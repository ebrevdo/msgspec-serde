# msgspec-flatbuffers

`msgspec-flatbuffers` turns FlatBuffers IDL files into typed Python modules with
two complementary APIs:

- Lazy, read-only views for accessing FlatBuffers without materializing the
  complete object graph.
- Frozen [`msgspec.Struct`](https://jcristharif.com/msgspec/) models for ordinary
  Python value semantics and FlatBuffer construction.

## Installation

Install the package for your user account:

```shell
python -m pip install --user msgspec-flatbuffers
```

This installs the `msgspec-flatbuffers` command in your local binary directory,
usually `~/.local/bin`. Make sure that directory is on `PATH`, then verify the
installation:

```shell
msgspec-flatbuffers --help
```

Python 3.12 or newer is required. Schema generation also requires the
FlatBuffers compiler, `flatc`, on `PATH`:

```shell
flatc --version
```

`flatc` is only needed to generate Python modules. Applications reading or
writing FlatBuffers through generated modules do not invoke it.

Prebuilt wheels include the native extension. Installing from source also
requires a Rust toolchain.

## Generate Python modules

Generate a module from one or more `.fbs` files:

```shell
msgspec-flatbuffers generate schemas/monster.fbs -o generated
```

Use `-I` for directories referenced by `include` statements:

```shell
msgspec-flatbuffers generate \
  schemas/monster.fbs \
  -I schemas/includes \
  -o generated
```

Each `.fbs` file produces one Python module. FlatBuffers namespaces become
Python package directories. For example, `monster.fbs` in namespace `Example`
produces `generated/example/monster.py`. Use `--package` to add a package prefix:

```shell
msgspec-flatbuffers generate \
  schemas/monster.fbs \
  -o generated \
  --package my_application.generated
```

When schemas are part of a larger source tree, `--project-root` establishes the
root used to resolve generated imports:

```shell
msgspec-flatbuffers generate \
  schemas/game/monster.fbs \
  -I schemas \
  -o generated \
  --project-root schemas
```

## Models and views

Given this schema:

```fbs
namespace Example;

table Weapon {
  name:string;
  damage:short = 0;
}

table Monster {
  name:string;
  inventory:[ubyte];
  scores:[uint];
  weapons:[Weapon];
}

root_type Monster;
```

the generated module exports `Weapon`, `WeaponView`, `Monster`, and
`MonsterView`.

### Construct a FlatBuffer

Create frozen msgspec models and encode the root model:

```python
import numpy as np

from example.monster import Monster, Weapon

monster = Monster(
    name="Orc",
    inventory=b"\x00\x01\x02",
    scores=np.array([10, 20, 30], dtype=np.float32),
    weapons=[Weapon(name="Axe", damage=12)],
)

buffer: memoryview = monster.to_flatbuffer()
```

`to_flatbuffer()` returns a read-only `memoryview` with immutable backing
storage, without copying the finished FlatBuffer. Convert it to `bytes` only
when another API requires the concrete `bytes` type:

```python
owned_buffer = bytes(monster.to_flatbuffer())
```

The default `initial_size=0` uses an automatic size estimate with 1 percent
padding. A positive value sets a larger minimum initial allocation when needed:

```python
buffer = monster.to_flatbuffer(initial_size=4096)
```

The positive minimum is used only when it exceeds the padded estimate. The
allocation still grows automatically if the estimate is too small.

### Read lazily

Create a view over a buffer and access only the fields you need:

```python
from example.monster import MonsterView

view = MonsterView.from_buffer(buffer)

print(view.name)
print(view.weapons[0].damage)
```

Views accept contiguous objects implementing Python's buffer protocol,
including `bytes`, `bytearray`, `memoryview`, and memory-mapped files. They expose
a read-only `memoryview`, but the original backing object must not be mutated
while a view is alive.

Allocated values and nested views are cached after access. Repeated indexed
access returns the same nested view object, and repeated scans reuse already
materialized entries:

```python
weapons = view.weapons
assert weapons is view.weapons
assert view.weapons[0] is view.weapons[0]
```

### Materialize a model

Convert a view into an independent msgspec model:

```python
model: Monster = view.to_model()
```

`to_model()` copies the full object graph into ordinary Python values. The
result is not cached, so separate calls produce separate snapshots.

The model itself is frozen, but its vector values are mutable. Numeric vectors
are writable NumPy arrays, and other non-byte vectors are Python lists:

```python
assert model.scores is not None
assert model.weapons is not None

model.scores[0] = 20
model.weapons.append(Weapon(name="Bow", damage=5))
```

## Field representations

Views preserve borrowed access where the FlatBuffers layout permits it. Models
use independent, owned Python values.

| FlatBuffers field | View value | Materialized model value |
|---|---|---|
| Scalar | `int`, `float`, or `bool` | `int`, `float`, or `bool` |
| String | Cached `str` | `str` |
| `[ubyte]` | Read-only `memoryview` | `bytes` |
| `[ubyte] (nested_flatbuffer: "Payload")` | Raw `memoryview` and typed `PayloadView` | `Payload` |
| String tag plus `[ubyte] (dynamic_flatbuffer: "type")` | Raw tag, `memoryview`, and `DynamicView` | `DynamicValue` |
| Numeric vector | Read-only NumPy array | Writable, owning NumPy array with the generated dtype |
| Enum vector | Read-only NumPy array | `list[Enum]` |
| String vector | Cached sequence | `list[str]` |
| Table field | Cached nested view | Nested msgspec model |
| Table or struct vector | Cached view sequence | `list` of models |
| Union | Discriminator and typed variant view | Selected variant model |
| Union vector | Discriminator array and cached variant views | `list` of variant models |

Numeric and union-discriminator vectors use the FlatBuffers little-endian
element type and support normal NumPy indexing without copying the vector.
Bounds checks still apply.

Materialized numeric arrays use native NumPy dtypes such as `np.uint32` and
`np.float64`. They own their storage, are writable, and do not share memory with
the source view.

## Msgspec conversion hooks

Pass the package's NumPy hooks to msgspec's standard conversion and JSON APIs:

```python
import msgspec

from msgspec_flatbuffers import dec_hook, enc_hook

data: dict[str, object] = msgspec.to_builtins(monster, enc_hook=enc_hook)
from_dict: Monster = msgspec.convert(
    data,
    type=Monster,
    dec_hook=dec_hook,
)

encoded: bytes = msgspec.json.encode(monster, enc_hook=enc_hook)
from_json: Monster = msgspec.json.decode(
    encoded,
    type=Monster,
    dec_hook=dec_hook,
)
```

`dec_hook` restores each generated NumPy dtype, including arrays inside nested
tables and lists. Decoded arrays own their storage and are writable.

Every generated table model uses its fully qualified IDL name as a stable
msgspec tag. The tag is encoded in the reserved
`__msgspec_flatbuffers_type__` field, allowing msgspec to decode table-valued
union fields and union vectors without a custom union codec.

## IDL-declared nested FlatBuffers

Use `nested_flatbuffer` when a byte vector always contains one known FlatBuffer
type:

```fbs
table Payload {
  serial:uint;
}

table Envelope {
  payload:[ubyte] (nested_flatbuffer: "Payload");
}
```

The materialized model uses the declared type directly:

```python
envelope = Envelope(payload=Payload(serial=17))
buffer = envelope.to_flatbuffer()

payload: Payload | None = envelope.payload
```

The view exposes both the raw bytes and a typed helper whose target is inferred
from the IDL:

```python
envelope_view = EnvelopeView.from_buffer(buffer)

raw: memoryview | None = envelope_view.payload
payload_view: PayloadView | None = envelope_view.payload_view()
```

`payload_view()` is zero-copy and cached. It validates the nested target's file
identifier when the target IDL declares one. If the stored payload is
size-prefixed, request that framing explicitly:

```python
payload_view = envelope_view.payload_view(size_prefixed=True)
```

Nested targets may contain unions. Building an `Envelope` encodes its nested
model automatically.

## Unannotated byte vectors

An unannotated `[ubyte]` field can also store a complete FlatBuffer. This is
useful when an application or plugin registry selects the payload type at
runtime. Use the generated `<field>_as()` helper with the selected view type:

```python
extension = base.extensions[0]
adam = extension.data_as(AdamExtensionView)
```

The nested view borrows the byte vector without copying it. Repeated calls with
the same target type and `size_prefixed` setting return the same view object. An
absent optional byte vector returns `None`.

The materialized form stores the payload as `bytes`:

```python
extension_model = extension.to_model()
adam_model = AdamExtensionView.from_buffer(extension_model.data).to_model()
```

This representation preserves payloads whose target type is selected by an
application or plugin registry.

## Dynamic FlatBuffers

Use `dynamic_flatbuffer` when a byte vector's root type is selected by a
sibling string tag. `dynamic_allow` declares the namespace of permitted
extension types without enumerating them:

```fbs
attribute "dynamic_flatbuffer";
attribute "dynamic_allow";

table Envelope {
  payload_type:string;
  payload:[ubyte] (
    dynamic_flatbuffer: "payload_type",
    dynamic_allow: "Example.Dynamic.*"
  );
}
```

The referenced type field must be a string with the same requiredness as the
byte vector. An absent tag and payload represent an absent value. Only a
terminal namespace wildcard such as `Example.Dynamic.*` is accepted.

Mark each extension root in its own schema:

```fbs
attribute "dynamic_extension";

table Metric (dynamic_extension) {
  name:string;
  values:[float];
}

root_type Metric;
```

The generated extension module registers its root model and view when imported.
No manual type registration is needed:

```python
from msgspec_flatbuffers import DynamicValue

envelope = Envelope(
    payload=DynamicValue(Metric(name="latency")),
)
```

The materialized model contains one `DynamicValue` field. Its concrete table
tag is written to the sibling string when the FlatBuffer is built. The view
retains the raw tag and byte vector and adds a cached dynamic helper:

```python
view = EnvelopeView.from_buffer(envelope.to_flatbuffer())
payload = view.payload_dynamic()

assert payload is not None
assert payload.tag == "Example.Dynamic.Metric"
assert isinstance(payload.value, MetricView)
```

The standard `enc_hook` and `dec_hook` use the same process-wide registry for
dictionary and JSON conversion. The dynamic wrapper stores the registered type
name separately from the ordinary, untagged extension model:

```json
{
  "__msgspec_flatbuffers_type__": "Example.Dynamic.Metric",
  "value": {
    "name": "latency",
    "values": [1.25, 2.5]
  }
}
```

The extension model itself has no msgspec tag. Msgspec tags are generated only
for tables that are alternatives in an IDL union.

An unknown tag inside the allowed namespace is materialized as an opaque
`DynamicValue`. Its tag and payload bytes survive dictionary, JSON, and
FlatBuffer round trips, allowing a newer extension to pass through an older
process without data loss. Tags outside the allowed namespace are rejected.

For optional lazy loading, associate a trusted tag with its generated module:

```python
from msgspec_flatbuffers import register_dynamic_module

register_dynamic_module(
    "Example.Dynamic.Metric",
    "example.dynamic.metric",
)
```

On the first registry miss, the mapped module is imported and registers its
generated root. Wire tags are never interpreted directly as Python import
paths.

## Unions

FlatBuffers unions provide a closed set of table alternatives:

```fbs
table Cat {
  name:string;
}

table Dog {
  name:string;
}

union Pet { Cat, Dog }

table Payload {
  favorite:Pet;
  residents:[Pet];
}
```

Materialized models use ordinary Python unions:

```python
payload = Payload(
    favorite=Cat(name="Miso"),
    residents=[Cat(name="Miso"), Dog(name="Pip")],
)

favorite: Cat | Dog | None = payload.favorite
residents: list[Cat | Dog] | None = payload.residents
```

The model builder selects each discriminator from the variant's Python type.
The lazy view exposes the discriminator and the selected typed view separately:

```python
favorite_type: Pet = payload_view.favorite_type
favorite_view: CatView | DogView | None = payload_view.favorite_view()

resident_types = payload_view.residents_type
residents = payload_view.residents
```

`residents_type` is a read-only NumPy array of discriminator values.
`residents` is a cached sequence of `CatView` and `DogView` objects. Repeated
access to the same union value returns the same view object.

The whole union vector may be absent. When present, every element must select a
table alternative; `NONE` and null elements are rejected. Unknown
discriminators raise `InvalidBufferError` when their payload is accessed or the
containing view is materialized.

## Size-prefixed buffers

FlatBuffers may be encoded with a four-byte size prefix:

```python
framed = monster.to_flatbuffer(size_prefixed=True)
framed_view = MonsterView.from_buffer(framed, size_prefixed=True)
```

`size_prefixed` defaults to `False`. A size-prefixed view is restricted to the
payload length declared by its prefix, so trailing or concatenated frames are
not part of its buffer.

The same option applies to either form of FlatBuffer stored in a byte vector:

```python
adam = extension.data_as(AdamExtensionView, size_prefixed=True)
payload = envelope_view.payload_view(size_prefixed=True)
```

## File identifiers

When the target schema declares a four-byte `file_identifier`, root views check
it automatically:

```fbs
file_identifier "MNST";
```

```python
view = MonsterView.from_buffer(buffer)
```

Pass `check_identifier=False` only when intentionally reading data without that
validation:

```python
view = MonsterView.from_buffer(buffer, check_identifier=False)
```

`<field>_as()`, `<field>_view()`, and `<field>_dynamic()` also follow the target
view's declared identifier automatically.

## Supported schemas and limitations

The generated API supports tables, structs, enums, optional scalars, strings,
nested tables, unions, IDL-declared and dynamically typed nested FlatBuffers,
and vectors of scalars, bytes, strings, tables, structs, and unions. Included
schemas and multiple declarations in one `.fbs` file are supported.

Current limitations:

- Union alternatives must be tables with unique target types. String and struct
  alternatives and duplicate-target aliases are not supported.
- Union vectors cannot contain `NONE` or null elements.
- One `.fbs` file cannot declare definitions in multiple namespaces.
- Nested struct construction is not supported.
- Fixed arrays and 64-bit offsets are not supported.

Unsupported schema features fail generation instead of producing a partial
Python API.
