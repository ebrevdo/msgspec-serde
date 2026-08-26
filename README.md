# msgspec-flatbuffers

## Tutorial

`msgspec-flatbuffers` generates Python modules from FlatBuffers IDL (`.fbs`)
files. Every generated table and struct provides two Python interfaces:

- A read-only view borrows a FlatBuffer and decodes each field when you access
  it.
- A mutable [`msgspec.Struct`](https://jcristharif.com/msgspec/) model owns its
  data and can build a new FlatBuffer.

The tutorial starts with a model and a view for one table. Later sections add
owned snapshots, validation through subclasses, JSON, unions, typed nested
FlatBuffers, and dynamic payloads. You can also run the complete example in
[`examples/tutorial`](https://github.com/ebrevdo/msgspec_flatbuffers/tree/main/examples/tutorial).

### Run the finished tutorial

From the repository root:

```shell
cd examples/tutorial
msgspec-flatbuffers generate \
  schemas/*.fbs \
  -I schemas \
  -o generated \
  --project-root schemas
PYTHONPATH=generated python demo.py
```

The first three output lines are:

```text
Orc Axe
Cat nightly snapshot
Tutorial.Metric
```

The fourth line contains the complete `Monster` model as JSON.

### 1. Install the tools

Install the Python package:

```shell
python -m pip install --user msgspec-flatbuffers
```

With `--user`, pip usually installs the `msgspec-flatbuffers` executable in
`~/.local/bin`. Add that directory to `PATH` if your shell cannot find the
executable, then check the installation:

```shell
msgspec-flatbuffers --help
```

Python 3.12 or newer is required. Generating modules also requires the
FlatBuffers compiler, `flatc`:

```shell
flatc --version
```

Only generation requires `flatc`. Importing and using a generated module does
not run it.

### 2. Write a schema

Create a project directory:

```shell
mkdir -p flatbuffers-tutorial/schemas
cd flatbuffers-tutorial
```

Save this as `schemas/monster.fbs`:

```fbs
namespace Tutorial;

enum Color : byte {
  Red = 1,
  Green,
  Blue,
}

struct Vec3 {
  x:float;
  y:float;
  z:float;
}

table Weapon {
  name:string (required);
  damage:short = 0;
}

table Monster {
  pos:Vec3;
  name:string (required);
  inventory:[ubyte];
  color:Color = Blue;
  weapons:[Weapon];
  scores:[float];
  tags:[string];
  optional_energy:int = null;
}

root_type Monster;
file_identifier "MNST";
```

### 3. Generate the Python module

Generate into a new `generated/` directory:

```shell
msgspec-flatbuffers generate \
  schemas/monster.fbs \
  -o generated \
  --project-root schemas
```

This command writes `generated/tutorial/monster.py`. The `Tutorial` namespace
becomes the `tutorial/` package directory. The `Monster` root table becomes the
`monster.py` module.

By default, the generator writes each definition to its own module under the
namespace package, matching `flatc --python`. For example, `Tutorial.Weapon`
becomes `generated/tutorial/weapon.py`. The root module re-exports dependency
names when they are unambiguous, so the tutorial can import `Weapon` from
`tutorial.monster`. The command prints the path of the root table's module.

Use `--gen-onefile` to place every definition from a schema in one module:

```shell
msgspec-flatbuffers generate \
  schemas/monster.fbs \
  -o generated \
  --project-root schemas \
  --gen-onefile
```

This writes all definitions to `generated/monster.py`, without a namespace
directory. One-file generation rejects a schema when two definitions have the
same short name. The default layout places those definitions in separate
modules.

You may pass several schema paths in one command. Use `-I` to add directories
for FlatBuffers `include` statements. The included schemas and the schemas that
import them must use the same one-file setting.

The examples below add `generated/` to `PYTHONPATH`. To generate modules inside
an application's source tree, use `--package` to add a package prefix:

```shell
msgspec-flatbuffers generate \
  schemas/monster.fbs \
  -o src \
  --project-root schemas \
  --package my_application.generated
```

You can then import the generated root module as
`my_application.generated.tutorial.monster`.

### 4. Build a FlatBuffer from a model

The generated module contains a model and a view for every table and struct.
Create `demo.py`:

```python
import numpy as np

from tutorial.monster import Color, Monster, Vec3, Weapon

monster = Monster(
    pos=Vec3(x=1.0, y=2.0, z=3.0),
    name="Orc",
    inventory=b"\x00\x01\x02",
    color=Color.Green,
    weapons=[Weapon(name="Axe", damage=12)],
    scores=np.array([10.0, 20.0, 30.0], dtype=np.float32),
    tags=["boss", "cave"],
    optional_energy=80,
)

buffer: memoryview = monster.to_flatbuffer()

assert buffer.readonly
```

Run it with:

```shell
PYTHONPATH=generated python demo.py
```

Later snippets extend the same `demo.py`.

`to_flatbuffer()` returns a read-only `memoryview`. Convert the result only when
another API requires `bytes`:

```python
owned_bytes = bytes(buffer)
```

With the default `initial_size=0`, `to_flatbuffer()` estimates the initial
allocation. Pass a positive value to set a minimum initial allocation:

```python
buffer = monster.to_flatbuffer(initial_size=4096)
```

`to_flatbuffer()` grows the allocation if the estimate is too small.

### 5. Read fields without materializing the object

A view borrows the FlatBuffer and decodes fields as you access them:

```python
from tutorial.monster import MonsterView

view = MonsterView.from_buffer(buffer)
position = view.pos

assert view.name == "Orc"
assert position is not None
assert position.z == 3.0
assert view.color is Color.Green
assert view.inventory is not None
assert view.inventory.readonly
```

Views return numeric and byte vectors without copying them. A numeric vector is
a read-only NumPy array over the FlatBuffer:

```python
assert view.scores is not None
assert view.scores.dtype == np.dtype("<f4")
assert not view.scores.flags.writeable
```

Strings, nested views, and vector elements are cached after their first access:

```python
weapons = view.weapons
assert weapons is not None
assert weapons is view.weapons
assert weapons[0].damage == 12
assert weapons[0] is weapons[0]
```

Views accept contiguous buffer objects, including `bytes`, `bytearray`,
`memoryview`, and memory-mapped files. A view exposes its buffer through a
read-only `memoryview`. Another reference can still change a mutable backing
object such as a `bytearray`, so do not mutate the backing object while a view
uses it.

### 6. Materialize independent Python data

Call `to_model()` to copy the data into an owned model:

```python
restored = view.to_model()

assert restored == monster
assert restored is not monster
assert restored.scores is not None
assert restored.scores.flags.owndata
assert restored.scores.flags.writeable
```

When you only need the model, decode it directly:

```python
restored = Monster.from_flatbuffer(buffer)
```

Both decoding methods copy the complete object graph. Numeric vectors become
writable NumPy arrays that own their storage. Other non-byte vectors become
Python lists. Each call returns an independent model.

Generated models, their arrays, and their lists are mutable:

```python
assert restored.weapons is not None

restored.name = "Goblin"
restored.scores[0] = 99.0
restored.weapons.append(Weapon(name="Bow", damage=5))
```

### 7. Add application validation

Subclass a generated model to add validation, methods, or Python-only state. To
decode a nested field as a model subclass, redeclare the inherited field with
the subclass type:

```python
from typing import ClassVar


class ValidatedWeapon(Weapon, dict=True):
    validation_kind: ClassVar[str] = "weapon"

    def __post_init__(self) -> None:
        if self.damage < 0:
            raise ValueError("damage cannot be negative")
        self.was_validated = True


class ValidatedMonster(Monster, dict=True):
    weapons: list[ValidatedWeapon] | None = None

    def __post_init__(self) -> None:
        self.was_validated = True
```

Call `from_flatbuffer()` on the subclass, or pass it to `to_model()`:

```python
validated = ValidatedMonster.from_flatbuffer(buffer)
validated_from_view = view.to_model(ValidatedMonster)

assert validated.was_validated
assert validated.weapons is not None
assert isinstance(validated.weapons[0], ValidatedWeapon)
assert validated_from_view.was_validated
```

FlatBuffer decoding calls `__post_init__()` on the requested root class and on
nested subclasses selected by its annotations. An annotation may select a
subclass inside an optional field, list, union, struct, or typed nested
FlatBuffer.

`ClassVar` values and attributes stored in an instance's `__dict__` are
Python-only state. The `dict=True` class option enables the instance dictionary.
Neither msgspec nor FlatBuffers serializes this state. Any other annotation
creates a msgspec field. The generated FlatBuffer APIs reject a subclass when
that field has no matching FlatBuffers field.

### 8. Use msgspec conversion and JSON

Pass the package's conversion hooks to msgspec when a model contains NumPy
arrays:

```python
import msgspec

from msgspec_flatbuffers import dec_hook, enc_hook

data = msgspec.to_builtins(monster, enc_hook=enc_hook)
from_dict = msgspec.convert(data, type=Monster, dec_hook=dec_hook)

encoded = msgspec.json.encode(monster, enc_hook=enc_hook)
from_json = msgspec.json.decode(
    encoded,
    type=Monster,
    dec_hook=dec_hook,
)

assert from_dict == monster
assert from_json == monster
```

`dec_hook` restores every generated NumPy dtype, including arrays nested inside
tables and lists. The restored arrays own their storage and are writable.

### 9. Add a FlatBuffers union

Save the following schema as `schemas/pets.fbs`:

```fbs
namespace Tutorial;

table Cat {
  name:string (required);
  lives:ubyte = 9;
}

table Dog {
  name:string (required);
  good:bool = true;
}

union Pet { Cat, Dog }

table PetList {
  favorite:Pet;
  residents:[Pet];
}

root_type PetList;
file_identifier "PETS";
```

Generate the new module:

```shell
msgspec-flatbuffers generate \
  schemas/pets.fbs \
  -o generated \
  --project-root schemas
```

Materialized models use ordinary Python union types:

```python
from tutorial.pet_list import Cat, Dog, PetList

pets = PetList(
    favorite=Cat(name="Miso"),
    residents=[Cat(name="Miso"), None, Dog(name="Pip")],
)
```

`to_flatbuffer()` chooses each FlatBuffers discriminator from the concrete
Python type. A view exposes both the selected value and its discriminator:

```python
from tutorial.pet_list import CatView, Pet, PetListView

pets_view = PetListView.from_buffer(pets.to_flatbuffer())

assert pets_view.favorite_type is Pet.Cat
assert isinstance(pets_view.favorite, CatView)
assert pets_view.residents_type is not None
assert pets_view.residents is not None
```

`residents_type` is a read-only NumPy array. `residents` is a cached sequence of
`CatView`, `DogView`, and `None` values. A `None` entry uses a `NONE`
discriminator and a null payload offset. A non-`NONE` discriminator with a null
payload is invalid.

Union alternatives receive msgspec tags based on their fully qualified IDL
names. Other generated tables remain untagged.

### 10. Put a typed FlatBuffer inside a byte vector

Use `nested_flatbuffer` when a byte vector always contains one known root type.
Save this as `schemas/nested_envelope.fbs`:

```fbs
include "pets.fbs";

namespace Tutorial;

table NestedEnvelope {
  snapshot:[ubyte] (nested_flatbuffer: "Tutorial.PetList");
  note:string;
}

root_type NestedEnvelope;
file_identifier "SNAP";
```

Generate it with the schema directory on the include path:

```shell
msgspec-flatbuffers generate \
  schemas/nested_envelope.fbs \
  -I schemas \
  -o generated \
  --project-root schemas
```

The model field uses `PetList`, not `bytes`:

```python
from tutorial.nested_envelope import NestedEnvelope, NestedEnvelopeView

envelope = NestedEnvelope(snapshot=pets, note="nightly snapshot")
envelope_view = NestedEnvelopeView.from_buffer(envelope.to_flatbuffer())
snapshot = envelope_view.snapshot
snapshot_raw = envelope_view.snapshot_raw

assert snapshot is not None
assert snapshot.favorite_type is Pet.Cat
assert snapshot_raw is not None
assert snapshot_raw.readonly
```

`snapshot` returns a cached `PetListView` over the byte vector. `snapshot_raw`
returns the same payload as a read-only `memoryview`. The reader checks the
`PetList` file identifier automatically.

If an incoming nested payload is size-prefixed, request that framing when you
open it:

```python
snapshot = envelope_view.snapshot_view(size_prefixed=True)
```

### 11. Add open-ended dynamic payloads

A union lists every permitted type in the IDL. A dynamic field stores a string
type name beside a byte vector. Generated extension modules can register new
types without changing the envelope schema.

Save the first extension as `schemas/metric.fbs`:

```fbs
namespace Tutorial;

attribute "dynamic_extension";

table Metric (dynamic_extension) {
  name:string (required);
  values:[float];
}

root_type Metric;
file_identifier "METR";
```

Save the envelope as `schemas/dynamic_envelope.fbs`:

```fbs
namespace Tutorial;

attribute "dynamic_flatbuffer";
attribute "dynamic_allow";

table DynamicEnvelope {
  payload_type:string;
  payload:[ubyte] (
    dynamic_flatbuffer: "payload_type",
    dynamic_allow: "Tutorial.*"
  );
}

root_type DynamicEnvelope;
file_identifier "DYNE";
```

Generate both modules:

```shell
msgspec-flatbuffers generate \
  schemas/metric.fbs \
  schemas/dynamic_envelope.fbs \
  -o generated \
  --project-root schemas
```

Importing the extension module registers `Tutorial.Metric`. The envelope module
exports `DynamicEnvelopePayload`, a wrapper for the dynamic value and its type
name:

```python
from tutorial.dynamic_envelope import (
    DynamicEnvelope,
    DynamicEnvelopePayload,
    DynamicEnvelopeView,
)
from tutorial.metric import Metric, MetricView

metric = Metric(
    name="latency",
    values=np.array([1.25, 2.5], dtype=np.float32),
)
dynamic = DynamicEnvelope(payload=DynamicEnvelopePayload(metric))
dynamic_view = DynamicEnvelopeView.from_buffer(dynamic.to_flatbuffer())

payload = dynamic_view.payload
payload_raw = dynamic_view.payload_raw
assert payload is not None
assert payload.tag == "Tutorial.Metric"
assert isinstance(payload.value, MetricView)
assert dynamic_view.payload_type_raw == "Tutorial.Metric"
assert payload_raw is not None
assert payload_raw.readonly
```

`payload_type_raw` and `payload_raw` expose the stored type name and byte vector.
`Metric` remains an ordinary, untagged msgspec model. Encode and decode the
wrapper with the same JSON hooks:

```python
encoded = msgspec.json.encode(dynamic, enc_hook=enc_hook)
decoded = msgspec.json.decode(
    encoded,
    type=DynamicEnvelope,
    dec_hook=dec_hook,
)

assert decoded == dynamic
```

The `Tutorial.Metric` wire tag resolves to the generated `Metric` class. It
cannot select an application subclass such as `ValidatedMetric`. To construct
that subclass while decoding, map `Metric` to `ValidatedMetric` with
`DynamicModelOverrides`:

```python
from msgspec_flatbuffers import DynamicModelOverrides


class ValidatedMetric(Metric, dict=True):
    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("metric name cannot be empty")
        self.was_validated = True


dynamic_overrides = DynamicModelOverrides(
    {
        Metric: ValidatedMetric,
    }
)

validated_buffer_model = DynamicEnvelope.from_flatbuffer(
    dynamic.to_flatbuffer(),
    dynamic_overrides=dynamic_overrides,
)
validated_json_model = msgspec.json.decode(
    encoded,
    type=DynamicEnvelope,
    dec_hook=dynamic_overrides.dec_hook,
)

assert validated_buffer_model.payload is not None
assert validated_json_model.payload is not None
assert isinstance(validated_buffer_model.payload.value, ValidatedMetric)
assert isinstance(validated_json_model.payload.value, ValidatedMetric)
```

`DynamicModelOverrides` is a mutable mapping from generated models to
application subclasses. The `|` operator returns a new mapping, and `|=` updates
the mapping on its left. If both mappings contain the same generated model, the
value on the right wins. Decoding uses the generated model when no override
exists.

An unknown type inside the allowed `Tutorial.*` namespace remains opaque. The
library preserves its type name and bytes across model, JSON, and FlatBuffer
round trips:

```python
opaque = DynamicEnvelope(
    payload=DynamicEnvelopePayload.opaque(
        "Tutorial.FutureMetric",
        b"payload from a newer plugin",
    )
)
forwarded = DynamicEnvelope.from_flatbuffer(opaque.to_flatbuffer())
forwarded_payload = forwarded.payload

assert forwarded_payload is not None
assert not forwarded_payload.is_known
assert forwarded_payload.data == b"payload from a newer plugin"
```

Tags outside `Tutorial.*` are rejected. The runtime never treats a wire tag as
a Python import path. For lazy imports, map each trusted tag to its module
explicitly:

```python
from msgspec_flatbuffers import register_dynamic_module

register_dynamic_module("Tutorial.Metric", "tutorial.metric")
```

### 12. Interpret an unannotated byte vector manually

Use an ordinary `[ubyte]` vector when the schema does not declare the payload
type. Save this as `schemas/raw_extensions.fbs`:

```fbs
namespace Tutorial;

table Extension {
  type_id:ulong;
  data:[ubyte];
}

table ExtensionList {
  extensions:[Extension];
}

root_type ExtensionList;
file_identifier "EXTS";
```

Generate the module:

```shell
msgspec-flatbuffers generate \
  schemas/raw_extensions.fbs \
  -o generated \
  --project-root schemas
```

`Extension.data` is `bytes` in the model. The generated `data_as()` helper reads
the vector through a chosen view type without copying it:

```python
from tutorial.extension_list import Extension, ExtensionList, ExtensionListView

metric_bytes = bytes(metric.to_flatbuffer())
extension_list = ExtensionList(
    extensions=[
        Extension(
            type_id=0xA813,
            data=metric_bytes,
        )
    ]
)
extension_list_view = ExtensionListView.from_buffer(
    extension_list.to_flatbuffer()
)
extensions = extension_list_view.extensions

assert extensions is not None
extension_view = extensions[0]
metric_view = extension_view.data_as(MetricView)

assert metric_view is not None
assert metric_view.name == "latency"
assert extension_view.data_as(MetricView) is metric_view
```

The application defines the meaning of `type_id`. Repeated `data_as()` calls
with the same view type and `size_prefixed` setting return the same cached view.

Use `nested_flatbuffer` when the IDL declares the payload type. Use a dynamic
field when extension modules register an open-ended set of type names.

### 13. Use size prefixes and file identifiers

Every tutorial schema declares a four-byte file identifier. The
`MonsterView.from_buffer()` and `Monster.from_flatbuffer()` calls in steps 5 and
6 check `MNST` by default.

Disable the check only when the input intentionally omits or replaces the
identifier:

```python
view = MonsterView.from_buffer(buffer, check_identifier=False)
```

FlatBuffers can also include a four-byte size prefix:

```python
framed = monster.to_flatbuffer(size_prefixed=True)
framed_view = MonsterView.from_buffer(framed, size_prefixed=True)
framed_model = Monster.from_flatbuffer(framed, size_prefixed=True)
```

A size-prefixed reader stops at the payload length declared by the prefix. The
view excludes trailing bytes and any frames concatenated after the payload.

### Field representations

| FlatBuffers field | View value | Materialized model value |
|---|---|---|
| Scalar | `int`, `float`, or `bool` | `int`, `float`, or `bool` |
| Enum | Generated enum | Generated enum |
| String | Cached `str` | `str` |
| `[ubyte]` | Read-only `memoryview` | `bytes` |
| Numeric vector | Read-only NumPy array | Writable, owning NumPy array |
| Enum vector | Read-only NumPy array | `list[Enum]` |
| String vector | Cached sequence | `list[str]` |
| Table field | Cached table view | Nested model |
| Struct field | Cached inline struct view | Nested struct model |
| Table or struct vector | Cached view sequence | `list` of models |
| Fixed numeric array | Read-only NumPy array | Writable, owning NumPy array |
| Fixed enum array | Read-only NumPy array | `list[Enum]` |
| Fixed struct array | Cached struct views | `list` of struct models |
| Union | Selected view plus `<field>_type` | Selected model |
| Union vector | Discriminator array and cached views | `list` of models or `None` values |
| IDL nested FlatBuffer | Typed nested view plus `<field>_raw` | Typed nested model |
| Dynamic FlatBuffer | `DynamicView` plus raw tag and bytes | Generated dynamic wrapper |

A view exposes numeric vectors with the FlatBuffers little-endian dtype.
Materialized arrays use the platform's native NumPy dtype.

### Supported schemas and limitations

Generated modules support tables, nested structs, fixed arrays, enums, optional
scalars, strings, nested tables, unions, and typed or dynamic nested
FlatBuffers. Vectors may contain scalars, bytes, strings, tables, structs, or
unions. A single `.fbs` file may declare several types in several namespaces.
Schemas may include other schemas.

Current limitations:

- Union alternatives must be tables with unique target types. String and struct
  alternatives and duplicate-target aliases are not supported.
- One-file generation cannot flatten identical short names from different
  namespaces.
- 64-bit offsets are not supported.

Unsupported schema features fail generation instead of producing a partial
Python API.

## Backward compatibility

Within one major SemVer version, we try to keep Python code generated by an
earlier release working with later `msgspec-flatbuffers` runtimes. This project
is young, so backward compatibility is a goal rather than a guarantee.

Forward compatibility is not guaranteed. Code generated by a later release may
not work with an earlier runtime.

Every generated module records the generator version and checks the installed
runtime during import:

- If the major versions differ in either direction, import raises
  `GeneratedCodeVersionError`.
- If the major versions match but the runtime minor version is older, import
  emits `GeneratedCodeVersionWarning` once for that generator/runtime version
  pair.
- Otherwise, import continues without a version warning. Patch-version
  differences do not produce a warning.

To suppress the older-runtime warning, set the process-wide flag before
importing generated modules:

```python
import msgspec_flatbuffers

msgspec_flatbuffers.warn_on_older_runtime = False
```

The flag does not suppress `GeneratedCodeVersionError` for a major-version
mismatch.
