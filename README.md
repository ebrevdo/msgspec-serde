# msgspec-serde

`msgspec-serde` brings together the performance and flexibility of `msgspec` with other
common data formats:
  * Representation of gridded data via `NumPy`-like arrays and views.
  * The cross-language and schema/IDL support of `FlatBuffers`.

It can generate typed `msgspec.Struct` models and read-only FlatBuffers
views from `.fbs` schemas.

Its JSON, MessagePack, and FlatBuffers codecs also support very fast encoding/decoding of
`msgspec.Struct` types containing NumPy arrays (with or without a FlatBuffers schema).

For example, in the [recorded benchmarks](benchmarks.md), the API encodes
FlatBuffers about 20–22 times faster and decodes them into complete models about
11–13 times faster than the official Python FlatBuffers API.

## Installation

In a Python 3.12 or newer environment, install the package:

```shell
python -m pip install msgspec-serde
```

Generating modules from IDL also requires `flatc`, the FlatBuffers compiler. Install it
separately and make sure `flatc --version` works in your shell. It is **not** required to:
  * Import, use, and ser/de of IDL-generated modules.
  * Import and use JSON/MessagePack ser/de of regular `Struct` containing NumPy arrays.

## Quick start

This takes you through generating `msgspec.Struct` from a FlatBuffers IDL, through ser/de.
If you're interested in efficient ser/de of regular `Struct` with NumPy arrays, you can skip
the generation step.

### Generating and subclassing `msgspec.Struct` from IDL

Save this schema as `reading.fbs`:

```fbs
namespace Example;

table Reading {
  name:string (required);
  values:[float] (required);
}

root_type Reading;
```

Generate the Python module:

```shell
msgspec_flatc generate reading.fbs -o generated
```

This creates `generated/example/reading.py`. The example below subclasses
`Reading` to reject blank names. Save it as `demo.py` in the same directory as
`reading.fbs`:

```python
import numpy as np

from example.reading import Reading, ReadingView
from msgspec_serde import flatbuffer, json, msgpack

class ValidatedReading(Reading):
    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be blank")


reading = ValidatedReading(
    name="latency", values=np.array([1.25, 2.5], dtype=np.float32)
)

buffer = flatbuffer.encode(reading)
view = flatbuffer.decode(buffer, type=ReadingView)
print(view.name, view.values.tolist())

model = view.to_model(ValidatedReading)
assert isinstance(model, ValidatedReading)
model.values[0] = 3.0
assert view.values[0] == 1.25

encoded_json = json.encode(model)
from_json = json.decode(encoded_json, type=ValidatedReading)
assert isinstance(from_json, ValidatedReading)

encoded_msgpack = msgpack.encode(model)
from_msgpack = msgpack.decode(encoded_msgpack, type=ValidatedReading)
assert isinstance(from_msgpack, ValidatedReading)
```

Run it from that directory:

```shell
PYTHONPATH=generated python demo.py
```

The script prints `latency [1.25, 2.5]`.

A view reads fields on demand from the serialized buffer. Numeric vectors are
read-only NumPy arrays over that buffer. Calling `to_model(ValidatedReading)`
copies the data into a mutable model whose arrays own their storage. To decode
directly into the subclass, use
`flatbuffer.decode(buffer, type=ValidatedReading)`.

Pass the subclass to `to_model()` or the decoder to run its `__post_init__()`
validator when constructing the model. Lazy views do not run model validation.

## Features

- Encode NumPy arrays without custom conversion hooks through the
  [JSON and MessagePack codecs](TUTORIAL.md#8-encode-json-and-messagepack).
  Reuse `Encoder` and `Decoder` instances when processing many values.
- [Decode into application subclasses](TUTORIAL.md#7-add-application-validation)
  and run their `__post_init__()` validation.
- Vectors of tables with a `(key)` field become dictionaries in models and
  read-only `TableMap` mappings in views. See
  [keyed table vectors](TUTORIAL.md#keyed-table-vectors).
- Read [typed nested FlatBuffers](TUTORIAL.md#10-put-a-typed-flatbuffer-inside-a-byte-vector).
- [Register additional dynamic payload types](TUTORIAL.md#11-add-open-ended-dynamic-payloads)
  without changing the envelope schema.
- Generate models and views for tables, structs, enums, fixed arrays, and
  unions. Schema `include` statements are supported. See
  [supported schemas and limitations](TUTORIAL.md#supported-schemas-and-limitations).

## Benchmarks

Selected-field reads are about 1.2 times faster than the official Python
FlatBuffers API in the recorded workloads. This measures reading a few fields,
separately from decoding the complete model.

In the recursive 1,000-object benchmark, MessagePack encodes fastest with
16-value vectors, while FlatBuffers decodes complete models fastest. With
256-value vectors, FlatBuffers is fastest in both directions.

See [benchmarks.md](benchmarks.md) for charts, methodology, environment details,
and reproducible commands.

## Tutorial

The [tutorial](TUTORIAL.md) walks through schema generation and the generated
APIs. The complete runnable example is in
[examples/tutorial](examples/tutorial).

## Backward compatibility

We aim to keep code generated by an earlier release working with later runtimes
in the same major version. This is a goal rather than a guarantee.
Major-version mismatches raise `GeneratedCodeVersionError`. Code generated by a
later release may not work with an earlier runtime. See
[runtime version checks and warning controls](TUTORIAL.md#generated-code-compatibility).
