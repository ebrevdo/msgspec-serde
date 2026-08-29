# FlatBuffers reflection bindings

This directory contains code generated from Google FlatBuffers
`reflection/reflection.fbs` at tag `v23.5.26`. The schema namespace was changed
from `reflection` to `msgspec_serde._reflection` so the generated imports remain
private to this package. No field definitions were changed.

The generated files are covered by the Apache License 2.0 in `LICENSE.txt`.
They are excluded from `ty` because they are compiler output. The handwritten
adapter in `msgspec_serde._schema_reader` remains type-checked.

Regenerate the files with a matching `flatc` executable:

```console
flatc --python --python-typing -o <output> reflection.fbs
```
