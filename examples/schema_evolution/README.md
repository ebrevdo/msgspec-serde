# Schema evolution example

This example keeps two versions of the same FlatBuffers table and the Python
modules generated from each version. It demonstrates both directions of
compatible schema evolution:

- A `ReadingView` generated from version 2 reads a version 1 buffer. Fields
  added in version 2 use their declared defaults.
- A `ReadingView` generated from version 1 reads a version 2 buffer. It ignores
  fields that did not exist in version 1.

The evolution is compatible because version 2 only appends fields. Existing
field order, types, and defaults do not change.

Run the example from the repository root:

```console
uv run python examples/schema_evolution/demo.py
```

The `fixtures` directory contains buffers and binary schemas produced by
`flatc 23.5.26`. They are retained as interoperability fixtures, not as exact
output snapshots. Tests assert their meaning rather than their bytes.
The same tests regenerate modules with the installed `flatc`, so CI can run
the suite against multiple compiler versions without changing the fixtures.

The checked-in `schema_evolution_generated` package makes the generated API
available for reading and experimentation without running `flatc`. Regenerate
it after an intentional generator change with:

```console
uv run msgspec_flatc generate \
  examples/schema_evolution/schemas/reading_v1.fbs \
  examples/schema_evolution/schemas/reading_v2.fbs \
  -o examples/schema_evolution \
  --project-root examples/schema_evolution \
  --package schema_evolution_generated.example.evolution \
  --gen-onefile
```
