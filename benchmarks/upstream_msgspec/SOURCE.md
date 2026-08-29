# Upstream msgspec benchmarks

This directory vendors part of msgspec's filesystem benchmark from the
following upstream revision:

- Repository: <https://github.com/msgspec/msgspec>
- Commit: `f51f378335b01dc0026dc6553a0b9e1915a8edae`
- Snapshot date: 2026-08-28

The retained scripts under `benchmarks/` began as copies of the upstream
filesystem encoding benchmark. They remain subject to the bundled BSD 3-Clause
`LICENSE`.

Local changes:

- Every generated `File` and `Directory` has a deterministic `list[int]` and
  `list[float]`. The default length is 16 values per list.
- The profile runner measures the list-backed msgspec model, NumPy array hooks,
  native `msgspec_serde` array handling, and the IDL-generated model.
- The encoding runner emits only the orjson, ormsgpack, and materialized
  FlatBuffers rows used by `benchmarks.md`.
- `schemas/filesystem.fbs` defines the same generated filesystem model for
  `msgspec-flatbuffers`.

The IDL numeric vectors become `np.ndarray[np.int32]` and
`np.ndarray[np.float64]` fields in the generated models. Schema compilation,
module import, fixture conversion, and round-trip validation happen before
timing starts.

Integer values are sampled across the complete signed 32-bit range. Floating-
point values are finite Python doubles sampled between -1,000,000 and 1,000,000,
which preserves full binary64-style precision in their text representations.

See `benchmarks.md` for the retained commands and report inputs.
