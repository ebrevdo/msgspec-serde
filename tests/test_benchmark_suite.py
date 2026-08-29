from contextlib import closing

from benchmarks._suite import (
    ADAPTERS,
    OPERATIONS,
    PROFILE_NAMES,
    BenchmarkDefinition,
    BenchmarkSuite,
    benchmark_keys,
)


def _run_all(definitions: tuple[BenchmarkDefinition, ...]) -> None:
    for definition in definitions:
        definition.function()


def test_report_matrix_contains_only_plotted_cases() -> None:
    keys = benchmark_keys()

    assert len(keys) == len(PROFILE_NAMES) * len(OPERATIONS) * len(ADAPTERS)
    assert {key.profile for key in keys} == set(PROFILE_NAMES)
    assert {key.operation for key in keys} == set(OPERATIONS)
    assert {key.adapter for key in keys} == set(ADAPTERS)


def test_comparison_benchmark_adapters_share_semantics() -> None:
    with closing(BenchmarkSuite(("small",))) as suite:
        definitions = tuple(suite.definition(key) for key in benchmark_keys(("small",)))
        assert len(definitions) == len(OPERATIONS) * len(ADAPTERS)
        assert {item.metadata["operation"] for item in definitions} == set(OPERATIONS)
        _run_all(definitions)

        wire_sizes = suite.cases[0].wire_sizes
        assert wire_sizes["msgspec_serde"] == wire_sizes["python_flatbuffers"]
        assert all(size > 0 for size in wire_sizes.values())

        with closing(
            BenchmarkSuite(
                ("small",),
                generated_root=suite.generated_root,
            )
        ) as shared:
            assert all(
                shared.definition(key).function() is not None
                for key in benchmark_keys(("small",))
            )
