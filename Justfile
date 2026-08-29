# List available recipes.
default:
    @just --list

# Install the locked Python project and development dependencies.
sync:
    uv sync --locked --all-groups

# Run the Python test suite.
test:
    uv run --locked --group benchmark pytest -q

# Run Python linting and type checking.
lint:
    uv run --locked ruff check .
    uv run --locked ty check --no-progress

# Check Rust formatting, linting, and tests.
rust-check:
    cargo fmt --manifest-path rust/Cargo.toml -- --check
    cargo clippy --manifest-path rust/Cargo.toml --all-targets --locked -- -D warnings
    cargo test --manifest-path rust/Cargo.toml --locked

# Run all Python and Rust checks.
check: lint test rust-check

# Build the wheel and source distribution.
build:
    uv build

# Run the benchmark suite and pass through optional arguments.
benchmark *args:
    uv run --locked --group benchmark python -m benchmarks.run_pinned -- python -m benchmarks.benchmark {{args}}

# Generate the benchmark report images from saved benchmark outputs.
benchmark-report *args:
    uv run --locked --group benchmark python -m benchmarks.report {{args}}

# Run msgspec's vendored JSON or MessagePack benchmark.
benchmark-upstream-encodings *args:
    uv run --locked --group upstream-benchmark python -m benchmarks.run_pinned --cwd benchmarks/upstream_msgspec -- python -m benchmarks.bench_encodings {{args}}

# Profile upstream Structs at 16- and 256-value vector sizes by default.
benchmark-upstream-profile *args:
    uv run --locked --group upstream-benchmark python -m benchmarks.run_pinned --cwd benchmarks/upstream_msgspec -- python -m benchmarks.profile_msgspec_serde {{args}}

# Freeze generated code for the current release.
freeze-release:
    uv run python tools/freeze_generated_compatibility.py

# Validate the committed generated-code snapshot.
check-release:
    python tools/freeze_generated_compatibility.py --check

# Run every local release check and build distributions.
release-check: check check-release build
