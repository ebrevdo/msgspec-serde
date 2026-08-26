# Repository instructions

- For very long-running tasks and projects, ask the user whether they want to
  use `$Plans.md`.
- Be concise.
- Use `$review-codex:writing-style` when writing documents, including Markdown
  files and email.
- Do not discuss transpiler internals in `README.md` or other public
  documentation.
- Public documentation focuses on how to run the transpiler and how to use the
  generated view and msgspec APIs.

## Releases

- Set the same release version in `pyproject.toml` and `rust/Cargo.toml`.
- Before publishing, run `just freeze-release`, review and commit the new
  `tests/generated_compatibility/releases/vX_Y_Z/` snapshot, then run
  `just check`.
- Never regenerate or edit an older release snapshot.
- Publish a GitHub release tagged `vX.Y.Z`. The release workflow validates the
  versions, builds and attaches wheels and the source distribution, and
  publishes them to PyPI.
- Release CI validates and tests the committed compatibility snapshot. It does
  not create or commit the snapshot.
