# FiniteVolumeMethod.jl (research fork)

Julia finite volume solver library; a research fork of Daniel VandenHeuvel's FiniteVolumeMethod.jl.

## Constraints

- This fork is not the registered package: `Pkg.add("FiniteVolumeMethod")` installs upstream. Registering the fork requires a new name and UUID.
- Preserve Daniel VandenHeuvel's license and upstream attribution.
- Scientific maturity claims follow `docs/src/capability_matrix.md`; CPU Float64 is the publication baseline and GPU support has limited audited scope. Known failures live in `test/KNOWN_FAILURES.md`.
- Only current Julia stable and LTS are release-supported targets.
- GitHub Actions are disabled for this fork; the local `make` lanes are the authoritative checks.

## Commands

- Docker lanes (need Docker with 12+ GB memory; run `make ci-build` first and after `Project.toml` changes): `make ci-fast`, `make ci-smoke`, `make ci-full-evidence`, `make ci-performance`, `make ci-release-audit`, `make ci-test`, `make ci-docs-ci`.
- Single test file: `TEST_FILE=test/geometry/geometry.jl make ci-test-file`.
- Formatting (Runic): `make ci-format` / `make ci-format-fix`; `scripts/install_hooks.sh` installs a Runic pre-commit hook.
- Published benchmarks run natively (no Docker): `make ci-published-benchmarks`.
- `make help` lists all targets.
