# Shared workflow and Spec Kit setup

This repository is enrolled in core Spec Kit for Codex and Claude Code. [AGENTS.md](../AGENTS.md) supplies shared guidance and [CLAUDE.md](../CLAUDE.md) imports it. Existing scientific documents and conventions remain authoritative.

## Provenance and scope

Specify CLI 1.0.13 generated `.specify/`, `.agents/skills/speckit-*` and `.claude/skills/speckit-*` from the shared pin `f1a548a39dba4e5e8600de1d2e0d3ff0c468d2a9` (GitHub Spec Kit v1.0.13, MIT). The original upstream MIT license is retained at `.specify/LICENSE`. Preserve generated resources; the [constitution](../.specify/memory/constitution.md) is authored project guidance. Pin/update guidance lives in `/home/sami/.agents/sources.md`.

```sh
specify init --here --integration codex --integration-options=--skills --script sh --force --non-interactive
specify integration install claude
specify integration status
```

The nonempty checkout was inspected before initialization; no prior `.specify` installation was overwritten. Do not blindly repeat forced initialization over authored artifacts. Codex is the CLI default and both integrations coexist. No optional extension, naming hook, publisher, live client setting, autosave or background job was installed. Enrollment does not restore retired plans or authorize scientific implementation.

## Start and recover

Open a fresh client session in the intended checkout for skill discovery. Codex uses `$speckit-specify`, `$speckit-plan`, `$speckit-tasks`, `$speckit-implement`; Claude uses `/speckit-specify`, `/speckit-plan`, `/speckit-tasks`, `/speckit-implement`. Integration status verifies CLI registration/files; fresh interactive discovery remains a next-session check.

Read the issue, acceptance criteria, responsibility and [maintained handoff](shared-workflow-handoff.md). For substantial authorized features, reuse or create `specs/<feature>/spec.md`, `plan.md` and `tasks.md`. Preserve the recorded branch and absolute worktree on transfer. Select the actual recorded feature explicitly:

```sh
export SPECIFY_FEATURE_DIRECTORY=specs/example-feature
.specify/scripts/bash/check-prerequisites.sh --json --paths-only
```

The example is not an active feature. `.specify/feature.json` is ignored checkout-local state, not shared task identity; numeric branch naming is unnecessary. Keep exact check evidence and outstanding limitations with the task artifacts. Work records link to them; installation itself neither publishes tasks nor migrates another tracker.

## Project policy and actual validation commands

Read README.md, docs/src/capability_matrix.md, docs/src/migration/v4.md and test/KNOWN_FAILURES.md. Preserve Daniel VandenHeuvel’s license and upstream attribution. This fork is not the registered package: Pkg.add("FiniteVolumeMethod") selects upstream, and registration requires a new name/UUID. Scientific maturity follows the capability matrix; CPU Float64 remains the publication baseline and GPU support has limited audited scope. Keep current stable/LTS support and local scientific evidence lanes.

Commands from the existing README/build/CI configuration, selected according to the actual change:

```sh
make ci-fast
make ci-smoke
make ci-full-evidence
make ci-format
make ci-performance
make ci-release-audit
make ci-published-benchmarks
```

The current README states Actions are disabled for this fork; its local lanes govern validation even though .github/workflows/README.md still describes automatic triggers. This enrollment does not enable CI or alter workflows. Docker lanes need prebuilt images/dependencies and substantial memory; full scientific evidence and benchmarks are outside this documentation-only change.

Workflow-only validation checks `specify integration status`, Bash syntax of generated scripts, explicit feature recovery using a disposable selector on `chore/shared-workflow`, and `git diff --check`. See the maintained handoff for actual executed checks and skipped scientific validation. Hooks/background enrollment remains confined to `/home/sami/.agents`.
