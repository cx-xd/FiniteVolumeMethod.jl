# Shared workflow enrollment handoff

- Issue and acceptance criteria: Authorized 16-repository shared-workflow rollout; the dedicated [Work tracking issue](https://github.com/S-AM-I/agents/issues/1) is maintained by the parent coordinator. Issues are disabled on this public research fork, so tracking uses S-AM-I/agents without changing repository settings. This repository's criteria are shared guidance, both pinned core Spec Kit integrations, project-specific principles, real validation commands, selector recovery and maintained evidence. Installation does not implement a scientific backlog item.
- Repository: `cx-xd/FiniteVolumeMethod.jl`.
- Absolute checkout: `/home/sami/Code/github.com/cx-xd/FiniteVolumeMethod.jl-shared-workflow`.
- Branch: `chore/shared-workflow`; enrollment commit: `c0257025b53749db42669f5c82ebffb7ca66d39a`. This evidence-only handoff commit follows it; use `git rev-parse HEAD` to record later transfer state. Explicit fetched base: `origin/main` at `7dbf886f88107c177312cd20958f3dcc0a650f7e`.
- Uncommitted changes: none at the tested enrollment commit; this handoff is the only subsequent authored change. Existing main and other worktrees were preserved.
- Feature/change directory and selector: `docs/shared-workflow.md`; no active scientific feature. Validation used a disposable `specs/workflow-enrollment-validation` with spec/plan/tasks on the nonnumeric task branch. Its artifacts and generated ignored `.specify/feature.json` pointer were removed afterward.
- Completed work: AGENTS.md shared guidance, CLAUDE.md entry point, authored constitution and workflow/check documentation, unmodified generated integration assets and original pinned MIT license. No optional extensions, publishing, hooks, autosave, background jobs, CI changes or scientific behavior changes.
- Checks: commands below ran at `c0257025b53749db42669f5c82ebffb7ca66d39a`; all exited 0. Integration status reported both clients, Codex default, zero modified/missing managed files and safe concurrent installation.
- Outstanding work and limitations: scientific tests/builds/campaigns deliberately skipped for this documentation-only setup; no dependencies installed. Check fresh interactive client discovery in the next session. Read docs/shared-workflow.md for native prerequisites and scientific limitations. Parent coordinates tracking, independent review and any authorized PR/merge; nothing has been pushed or merged here.
- Next action: review the scoped enrollment diff and link this handoff from the Work record; preserve the exact branch/path on transfer.
- Responsibility: Codex scientific-rollout implementer owns this preparation under the parent coordinator's authorized delegation. No transfer to another implementer has been made by this record.
- PR/review/merge evidence: no PR or merge at this stage; prepared local changes are ready for review, not Done.
- Updated: 2026-10-03T00:11:13+01:00 (Europe/London).

## Executed checks

- `specify integration status` — exit 0.
- `bash -n .specify/scripts/bash/check-prerequisites.sh` — exit 0.
- `bash -n .specify/scripts/bash/common.sh` — exit 0.
- `bash -n .specify/scripts/bash/create-new-feature.sh` — exit 0.
- `bash -n .specify/scripts/bash/resolve-template.sh` — exit 0.
- `bash -n .specify/scripts/bash/setup-plan.sh` — exit 0.
- `bash -n .specify/scripts/bash/setup-tasks.sh` — exit 0.
- `SPECIFY_FEATURE_DIRECTORY=specs/workflow-enrollment-validation .specify/scripts/bash/check-prerequisites.sh --json --require-spec --require-tasks --include-tasks` — exit 0.
- `julia --startup-file=no --history-file=no -e 'using TOML; TOML.parsefile("Project.toml"); println("Project.toml parsed")'` — exit 0.
- `make help` — exit 0.
- `git diff --check` — exit 0.

Independent review found the root ignore pattern also suppressed the generated
`.specify/.gitignore`. A narrow exception now preserves this managed file in Git;
checkout-local feature pointers remain ignored in clean clones.
