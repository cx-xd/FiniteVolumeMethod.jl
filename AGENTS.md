# Shared project guidance

Read README.md, docs/src/capability_matrix.md, docs/src/migration/v4.md and test/KNOWN_FAILURES.md. Preserve Daniel VandenHeuvel’s license and upstream attribution. This fork is not the registered package: Pkg.add("FiniteVolumeMethod") selects upstream, and registration requires a new name/UUID. Scientific maturity follows the capability matrix; CPU Float64 remains the publication baseline and GPU support has limited audited scope. Keep current stable/LTS support and local scientific evidence lanes.

For tasks tracked in GitHub [Work](https://github.com/users/S-AM-I/projects/11), read `/home/sami/.agents/docs/workflow.md` and use the canonical `github-issues` skill. A backlog entry is context, not authorization. Read the maintained handoff and current responsibility before editing; use one active implementer per issue and transfer responsibility explicitly. Preserve existing task records rather than duplicating publishers.

Substantial authorized features use core Spec Kit specifications, plans and tasks. Small fixes use a concise plan and relevant checks. Reuse approved native/Spec Kit artifacts and authorization; Superpowers supports them without competing plans or repeated approvals. Recover from the recorded feature path using `SPECIFY_FEATURE_DIRECTORY` or ignored `.specify/feature.json`, independently of the branch name.

Use task-based branches without client identity prefixes and sibling worktrees named `<repo>-<task>` under `/home/sami/Code/github.com/cx-xd/`. Preserve the exact branch and absolute checkout on handoff. Update [the maintained handoff](docs/shared-workflow-handoff.md); feature-specific tasks may keep their handoff beside their artifacts and link it there. Completion requires applicable acceptance criteria, review and merge evidence.

[Workflow setup and checks](docs/shared-workflow.md) documents this enrollment. No hooks, autosave or background jobs are enrolled here; current shared hook enrollment is only `/home/sami/.agents`. Preserve licenses and native tooling; keep credentials and live client settings outside Git.
