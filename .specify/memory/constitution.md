# FiniteVolumeMethod.jl research fork constitution

## Project principles

### Scientific scope and provenance

Read README.md, docs/src/capability_matrix.md, docs/src/migration/v4.md and test/KNOWN_FAILURES.md. Preserve Daniel VandenHeuvel’s license and upstream attribution. This fork is not the registered package: Pkg.add("FiniteVolumeMethod") selects upstream, and registration requires a new name/UUID. Scientific maturity follows the capability matrix; CPU Float64 remains the publication baseline and GPU support has limited audited scope. Keep current stable/LTS support and local scientific evidence lanes.

### Native architecture and reproducibility

Keep the established language, package boundaries, native validation commands and reproducible inputs. Changes to scientific assumptions, datasets or outputs need explicit scope and appropriate evidence; workflow installation does not establish scientific validity. Preserve upstream licenses and attribution.

### Proportionate validation

Run relevant existing checks from `docs/shared-workflow.md`. Record exact commands, outcomes and skipped prerequisites. Documentation-only setup needs integration, shell syntax, selector recovery and diff checks; it does not require dependency installation or expensive simulations.

### Authorized shared execution

Use Work for tasks tracked there, one active implementer per issue, and a maintained handoff with exact path, branch, feature selector and evidence. Substantial authorized features use specification, plan and tasks; small fixes use concise plans and appropriate checks. Reuse approved artifacts and authorization rather than generating competing plans.

## Governance

Read AGENTS.md and existing project documents before changing scope. Preserve native scientific conventions and reconcile conflicts explicitly. A task is complete only after its acceptance criteria, applicable review and merge requirements pass. Hooks, autosave, publishers and background jobs require separate enrollment; none is introduced here. Amend project principles with an explicit rationale in the reviewed change.

**Version**: 1.0.0 | **Ratified**: 2026-10-03 | **Last Amended**: 2026-10-03
