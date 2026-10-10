# FabStack professionalization roadmap

This roadmap turns the current package organization into real internal
boundaries. Refactors are facade-first and behavior-preserving: each existing
class, import path, serialized key, operation name, and UI workflow remains
available while its implementation is extracted behind explicit contracts.

## Current hotspots

| Priority | Area | Current pressure | Target boundary |
| --- | --- | --- | --- |
| Complete | Plotting and export | The former 5,333-line mixin is separated into display-profile, visual-adjustment, canvas-rendering, and export-action mixins behind its original facade. Its two largest algorithms are presentation-free rendering services with typed host contracts. | Keep the aggregate facade, delegate signatures, method surface, and rendering parity gates stable. |
| Complete | Operation lifecycle | The ten stable operation keys and all domain mutation calls are centralized behind typed, route-aware execution boundaries. Assistant execution continues through the interactive path; persistence and replay retain their explicit legacy validation/default policies. | Keep shared keys, adapters, host protocols, inverse-AST gates, and route-specific policy tests stable. |
| Critical | `MainWindow` | A 2,570-line composition root also owns overlays, settings, recipes, replay, and history. | Composition and lifecycle only; feature controllers expose explicit host contracts. |
| High | Planner | Model transport, conversation repair, parsing, policy, validation, and deterministic planning share 2,981 lines. | Planner facade over isolated backend, parser, resolver, policy, and validation modules. |
| High | Material library | Qt construction, storage, catalog mutation, six compatibility editors, restrictions, and settings share one dialog. | Transactional dialog plus catalog, compatibility, and restriction components. |
| High | UI mixins | Operations, assistant, and material mixins rely on undocumented `self` attributes and methods owned elsewhere. | Aggregate compatibility mixins composed from typed, focused capabilities. |

## Staged execution

1. **Complete:** freeze public method surfaces and behavioral baselines.
2. **Complete:** decompose plotting/export behind its stable aggregate mixin.
3. **Complete:** extract rendering constraints and display mapping behind wrapper methods.
4. **Complete:** define the operation command contract and centralize interactive,
   replay, assistant, and persisted operation mutation paths.
5. **In progress:** split the planner bottom-up while keeping `LocalStepPlanner` and its monkeypatch
   seams stable.
6. Make the material dialog transactional and replace duplicated compatibility
   editors with one tested definition.
7. Split operation, assistant, material, layout, and theme UI responsibilities
   behind aggregate mixins and explicit host protocols.
8. Reduce `MainWindow` to composition, lifecycle, and stable delegating facades.
9. Isolate documented domain algorithms and schemas only after all presentation
   callers use explicit boundaries.

## Required gate for every stage

- Normalized AST or wrapper-delegation parity for mechanically moved methods.
- Exact `.fab` fixture compatibility and stable serialized keys.
- Exact planner command output and manual-lithography pause behavior.
- Deep equality for rendering profiles, grids, row heights, provenance roles, and
  export metadata—not screenshot similarity alone.
- Unit/architecture suite, all 62 rendering regressions, and planner/UI stress
  suite remain green.
- Package asset installation, PyInstaller build, and frozen launch remain green
  for release-relevant stages.
- Root compatibility facades and `MainWindow`'s direct base order remain stable.

## Definition of professional completion

The goal is complete only when all of the following are demonstrated:

1. Every operation has one authoritative semantic path across UI, replay,
   assistant execution, and persisted process loading.
2. Required collaborators are statically declared; required behavior is not
   discovered through `hasattr` or silently skipped.
3. Production functions stay below 150 lines and complexity 20 unless a pure,
   directly tested algorithm has a documented waiver. UI/controller methods stay
   below 100 lines.
4. Rendering constraints, display mapping, plotting, interaction, and exporters
   are independently testable.
5. Process files, project sessions, planner commands, history records, rendering
   profiles, and material libraries each have one schema authority.
6. Cancelling a dialog causes no domain mutation or persistent write.
7. No `except Exception: pass`; broad boundary catches log actionable context.
8. Global Qt, Matplotlib, and filesystem mutations occur only in explicit
   bootstrap or persistence actions.
9. Substantial duplicated behavior—especially operation dispatch and material
   compatibility editing—is eliminated.
10. All behavior, package, build, and facade gates pass after the final audit.

Optimization (caching, dispatch tables, regex indexes, immutable render inputs,
and data-driven compatibility widgets) follows structural parity. It must not be
mixed into a mechanical extraction.
