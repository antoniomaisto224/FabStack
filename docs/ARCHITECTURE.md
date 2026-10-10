# FabStack architecture

FabStack uses a compatibility-first package structure. Product behavior,
serialized `.fab` data, application resources, and the existing launch commands
remain stable while implementation code is grouped by responsibility.

The staged work that turns these package boundaries into independently testable
components is tracked in [the professionalization roadmap](REFACTORING_ROADMAP.md).

## Package map

```text
fabstack/
├── application.py                 Qt application bootstrap
├── resource_paths.py              side-effect-free bundle lookup
├── resources.py                   bundle and per-user data paths
├── terminology.py                 stable operation keys and display labels
├── version.py                     runtime version
├── assets/                         installed defaults and UI images
├── assistant/
│   └── planner.py                 local-model adapter and deterministic planner
├── config/
│   └── visualization.py           shared display limits and clamps
├── domain/
│   └── cross_section.py           physical process and geometry engine
├── materials/
│   ├── chemistry.py               chemical-name display helpers
│   └── schema.py                  material-library validation
├── operations/
│   ├── contracts.py               canonical keys and boundary TypedDicts
│   └── execution.py               route-aware mutation kernels and replay
├── persistence/
│   └── process_file.py            restricted loading and atomic saving
├── rendering/
│   ├── constants.py               shared rendering and export constants
│   ├── contracts.py               typed host capabilities for render services
│   ├── scaling.py                 pure scalar display transforms
│   ├── scene_constraints.py       process-wide display constraint kernel
│   └── segment_targets.py         per-column target-row kernel
└── presentation/
    ├── copy.py                     legacy presentation terminology re-export
    ├── main_window.py              MainWindow composition and process replay
    ├── dialogs/
    │   └── material_library.py     material-library editor
    └── mixins/
        ├── plotting_export_mixin.py stable aggregate compatibility facade
        ├── display_profile_mixin.py display constraints and mapping
        ├── visual_adjustment_mixin.py visual override controls
        ├── canvas_rendering_mixin.py plotting and canvas interaction
        ├── export_actions_mixin.py  image, document, and text exports
        └── ...                     other focused Qt behavior groups
```

Tests are grouped as `tests/unit`, `tests/regression`, `tests/stress`, and
`tests/fixtures`. Root validation modules are thin compatibility runners.

## Dependency direction

```text
application -> presentation -> assistant / persistence / materials / config
                           -> operations / rendering / domain

persistence -> NumPy only
domain      -> NumPy only
rendering   -> config / NumPy only
operations  -> terminology / NumPy only
```

The domain and persistence layers must not import Qt. Presentation modules may
coordinate the lower layers. Cross-mixin calls are part of `MainWindow`'s
current internal interface; keep the base-class order in
`fabstack/presentation/main_window.py` stable during behavior-preserving work.

## Stable contracts

- `.fab` format version, dictionary keys, operation keys, NumPy shapes, and
  replay order are compatibility boundaries.
- Resource filenames and their flat PyInstaller bundle layout remain stable;
  source and installed-package copies live under `fabstack/assets`.
- User data remains in the platform-specific `FabStack` directory, with
  `FABSTACK_DATA_DIR` available as an override.
- Root launchers and import facades remain available for existing tooling.
- New code should use `fabstack.*` imports; compatibility facades should not
  acquire new behavior.

Compatibility facades preserve the application classes, helpers, and uppercase
constants listed in their explicit `__all__`. Names that happened to be imported
inside the former implementation modules (for example `os` or Qt classes) are
implementation details, not part of the supported facade API.

## Fine-tuning guidance

The historical plotting/export mixin is now a stable aggregate facade over four
responsibility-focused mixins. Its two large display algorithms live in
presentation-free rendering services behind exact-signature mixin delegates and
narrow host protocols. Keep `PlottingExportMixin`, those delegate signatures,
and `MainWindow`'s direct base order stable while callers migrate. Operation
keys and mutation calls now have a shared contract/execution boundary; route-
specific validation, presentation, and legacy behavior remain explicit.
Planner and dialog modules still require the same facade-first decomposition.
