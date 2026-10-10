"""Validation scenarios for FabStack dimension preservation.

Run from the project root:
    QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=/private/tmp/mpl ./venv/bin/python dimension_preservation_validation.py
"""

import os
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VALIDATION_OUTPUT_DIR = Path(
    os.environ.get("FABSTACK_VALIDATION_OUTPUT_DIR", "")
    or tempfile.mkdtemp(prefix="fabstack-validation-")
).resolve()
VALIDATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIXTURES_DIR = ROOT / "tests" / "fixtures"

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")
os.environ.setdefault("FABSTACK_DATA_DIR", str(VALIDATION_OUTPUT_DIR / "user-data"))

from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox, QPushButton, QSpinBox  # noqa: E402

from fabstack.presentation.main_window import MainWindow  # noqa: E402
from fabstack.presentation.mixins.plotting_export_mixin import (  # noqa: E402
    FEATURE_EMPHASIS_DEFAULT,
    FEATURE_EMPHASIS_MAX,
    FEATURE_EMPHASIS_MIN,
    MAX_LAYER_PX,
    MAX_THICKNESS_NM,
    MIN_LAYER_PX,
    MIN_VOID_PX,
    MIN_SUBSTRATE_CORE_PX,
    MIN_THICKNESS_NM,
    ORDER_MARGIN_PX,
    VISUAL_RENDER_SCALE_BASE,
    thickness_to_raw_visual_px,
)
from fabstack.domain.cross_section import CrossSection  # noqa: E402
from fabstack.persistence.process_file import load_process_file  # noqa: E402


TOL_UM = 1e-6
BASE_SUBSTRATE_MATERIAL = "MEMpax"


def _patch_message_boxes():
    QMessageBox.warning = lambda *args, **kwargs: QMessageBox.Ok
    QMessageBox.information = lambda *args, **kwargs: QMessageBox.Ok
    QMessageBox.question = lambda *args, **kwargs: QMessageBox.Yes


def _new_window():
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.update_plot = lambda *args, **kwargs: None
    return app, win


def _run_commands(name, commands):
    _app, win = _new_window()
    for command in commands:
        ok, message = win._execute_assistant_command(command)
        if not ok:
            raise AssertionError(f"{name}: command failed: {message}")
    path = VALIDATION_OUTPUT_DIR / f"{name}.fab"
    win._save_process_to_path(str(path))
    return win, path


def _material_stats(state, material_names, material):
    if material not in material_names:
        return None
    grid = state.get("grid")
    heights = state.get("row_heights_um")
    if grid is None or heights is None:
        return None
    grid = np.asarray(grid)
    heights = np.asarray(heights, dtype=float)
    if heights.shape[0] != grid.shape[0]:
        heights = np.ones(grid.shape[0], dtype=float)
    material_index = material_names.index(material)
    values = np.asarray([
        float(np.sum(heights[grid[:, x] == material_index]))
        for x in range(grid.shape[1])
    ])
    nonzero = values[values > TOL_UM]
    if nonzero.size == 0:
        return None
    return {
        "count": int(nonzero.size),
        "min": float(nonzero.min()),
        "median": float(np.median(nonzero)),
        "max": float(nonzero.max()),
        "sum": float(values.sum()),
    }


def _material_column_heights(state, material_names, material):
    if material not in material_names:
        raise AssertionError(f"Missing material {material!r}")
    grid = np.asarray(state.get("grid"))
    heights = np.asarray(state.get("row_heights_um"), dtype=float)
    if grid.ndim != 2 or heights.shape[0] != grid.shape[0]:
        raise AssertionError(f"Invalid state geometry for {state.get('title', '')!r}")
    material_index = material_names.index(material)
    return np.asarray([
        float(np.sum(heights[grid[:, x] == material_index]))
        for x in range(grid.shape[1])
    ])


def _assert_history_substrate_max(win, substrate_material, expected_um):
    for idx, state in enumerate(win.history):
        values = _material_column_heights(state, win.material_names, substrate_material)
        max_um = float(values.max()) if values.size else 0.0
        if abs(max_um - float(expected_um)) > TOL_UM:
            raise AssertionError(
                f"History state {idx} {state.get('title', '')!r} substrate max "
                f"should be {expected_um} um, got {max_um}"
            )


def _state_by_operation(history, operation, occurrence=1):
    seen = 0
    for state in history:
        if state.get("operation") == operation:
            seen += 1
            if seen == occurrence:
                return state
    raise AssertionError(f"Missing operation {operation!r} occurrence {occurrence}")


def _dynamic_rows_for_um(
    win,
    height_um,
    profile=None,
    material_idx=None,
    *,
    is_substrate_core=False,
):
    if profile is None:
        profile = win._process_display_adjustment_profile()
    return int(
        win._display_segment_target_height(
            float(height_um),
            material_idx=material_idx,
            process_profile=profile,
            is_substrate_core=is_substrate_core,
        )
    )


def _material_column_heights_from_xsec(xsec, material_index: int):
    grid = np.asarray(xsec.grid)
    heights = np.asarray(xsec.copy_row_heights(), dtype=float)
    return np.asarray([
        float(np.sum(heights[grid[:, x] == material_index]))
        for x in range(grid.shape[1])
    ])


def _contiguous_clusters(cols):
    cols = [int(col) for col in cols]
    if not cols:
        return []
    clusters = []
    start = prev = cols[0]
    for col in cols[1:]:
        if col == prev + 1:
            prev = col
            continue
        clusters.append((start, prev))
        start = prev = col
    clusters.append((start, prev))
    return clusters


def test_isotropic_physical_front_geometry():
    compact = CrossSection(
        width_px=100,
        height_px=575,
        substrate_index=1,
        void_index=0,
        substrate_thickness_px=525,
        substrate_thickness_um=525.0,
        top_headroom_um=50.0,
    )
    compact_heights = compact.copy_row_heights()
    if compact.grid.shape[0] != 2:
        raise AssertionError(f"Substrate should initialize compactly, got shape={compact.grid.shape}")
    if abs(float(np.sum(compact_heights[compact.grid[:, 0] == 1])) - 525.0) > TOL_UM:
        raise AssertionError("Compact substrate should preserve exact physical thickness")

    compact.deposit_physical(2, 0.001)
    compact_heights = compact.copy_row_heights()
    film_um = float(np.sum(compact_heights[compact.grid[:, 0] == 2]))
    if abs(film_um - 0.001) > TOL_UM:
        raise AssertionError(f"1 nm film should remain exact, got {film_um:g} um")

    shallow = CrossSection(
        width_px=100,
        height_px=150,
        substrate_index=1,
        void_index=0,
        substrate_thickness_px=100,
    )
    shallow_mask = shallow.window_mask(0.40, 0.60)
    shallow.etch_physical(1, 0.2, mask=shallow_mask, mode="isotropic")
    shallow_heights = _material_column_heights_from_xsec(shallow, 1)
    shallow_changed = np.where(shallow_heights < shallow_heights.max() - TOL_UM)[0]
    shallow_open = np.where(shallow_mask)[0]
    if abs(float(shallow_heights.min()) - 99.8) > TOL_UM:
        raise AssertionError(f"0.2 um isotropic etch min height wrong: {shallow_heights.min():g}")
    if not np.array_equal(shallow_changed, shallow_open):
        raise AssertionError(
            f"0.2 um isotropic etch should not force sub-grid lateral undercut: "
            f"changed={shallow_changed}, open={shallow_open}"
        )

    aniso = CrossSection(
        width_px=120,
        height_px=180,
        substrate_index=1,
        void_index=0,
        substrate_thickness_px=120,
    )
    deep_mask = aniso.window_mask(0.45, 0.55)
    aniso.etch_physical(1, 20.0, mask=deep_mask, mode="anisotropic")
    aniso_heights = _material_column_heights_from_xsec(aniso, 1)
    aniso_changed = np.where(aniso_heights < aniso_heights.max() - TOL_UM)[0]
    deep_open = np.where(deep_mask)[0]
    if not np.array_equal(aniso_changed, deep_open):
        raise AssertionError(
            f"Anisotropic etch should stay inside opening: changed={aniso_changed}, open={deep_open}"
        )

    iso = CrossSection(
        width_px=120,
        height_px=180,
        substrate_index=1,
        void_index=0,
        substrate_thickness_px=120,
    )
    iso.etch_physical(1, 20.0, mask=deep_mask, mode="isotropic")
    iso_heights = _material_column_heights_from_xsec(iso, 1)
    iso_changed = np.where(iso_heights < iso_heights.max() - TOL_UM)[0]
    left_undercut = int(deep_open[0] - iso_changed[0])
    right_undercut = int(iso_changed[-1] - deep_open[-1])
    if not (18 <= left_undercut <= 20 and 18 <= right_undercut <= 20):
        raise AssertionError(
            f"20 um isotropic etch should undercut about 20 columns, got "
            f"left={left_undercut}, right={right_undercut}"
        )
    if abs(float(iso_heights.min()) - 100.0) > TOL_UM:
        raise AssertionError(f"20 um isotropic etch vertical depth wrong: min={iso_heights.min():g}")

    buried = CrossSection(
        width_px=60,
        height_px=120,
        substrate_index=1,
        void_index=0,
        substrate_thickness_px=80,
    )
    buried.deposit_physical(2, 5.0)
    before_buried = _material_column_heights_from_xsec(buried, 1)
    buried.etch_physical(1, 10.0, mask=np.ones(buried.width, dtype=bool), mode="isotropic")
    after_buried = _material_column_heights_from_xsec(buried, 1)
    if not np.allclose(before_buried, after_buried, atol=TOL_UM, rtol=0.0):
        raise AssertionError("Isotropic etch should not reach buried target material through unrelated top layer")

    thin_target = CrossSection(
        width_px=60,
        height_px=120,
        substrate_index=1,
        void_index=0,
        substrate_thickness_px=80,
    )
    thin_target.deposit_physical(2, 2.0)
    thin_target.etch_physical(2, 20.0, mask=np.ones(thin_target.width, dtype=bool), mode="isotropic")
    if _material_column_heights_from_xsec(thin_target, 2).max() > TOL_UM:
        raise AssertionError("Over-depth isotropic etch should remove only available target film")
    substrate_after = _material_column_heights_from_xsec(thin_target, 1)
    if abs(float(substrate_after.min()) - 80.0) > TOL_UM:
        raise AssertionError("Over-depth isotropic etch should not continue into substrate")

    separate = CrossSection(
        width_px=100,
        height_px=150,
        substrate_index=1,
        void_index=0,
        substrate_thickness_px=100,
    )
    separate_mask = separate.window_mask(0.20, 0.25) | separate.window_mask(0.75, 0.80)
    separate.etch_physical(1, 5.0, mask=separate_mask, mode="isotropic")
    separate_heights = _material_column_heights_from_xsec(separate, 1)
    separate_changed = np.where(separate_heights < separate_heights.max() - TOL_UM)[0]
    clusters = _contiguous_clusters(separate_changed)
    if len(clusters) != 2:
        raise AssertionError(f"Separate isotropic windows should remain separate, got clusters={clusters}")

    if not iso.visual_geometry:
        raise AssertionError("Isotropic etch should record vector front metadata")
    first_front = iso.visual_geometry[0]
    if first_front.get("type") != "isotropic_etch_front" or not first_front.get("polylines"):
        raise AssertionError(f"Invalid isotropic visual geometry metadata: {first_front}")


def test_adaptive_display_layer_order():
    _app, win = _new_window()
    win.material_names = ["Void", "Silicon (Si)", "Thin", "Thick"]
    win.material_colors = ["#ffffff", "#aaaaaa", "#facc15", "#2563eb"]

    xsec = CrossSection(
        width_px=80,
        height_px=575,
        substrate_index=1,
        void_index=0,
        substrate_thickness_px=525,
        substrate_thickness_um=525.0,
        top_headroom_um=50.0,
    )
    xsec.deposit_physical(2, 1.0)
    xsec.deposit_physical(3, 8.0)
    state = {
        "grid": xsec.copy_state(),
        "row_heights_um": xsec.copy_row_heights(),
        "visual_geometry": xsec.copy_visual_geometry(),
        "title": "adaptive display",
    }
    win.xsec = xsec
    win.history = [state]
    display_grid = win._display_grid_auto(
        state["grid"],
        row_heights_um=state["row_heights_um"],
        history_index=0,
    )
    thin_rows = int(np.sum(display_grid[:, 0] == 2))
    thick_rows = int(np.sum(display_grid[:, 0] == 3))
    substrate_rows = int(np.sum(display_grid[:, 0] == 1))
    if thin_rows <= 0:
        raise AssertionError("1 um layer should remain visible")
    if thick_rows <= thin_rows:
        raise AssertionError(
            f"8 um layer should display thicker than 1 um layer, got {thick_rows} <= {thin_rows}"
        )
    if substrate_rows <= thick_rows:
        raise AssertionError(
            f"525 um substrate should display thicker than an 8 um layer, got {substrate_rows} <= {thick_rows}"
        )

    # A substrate-only view uses the compact structural-body scale.
    _app2, win2 = _new_window()
    win2.material_names = ["Void", "Silicon (Si)"]
    win2.material_colors = ["#ffffff", "#aaaaaa"]
    xsec_small = CrossSection(
        width_px=80, height_px=250, substrate_index=1, void_index=0,
        substrate_thickness_px=200, substrate_thickness_um=200.0, top_headroom_um=50.0,
    )
    xsec_big = CrossSection(
        width_px=80, height_px=575, substrate_index=1, void_index=0,
        substrate_thickness_px=525, substrate_thickness_um=525.0, top_headroom_um=50.0,
    )
    win2.history = []
    win2.xsec = xsec_small
    small_display = win2._display_grid_auto(
        xsec_small.grid, row_heights_um=xsec_small.copy_row_heights(), history_index=None,
    )
    win2.xsec = xsec_big
    big_display = win2._display_grid_auto(
        xsec_big.grid, row_heights_um=xsec_big.copy_row_heights(), history_index=None,
    )
    small_substrate_rows = int(np.sum(small_display[:, 0] == 1))
    big_substrate_rows = int(np.sum(big_display[:, 0] == 1))
    if big_substrate_rows <= small_substrate_rows:
        raise AssertionError(
            f"A 525um substrate should display thicker than a 200um substrate, "
            f"got {big_substrate_rows} <= {small_substrate_rows}"
        )


def test_visual_scale_control_points():
    # Reference points for the scene-independent fifth-root scale.
    at_min = thickness_to_raw_visual_px(MIN_THICKNESS_NM)
    if abs(at_min - MIN_LAYER_PX) > 1e-9:
        raise AssertionError(f"1nm should map to {MIN_LAYER_PX}px, got {at_min:g}")
    at_max = thickness_to_raw_visual_px(MAX_THICKNESS_NM)
    if abs(at_max - MAX_LAYER_PX) > 1e-9:
        raise AssertionError(f"1000um should map to {MAX_LAYER_PX}px, got {at_max:g}")

    # Only the 1nm readability endpoint is clamped. Above 1mm the same power
    # law continues, rather than flattening at a cap.
    below = thickness_to_raw_visual_px(MIN_THICKNESS_NM / 10.0)
    if abs(below - MIN_LAYER_PX) > 1e-9:
        raise AssertionError(f"Below 1nm should clamp at {MIN_LAYER_PX}px, got {below:g}")
    above = thickness_to_raw_visual_px(MAX_THICKNESS_NM * 10.0)
    expected_above = MAX_LAYER_PX * (10.0 ** 0.20)
    if abs(above - expected_above) > 1e-6:
        raise AssertionError(
            f"Above 1000um should continue the fifth-root law, expected {expected_above:g}px, got {above:g}"
        )

    # Every decade from 1nm through 1mm must be visible and distinct.
    samples_nm = [1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0]
    values = [thickness_to_raw_visual_px(nm) for nm in samples_nm]
    for prev_nm, curr_nm, prev_px, curr_px in zip(samples_nm, samples_nm[1:], values, values[1:]):
        if curr_px <= prev_px:
            raise AssertionError(
                f"Each physical decade must be strictly distinct: "
                f"{prev_nm:g}nm->{prev_px:g}px vs {curr_nm:g}nm->{curr_px:g}px"
            )

    # Power-law ratios are scale-covariant and independent of the scene.
    hundred_um_px = thickness_to_raw_visual_px(100_000.0)
    two_hundred_um_px = thickness_to_raw_visual_px(200_000.0)
    ratio = two_hundred_um_px / hundred_um_px
    expected_ratio = 2.0 ** 0.20
    if abs(ratio - expected_ratio) > 1e-6:
        raise AssertionError(f"200um should follow the fifth-root ratio, got ratio={ratio:g}")

    if MIN_LAYER_PX * VISUAL_RENDER_SCALE_BASE < 6.0:
        raise AssertionError(
            f"The 1nm band must retain at least six visible pixels at default scale, "
            f"got {MIN_LAYER_PX * VISUAL_RENDER_SCALE_BASE:g}"
        )


def test_feature_emphasis_control_range_and_persistence():
    _app, win = _new_window()
    if abs(float(win.feature_scale_factor) - FEATURE_EMPHASIS_DEFAULT) > 1e-9:
        raise AssertionError(f"New processes should start balanced, got {win.feature_scale_factor}")
    if win.feature_scale_slider is not None or win.feature_scale_value_label is not None:
        raise AssertionError("The duplicate feature-emphasis control must not appear beside Substrate")
    if win.feature_scale_overlay_slider.value() != 50:
        raise AssertionError("Canvas feature emphasis should initialise at Balanced · 0.50")

    samples_um = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    rows_by_factor = []
    body_rows = []
    for factor in (FEATURE_EMPHASIS_MIN, FEATURE_EMPHASIS_DEFAULT, FEATURE_EMPHASIS_MAX):
        win.feature_scale_factor = factor
        rows = [win._display_base_feature_rows(value) for value in samples_um]
        if any(curr <= prev for prev, curr in zip(rows, rows[1:])):
            raise AssertionError(
                f"Every decade from 1nm to 1mm must remain distinct at emphasis {factor:g}: {rows}"
            )
        if rows[0] < int(MIN_LAYER_PX):
            raise AssertionError(f"The 1nm endpoint disappeared at emphasis {factor:g}: {rows}")
        rows_by_factor.append(rows)
        body_rows.append(win._display_structural_body_rows(525.0))

    for feature_index, physical_um in enumerate(samples_um):
        factor_rows = [rows[feature_index] for rows in rows_by_factor]
        if factor_rows != sorted(factor_rows):
            raise AssertionError(
                f"Feature emphasis must be monotone for {physical_um:g}um, got {factor_rows}"
            )
    if rows_by_factor[0][-1] >= rows_by_factor[-1][-1]:
        raise AssertionError("Compact and Expanded must visibly differ for a 1mm feature")
    if body_rows != [175, 175, 175]:
        raise AssertionError(
            f"Feature emphasis must not resize a 525um structural substrate, got {body_rows}"
        )

    win.feature_scale_factor = 0.75
    data = win._make_process_dict()
    _app, loaded = _new_window()
    loaded._load_process_dict(data)
    if abs(float(loaded.feature_scale_factor) - 0.75) > 1e-9:
        raise AssertionError("Feature emphasis was not preserved by project save/load")
    if loaded.feature_scale_slider is not None or loaded.feature_scale_value_label is not None:
        raise AssertionError("Loading a project must not restore the removed toolbar control")
    if loaded.feature_scale_overlay_slider.value() != 75:
        raise AssertionError("Loaded canvas feature emphasis does not match the saved value")

    legacy_dynamic = dict(data)
    legacy_dynamic.pop("feature_scale_factor", None)
    legacy_dynamic["visualisation_mode"] = "dynamic"
    loaded._load_process_dict(legacy_dynamic)
    if abs(float(loaded.feature_scale_factor) - FEATURE_EMPHASIS_DEFAULT) > 1e-9:
        raise AssertionError("Legacy dynamic projects should migrate to Balanced · 0.50")

    legacy_frozen = dict(data)
    legacy_frozen.pop("feature_scale_factor", None)
    legacy_frozen["visualisation_mode"] = "finalised"
    loaded._load_process_dict(legacy_frozen)
    if abs(float(loaded.feature_scale_factor) - FEATURE_EMPHASIS_MAX) > 1e-9:
        raise AssertionError("Legacy frozen projects should retain the former Expanded mapping")

    loaded.visualisation_mode = "finalised"
    loaded.visual_thickness_overrides = {"layer:test": 20}
    loaded.visual_thickness_step_overrides = {0: {"layer:test": 20}}
    if not loaded._set_feature_scale_factor(FEATURE_EMPHASIS_DEFAULT):
        raise AssertionError("Changing emphasis should succeed after confirming the frozen-view warning")
    if loaded._normalised_visualisation_mode() != "dynamic":
        raise AssertionError("Changing feature emphasis should return a frozen view to dynamic mode")
    if loaded.visual_thickness_overrides or loaded.visual_thickness_step_overrides:
        raise AssertionError("Changing feature emphasis should clear incompatible frozen overrides")


def _masked_substrate_etch_window(nominal_um, depth_um):
    commands = [
        {"action": "set_substrate", "material": "Silicon (Si)", "thickness_um": float(nominal_um)},
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {
            "action": "operation",
            "operation": "Spincoating",
            "material": "PR_SU-8 Photoresist",
            "thickness_um": 5.0,
        },
        {
            "action": "operation",
            "operation": "Lithography",
            "material": "PR_SU-8 Photoresist",
            "windows": [(0.40, 0.60)],
            "invert_lithography": False,
        },
        {
            "action": "operation",
            "operation": "Etch",
            "material": "Silicon (Si)",
            "depth_um": float(depth_um),
            "etch_mode": "anisotropic",
        },
    ]
    _app, win = _new_window()
    for command in commands:
        ok, message = win._execute_assistant_command(command)
        if not ok:
            raise AssertionError(f"{nominal_um:g}/{depth_um:g}um substrate etch failed: {message}")
    return win, win.history[-1]


def _rendered_substrate_window_rows(win, state):
    state_index = next(idx for idx, item in enumerate(win.history) if item is state)
    profile = win._process_display_adjustment_profile()
    display = np.asarray(win._display_grid_auto(
        np.asarray(state.get("grid")),
        row_heights_um=np.asarray(state.get("row_heights_um"), dtype=float),
        history_index=state_index,
        visual_geometry=state.get("visual_geometry", []),
        process_profile=profile,
    ))
    substrate_index = win.material_names.index("Silicon (Si)")
    outside = int(np.sum(display[:, 0] == substrate_index))
    center = int(np.sum(display[:, 150] == substrate_index))
    return outside, center, outside - center, profile


def test_deep_partial_etch_preserves_visible_substrate_core():
    win, state = _masked_substrate_etch_window(500.0, 200.0)
    physical = _material_column_heights(state, win.material_names, "Silicon (Si)")
    if abs(float(physical[0]) - 500.0) > TOL_UM or abs(float(physical[150]) - 300.0) > TOL_UM:
        raise AssertionError(
            f"A 500um substrate etched by 200um must retain a 300um core, got {physical[0]:g}/{physical[150]:g}"
        )

    displayed_gaps = []
    for factor in (FEATURE_EMPHASIS_MIN, FEATURE_EMPHASIS_DEFAULT, FEATURE_EMPHASIS_MAX):
        win.feature_scale_factor = factor
        outside, center, gap, profile = _rendered_substrate_window_rows(win, state)
        displayed_gaps.append(gap)
        if center < 100:
            raise AssertionError(
                f"The 300um remaining core must stay substantial at emphasis {factor:g}, got {center} rows"
            )
        if center < gap + int(ORDER_MARGIN_PX):
            raise AssertionError(
                f"Physical core 300um > etch 200um must remain visible in that order at {factor:g}: "
                f"core={center}, gap={gap}, outside={outside}"
            )
        expected_gap = int(profile["resolved_feature_rows"][win._display_feature_profile_key(200.0)])
        if gap != expected_gap:
            raise AssertionError(
                f"The rendered trench must use the global 200um feature target, expected {expected_gap}, got {gap}"
            )
    if displayed_gaps != sorted(displayed_gaps) or len(set(displayed_gaps)) != 3:
        raise AssertionError(f"Feature emphasis should progressively resize the 200um etch: {displayed_gaps}")

    shallow, shallow_state = _masked_substrate_etch_window(525.0, 20.0)
    shallow.feature_scale_factor = FEATURE_EMPHASIS_DEFAULT
    shallow_outside, _shallow_core, _shallow_gap, _profile = _rendered_substrate_window_rows(
        shallow, shallow_state
    )
    if shallow_outside != 175:
        raise AssertionError(f"A shallow 525/20um scene should keep the compact 175-row body, got {shallow_outside}")

    deep, deep_state = _masked_substrate_etch_window(500.0, 400.0)
    deep.feature_scale_factor = FEATURE_EMPHASIS_DEFAULT
    _outside, deep_core, deep_gap, _profile = _rendered_substrate_window_rows(deep, deep_state)
    if deep_gap < deep_core + int(ORDER_MARGIN_PX):
        raise AssertionError(
            f"Physical etch 400um > core 100um must remain visible in that order: gap={deep_gap}, core={deep_core}"
        )

    near_through, near_state = _masked_substrate_etch_window(500.0, 499.0)
    near_through.feature_scale_factor = FEATURE_EMPHASIS_DEFAULT
    _outside, near_core, _gap, _profile = _rendered_substrate_window_rows(near_through, near_state)
    if near_core < int(MIN_SUBSTRATE_CORE_PX):
        raise AssertionError(f"A physical 1um residual core must remain visible, got {near_core} rows")

    through, through_state = _masked_substrate_etch_window(500.0, 500.0)
    through.feature_scale_factor = FEATURE_EMPHASIS_DEFAULT
    through_physical = _material_column_heights(through_state, through.material_names, "Silicon (Si)")
    _outside, through_core, _gap, _profile = _rendered_substrate_window_rows(through, through_state)
    if float(through_physical[150]) > TOL_UM or through_core != 0:
        raise AssertionError(
            f"Only a true 500/500um through-etch may remove the core, got physical/display "
            f"{through_physical[150]:g}um/{through_core} rows"
        )


def test_near_tie_feature_order_is_visible_without_exaggeration():
    _app, win = _new_window()
    win.material_names = ["Void", "Silicon (Si)", "Near-tie film"]
    win.material_colors = ["#ffffff", "#aaaaaa", "#ff0000"]
    win.xsec = CrossSection(
        width_px=1,
        height_px=500,
        substrate_index=1,
        void_index=0,
        substrate_thickness_px=500,
        substrate_thickness_um=500.0,
        top_headroom_um=0.0,
    )

    def state(film_um, gap_um):
        return {
            "grid": np.asarray([2, 0, 1], dtype=int).reshape(-1, 1),
            "row_heights_um": np.asarray([film_um, gap_um, 500.0 - gap_um], dtype=float),
            "visual_geometry": [],
        }

    # One process contains both directions of the same close comparison, so
    # the global targets must settle on one consistent, non-contradictory map.
    win.history = [state(10.4, 10.0), state(10.0, 10.4)]
    profile = win._process_display_adjustment_profile()
    if not profile.get("constraint_converged") or profile.get("constraint_violations"):
        raise AssertionError(f"Near-tie constraints did not converge cleanly: {profile}")
    ten = int(profile["resolved_feature_rows"][win._display_feature_profile_key(10.0)])
    ten_point_four = int(profile["resolved_feature_rows"][win._display_feature_profile_key(10.4)])
    difference = ten_point_four - ten
    if difference != 1:
        raise AssertionError(
            f"A real but sub-5% size difference should get one visible row, not an eight-row exaggeration: "
            f"10um={ten}, 10.4um={ten_point_four}"
        )
    if win._display_physical_order_margin(10.0, 10.0) != 0:
        raise AssertionError("Physically equal dimensions must be allowed to display equally")
    if win._display_physical_order_margin(10.5, 10.0) != int(ORDER_MARGIN_PX):
        raise AssertionError("A 5% physical difference should receive the full readable separation")


def _process_wide_scaling_window():
    _app, win = _new_window()
    late_layer_count = 12
    win.material_names = (
        ["Void", "Silicon (Si)", "Thin marker"]
        + [f"Late thick {idx}" for idx in range(late_layer_count)]
    )
    win.material_colors = ["#ffffff"] * len(win.material_names)

    xsec = CrossSection(
        width_px=80,
        height_px=575,
        substrate_index=1,
        void_index=0,
        substrate_thickness_px=525,
        substrate_thickness_um=525.0,
        top_headroom_um=50.0,
    )
    xsec.deposit_physical(2, 0.001)
    early = {
        "grid": xsec.copy_state(),
        "row_heights_um": xsec.copy_row_heights(),
        "visual_geometry": xsec.copy_visual_geometry(),
        "title": "early thin marker",
    }
    for offset in range(late_layer_count):
        xsec.deposit_physical(3 + offset, 20.0)
    late = {
        "grid": xsec.copy_state(),
        "row_heights_um": xsec.copy_row_heights(),
        "visual_geometry": xsec.copy_visual_geometry(),
        "title": "late thick stack",
    }
    win.xsec = xsec
    win.history = [early, late]
    return win, early, late


def test_process_wide_scaling_reaches_previous_steps():
    win, early, _late = _process_wide_scaling_window()
    local_display = win._display_grid_auto(
        early["grid"],
        row_heights_um=early["row_heights_um"],
        history_index=0,
        apply_process_profile=False,
    )
    local_thin = int(np.sum(local_display[:, 0] == 2))

    process_display = win._display_grid_auto(
        early["grid"],
        row_heights_um=early["row_heights_um"],
        history_index=0,
    )
    process_thin = int(np.sum(process_display[:, 0] == 2))
    local_substrate = int(np.sum(local_display[:, 0] == 1))
    process_substrate = int(np.sum(process_display[:, 0] == 1))
    if process_thin != local_thin:
        raise AssertionError(
            f"Minimum visible layers should stay at the {MIN_LAYER_PX:g}px floor, "
            f"got process={process_thin}, local={local_thin}"
        )
    if process_thin != int(MIN_LAYER_PX):
        raise AssertionError(f"Thin marker should render at the {MIN_LAYER_PX:g}px floor, got {process_thin}")
    if process_substrate != local_substrate:
        raise AssertionError(
            f"Process-wide canvas sizing should not change the substrate scale, "
            f"got process={process_substrate}, local={local_substrate}"
        )


def test_finalised_visual_override_scope_consistency():
    win, early, late = _process_wide_scaling_window()
    key = win._display_material_count_profile_key(2, 0.001)

    win.visualisation_mode = "pending"
    win.visual_thickness_overrides = {key: 13}
    early_display = win._display_grid_auto(
        early["grid"],
        row_heights_um=early["row_heights_um"],
        history_index=0,
    )
    late_display = win._display_grid_auto(
        late["grid"],
        row_heights_um=late["row_heights_um"],
        history_index=1,
    )
    early_rows = int(np.sum(early_display[:, 0] == 2))
    late_rows = int(np.sum(late_display[:, 0] == 2))
    if early_rows != 13 or late_rows != 13:
        raise AssertionError(
            f"Finalised visual override should apply wherever that segment exists, got early={early_rows}, late={late_rows}"
        )

    win.reject_visualisation()
    if win.visualisation_mode != "dynamic" or win.visual_thickness_overrides:
        raise AssertionError("Reject should clear finalised overrides and return to dynamic mode")
    early_display = win._display_grid_auto(
        early["grid"],
        row_heights_um=early["row_heights_um"],
        history_index=0,
    )
    early_rows = int(np.sum(early_display[:, 0] == 2))
    if early_rows != int(MIN_LAYER_PX):
        raise AssertionError(f"After Reject, dynamic {MIN_LAYER_PX:g}px floor should return, got {early_rows}")


def _substrate_rebuild_commands():
    return [
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 5.0},
        {
            "action": "operation",
            "operation": "Lithography",
            "material": "PR_SU-8 Photoresist",
            "windows": [(0.40, 0.60)],
            "invert_lithography": False,
        },
        {
            "action": "operation",
            "operation": "Etch",
            "material": "Silicon (Si)",
            "depth_um": 20.0,
            "etch_mode": "anisotropic",
        },
    ]


def _substrate_rebuild_window():
    _app, win = _new_window()
    for command in _substrate_rebuild_commands():
        ok, message = win._execute_assistant_command(command)
        if not ok:
            raise AssertionError(f"substrate rebuild command failed: {message}")
    return win


def test_substrate_change_rebuilds_previous_steps():
    win = _substrate_rebuild_window()
    if getattr(win, "process_steps_replay_enabled", False):
        raise AssertionError("Regression setup should start as snapshot-only history")
    before_count = len(win.history)
    selected_row = min(2, before_count - 1)
    win.history_list.setCurrentRow(selected_row)

    win.substrate_thick_spin.setValue(525)
    win.on_substrate_changed()

    if len(win.history) != before_count:
        raise AssertionError(f"Substrate propagation should preserve history length, got {len(win.history)}")
    if not getattr(win, "process_steps_replay_enabled", False):
        raise AssertionError("Substrate propagation should enable deterministic replay for future edits")
    if win.history_list.currentRow() != selected_row:
        raise AssertionError(
            f"Substrate propagation should preserve selected row {selected_row}, "
            f"got {win.history_list.currentRow()}"
        )
    _assert_history_substrate_max(win, "Silicon (Si)", 525.0)

    first_op = _state_by_operation(win.history, "Spincoating")
    values = _material_column_heights(first_op, win.material_names, "Silicon (Si)")
    if abs(float(values.min()) - 525.0) > TOL_UM or abs(float(values.max()) - 525.0) > TOL_UM:
        raise AssertionError("Previous spincoat step should be rebuilt on a 525 um substrate")


def test_assistant_set_substrate_rebuilds_without_clearing_history():
    win = _substrate_rebuild_window()
    before_count = len(win.history)
    ok, message = win._execute_assistant_command({
        "action": "set_substrate",
        "material": "Silicon (Si)",
        "thickness_um": 525.0,
    })
    if not ok:
        raise AssertionError(f"Assistant set_substrate failed: {message}")
    if len(win.history) != before_count:
        raise AssertionError("Assistant set_substrate should rebuild existing steps, not clear history")
    if not getattr(win, "process_steps_replay_enabled", False):
        raise AssertionError("Assistant set_substrate should leave process in replay-enabled mode")
    _assert_history_substrate_max(win, "Silicon (Si)", 525.0)
    _state_by_operation(win.history, "Spincoating")
    _state_by_operation(win.history, "Lithography")
    _state_by_operation(win.history, "Etch")


def test_assistant_set_substrate_material_rebuilds_history():
    _app, win = _new_window()
    for command in (
        {"action": "add_material", "material": "Gold"},
        {"action": "operation", "operation": "Deposit", "material": "Gold", "thickness_um": 0.1},
    ):
        ok, message = win._execute_assistant_command(command)
        if not ok:
            raise AssertionError(f"substrate material setup failed: {message}")
    before_count = len(win.history)

    ok, message = win._execute_assistant_command({
        "action": "set_substrate",
        "material": BASE_SUBSTRATE_MATERIAL,
        "thickness_um": 525.0,
    })
    if not ok:
        raise AssertionError(f"Assistant substrate material change failed: {message}")
    if len(win.history) != before_count:
        raise AssertionError("Changing substrate material should rebuild existing history, not clear it")
    _assert_history_substrate_max(win, BASE_SUBSTRATE_MATERIAL, 525.0)
    gold_state = _state_by_operation(win.history, "Deposit")
    stats = _material_stats(gold_state, win.material_names, "Gold")
    if stats is None or abs(float(stats["median"]) - 0.1) > TOL_UM:
        raise AssertionError(f"Gold deposit should be preserved after substrate material change, got {stats}")


def _central_anisotropic_etch_window(depth_um):
    commands = [
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 5.0},
        {
            "action": "operation",
            "operation": "Lithography",
            "material": "PR_SU-8 Photoresist",
            "windows": [(0.40, 0.60)],
            "invert_lithography": False,
        },
        {
            "action": "operation",
            "operation": "Etch",
            "material": "Silicon (Si)",
            "depth_um": float(depth_um),
            "etch_mode": "anisotropic",
        },
    ]
    _app, win = _new_window()
    for command in commands:
        ok, message = win._execute_assistant_command(command)
        if not ok:
            raise AssertionError(f"central etch command failed: {message}")
    return win, win.history[-1]


def test_anisotropic_etch_cutout_has_no_false_material_layer():
    win, state = _central_anisotropic_etch_window(30.0)
    grid = np.asarray(state.get("grid"))
    row_heights = np.asarray(state.get("row_heights_um"), dtype=float)
    silicon_index = win.material_names.index("Silicon (Si)")

    display_grid = win._display_grid_auto(
        grid,
        row_heights_um=row_heights,
        history_index=len(win.history) - 1,
    )
    info = getattr(win, "_last_display_remap_info", {}) or {}
    if info.get("etch_depth_overlays"):
        raise AssertionError(f"Anisotropic etch should render direct void rows, not overlays: {info}")

    # The substrate body is compact, while removed depth uses the process-zone
    # scale and genuinely shrinks the remaining structural segment.
    outside_rows = np.where(display_grid[:, 0] == silicon_index)[0]
    center_rows = np.where(display_grid[:, 150] == silicon_index)[0]
    if outside_rows.size == 0 or center_rows.size == 0:
        raise AssertionError("Etch display should contain substrate inside and outside the etched window")
    if center_rows.size >= outside_rows.size:
        raise AssertionError(
            f"Etched substrate (170um remaining) should display thinner than unetched (200um), "
            f"got outside={outside_rows.size}, center={center_rows.size}"
        )
    profile = win._process_display_adjustment_profile()
    expected_outside_rows = _dynamic_rows_for_um(
        win, 200.0, profile, material_idx=silicon_index, is_substrate_core=True
    )
    expected_center_rows = _dynamic_rows_for_um(
        win, 170.0, profile, material_idx=silicon_index, is_substrate_core=True
    )
    if abs(outside_rows.size - expected_outside_rows) > 1:
        raise AssertionError(f"Unetched substrate should be {expected_outside_rows}px, got {outside_rows.size}")
    if abs(center_rows.size - expected_center_rows) > 1:
        raise AssertionError(f"Etched substrate should be {expected_center_rows}px, got {center_rows.size}")

    # The etch depth itself shows up as its own distinctly-sized void
    # segment directly above the (now visibly shorter) substrate.
    columns = info.get("column_segments") or []
    if len(columns) <= 150:
        raise AssertionError("Expected column_segments metadata for the etched column")
    center_void = next(
        (seg for seg in columns[150]
         if int(seg.get("material", win.VOID_INDEX)) == win.VOID_INDEX
         and bool(seg.get("in_substrate_recess", False))),
        None,
    )
    if center_void is None:
        raise AssertionError("Etched column should have its own void segment above the substrate")
    void_rows = int(center_void["display_end"]) - int(center_void["display_start"])
    # The resolved visible trench is keyed to actual substrate removal. The
    # opened PR changes the surface topology but must not invent a second
    # depth scale for the same substrate bottom.
    expected_void_rows = _dynamic_rows_for_um(win, 30.0, profile)
    if abs(void_rows - expected_void_rows) > 1:
        raise AssertionError(
            f"Etched void and 30um substrate removal should share {expected_void_rows} rows, got {void_rows}"
        )

def test_substrate_visual_override_preserves_etched_drop():
    commands = [
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 5.0},
        {
            "action": "operation",
            "operation": "Lithography",
            "material": "PR_SU-8 Photoresist",
            "windows": [(0.40, 0.60)],
            "invert_lithography": False,
        },
        {"action": "operation", "operation": "Etch", "material": "Silicon (Si)", "depth_um": 30.0, "etch_mode": "anisotropic"},
    ]
    _app, win = _new_window()
    for command in commands:
        ok, message = win._execute_assistant_command(command)
        if not ok:
            raise AssertionError(f"substrate override etch command failed: {message}")

    state = win.history[-1]
    grid = np.asarray(state.get("grid"))
    row_heights = np.asarray(state.get("row_heights_um"), dtype=float)
    silicon_index = win.material_names.index("Silicon (Si)")
    expected_void_rows = _dynamic_rows_for_um(win, 30.0)

    def render():
        profile = win._process_display_adjustment_profile()
        display_grid = win._display_grid_auto(
            grid,
            row_heights_um=row_heights,
            history_index=len(win.history) - 1,
            visual_geometry=state.get("visual_geometry", []),
            process_profile=profile,
        )
        info = getattr(win, "_last_display_remap_info", {}) or {}
        columns = info.get("column_segments") or []
        outside_rows = int(np.sum(display_grid[:, 0] == silicon_index))
        center_rows = int(np.sum(display_grid[:, 150] == silicon_index))
        center_void = next(
            (seg for seg in columns[150]
             if int(seg.get("material", win.VOID_INDEX)) == win.VOID_INDEX
             and bool(seg.get("in_substrate_recess", False))),
            None,
        )
        void_rows = (
            int(center_void["display_end"]) - int(center_void["display_start"])
            if center_void is not None else 0
        )
        return outside_rows, center_rows, void_rows

    # Baseline: the void notch and rows removed from the substrate share the
    # same globally resolved 30um target.
    outside_rows, center_rows, void_rows = render()
    if center_rows >= outside_rows:
        raise AssertionError(f"Etched substrate should display thinner, got outside={outside_rows}, center={center_rows}")
    if abs(void_rows - expected_void_rows) > 1:
        raise AssertionError(f"Baseline void notch should be {expected_void_rows}px, got {void_rows}")

    # Overriding the OUTSIDE (unetched, 200um) column's visual size should
    # only affect that column - not the etched column's substrate, and
    # never the void notch (which stays a direct function of real depth,
    # fully independent of any material override).
    win.visualisation_mode = "pending"
    win.visual_thickness_overrides = {
        win._display_material_count_profile_key(silicon_index, 200.0): 90
    }
    outside_rows, center_rows, void_rows = render()
    if outside_rows != 90:
        raise AssertionError(f"Overriding the unetched (200um) key should resize that column, got {outside_rows}")
    if center_rows == 90:
        raise AssertionError("Overriding the unetched (200um) key should not affect the etched (170um) column")
    if abs(void_rows - expected_void_rows) > 1:
        raise AssertionError(
            f"Void notch should be unaffected by a material override, expected {expected_void_rows}px, got {void_rows}"
        )

    # Symmetric check: overriding the ETCHED (170um) column's key instead.
    win.visual_thickness_overrides = {
        win._display_material_count_profile_key(silicon_index, 170.0): 130
    }
    outside_rows, center_rows, void_rows = render()
    if center_rows != 130:
        raise AssertionError(f"Overriding the etched (170um) key should resize that column, got {center_rows}")
    if outside_rows == 130:
        raise AssertionError("Overriding the etched (170um) key should not affect the unetched (200um) column")
    if abs(void_rows - expected_void_rows) > 1:
        raise AssertionError(
            f"Void notch should still be unaffected by a material override, expected {expected_void_rows}px, got {void_rows}"
        )


def test_deposit_after_etch_remains_visible_in_trench():
    commands = [
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "add_material", "material": "Gold"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 5.0},
        {
            "action": "operation",
            "operation": "Lithography",
            "material": "PR_SU-8 Photoresist",
            "windows": [(0.40, 0.60)],
            "invert_lithography": False,
        },
        {"action": "operation", "operation": "Etch", "material": "Silicon (Si)", "depth_um": 30.0, "etch_mode": "anisotropic"},
        {"action": "operation", "operation": "Deposit", "material": "Gold", "thickness_um": 2.0},
    ]
    _app, win = _new_window()
    for command in commands:
        ok, message = win._execute_assistant_command(command)
        if not ok:
            raise AssertionError(f"deposit-after-etch command failed: {message}")

    state = win.history[-1]
    grid = np.asarray(state.get("grid"))
    row_heights = np.asarray(state.get("row_heights_um"), dtype=float)
    gold_index = win.material_names.index("Gold")

    physical_gold = np.asarray([
        float(np.sum(row_heights[grid[:, x] == gold_index]))
        for x in range(grid.shape[1])
    ])
    if abs(float(physical_gold[0]) - 2.0) > TOL_UM or abs(float(physical_gold[150]) - 2.0) > TOL_UM:
        raise AssertionError(
            f"Directional deposition should place 2 um gold outside and inside trench, "
            f"got outside={physical_gold[0]:g}, inside={physical_gold[150]:g}"
        )

    display_grid = win._display_grid_auto(
        grid,
        row_heights_um=row_heights,
        history_index=len(win.history) - 1,
    )
    display_gold = np.asarray([
        int(np.sum(display_grid[:, x] == gold_index))
        for x in range(display_grid.shape[1])
    ])
    if display_gold[150] <= 0:
        raise AssertionError("Gold deposited inside etched trench should remain visible in display grid")
    central_gold_rows = np.where(display_grid[:, 150] == gold_index)[0]
    if central_gold_rows.size == 0:
        raise AssertionError("Gold deposited inside etched trench should have displayed rows")
    info = getattr(win, "_last_display_remap_info", {}) or {}
    if info.get("etch_depth_overlays"):
        raise AssertionError(f"Deposit-after-etch should use direct grid rows, not overlays: {info}")
    display_void = np.asarray([
        int(np.sum(display_grid[:, x] == win.VOID_INDEX))
        for x in range(display_grid.shape[1])
    ])
    if int(display_void[150]) <= int(display_void[0]):
        raise AssertionError(
            f"Partially filled trench should keep direct visible void rows, "
            f"outside={display_void[0]}, inside={display_void[150]}"
        )
    if central_gold_rows.min() <= np.where(display_grid[:, 150] == win.VOID_INDEX)[0].min():
        raise AssertionError("Gold should sit below the remaining open trench void in the center column")


def test_visual_thickness_pick_deposit_inside_etched_trench():
    commands = [
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "add_material", "material": "Gold"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 5.0},
        {
            "action": "operation",
            "operation": "Lithography",
            "material": "PR_SU-8 Photoresist",
            "windows": [(0.40, 0.60)],
            "invert_lithography": False,
        },
        {"action": "operation", "operation": "Etch", "material": "Silicon (Si)", "depth_um": 30.0, "etch_mode": "anisotropic"},
        {"action": "operation", "operation": "Deposit", "material": "Gold", "thickness_um": 2.0},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 5.0},
    ]
    _app, win = _new_window()
    for command in commands:
        ok, message = win._execute_assistant_command(command)
        if not ok:
            raise AssertionError(f"visual-pick trench command failed: {message}")

    state = win.history[-1]
    state_index = len(win.history) - 1
    grid = np.asarray(state.get("grid"))
    row_heights = np.asarray(state.get("row_heights_um"), dtype=float)
    profile = win._process_display_adjustment_profile()

    win.canvas.ax.clear()
    win.plot_grid(
        win.canvas.ax,
        grid,
        title=None,
        main_axis=True,
        history_index=state_index,
        row_heights_um=row_heights,
        visual_geometry=state.get("visual_geometry", []),
        process_profile=profile,
    )

    display_grid = np.asarray(getattr(win, "_last_main_display_grid", None))
    gold_index = win.material_names.index("Gold")
    center_gold_rows = np.where(display_grid[:, 150] == gold_index)[0]
    if center_gold_rows.size == 0:
        raise AssertionError("Gold should be visible in the etched center column before picking")
    click_row = int(center_gold_rows[center_gold_rows.size // 2])

    class _Event:
        pass

    event = _Event()
    event.inaxes = win.canvas.ax
    event.xdata = 150.0
    event.ydata = win._plot_y_for_visual_scale(
        float(click_row),
        int(display_grid.shape[0]),
        win._visual_scale_factor_clamped(),
    )

    pick = win._visual_layer_pick_from_event(event)
    if pick is None:
        raise AssertionError("Visual thickness picker should select deposited Gold inside the etched trench")
    if int(pick.get("material_idx", -1)) != gold_index:
        raise AssertionError(f"Picker should select Gold, got {pick}")
    if abs(float(pick.get("height_um", 0.0)) - 2.0) > TOL_UM:
        raise AssertionError(f"Picker should report the 2 um Gold deposit thickness, got {pick}")
    expected_key = win._display_material_count_profile_key(gold_index, 2.0)
    if pick.get("key") != expected_key:
        raise AssertionError(f"Picker key should target the deposited Gold layer, got {pick}")
    expected_px = int(np.sum(display_grid[:, 150] == gold_index))
    if int(pick.get("current_px", 0)) != expected_px:
        raise AssertionError(f"Picker should report current Gold display rows {expected_px}, got {pick}")


def test_etch_depth_display_after_spincoat():
    commands = [
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 5.0},
        {
            "action": "operation",
            "operation": "Lithography",
            "material": "PR_SU-8 Photoresist",
            "windows": [(0.40, 0.60)],
            "invert_lithography": False,
        },
        {"action": "operation", "operation": "Etch", "material": "Silicon (Si)", "depth_um": 20.0, "etch_mode": "anisotropic"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 5.0},
    ]
    _app, win = _new_window()
    for command in commands:
        ok, message = win._execute_assistant_command(command)
        if not ok:
            raise AssertionError(f"etch-depth display command failed: {message}")

    state = win.history[-1]
    grid = np.asarray(state.get("grid"))
    row_heights = np.asarray(state.get("row_heights_um"), dtype=float)
    silicon_index = win.material_names.index("Silicon (Si)")
    substrate_physical = np.asarray([
        float(np.sum(row_heights[grid[:, x] == silicon_index]))
        for x in range(grid.shape[1])
    ])
    physical_depth = float(substrate_physical.max() - substrate_physical.min())
    if abs(physical_depth - 20.0) > TOL_UM:
        raise AssertionError(f"Expected 20 um physical etch depth, got {physical_depth:g}")

    display_grid = win._display_grid_auto(
        grid,
        row_heights_um=row_heights,
        history_index=len(win.history) - 1,
    )
    info = getattr(win, "_last_display_remap_info", {}) or {}
    if info.get("etch_depth_overlays"):
        raise AssertionError(f"Filled trench should render from direct grid rows, not overlays: {info}")
    pr_index = win.material_names.index("PR_SU-8 Photoresist")
    display_pr = np.asarray([
        int(np.sum(display_grid[:, x] == pr_index))
        for x in range(display_grid.shape[1])
    ])
    if int(display_pr[150]) <= int(display_pr[0]):
        raise AssertionError(
            f"Spincoat should visibly fill etched trench with thicker center PR, "
            f"outside={display_pr[0]}, inside={display_pr[150]}"
        )

    etch_state = _state_by_operation(win.history, "Etch")
    etch_grid = np.asarray(etch_state.get("grid"))
    etch_heights = np.asarray(etch_state.get("row_heights_um"), dtype=float)
    etch_display = win._display_grid_auto(
        etch_grid,
        row_heights_um=etch_heights,
        history_index=next(i for i, item in enumerate(win.history) if item is etch_state),
    )
    etch_info = getattr(win, "_last_display_remap_info", {}) or {}
    if etch_info.get("etch_depth_overlays"):
        raise AssertionError(f"Open etch step should render direct void rows, not overlays: {etch_info}")
    etch_void = np.asarray([
        int(np.sum(etch_display[:, x] == win.VOID_INDEX))
        for x in range(etch_display.shape[1])
    ])
    if int(etch_void[150]) <= int(etch_void[0]):
        raise AssertionError(
            f"Open etch step should show direct void in center column, "
            f"outside={etch_void[0]}, inside={etch_void[150]}"
        )

    if not any(item.get("type") == "etch_depth_feature" for item in state.get("visual_geometry", []) or []):
        raise AssertionError("Etch depth feature metadata should persist after spincoat")


def test_etch_depth_matches_equal_deposit_display_height():
    commands = _base_substrate_commands() + [
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "add_material", "material": "Gold"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 1.0},
        {
            "action": "operation",
            "operation": "Lithography",
            "material": "PR_SU-8 Photoresist",
            "windows": [(0.40, 0.60)],
            "invert_lithography": False,
        },
        {
            "action": "operation",
            "operation": "Etch",
            "material": BASE_SUBSTRATE_MATERIAL,
            "depth_um": 10.0,
            "etch_mode": "anisotropic",
        },
        {"action": "operation", "operation": "Deposit", "material": "Gold", "thickness_um": 10.0},
    ]
    win, path = _run_commands("FabStack_dimtest_equal_etch_deposit_display", commands)
    profile = win._process_display_adjustment_profile()

    etch_state = _state_by_operation(win.history, "Etch", occurrence=1)
    deposit_state = _state_by_operation(win.history, "Deposit", occurrence=1)
    substrate_index = win.material_names.index(BASE_SUBSTRATE_MATERIAL)
    gold_index = win.material_names.index("Gold")
    # The substrate-core display reflects remaining thickness as nominal body
    # minus process-zone etch depth (525um -> 515um after a 10um etch).
    # The etch also shows up as its own void segment. The opened PR changes
    # topology, but the notch height remains the same resolved 10um substrate
    # depth. Gold uses the same feature law.
    expected_etch_notch_rows = _dynamic_rows_for_um(win, 10.0, profile)
    expected_gold_rows = _dynamic_rows_for_um(win, 10.0, profile)

    etch_grid = np.asarray(etch_state.get("grid"))
    etch_heights = np.asarray(etch_state.get("row_heights_um"), dtype=float)
    etch_display = win._display_grid_auto(
        etch_grid,
        row_heights_um=etch_heights,
        history_index=next(i for i, state in enumerate(win.history) if state is etch_state),
        visual_geometry=etch_state.get("visual_geometry", []),
        process_profile=profile,
    )
    outside_rows = np.where(etch_display[:, 0] == substrate_index)[0]
    center_rows = np.where(etch_display[:, 150] == substrate_index)[0]
    if outside_rows.size == 0 or center_rows.size == 0:
        raise AssertionError("Etch display should contain substrate inside and outside the window")
    if center_rows.size >= outside_rows.size:
        raise AssertionError(
            f"Etched substrate should display thinner than unetched, "
            f"got outside={outside_rows.size}, center={center_rows.size}"
        )
    etch_columns = (getattr(win, "_last_display_remap_info", {}) or {}).get("column_segments") or []
    center_void = next(
        (seg for seg in etch_columns[150]
         if int(seg.get("material", win.VOID_INDEX)) == win.VOID_INDEX
         and bool(seg.get("in_substrate_recess", False))),
        None,
    )
    if center_void is None:
        raise AssertionError("Etched column should have its own void segment above the substrate")
    void_rows = int(center_void["display_end"]) - int(center_void["display_start"])
    if abs(void_rows - expected_etch_notch_rows) > 1:
        raise AssertionError(
            f"Etched void and 10um substrate removal should share {expected_etch_notch_rows} rows, got {void_rows}"
        )

    deposit_grid = np.asarray(deposit_state.get("grid"))
    deposit_heights = np.asarray(deposit_state.get("row_heights_um"), dtype=float)
    deposit_display = win._display_grid_auto(
        deposit_grid,
        row_heights_um=deposit_heights,
        history_index=next(i for i, state in enumerate(win.history) if state is deposit_state),
        visual_geometry=deposit_state.get("visual_geometry", []),
        process_profile=profile,
    )
    gold_rows = int(np.sum(deposit_display[:, 150] == gold_index))
    if gold_rows != expected_gold_rows:
        raise AssertionError(
            f"10 um deposited gold should display as {expected_gold_rows} px, got {gold_rows}"
        )
    return path


def test_etch_geometry_display_stress_every_step():
    commands = _base_substrate_commands() + [
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "add_material", "material": "Gold"},
        {"action": "add_material", "material": "Silicon Dioxide (SiO2)"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 5.0},
        {
            "action": "operation",
            "operation": "Lithography",
            "material": "PR_SU-8 Photoresist",
            "windows": [(0.40, 0.60)],
            "invert_lithography": False,
        },
        {
            "action": "operation",
            "operation": "Etch",
            "material": BASE_SUBSTRATE_MATERIAL,
            "depth_um": 30.0,
            "etch_mode": "anisotropic",
        },
        {"action": "operation", "operation": "Deposit", "material": "Gold", "thickness_um": 2.0},
        {"action": "operation", "operation": "Deposit", "material": "Silicon Dioxide (SiO2)", "thickness_um": 0.1},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 5.0},
    ]
    win, path = _run_commands("FabStack_dimtest_etch_geometry_display_stress", commands)

    etch_state = _state_by_operation(win.history, "Etch", occurrence=1)
    gold_state = _state_by_operation(win.history, "Deposit", occurrence=1)
    oxide_state = _state_by_operation(win.history, "Deposit", occurrence=2)
    final_spin = _state_by_operation(win.history, "Spincoating", occurrence=2)

    gold_index = win.material_names.index("Gold")
    oxide_index = win.material_names.index("Silicon Dioxide (SiO2)")
    pr_index = win.material_names.index("PR_SU-8 Photoresist")

    for state, expected_depth in (
        (etch_state, 30.0),
        (gold_state, 30.0),
        (oxide_state, 30.0),
        (final_spin, 30.0),
    ):
        substrate = _material_column_heights(state, win.material_names, BASE_SUBSTRATE_MATERIAL)
        depth = float(substrate.max() - substrate.min())
        if abs(depth - expected_depth) > TOL_UM:
            raise AssertionError(
                f"{state.get('title', '')!r} should keep a {expected_depth:g} um etched recess, got {depth:g}"
            )

    gold_grid = np.asarray(gold_state.get("grid"))
    gold_heights = np.asarray(gold_state.get("row_heights_um"), dtype=float)
    gold_physical = np.asarray([
        float(np.sum(gold_heights[gold_grid[:, x] == gold_index]))
        for x in range(gold_grid.shape[1])
    ])
    if abs(float(gold_physical[150]) - 2.0) > TOL_UM or abs(float(gold_physical[0]) - 2.0) > TOL_UM:
        raise AssertionError(f"Gold deposit should remain 2 um inside and outside etched trench, got {gold_physical}")

    oxide_grid = np.asarray(oxide_state.get("grid"))
    oxide_heights = np.asarray(oxide_state.get("row_heights_um"), dtype=float)
    oxide_physical = np.asarray([
        float(np.sum(oxide_heights[oxide_grid[:, x] == oxide_index]))
        for x in range(oxide_grid.shape[1])
    ])
    if abs(float(oxide_physical[150]) - 0.1) > TOL_UM or abs(float(oxide_physical[0]) - 0.1) > TOL_UM:
        raise AssertionError(f"Oxide deposit should remain 0.1 um inside and outside etched trench, got {oxide_physical}")

    final_grid = np.asarray(final_spin.get("grid"))
    final_heights = np.asarray(final_spin.get("row_heights_um"), dtype=float)
    final_pr = np.asarray([
        float(np.sum(final_heights[final_grid[:, x] == pr_index]))
        for x in range(final_grid.shape[1])
    ])
    if final_pr[150] <= final_pr[0] + 25.0:
        raise AssertionError(
            f"Planarized spincoat should fill the etched recess, got PR columns {final_pr[0]} and {final_pr[150]}"
        )

    _assert_etch_geometry_consistent_every_step(win)
    return path


def test_multiscale_trench_fill_layer_order():
    commands = _base_substrate_commands() + [
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "add_material", "material": "Chromium"},
        {"action": "add_material", "material": "Gold"},
        {"action": "add_material", "material": "Silicon Dioxide (SiO2)"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 1.0},
        {
            "action": "operation",
            "operation": "Lithography",
            "material": "PR_SU-8 Photoresist",
            "windows": [(0.40, 0.60)],
            "invert_lithography": False,
        },
        {
            "action": "operation",
            "operation": "Etch",
            "material": BASE_SUBSTRATE_MATERIAL,
            "depth_um": 10.0,
            "etch_mode": "anisotropic",
        },
        {"action": "operation", "operation": "Deposit", "material": "Chromium", "thickness_um": 0.001},
        {"action": "operation", "operation": "Deposit", "material": "Gold", "thickness_um": 0.01},
        {"action": "operation", "operation": "Deposit", "material": "Silicon Dioxide (SiO2)", "thickness_um": 0.1},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 1.0},
    ]
    win, path = _run_commands("FabStack_dimtest_multiscale_trench_fill_order", commands)

    final_deposit = _state_by_operation(win.history, "Deposit", occurrence=3)
    final_spin = _state_by_operation(win.history, "Spincoating", occurrence=2)
    for state in (final_deposit, final_spin):
        substrate = _material_column_heights(state, win.material_names, BASE_SUBSTRATE_MATERIAL)
        etch_depth = float(substrate.max() - substrate.min())
        if abs(etch_depth - 10.0) > TOL_UM:
            raise AssertionError(f"{state.get('title', '')!r} should keep 10 um etched depth, got {etch_depth:g}")

    for material, expected in (
        ("Chromium", 0.001),
        ("Gold", 0.01),
        ("Silicon Dioxide (SiO2)", 0.1),
    ):
        stats = _material_stats(final_deposit, win.material_names, material)
        if stats is None:
            raise AssertionError(f"Missing {material} in multiscale trench deposit state")
        for key in ("min", "median", "max"):
            if abs(float(stats[key]) - expected) > TOL_UM:
                raise AssertionError(
                    f"{material} {key} should be {expected:g} um after deposit, got {stats[key]:g}"
                )

    profile = win._process_display_adjustment_profile()
    grid = np.asarray(final_spin.get("grid"))
    row_heights = np.asarray(final_spin.get("row_heights_um"), dtype=float)
    win._display_grid_auto(
        grid,
        row_heights_um=row_heights,
        history_index=next(i for i, state in enumerate(win.history) if state is final_spin),
        visual_geometry=final_spin.get("visual_geometry", []),
        process_profile=profile,
    )
    info = getattr(win, "_last_display_remap_info", {}) or {}
    if info.get("etch_depth_overlays"):
        raise AssertionError(f"Multiscale trench fill should render direct grid rows, not overlays: {info}")

    display_grid = win._display_grid_auto(
        grid,
        row_heights_um=row_heights,
        history_index=next(i for i, state in enumerate(win.history) if state is final_spin),
        visual_geometry=final_spin.get("visual_geometry", []),
        process_profile=profile,
    )
    center_col = display_grid[:, 150]
    expected_order = ["Silicon Dioxide (SiO2)", "Gold", "Chromium"]
    row_ranges = {}
    for material in expected_order:
        material_index = win.material_names.index(material)
        rows = np.where(center_col == material_index)[0]
        if rows.size == 0:
            raise AssertionError(f"{material} should be visible in the etched center column")
        row_ranges[material] = (int(rows.min()), int(rows.max()), int(rows.size))

    if not (
        row_ranges["Silicon Dioxide (SiO2)"][0] < row_ranges["Gold"][0] < row_ranges["Chromium"][0]
    ):
        raise AssertionError(f"Trench fill display order should be {expected_order}, got row ranges {row_ranges}")
    expected_sizes = {
        "Chromium": _dynamic_rows_for_um(win, 0.001, profile),
        "Gold": _dynamic_rows_for_um(win, 0.01, profile),
        "Silicon Dioxide (SiO2)": _dynamic_rows_for_um(win, 0.1, profile),
    }
    for material, expected_rows in expected_sizes.items():
        if row_ranges[material][2] != expected_rows:
            raise AssertionError(
                f"{material} should render as {expected_rows} pixels with the dynamic minimum scale, got {row_ranges}"
            )

    _assert_etch_geometry_consistent_every_step(win)
    return path


def test_power_scale_population_independent_order_sweep_deposit_and_spincoat():
    """
    render_px(t) must depend only on t: identical physical thickness renders
    identically for deposit vs. spincoat and regardless of how many other
    distinct thicknesses exist elsewhere in the process.  The structural
    substrate uses the same intrinsic law and is calibrated to 175 rows at
    the common 525um wafer thickness.
    """
    _app, win = _new_window()
    thicknesses = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 200.0, 400.0]
    labels = ["1nm", "10nm", "100nm", "1um", "10um", "100um", "200um", "400um"]
    deposit_materials = [f"DepScale_{label}" for label in labels]
    spin_materials = [f"SpinScale_{label} Photoresist" for label in labels]

    for idx, name in enumerate(deposit_materials + spin_materials):
        if name not in win.material_names:
            win.material_names.append(name)
            hue = (idx * 37) % 255
            win.material_colors.append(f"#{hue:02x}{(180 - idx * 7) % 255:02x}{(80 + idx * 11) % 255:02x}")
            if name != "Void" and win.substrate_combo.findText(name) < 0:
                win.substrate_combo.addItem(name)
    win.refresh_operation_material_combo()
    win.refresh_materials_list()
    win.update_etchant_options()
    win.update_deposit_methods_for_material()

    commands = [{"action": "set_substrate", "material": "Silicon (Si)", "thickness_um": 525.0}]
    commands.extend(
        {"action": "operation", "operation": "Deposit", "material": material, "thickness_um": thickness}
        for material, thickness in zip(deposit_materials, thicknesses)
    )
    commands.extend(
        {"action": "operation", "operation": "Spincoating", "material": material, "thickness_um": thickness}
        for material, thickness in zip(spin_materials, thicknesses)
    )

    def rows_for(display_grid, material):
        material_index = win.material_names.index(material)
        rows = int(np.sum(np.asarray(display_grid)[:, 0] == material_index))
        if rows < int(MIN_LAYER_PX):
            raise AssertionError(f"{material} should render at least {MIN_LAYER_PX:g}px, got {rows}")
        return rows

    first_deposit_material = deposit_materials[0]
    early_rows_for_first_deposit = None
    for step_idx, command in enumerate(commands):
        ok, message = win._execute_assistant_command(command)
        if not ok:
            raise AssertionError(f"power scale sweep command failed: {message}")
        if step_idx == 1:
            # Right after the first deposit, before any other distinct
            # thickness exists in the process.
            state = win.history[-1]
            display_grid = win._display_grid_auto(
                np.asarray(state.get("grid")),
                row_heights_um=np.asarray(state.get("row_heights_um"), dtype=float),
                history_index=len(win.history) - 1,
            )
            early_rows_for_first_deposit = rows_for(display_grid, first_deposit_material)

    state = win.history[-1]
    display_grid = win._display_grid_auto(
        np.asarray(state.get("grid")),
        row_heights_um=np.asarray(state.get("row_heights_um"), dtype=float),
        history_index=len(win.history) - 1,
        visual_geometry=state.get("visual_geometry", []),
    )
    display_grid = np.asarray(display_grid)

    late_rows_for_first_deposit = rows_for(display_grid, first_deposit_material)
    if late_rows_for_first_deposit != early_rows_for_first_deposit:
        raise AssertionError(
            f"{first_deposit_material} pixel height must not depend on how many other "
            f"distinct thicknesses exist elsewhere in the process: "
            f"{early_rows_for_first_deposit}px early vs {late_rows_for_first_deposit}px late"
        )

    deposit_rows = [rows_for(display_grid, material) for material in deposit_materials]
    spin_rows = [rows_for(display_grid, material) for material in spin_materials]
    if deposit_rows != sorted(deposit_rows):
        raise AssertionError(f"Deposit order sweep should render monotonically by thickness, got {deposit_rows}")
    if spin_rows != sorted(spin_rows):
        raise AssertionError(f"Spincoat order sweep should render monotonically by thickness, got {spin_rows}")
    if deposit_rows != spin_rows:
        raise AssertionError(f"Same physical thickness should render the same for deposit/spincoat: {deposit_rows} vs {spin_rows}")

    # 400um vs 200um follows the same fifth-root ratio on the live rendering
    # path, not just in the standalone helper.
    two_hundred_px = deposit_rows[-2]
    four_hundred_px = deposit_rows[-1]
    ratio = four_hundred_px / two_hundred_px
    expected_ratio = 2.0 ** 0.20
    if abs(ratio - expected_ratio) > 0.03:
        raise AssertionError(
            f"400um/200um should follow the fifth-root ratio, got ratio={ratio:g}: {deposit_rows}"
        )

    # The structural substrate uses its explicit compact body scale and must
    # remain stable regardless of the large process-zone layer population.
    substrate_index = win.material_names.index("Silicon (Si)")
    substrate_rows = int(np.sum(display_grid[:, 0] == substrate_index))
    expected_substrate_rows = win._display_substrate_core_target_height(
        525.0,
        win._process_display_adjustment_profile(),
    )
    if substrate_rows != expected_substrate_rows:
        raise AssertionError(
            f"Substrate should render through the compact body scale, "
            f"expected {expected_substrate_rows}px, got {substrate_rows}px"
        )
    return None


def test_substrate_anchored_20um_etch_beats_nm_film_stack():
    """Regression for the reported 525um substrate / 20um trench process."""
    commands = [
        {"action": "set_substrate", "material": "Silicon (Si)", "thickness_um": 525.0},
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "add_material", "material": "Aluminum Nitride (AlN)"},
        {"action": "add_material", "material": "Silicon Carbide (SiC)"},
        {"action": "add_material", "material": "Titanium Nitride (TiN)"},
        {
            "action": "operation",
            "operation": "Spincoating",
            "material": "PR_SU-8 Photoresist",
            "thickness_um": 5.0,
        },
        {
            "action": "operation",
            "operation": "Lithography",
            "material": "PR_SU-8 Photoresist",
            "windows": [(0.40, 0.60)],
            "invert_lithography": False,
        },
        {
            "action": "operation",
            "operation": "Etch",
            "material": "Silicon (Si)",
            "depth_um": 20.0,
            "etch_mode": "anisotropic",
        },
        {
            "action": "operation",
            "operation": "Deposit",
            "material": "Aluminum Nitride (AlN)",
            "thickness_um": 0.1,
        },
        {
            "action": "operation",
            "operation": "Deposit",
            "material": "Silicon Carbide (SiC)",
            "thickness_um": 0.01,
        },
        {
            "action": "operation",
            "operation": "Deposit",
            "material": "Titanium Nitride (TiN)",
            "thickness_um": 0.001,
        },
    ]
    _app, win = _new_window()
    for command in commands:
        ok, message = win._execute_assistant_command(command)
        if not ok:
            raise AssertionError(f"20um substrate/film regression command failed: {message}")

    state = win.history[-1]
    grid = np.asarray(state.get("grid"))
    row_heights = np.asarray(state.get("row_heights_um"), dtype=float)
    display_grid = np.asarray(
        win._display_grid_auto(
            grid,
            row_heights_um=row_heights,
            history_index=len(win.history) - 1,
            visual_geometry=state.get("visual_geometry", []),
        )
    )
    columns = (getattr(win, "_last_display_remap_info", {}) or {}).get("column_segments") or []
    outside_col = 0
    trench_col = 150
    if len(columns) <= trench_col:
        raise AssertionError("Expected display segment metadata for the trench center")

    substrate_index = win.material_names.index("Silicon (Si)")
    film_indices = [
        win.material_names.index(name)
        for name in (
            "Aluminum Nitride (AlN)",
            "Silicon Carbide (SiC)",
            "Titanium Nitride (TiN)",
        )
    ]
    outside_substrate = int(np.sum(display_grid[:, outside_col] == substrate_index))
    trench_substrate = int(np.sum(display_grid[:, trench_col] == substrate_index))
    displayed_etch_depth = outside_substrate - trench_substrate
    displayed_film_stack = int(
        sum(np.sum(display_grid[:, trench_col] == material_index) for material_index in film_indices)
    )

    def top_solid_row(col):
        starts = [
            int(segment["display_start"])
            for segment in columns[col]
            if int(segment.get("material", win.VOID_INDEX)) != win.VOID_INDEX
        ]
        return min(starts) if starts else None

    outside_surface = top_solid_row(outside_col)
    trench_surface = top_solid_row(trench_col)
    if outside_surface is None or trench_surface is None:
        raise AssertionError("Both outside and trench columns must contain a solid surface")
    displayed_surface_drop = trench_surface - outside_surface

    if displayed_etch_depth <= displayed_film_stack:
        raise AssertionError(
            f"20um substrate depth must display larger than the 0.111um three-film stack: "
            f"depth={displayed_etch_depth}px, films={displayed_film_stack}px"
        )
    if displayed_surface_drop <= 0:
        raise AssertionError(
            f"The opened PR must leave a visible surface step: "
            f"surface_drop={displayed_surface_drop}px, films={displayed_film_stack}px"
        )
    if any(int(np.sum(display_grid[:, trench_col] == idx)) < int(MIN_LAYER_PX) for idx in film_indices):
        raise AssertionError(
            f"Every 1nm-scale deposited film must receive at least {MIN_LAYER_PX:g} readable rows"
        )
    if outside_substrate != 175:
        raise AssertionError(
            f"The 525um substrate should use the calibrated two-thirds body target: "
            f"substrate={outside_substrate}px, canvas={display_grid.shape[0]}px"
        )
    if display_grid.shape[0] >= 360:
        raise AssertionError(
            f"The readable process must remain understandable in one scrollable view; got {display_grid.shape[0]}px"
        )
    expected_depth = _dynamic_rows_for_um(
        win,
        20.0,
        win._process_display_adjustment_profile(),
        material_idx=win.VOID_INDEX,
    )
    if displayed_etch_depth != expected_depth:
        raise AssertionError(
            f"20um removal and its trench must share one global target, "
            f"expected {expected_depth}, got {displayed_etch_depth}"
        )


def test_readability_band_segmentation_error_is_bounded():
    """Splitting a sub-resolution film may add bands, never unbounded height."""
    _app, win = _new_window()
    win.material_names = ["Void", "Silicon (Si)", "Single", "Part A", "Part B", "Part C"]
    win.material_colors = ["#ffffff"] * len(win.material_names)

    def rendered_height(parts):
        xsec = CrossSection(
            width_px=20,
            height_px=575,
            substrate_index=1,
            void_index=0,
            substrate_thickness_px=525,
            substrate_thickness_um=525.0,
            top_headroom_um=50.0,
        )
        for material_idx, thickness_um in parts:
            xsec.deposit_physical(material_idx, thickness_um)
        state = {
            "grid": xsec.copy_state(),
            "row_heights_um": xsec.copy_row_heights(),
            "visual_geometry": xsec.copy_visual_geometry(),
        }
        win.xsec = xsec
        win.history = [state]
        display = np.asarray(
            win._display_grid_auto(
                state["grid"],
                row_heights_um=state["row_heights_um"],
                history_index=0,
            )
        )
        return int(np.sum(display[:, 0] != win.VOID_INDEX)), int(np.sum(display[:, 0] == 1))

    single_height, single_substrate = rendered_height([(2, 0.111)])
    split_height, split_substrate = rendered_height([(3, 0.1), (4, 0.01), (5, 0.001)])
    if single_substrate != split_substrate:
        raise AssertionError("Film segmentation must never change the rendered substrate core")
    # Each visible band receives its readability floor; the two physical
    # order boundaries may additionally require one explicit order margin.
    maximum_extra_rows = int(3 * MIN_LAYER_PX + 2 * ORDER_MARGIN_PX)
    if split_height - single_height > maximum_extra_rows:
        raise AssertionError(
            f"Three independently visible bands may add at most {maximum_extra_rows} rows versus one merged film, "
            f"got single={single_height}px, split={split_height}px"
        )


def _assert_stats(win, state, material, expected, *, count=None):
    stats = _material_stats(state, win.material_names, material)
    if stats is None:
        raise AssertionError(f"{material}: no material remaining in state {state.get('title')!r}")
    for key in ("min", "median", "max"):
        delta = abs(stats[key] - expected)
        if delta > TOL_UM:
            raise AssertionError(
                f"{material} {key} expected {expected:g} um, got {stats[key]:g} um "
                f"in state {state.get('title')!r}"
            )
    if count is not None and stats["count"] != count:
        raise AssertionError(
            f"{material} count expected {count}, got {stats['count']} in state {state.get('title')!r}"
        )
    return stats


def _assert_planarized_spincoat(stats, requested_thickness_um: float, state_title: str):
    if stats is None:
        raise AssertionError(f"Missing planarized spincoat material in {state_title!r}")
    requested = float(requested_thickness_um)
    if stats["min"] < requested - TOL_UM:
        raise AssertionError(
            f"Planarized spincoat min should stay at least {requested:g} um, "
            f"got {stats['min']:g} um in {state_title!r}"
        )
    if stats["max"] <= requested + TOL_UM:
        raise AssertionError(
            f"Planarized spincoat should visibly fill lower topography above "
            f"{requested:g} um in {state_title!r}; stats={stats}"
        )


def _assert_consistent_display_profile(win, substrate_material=BASE_SUBSTRATE_MATERIAL):
    heights = []
    for idx, state in enumerate(win.history):
        grid = np.asarray(state.get("grid"))
        row_heights = np.asarray(state.get("row_heights_um"), dtype=float)
        if grid.ndim != 2 or grid.size == 0:
            raise AssertionError(f"History step {idx} has invalid grid")
        if row_heights.shape[0] != grid.shape[0]:
            row_heights = np.ones(grid.shape[0], dtype=float)

        display_grid = win._display_grid_auto(
            grid,
            row_heights_um=row_heights,
            history_index=idx,
        )
        height = int(np.asarray(display_grid).shape[0])
        heights.append(height)
        if height <= 0:
            raise AssertionError(f"History step {idx} display is empty")
        info = getattr(win, "_last_display_remap_info", {}) or {}
        column_segments = list(info.get("column_segments", []) or [])
        if not column_segments or not any(column_segments):
            raise AssertionError(f"History step {idx} has no column segments: {info}")

        for material_index in np.unique(grid):
            material_index = int(material_index)
            if material_index == win.VOID_INDEX:
                continue
            physical_cols = np.asarray([
                float(np.sum(row_heights[grid[:, x] == material_index]))
                for x in range(grid.shape[1])
            ])
            display_cols = np.asarray([
                int(np.sum(display_grid[:, x] == material_index))
                for x in range(display_grid.shape[1])
            ])
            if np.any(physical_cols > TOL_UM) and not np.any(display_cols > 0):
                name = (
                    win.material_names[material_index]
                    if 0 <= material_index < len(win.material_names)
                    else str(material_index)
                )
                raise AssertionError(f"{name} is physically present but not visible in step {idx}")

    return {"active": True, "target_height": None}, heights


def _assert_process_wide_material_display_counts(win):
    profile = win._process_display_adjustment_profile()
    target_height = int(profile.get("target_height", 0) or 0)
    if target_height <= 0:
        raise AssertionError(f"Invalid process display profile target height: {profile}")

    observed = {}
    examples = {}
    heights = set()
    for idx, state in enumerate(win.history):
        grid = np.asarray(state.get("grid"))
        row_heights = np.asarray(state.get("row_heights_um"), dtype=float)
        if grid.ndim != 2 or row_heights.shape[0] != grid.shape[0]:
            raise AssertionError(f"Invalid geometry in history state {idx}: {state.get('title', '')!r}")

        display_grid = win._display_grid_auto(
            grid,
            row_heights_um=row_heights,
            history_index=idx,
            visual_geometry=state.get("visual_geometry", []),
            process_profile=profile,
        )
        display_grid = np.asarray(display_grid)
        heights.add(int(display_grid.shape[0]))
        if int(display_grid.shape[0]) != target_height:
            raise AssertionError(
                f"History state {idx} display height should be process-wide {target_height}, "
                f"got {display_grid.shape[0]}"
            )

        info = getattr(win, "_last_display_remap_info", {}) or {}
        for col, column_segments in enumerate(info.get("column_segments", []) or []):
            for segment in column_segments or []:
                material_index = int(segment.get("material", win.VOID_INDEX))
                if material_index == win.VOID_INDEX:
                    continue
                # The local topography-fill portion of a planarizing film is
                # deliberately elastic in display space: it consumes the
                # resolved recess/topography budget so the physical planar
                # interface stays planar. Photoresist provenance may merge
                # across later process steps, so its overburden is checked by
                # the dedicated spin-coat invariant test; this general check
                # continues to cover every non-planarizing layer.
                if (
                    bool(segment.get("is_planarizer_fill", False))
                    or material_index in win._display_photoresist_indices()
                ):
                    continue
                physical_um = round(
                    float(segment.get("y1_um", 0.0)) - float(segment.get("y0_um", 0.0)),
                    9,
                )
                if physical_um <= TOL_UM:
                    continue
                display_rows = int(segment.get("display_end", 0)) - int(segment.get("display_start", 0))
                key = (material_index, physical_um)
                observed.setdefault(key, set()).add(display_rows)
                examples.setdefault(
                    key,
                    (
                        idx,
                        col,
                        win.material_names[material_index]
                        if 0 <= material_index < len(win.material_names)
                        else str(material_index),
                    ),
                )

    if len(heights) != 1:
        raise AssertionError(f"All history states should share one display height, got {sorted(heights)}")

    mismatches = [
        (key, sorted(values), examples.get(key))
        for key, values in observed.items()
        if len(values) > 1
    ]
    if mismatches:
        raise AssertionError(f"Continuous segment display thickness changed across steps: {mismatches[:8]}")


def _assert_etch_geometry_consistent_every_step(win):
    profile = win._process_display_adjustment_profile()
    target_height = int(profile.get("target_height", 0) or 0)
    if target_height <= 0:
        raise AssertionError(f"Invalid stress display profile: {profile}")

    for idx, state in enumerate(win.history):
        grid = np.asarray(state.get("grid"))
        row_heights = np.asarray(state.get("row_heights_um"), dtype=float)
        if grid.ndim != 2 or row_heights.shape[0] != grid.shape[0]:
            raise AssertionError(f"Invalid stress geometry in step {idx}: {state.get('title', '')!r}")

        visual_geometry = list(state.get("visual_geometry", []) or [])
        display_grid = win._display_grid_auto(
            grid,
            row_heights_um=row_heights,
            history_index=idx,
            visual_geometry=visual_geometry,
            process_profile=profile,
        )
        display_grid = np.asarray(display_grid)
        if int(display_grid.shape[0]) != target_height:
            raise AssertionError(
                f"Stress step {idx} should use shared display height {target_height}, "
                f"got {display_grid.shape[0]}"
            )

        for material_index in np.unique(grid):
            material_index = int(material_index)
            if material_index == win.VOID_INDEX:
                continue
            physical_cols = np.asarray([
                float(np.sum(row_heights[grid[:, x] == material_index]))
                for x in range(grid.shape[1])
            ])
            display_cols = np.asarray([
                int(np.sum(display_grid[:, x] == material_index))
                for x in range(display_grid.shape[1])
            ])
            if np.any(physical_cols > TOL_UM) and not np.any(display_cols > 0):
                name = (
                    win.material_names[material_index]
                    if 0 <= material_index < len(win.material_names)
                    else str(material_index)
                )
                raise AssertionError(f"{name} is physically present but invisible in stress step {idx}")

        info = getattr(win, "_last_display_remap_info", {}) or {}
        if info.get("etch_depth_overlays"):
            raise AssertionError(f"Stress step {idx} should render etch geometry directly, not with overlays: {info}")

        for item in visual_geometry:
            if not isinstance(item, dict) or item.get("type") != "etch_depth_feature":
                continue
            if str(item.get("mode", "anisotropic") or "anisotropic").lower() != "anisotropic":
                continue
            for feature in item.get("features", []) or []:
                open_intervals = win._open_etch_void_intervals(grid, row_heights, feature)
                fill_intervals = win._etch_fill_intervals(grid, row_heights, feature)
                if open_intervals:
                    for x_start, x_end, open_y0, open_y1 in open_intervals:
                        for col in range(int(x_start), int(x_end) + 1):
                            visible_void = int(np.sum(display_grid[:, col] == win.VOID_INDEX))
                            if visible_void <= 12:
                                raise AssertionError(
                                    f"Open etched void should render as direct visible void rows in step {idx}, "
                                    f"col={col}, interval={(x_start, x_end, open_y0, open_y1)}, visible_void={visible_void}"
                                )
                if fill_intervals:
                    for x_start, x_end, material_index, fill_y0, fill_y1 in fill_intervals:
                        for col in range(int(x_start), int(x_end) + 1):
                            visible_material = int(np.sum(display_grid[:, col] == int(material_index)))
                            if visible_material <= 0:
                                material_name = (
                                    win.material_names[int(material_index)]
                                    if 0 <= int(material_index) < len(win.material_names)
                                    else str(material_index)
                                )
                                raise AssertionError(
                                    f"Etch fill material {material_name} should render directly in step {idx}, "
                                    f"col={col}, interval={(x_start, x_end, material_index, fill_y0, fill_y1)}"
                                )


def _assert_gap_layer_visual_relation(
    win,
    state,
    layer_materials,
    *,
    substrate_material=BASE_SUBSTRATE_MATERIAL,
):
    if substrate_material not in win.material_names:
        raise AssertionError(f"Missing substrate material {substrate_material!r}")

    substrate_index = win.material_names.index(substrate_material)
    layer_indices = [
        win.material_names.index(material)
        for material in layer_materials
        if material in win.material_names
    ]
    if not layer_indices:
        raise AssertionError(f"No layer materials found from {layer_materials!r}")

    grid = np.asarray(state.get("grid"))
    row_heights = np.asarray(state.get("row_heights_um"), dtype=float)
    if row_heights.shape[0] != grid.shape[0]:
        row_heights = np.ones(grid.shape[0], dtype=float)

    physical_substrate = np.asarray([
        float(np.sum(row_heights[grid[:, x] == substrate_index]))
        for x in range(grid.shape[1])
    ])
    gap_cols = np.where(physical_substrate < physical_substrate.max() - TOL_UM)[0]
    if gap_cols.size == 0:
        raise AssertionError(f"No substrate gap found in state {state.get('title')!r}")
    deepest_substrate = float(physical_substrate[gap_cols].min())
    comparison_cols = gap_cols[
        np.where(physical_substrate[gap_cols] <= deepest_substrate + TOL_UM)[0]
    ]
    if comparison_cols.size == 0:
        comparison_cols = gap_cols

    display_grid = win._display_grid_auto(grid, row_heights_um=row_heights)
    display_substrate = np.asarray([
        int(np.sum(display_grid[:, x] == substrate_index))
        for x in range(display_grid.shape[1])
    ])
    display_gap = int(display_substrate.max() - display_substrate.min())
    physical_gap = float(physical_substrate.max() - physical_substrate.min())

    physical_layer_totals = []
    display_layer_totals = []
    for x in comparison_cols:
        physical_layer_totals.append(
            float(sum(np.sum(row_heights[grid[:, x] == idx]) for idx in layer_indices))
        )
        display_layer_totals.append(
            int(sum(np.sum(display_grid[:, x] == idx) for idx in layer_indices))
        )
    physical_layers = float(max(physical_layer_totals))
    display_layers = int(max(display_layer_totals))

    if physical_layers > physical_gap * 1.2 + TOL_UM and display_layers <= display_gap:
        raise AssertionError(
            f"Layers are physically thicker than gap but do not display thicker: "
            f"layers={physical_layers:g} um/{display_layers} rows, "
            f"gap={physical_gap:g} um/{display_gap} rows in {state.get('title')!r}"
        )
    if physical_layers < physical_gap * 0.8 - TOL_UM and display_layers >= display_gap:
        raise AssertionError(
            f"Layers are physically thinner than gap but do not display thinner: "
            f"layers={physical_layers:g} um/{display_layers} rows, "
            f"gap={physical_gap:g} um/{display_gap} rows in {state.get('title')!r}"
        )

    # Direction alone isn't enough - a population-dependent scale (like the
    # rank+log scheme this replaced) can keep the direction right while still
    # squashing a large physical ratio into a near-1:1 display ratio. Bound
    # the distortion too, but only once both sides are comfortably clear of
    # the floor/cap where true proportionality is expected to hold.
    clear_of_floor_nm = MIN_THICKNESS_NM * 50.0
    clear_of_cap_nm = MAX_THICKNESS_NM * 0.5
    physical_gap_nm = physical_gap * 1000.0
    physical_layers_nm = physical_layers * 1000.0
    if (
        display_gap > 0
        and clear_of_floor_nm < physical_gap_nm < clear_of_cap_nm
        and clear_of_floor_nm < physical_layers_nm < clear_of_cap_nm
    ):
        ratio_physical = physical_layers / physical_gap
        ratio_display = float(display_layers) / float(display_gap)
        # Generous multiplicative slack: real stacks can legitimately shift
        # the ratio (e.g. several floor-anchored segments summed vs. one
        # contiguous gap segment), this only catches gross distortion.
        if ratio_display > ratio_physical * 6.0 + 1.0 or ratio_display * 6.0 + 1.0 < ratio_physical:
            raise AssertionError(
                f"Display ratio is grossly distorted relative to the physical ratio: "
                f"physical layers/gap={ratio_physical:g}, display layers/gap={ratio_display:g} "
                f"(layers={physical_layers:g} um/{display_layers} rows, "
                f"gap={physical_gap:g} um/{display_gap} rows) in {state.get('title')!r}"
            )

    return {
        "physical_gap": physical_gap,
        "display_gap": display_gap,
        "physical_layers": physical_layers,
        "display_layers": display_layers,
    }


def _assert_material_visual_order_in_gap(
    win,
    state,
    thicker_material,
    thinner_material,
    *,
    substrate_material=BASE_SUBSTRATE_MATERIAL,
):
    for material in (substrate_material, thicker_material, thinner_material):
        if material not in win.material_names:
            raise AssertionError(f"Missing material {material!r}")

    substrate_index = win.material_names.index(substrate_material)
    thicker_index = win.material_names.index(thicker_material)
    thinner_index = win.material_names.index(thinner_material)

    grid = np.asarray(state.get("grid"))
    row_heights = np.asarray(state.get("row_heights_um"), dtype=float)
    if row_heights.shape[0] != grid.shape[0]:
        row_heights = np.ones(grid.shape[0], dtype=float)

    physical_substrate = np.asarray([
        float(np.sum(row_heights[grid[:, x] == substrate_index]))
        for x in range(grid.shape[1])
    ])
    gap_cols = np.where(physical_substrate < physical_substrate.max() - TOL_UM)[0]
    if gap_cols.size == 0:
        raise AssertionError(f"No substrate gap found in state {state.get('title')!r}")
    deepest_substrate = float(physical_substrate[gap_cols].min())
    comparison_cols = gap_cols[
        np.where(physical_substrate[gap_cols] <= deepest_substrate + TOL_UM)[0]
    ]
    if comparison_cols.size == 0:
        comparison_cols = gap_cols

    physical_thicker = max(
        float(np.sum(row_heights[grid[:, x] == thicker_index]))
        for x in comparison_cols
    )
    physical_thinner = max(
        float(np.sum(row_heights[grid[:, x] == thinner_index]))
        for x in comparison_cols
    )
    if physical_thicker <= physical_thinner + TOL_UM:
        raise AssertionError(
            f"Expected {thicker_material} to be physically thicker than {thinner_material}, "
            f"got {physical_thicker:g} <= {physical_thinner:g}"
        )

    display_grid = win._display_grid_auto(grid, row_heights_um=row_heights)
    display_thicker = max(
        int(np.sum(display_grid[:, x] == thicker_index))
        for x in comparison_cols
    )
    display_thinner = max(
        int(np.sum(display_grid[:, x] == thinner_index))
        for x in comparison_cols
    )
    if display_thicker < display_thinner:
        raise AssertionError(
            f"{thicker_material} is physically thicker than {thinner_material} but "
            f"displays thinner in {state.get('title')!r}: "
            f"{display_thicker} < {display_thinner}"
        )
    return {
        "physical_thicker": physical_thicker,
        "physical_thinner": physical_thinner,
        "display_thicker": display_thicker,
        "display_thinner": display_thinner,
    }


def _assert_material_display_consistent_across_gap(
    win,
    state,
    material,
    *,
    substrate_material=BASE_SUBSTRATE_MATERIAL,
):
    for required in (substrate_material, material):
        if required not in win.material_names:
            raise AssertionError(f"Missing material {required!r}")

    substrate_index = win.material_names.index(substrate_material)
    material_index = win.material_names.index(material)
    grid = np.asarray(state.get("grid"))
    row_heights = np.asarray(state.get("row_heights_um"), dtype=float)
    if row_heights.shape[0] != grid.shape[0]:
        row_heights = np.ones(grid.shape[0], dtype=float)

    physical_substrate = np.asarray([
        float(np.sum(row_heights[grid[:, x] == substrate_index]))
        for x in range(grid.shape[1])
    ])
    gap_cols = np.where(physical_substrate < physical_substrate.max() - TOL_UM)[0]
    nongap_cols = np.where(physical_substrate >= physical_substrate.max() - TOL_UM)[0]
    if gap_cols.size == 0 or nongap_cols.size == 0:
        raise AssertionError(f"Need both gap and non-gap columns in {state.get('title')!r}")

    def physical_values(cols):
        values = np.asarray([
            float(np.sum(row_heights[grid[:, x] == material_index]))
            for x in cols
        ])
        return values[values > TOL_UM]

    physical_gap = physical_values(gap_cols)
    physical_nongap = physical_values(nongap_cols)
    if physical_gap.size == 0 or physical_nongap.size == 0:
        raise AssertionError(f"{material} must exist both inside and outside the gap")
    if (
        abs(float(np.median(physical_gap)) - float(np.median(physical_nongap))) > TOL_UM
        or abs(float(physical_gap.min()) - float(physical_gap.max())) > TOL_UM
        or abs(float(physical_nongap.min()) - float(physical_nongap.max())) > TOL_UM
    ):
        raise AssertionError(
            f"{material} is not physically equal inside/outside gap in {state.get('title')!r}: "
            f"gap={physical_gap}, nongap={physical_nongap}"
        )

    display_grid = win._display_grid_auto(grid, row_heights_um=row_heights)

    def display_values(cols):
        values = np.asarray([
            int(np.sum(display_grid[:, x] == material_index))
            for x in cols
        ])
        return values[values > 0]

    display_gap = display_values(gap_cols)
    display_nongap = display_values(nongap_cols)
    if display_gap.size == 0 or display_nongap.size == 0:
        raise AssertionError(f"{material} must display both inside and outside the gap")
    if int(display_gap.min()) != int(display_gap.max()):
        raise AssertionError(f"{material} display varies inside gap: {display_gap}")
    if int(display_nongap.min()) != int(display_nongap.max()):
        raise AssertionError(f"{material} display varies outside gap: {display_nongap}")
    if int(display_gap[0]) != int(display_nongap[0]):
        raise AssertionError(
            f"{material} has equal physical thickness but different display thickness "
            f"in {state.get('title')!r}: gap={int(display_gap[0])}, "
            f"nongap={int(display_nongap[0])}"
        )
    return {
        "physical_um": float(np.median(physical_gap)),
        "display_rows": int(display_gap[0]),
    }


def _base_substrate_commands():
    return [
        {"action": "add_material", "material": BASE_SUBSTRATE_MATERIAL},
        {"action": "set_substrate", "material": BASE_SUBSTRATE_MATERIAL, "thickness_um": 525.0},
    ]


def _selected_substrate_material(win, fallback=BASE_SUBSTRATE_MATERIAL):
    combo = getattr(win, "substrate_combo", None)
    if combo is not None:
        try:
            selected = str(combo.currentText() or "").strip()
        except Exception:
            selected = ""
        if selected and selected in win.material_names:
            return selected
    if fallback in win.material_names:
        return fallback
    if len(win.material_names) > 1:
        return win.material_names[1]
    return fallback


def test_shallow_isotropic_etch():
    commands = _base_substrate_commands() + [
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 10.0},
        {
            "action": "operation",
            "operation": "Lithography",
            "material": "PR_SU-8 Photoresist",
            "windows": [(0.35, 0.65)],
            "invert_lithography": False,
        },
        {
            "action": "operation",
            "operation": "Etch",
            "material": BASE_SUBSTRATE_MATERIAL,
            "depth_um": 0.2,
            "etch_mode": "isotropic",
            "method": "HF",
        },
    ]
    win, path = _run_commands("FabStack_dimtest_shallow_isotropic", commands)
    state = _state_by_operation(win.history, "Etch")
    stats = _material_stats(state, win.material_names, BASE_SUBSTRATE_MATERIAL)
    if stats is None or abs(stats["min"] - 524.8) > TOL_UM:
        raise AssertionError(f"Shallow isotropic etch should remove 0.2 um vertically, stats={stats}")
    return path


def test_spincoat_over_topography():
    commands = _base_substrate_commands() + [
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "add_material", "material": "Chromium"},
        {"action": "add_material", "material": "Gold"},
        {"action": "add_material", "material": "Silicon Dioxide (SiO2)"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 10.0},
        {
            "action": "operation",
            "operation": "Lithography",
            "material": "PR_SU-8 Photoresist",
            "windows": [(0.35, 0.65)],
            "invert_lithography": False,
        },
        {"action": "operation", "operation": "Etch", "material": BASE_SUBSTRATE_MATERIAL, "depth_um": 0.2, "etch_mode": "isotropic"},
        {"action": "operation", "operation": "Deposit", "material": "Chromium", "thickness_um": 0.015},
        {"action": "operation", "operation": "Deposit", "material": "Gold", "thickness_um": 0.15},
        {"action": "operation", "operation": "Lift-off", "material": "PR_SU-8 Photoresist"},
        {"action": "operation", "operation": "Deposit", "material": "Silicon Dioxide (SiO2)", "thickness_um": 0.1},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 10.0},
    ]
    win, path = _run_commands("FabStack_dimtest_spincoat_over_topography", commands)
    state = _state_by_operation(win.history, "Spincoating", occurrence=2)
    stats = _material_stats(state, win.material_names, "PR_SU-8 Photoresist")
    _assert_planarized_spincoat(stats, 10.0, state.get("title", ""))
    _assert_stats(win, state, "Silicon Dioxide (SiO2)", 0.1, count=300)
    _assert_consistent_display_profile(win)
    return path


def test_spincoat_fills_recess_and_preserves_planar_interface():
    commands = _base_substrate_commands() + [
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "add_material", "material": "Aluminum Nitride (AlN)"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 1.0},
        {
            "action": "operation",
            "operation": "Lithography",
            "material": "PR_SU-8 Photoresist",
            "windows": [(0.40, 0.60)],
            "invert_lithography": False,
        },
        {"action": "operation", "operation": "Etch", "material": BASE_SUBSTRATE_MATERIAL, "depth_um": 20.0, "etch_mode": "anisotropic"},
        {"action": "operation", "operation": "Deposit", "material": "Aluminum Nitride (AlN)", "thickness_um": 0.1},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 1.0},
    ]
    win, path = _run_commands("FabStack_dimtest_spincoat_recess_interface", commands)
    state = _state_by_operation(win.history, "Spincoating", occurrence=2)
    state_index = next(idx for idx, item in enumerate(win.history) if item is state)
    grid = np.asarray(state.get("grid"))
    heights = np.asarray(state.get("row_heights_um"), dtype=float)
    edges = np.concatenate(([0.0], np.cumsum(heights)))

    physical_tops = []
    for col in range(grid.shape[1]):
        non_void = np.where(grid[:, col] != win.VOID_INDEX)[0]
        physical_tops.append(float(edges[int(non_void.min())]))
    if np.ptp(np.asarray(physical_tops, dtype=float)) > TOL_UM:
        raise AssertionError(f"Spin coat physical free surface is not planar: {physical_tops}")

    substrate = _material_column_heights(state, win.material_names, BASE_SUBSTRATE_MATERIAL)
    if abs(float(substrate[0]) - 525.0) > TOL_UM or abs(float(substrate[150]) - 505.0) > TOL_UM:
        raise AssertionError(
            f"Spin coating must not move the etched substrate, got outside/inside={substrate[0]:g}/{substrate[150]:g} um"
        )
    pr = _material_column_heights(state, win.material_names, "PR_SU-8 Photoresist")
    if float(pr[150]) <= float(pr[0]) + 18.0:
        raise AssertionError(
            f"Planarizer should consume the remaining trench height, got outside/inside={pr[0]:g}/{pr[150]:g} um"
        )

    profile = win._process_display_adjustment_profile()
    win._display_grid_auto(
        grid,
        row_heights_um=heights,
        history_index=state_index,
        visual_geometry=state.get("visual_geometry", []),
        process_profile=profile,
    )
    columns = (getattr(win, "_last_display_remap_info", {}) or {}).get("column_segments", [])
    if not columns:
        raise AssertionError("Spin-coat display did not produce column segments")
    planarizer_grid = np.asarray(
        state.get("planarizer_role_grid", np.zeros_like(grid, dtype=np.int32)),
        dtype=np.int32,
    )
    current_planarizer_role = max(
        (abs(int(value)) for value in np.unique(planarizer_grid) if int(value) != 0),
        default=0,
    )

    display_tops = []
    overburden_rows = []
    for segments in columns:
        solids = [seg for seg in segments if int(seg.get("material", win.VOID_INDEX)) != win.VOID_INDEX]
        if not solids:
            raise AssertionError("Spin-coated display contains an empty column")
        display_tops.append(min(int(seg.get("display_start", 0)) for seg in solids))
        overburden = [
            seg for seg in segments
            if int(seg.get("planarizer_role", 0)) == int(current_planarizer_role)
            and bool(seg.get("is_planarizer_overburden", False))
        ]
        if len(overburden) != 1:
            raise AssertionError(f"Expected one explicit spin overburden per column, got {overburden}")
        overburden_rows.append(
            int(overburden[0].get("display_end", 0)) - int(overburden[0].get("display_start", 0))
        )
    if len(set(display_tops)) != 1:
        raise AssertionError(f"Displayed spin free surface must be one plane, got {sorted(set(display_tops))}")
    if len(set(overburden_rows)) != 1 or overburden_rows[0] < int(MIN_LAYER_PX):
        raise AssertionError(f"The common 1 um spin overburden must keep one visible scale, got {overburden_rows}")

    outside = columns[0]
    center = columns[150]
    center_recess = [seg for seg in center if bool(seg.get("in_substrate_recess", False))]
    if any(int(seg.get("material", win.VOID_INDEX)) == win.VOID_INDEX for seg in center_recess):
        raise AssertionError(f"Spin coat should leave no void inside the filled recess: {center_recess}")
    outside_core = next(seg for seg in outside if bool(seg.get("is_substrate_core", False)))
    center_core = next(seg for seg in center if bool(seg.get("is_substrate_core", False)))
    displayed_gap = (
        int(outside_core.get("display_end", 0)) - int(outside_core.get("display_start", 0))
        - int(center_core.get("display_end", 0)) + int(center_core.get("display_start", 0))
    )
    recess_fill = sum(
        int(seg.get("display_end", 0)) - int(seg.get("display_start", 0))
        for seg in center_recess
    )
    if recess_fill != displayed_gap:
        raise AssertionError(
            f"Recess must have one conserved display budget, got fill={recess_fill} rows and gap={displayed_gap} rows"
        )

    for factor in (FEATURE_EMPHASIS_MIN, FEATURE_EMPHASIS_DEFAULT, FEATURE_EMPHASIS_MAX):
        win.feature_scale_factor = factor
        factor_profile = win._process_display_adjustment_profile()
        win._display_grid_auto(
            grid,
            row_heights_um=heights,
            history_index=state_index,
            visual_geometry=state.get("visual_geometry", []),
            process_profile=factor_profile,
        )
        factor_columns = (getattr(win, "_last_display_remap_info", {}) or {}).get("column_segments", [])
        factor_tops = []
        for segments in factor_columns:
            solids = [seg for seg in segments if int(seg.get("material", win.VOID_INDEX)) != win.VOID_INDEX]
            factor_tops.append(min(int(seg.get("display_start", 0)) for seg in solids))
        if len(set(factor_tops)) != 1:
            raise AssertionError(
                f"Feature emphasis {factor:g} broke the planar spin-coat free surface: {sorted(set(factor_tops))}"
            )
        factor_outside = factor_columns[0]
        factor_center = factor_columns[150]
        factor_recess = [seg for seg in factor_center if bool(seg.get("in_substrate_recess", False))]
        if any(int(seg.get("material", win.VOID_INDEX)) == win.VOID_INDEX for seg in factor_recess):
            raise AssertionError(f"Feature emphasis {factor:g} reopened a spin-filled trench")
        outside_core = next(seg for seg in factor_outside if bool(seg.get("is_substrate_core", False)))
        center_core = next(seg for seg in factor_center if bool(seg.get("is_substrate_core", False)))
        factor_gap = (
            int(outside_core.get("display_end", 0)) - int(outside_core.get("display_start", 0))
            - int(center_core.get("display_end", 0)) + int(center_core.get("display_start", 0))
        )
        factor_fill = sum(
            int(seg.get("display_end", 0)) - int(seg.get("display_start", 0))
            for seg in factor_recess
        )
        if factor_fill != factor_gap:
            raise AssertionError(
                f"Feature emphasis {factor:g} broke spin-fill conservation: fill={factor_fill}, gap={factor_gap}"
            )
    return path


def _planarized_recess_commands():
    return [
        {"action": "set_substrate", "material": "Silicon (Si)", "thickness_um": 500.0},
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {
            "action": "operation",
            "operation": "Spincoating",
            "material": "PR_SU-8 Photoresist",
            "thickness_um": 1.0,
        },
        {
            "action": "operation",
            "operation": "Lithography",
            "material": "PR_SU-8 Photoresist",
            "windows": [(0.40, 0.60)],
            "invert_lithography": False,
        },
        {
            "action": "operation",
            "operation": "Etch",
            "material": "Silicon (Si)",
            "depth_um": 20.0,
            "etch_mode": "anisotropic",
        },
        {"action": "operation", "operation": "Strip material", "material": "PR_SU-8 Photoresist"},
        {
            "action": "operation",
            "operation": "Spincoating",
            "material": "PR_SU-8 Photoresist",
            "thickness_um": 1.0,
        },
    ]


def _window_from_commands(commands):
    _app, win = _new_window()
    for command in commands:
        ok, message = win._execute_assistant_command(command)
        if not ok:
            raise AssertionError(f"Command failed: {command}: {message}")
    return win


def _render_last_state_segments(win, factor):
    return _render_state_segments(win, len(win.history) - 1, factor)


def _render_state_segments(win, state_index, factor):
    win.feature_scale_factor = float(factor)
    state_index = int(state_index)
    state = win.history[state_index]
    profile = win._process_display_adjustment_profile()
    win._display_grid_auto(
        np.asarray(state.get("grid")),
        row_heights_um=np.asarray(state.get("row_heights_um"), dtype=float),
        history_index=state_index,
        visual_geometry=state.get("visual_geometry", []),
        process_profile=profile,
    )
    info = getattr(win, "_last_display_remap_info", {}) or {}
    return state, profile, info


def _material_top_from_display_bottom(win, info, material_name, col):
    material_idx = win.material_names.index(material_name)
    height = int(info.get("display_height", 0))
    segments = [
        segment for segment in info.get("column_segments", [])[int(col)]
        if int(segment.get("material", -1)) == int(material_idx)
        and not bool(segment.get("is_structural_body", False))
    ]
    if not segments:
        return None
    return max(height - int(segment.get("display_start", height)) for segment in segments)


def test_bond_after_planarized_recess_preserves_support_plane():
    pre = _window_from_commands(_planarized_recess_commands())
    post = _window_from_commands(
        _planarized_recess_commands()
        + [{
            "action": "operation",
            "operation": "Bond",
            "material": "Silicon (Si)",
            "thickness_um": 100.0,
        }]
    )

    for factor in (FEATURE_EMPHASIS_MIN, FEATURE_EMPHASIS_DEFAULT, FEATURE_EMPHASIS_MAX):
        _pre_state, pre_profile, pre_info = _render_last_state_segments(pre, factor)
        post_state, post_profile, post_info = _render_last_state_segments(post, factor)
        pre_tops = [
            _material_top_from_display_bottom(pre, pre_info, "PR_SU-8 Photoresist", col)
            for col in (0, 150, 299)
        ]
        post_tops = [
            _material_top_from_display_bottom(post, post_info, "PR_SU-8 Photoresist", col)
            for col in (0, 150, 299)
        ]
        if len(set(pre_tops)) != 1 or post_tops != pre_tops:
            raise AssertionError(
                f"Bond must not lift a planar layer over the20um recess at {factor:g}: "
                f"before={pre_tops}, after={post_tops}"
            )
        gap_key = post._display_feature_profile_key(20.0)
        if (
            int(pre_profile["resolved_feature_rows"][gap_key])
            != int(post_profile["resolved_feature_rows"][gap_key])
        ):
            raise AssertionError(
                f"Bond must not remap the existing20um etch at {factor:g}"
            )

        height = int(post_info.get("display_height", 0))
        body_spans = []
        for col in (0, 150, 299):
            body = [
                segment for segment in post_info["column_segments"][col]
                if int(segment.get("structural_body_id", 0)) == 2
            ]
            if len(body) != 1:
                raise AssertionError(f"Missing rigid bonded body in column {col}: {body}")
            body_spans.append((
                height - int(body[0]["display_end"]),
                height - int(body[0]["display_start"]),
            ))
        if len(set(body_spans)) != 1:
            raise AssertionError(
                f"Bonded substrate must be flat above the recess at {factor:g}: {body_spans}"
            )
        if not np.any(np.asarray(post_state.get("planarizer_role_grid")) != 0):
            raise AssertionError("Bond state lost the prior spincoat provenance")

    data = post._make_process_dict()
    _app, loaded = _new_window()
    loaded._load_process_dict(data)
    loaded_state = loaded.history[-1]
    if not np.any(np.asarray(loaded_state.get("planarizer_role_grid")) != 0):
        raise AssertionError("Save/load or replay lost planarizer provenance after Bond")


def test_second_lithography_preserves_prior_planarizer_plane():
    pre = _window_from_commands(_planarized_recess_commands())
    post = _window_from_commands(
        _planarized_recess_commands()
        + [{
            "action": "operation",
            "operation": "Lithography",
            "material": "PR_SU-8 Photoresist",
            "windows": [(0.20, 0.30)],
            "invert_lithography": False,
        }]
    )
    pre_state = pre.history[-1]
    post_state = post.history[-1]
    pre_grid = np.asarray(pre_state.get("grid"))
    post_grid = np.asarray(post_state.get("grid"))
    pre_heights = np.asarray(pre_state.get("row_heights_um"), dtype=float)
    post_heights = np.asarray(post_state.get("row_heights_um"), dtype=float)
    if not np.array_equal(pre_heights, post_heights):
        raise AssertionError("Lithography must not change physical row heights")
    if not np.array_equal(
        np.asarray(pre_state.get("structural_role_grid")),
        np.asarray(post_state.get("structural_role_grid")),
    ):
        raise AssertionError("Lithography must not move structural-body cells")
    changed = pre_grid != post_grid
    expected_window = post.xsec.window_mask(0.20, 0.30)
    pr_index = post.material_names.index("PR_SU-8 Photoresist")
    changed_rows, changed_cols = np.where(changed)
    if changed_cols.size == 0:
        raise AssertionError("Second lithography did not open its requested window")
    if not np.all(expected_window[changed_cols]):
        raise AssertionError("Second lithography changed material outside its own window")
    if not np.all(pre_grid[changed_rows, changed_cols] == pr_index):
        raise AssertionError("Second lithography changed a non-photoresist material")

    for factor in (FEATURE_EMPHASIS_MIN, FEATURE_EMPHASIS_DEFAULT, FEATURE_EMPHASIS_MAX):
        _state, pre_profile, pre_info = _render_last_state_segments(pre, factor)
        _state, post_profile, post_info = _render_last_state_segments(post, factor)
        pre_tops = [
            _material_top_from_display_bottom(pre, pre_info, "PR_SU-8 Photoresist", col)
            for col in (0, 150, 299)
        ]
        post_tops = [
            _material_top_from_display_bottom(post, post_info, "PR_SU-8 Photoresist", col)
            for col in (0, 150, 299)
        ]
        if post_tops[0] != pre_tops[0] or post_tops[1:] != pre_tops[1:]:
            raise AssertionError(
                f"Second lithography shifted surviving planar material at {factor:g}: "
                f"before={pre_tops}, after={post_tops}"
            )
        if _material_top_from_display_bottom(
            post,
            post_info,
            "PR_SU-8 Photoresist",
            75,
        ) is not None:
            raise AssertionError("Positive second lithography must leave its selected window open")
        gap_key = post._display_feature_profile_key(20.0)
        if (
            int(pre_profile["resolved_feature_rows"][gap_key])
            != int(post_profile["resolved_feature_rows"][gap_key])
            or int(pre_profile["structural_body_rows"][1])
            != int(post_profile["structural_body_rows"][1])
        ):
            raise AssertionError(
                f"Second lithography remapped the original etch/substrate at {factor:g}"
            )


def test_rigid_bond_planes_persist_through_descendants():
    commands = [
        {"action": "set_substrate", "material": "Silicon (Si)", "thickness_um": 500.0},
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 1.0},
        {"action": "operation", "operation": "Lithography", "material": "PR_SU-8 Photoresist", "windows": [(0.40, 0.60)], "invert_lithography": False},
        {"action": "operation", "operation": "Etch", "material": "Silicon (Si)", "depth_um": 20.0, "etch_mode": "anisotropic"},
        {"action": "operation", "operation": "Strip material", "material": "PR_SU-8 Photoresist"},
        {"action": "operation", "operation": "Bond", "material": "Silicon (Si)", "thickness_um": 100.0},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 2.0},
        {"action": "operation", "operation": "Lithography", "material": "PR_SU-8 Photoresist", "windows": [(0.20, 0.30)], "invert_lithography": False},
    ]
    win = _window_from_commands(commands)
    descendant_indices = [
        idx for idx, state in enumerate(win.history)
        if state.get("operation") in {"Bond", "Spincoating", "Lithography"}
        and any(
            int(value) == 2
            for value in np.unique(np.asarray(state.get("structural_role_grid")))
        )
    ]
    if len(descendant_indices) != 3:
        raise AssertionError(f"Expected Bond/Spin/Lithography descendants, got {descendant_indices}")

    for factor in (FEATURE_EMPHASIS_MIN, FEATURE_EMPHASIS_DEFAULT, FEATURE_EMPHASIS_MAX):
        win.feature_scale_factor = factor
        profile = win._process_display_adjustment_profile()
        reference_span = None
        for idx in descendant_indices:
            state = win.history[idx]
            win._display_grid_auto(
                np.asarray(state.get("grid")),
                row_heights_um=np.asarray(state.get("row_heights_um"), dtype=float),
                history_index=idx,
                process_profile=profile,
            )
            info = getattr(win, "_last_display_remap_info", {}) or {}
            spans = []
            for col in (0, 75, 150, 299):
                body = [
                    segment for segment in info["column_segments"][col]
                    if int(segment.get("structural_body_id", 0)) == 2
                ]
                if len(body) != 1:
                    raise AssertionError(f"Descendant lost bonded body2 at {factor:g}/{idx}/{col}")
                spans.append((int(body[0]["display_start"]), int(body[0]["display_end"])))
            if len(set(spans)) != 1:
                raise AssertionError(
                    f"Rigid body bent in descendant state {idx} at {factor:g}: {spans}"
                )
            if reference_span is None:
                reference_span = spans[0]
            elif spans[0] != reference_span:
                raise AssertionError(
                    f"Rigid body moved between Bond and later operations at {factor:g}: "
                    f"{reference_span} -> {spans[0]}"
                )


def test_partial_etch_of_planarizer_keeps_real_step():
    base = [
        {"action": "set_substrate", "material": "Silicon (Si)", "thickness_um": 500.0},
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "add_material", "material": "Polyimide"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 1.0},
        {"action": "operation", "operation": "Lithography", "material": "PR_SU-8 Photoresist", "windows": [(0.40, 0.60)], "invert_lithography": False},
        {"action": "operation", "operation": "Etch", "material": "Silicon (Si)", "depth_um": 20.0, "etch_mode": "anisotropic"},
        {"action": "operation", "operation": "Strip material", "material": "PR_SU-8 Photoresist"},
        {"action": "operation", "operation": "Spincoating", "material": "Polyimide", "thickness_um": 5.0},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 1.0},
        {"action": "operation", "operation": "Lithography", "material": "PR_SU-8 Photoresist", "windows": [(0.40, 0.60)], "invert_lithography": False},
    ]

    # Exercise an etch within the overburden, exactly through it, and into
    # the recess fill.  None may be visually flattened back to the old plane.
    for etch_depth_um in (0.001, 1.0, 2.0, 5.0, 8.0, 12.0, 12.5, 13.0):
        win = _window_from_commands(base + [{
            "action": "operation",
            "operation": "Etch",
            "material": "Polyimide",
            "depth_um": etch_depth_um,
            "etch_mode": "anisotropic",
        }])
        state = win.history[-1]
        grid = np.asarray(state.get("grid"))
        heights = np.asarray(state.get("row_heights_um"), dtype=float)
        edges = np.concatenate(([0.0], np.cumsum(heights)))
        pi_index = win.material_names.index("Polyimide")

        physical_tops = []
        for col in (0, 150):
            rows = np.where(grid[:, col] == pi_index)[0]
            if not rows.size:
                raise AssertionError(
                    f"The {etch_depth_um:g}um etch unexpectedly removed all Polyimide in column {col}"
                )
            physical_tops.append(float(edges[-1] - edges[int(rows.min())]))
        physical_step = physical_tops[0] - physical_tops[1]
        if abs(physical_step - etch_depth_um) > TOL_UM:
            raise AssertionError(
                f"Physical Polyimide step should be {etch_depth_um:g}um, got {physical_step:g}um"
            )

        for factor in (FEATURE_EMPHASIS_MIN, FEATURE_EMPHASIS_DEFAULT, FEATURE_EMPHASIS_MAX):
            _state, profile, info = _render_last_state_segments(win, factor)
            outside_top = _material_top_from_display_bottom(win, info, "Polyimide", 0)
            etched_top = _material_top_from_display_bottom(win, info, "Polyimide", 150)
            gap_key = win._display_feature_profile_key(etch_depth_um)
            if gap_key not in profile.get("resolved_feature_rows", {}):
                raise AssertionError(
                    f"The {etch_depth_um:g}um planarizer removal has no resolved feature scale"
                )
            expected_step = int(profile["resolved_feature_rows"][gap_key])
            displayed_step = (
                int(outside_top) - int(etched_top)
                if outside_top is not None and etched_top is not None
                else 0
            )
            if displayed_step != expected_step:
                raise AssertionError(
                    f"A real {etch_depth_um:g}um planarizer etch did not receive its calibrated "
                    f"scale at emphasis {factor:g}: displayed={displayed_step}, "
                    f"expected={expected_step}, outside/etched={outside_top}/{etched_top}"
                )
            center_pi_rows = sum(
                int(segment.get("display_end", 0)) - int(segment.get("display_start", 0))
                for segment in info.get("column_segments", [])[150]
                if int(segment.get("material", -1)) == int(pi_index)
            )
            if center_pi_rows < int(MIN_LAYER_PX):
                raise AssertionError(
                    f"Remaining Polyimide became unreadable after {etch_depth_um:g}um etch: "
                    f"{center_pi_rows} rows"
                )
            remaining_pi_um = 25.0 - etch_depth_um
            if remaining_pi_um > etch_depth_um + TOL_UM and not center_pi_rows > displayed_step:
                raise AssertionError(
                    f"Remaining {remaining_pi_um:g}um planarizer stack must display larger than "
                    f"the {etch_depth_um:g}um etch: remaining/gap={center_pi_rows}/{displayed_step}"
                )
            if abs(remaining_pi_um - etch_depth_um) <= TOL_UM and center_pi_rows != displayed_step:
                raise AssertionError(
                    f"Equal remaining/gap {remaining_pi_um:g}um features must share one scale: "
                    f"{center_pi_rows}/{displayed_step}"
                )
            if remaining_pi_um + TOL_UM < etch_depth_um and not center_pi_rows < displayed_step:
                raise AssertionError(
                    f"Remaining {remaining_pi_um:g}um planarizer stack must display smaller than "
                    f"the {etch_depth_um:g}um etch: remaining/gap={center_pi_rows}/{displayed_step}"
                )

            pr_index = win.material_names.index("PR_SU-8 Photoresist")
            mask_rows = sum(
                int(segment.get("display_end", 0)) - int(segment.get("display_start", 0))
                for segment in info.get("column_segments", [])[0]
                if int(segment.get("material", -1)) == int(pr_index)
            )
            if etch_depth_um < 1.0 - TOL_UM and not displayed_step < mask_rows:
                raise AssertionError(
                    f"{etch_depth_um:g}um etch must display smaller than the 1um mask: "
                    f"etch/mask={displayed_step}/{mask_rows}"
                )
            if abs(etch_depth_um - 1.0) <= TOL_UM and displayed_step != mask_rows:
                raise AssertionError(
                    f"Equal 1um etch/mask must share one scale: {displayed_step}/{mask_rows}"
                )
            if etch_depth_um > 1.0 + TOL_UM and not displayed_step > mask_rows:
                raise AssertionError(
                    f"{etch_depth_um:g}um etch must display larger than the 1um mask: "
                    f"etch/mask={displayed_step}/{mask_rows}"
                )

    monotonic = _window_from_commands(base + [
        {
            "action": "operation",
            "operation": "Etch",
            "material": "Polyimide",
            "depth_um": 0.001,
            "etch_mode": "anisotropic",
        },
        {"action": "add_material", "material": "Silicon Dioxide (SiO2)"},
        {
            "action": "operation",
            "operation": "Deposit",
            "material": "Silicon Dioxide (SiO2)",
            "thickness_um": 6.0,
        },
    ])
    for factor in (FEATURE_EMPHASIS_MIN, FEATURE_EMPHASIS_DEFAULT, FEATURE_EMPHASIS_MAX):
        monotonic.feature_scale_factor = factor
        profile = monotonic._process_display_adjustment_profile()
        resolved = profile.get("resolved_feature_rows", {})
        keys = [
            monotonic._display_feature_profile_key(value)
            for value in (4.999, 5.0, 6.0)
        ]
        if any(key not in resolved for key in keys):
            raise AssertionError(f"Missing promoted monotonic feature keys: {keys}")
        rows = tuple(int(resolved[key]) for key in keys)
        if not rows[0] < rows[1] < rows[2]:
            raise AssertionError(
                f"Partial-etch promotion inverted4.999/5/6um order at {factor:g}: {rows}"
            )
        if profile.get("constraint_violations"):
            raise AssertionError(
                f"Partial-etch monotonic profile has violations: {profile['constraint_violations']}"
            )


def test_flipped_planarizer_preserves_unequal_fill_depths():
    commands = [
        {"action": "set_substrate", "material": "Silicon (Si)", "thickness_um": 500.0},
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 1.0},
        {"action": "operation", "operation": "Lithography", "material": "PR_SU-8 Photoresist", "windows": [(0.20, 0.30)], "invert_lithography": False},
        {"action": "operation", "operation": "Etch", "material": "Silicon (Si)", "depth_um": 20.0, "etch_mode": "anisotropic"},
        {"action": "operation", "operation": "Strip material", "material": "PR_SU-8 Photoresist"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 1.0},
        {"action": "operation", "operation": "Lithography", "material": "PR_SU-8 Photoresist", "windows": [(0.70, 0.80)], "invert_lithography": False},
        {"action": "operation", "operation": "Etch", "material": "Silicon (Si)", "depth_um": 40.0, "etch_mode": "anisotropic"},
        {"action": "operation", "operation": "Strip material", "material": "PR_SU-8 Photoresist"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 1.0},
        {"action": "operation", "operation": "Flip", "material": ""},
    ]
    win = _window_from_commands(commands)
    before_index = len(win.history) - 2
    after_index = len(win.history) - 1
    before_roles = np.asarray(win.history[before_index].get("planarizer_role_grid"))
    role_id = max(
        (abs(int(value)) for value in np.unique(before_roles) if int(value) != 0),
        default=0,
    )
    if role_id <= 0:
        raise AssertionError("The final planarizing spin coat has no persistent role")

    def role_rows(info, col, signed_role):
        return sum(
            int(segment.get("display_end", 0)) - int(segment.get("display_start", 0))
            for segment in info.get("column_segments", [])[int(col)]
            if int(segment.get("planarizer_role", 0)) == int(signed_role)
        )

    expected_by_factor = {}
    for factor in (FEATURE_EMPHASIS_MIN, FEATURE_EMPHASIS_DEFAULT, FEATURE_EMPHASIS_MAX):
        _state, _profile, before_info = _render_state_segments(win, before_index, factor)
        _state, _profile, after_info = _render_state_segments(win, after_index, factor)
        before_fill = (
            role_rows(before_info, 75, -role_id),
            role_rows(before_info, 225, -role_id),
        )
        after_fill = (
            role_rows(after_info, 75, -role_id),
            role_rows(after_info, 225, -role_id),
        )
        if not (before_fill[0] < before_fill[1]):
            raise AssertionError(f"Before Flip,20um/40um fill order is wrong: {before_fill}")
        if after_fill != before_fill:
            raise AssertionError(
                f"Flip changed or equalised the20um/40um fills at {factor:g}: "
                f"before={before_fill}, after={after_fill}"
            )
        overburden_rows = [
            role_rows(after_info, col, role_id) for col in (0, 75, 150, 225, 299)
        ]
        if len(set(overburden_rows)) != 1 or overburden_rows[0] < int(MIN_LAYER_PX):
            raise AssertionError(
                f"Flip changed the common visible overburden at {factor:g}: {overburden_rows}"
            )
        height = int(after_info.get("display_height", 0))
        outer_planes = []
        for col in (0, 75, 150, 225, 299):
            positive = [
                segment for segment in after_info["column_segments"][col]
                if int(segment.get("planarizer_role", 0)) == int(role_id)
            ]
            outer_planes.append(height - max(int(segment["display_end"]) for segment in positive))
        if len(set(outer_planes)) != 1:
            raise AssertionError(
                f"Flipped planarizer outer surface is not flat at {factor:g}: {outer_planes}"
            )
        expected_by_factor[float(factor)] = (after_fill, overburden_rows[0], outer_planes[0])

    data = win._make_process_dict()
    _app, loaded = _new_window()
    loaded._load_process_dict(data)
    loaded_metadata = loaded.history[-1].get("planarizer_roles", {}) or {}
    loaded_role = loaded_metadata.get(role_id, loaded_metadata.get(str(role_id), {}))
    if int((loaded_role or {}).get("orientation", 0)) != -1:
        raise AssertionError(f"Save/load lost flipped planarizer orientation: {loaded_role}")
    for factor, expected in expected_by_factor.items():
        _state, _profile, loaded_info = _render_last_state_segments(loaded, factor)
        loaded_fill = (
            role_rows(loaded_info, 75, -role_id),
            role_rows(loaded_info, 225, -role_id),
        )
        height = int(loaded_info.get("display_height", 0))
        positive = [
            segment for segment in loaded_info["column_segments"][75]
            if int(segment.get("planarizer_role", 0)) == int(role_id)
        ]
        loaded_values = (
            loaded_fill,
            role_rows(loaded_info, 75, role_id),
            height - max(int(segment["display_end"]) for segment in positive),
        )
        if loaded_values != expected:
            raise AssertionError(
                f"Save/load changed flipped planarizer display at {factor:g}: "
                f"expected={expected}, loaded={loaded_values}"
            )


def test_planar_membrane_uses_real_support_void_after_flip_back():
    base = [
        {"action": "set_substrate", "material": "Silicon (Si)", "thickness_um": 500.0},
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "add_material", "material": "Polyimide"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 1.0},
        {"action": "operation", "operation": "Lithography", "material": "PR_SU-8 Photoresist", "windows": [(0.40, 0.60)], "invert_lithography": False},
        {"action": "operation", "operation": "Etch", "material": "Silicon (Si)", "depth_um": 20.0, "etch_mode": "anisotropic"},
        {"action": "operation", "operation": "Strip material", "material": "PR_SU-8 Photoresist"},
        {"action": "operation", "operation": "Spincoating", "material": "Polyimide", "thickness_um": 5.0},
        {"action": "operation", "operation": "Flip", "material": ""},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 1.0},
        {"action": "operation", "operation": "Lithography", "material": "PR_SU-8 Photoresist", "windows": [(0.40, 0.60)], "invert_lithography": False},
        {"action": "operation", "operation": "Etch", "material": "Silicon (Si)", "depth_um": 480.0, "etch_mode": "anisotropic"},
    ]

    def build(remove_fill_um):
        commands = list(base)
        if remove_fill_um > 0.0:
            commands.append({
                "action": "operation",
                "operation": "Etch",
                "material": "Polyimide",
                "depth_um": float(remove_fill_um),
                "etch_mode": "anisotropic",
            })
        commands.extend([
            {"action": "operation", "operation": "Strip material", "material": "PR_SU-8 Photoresist"},
            {"action": "operation", "operation": "Flip", "material": ""},
        ])
        return _window_from_commands(commands)

    for removed_fill_um in (0.0, 10.0):
        win = build(removed_fill_um)
        state = win.history[-1]
        metadata = state.get("planarizer_roles", {}) or {}
        role_grid = np.asarray(state.get("planarizer_role_grid"), dtype=np.int32)
        role_id = max(
            (abs(int(value)) for value in np.unique(role_grid) if int(value) != 0),
            default=0,
        )
        role = metadata.get(role_id, metadata.get(str(role_id), {}))
        if int((role or {}).get("orientation", 0)) != 1:
            raise AssertionError(f"Flip-back did not restore planarizer orientation: {role}")

        remaining_fill_um = 20.0 - removed_fill_um
        for factor in (FEATURE_EMPHASIS_MIN, FEATURE_EMPHASIS_DEFAULT, FEATURE_EMPHASIS_MAX):
            _state, profile, info = _render_last_state_segments(win, factor)
            columns = info.get("column_segments", [])
            height = int(info.get("display_height", 0))
            outer_planes = []
            for col in (0, 150, 299):
                positive = [
                    segment for segment in columns[col]
                    if int(segment.get("planarizer_role", 0)) == int(role_id)
                ]
                if len(positive) != 1:
                    raise AssertionError(f"Missing intact membrane overburden at {col}: {positive}")
                outer_planes.append(
                    height - int(positive[0].get("display_start", height))
                )
            if len(set(outer_planes)) != 1:
                raise AssertionError(
                    f"Support etch bent the intact membrane plane at {factor:g}: {outer_planes}"
                )

            fill_rows = sum(
                int(segment.get("display_end", 0)) - int(segment.get("display_start", 0))
                for segment in columns[150]
                if int(segment.get("planarizer_role", 0)) == -int(role_id)
            )
            fill_key = win._display_feature_profile_key(remaining_fill_um)
            expected_fill = int(profile["resolved_feature_rows"][fill_key])
            if fill_rows != expected_fill:
                raise AssertionError(
                    f"Support void inflated the unchanged {remaining_fill_um:g}um fill at {factor:g}: "
                    f"displayed={fill_rows}, expected={expected_fill}"
                )
            support_void_rows = sum(
                int(segment.get("display_end", 0)) - int(segment.get("display_start", 0))
                for segment in columns[150]
                if int(segment.get("material", win.VOID_INDEX)) == int(win.VOID_INDEX)
            )
            if support_void_rows <= fill_rows:
                raise AssertionError(
                    f"Physical support void must display larger than buried fill at {factor:g}: "
                    f"void/fill={support_void_rows}/{fill_rows}"
                )
            if removed_fill_um > 0.0:
                removed_key = win._display_feature_profile_key(removed_fill_um)
                contexts = profile.get("planarizer_partition_contexts", [])
                if not any(
                    context.get("total_removed_key") == removed_key
                    for context in contexts
                ):
                    raise AssertionError(
                        f"Backside {removed_fill_um:g}um fill removal has no explicit feature context"
                    )


def test_bond_is_rigid_structural_body_with_clearance():
    commands = _base_substrate_commands() + [
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "add_material", "material": "Aluminum Nitride (AlN)"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 1.0},
        {
            "action": "operation",
            "operation": "Lithography",
            "material": "PR_SU-8 Photoresist",
            "windows": [(0.40, 0.60)],
            "invert_lithography": False,
        },
        {"action": "operation", "operation": "Etch", "material": BASE_SUBSTRATE_MATERIAL, "depth_um": 20.0, "etch_mode": "anisotropic"},
        {"action": "operation", "operation": "Deposit", "material": "Aluminum Nitride (AlN)", "thickness_um": 0.1},
        {"action": "operation", "operation": "Bond", "material": BASE_SUBSTRATE_MATERIAL, "thickness_um": 100.0},
    ]
    win, path = _run_commands("FabStack_dimtest_rigid_bond_clearance", commands)
    pre = _state_by_operation(win.history, "Deposit", occurrence=1)
    bonded = _state_by_operation(win.history, "Bond", occurrence=1)
    bonded_index = next(idx for idx, item in enumerate(win.history) if item is bonded)

    pre_roles = np.asarray(pre.get("structural_role_grid"), dtype=np.int16)
    pre_heights = np.asarray(pre.get("row_heights_um"), dtype=float)
    roles = np.asarray(bonded.get("structural_role_grid"), dtype=np.int16)
    heights = np.asarray(bonded.get("row_heights_um"), dtype=float)
    if set(int(value) for value in np.unique(roles)) != {0, 1, 2}:
        raise AssertionError(f"Bond should retain process/original/bonded identities, got roles {np.unique(roles)}")
    for col in (0, 150, 299):
        original_before = float(np.sum(pre_heights[pre_roles[:, col] == 1]))
        original_after = float(np.sum(heights[roles[:, col] == 1]))
        bonded_body = float(np.sum(heights[roles[:, col] == 2]))
        if abs(original_after - original_before) > TOL_UM:
            raise AssertionError(
                f"Bond moved or resized the original substrate in column {col}: {original_before:g} -> {original_after:g} um"
            )
        if abs(bonded_body - 100.0) > TOL_UM:
            raise AssertionError(f"Bonded structural body must remain exactly 100 um, got {bonded_body:g} in column {col}")

    profile = win._process_display_adjustment_profile()

    def display_segments(state, history_index, active_profile=None):
        win._display_grid_auto(
            np.asarray(state.get("grid")),
            row_heights_um=np.asarray(state.get("row_heights_um"), dtype=float),
            history_index=history_index,
            visual_geometry=state.get("visual_geometry", []),
            process_profile=active_profile or profile,
        )
        return (getattr(win, "_last_display_remap_info", {}) or {}).get("column_segments", [])

    pre_index = next(idx for idx, item in enumerate(win.history) if item is pre)
    pre_columns = display_segments(pre, pre_index)
    bonded_columns = display_segments(bonded, bonded_index)
    cap_spans = []
    for col, segments in enumerate(bonded_columns):
        cap = [seg for seg in segments if int(seg.get("structural_body_id", 0)) == 2]
        if len(cap) != 1:
            raise AssertionError(f"Bonded wafer must be one separate structural segment in column {col}: {cap}")
        cap_spans.append((int(cap[0]["display_start"]), int(cap[0]["display_end"])))
    if len(set(cap_spans)) != 1:
        raise AssertionError(f"Rigid bond top and bottom must stay aligned, got {sorted(set(cap_spans))}")
    if cap_spans[0][1] - cap_spans[0][0] < int(MIN_LAYER_PX):
        raise AssertionError(f"Bonded wafer is not visibly rendered: {cap_spans[0]}")

    center = bonded_columns[150]
    cap_index = next(idx for idx, seg in enumerate(center) if int(seg.get("structural_body_id", 0)) == 2)
    clearance = next(
        (seg for seg in center[cap_index + 1:] if int(seg.get("material", win.VOID_INDEX)) == win.VOID_INDEX),
        None,
    )
    if clearance is None or int(clearance["display_end"]) <= int(clearance["display_start"]):
        raise AssertionError("The trench must remain as visible clearance below the rigid bonded wafer")

    aln_index = win.material_names.index("Aluminum Nitride (AlN)")
    for col in (0, 150):
        pre_aln = [seg for seg in pre_columns[col] if int(seg.get("material", -1)) == aln_index]
        post_aln = [seg for seg in bonded_columns[col] if int(seg.get("material", -1)) == aln_index]
        pre_spans = [(int(seg["display_start"]), int(seg["display_end"])) for seg in pre_aln]
        post_spans = [(int(seg["display_start"]), int(seg["display_end"])) for seg in post_aln]
        if pre_spans != post_spans:
            raise AssertionError(
                f"Bond must not lift the existing AlN in column {col}: before={pre_spans}, after={post_spans}"
            )

    factor_cap_heights = []
    for factor in (FEATURE_EMPHASIS_MIN, FEATURE_EMPHASIS_DEFAULT, FEATURE_EMPHASIS_MAX):
        win.feature_scale_factor = factor
        factor_profile = win._process_display_adjustment_profile()
        factor_pre = display_segments(pre, pre_index, factor_profile)
        factor_bonded = display_segments(bonded, bonded_index, factor_profile)
        factor_spans = []
        for segments in factor_bonded:
            cap = [seg for seg in segments if int(seg.get("structural_body_id", 0)) == 2]
            if len(cap) != 1:
                raise AssertionError(f"Emphasis {factor:g} lost the bonded structural body: {cap}")
            factor_spans.append((int(cap[0]["display_start"]), int(cap[0]["display_end"])))
        if len(set(factor_spans)) != 1:
            raise AssertionError(f"Emphasis {factor:g} bent the rigid bonded substrate: {factor_spans}")
        factor_cap_heights.append(factor_spans[0][1] - factor_spans[0][0])
        aln_index = win.material_names.index("Aluminum Nitride (AlN)")
        for col in (0, 150):
            pre_aln = [seg for seg in factor_pre[col] if int(seg.get("material", -1)) == aln_index]
            post_aln = [seg for seg in factor_bonded[col] if int(seg.get("material", -1)) == aln_index]
            pre_spans = [(int(seg["display_start"]), int(seg["display_end"])) for seg in pre_aln]
            post_spans = [(int(seg["display_start"]), int(seg["display_end"])) for seg in post_aln]
            if pre_spans != post_spans:
                raise AssertionError(
                    f"Emphasis {factor:g} made bonding lift the existing layer in column {col}: "
                    f"before={pre_spans}, after={post_spans}"
                )
    expected_cap_height = win._display_structural_body_rows(100.0)
    if factor_cap_heights != [expected_cap_height] * 3:
        raise AssertionError(
            f"Bonded substrate thickness must be independent of feature emphasis, got {factor_cap_heights}"
        )

    data = win._make_process_dict()
    _app, loaded = _new_window()
    loaded._load_process_dict(data)
    loaded_roles = np.asarray(loaded.xsec.copy_structural_roles(), dtype=np.int16)
    if not {1, 2}.issubset(set(int(value) for value in np.unique(loaded_roles))):
        raise AssertionError("Save/load lost original or bonded substrate provenance")
    ok, message = loaded._execute_assistant_command({"action": "operation", "operation": "Flip", "material": ""})
    if not ok:
        raise AssertionError(f"Flip after bond failed: {message}")
    flipped_roles = np.asarray(loaded.xsec.copy_structural_roles(), dtype=np.int16)
    if not {1, 2}.issubset(set(int(value) for value in np.unique(flipped_roles))):
        raise AssertionError("Flip lost original or bonded substrate provenance")
    return path


def test_rigid_bond_stays_flat_over_multi_etch_film_topography():
    """Regression for the user's two-etch, multi-film bonding workflow."""
    commands = [
        {"action": "set_substrate", "material": "Silicon (Si)", "thickness_um": 525.0},
        {"action": "add_material", "material": "PR_Epoxy-based Photoresist"},
        {"action": "add_material", "material": "Aluminum Nitride (AlN)"},
        {"action": "add_material", "material": "Silicon Carbide (SiC)"},
        {"action": "add_material", "material": "Titanium Nitride (TiN)"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_Epoxy-based Photoresist", "thickness_um": 1.0},
        {"action": "operation", "operation": "Lithography", "material": "PR_Epoxy-based Photoresist", "windows": [(0.35, 0.65)], "invert_lithography": False},
        {"action": "operation", "operation": "Etch", "material": "Silicon (Si)", "depth_um": 200.0, "etch_mode": "anisotropic"},
        {"action": "operation", "operation": "Deposit", "material": "Aluminum Nitride (AlN)", "thickness_um": 0.1},
        {"action": "operation", "operation": "Deposit", "material": "Silicon Carbide (SiC)", "thickness_um": 0.01},
        {"action": "operation", "operation": "Deposit", "material": "Titanium Nitride (TiN)", "thickness_um": 0.001},
        {"action": "operation", "operation": "Lift-off", "material": "PR_Epoxy-based Photoresist"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_Epoxy-based Photoresist", "thickness_um": 1.0},
        {
            "action": "operation",
            "operation": "Lithography",
            "material": "PR_Epoxy-based Photoresist",
            "windows": [
                (0.08930633072161996, 0.19765442202425673),
                (0.7728842885764373, 0.8812323798790739),
            ],
            "invert_lithography": False,
        },
        {"action": "operation", "operation": "Etch", "material": "Silicon (Si)", "depth_um": 20.0, "etch_mode": "anisotropic"},
        {"action": "operation", "operation": "Deposit", "material": "Aluminum Nitride (AlN)", "thickness_um": 10.0},
        {"action": "operation", "operation": "Lift-off", "material": "PR_Epoxy-based Photoresist"},
        {"action": "operation", "operation": "Bond", "material": "Silicon (Si)", "thickness_um": 100.0},
    ]
    win = _window_from_commands(commands)
    bonded_index = len(win.history) - 1
    bonded = win.history[bonded_index]
    pre_index = bonded_index - 1
    pre = win.history[pre_index]

    def visible_nonbonded_solids(info, col):
        return [
            (
                int(segment.get("material", win.VOID_INDEX)),
                int(segment.get("structural_body_id", 0)),
                int(segment.get("planarizer_role", 0)),
                int(segment.get("display_start", 0)),
                int(segment.get("display_end", 0)),
            )
            for segment in info.get("column_segments", [])[col]
            if int(segment.get("material", win.VOID_INDEX)) != int(win.VOID_INDEX)
            and int(segment.get("structural_body_id", 0)) != 2
        ]

    for factor in (FEATURE_EMPHASIS_MIN, FEATURE_EMPHASIS_DEFAULT, FEATURE_EMPHASIS_MAX):
        win.feature_scale_factor = factor
        profile = win._process_display_adjustment_profile()
        if profile.get("constraint_violations"):
            raise AssertionError(
                f"Multi-etch Bond profile has constraint violations at {factor:g}: "
                f"{profile.get('constraint_violations')}"
            )
        if not profile.get("bond_attachment_equations"):
            raise AssertionError("Multi-etch Bond produced no attachment equations")
        win._display_grid_auto(
            np.asarray(pre.get("grid")),
            row_heights_um=np.asarray(pre.get("row_heights_um"), dtype=float),
            history_index=pre_index,
            visual_geometry=pre.get("visual_geometry", []),
            process_profile=profile,
        )
        pre_info = getattr(win, "_last_display_remap_info", {}) or {}
        win._display_grid_auto(
            np.asarray(bonded.get("grid")),
            row_heights_um=np.asarray(bonded.get("row_heights_um"), dtype=float),
            history_index=bonded_index,
            visual_geometry=bonded.get("visual_geometry", []),
            process_profile=profile,
        )
        bonded_info = getattr(win, "_last_display_remap_info", {}) or {}

        body_spans = []
        for col, segments in enumerate(bonded_info.get("column_segments", [])):
            body = [
                segment for segment in segments
                if int(segment.get("structural_body_id", 0)) == 2
            ]
            if len(body) != 1:
                raise AssertionError(
                    f"Bonded wafer must remain one slab segment in column {col}: {body}"
                )
            body_spans.append((
                int(body[0].get("display_start", 0)),
                int(body[0].get("display_end", 0)),
            ))
        if len(set(body_spans)) != 1:
            raise AssertionError(
                f"Bonded slab stepped over the multi-etch topography at {factor:g}: "
                f"{sorted(set(body_spans))}"
            )

        for col in range(len(body_spans)):
            if visible_nonbonded_solids(pre_info, col) != visible_nonbonded_solids(
                bonded_info,
                col,
            ):
                raise AssertionError(
                    f"Bond moved an existing solid in column {col} at {factor:g}"
                )

        # The corrected flat slab must not erase the real cavity beneath it.
        cavity_col = 150
        cavity_segments = bonded_info["column_segments"][cavity_col]
        body_index = next(
            idx for idx, segment in enumerate(cavity_segments)
            if int(segment.get("structural_body_id", 0)) == 2
        )
        clearance = cavity_segments[body_index + 1]
        film_stack_rows = int(sum(
            int(segment.get("display_end", 0))
            - int(segment.get("display_start", 0))
            for segment in cavity_segments[body_index + 2:]
            if int(segment.get("material", win.VOID_INDEX)) != int(win.VOID_INDEX)
            and int(segment.get("structural_body_id", 0)) == 0
        ))
        clearance_rows = int(
            int(clearance.get("display_end", 0))
            - int(clearance.get("display_start", 0))
        )
        if (
            int(clearance.get("material", win.VOID_INDEX)) != int(win.VOID_INDEX)
            or clearance_rows < int(MIN_VOID_PX)
        ):
            raise AssertionError(
                f"Flat bond erased its visible etched clearance at {factor:g}: {clearance}"
            )
        if clearance_rows < film_stack_rows + int(ORDER_MARGIN_PX):
            raise AssertionError(
                f"199.889um bond clearance must visibly exceed the 0.111um film stack "
                f"at {factor:g}: gap={clearance_rows}, stack={film_stack_rows}"
            )


def test_bond_equal_contact_composites_are_resolved_process_wide():
    """H(10um) and H(5um)+H(5um) contacts must share one bond plane."""
    _app, win = _new_window()
    for material in (
        "Aluminum Nitride (AlN)",
        "Silicon Carbide (SiC)",
        "Titanium Nitride (TiN)",
    ):
        ok, message = win._execute_assistant_command({
            "action": "add_material",
            "material": material,
        })
        if not ok:
            raise AssertionError(f"Could not prepare composite contact: {message}")

    silicon = win.material_names.index("Silicon (Si)")
    aln = win.material_names.index("Aluminum Nitride (AlN)")
    sic = win.material_names.index("Silicon Carbide (SiC)")
    tin = win.material_names.index("Titanium Nitride (TiN)")
    void = int(win.VOID_INDEX)
    row_heights = np.asarray([100.0, 5.0, 5.0, 500.0], dtype=float)
    pre_grid = np.asarray([
        [void, void],
        [aln, sic],
        [aln, tin],
        [silicon, silicon],
    ], dtype=np.int16)
    bonded_grid = pre_grid.copy()
    bonded_grid[0, :] = silicon
    pre_roles = np.asarray([
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 1],
    ], dtype=np.int16)
    bonded_roles = pre_roles.copy()
    bonded_roles[0, :] = 2
    bodies = {
        1: {
            "body_id": 1,
            "material_index": silicon,
            "nominal_thickness_um": 500.0,
            "kind": "substrate",
        },
        2: {
            "body_id": 2,
            "material_index": silicon,
            "nominal_thickness_um": 100.0,
            "kind": "bonded_substrate",
        },
    }

    def state(grid, roles, *, operation=None):
        return {
            "title": operation or "Equal-height contact pads",
            "operation": operation,
            "material_name": "Silicon (Si)" if operation == "Bond" else None,
            "grid": grid.copy(),
            "row_heights_um": row_heights.copy(),
            "structural_role_grid": roles.copy(),
            "structural_bodies": {
                body_id: dict(metadata)
                for body_id, metadata in bodies.items()
                if body_id == 1 or operation == "Bond"
            },
            "planarizer_role_grid": np.zeros_like(grid, dtype=np.int32),
            "planarizer_roles": {},
            "visual_geometry": [],
        }

    pre = state(pre_grid, pre_roles)
    bonded = state(bonded_grid, bonded_roles, operation="Bond")
    win.history = [pre, bonded]

    def solids(info, col, *, exclude_body2=False):
        return [
            (
                int(segment.get("material", void)),
                int(segment.get("structural_body_id", 0)),
                int(segment.get("display_start", 0)),
                int(segment.get("display_end", 0)),
            )
            for segment in info.get("column_segments", [])[col]
            if int(segment.get("material", void)) != void
            and not (
                exclude_body2
                and int(segment.get("structural_body_id", 0)) == 2
            )
        ]

    for factor in (FEATURE_EMPHASIS_MIN, FEATURE_EMPHASIS_DEFAULT, FEATURE_EMPHASIS_MAX):
        win.feature_scale_factor = factor
        profile = win._process_display_adjustment_profile()
        if profile.get("constraint_violations"):
            raise AssertionError(
                f"Composite contact constraints failed at {factor:g}: {profile}"
            )
        equations = profile.get("bond_attachment_equations", [])
        if not equations:
            raise AssertionError("Equal contact stacks produced no attachment equation")
        five_key = win._display_feature_profile_key(5.0)
        ten_key = win._display_feature_profile_key(10.0)
        resolved = profile.get("resolved_feature_rows", {})
        if int(resolved[ten_key]) != 2 * int(resolved[five_key]):
            raise AssertionError(
                f"H(10) must equal H(5)+H(5) at {factor:g}: "
                f"{resolved[ten_key]} vs {resolved[five_key]}+{resolved[five_key]}"
            )

        infos = []
        for history_index, item in enumerate((pre, bonded)):
            win._display_grid_auto(
                np.asarray(item["grid"]),
                row_heights_um=np.asarray(item["row_heights_um"], dtype=float),
                history_index=history_index,
                process_profile=profile,
            )
            infos.append(getattr(win, "_last_display_remap_info", {}) or {})
        pre_info, bonded_info = infos
        body_spans = []
        for col in (0, 1):
            body = [
                segment for segment in bonded_info["column_segments"][col]
                if int(segment.get("structural_body_id", 0)) == 2
            ]
            if len(body) != 1:
                raise AssertionError(f"Composite contact lost bonded body: {body}")
            body_spans.append((body[0]["display_start"], body[0]["display_end"]))
            if solids(pre_info, col) != solids(
                bonded_info,
                col,
                exclude_body2=True,
            ):
                raise AssertionError(
                    f"Bond moved an equal-height contact stack in column {col} at {factor:g}"
                )
        if len(set(body_spans)) != 1:
            raise AssertionError(
                f"Equal physical contact stacks bent the slab at {factor:g}: {body_spans}"
            )


def test_bond_stays_flat_over_complete_substrate_through_hole():
    commands = [
        {"action": "set_substrate", "material": "Silicon (Si)", "thickness_um": 500.0},
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 1.0},
        {"action": "operation", "operation": "Lithography", "material": "PR_SU-8 Photoresist", "windows": [(0.40, 0.60)], "invert_lithography": False},
        {"action": "operation", "operation": "Etch", "material": "Silicon (Si)", "depth_um": 500.0, "etch_mode": "anisotropic"},
        {"action": "operation", "operation": "Strip material", "material": "PR_SU-8 Photoresist"},
        {"action": "operation", "operation": "Bond", "material": "Silicon (Si)", "thickness_um": 100.0},
    ]
    win = _window_from_commands(commands)
    bonded_index = len(win.history) - 1
    bonded = win.history[bonded_index]
    pre_index = bonded_index - 1
    pre = win.history[pre_index]
    roles = np.asarray(bonded.get("structural_role_grid"), dtype=np.int16)
    heights = np.asarray(bonded.get("row_heights_um"), dtype=float)
    if float(np.sum(heights[roles[:, 150] == 1])) > TOL_UM:
        raise AssertionError("The centre column is not a physical through-hole")

    for factor in (FEATURE_EMPHASIS_MIN, FEATURE_EMPHASIS_DEFAULT, FEATURE_EMPHASIS_MAX):
        win.feature_scale_factor = factor
        profile = win._process_display_adjustment_profile()
        if profile.get("constraint_violations"):
            raise AssertionError(
                f"Through-hole Bond has constraint violations at {factor:g}: "
                f"{profile.get('constraint_violations')}"
            )
        full_key = win._display_feature_profile_key(500.0)
        full_rows = int(profile["resolved_feature_rows"][full_key])
        body_rows = int(profile["structural_body_rows"][1])
        if full_rows != body_rows:
            raise AssertionError(
                f"A full removal must equal its nominal body budget at {factor:g}: "
                f"gap={full_rows}, body={body_rows}"
            )

        infos = []
        for history_index, state in ((pre_index, pre), (bonded_index, bonded)):
            win._display_grid_auto(
                np.asarray(state.get("grid")),
                row_heights_um=np.asarray(state.get("row_heights_um"), dtype=float),
                history_index=history_index,
                visual_geometry=state.get("visual_geometry", []),
                process_profile=profile,
            )
            infos.append(getattr(win, "_last_display_remap_info", {}) or {})
        pre_info, bonded_info = infos

        body_spans = []
        for col, segments in enumerate(bonded_info["column_segments"]):
            body = [
                segment for segment in segments
                if int(segment.get("structural_body_id", 0)) == 2
            ]
            if len(body) != 1:
                raise AssertionError(f"Through-hole Bond lost body2 in column {col}: {body}")
            body_spans.append((body[0]["display_start"], body[0]["display_end"]))
        if len(set(body_spans)) != 1:
            raise AssertionError(
                f"Bond bent over a complete through-hole at {factor:g}: "
                f"{sorted(set(body_spans))}"
            )

        center = bonded_info["column_segments"][150]
        body_index = next(
            idx for idx, segment in enumerate(center)
            if int(segment.get("structural_body_id", 0)) == 2
        )
        through_void = center[body_index + 1]
        through_rows = int(
            int(through_void.get("display_end", 0))
            - int(through_void.get("display_start", 0))
        )
        if (
            int(through_void.get("material", win.VOID_INDEX)) != int(win.VOID_INDEX)
            or through_rows != body_rows
        ):
            raise AssertionError(
                f"Bond flattened or resized the real through-hole at {factor:g}: "
                f"void={through_void}, expected rows={body_rows}"
            )

        for col in (0, 150, 299):
            pre_solids = [
                (s["material"], s.get("structural_body_id"), s["display_start"], s["display_end"])
                for s in pre_info["column_segments"][col]
                if int(s.get("material", win.VOID_INDEX)) != int(win.VOID_INDEX)
            ]
            post_solids = [
                (s["material"], s.get("structural_body_id"), s["display_start"], s["display_end"])
                for s in bonded_info["column_segments"][col]
                if int(s.get("material", win.VOID_INDEX)) != int(win.VOID_INDEX)
                and int(s.get("structural_body_id", 0)) != 2
            ]
            if pre_solids != post_solids:
                raise AssertionError(
                    f"Bond moved an existing through-hole support in column {col} at {factor:g}"
                )

    data = win._make_process_dict()
    _app, loaded = _new_window()
    loaded._load_process_dict(data)
    ok, message = loaded._execute_assistant_command({
        "action": "operation",
        "operation": "Flip",
        "material": "",
    })
    if not ok:
        raise AssertionError(f"Could not flip saved through-hole Bond: {message}")
    flipped = loaded.history[-1]
    profile = loaded._process_display_adjustment_profile()
    loaded._display_grid_auto(
        np.asarray(flipped.get("grid")),
        row_heights_um=np.asarray(flipped.get("row_heights_um"), dtype=float),
        history_index=len(loaded.history) - 1,
        process_profile=profile,
    )
    info = getattr(loaded, "_last_display_remap_info", {}) or {}
    flipped_spans = []
    for segments in info.get("column_segments", []):
        body = [
            segment for segment in segments
            if int(segment.get("structural_body_id", 0)) == 2
        ]
        if len(body) != 1:
            raise AssertionError(f"Flip/save-load lost the bonded slab: {body}")
        flipped_spans.append((body[0]["display_start"], body[0]["display_end"]))
    if len(set(flipped_spans)) != 1:
        raise AssertionError(
            f"Flip/save-load bent the slab over the through-hole: {sorted(set(flipped_spans))}"
        )


def test_full_removal_tracks_final_ordered_body_budget():
    """A later etch of body2 must not bend its earlier Bond over body1's hole."""
    commands = [
        {"action": "set_substrate", "material": "Silicon (Si)", "thickness_um": 500.0},
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 1.0},
        {"action": "operation", "operation": "Lithography", "material": "PR_SU-8 Photoresist", "windows": [(0.40, 0.60)], "invert_lithography": False},
        {"action": "operation", "operation": "Etch", "material": "Silicon (Si)", "depth_um": 500.0, "etch_mode": "anisotropic"},
        {"action": "operation", "operation": "Strip material", "material": "PR_SU-8 Photoresist"},
        {"action": "operation", "operation": "Bond", "material": "Silicon (Si)", "thickness_um": 100.0},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 1.0},
        {"action": "operation", "operation": "Lithography", "material": "PR_SU-8 Photoresist", "windows": [(0.10, 0.25)], "invert_lithography": False},
        {"action": "operation", "operation": "Etch", "material": "Silicon (Si)", "depth_um": 49.0, "etch_mode": "anisotropic"},
    ]
    win = _window_from_commands(commands)
    bond_index = next(
        idx for idx, state in enumerate(win.history)
        if state.get("operation") == "Bond"
    )
    final_index = len(win.history) - 1
    final_state = win.history[final_index]
    roles = np.asarray(final_state.get("structural_role_grid"), dtype=np.int16)
    heights = np.asarray(final_state.get("row_heights_um"), dtype=float)
    body2_physical = [
        float(np.sum(heights[roles[:, col] == 2]))
        for col in (0, 50)
    ]
    if (
        abs(body2_physical[0] - 100.0) > TOL_UM
        or abs(body2_physical[1] - 51.0) > TOL_UM
    ):
        raise AssertionError(
            f"The later 49um etch must remove only body2 in its window: {body2_physical}"
        )

    for factor in (FEATURE_EMPHASIS_MIN, FEATURE_EMPHASIS_DEFAULT, FEATURE_EMPHASIS_MAX):
        win.feature_scale_factor = factor
        profile = win._process_display_adjustment_profile()
        if not profile.get("constraint_converged") or profile.get("constraint_violations"):
            raise AssertionError(
                f"Descendant body-budget constraints failed at {factor:g}: "
                f"{profile.get('constraint_violations')}"
            )
        full_key = win._display_feature_profile_key(500.0)
        full_rows = int(profile["resolved_feature_rows"][full_key])
        body1_rows = int(profile["structural_body_rows"][1])
        if full_rows != body1_rows:
            raise AssertionError(
                f"Body1's through-removal must follow its final ordered budget at {factor:g}: "
                f"gap={full_rows}, body={body1_rows}"
            )

        infos = []
        for history_index in (bond_index, final_index):
            state = win.history[history_index]
            win._display_grid_auto(
                np.asarray(state.get("grid")),
                row_heights_um=np.asarray(state.get("row_heights_um"), dtype=float),
                history_index=history_index,
                visual_geometry=state.get("visual_geometry", []),
                process_profile=profile,
            )
            infos.append(getattr(win, "_last_display_remap_info", {}) or {})
        bond_info, final_info = infos

        bond_spans = []
        for col, segments in enumerate(bond_info.get("column_segments", [])):
            body = [
                segment for segment in segments
                if int(segment.get("structural_body_id", 0)) == 2
            ]
            if len(body) != 1:
                raise AssertionError(
                    f"Earlier Bond lost its rigid slab in column {col} at {factor:g}: {body}"
                )
            bond_spans.append((
                int(body[0].get("display_start", 0)),
                int(body[0].get("display_end", 0)),
            ))
        if len(set(bond_spans)) != 1:
            raise AssertionError(
                f"A later body2 etch bent the earlier Bond at {factor:g}: "
                f"{sorted(set(bond_spans))}"
            )

        final_spans = {}
        for col in (0, 50, 150, 299):
            body = [
                segment for segment in final_info["column_segments"][col]
                if int(segment.get("structural_body_id", 0)) == 2
            ]
            if len(body) != 1:
                raise AssertionError(
                    f"Later etch fragmented body2 in column {col} at {factor:g}: {body}"
                )
            final_spans[col] = (
                int(body[0].get("display_start", 0)),
                int(body[0].get("display_end", 0)),
            )
        if len({span[1] for span in final_spans.values()}) != 1:
            raise AssertionError(
                f"The later etch moved body2's bonded interface at {factor:g}: {final_spans}"
            )
        displayed_etch = final_spans[50][0] - final_spans[0][0]
        expected_etch = int(
            profile["resolved_feature_rows"][win._display_feature_profile_key(49.0)]
        )
        if displayed_etch != expected_etch:
            raise AssertionError(
                f"The later body2 etch must remain visibly scaled at {factor:g}: "
                f"expected={expected_etch}, got={displayed_etch}"
            )


def test_equal_nominal_bodies_share_final_display_budget():
    """Identical substrate thicknesses stay identical after a later local etch."""
    commands = [
        {"action": "set_substrate", "material": "Silicon (Si)", "thickness_um": 500.0},
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 1.0},
        {"action": "operation", "operation": "Lithography", "material": "PR_SU-8 Photoresist", "windows": [(0.40, 0.60)], "invert_lithography": False},
        {"action": "operation", "operation": "Etch", "material": "Silicon (Si)", "depth_um": 500.0, "etch_mode": "anisotropic"},
        {"action": "operation", "operation": "Strip material", "material": "PR_SU-8 Photoresist"},
        {"action": "operation", "operation": "Bond", "material": "Silicon (Si)", "thickness_um": 500.0},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 1.0},
        {"action": "operation", "operation": "Lithography", "material": "PR_SU-8 Photoresist", "windows": [(0.10, 0.25)], "invert_lithography": False},
        {"action": "operation", "operation": "Etch", "material": "Silicon (Si)", "depth_um": 249.0, "etch_mode": "anisotropic"},
    ]
    win = _window_from_commands(commands)
    bond_index = next(
        idx for idx, state in enumerate(win.history)
        if state.get("operation") == "Bond"
    )
    final_index = len(win.history) - 1

    for factor in (FEATURE_EMPHASIS_MIN, FEATURE_EMPHASIS_DEFAULT, FEATURE_EMPHASIS_MAX):
        win.feature_scale_factor = factor
        profile = win._process_display_adjustment_profile()
        if not profile.get("constraint_converged") or profile.get("constraint_violations"):
            raise AssertionError(
                f"Equal-body constraints failed at {factor:g}: "
                f"{profile.get('constraint_violations')}"
            )
        body1_rows = int(profile["structural_body_rows"][1])
        body2_rows = int(profile["structural_body_rows"][2])
        full_rows = int(
            profile["resolved_feature_rows"][win._display_feature_profile_key(500.0)]
        )
        if body1_rows != body2_rows or full_rows != body1_rows:
            raise AssertionError(
                f"Equal 500um substrates must share the through-hole budget at {factor:g}: "
                f"body1={body1_rows}, body2={body2_rows}, gap={full_rows}"
            )

        rendered = []
        for history_index in (bond_index, final_index):
            state = win.history[history_index]
            win._display_grid_auto(
                np.asarray(state.get("grid")),
                row_heights_um=np.asarray(state.get("row_heights_um"), dtype=float),
                history_index=history_index,
                visual_geometry=state.get("visual_geometry", []),
                process_profile=profile,
            )
            rendered.append(getattr(win, "_last_display_remap_info", {}) or {})
        bond_info, final_info = rendered

        bond_body2_spans = []
        for segments in bond_info["column_segments"]:
            body2 = [
                segment for segment in segments
                if int(segment.get("structural_body_id", 0)) == 2
            ]
            if len(body2) != 1:
                raise AssertionError(f"Equal-body Bond lost body2 at {factor:g}: {body2}")
            bond_body2_spans.append((
                int(body2[0]["display_start"]),
                int(body2[0]["display_end"]),
            ))
        if len(set(bond_body2_spans)) != 1:
            raise AssertionError(
                f"Equal-body Bond bent over the through-hole at {factor:g}: "
                f"{sorted(set(bond_body2_spans))}"
            )

        intact_segments = bond_info["column_segments"][0]
        intact_rows = {}
        for body_id in (1, 2):
            body = [
                segment for segment in intact_segments
                if int(segment.get("structural_body_id", 0)) == body_id
            ]
            if len(body) != 1:
                raise AssertionError(
                    f"Missing intact equal body {body_id} at {factor:g}: {body}"
                )
            intact_rows[body_id] = int(body[0]["display_end"]) - int(body[0]["display_start"])
        if intact_rows[1] != intact_rows[2]:
            raise AssertionError(
                f"Equal 500um intact bodies render differently at {factor:g}: {intact_rows}"
            )

        final_spans = {}
        for col in (0, 50, 150, 299):
            body2 = [
                segment for segment in final_info["column_segments"][col]
                if int(segment.get("structural_body_id", 0)) == 2
            ]
            if len(body2) != 1:
                raise AssertionError(
                    f"Later etch lost equal-nominal body2 at {factor:g}/{col}: {body2}"
                )
            final_spans[col] = (
                int(body2[0]["display_start"]),
                int(body2[0]["display_end"]),
            )
        if len({span[1] for span in final_spans.values()}) != 1:
            raise AssertionError(
                f"Later etch moved the equal body's bonded face at {factor:g}: {final_spans}"
            )
        displayed_etch = final_spans[50][0] - final_spans[0][0]
        expected_etch = int(
            profile["resolved_feature_rows"][win._display_feature_profile_key(249.0)]
        )
        if displayed_etch != expected_etch:
            raise AssertionError(
                f"Equal-body 249um etch lost its visible scale at {factor:g}: "
                f"expected={expected_etch}, got={displayed_etch}"
            )


def test_body_ordering_enters_equal_nominal_group():
    """Growth from a smaller body must reach every equal larger substrate."""
    commands = [
        {"action": "set_substrate", "material": "Silicon (Si)", "thickness_um": 500.0},
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 1.0},
        {"action": "operation", "operation": "Lithography", "material": "PR_SU-8 Photoresist", "windows": [(0.40, 0.60)], "invert_lithography": False},
        {"action": "operation", "operation": "Etch", "material": "Silicon (Si)", "depth_um": 500.0, "etch_mode": "anisotropic"},
        {"action": "operation", "operation": "Strip material", "material": "PR_SU-8 Photoresist"},
        {"action": "operation", "operation": "Bond", "material": "Silicon (Si)", "thickness_um": 500.0},
        {"action": "operation", "operation": "Bond", "material": "Silicon (Si)", "thickness_um": 100.0},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 1.0},
        {"action": "operation", "operation": "Lithography", "material": "PR_SU-8 Photoresist", "windows": [(0.10, 0.25)], "invert_lithography": False},
        {"action": "operation", "operation": "Etch", "material": "Silicon (Si)", "depth_um": 49.0, "etch_mode": "anisotropic"},
    ]
    win = _window_from_commands(commands)
    bond_indices = [
        idx for idx, state in enumerate(win.history)
        if state.get("operation") == "Bond"
    ]
    if len(bond_indices) != 2:
        raise AssertionError(f"Expected two Bond states, got {bond_indices}")
    final_index = len(win.history) - 1
    final_state = win.history[final_index]
    roles = np.asarray(final_state.get("structural_role_grid"), dtype=np.int16)
    heights = np.asarray(final_state.get("row_heights_um"), dtype=float)
    body3_physical = [
        float(np.sum(heights[roles[:, col] == 3]))
        for col in (0, 50)
    ]
    if (
        abs(body3_physical[0] - 100.0) > TOL_UM
        or abs(body3_physical[1] - 51.0) > TOL_UM
    ):
        raise AssertionError(f"Later 49um etch did not stay in body3: {body3_physical}")

    for factor in (FEATURE_EMPHASIS_MIN, FEATURE_EMPHASIS_DEFAULT, FEATURE_EMPHASIS_MAX):
        win.feature_scale_factor = factor
        profile = win._process_display_adjustment_profile()
        if not profile.get("constraint_converged") or profile.get("constraint_violations"):
            raise AssertionError(
                f"Ordered equal-body groups failed at {factor:g}: "
                f"{profile.get('constraint_violations')}"
            )
        body_rows = profile["structural_body_rows"]
        if int(body_rows[1]) != int(body_rows[2]):
            raise AssertionError(
                f"Equal 500um bodies diverged after 100um-body growth at {factor:g}: "
                f"{body_rows}"
            )
        margin = win._display_physical_order_margin(500.0, 100.0)
        if int(body_rows[1]) < int(body_rows[3]) + int(margin):
            raise AssertionError(
                f"500um group must remain larger than the grown 100um body at {factor:g}: "
                f"{body_rows}"
            )
        full_rows = int(
            profile["resolved_feature_rows"][win._display_feature_profile_key(500.0)]
        )
        if full_rows != int(body_rows[1]):
            raise AssertionError(
                f"Through-hole path did not follow the ordered 500um group at {factor:g}: "
                f"gap={full_rows}, bodies={body_rows}"
            )

        rendered = {}
        for history_index in (*bond_indices, final_index):
            state = win.history[history_index]
            win._display_grid_auto(
                np.asarray(state.get("grid")),
                row_heights_um=np.asarray(state.get("row_heights_um"), dtype=float),
                history_index=history_index,
                visual_geometry=state.get("visual_geometry", []),
                process_profile=profile,
            )
            rendered[history_index] = getattr(win, "_last_display_remap_info", {}) or {}

        for history_index, body_id in zip(bond_indices, (2, 3)):
            spans = []
            for segments in rendered[history_index]["column_segments"]:
                body = [
                    segment for segment in segments
                    if int(segment.get("structural_body_id", 0)) == body_id
                ]
                if len(body) != 1:
                    raise AssertionError(
                        f"Bond state {history_index} lost body{body_id} at {factor:g}: {body}"
                    )
                spans.append((
                    int(body[0]["display_start"]),
                    int(body[0]["display_end"]),
                ))
            if len(set(spans)) != 1:
                raise AssertionError(
                    f"Bonded body{body_id} bent after descendant processing at {factor:g}: "
                    f"{sorted(set(spans))}"
                )

        final_spans = {}
        for col in (0, 50, 150, 299):
            body3 = [
                segment for segment in rendered[final_index]["column_segments"][col]
                if int(segment.get("structural_body_id", 0)) == 3
            ]
            if len(body3) != 1:
                raise AssertionError(f"Final state lost body3 at {factor:g}/{col}: {body3}")
            final_spans[col] = (
                int(body3[0]["display_start"]),
                int(body3[0]["display_end"]),
            )
        if len({span[1] for span in final_spans.values()}) != 1:
            raise AssertionError(
                f"Body3's bonded face moved after etching at {factor:g}: {final_spans}"
            )
        displayed_etch = final_spans[50][0] - final_spans[0][0]
        expected_etch = int(
            profile["resolved_feature_rows"][win._display_feature_profile_key(49.0)]
        )
        if displayed_etch != expected_etch:
            raise AssertionError(
                f"Body3's later etch lost visibility at {factor:g}: "
                f"expected={expected_etch}, got={displayed_etch}"
            )


def test_bonded_substrate_etch_uses_body_local_core_scaling():
    _app, win = _new_window()
    commands = [
        {"action": "set_substrate", "material": "Silicon (Si)", "thickness_um": 500.0},
        {"action": "operation", "operation": "Bond", "material": "Silicon (Si)", "thickness_um": 100.0},
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 5.0},
        {
            "action": "operation",
            "operation": "Lithography",
            "material": "PR_SU-8 Photoresist",
            "windows": [(0.40, 0.60)],
            "invert_lithography": False,
        },
        {
            "action": "operation",
            "operation": "Etch",
            "material": "Silicon (Si)",
            "depth_um": 40.0,
            "etch_mode": "anisotropic",
        },
    ]
    for command in commands:
        ok, message = win._execute_assistant_command(command)
        if not ok:
            raise AssertionError(f"Bonded-substrate etch command failed: {message}")

    etched = win.history[-1]
    roles = np.asarray(etched.get("structural_role_grid"), dtype=np.int16)
    heights = np.asarray(etched.get("row_heights_um"), dtype=float)
    body2_physical = [float(np.sum(heights[roles[:, col] == 2])) for col in (0, 150)]
    if abs(body2_physical[0] - 100.0) > TOL_UM or abs(body2_physical[1] - 60.0) > TOL_UM:
        raise AssertionError(f"A 40um etch must leave 60um of the 100um bonded wafer: {body2_physical}")

    def check_state(state, factor):
        win.feature_scale_factor = factor
        profile = win._process_display_adjustment_profile()
        state_index = next(idx for idx, item in enumerate(win.history) if item is state)
        win._display_grid_auto(
            np.asarray(state.get("grid")),
            row_heights_um=np.asarray(state.get("row_heights_um"), dtype=float),
            history_index=state_index,
            visual_geometry=state.get("visual_geometry", []),
            process_profile=profile,
        )
        columns = (getattr(win, "_last_display_remap_info", {}) or {}).get("column_segments", [])

        def body_rows(col, body_id):
            return sum(
                int(segment.get("display_end", 0)) - int(segment.get("display_start", 0))
                for segment in columns[col]
                if int(segment.get("structural_body_id", 0)) == int(body_id)
            )

        body2_outside = body_rows(0, 2)
        body2_center = body_rows(150, 2)
        displayed_gap = body2_outside - body2_center
        expected_gap = int(profile["resolved_feature_rows"][win._display_feature_profile_key(40.0)])
        if displayed_gap != expected_gap:
            raise AssertionError(
                f"Bonded wafer must use its local 40um feature scale at {factor:g}: "
                f"expected {expected_gap}, got {displayed_gap}"
            )
        if body2_center < displayed_gap + int(ORDER_MARGIN_PX):
            raise AssertionError(
                f"Bonded-wafer core 60um > etch 40um must stay in that order at {factor:g}: "
                f"core={body2_center}, gap={displayed_gap}"
            )
        body1_outside = body_rows(0, 1)
        if body1_outside < body2_outside + int(ORDER_MARGIN_PX):
            raise AssertionError(
                f"Original wafer 500um > bonded wafer 100um must stay in that order at {factor:g}: "
                f"body1={body1_outside}, body2={body2_outside}"
            )
        contexts = [
            context for context in profile.get("gap_contexts", [])
            if int(context.get("structural_body_id", 0)) == 2
        ]
        if not contexts or any(abs(float(context.get("nominal_um", 0.0)) - 100.0) > TOL_UM for context in contexts):
            raise AssertionError(f"Missing body-local bonded-substrate constraint context: {contexts}")
        return displayed_gap

    gaps = [check_state(etched, factor) for factor in (
        FEATURE_EMPHASIS_MIN,
        FEATURE_EMPHASIS_DEFAULT,
        FEATURE_EMPHASIS_MAX,
    )]
    if gaps != sorted(gaps) or len(set(gaps)) != 3:
        raise AssertionError(f"Bonded-wafer etch should respond monotonically to emphasis: {gaps}")

    win.feature_scale_factor = FEATURE_EMPHASIS_DEFAULT
    data = win._make_process_dict()
    _app, loaded = _new_window()
    loaded._load_process_dict(data)
    loaded_etched = loaded.history[-1]
    win = loaded
    check_state(loaded_etched, FEATURE_EMPHASIS_DEFAULT)
    ok, message = win._execute_assistant_command({"action": "operation", "operation": "Flip", "material": ""})
    if not ok:
        raise AssertionError(f"Flip after bonded-substrate etch failed: {message}")
    check_state(win.history[-1], FEATURE_EMPHASIS_DEFAULT)

    # A bonded wafer can itself be at the lower range endpoint. It is still a
    # substrate-like structural body and must receive the visibility floor.
    _app, nano = _new_window()
    ok, message = nano._execute_assistant_command({
        "action": "operation",
        "operation": "Bond",
        "material": "Silicon (Si)",
        "thickness_um": 0.001,
    })
    if not ok:
        raise AssertionError(f"1nm bonded wafer failed: {message}")
    nano_state = nano.history[-1]
    nano_profile = nano._process_display_adjustment_profile()
    nano._display_grid_auto(
        np.asarray(nano_state.get("grid")),
        row_heights_um=np.asarray(nano_state.get("row_heights_um"), dtype=float),
        history_index=len(nano.history) - 1,
        visual_geometry=nano_state.get("visual_geometry", []),
        process_profile=nano_profile,
    )
    nano_columns = (getattr(nano, "_last_display_remap_info", {}) or {}).get("column_segments", [])
    nano_cap = next(
        segment for segment in nano_columns[0]
        if int(segment.get("structural_body_id", 0)) == 2
    )
    nano_rows = int(nano_cap["display_end"]) - int(nano_cap["display_start"])
    if nano_rows < int(MIN_LAYER_PX):
        raise AssertionError(f"A 1nm bonded substrate must remain visible, got {nano_rows} rows")


def test_external_multibody_bond_preserves_roles_and_buried_etch_scale():
    _app, source = _new_window()
    for command in [
        {"action": "set_substrate", "material": "Silicon (Si)", "thickness_um": 500.0},
        {"action": "operation", "operation": "Bond", "material": "Silicon (Si)", "thickness_um": 100.0},
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 5.0},
        {
            "action": "operation",
            "operation": "Lithography",
            "material": "PR_SU-8 Photoresist",
            "windows": [(0.40, 0.60)],
            "invert_lithography": False,
        },
        {
            "action": "operation",
            "operation": "Etch",
            "material": "Silicon (Si)",
            "depth_um": 40.0,
            "etch_mode": "anisotropic",
        },
    ]:
        ok, message = source._execute_assistant_command(command)
        if not ok:
            raise AssertionError(f"External source construction failed: {message}")

    source_data = source._make_process_dict()
    _app, target = _new_window()
    source_info = target._build_bond_source_info(source_data, "multibody_source.fab")
    target._set_bond_source_info(source_info)
    ok, message = target._execute_assistant_command({
        "action": "operation",
        "operation": "Bond",
        "material": "Silicon (Si)",
        "thickness_um": 0.001,
    })
    if not ok:
        raise AssertionError(f"External multi-body bond failed: {message}")

    def assert_physical_roles(window, state):
        roles = np.asarray(state.get("structural_role_grid"), dtype=np.int16)
        heights = np.asarray(state.get("row_heights_um"), dtype=float)
        present = {int(value) for value in np.unique(roles) if int(value) > 0}
        if not {1, 2, 3}.issubset(present):
            raise AssertionError(f"External role remap collapsed a wafer body: {sorted(present)}")
        expected = {
            1: (200.0, 200.0),
            2: (500.0, 500.0),
            3: (100.0, 60.0),
        }
        for body_id, (outside_expected, center_expected) in expected.items():
            outside = float(np.sum(heights[roles[:, 0] == body_id]))
            center = float(np.sum(heights[roles[:, 150] == body_id]))
            if abs(outside - outside_expected) > TOL_UM or abs(center - center_expected) > TOL_UM:
                raise AssertionError(
                    f"External body {body_id} physical geometry changed: "
                    f"expected {outside_expected:g}/{center_expected:g}, got {outside:g}/{center:g}"
                )

    def assert_buried_display(window, state, factor):
        window.feature_scale_factor = factor
        profile = window._process_display_adjustment_profile()
        state_index = next(idx for idx, item in enumerate(window.history) if item is state)
        window._display_grid_auto(
            np.asarray(state.get("grid")),
            row_heights_um=np.asarray(state.get("row_heights_um"), dtype=float),
            history_index=state_index,
            visual_geometry=state.get("visual_geometry", []),
            process_profile=profile,
        )
        columns = (getattr(window, "_last_display_remap_info", {}) or {}).get("column_segments", [])

        def body_rows(col, body_id):
            return sum(
                int(segment.get("display_end", 0)) - int(segment.get("display_start", 0))
                for segment in columns[col]
                if int(segment.get("structural_body_id", 0)) == int(body_id)
            )

        outside = body_rows(0, 3)
        center = body_rows(150, 3)
        gap = outside - center
        expected_gap = int(profile["resolved_feature_rows"][window._display_feature_profile_key(40.0)])
        if gap != expected_gap:
            raise AssertionError(
                f"Buried imported 40um etch lost its feature target at {factor:g}: "
                f"expected {expected_gap}, got {gap}"
            )
        if center < gap + int(ORDER_MARGIN_PX):
            raise AssertionError(
                f"Imported body core 60um > gap 40um must stay ordered at {factor:g}: "
                f"core={center}, gap={gap}"
            )
        if not any(
            int(context.get("structural_body_id", 0)) == 3
            and abs(float(profile["feature_values_um"][context["gap_key"]]) - 40.0) <= TOL_UM
            for context in profile.get("gap_contexts", [])
        ):
            raise AssertionError("Buried imported body has no independent body-local gap context")

    final_state = target.history[-1]
    assert_physical_roles(target, final_state)
    for factor in (FEATURE_EMPHASIS_MIN, FEATURE_EMPHASIS_DEFAULT, FEATURE_EMPHASIS_MAX):
        assert_buried_display(target, final_state, factor)

    if not target._ensure_process_steps_for_substrate_rebuild():
        raise AssertionError("External multi-body bonding should retain enough metadata for replay")
    target.rebuild_process_from_steps(update_view=False)
    replayed = target.history[-1]
    assert_physical_roles(target, replayed)
    assert_buried_display(target, replayed, FEATURE_EMPHASIS_DEFAULT)

    target.feature_scale_factor = FEATURE_EMPHASIS_DEFAULT
    replay_data = target._make_process_dict()
    _app, loaded = _new_window()
    loaded._load_process_dict(replay_data)
    loaded_state = loaded.history[-1]
    assert_physical_roles(loaded, loaded_state)
    assert_buried_display(loaded, loaded_state, FEATURE_EMPHASIS_DEFAULT)


def test_external_bond_remaps_planarizer_metadata_once():
    source = _window_from_commands(_planarized_recess_commands())
    source_data = source._make_process_dict()
    _app, target = _new_window()
    source_info = target._build_bond_source_info(
        source_data,
        "planarized_external_source.fab",
    )
    target._set_bond_source_info(source_info)
    ok, message = target._execute_assistant_command({
        "action": "operation",
        "operation": "Bond",
        "material": "Silicon (Si)",
        "thickness_um": 0.001,
    })
    if not ok:
        raise AssertionError(f"External planarized-stack bond failed: {message}")

    def assert_one_role(window, state, label):
        grid = np.asarray(state.get("planarizer_role_grid"), dtype=np.int32)
        used_ids = {
            abs(int(value)) for value in np.unique(grid) if int(value) != 0
        }
        metadata = state.get("planarizer_roles", {}) or {}
        metadata_ids = {
            int(key) for key in metadata
            if str(key).lstrip("-").isdigit() and int(key) > 0
        }
        if used_ids != metadata_ids or len(used_ids) != 1:
            raise AssertionError(
                f"{label} duplicated or orphaned imported planarizer metadata: "
                f"used={sorted(used_ids)}, metadata={sorted(metadata_ids)}"
            )
        role_id = next(iter(used_ids))
        role = metadata.get(role_id, metadata.get(str(role_id), {}))
        if int((role or {}).get("orientation", 0)) != -1:
            raise AssertionError(
                f"{label} did not invert external planarizer orientation: {role}"
            )
        fill = list((role or {}).get("fill_thickness_um_by_column", []) or [])
        if len(fill) != grid.shape[1] or abs(float(fill[150]) - 20.0) > TOL_UM:
            raise AssertionError(
                f"{label} lost imported per-column fill metadata: center={fill[150] if len(fill) > 150 else None}"
            )

    assert_one_role(target, target.history[-1], "External bond")
    if not target._ensure_process_steps_for_substrate_rebuild():
        raise AssertionError("External planarized bond should be replayable")
    target.rebuild_process_from_steps(update_view=False)
    assert_one_role(target, target.history[-1], "External bond replay")

    data = target._make_process_dict()
    _app, loaded = _new_window()
    loaded._load_process_dict(data)
    assert_one_role(loaded, loaded.history[-1], "External bond save/load")


def test_cross_body_etch_compares_complete_step_to_layers():
    def build(film_um):
        _app, window = _new_window()
        commands = [
            {"action": "set_substrate", "material": "Silicon (Si)", "thickness_um": 500.0},
            {"action": "operation", "operation": "Bond", "material": "Silicon (Si)", "thickness_um": 100.0},
            {"action": "add_material", "material": "PR_SU-8 Photoresist"},
            {"action": "add_material", "material": "Silicon Dioxide (SiO2)"},
            {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 5.0},
            {
                "action": "operation",
                "operation": "Lithography",
                "material": "PR_SU-8 Photoresist",
                "windows": [(0.40, 0.60)],
                "invert_lithography": False,
            },
            {
                "action": "operation",
                "operation": "Etch",
                "material": "Silicon (Si)",
                "depth_um": 140.0,
                "etch_mode": "anisotropic",
            },
            {
                "action": "operation",
                "operation": "Deposit",
                "material": "Silicon Dioxide (SiO2)",
                "thickness_um": float(film_um),
            },
        ]
        for command in commands:
            ok, message = window._execute_assistant_command(command)
            if not ok:
                raise AssertionError(f"Cross-body {film_um:g}um comparison failed: {message}")
        return window, window.history[-1]

    for film_um in (120.0, 140.0, 160.0):
        win, state = build(film_um)
        roles = np.asarray(state.get("structural_role_grid"), dtype=np.int16)
        heights = np.asarray(state.get("row_heights_um"), dtype=float)
        removed = sum(
            float(np.sum(heights[roles[:, 0] == body_id]))
            - float(np.sum(heights[roles[:, 150] == body_id]))
            for body_id in (1, 2)
        )
        if abs(removed - 140.0) > TOL_UM:
            raise AssertionError(f"Cross-body physical etch should total 140um, got {removed:g}")

        for factor in (FEATURE_EMPHASIS_MIN, FEATURE_EMPHASIS_DEFAULT, FEATURE_EMPHASIS_MAX):
            win.feature_scale_factor = factor
            profile = win._process_display_adjustment_profile()
            win._display_grid_auto(
                np.asarray(state.get("grid")),
                row_heights_um=np.asarray(state.get("row_heights_um"), dtype=float),
                history_index=len(win.history) - 1,
                visual_geometry=state.get("visual_geometry", []),
                process_profile=profile,
            )
            columns = (getattr(win, "_last_display_remap_info", {}) or {}).get("column_segments", [])

            def rows(col, predicate):
                return sum(
                    int(segment.get("display_end", 0)) - int(segment.get("display_start", 0))
                    for segment in columns[col]
                    if predicate(segment)
                )

            complete_step = sum(
                rows(0, lambda segment, bid=body_id: int(segment.get("structural_body_id", 0)) == bid)
                - rows(150, lambda segment, bid=body_id: int(segment.get("structural_body_id", 0)) == bid)
                for body_id in (1, 2)
            )
            oxide_index = win.material_names.index("Silicon Dioxide (SiO2)")
            film_rows = rows(150, lambda segment: int(segment.get("material", -1)) == oxide_index)
            if film_um < 140.0 and complete_step < film_rows + int(ORDER_MARGIN_PX):
                raise AssertionError(
                    f"140um complete step must exceed {film_um:g}um layer at {factor:g}: "
                    f"step={complete_step}, layer={film_rows}"
                )
            if film_um == 140.0 and complete_step != film_rows:
                raise AssertionError(
                    f"Equal 140um cross-body step/layer must display equally at {factor:g}: "
                    f"step={complete_step}, layer={film_rows}"
                )
            if film_um > 140.0 and film_rows < complete_step + int(ORDER_MARGIN_PX):
                raise AssertionError(
                    f"{film_um:g}um layer must exceed 140um complete step at {factor:g}: "
                    f"layer={film_rows}, step={complete_step}"
                )
            if not any(
                bool(context.get("is_cumulative", False))
                and abs(float(profile["feature_values_um"][context["gap_key"]]) - 140.0) <= TOL_UM
                for context in profile.get("gap_contexts", [])
            ):
                raise AssertionError("Cross-body etch is missing its cumulative 140um surface-step context")


def test_buried_cavity_is_not_a_cumulative_surface_step():
    _app, win = _new_window()
    commands = [
        {"action": "set_substrate", "material": "Silicon (Si)", "thickness_um": 500.0},
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "add_material", "material": "Silicon Dioxide (SiO2)"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 5.0},
        {
            "action": "operation",
            "operation": "Lithography",
            "material": "PR_SU-8 Photoresist",
            "windows": [(0.40, 0.60)],
            "invert_lithography": False,
        },
        {
            "action": "operation",
            "operation": "Etch",
            "material": "Silicon (Si)",
            "depth_um": 40.0,
            "etch_mode": "anisotropic",
        },
        {"action": "operation", "operation": "Strip material", "material": "PR_SU-8 Photoresist"},
        {"action": "operation", "operation": "Bond", "material": "Silicon (Si)", "thickness_um": 100.0},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 5.0},
        {
            "action": "operation",
            "operation": "Lithography",
            "material": "PR_SU-8 Photoresist",
            "windows": [(0.40, 0.60)],
            "invert_lithography": False,
        },
        {
            "action": "operation",
            "operation": "Etch",
            "material": "Silicon (Si)",
            "depth_um": 20.0,
            "etch_mode": "anisotropic",
        },
        {
            "action": "operation",
            "operation": "Deposit",
            "material": "Silicon Dioxide (SiO2)",
            "thickness_um": 40.0,
        },
    ]
    for command in commands:
        ok, message = win._execute_assistant_command(command)
        if not ok:
            raise AssertionError(f"Disconnected buried-cavity process failed: {message}")

    state = win.history[-1]
    roles = np.asarray(state.get("structural_role_grid"), dtype=np.int16)
    heights = np.asarray(state.get("row_heights_um"), dtype=float)
    body1_drop = float(np.sum(heights[roles[:, 0] == 1]) - np.sum(heights[roles[:, 150] == 1]))
    body2_drop = float(np.sum(heights[roles[:, 0] == 2]) - np.sum(heights[roles[:, 150] == 2]))
    if abs(body1_drop - 40.0) > TOL_UM or abs(body2_drop - 20.0) > TOL_UM:
        raise AssertionError(f"Expected buried/exposed removals 40/20um, got {body1_drop:g}/{body2_drop:g}")

    for factor in (FEATURE_EMPHASIS_MIN, FEATURE_EMPHASIS_DEFAULT, FEATURE_EMPHASIS_MAX):
        win.feature_scale_factor = factor
        profile = win._process_display_adjustment_profile()
        false_contexts = [
            context for context in profile.get("gap_contexts", [])
            if bool(context.get("is_cumulative", False))
            and abs(float(profile["feature_values_um"][context["gap_key"]]) - 60.0) <= TOL_UM
        ]
        if false_contexts:
            raise AssertionError(
                f"A remaining 80um cap must disconnect the buried 40um cavity from the exposed 20um step: "
                f"{false_contexts}"
            )
        win._display_grid_auto(
            np.asarray(state.get("grid")),
            row_heights_um=heights,
            history_index=len(win.history) - 1,
            visual_geometry=state.get("visual_geometry", []),
            process_profile=profile,
        )
        columns = (getattr(win, "_last_display_remap_info", {}) or {}).get("column_segments", [])

        def rows(col, predicate):
            return sum(
                int(segment.get("display_end", 0)) - int(segment.get("display_start", 0))
                for segment in columns[col]
                if predicate(segment)
            )

        exposed_step = (
            rows(0, lambda segment: int(segment.get("structural_body_id", 0)) == 2)
            - rows(150, lambda segment: int(segment.get("structural_body_id", 0)) == 2)
        )
        oxide_index = win.material_names.index("Silicon Dioxide (SiO2)")
        film_rows = rows(150, lambda segment: int(segment.get("material", -1)) == oxide_index)
        if film_rows < exposed_step + int(ORDER_MARGIN_PX):
            raise AssertionError(
                f"40um film must compare to the exposed 20um cap step, not a false 60um sum at {factor:g}: "
                f"film={film_rows}, exposed_step={exposed_step}"
            )


def test_nonstructural_interlayer_controls_cumulative_connectivity():
    _app, win = _new_window()
    win.material_names = [
        "Void",
        "Silicon (Si)",
        "Silicon Dioxide (SiO2)",
        "Aluminum Nitride (AlN)",
    ]
    win.material_colors = ["#ffffff", "#aaaaaa", "#55aaff", "#ff5500"]
    xsec = CrossSection(
        width_px=2,
        height_px=500,
        substrate_index=1,
        void_index=0,
        substrate_thickness_px=500,
        substrate_thickness_um=500.0,
        top_headroom_um=0.0,
    )
    xsec.structural_bodies = {
        1: {
            "body_id": 1,
            "material_index": 1,
            "nominal_thickness_um": 500.0,
            "kind": "substrate",
        },
        2: {
            "body_id": 2,
            "material_index": 1,
            "nominal_thickness_um": 100.0,
            "kind": "bonded_substrate",
        },
    }
    xsec._next_structural_body_id = 3
    win.xsec = xsec

    def make_state(*, separator_open, film_um):
        # Top -> bottom: comparison film, a cap that is fully removed only in
        # the centre, a 10um role-0 bonding plane, then a lower wafer with a
        # sealed 40um cavity. The common row geometry makes the separator's
        # physical connectivity explicit and independent of material names.
        grid = np.asarray([
            [3, 3],
            [1, 0],
            [2, 0 if separator_open else 2],
            [1, 0],
            [1, 1],
        ], dtype=np.int16)
        roles = np.asarray([
            [0, 0],
            [2, 0],
            [0, 0],
            [1, 0],
            [1, 1],
        ], dtype=np.int16)
        return {
            "grid": grid,
            "row_heights_um": np.asarray([film_um, 100.0, 10.0, 40.0, 460.0], dtype=float),
            "visual_geometry": [],
            "structural_role_grid": roles,
            "structural_bodies": xsec.copy_structural_bodies(),
            "operation": "Deposit",
            "material_name": "Aluminum Nitride (AlN)",
            "title": "open separator" if separator_open else "solid separator",
        }

    solid_state = make_state(separator_open=False, film_um=120.0)
    open_state = make_state(separator_open=True, film_um=160.0)

    def render(state, factor):
        win.history = [state]
        win.feature_scale_factor = factor
        profile = win._process_display_adjustment_profile()
        win._display_grid_auto(
            state["grid"],
            row_heights_um=state["row_heights_um"],
            history_index=0,
            process_profile=profile,
            structural_role_grid=state["structural_role_grid"],
        )
        columns = (getattr(win, "_last_display_remap_info", {}) or {}).get("column_segments", [])

        def rows(col, predicate):
            return sum(
                int(segment.get("display_end", 0)) - int(segment.get("display_start", 0))
                for segment in columns[col]
                if predicate(segment)
            )

        film_rows = rows(1, lambda segment: int(segment.get("material", -1)) == 3)
        cap_gap = (
            rows(0, lambda segment: int(segment.get("structural_body_id", 0)) == 2)
            - rows(1, lambda segment: int(segment.get("structural_body_id", 0)) == 2)
        )
        separator_gap = rows(0, lambda segment: int(segment.get("material", -1)) == 2) - rows(
            1,
            lambda segment: int(segment.get("material", -1)) == 2,
        )
        lower_gap = (
            rows(0, lambda segment: int(segment.get("structural_body_id", 0)) == 1)
            - rows(1, lambda segment: int(segment.get("structural_body_id", 0)) == 1)
        )
        return profile, film_rows, cap_gap, separator_gap, lower_gap

    for factor in (FEATURE_EMPHASIS_MIN, FEATURE_EMPHASIS_DEFAULT, FEATURE_EMPHASIS_MAX):
        profile, film_rows, cap_gap, _separator_gap, _lower_gap = render(solid_state, factor)
        cumulative_depths = [
            float(profile["feature_values_um"][context["gap_key"]])
            for context in profile.get("gap_contexts", [])
            if bool(context.get("is_cumulative", False))
        ]
        if any(abs(depth - 140.0) <= TOL_UM for depth in cumulative_depths):
            raise AssertionError(
                f"A solid 10um interlayer must block the buried 40um cavity at {factor:g}: "
                f"cumulative={cumulative_depths}"
            )
        if film_rows < cap_gap + int(ORDER_MARGIN_PX):
            raise AssertionError(
                f"120um film must exceed only the exposed 100um cap opening at {factor:g}: "
                f"film={film_rows}, cap={cap_gap}"
            )

        profile, film_rows, cap_gap, separator_gap, lower_gap = render(open_state, factor)
        cumulative = [
            context for context in profile.get("gap_contexts", [])
            if bool(context.get("is_cumulative", False))
            and abs(float(profile["feature_values_um"][context["gap_key"]]) - 150.0) <= TOL_UM
        ]
        if not cumulative:
            raise AssertionError(
                f"Removing the 10um separator must connect 100+10+40um at {factor:g}"
            )
        complete_gap = cap_gap + separator_gap + lower_gap
        if film_rows < complete_gap + int(ORDER_MARGIN_PX):
            raise AssertionError(
                f"160um film must exceed connected 150um cap+separator+cavity at {factor:g}: "
                f"film={film_rows}, gap={complete_gap}"
            )


def test_distinct_open_interlayers_keep_component_scaling():
    _app, win = _new_window()
    win.material_names = [
        "Void",
        "Silicon (Si)",
        "Silicon Dioxide (SiO2)",
        "Silicon Nitride (SiN)",
        "Aluminum Nitride (AlN)",
    ]
    win.material_colors = ["#ffffff", "#aaaaaa", "#55aaff", "#aa55ff", "#ff5500"]
    xsec = CrossSection(
        width_px=2,
        height_px=500,
        substrate_index=1,
        void_index=0,
        substrate_thickness_px=500,
        substrate_thickness_um=500.0,
        top_headroom_um=0.0,
    )
    xsec.structural_bodies = {
        1: {
            "body_id": 1,
            "material_index": 1,
            "nominal_thickness_um": 500.0,
            "kind": "substrate",
        },
        2: {
            "body_id": 2,
            "material_index": 1,
            "nominal_thickness_um": 100.0,
            "kind": "bonded_substrate",
        },
    }
    xsec._next_structural_body_id = 3
    win.xsec = xsec

    # Top -> bottom: 150um comparison film, a fully opened 100um cap,
    # two physically distinct cleared 5um interface films, and a 40um
    # opening in the lower wafer. The connected recess is exactly 150um.
    grid = np.asarray([
        [4, 4],
        [1, 0],
        [2, 0],
        [3, 0],
        [1, 0],
        [1, 1],
    ], dtype=np.int16)
    roles = np.asarray([
        [0, 0],
        [2, 0],
        [0, 0],
        [0, 0],
        [1, 0],
        [1, 1],
    ], dtype=np.int16)
    state = {
        "grid": grid,
        "row_heights_um": np.asarray([150.0, 100.0, 5.0, 5.0, 40.0, 460.0]),
        "visual_geometry": [],
        "structural_role_grid": roles,
        "structural_bodies": xsec.copy_structural_bodies(),
        "operation": "Deposit",
        "material_name": "Aluminum Nitride (AlN)",
        "title": "two open interface films",
    }
    win.history = [state]

    for factor in (FEATURE_EMPHASIS_MIN, FEATURE_EMPHASIS_DEFAULT, FEATURE_EMPHASIS_MAX):
        win.feature_scale_factor = factor
        profile = win._process_display_adjustment_profile()
        win._display_grid_auto(
            grid,
            row_heights_um=state["row_heights_um"],
            history_index=0,
            process_profile=profile,
            structural_role_grid=roles,
        )
        columns = (getattr(win, "_last_display_remap_info", {}) or {}).get(
            "column_segments",
            [],
        )

        def rows(col, predicate):
            return sum(
                int(segment.get("display_end", 0)) - int(segment.get("display_start", 0))
                for segment in columns[col]
                if predicate(segment)
            )

        film_rows = rows(1, lambda segment: int(segment.get("material", -1)) == 4)
        actual_step = sum(
            rows(0, lambda segment, material=material: int(segment.get("material", -1)) == material)
            - rows(1, lambda segment, material=material: int(segment.get("material", -1)) == material)
            for material in (1, 2, 3)
        )
        cumulative = [
            context for context in profile.get("gap_contexts", [])
            if bool(context.get("is_cumulative", False))
            and abs(float(profile["feature_values_um"][context["gap_key"]]) - 150.0) <= TOL_UM
        ]
        expected_components = sorted([5.0, 5.0, 40.0, 100.0])
        if not cumulative:
            raise AssertionError(f"Missing cumulative 150um context at {factor:g}")
        component_values = sorted(
            float(profile["feature_values_um"][key])
            for key in cumulative[0].get("component_gap_keys", ())
        )
        if component_values != expected_components:
            raise AssertionError(
                f"Distinct open interface films must remain 100+5+5+40 at {factor:g}, "
                f"got {component_values}"
            )
        if film_rows != actual_step:
            raise AssertionError(
                f"150um film must equal the displayed 100+5+5+40um connected step at {factor:g}: "
                f"film={film_rows}, gap={actual_step}"
            )


def test_same_material_open_interlayer_uses_local_reference_merge():
    _app, win = _new_window()
    win.material_names = [
        "Void",
        "Silicon (Si)",
        "Silicon Dioxide (SiO2)",
        "Aluminum Nitride (AlN)",
    ]
    win.material_colors = ["#ffffff", "#aaaaaa", "#55aaff", "#ff5500"]
    xsec = CrossSection(
        width_px=3,
        height_px=500,
        substrate_index=1,
        void_index=0,
        substrate_thickness_px=500,
        substrate_thickness_um=500.0,
        top_headroom_um=0.0,
    )
    xsec.structural_bodies = {
        1: {
            "body_id": 1,
            "material_index": 1,
            "nominal_thickness_um": 500.0,
            "kind": "substrate",
        },
        2: {
            "body_id": 2,
            "material_index": 1,
            "nominal_thickness_um": 100.0,
            "kind": "bonded_substrate",
        },
    }
    xsec._next_structural_body_id = 3
    win.xsec = xsec

    # Col0 is the intact local reference for the centre opening. Its two
    # adjacent SiO2 rows render as one 10um feature. Col2 has an unrelated
    # lateral opening in only the lower row and must not split col0 into5+5.
    grid = np.asarray([
        [3, 3, 3],
        [1, 0, 1],
        [2, 0, 2],
        [2, 0, 0],
        [1, 0, 1],
        [1, 1, 1],
    ], dtype=np.int16)
    roles = np.asarray([
        [0, 0, 0],
        [2, 0, 2],
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 1],
        [1, 1, 1],
    ], dtype=np.int16)
    state = {
        "grid": grid,
        "row_heights_um": np.asarray([150.0, 100.0, 5.0, 5.0, 40.0, 460.0]),
        "visual_geometry": [],
        "structural_role_grid": roles,
        "structural_bodies": xsec.copy_structural_bodies(),
        "operation": "Deposit",
        "material_name": "Aluminum Nitride (AlN)",
        "title": "local same-material interface reference",
    }
    win.history = [state]

    for factor in (FEATURE_EMPHASIS_MIN, FEATURE_EMPHASIS_DEFAULT, FEATURE_EMPHASIS_MAX):
        win.feature_scale_factor = factor
        profile = win._process_display_adjustment_profile()
        win._display_grid_auto(
            grid,
            row_heights_um=state["row_heights_um"],
            history_index=0,
            process_profile=profile,
            structural_role_grid=roles,
        )
        columns = (getattr(win, "_last_display_remap_info", {}) or {}).get(
            "column_segments",
            [],
        )

        def rows(col, predicate):
            return sum(
                int(segment.get("display_end", 0)) - int(segment.get("display_start", 0))
                for segment in columns[col]
                if predicate(segment)
            )

        film_rows = rows(1, lambda segment: int(segment.get("material", -1)) == 3)
        actual_step = sum(
            rows(0, lambda segment, material=material: int(segment.get("material", -1)) == material)
            - rows(1, lambda segment, material=material: int(segment.get("material", -1)) == material)
            for material in (1, 2)
        )
        cumulative = [
            context for context in profile.get("gap_contexts", [])
            if bool(context.get("is_cumulative", False))
            and abs(float(profile["feature_values_um"][context["gap_key"]]) - 150.0) <= TOL_UM
        ]
        if not cumulative:
            raise AssertionError(f"Missing cumulative 150um context at {factor:g}")
        component_values = sorted(
            float(profile["feature_values_um"][key])
            for key in cumulative[0].get("component_gap_keys", ())
        )
        if component_values != [10.0, 40.0, 100.0]:
            raise AssertionError(
                f"The intact local reference must merge the adjacent SiO2 interface to10um at "
                f"{factor:g}, got {component_values}"
            )
        if film_rows != actual_step:
            raise AssertionError(
                f"150um film must equal the locally rendered 100+10+40um connected step at "
                f"{factor:g}: film={film_rows}, gap={actual_step}"
            )


def test_multiple_interface_regions_use_nearest_local_reference():
    _app, win = _new_window()
    win.material_names = [
        "Void",
        "Silicon (Si)",
        "Silicon Dioxide (SiO2)",
        "Silicon Nitride (SiN)",
        "Aluminum Nitride (AlN)",
    ]
    win.material_colors = ["#ffffff", "#aaaaaa", "#55aaff", "#aa55ff", "#ff5500"]
    xsec = CrossSection(
        width_px=4,
        height_px=500,
        substrate_index=1,
        void_index=0,
        substrate_thickness_px=500,
        substrate_thickness_um=500.0,
        top_headroom_um=0.0,
    )
    xsec.structural_bodies = {
        1: {
            "body_id": 1,
            "material_index": 1,
            "nominal_thickness_um": 500.0,
            "kind": "substrate",
        },
        2: {
            "body_id": 2,
            "material_index": 1,
            "nominal_thickness_um": 100.0,
            "kind": "bonded_substrate",
        },
    }
    xsec._next_structural_body_id = 3
    win.xsec = xsec

    # Col0/1 form a same-material interface region (10um). Col2/3 form a
    # distinct-material region (5+5um). Each open column must inherit its
    # nearest intact region rather than one body-pair-wide reference.
    grid = np.asarray([
        [4, 4, 4, 4],
        [1, 0, 1, 0],
        [2, 0, 2, 0],
        [2, 0, 3, 0],
        [1, 0, 1, 0],
        [1, 1, 1, 1],
    ], dtype=np.int16)
    roles = np.asarray([
        [0, 0, 0, 0],
        [2, 0, 2, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 1, 0],
        [1, 1, 1, 1],
    ], dtype=np.int16)
    state = {
        "grid": grid,
        "row_heights_um": np.asarray([150.0, 100.0, 5.0, 5.0, 40.0, 460.0]),
        "visual_geometry": [],
        "structural_role_grid": roles,
        "structural_bodies": xsec.copy_structural_bodies(),
        "operation": "Deposit",
        "material_name": "Aluminum Nitride (AlN)",
        "title": "two lateral interface regions",
    }
    win.history = [state]

    for factor in (FEATURE_EMPHASIS_MIN, FEATURE_EMPHASIS_DEFAULT, FEATURE_EMPHASIS_MAX):
        win.feature_scale_factor = factor
        profile = win._process_display_adjustment_profile()
        win._display_grid_auto(
            grid,
            row_heights_um=state["row_heights_um"],
            history_index=0,
            process_profile=profile,
            structural_role_grid=roles,
        )
        columns = (getattr(win, "_last_display_remap_info", {}) or {}).get(
            "column_segments",
            [],
        )

        def rows(col, predicate):
            return sum(
                int(segment.get("display_end", 0)) - int(segment.get("display_start", 0))
                for segment in columns[col]
                if predicate(segment)
            )

        film_rows = rows(1, lambda segment: int(segment.get("material", -1)) == 4)
        left_step = sum(
            rows(0, lambda segment, material=material: int(segment.get("material", -1)) == material)
            - rows(1, lambda segment, material=material: int(segment.get("material", -1)) == material)
            for material in (1, 2, 3)
        )
        right_step = sum(
            rows(2, lambda segment, material=material: int(segment.get("material", -1)) == material)
            - rows(3, lambda segment, material=material: int(segment.get("material", -1)) == material)
            for material in (1, 2, 3)
        )
        component_patterns = {
            tuple(sorted(
                float(profile["feature_values_um"][key])
                for key in context.get("component_gap_keys", ())
            ))
            for context in profile.get("gap_contexts", [])
            if bool(context.get("is_cumulative", False))
            and abs(float(profile["feature_values_um"][context["gap_key"]]) - 150.0) <= TOL_UM
        }
        expected_patterns = {
            (10.0, 40.0, 100.0),
            (5.0, 5.0, 40.0, 100.0),
        }
        if not expected_patterns.issubset(component_patterns):
            raise AssertionError(
                f"Both local interface topologies must be retained at {factor:g}: "
                f"got {component_patterns}"
            )
        if film_rows != left_step or film_rows != right_step:
            raise AssertionError(
                f"150um film must equal both locally rendered connected steps at {factor:g}: "
                f"film={film_rows}, left={left_step}, right={right_step}"
            )


def test_spincoat_does_not_plug_an_unsupported_through_hole():
    win, through_state = _masked_substrate_etch_window(500.0, 500.0)
    ok, message = win._execute_assistant_command({
        "action": "operation",
        "operation": "Spincoating",
        "material": "PR_SU-8 Photoresist",
        "thickness_um": 5.0,
    })
    if not ok:
        raise AssertionError(f"Spin coat after through-etch failed: {message}")
    state = win.history[-1]
    grid = np.asarray(state.get("grid"))
    if np.any(grid[:, 150] != win.VOID_INDEX):
        raise AssertionError(
            "An unsupported through-hole must remain open; spin coating cannot invent a bottom support surface"
        )
    pr = _material_column_heights(state, win.material_names, "PR_SU-8 Photoresist")
    if float(pr[150]) > TOL_UM or float(pr[0]) <= TOL_UM:
        raise AssertionError(
            f"Spin coat should remain on supported regions only, got outside/through={pr[0]:g}/{pr[150]:g}um"
        )


def test_inverse_lithography_after_topography():
    commands = _base_substrate_commands() + [
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "add_material", "material": "Silicon Dioxide (SiO2)"},
        {"action": "operation", "operation": "Etch", "material": BASE_SUBSTRATE_MATERIAL, "depth_um": 0.2, "etch_mode": "isotropic"},
        {"action": "operation", "operation": "Deposit", "material": "Silicon Dioxide (SiO2)", "thickness_um": 0.1},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 10.0},
        {
            "action": "operation",
            "operation": "Lithography",
            "material": "PR_SU-8 Photoresist",
            "windows": [(0.35, 0.65)],
            "invert_lithography": True,
        },
    ]
    win, path = _run_commands("FabStack_dimtest_inverse_lithography_topography", commands)
    state = _state_by_operation(win.history, "Lithography")
    _assert_stats(win, state, "PR_SU-8 Photoresist", 10.0, count=90)
    _assert_stats(win, state, "Silicon Dioxide (SiO2)", 0.1, count=300)
    return path


def test_sub_nm_deposit_and_etch():
    commands = _base_substrate_commands() + [
        {"action": "add_material", "material": "Gold"},
        {"action": "operation", "operation": "Deposit", "material": "Gold", "thickness_um": 0.0005},
        {"action": "operation", "operation": "Etch", "material": "Gold", "depth_um": 0.0004, "etch_mode": "anisotropic"},
    ]
    win, path = _run_commands("FabStack_dimtest_sub_nm_deposit_etch", commands)
    state = _state_by_operation(win.history, "Etch")
    _assert_stats(win, state, "Gold", 0.0001, count=300)
    return path


def test_flip_then_process():
    commands = _base_substrate_commands() + [
        {"action": "add_material", "material": "Gold"},
        {"action": "add_material", "material": "Chromium"},
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "operation", "operation": "Deposit", "material": "Gold", "thickness_um": 0.15},
        {"action": "operation", "operation": "Flip", "material": ""},
        {"action": "operation", "operation": "Deposit", "material": "Chromium", "thickness_um": 0.015},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 10.0},
    ]
    win, path = _run_commands("FabStack_dimtest_flip_then_process", commands)
    final = win.history[-1]
    _assert_stats(win, final, "Gold", 0.15, count=300)
    _assert_stats(win, final, "Chromium", 0.015, count=300)
    _assert_stats(win, final, "PR_SU-8 Photoresist", 10.0, count=300)
    return path


def test_gap_layer_display_consistency_stress():
    commands = _base_substrate_commands() + [
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "add_material", "material": "Chromium"},
        {"action": "add_material", "material": "Gold"},
        {"action": "add_material", "material": "Silicon Dioxide (SiO2)"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 10.0},
        {
            "action": "operation",
            "operation": "Lithography",
            "material": "PR_SU-8 Photoresist",
            "windows": [(0.35, 0.65)],
            "invert_lithography": False,
        },
        {"action": "operation", "operation": "Etch", "material": BASE_SUBSTRATE_MATERIAL, "depth_um": 0.2, "etch_mode": "isotropic"},
        {"action": "operation", "operation": "Deposit", "material": "Chromium", "thickness_um": 0.015},
        {"action": "operation", "operation": "Deposit", "material": "Gold", "thickness_um": 0.15},
        {"action": "operation", "operation": "Deposit", "material": "Silicon Dioxide (SiO2)", "thickness_um": 0.1},
    ]
    win, path = _run_commands("FabStack_dimtest_gap_layer_consistency_stress", commands)

    chromium_deposit = _state_by_operation(win.history, "Deposit", occurrence=1)
    _assert_material_display_consistent_across_gap(win, chromium_deposit, "Chromium")

    gold_deposit = _state_by_operation(win.history, "Deposit", occurrence=2)
    _assert_material_display_consistent_across_gap(win, gold_deposit, "Chromium")
    _assert_material_display_consistent_across_gap(win, gold_deposit, "Gold")

    oxide_deposit = _state_by_operation(win.history, "Deposit", occurrence=3)
    _assert_material_display_consistent_across_gap(win, oxide_deposit, "Chromium")
    _assert_material_display_consistent_across_gap(win, oxide_deposit, "Gold")
    _assert_material_display_consistent_across_gap(win, oxide_deposit, "Silicon Dioxide (SiO2)")
    _assert_material_visual_order_in_gap(win, oxide_deposit, "Gold", "Silicon Dioxide (SiO2)")
    _assert_consistent_display_profile(win)
    return path


def test_process_wide_visual_consistency_many_steps():
    commands = [
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "add_material", "material": "Gold"},
        {"action": "add_material", "material": "Chromium"},
        {"action": "operation", "operation": "Deposit", "material": "Chromium", "thickness_um": 0.01},
        {"action": "operation", "operation": "Deposit", "material": "Gold", "thickness_um": 0.1},
    ]
    for idx in range(6):
        commands.extend([
            {
                "action": "operation",
                "operation": "Spincoating",
                "material": "PR_SU-8 Photoresist",
                "thickness_um": 1.0 + 0.5 * idx,
            },
            {
                "action": "operation",
                "operation": "Lithography",
                "material": "PR_SU-8 Photoresist",
                "windows": [(0.15 + 0.02 * idx, 0.28 + 0.02 * idx), (0.55, 0.68)],
                "invert_lithography": False,
            },
            {
                "action": "operation",
                "operation": "Etch",
                "material": "PR_SU-8 Photoresist",
                "depth_um": 2.5 * (idx + 1),
                "etch_mode": "anisotropic",
            },
            {
                "action": "operation",
                "operation": "Deposit",
                "material": "Gold" if idx % 2 == 0 else "Chromium",
                "thickness_um": 0.025 * (idx + 1),
            },
        ])

    _app, win = _new_window()
    for command in commands:
        ok, message = win._execute_assistant_command(command)
        if not ok:
            raise AssertionError(f"many-step visual consistency command failed: {message}")

    if len(win.history) < 20:
        raise AssertionError(f"Stress process should contain many history states, got {len(win.history)}")
    _assert_process_wide_material_display_counts(win)
    return None


def test_rebuild_user_testprocess():
    _app, win = _new_window()
    data = load_process_file(FIXTURES_DIR / "testprocess.fab")
    steps = win.process_steps_from_history_metadata(data.get("history", []))
    if not steps:
        raise AssertionError("Could not derive process steps from testprocess.fab")

    for material in data.get("material_names", []):
        if material and material != "Void":
            win._execute_assistant_command({"action": "add_material", "material": material})
    ok, message = win._execute_assistant_command({
        "action": "set_substrate",
        "material": data.get("substrate_name", BASE_SUBSTRATE_MATERIAL),
        "thickness_um": float(data.get("substrate_thickness", 525) or 525),
    })
    if not ok:
        raise AssertionError(f"testprocess substrate setup failed: {message}")
    substrate_material = _selected_substrate_material(
        win,
        data.get("substrate_name", BASE_SUBSTRATE_MATERIAL),
    )
    for step in steps:
        command = {
            "action": "operation",
            "operation": step.get("operation", ""),
            "material": step.get("material_name", ""),
        }
        for key in ("thickness_um", "depth_um", "etch_mode", "method", "windows", "invert_lithography"):
            if key in step:
                command[key] = step[key]
        ok, message = win._execute_assistant_command(command)
        if not ok:
            raise AssertionError(f"testprocess rebuild failed: {message}")

    path = VALIDATION_OUTPUT_DIR / "FabStack_dimtest_rebuilt_testprocess.fab"
    win._save_process_to_path(str(path))
    second_spin = _state_by_operation(win.history, "Spincoating", occurrence=2)
    stats = _material_stats(second_spin, win.material_names, "PR_SU-8 Photoresist")
    _assert_planarized_spincoat(stats, 10.0, second_spin.get("title", ""))
    final_oxide_etch = _state_by_operation(win.history, "Etch", occurrence=2)
    _assert_stats(win, final_oxide_etch, "Silicon Dioxide (SiO2)", 0.1, count=90)
    _assert_consistent_display_profile(win, substrate_material=substrate_material)
    gold_deposit = _state_by_operation(win.history, "Deposit", occurrence=2)
    _assert_gap_layer_visual_relation(
        win,
        gold_deposit,
        ["Chromium", "Gold"],
        substrate_material=substrate_material,
    )
    oxide_deposit = _state_by_operation(win.history, "Deposit", occurrence=3)
    _assert_gap_layer_visual_relation(
        win,
        oxide_deposit,
        ["Chromium", "Gold", "Silicon Dioxide (SiO2)"],
        substrate_material=substrate_material,
    )
    for state in (
        oxide_deposit,
        _state_by_operation(win.history, "Lithography", occurrence=2),
        final_oxide_etch,
        win.history[-1],
    ):
        _assert_material_visual_order_in_gap(
            win,
            state,
            "Gold",
            "Silicon Dioxide (SiO2)",
            substrate_material=substrate_material,
        )
    return path


def test_saved_testprocess_display_profile():
    _app, win = _new_window()
    path = FIXTURES_DIR / "testprocess.fab"
    data = load_process_file(path)
    win.material_names = data.get("material_names", win.material_names)
    win.material_colors = data.get("material_colors", win.material_colors)
    win.VOID_INDEX = data.get("void_index", win.VOID_INDEX)
    win.history = data.get("history", [])
    if not win.history:
        raise AssertionError("testprocess.fab has no saved history")
    _assert_consistent_display_profile(
        win,
        substrate_material=data.get("substrate_name", BASE_SUBSTRATE_MATERIAL),
    )
    return path


def test_creation_vt_controls_removed():
    _app, win = _new_window()
    removed = [
        "substrate_vt_spin",
        "substrate_auto_vt_preview_label",
        "vt_spin",
        "vt_row_widget",
        "auto_vt_preview_label",
    ]
    present = [name for name in removed if hasattr(win, name)]
    if present:
        raise AssertionError(f"Creation-time VT controls should be removed, still present: {present}")
    if win.visualisation_mode != "dynamic":
        raise AssertionError(f"New process should start in dynamic mode, got {win.visualisation_mode!r}")
    if any(button is not None for button in (
        win.finalise_visualisation_button,
        win.accept_visualisation_button,
        win.reject_visualisation_button,
    )):
        raise AssertionError("Finalise/Accept/Reject controls should be removed from the UI")
    if win.adjust_visual_thickness_button.text() != "Adjust display height…":
        raise AssertionError("The display-height control has the wrong label")
    if win.adjust_visual_thickness_button.isHidden():
        raise AssertionError("Adjust display should remain visible beside the substrate controls")
    win.adjust_visual_thickness_button.click()
    if not win.visual_thickness_pick_mode:
        raise AssertionError("Adjust display should directly enable canvas picking")
    if win.visualisation_mode != "dynamic":
        raise AssertionError("Entering display pick mode alone must not freeze the visualisation")


def test_feature_direct_edit_applies_persists_and_cancel_rolls_back():
    _app, win = _new_window()
    for command in (
        {"action": "add_material", "material": "Gold"},
        {"action": "operation", "operation": "Deposit", "material": "Gold", "thickness_um": 2.0},
    ):
        ok, message = win._execute_assistant_command(command)
        if not ok:
            raise AssertionError(f"Feature direct-edit setup failed: {message}")

    state = win.history[-1]
    history_index = len(win.history) - 1
    gold_index = win.material_names.index("Gold")
    display = win._display_grid_auto(
        np.asarray(state.get("grid")),
        row_heights_um=np.asarray(state.get("row_heights_um"), dtype=float),
        history_index=history_index,
    )
    current_rows = int(np.sum(display[:, 0] == gold_index))
    target_rows = current_rows + 7
    key = win._display_material_count_profile_key(gold_index, 2.0)
    pick = {
        "key": key,
        "material_idx": gold_index,
        "material_name": "Gold",
        "height_um": 2.0,
        "current_px": current_rows,
        "history_index": history_index,
    }
    physical_before = np.asarray(state.get("row_heights_um"), dtype=float).copy()

    original_exec = QDialog.exec_

    def apply_feature(dialog):
        if dialog.windowTitle() != "Adjust display height":
            raise AssertionError(f"Direct visual editor has the wrong title: {dialog.windowTitle()!r}")
        spin = dialog.findChild(QSpinBox)
        apply_button = next(
            button for button in dialog.findChildren(QPushButton)
            if button.text() == "Save"
        )
        spin.setValue(target_rows)
        apply_button.click()
        return dialog.result()

    QDialog.exec_ = apply_feature
    try:
        win._show_visual_thickness_pick_dialog(pick)
    finally:
        QDialog.exec_ = original_exec

    if win._normalised_visualisation_mode() != "finalised":
        raise AssertionError("Applying Feature should persist the selected display override")
    if int(win.visual_thickness_overrides.get(key, 0)) != target_rows:
        raise AssertionError(
            f"Feature did not store the selected display height: {win.visual_thickness_overrides}"
        )
    adjusted = win._display_grid_auto(
        np.asarray(state.get("grid")),
        row_heights_um=np.asarray(state.get("row_heights_um"), dtype=float),
        history_index=history_index,
    )
    if int(np.sum(adjusted[:, 0] == gold_index)) != target_rows:
        raise AssertionError("Feature override was stored but not rendered")
    if not np.array_equal(
        physical_before,
        np.asarray(state.get("row_heights_um"), dtype=float),
    ):
        raise AssertionError("Feature must never change physical process thicknesses")

    data = win._make_process_dict()
    _app, loaded = _new_window()
    loaded._load_process_dict(data)
    if (
        loaded._normalised_visualisation_mode() != "finalised"
        or int(loaded.visual_thickness_overrides.get(key, 0)) != target_rows
    ):
        raise AssertionError("Feature override did not survive project save/load")

    win.reject_visualisation()
    if win._normalised_visualisation_mode() != "dynamic":
        raise AssertionError("Test setup could not restore dynamic mode")

    def cancel_feature(dialog):
        dialog.reject()
        return dialog.result()

    QDialog.exec_ = cancel_feature
    try:
        win._show_visual_thickness_pick_dialog(pick)
    finally:
        QDialog.exec_ = original_exec
    if win._normalised_visualisation_mode() != "dynamic" or win.visual_thickness_overrides:
        raise AssertionError("Cancelling Feature must restore the original dynamic visualisation")


def test_finalise_freezes_accepts_and_resets_on_process_change():
    _app, win = _new_window()
    before_heights = [np.asarray(state.get("row_heights_um"), dtype=float).copy() for state in win.history]

    for command in [
        {"action": "add_material", "material": "Gold"},
        {"action": "operation", "operation": "Deposit", "material": "Gold", "thickness_um": 2.0},
    ]:
        ok, message = win._execute_assistant_command(command)
        if not ok:
            raise AssertionError(f"finalise setup command failed: {message}")

    state = win.history[-1]
    gold_index = win.material_names.index("Gold")
    history_index = len(win.history) - 1
    dynamic_display = win._display_grid_auto(
        np.asarray(state.get("grid")),
        row_heights_um=np.asarray(state.get("row_heights_um"), dtype=float),
        history_index=history_index,
    )
    dynamic_gold_rows = int(np.sum(dynamic_display[:, 0] == gold_index))

    win.finalise_visualisation()
    if win.visualisation_mode != "pending":
        raise AssertionError(f"Finalise should enter pending mode, got {win.visualisation_mode!r}")
    key = win._display_material_count_profile_key(gold_index, 2.0)
    if win.visual_thickness_overrides.get(key) != dynamic_gold_rows:
        raise AssertionError(
            f"Finalise should freeze Gold at {dynamic_gold_rows}px, got {win.visual_thickness_overrides.get(key)}"
        )
    frozen_display = win._display_grid_auto(
        np.asarray(state.get("grid")),
        row_heights_um=np.asarray(state.get("row_heights_um"), dtype=float),
        history_index=history_index,
    )
    if int(np.sum(frozen_display[:, 0] == gold_index)) != dynamic_gold_rows:
        raise AssertionError("Frozen display should initially match dynamic display")

    win._set_process_visual_override(key, 44)
    adjusted_display = win._display_grid_auto(
        np.asarray(state.get("grid")),
        row_heights_um=np.asarray(state.get("row_heights_um"), dtype=float),
        history_index=history_index,
    )
    if int(np.sum(adjusted_display[:, 0] == gold_index)) != 44:
        raise AssertionError("Finalised visual override should change display rows only")
    for before, after_state in zip(before_heights, win.history):
        after = np.asarray(after_state.get("row_heights_um"), dtype=float)
        if before.shape == after.shape and not np.allclose(before, after, atol=0.0, rtol=0.0):
            raise AssertionError("Visual finalisation should not mutate physical row heights")

    win.accept_visualisation()
    if win.visualisation_mode != "finalised":
        raise AssertionError(f"Accept should store finalised mode, got {win.visualisation_mode!r}")
    data = win._make_process_dict()
    if data.get("visualisation_mode") != "finalised" or data.get("visual_thickness_overrides", {}).get(key) != 44:
        raise AssertionError("Accepted finalised visualisation should persist mode and overrides")

    ok, message = win._execute_assistant_command(
        {"action": "operation", "operation": "Deposit", "material": "Gold", "thickness_um": 1.0}
    )
    if not ok:
        raise AssertionError(f"process-change reset command failed: {message}")
    if win.visualisation_mode != "dynamic" or win.visual_thickness_overrides:
        raise AssertionError("Adding a process step after finalise should clear overrides and return to dynamic mode")


def test_finalise_allows_etch_void_override_and_reject():
    commands = [
        {"action": "add_material", "material": "PR_SU-8 Photoresist"},
        {"action": "operation", "operation": "Spincoating", "material": "PR_SU-8 Photoresist", "thickness_um": 5.0},
        {
            "action": "operation",
            "operation": "Lithography",
            "material": "PR_SU-8 Photoresist",
            "windows": [(0.40, 0.60)],
            "invert_lithography": False,
        },
        {"action": "operation", "operation": "Etch", "material": "Silicon (Si)", "depth_um": 30.0, "etch_mode": "anisotropic"},
    ]
    _app, win = _new_window()
    for command in commands:
        ok, message = win._execute_assistant_command(command)
        if not ok:
            raise AssertionError(f"finalise etch setup command failed: {message}")

    win.finalise_visualisation()
    if win.visualisation_mode != "pending":
        raise AssertionError("Etch finalise should enter pending mode")

    state = win.history[-1]
    history_index = len(win.history) - 1

    def center_void_rows():
        display_grid = win._display_grid_auto(
            np.asarray(state.get("grid")),
            row_heights_um=np.asarray(state.get("row_heights_um"), dtype=float),
            history_index=history_index,
        )
        info = getattr(win, "_last_display_remap_info", {}) or {}
        columns = info.get("column_segments") or []
        center_void = next(
            (seg for seg in columns[150]
             if int(seg.get("material", win.VOID_INDEX)) == win.VOID_INDEX
             and bool(seg.get("in_substrate_recess", False))),
            None,
        )
        if center_void is None:
            raise AssertionError("Etched column should have a selectable void segment")
        return display_grid, center_void, int(center_void["display_end"]) - int(center_void["display_start"])

    display_grid, center_void, auto_rows = center_void_rows()
    expected_key = win._display_material_count_profile_key(win.VOID_INDEX, 30.0)
    if win.visual_thickness_overrides.get(expected_key) != auto_rows:
        raise AssertionError("Finalise should freeze the etched void span")

    win._last_main_display_grid = np.asarray(display_grid).copy()
    win._last_main_display_history_index = history_index
    win._last_main_display_remap_info = dict(getattr(win, "_last_display_remap_info", {}) or {})
    event = type("Evt", (), {})()
    event.inaxes = win.canvas.ax
    event.xdata = 150.0
    event.ydata = win._plot_y_for_visual_scale(
        (float(center_void["display_start"]) + float(center_void["display_end"])) / 2.0,
        int(display_grid.shape[0]),
        win._visual_scale_factor_clamped(),
    )
    pick = win._visual_layer_pick_from_event(event)
    if not pick or pick.get("key") != expected_key or pick.get("material_idx") != win.VOID_INDEX:
        raise AssertionError(f"Canvas picking should support etched void segments, got {pick}")

    win._set_process_visual_override(expected_key, 175)
    _display_grid, _center_void, rows_after_override = center_void_rows()
    if rows_after_override != 175:
        raise AssertionError(f"Etched void override should render at 175px, got {rows_after_override}")

    win.reject_visualisation()
    if win.visualisation_mode != "dynamic" or win.visual_thickness_overrides:
        raise AssertionError("Reject should clear etch visual overrides and return to dynamic mode")


def test_legacy_visual_overrides_ignored_without_mode():
    _app, win = _new_window()
    data = win._make_process_dict()
    substrate_index = win.material_names.index(win.substrate_combo.currentText())
    key = win._display_material_count_profile_key(substrate_index, float(win.substrate_thick_spin.value()))
    data.pop("visualisation_mode", None)
    data["visual_thickness_overrides"] = {key: 333}
    data["visual_thickness_step_overrides"] = {"0": {key: 333}}
    win._load_process_dict(data)
    if win.visualisation_mode != "dynamic":
        raise AssertionError("Legacy files without visualisation_mode should load in dynamic mode")
    if win.visual_thickness_overrides or win.visual_thickness_step_overrides:
        raise AssertionError("Legacy visual overrides should be ignored when visualisation_mode is missing")


def test_global_gap_stack_constraints_and_identical_bottoms():
    """Hard acceptance for atomic, aggregate, and globally shared gap order."""
    _app, win = _new_window()
    win.material_names = [
        "Void", "Silicon (Si)", "A", "B", "C", "D", "E", "F",
    ]
    win.material_colors = ["#ffffff"] * len(win.material_names)
    win.xsec = CrossSection(
        width_px=1,
        height_px=575,
        substrate_index=1,
        void_index=0,
        substrate_thickness_px=525,
        substrate_thickness_um=525.0,
        top_headroom_um=50.0,
    )

    def state(materials, heights):
        return {
            "grid": np.asarray(materials, dtype=int).reshape(-1, 1),
            "row_heights_um": np.asarray(heights, dtype=float),
            "visual_geometry": [],
        }

    # 20um gap above a 100/10/1nm local stack.
    shallow_stack = state(
        [2, 3, 4, 0, 1],
        [0.1, 0.01, 0.001, 19.889, 505.0],
    )
    # Mixed case: each 6um layer is smaller than a 10um gap, while their
    # contiguous 12um stack is larger than that same gap.
    mixed_stack = state([5, 6, 0, 1], [6.0, 6.0, 1.0, 515.0])
    # Same 20um substrate depth under different local films: its bottom must
    # be identical to shallow_stack, not corrected per column/context.
    same_depth_other_stack = state([7, 0, 1], [1.0, 19.0, 505.0])
    # Reverse atomic order: a 100nm layer is larger than a 50nm gap.
    reverse_atomic = state([2, 3, 0, 1], [0.1, 0.01, 0.01, 524.95])
    win.history = [shallow_stack, mixed_stack, same_depth_other_stack, reverse_atomic]

    profile = win._process_display_adjustment_profile()
    if not profile.get("constraint_converged") or profile.get("constraint_violations"):
        raise AssertionError(f"Constraint solver did not produce an audited fixed point: {profile}")
    nominal_rows = int(profile.get("nominal_substrate_rows", 0))
    compact_upper_bound = 180 + 2 * int(ORDER_MARGIN_PX)
    if not 175 <= nominal_rows <= compact_upper_bound:
        raise AssertionError(
            f"Normal 525um scenes should stay near the compact 175-row substrate while preserving "
            f"strict substrate-depth order, got {nominal_rows} (limit {compact_upper_bound})"
        )

    def rows(height_um):
        key = win._display_feature_profile_key(height_um)
        return int(profile["resolved_feature_rows"][key])

    nano_stack_rows = rows(0.1) + rows(0.01) + rows(0.001)
    if rows(20.0) < nano_stack_rows + int(ORDER_MARGIN_PX):
        raise AssertionError(
            f"20um gap must visibly exceed its 0.111um local stack: {rows(20.0)} vs {nano_stack_rows}"
        )
    if rows(10.0) < rows(6.0) + int(ORDER_MARGIN_PX):
        raise AssertionError("10um gap must exceed each local 6um atomic layer")
    if 2 * rows(6.0) < rows(10.0) + int(ORDER_MARGIN_PX):
        raise AssertionError("The local 12um stack must exceed the 10um gap")
    if rows(0.1) < rows(0.05) + int(ORDER_MARGIN_PX):
        raise AssertionError("100nm layer must visibly exceed a 50nm gap")

    core_rows = []
    for history_index, source in ((0, shallow_stack), (2, same_depth_other_stack)):
        display = win._display_grid_auto(
            source["grid"],
            row_heights_um=source["row_heights_um"],
            history_index=history_index,
            process_profile=profile,
        )
        core_rows.append(int(np.sum(display[:, 0] == 1)))
    if core_rows[0] != core_rows[1]:
        raise AssertionError(
            f"Equal 20um etch depths must share one substrate bottom, got core rows {core_rows}"
        )


def test_one_nm_band_has_six_rendered_device_pixels():
    """Validate visibility after the real axis/canvas transform, not just rows."""
    _app, win = _new_window()
    win.material_names = ["Void", "Silicon (Si)", "1 nm marker"]
    win.material_colors = ["#ffffff", "#aaaaaa", "#ff0000"]
    xsec = CrossSection(
        width_px=80,
        height_px=575,
        substrate_index=1,
        void_index=0,
        substrate_thickness_px=525,
        substrate_thickness_um=525.0,
        top_headroom_um=50.0,
    )
    xsec.deposit_physical(2, 0.001)
    source = {
        "grid": xsec.copy_state(),
        "row_heights_um": xsec.copy_row_heights(),
        "visual_geometry": xsec.copy_visual_geometry(),
    }
    win.xsec = xsec
    win.history = [source]
    profile = win._process_display_adjustment_profile()
    win.canvas.ax.clear()
    win.plot_grid(
        win.canvas.ax,
        source["grid"],
        main_axis=True,
        history_index=0,
        row_heights_um=source["row_heights_um"],
        process_profile=profile,
    )
    win.canvas.draw()

    info = win._last_main_display_remap_info
    marker = next(
        segment for segment in info["column_segments"][0]
        if int(segment.get("material", -1)) == 2
    )
    display_height = int(info["display_height"])
    y0 = win._plot_y_for_visual_scale(
        float(marker["display_start"]) - 0.5,
        display_height,
    )
    y1 = win._plot_y_for_visual_scale(
        float(marker["display_end"]) - 0.5,
        display_height,
    )
    device_y0 = win.canvas.ax.transData.transform((0.0, y0))[1]
    device_y1 = win.canvas.ax.transData.transform((0.0, y1))[1]
    visible_pixels = abs(float(device_y1) - float(device_y0))
    if visible_pixels < 6.0:
        raise AssertionError(
            f"A 1nm feature must occupy at least six rendered device pixels, got {visible_pixels:g}"
        )


def main():
    _patch_message_boxes()
    tests = [
        test_isotropic_physical_front_geometry,
        test_adaptive_display_layer_order,
        test_visual_scale_control_points,
        test_feature_emphasis_control_range_and_persistence,
        test_deep_partial_etch_preserves_visible_substrate_core,
        test_near_tie_feature_order_is_visible_without_exaggeration,
        test_process_wide_scaling_reaches_previous_steps,
        test_finalised_visual_override_scope_consistency,
        test_substrate_change_rebuilds_previous_steps,
        test_assistant_set_substrate_rebuilds_without_clearing_history,
        test_assistant_set_substrate_material_rebuilds_history,
        test_anisotropic_etch_cutout_has_no_false_material_layer,
        test_substrate_visual_override_preserves_etched_drop,
        test_deposit_after_etch_remains_visible_in_trench,
        test_visual_thickness_pick_deposit_inside_etched_trench,
        test_etch_depth_display_after_spincoat,
        test_etch_depth_matches_equal_deposit_display_height,
        test_etch_geometry_display_stress_every_step,
        test_multiscale_trench_fill_layer_order,
        test_power_scale_population_independent_order_sweep_deposit_and_spincoat,
        test_substrate_anchored_20um_etch_beats_nm_film_stack,
        test_readability_band_segmentation_error_is_bounded,
        test_shallow_isotropic_etch,
        test_spincoat_over_topography,
        test_spincoat_fills_recess_and_preserves_planar_interface,
        test_bond_after_planarized_recess_preserves_support_plane,
        test_second_lithography_preserves_prior_planarizer_plane,
        test_rigid_bond_planes_persist_through_descendants,
        test_partial_etch_of_planarizer_keeps_real_step,
        test_flipped_planarizer_preserves_unequal_fill_depths,
        test_planar_membrane_uses_real_support_void_after_flip_back,
        test_bond_is_rigid_structural_body_with_clearance,
        test_rigid_bond_stays_flat_over_multi_etch_film_topography,
        test_bond_equal_contact_composites_are_resolved_process_wide,
        test_bond_stays_flat_over_complete_substrate_through_hole,
        test_full_removal_tracks_final_ordered_body_budget,
        test_equal_nominal_bodies_share_final_display_budget,
        test_body_ordering_enters_equal_nominal_group,
        test_bonded_substrate_etch_uses_body_local_core_scaling,
        test_external_multibody_bond_preserves_roles_and_buried_etch_scale,
        test_external_bond_remaps_planarizer_metadata_once,
        test_cross_body_etch_compares_complete_step_to_layers,
        test_buried_cavity_is_not_a_cumulative_surface_step,
        test_nonstructural_interlayer_controls_cumulative_connectivity,
        test_distinct_open_interlayers_keep_component_scaling,
        test_same_material_open_interlayer_uses_local_reference_merge,
        test_multiple_interface_regions_use_nearest_local_reference,
        test_spincoat_does_not_plug_an_unsupported_through_hole,
        test_inverse_lithography_after_topography,
        test_sub_nm_deposit_and_etch,
        test_flip_then_process,
        test_gap_layer_display_consistency_stress,
        test_process_wide_visual_consistency_many_steps,
        test_rebuild_user_testprocess,
        test_saved_testprocess_display_profile,
        test_creation_vt_controls_removed,
        test_feature_direct_edit_applies_persists_and_cancel_rolls_back,
        test_finalise_freezes_accepts_and_resets_on_process_change,
        test_finalise_allows_etch_void_override_and_reject,
        test_legacy_visual_overrides_ignored_without_mode,
        test_global_gap_stack_constraints_and_identical_bottoms,
        test_one_nm_band_has_six_rendered_device_pixels,
    ]
    saved = []
    for test in tests:
        path = test()
        if path is not None:
            saved.append(path.name)
            print(f"PASS {test.__name__}: {path.name}")
        else:
            print(f"PASS {test.__name__}")
    print("Saved mocks:")
    for name in saved:
        print(f"- {name}")
    print(f"Validation artifacts: {VALIDATION_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
