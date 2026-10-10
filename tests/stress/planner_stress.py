"""Stress test for FabStack local LLM command planning.

Runs human-quality prompts through LocalStepPlanner and checks that the
returned command plan is complete or asks for the right missing detail.
"""

import argparse
import json
from pathlib import Path
from types import MethodType

from fabstack.assistant.planner import LocalStepPlanner


ROOT = Path(__file__).resolve().parents[2]


def load_materials():
    data = json.loads(
        (ROOT / "fabstack" / "assets" / "materials_library.json").read_text()
    )
    materials = data["materials"]
    library = [
        material
        for names in materials.values()
        if isinstance(names, list)
        for material in names
    ]
    return materials, library


def build_context(library, photoresists, active=None, thickness=None, order=None, substrate="Silicon (Si) (200 um)"):
    active = active or ["Silicon (Si)"]
    thickness = thickness if thickness is not None else {"Silicon (Si)": 200.0}
    order = order if order is not None else list(reversed(active))
    return {
        "substrate": substrate,
        "active_materials": active,
        "library_materials": library,
        "photoresist_materials": photoresists,
        "process_flow": [],
        "material_thickness_um": thickness,
        "top_material_order": order,
    }


def make_cases(library, photoresists):
    base = build_context(library, photoresists)
    stack = build_context(
        library,
        photoresists,
        active=["Silicon (Si)", "Titanium", "Chromium", "Gold", "PR_SU-8 Photoresist", "MEMpax"],
        thickness={
            "Silicon (Si)": 525.0,
            "Titanium": 0.01,
            "Chromium": 0.01,
            "Gold": 0.1,
            "PR_SU-8 Photoresist": 2.0,
            "MEMpax": 525.0,
        },
        order=["PR_SU-8 Photoresist", "Gold", "Chromium", "Titanium", "Silicon (Si)"],
        substrate="Silicon (Si) (525 um)",
    )
    pr_ctx = build_context(
        library,
        photoresists,
        active=["Silicon (Si)", "PR_SU-8 Photoresist"],
        thickness={"Silicon (Si)": 200.0, "PR_SU-8 Photoresist": 2.0},
        order=["PR_SU-8 Photoresist", "Silicon (Si)"],
    )
    thick_silicon = build_context(
        library,
        photoresists,
        active=["Silicon (Si)"],
        thickness={"Silicon (Si)": 525.0},
        order=["Silicon (Si)"],
    )

    return [
        {
            "name": "target_missing_su8",
            "prompt": (
                "hey, make a stack for me. first on a 525 um wafer, deposit 10 nm of titanium, "
                "then 100 nm of gold. then spincoat su8 and make two windows 0.2 far from the "
                "edges. then etch titanium and gold all the way through. then etch 200 um the "
                "silicon. finally strip the photoresist and bond the stack with a 525 mempax substrate"
            ),
            "ctx": base,
            "status": "needs_clarification",
            "missing": ["PR_SU-8 Photoresist"],
        },
        {
            "name": "target_complete",
            "prompt": (
                "on a 525 um wafer, deposit 10 nm titanium and 100 nm gold, spincoat su8 5 um, "
                "make two windows 0.2 far from the edges, etch titanium and gold all the way "
                "through, etch 200 um silicon, strip the photoresist, bond with a 525 mempax substrate"
            ),
            "ctx": base,
            "status": "ready",
            "expect": [
                ("set_substrate", "", "Silicon (Si)"),
                ("operation", "Deposit", "Titanium"),
                ("operation", "Deposit", "Gold"),
                ("operation", "Spincoating", "PR_SU-8 Photoresist"),
                ("operation", "Lithography", "PR_SU-8 Photoresist"),
                ("operation", "Etch", "Gold"),
                ("operation", "Etch", "Titanium"),
                ("operation", "Etch", "Silicon (Si)"),
                ("operation", "Strip material", "PR_SU-8 Photoresist"),
                ("operation", "Bond", "MEMpax"),
            ],
            "values": [
                ("set_substrate", "", "Silicon (Si)", "thickness_um", 525.0),
                ("operation", "Deposit", "Titanium", "thickness_um", 0.01),
                ("operation", "Deposit", "Gold", "thickness_um", 0.1),
                ("operation", "Spincoating", "PR_SU-8 Photoresist", "thickness_um", 5.0),
                ("operation", "Etch", "Gold", "depth_um", 0.1),
                ("operation", "Etch", "Titanium", "depth_um", 0.01),
                ("operation", "Etch", "Silicon (Si)", "depth_um", 200.0),
                ("operation", "Bond", "MEMpax", "thickness_um", 525.0),
            ],
        },
        {
            "name": "chrome_gold_resist",
            "prompt": "Deposit 10 nm of chrome, then deposit 100 nm of gold, then spincoat 1 um of photoresist material",
            "ctx": base,
            "status": "ready",
            "expect": [
                ("operation", "Deposit", "Chromium"),
                ("operation", "Deposit", "Gold"),
                ("operation", "Spincoating", "Photoresist"),
            ],
        },
        {
            "name": "complete_and_half_etch",
            "prompt": "etch completely gold and half the chrome",
            "ctx": stack,
            "status": "ready",
            "expect": [("operation", "Etch", "Gold"), ("operation", "Etch", "Chromium")],
            "values": [
                ("operation", "Etch", "Gold", "depth_um", 0.1),
                ("operation", "Etch", "Chromium", "depth_um", 0.005),
            ],
        },
        {
            "name": "unit_reuse_decimal_comma",
            "prompt": "etch gold of 0.1 um and 0,01 chrome",
            "ctx": stack,
            "status": "ready",
            "expect": [("operation", "Etch", "Gold"), ("operation", "Etch", "Chromium")],
            "values": [
                ("operation", "Etch", "Gold", "depth_um", 0.1),
                ("operation", "Etch", "Chromium", "depth_um", 0.01),
            ],
        },
        {
            "name": "edge_windows_existing_pr",
            "prompt": "make two windows 0.2 far from the edges",
            "ctx": pr_ctx,
            "status": "ready",
            "expect": [("operation", "Lithography", "PR_SU-8 Photoresist")],
            "windows": [(0.0, 0.2), (0.8, 1.0)],
        },
        {
            "name": "bond_unitless_substrate",
            "prompt": "bond the stack with a 525 mempax substrate",
            "ctx": stack,
            "status": "ready",
            "expect": [("operation", "Bond", "MEMpax")],
        },
        {"name": "spincoat_missing_thickness", "prompt": "spincoat su8", "ctx": base, "status": "needs_clarification", "missing": ["thickness"]},
        {"name": "deposit_missing_thickness", "prompt": "deposit gold", "ctx": base, "status": "needs_clarification", "missing": ["thickness"]},
        {"name": "complete_etch_missing_context", "prompt": "etch gold all the way through", "ctx": base, "status": "needs_clarification", "missing": ["current deposited thickness"]},
        {
            "name": "glass_substrate_aluminum",
            "prompt": "on a 525 um glass substrate deposit 50 nm aluminum",
            "ctx": base,
            "status": "ready",
            "expect": [("set_substrate", "", "Glass"), ("operation", "Deposit", "Aluminum")],
        },
        {
            "name": "compound_materials_prefer_full_name",
            "prompt": (
                "deposit 100 nm aluminium nitide, then deposit 50 nm titanium nitride, "
                "then deposit 25 nm aluminum"
            ),
            "ctx": base,
            "status": "ready",
            "expect": [
                ("operation", "Deposit", "Aluminum Nitride (AlN)"),
                ("operation", "Deposit", "Titanium Nitride (TiN)"),
                ("operation", "Deposit", "Aluminum"),
            ],
            "values": [
                ("operation", "Deposit", "Aluminum Nitride (AlN)", "thickness_um", 0.1),
                ("operation", "Deposit", "Titanium Nitride (TiN)", "thickness_um", 0.05),
                ("operation", "Deposit", "Aluminum", "thickness_um", 0.025),
            ],
        },
        {
            "name": "implicit_deposit_two_layers",
            "prompt": "first 10 nm titanium and then 100 nm gold",
            "ctx": base,
            "status": "ready",
            "expect": [("operation", "Deposit", "Titanium"), ("operation", "Deposit", "Gold")],
        },
        {
            "name": "spin_strip_pronoun",
            "prompt": "spincoat 2 um SU8 then strip it",
            "ctx": base,
            "status": "ready",
            "expect": [
                ("operation", "Spincoating", "PR_SU-8 Photoresist"),
                ("operation", "Strip material", "PR_SU-8 Photoresist"),
            ],
        },
        {
            "name": "litho_range_existing_pr",
            "prompt": "litho window 0.35 to 0.65",
            "ctx": pr_ctx,
            "status": "ready",
            "expect": [("operation", "Lithography", "PR_SU-8 Photoresist")],
            "windows": [(0.35, 0.65)],
        },
        {
            "name": "manual_lithography_pause",
            "prompt": "spincoat su8 2 um then custom litho window then etch silicon 5 um",
            "ctx": base,
            "status": "ready",
            "expect": [
                ("operation", "Spincoating", "PR_SU-8 Photoresist"),
                ("operation", "Lithography", "PR_SU-8 Photoresist"),
                ("operation", "Etch", "Silicon (Si)"),
            ],
            "values": [
                ("operation", "Spincoating", "PR_SU-8 Photoresist", "thickness_um", 2.0),
                ("operation", "Lithography", "PR_SU-8 Photoresist", "windowing_mode", "manual"),
                ("operation", "Etch", "Silicon (Si)", "depth_um", 5.0),
            ],
        },
        {
            "name": "oxide_deposit",
            "prompt": "deposit 500 nm oxide on silicon",
            "ctx": base,
            "status": "ready",
            "expect": [("operation", "Deposit", "Silicon Dioxide (SiO2)")],
        },
        {
            "name": "drie_silicon",
            "prompt": "DRIE silicon 50 um",
            "ctx": base,
            "status": "ready",
            "expect": [("operation", "Etch", "Silicon (Si)")],
            "values": [("operation", "Etch", "Silicon (Si)", "etch_mode", "anisotropic")],
        },
        {
            "name": "remaining_depth_fill_after_drie",
            "prompt": (
                "hey lets make a stack. first deposit photoresist su8 for 2um. "
                "then make a litho of 0.5 in the center. then DRIE etching of silicon for 100 um. "
                "then deposit 20 nm of aluminium and 100 nm of gold. "
                "then the remaining depth of the etching filled with silicon oxide. "
                "strip the photoresist and bond to a mempax of 525 um"
            ),
            "ctx": base,
            "status": "ready",
            "expect": [
                ("operation", "Spincoating", "PR_SU-8 Photoresist"),
                ("operation", "Lithography", "PR_SU-8 Photoresist"),
                ("operation", "Etch", "Silicon (Si)"),
                ("operation", "Deposit", "Aluminum"),
                ("operation", "Deposit", "Gold"),
                ("operation", "Deposit", "Silicon Dioxide (SiO2)"),
                ("operation", "Strip material", "PR_SU-8 Photoresist"),
                ("operation", "Bond", "MEMpax"),
            ],
            "values": [
                ("operation", "Spincoating", "PR_SU-8 Photoresist", "thickness_um", 2.0),
                ("operation", "Etch", "Silicon (Si)", "depth_um", 100.0),
                ("operation", "Etch", "Silicon (Si)", "method", "DRIE (Bosch process)"),
                ("operation", "Deposit", "Aluminum", "thickness_um", 0.02),
                ("operation", "Deposit", "Gold", "thickness_um", 0.1),
                ("operation", "Deposit", "Silicon Dioxide (SiO2)", "thickness_um", 99.88),
                ("operation", "Bond", "MEMpax", "thickness_um", 525.0),
            ],
            "windows": [(0.25, 0.75)],
        },
        {"name": "bond_missing_thickness", "prompt": "bond glass", "ctx": base, "status": "needs_clarification", "missing": ["thickness"]},
        {
            "name": "set_substrate_quartz_gold",
            "prompt": "set substrate quartz 300 um and deposit 100 nm gold",
            "ctx": base,
            "status": "ready",
            "expect": [("set_substrate", "", "Quartz"), ("operation", "Deposit", "Gold")],
        },
        {
            "name": "spin_pmma_liftoff",
            "prompt": "spincoat PMMA 2 um then lift off",
            "ctx": base,
            "status": "ready",
            "expect": [
                ("operation", "Spincoating", "PMMA (Poly(methyl methacrylate))"),
                ("operation", "Lift-off", "PMMA (Poly(methyl methacrylate))"),
            ],
        },
        {
            "name": "half_silicon",
            "prompt": "etch half silicon",
            "ctx": thick_silicon,
            "status": "ready",
            "expect": [("operation", "Etch", "Silicon (Si)")],
            "values": [("operation", "Etch", "Silicon (Si)", "depth_um", 262.5)],
        },
    ]


def has_command(plan, action, operation, material):
    for command in plan.get("commands", []):
        if command.get("action") != action:
            continue
        if action == "operation" and command.get("operation") != operation:
            continue
        if str(command.get("material", "")).lower() == material.lower():
            return True
    return False


def has_windows(plan, expected):
    expected_rounded = [(round(float(a), 6), round(float(b), 6)) for a, b in expected]
    for command in plan.get("commands", []):
        if command.get("action") == "operation" and command.get("operation") == "Lithography":
            got = [(round(float(a), 6), round(float(b), 6)) for a, b in command.get("windows", [])]
            return got == expected_rounded
    return False


def find_command(plan, action, operation, material):
    for command in plan.get("commands", []):
        if command.get("action") != action:
            continue
        if action == "operation" and command.get("operation") != operation:
            continue
        if str(command.get("material", "")).lower() == material.lower():
            return command
    return None


def value_matches(actual, expected):
    try:
        actual = float(actual)
        expected = float(expected)
    except (TypeError, ValueError):
        return str(actual or "").strip().lower() == str(expected or "").strip().lower()
    return abs(actual - expected) <= max(1e-9, abs(expected) * 1e-6)


def evaluate_case(plan, case):
    ok = plan.get("status") == case["status"]
    notes = []
    if not ok:
        notes.append(f"status {plan.get('status')} != {case['status']}")

    for action, operation, material in case.get("expect", []):
        if not has_command(plan, action, operation, material):
            ok = False
            notes.append(f"missing command {action}:{operation}:{material}")

    for action, operation, material, field, expected in case.get("values", []):
        command = find_command(plan, action, operation, material)
        if not command:
            ok = False
            notes.append(f"missing value command {action}:{operation}:{material}")
            continue
        if not value_matches(command.get(field), expected):
            ok = False
            notes.append(
                f"{action}:{operation}:{material} {field}={command.get(field)!r}, expected {expected!r}"
            )

    missing_text = " | ".join(plan.get("missing", []))
    for expected_missing in case.get("missing", []):
        if expected_missing.lower() not in missing_text.lower():
            ok = False
            notes.append(f"missing clarification containing {expected_missing!r}")

    if case.get("windows") and not has_windows(plan, case["windows"]):
        ok = False
        notes.append("window mismatch")

    return ok, notes


def make_ui_validator():
    from PyQt5.QtWidgets import QApplication
    from fabstack.presentation.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    def validate(plan):
        if plan.get("status") != "ready":
            return []
        return window._validate_assistant_commands(plan.get("commands", []))

    return app, window, validate


def run_manual_lithography_pause_flow(window):
    from PyQt5.QtWidgets import QMessageBox

    prompt = "set substrate silicon 200 um then spincoat su8 2 um then custom litho window then etch silicon 5 um"
    planner = LocalStepPlanner()
    plan = planner._fallback_command_plan(prompt, window._build_assistant_context())
    if plan.get("status") != "ready":
        return False, [f"manual pause plan status={plan.get('status')} missing={plan.get('missing')}"]

    original_question = QMessageBox.question
    original_warning = QMessageBox.warning
    try:
        QMessageBox.question = lambda *args, **kwargs: QMessageBox.Yes
        QMessageBox.warning = lambda *args, **kwargs: QMessageBox.Ok
        window._assistant_pending_plan = plan
        window._assistant_manual_lithography_pause = None
        window._apply_assistant_pending_plan()
        if not window._assistant_is_paused_for_manual_lithography():
            return False, ["runner did not pause for manual lithography"]

        window.add_litho_window_range(0.25, 0.75, show_warning=False, refresh_plot=False)
        window._apply_assistant_pending_plan()
    finally:
        QMessageBox.question = original_question
        QMessageBox.warning = original_warning
        dialog = getattr(window, "window_dialog", None)
        if dialog is not None:
            dialog.close()

    if window._assistant_is_paused_for_manual_lithography():
        return False, ["runner stayed paused after adding windows and continuing"]

    operations = [
        str(item.get("operation", ""))
        for item in getattr(window, "history", [])
        if isinstance(item, dict)
    ]
    missing = [
        operation
        for operation in ("Spincoating", "Lithography", "Etch")
        if operation not in operations
    ]
    if missing:
        return False, [f"history missing operations: {missing}; got {operations}"]
    return True, []


def make_planner(mock_qwen=False, partial_qwen=False):
    planner = LocalStepPlanner()
    if mock_qwen:
        def fake_model(self, _request, _context):
            return json.dumps({
                "status": "needs_clarification",
                "message": "forced repair path",
                "missing": ["forced repair path"],
                "commands": [],
            })

        planner._generate_command_json_with_model = MethodType(fake_model, planner)
    elif partial_qwen:
        def fake_model(self, _request, _context):
            return json.dumps({
                "status": "ready",
                "message": "Commands ready.",
                "missing": [],
                "commands": [
                    {"action": "set_substrate", "material": "Silicon (Si)", "thickness_um": 525.0}
                ],
            })

        planner._generate_command_json_with_model = MethodType(fake_model, planner)
    return planner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock-qwen", action="store_true", help="Force weak model output to test deterministic repair.")
    parser.add_argument(
        "--partial-qwen",
        action="store_true",
        help="Force a dangerous partial ready plan to test request coverage rejection.",
    )
    parser.add_argument(
        "--ui-validate",
        action="store_true",
        help="Also pass ready plans through MainWindow's assistant command validator.",
    )
    args = parser.parse_args()

    materials, library = load_materials()
    cases = make_cases(library, materials["Photoresists"])
    planner = make_planner(mock_qwen=args.mock_qwen, partial_qwen=args.partial_qwen)
    app = None
    window = None
    ui_validate = None
    if args.ui_validate:
        app, window, ui_validate = make_ui_validator()

    passed = 0
    total = len(cases)
    try:
        for index, case in enumerate(cases, 1):
            plan = planner.generate_command_plan(case["prompt"], case["ctx"])
            ok, notes = evaluate_case(plan, case)
            if ui_validate is not None:
                ui_errors = ui_validate(plan)
                if ui_errors:
                    ok = False
                    notes.extend(f"UI validator: {error}" for error in ui_errors)
            if ok:
                passed += 1
            print(
                f"[{index:02d}] {case['name']}: {'PASS' if ok else 'FAIL'} "
                f"status={plan.get('status')} cmds={len(plan.get('commands', []))}",
                flush=True,
            )
            if not ok:
                print("  notes:", "; ".join(notes), flush=True)
                print("  missing:", plan.get("missing"), flush=True)
                for command in plan.get("commands", []):
                    print(" ", command, flush=True)

        if args.ui_validate and window is not None:
            total += 1
            ok, notes = run_manual_lithography_pause_flow(window)
            if ok:
                passed += 1
            print(
                f"[ui] manual_lithography_pause_flow: {'PASS' if ok else 'FAIL'}",
                flush=True,
            )
            if not ok:
                print("  notes:", "; ".join(notes), flush=True)
    finally:
        if window is not None:
            dialog = getattr(window, "window_dialog", None)
            if dialog is not None:
                dialog.close()
            if not args.ui_validate:
                window.close()
        if app is not None:
            app.quit()

    print(f"\nSCORE {passed}/{total} = {passed / total:.0%}", flush=True)
    raise SystemExit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
