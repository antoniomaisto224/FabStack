"""Focused regression tests for production-safety and geometry invariants."""

import ast
import copy
import hashlib
import os
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PyQt5.QtWidgets import QApplication, QMessageBox


_TEST_DATA_DIR = tempfile.mkdtemp(prefix="fabstack-test-data-")
os.environ.setdefault("FABSTACK_DATA_DIR", _TEST_DATA_DIR)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="fabstack-mpl-"))

from fabstack.presentation.mixins.operations_mixin import OperationsMixin  # noqa: E402
from fabstack.presentation.main_window import MainWindow  # noqa: E402
from fabstack.presentation.mixins.process_persistence_mixin import ProcessPersistenceMixin  # noqa: E402
from fabstack.materials.schema import MaterialLibraryError, validate_material_library  # noqa: E402
from fabstack.domain.cross_section import CrossSection  # noqa: E402
from fabstack.persistence.process_file import (  # noqa: E402
    ProcessFileError,
    RestrictedProcessUnpickler,
    load_process_file,
    save_process_file,
)
from fabstack.terminology import OPERATION_LABELS, operation_label  # noqa: E402
from fabstack.version import __version__  # noqa: E402


class _MaliciousPayload:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return (Path.touch, (self.marker,))


class ProcessFileTests(unittest.TestCase):
    def test_round_trip_preserves_numpy_geometry(self):
        xsec = CrossSection(width_px=8, substrate_thickness_um=100.0)
        data = {
            "format_version": 1,
            "material_names": ["Void", "Silicon (Si)"],
            "material_colors": ["#ffffff", "#b0b0b0"],
            "history": [{"title": "Initial", "grid": xsec.copy_state()}],
            "process_steps": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roundtrip.fab"
            save_process_file(path, data)
            loaded = load_process_file(path)
        np.testing.assert_array_equal(loaded["history"][0]["grid"], xsec.grid)

    def test_numpy_private_constructor_aliases_are_remapped_locally(self):
        unpickler = RestrictedProcessUnpickler.__new__(RestrictedProcessUnpickler)
        legacy = unpickler.find_class("numpy.core.numeric", "_frombuffer")
        current = unpickler.find_class("numpy._core.numeric", "_frombuffer")
        self.assertIs(legacy, current)

    def test_restricted_loader_rejects_executable_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "executed"
            path = Path(directory) / "malicious.fab"
            path.write_bytes(pickle.dumps(_MaliciousPayload(marker)))
            with self.assertRaises(ProcessFileError):
                load_process_file(path)
            self.assertFalse(marker.exists())

    def test_existing_process_fixture_remains_compatible(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "testprocess.fab"
        data = load_process_file(fixture)
        self.assertTrue(data["history"])

    def test_malformed_history_is_rejected_before_activation(self):
        data = {
            "format_version": 1,
            "material_names": ["Void", "Silicon (Si)"],
            "material_colors": ["#ffffff", "#b0b0b0"],
            "history": [{}],
            "process_steps": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "malformed.fab"
            path.write_bytes(pickle.dumps(data))
            with self.assertRaises(ProcessFileError):
                load_process_file(path)

    def test_history_material_indices_and_dimensions_are_validated(self):
        base = {
            "format_version": 1,
            "material_names": ["Void", "Silicon (Si)"],
            "material_colors": ["#ffffff", "#b0b0b0"],
            "process_steps": None,
        }
        invalid_snapshots = [
            {"title": "Bad material", "grid": np.array([[2]], dtype=np.int16)},
            {
                "title": "Bad heights",
                "grid": np.array([[1], [1]], dtype=np.int16),
                "row_heights_um": np.array([1.0]),
            },
            {
                "title": "Bad roles",
                "grid": np.array([[1]], dtype=np.int16),
                "structural_role_grid": np.zeros((2, 1), dtype=np.int16),
            },
        ]
        for snapshot in invalid_snapshots:
            with self.subTest(snapshot=snapshot["title"]):
                with self.assertRaises(ProcessFileError):
                    save_process_file(
                        Path(tempfile.gettempdir()) / "invalid-process.fab",
                        {**base, "history": [snapshot]},
                    )

    def test_replay_steps_are_validated_before_activation(self):
        base = {
            "format_version": 1,
            "material_names": ["Void", "Silicon (Si)"],
            "material_colors": ["#ffffff", "#b0b0b0"],
            "history": [],
        }
        invalid_steps = [
            {},
            {"operation": "Unknown"},
            {"operation": "Deposit", "material_name": "Silicon (Si)"},
            {
                "operation": "Etch",
                "material_name": "Silicon (Si)",
                "depth_um": float("nan"),
            },
            {
                "operation": "Lithography",
                "material_name": "Silicon (Si)",
                "windows": [(0.7, 0.3)],
            },
            {
                "operation": "Lithography",
                "material_name": "Silicon (Si)",
                "windows": [],
                "window_start": "bad",
                "window_end": 0.8,
            },
        ]
        for step in invalid_steps:
            with self.subTest(step=step):
                with self.assertRaises(ProcessFileError):
                    save_process_file(
                        Path(tempfile.gettempdir()) / "invalid-step.fab",
                        {**base, "process_steps": [step]},
                    )

    def test_failed_save_is_reported_to_close_guard(self):
        owner = SimpleNamespace(
            history=[{"title": "Unsaved"}],
            current_process_path="/unwritable/process.fab",
            _make_process_dict=lambda: {
                "format_version": 1,
                "material_names": ["Void"],
                "material_colors": ["#ffffff"],
                "history": [],
                "process_steps": [],
            },
        )
        with (
            patch(
                "fabstack.presentation.mixins.process_persistence_mixin.save_process_file",
                side_effect=OSError("disk full"),
            ),
            patch("fabstack.presentation.mixins.process_persistence_mixin.QMessageBox.warning"),
        ):
            saved = ProcessPersistenceMixin._save_process_to_path(
                owner,
                owner.current_process_path,
            )
        self.assertFalse(saved)

        owner._save_process_to_path = lambda _path: False
        self.assertFalse(ProcessPersistenceMixin.save_process(owner))


class LithographyInvariantTests(unittest.TestCase):
    def test_canvas_edges_map_to_full_normalized_range(self):
        axis = object()
        owner = SimpleNamespace(
            xsec=SimpleNamespace(width=300),
            canvas=SimpleNamespace(ax=axis),
        )
        left = SimpleNamespace(inaxes=axis, xdata=-0.5)
        right = SimpleNamespace(inaxes=axis, xdata=299.5)
        self.assertEqual(OperationsMixin._event_to_window_fraction(owner, left), 0.0)
        self.assertEqual(OperationsMixin._event_to_window_fraction(owner, right), 1.0)

    def test_full_width_window_includes_every_column(self):
        xsec = CrossSection(width_px=300, substrate_thickness_um=100.0)
        self.assertTrue(np.all(xsec.window_mask(0.0, 1.0)))

    def test_negative_tone_empty_union_removes_exposed_resist(self):
        xsec = CrossSection(width_px=8, substrate_index=1, substrate_thickness_um=100.0)
        xsec.deposit_physical(2, 1.0)
        self.assertTrue(np.any(xsec.grid == 2))
        xsec.pattern_pr_lithography(2, [], tone="negative")
        self.assertFalse(np.any(xsec.grid == 2))

    def test_preview_and_execution_share_only_explicit_windows(self):
        owner = SimpleNamespace(litho_windows=[])
        self.assertEqual(OperationsMixin.effective_lithography_windows(owner), [])
        owner.litho_windows = [(-0.1, 0.4), (0.8, 1.2), (0.5, 0.5), ("bad", 1)]
        self.assertEqual(
            OperationsMixin.effective_lithography_windows(owner),
            [(0.0, 0.4), (0.8, 1.0)],
        )


class ProcessVisualizationAlignmentTests(unittest.TestCase):
    @staticmethod
    def _cross_section_provenance_state(xsec):
        return {
            "grid": xsec.copy_state(),
            "row_heights": xsec.copy_row_heights(),
            "structural_roles": xsec.copy_structural_roles(),
            "planarizer_roles": xsec.copy_planarizer_roles(),
            "structural_bodies": xsec.copy_structural_bodies(),
            "planarizer_metadata": xsec.copy_planarizer_metadata(),
            "visual_geometry": xsec.copy_visual_geometry(),
        }

    def _assert_cross_section_provenance_equal(self, left, right):
        left_state = self._cross_section_provenance_state(left)
        right_state = self._cross_section_provenance_state(right)
        for key in ("grid", "row_heights", "structural_roles", "planarizer_roles"):
            np.testing.assert_array_equal(left_state[key], right_state[key])
        for key in ("structural_bodies", "planarizer_metadata", "visual_geometry"):
            self.assertEqual(
                pickle.dumps(left_state[key]),
                pickle.dumps(right_state[key]),
            )

    def _assert_shared_material_operation_characterization(self):
        from inspect import signature

        import fabstack.operations.execution as operation_execution
        from fabstack.operations.contracts import UNSET
        from fabstack.operations.execution import (
            apply_process_step_to_current_xsec,
            execute_replay_bake_operation,
            execute_replay_bond_operation,
            execute_replay_flip_operation,
            execute_replay_material_operation,
            execute_ui_bake_operation,
            execute_ui_external_bond_operation,
            execute_ui_flip_operation,
            execute_ui_material_operation,
            execute_ui_simple_bond_operation,
        )

        for function in (
            operation_execution._execute_material_operation,
            execute_ui_material_operation,
            execute_replay_material_operation,
        ):
            for parameter in (
                "thickness_um",
                "depth_um",
                "mask",
                "etch_mode",
                "windows",
                "tone",
            ):
                self.assertIs(
                    signature(function).parameters[parameter].default,
                    UNSET,
                )

        def flat_cross_section():
            return CrossSection(
                width_px=8,
                substrate_index=1,
                substrate_thickness_um=20.0,
                top_headroom_um=8.0,
            )

        def deposited_cross_section():
            xsec = flat_cross_section()
            xsec.deposit_physical(2, 1.25)
            return xsec

        def uneven_cross_section():
            xsec = flat_cross_section()
            mask = np.zeros(xsec.width, dtype=bool)
            mask[2:6] = True
            xsec.etch_physical(1, 3.0, mask=mask, mode="anisotropic")
            return xsec

        cases = (
            (
                "Clean / Surface prep",
                -1,
                None,
                flat_cross_section,
                lambda _xsec: None,
            ),
            (
                "Strip material",
                2,
                None,
                deposited_cross_section,
                lambda xsec: xsec.strip_top_only(2),
            ),
            (
                "Deposit",
                2,
                1.25,
                flat_cross_section,
                lambda xsec: xsec.deposit_physical(2, 1.25),
            ),
            (
                "Spincoating",
                3,
                0.75,
                uneven_cross_section,
                lambda xsec: xsec.spincoat_physical(3, 0.75),
            ),
        )
        for operation, material_index, thickness_um, setup, direct_call in cases:
            direct = setup()
            ui_routed = setup()
            replay_routed = setup()
            direct_call(direct)

            def unexpected_converter(*_args, **_kwargs):
                raise AssertionError("physical routes must not use pixel conversion")

            ui_host = SimpleNamespace(
                xsec=ui_routed,
                um_to_process_px=unexpected_converter,
            )
            replay_host = SimpleNamespace(
                xsec=replay_routed,
                um_to_process_px=unexpected_converter,
            )
            kwargs = {} if thickness_um is None else {"thickness_um": thickness_um}
            with self.subTest(real_route="ui", operation=operation):
                self.assertTrue(
                    execute_ui_material_operation(
                        ui_host,
                        operation,
                        material_index,
                        **kwargs,
                    )
                )
                self._assert_cross_section_provenance_equal(direct, ui_routed)
            with self.subTest(real_route="replay", operation=operation):
                self.assertTrue(
                    execute_replay_material_operation(
                        replay_host,
                        operation,
                        material_index,
                        **kwargs,
                    )
                )
                self._assert_cross_section_provenance_equal(direct, replay_routed)

        for etch_mode in ("anisotropic", "isotropic"):
            direct = flat_cross_section()
            ui_routed = flat_cross_section()
            replay_routed = flat_cross_section()
            direct_mask = np.zeros(direct.width, dtype=bool)
            ui_mask = np.zeros(ui_routed.width, dtype=bool)
            replay_mask = np.zeros(replay_routed.width, dtype=bool)
            direct_mask[2:6] = True
            ui_mask[2:6] = True
            replay_mask[2:6] = True
            direct.etch_physical(
                1,
                depth_um=2.5,
                mask=direct_mask,
                mode=etch_mode,
            )

            def unexpected_etch_converter(*_args, **_kwargs):
                raise AssertionError("physical etch must not use pixel conversion")

            ui_host = SimpleNamespace(
                xsec=ui_routed,
                um_to_process_px=unexpected_etch_converter,
            )
            replay_host = SimpleNamespace(
                xsec=replay_routed,
                um_to_process_px=unexpected_etch_converter,
            )
            self.assertTrue(
                execute_ui_material_operation(
                    ui_host,
                    "Etch",
                    1,
                    depth_um=2.5,
                    mask=ui_mask,
                    etch_mode=etch_mode,
                )
            )
            self.assertTrue(
                execute_replay_material_operation(
                    replay_host,
                    "Etch",
                    1,
                    depth_um=2.5,
                    mask=replay_mask,
                    etch_mode=etch_mode,
                )
            )
            with self.subTest(real_etch=etch_mode, route="ui"):
                self._assert_cross_section_provenance_equal(direct, ui_routed)
            with self.subTest(real_etch=etch_mode, route="replay"):
                self._assert_cross_section_provenance_equal(direct, replay_routed)

        def liftoff_cross_section():
            xsec = flat_cross_section()
            xsec.deposit_physical(2, 1.0)
            xsec.deposit_physical(3, 0.5)
            return xsec

        direct_liftoff = liftoff_cross_section()
        ui_liftoff = liftoff_cross_section()
        replay_liftoff = liftoff_cross_section()
        direct_liftoff.liftoff(2)
        self.assertTrue(
            execute_ui_material_operation(
                SimpleNamespace(xsec=ui_liftoff),
                "Lift-off",
                2,
            )
        )
        self.assertTrue(
            execute_replay_material_operation(
                SimpleNamespace(xsec=replay_liftoff),
                "Lift-off",
                2,
            )
        )
        self._assert_cross_section_provenance_equal(direct_liftoff, ui_liftoff)
        self._assert_cross_section_provenance_equal(direct_liftoff, replay_liftoff)

        def lithography_cross_section():
            xsec = flat_cross_section()
            xsec.deposit_physical(2, 1.0)
            return xsec

        for tone in ("positive", "negative"):
            direct_lithography = lithography_cross_section()
            ui_lithography = lithography_cross_section()
            replay_lithography = lithography_cross_section()
            direct_windows = [(0.25, 0.75)]
            ui_windows = [(0.25, 0.75)]
            replay_windows = [(0.25, 0.75)]
            direct_lithography.pattern_pr_lithography(
                2,
                direct_windows,
                tone=tone,
            )
            self.assertTrue(
                execute_ui_material_operation(
                    SimpleNamespace(xsec=ui_lithography),
                    "Lithography",
                    2,
                    windows=ui_windows,
                    tone=tone,
                )
            )
            self.assertTrue(
                execute_replay_material_operation(
                    SimpleNamespace(xsec=replay_lithography),
                    "Lithography",
                    2,
                    windows=replay_windows,
                    tone=tone,
                )
            )
            with self.subTest(real_lithography=tone, route="ui"):
                self._assert_cross_section_provenance_equal(
                    direct_lithography,
                    ui_lithography,
                )
            with self.subTest(real_lithography=tone, route="replay"):
                self._assert_cross_section_provenance_equal(
                    direct_lithography,
                    replay_lithography,
                )

        def bake_cross_section():
            xsec = flat_cross_section()
            xsec.deposit_physical(2, 1.0)
            xsec.deposit_physical(3, 0.5)
            mask = np.zeros(xsec.width, dtype=bool)
            mask[:4] = True
            xsec.etch_physical(3, 0.5, mask=mask, mode="anisotropic")
            return xsec

        def old_bake_mutation(host, material_index):
            baked_idx = host._bake_material(material_index)
            if baked_idx != material_index:
                grid = host.xsec.grid
                h, w = grid.shape
                for x in range(w):
                    col = grid[:, x]
                    nz = np.where(col != host.VOID_INDEX)[0]
                    if nz.size == 0:
                        continue
                    y0 = nz.min()
                    if col[y0] != material_index:
                        continue
                    y = y0
                    while y < h and col[y] == material_index:
                        col[y] = baked_idx
                        y += 1
                host.xsec.grid = grid
            return baked_idx

        def bake_host(xsec):
            registry = []

            def resolve(material_index):
                registry.append(("resolved", material_index, 4))
                return 4

            return SimpleNamespace(
                VOID_INDEX=0,
                xsec=xsec,
                registry=registry,
                _bake_material=resolve,
            )

        direct_bake_host = bake_host(bake_cross_section())
        ui_bake_host = bake_host(bake_cross_section())
        replay_bake_host = bake_host(bake_cross_section())
        self.assertEqual(old_bake_mutation(direct_bake_host, 2), 4)
        self.assertEqual(execute_ui_bake_operation(ui_bake_host, 2), 4)
        self.assertEqual(execute_replay_bake_operation(replay_bake_host, 2), 4)
        self._assert_cross_section_provenance_equal(
            direct_bake_host.xsec,
            ui_bake_host.xsec,
        )
        self._assert_cross_section_provenance_equal(
            direct_bake_host.xsec,
            replay_bake_host.xsec,
        )
        self.assertTrue(np.any(ui_bake_host.xsec.grid == 4))
        self.assertTrue(np.any(ui_bake_host.xsec.grid == 2))
        self.assertEqual(ui_bake_host.registry, [("resolved", 2, 4)])
        self.assertEqual(replay_bake_host.registry, [("resolved", 2, 4)])

        def clear_xsec_after_resolution(host):
            def resolve(_material_index):
                host.xsec = None
                return 4

            host._bake_material = resolve
            return host

        guarded_bake_host = clear_xsec_after_resolution(
            SimpleNamespace(VOID_INDEX=0, xsec=bake_cross_section())
        )
        self.assertEqual(execute_ui_bake_operation(guarded_bake_host, 2), 4)
        unguarded_bake_host = clear_xsec_after_resolution(
            SimpleNamespace(VOID_INDEX=0, xsec=bake_cross_section())
        )
        with self.assertRaises(AttributeError):
            execute_replay_bake_operation(unguarded_bake_host, 2)
        no_change_bake_host = SimpleNamespace(
            VOID_INDEX=0,
            xsec=None,
            _bake_material=lambda material_index: material_index,
        )
        self.assertEqual(execute_ui_bake_operation(no_change_bake_host, 2), 2)
        self.assertEqual(execute_replay_bake_operation(no_change_bake_host, 2), 2)

        truthy_result = object()
        truthy_flip_host = SimpleNamespace(
            xsec=SimpleNamespace(flip_active_stack=lambda: truthy_result)
        )
        self.assertIs(execute_ui_flip_operation(truthy_flip_host), truthy_result)
        self.assertIs(execute_replay_flip_operation(truthy_flip_host), truthy_result)
        false_flip_host = SimpleNamespace(
            xsec=SimpleNamespace(flip_active_stack=lambda: False)
        )
        self.assertIs(execute_ui_flip_operation(false_flip_host), False)
        self.assertIs(execute_replay_flip_operation(false_flip_host), False)
        replay_false_host = SimpleNamespace(
            xsec=SimpleNamespace(flip_active_stack=lambda: False),
            material_index=lambda _name: -1,
        )
        self.assertTrue(
            apply_process_step_to_current_xsec(
                replay_false_host,
                {"operation": "Flip"},
            )
        )
        raising_flip_host = SimpleNamespace(
            xsec=SimpleNamespace(
                flip_active_stack=lambda: (_ for _ in ()).throw(RuntimeError("flip"))
            )
        )
        with self.assertRaisesRegex(RuntimeError, "flip"):
            execute_ui_flip_operation(raising_flip_host)
        with self.assertRaisesRegex(RuntimeError, "flip"):
            execute_replay_flip_operation(raising_flip_host)

        self.assertIs(
            signature(execute_replay_bond_operation).return_annotation,
            None,
        )

        class PhysicalBondCrossSection:
            def __init__(self):
                self.calls = []

            def bond_physical(self, material_index, thickness_um):
                self.calls.append(("bond_physical", material_index, thickness_um))

            def bond(self, *_args, **_kwargs):
                raise AssertionError("physical Bond must not use the legacy route")

        ui_bond_xsec = PhysicalBondCrossSection()
        ui_bond_host = SimpleNamespace(
            xsec=ui_bond_xsec,
            um_to_process_px=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("UI Bond must not convert to pixels")
            ),
        )
        self.assertIsNone(execute_ui_simple_bond_operation(ui_bond_host, 4, 1.25))
        self.assertEqual(ui_bond_xsec.calls, [("bond_physical", 4, 1.25)])

        ui_legacy_bond_calls = []
        ui_legacy_bond_host = SimpleNamespace(
            xsec=SimpleNamespace(
                bond=lambda *args, **kwargs: ui_legacy_bond_calls.append(
                    (args, kwargs)
                )
            ),
            um_to_process_px=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("UI Bond must not convert to pixels")
            ),
        )
        with self.assertRaises(AttributeError):
            execute_ui_simple_bond_operation(ui_legacy_bond_host, 4, 1.25)
        self.assertEqual(ui_legacy_bond_calls, [])

        raw_external_result = ("source.fab", {"operation": "Bond"}, ["Gold"])
        external_calls = []
        ui_external_host = SimpleNamespace(
            _apply_bond_with_external_stack=lambda material_index, thickness_um: (
                external_calls.append((material_index, thickness_um))
                or raw_external_result
            )
        )
        self.assertIs(
            execute_ui_external_bond_operation(ui_external_host, 6, 2.5),
            raw_external_result,
        )
        self.assertEqual(external_calls, [(6, 2.5)])
        raising_external_host = SimpleNamespace(
            _apply_bond_with_external_stack=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("external Bond")
            )
        )
        with self.assertRaisesRegex(RuntimeError, "external Bond"):
            execute_ui_external_bond_operation(raising_external_host, 6, 2.5)

        direct_bond = flat_cross_section()
        ui_routed_bond = flat_cross_section()
        replay_routed_bond = flat_cross_section()
        direct_bond.bond_physical(2, 1.5)
        self.assertIsNone(
            execute_ui_simple_bond_operation(
                SimpleNamespace(xsec=ui_routed_bond),
                2,
                1.5,
            )
        )
        self.assertIsNone(
            execute_replay_bond_operation(
                SimpleNamespace(xsec=replay_routed_bond),
                2,
                1.5,
                thickness_px=99,
                source_info=None,
            )
        )
        self._assert_cross_section_provenance_equal(direct_bond, ui_routed_bond)
        self._assert_cross_section_provenance_equal(direct_bond, replay_routed_bond)

        replay_physical_trace = []
        replay_physical_host = SimpleNamespace(
            xsec=SimpleNamespace(
                bond_physical=lambda material_index, thickness_um: (
                    replay_physical_trace.append(
                        ("physical", material_index, thickness_um)
                    )
                )
            ),
            material_index=lambda material_name: 7,
            um_to_process_px=lambda value, *, mode, minimum: (
                replay_physical_trace.append(("convert", value, mode, minimum)) or 11
            ),
        )
        self.assertTrue(
            apply_process_step_to_current_xsec(
                replay_physical_host,
                {
                    "operation": "Bond",
                    "material_name": "Gold",
                    "thickness_um": "1.25",
                },
            )
        )
        self.assertEqual(
            replay_physical_trace,
            [("convert", 1.25, "ceil", 1), ("physical", 7, 1.25)],
        )

        replay_legacy_trace = []
        replay_legacy_host = SimpleNamespace(
            xsec=SimpleNamespace(
                bond=lambda material_index, *, thickness_px: (
                    replay_legacy_trace.append(
                        ("legacy", material_index, thickness_px)
                    )
                )
            ),
            material_index=lambda material_name: 8,
            um_to_process_px=lambda value, *, mode, minimum: (
                replay_legacy_trace.append(("convert", value, mode, minimum)) or 12
            ),
        )
        self.assertTrue(
            apply_process_step_to_current_xsec(
                replay_legacy_host,
                {
                    "operation": "Bond",
                    "material_name": "Gold",
                    "thickness_um": 2.5,
                },
            )
        )
        self.assertEqual(
            replay_legacy_trace,
            [("convert", 2.5, "ceil", 1), ("legacy", 8, 12)],
        )

        class ExactBondSource(dict):
            pass

        exact_source = ExactBondSource(source="selected")
        previous_source = object()
        replay_external_trace = []
        replay_external_host = SimpleNamespace(
            xsec=SimpleNamespace(),
            bond_source_info=previous_source,
            material_index=lambda material_name: 9,
            um_to_process_px=lambda value, *, mode, minimum: (
                replay_external_trace.append(("convert", value, mode, minimum)) or 13
            ),
        )

        def replay_external_helper(material_index, thickness_um):
            replay_external_trace.append(
                (
                    "external",
                    replay_external_host.bond_source_info,
                    material_index,
                    thickness_um,
                )
            )
            return raw_external_result

        replay_external_host._apply_bond_with_external_stack = replay_external_helper
        self.assertTrue(
            apply_process_step_to_current_xsec(
                replay_external_host,
                {
                    "operation": "Bond",
                    "material_name": "Gold",
                    "thickness_um": 3.75,
                    "bond_source_info": exact_source,
                },
            )
        )
        self.assertEqual(replay_external_trace[0], ("convert", 3.75, "ceil", 1))
        self.assertEqual(replay_external_trace[1][0], "external")
        self.assertIs(replay_external_trace[1][1], exact_source)
        self.assertEqual(replay_external_trace[1][2:], (9, 3.75))
        self.assertIs(replay_external_host.bond_source_info, previous_source)

        failing_previous_source = object()
        failing_source = ExactBondSource(source="failure")
        failing_external_host = SimpleNamespace(
            xsec=SimpleNamespace(),
            bond_source_info=failing_previous_source,
        )

        def fail_external_bond(*_args):
            self.assertIs(failing_external_host.bond_source_info, failing_source)
            raise RuntimeError("replay external Bond")

        failing_external_host._apply_bond_with_external_stack = fail_external_bond
        with self.assertRaisesRegex(RuntimeError, "replay external Bond"):
            execute_replay_bond_operation(
                failing_external_host,
                10,
                4.0,
                thickness_px=14,
                source_info=failing_source,
            )
        self.assertIs(failing_external_host.bond_source_info, failing_previous_source)

        class DictLikeBondSource:
            pass

        dict_like_source = DictLikeBondSource()
        non_dict_trace = []
        non_dict_previous_source = object()
        non_dict_host = SimpleNamespace(
            xsec=SimpleNamespace(
                bond_physical=lambda material_index, thickness_um: non_dict_trace.append(
                    ("physical", material_index, thickness_um)
                )
            ),
            bond_source_info=non_dict_previous_source,
            _apply_bond_with_external_stack=lambda *_args: (_ for _ in ()).throw(
                AssertionError("non-dict source must not use external Bond")
            ),
        )
        self.assertIsNone(
            execute_replay_bond_operation(
                non_dict_host,
                11,
                4.5,
                thickness_px=15,
                source_info=dict_like_source,
            )
        )
        self.assertEqual(non_dict_trace, [("physical", 11, 4.5)])
        self.assertIs(non_dict_host.bond_source_info, non_dict_previous_source)

        converter_failure_source = ExactBondSource(source="not-installed")
        converter_failure_previous = object()
        converter_failure_trace = []
        converter_failure_host = SimpleNamespace(
            xsec=SimpleNamespace(),
            bond_source_info=converter_failure_previous,
            material_index=lambda material_name: 12,
            um_to_process_px=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("Bond conversion")
            ),
            _apply_bond_with_external_stack=lambda *_args: converter_failure_trace.append(
                "external"
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "Bond conversion"):
            apply_process_step_to_current_xsec(
                converter_failure_host,
                {
                    "operation": "Bond",
                    "material_name": "Gold",
                    "thickness_um": 5.0,
                    "bond_source_info": converter_failure_source,
                },
            )
        self.assertIs(
            converter_failure_host.bond_source_info,
            converter_failure_previous,
        )
        self.assertEqual(converter_failure_trace, [])

        class PhysicalCrossSection:
            def __init__(self):
                self.calls = []

            def deposit_physical(self, material_index, thickness_um):
                self.calls.append(("deposit_physical", material_index, thickness_um))

            def spincoat_physical(self, material_index, thickness_um):
                self.calls.append(("spincoat_physical", material_index, thickness_um))

            def strip_top_only(self, material_index):
                self.calls.append(("strip_top_only", material_index))

            def etch_physical(self, material_index, *, depth_um, mask, mode):
                self.calls.append(
                    ("etch_physical", material_index, depth_um, mask, mode)
                )

            def liftoff(self, material_index):
                self.calls.append(("liftoff", material_index))

            def pattern_pr_lithography(self, material_index, windows, *, tone):
                self.calls.append(
                    ("pattern_pr_lithography", material_index, windows, tone)
                )

        physical_xsec = PhysicalCrossSection()
        physical_converter_calls = []
        physical_host = SimpleNamespace(
            xsec=physical_xsec,
            um_to_process_px=lambda *args, **kwargs: physical_converter_calls.append(
                (args, kwargs)
            ),
        )
        self.assertTrue(
            execute_replay_material_operation(
                physical_host,
                "Deposit",
                4,
                thickness_um=1.25,
            )
        )
        physical_windows = [("0.2", "0.8")]
        self.assertTrue(
            execute_ui_material_operation(
                physical_host,
                "Lithography",
                8,
                windows=physical_windows,
                tone="negative",
            )
        )
        self.assertTrue(
            execute_replay_material_operation(
                physical_host,
                "Spincoating",
                5,
                thickness_um=2.5,
            )
        )
        physical_mask = object()
        physical_mode = object()
        self.assertTrue(
            execute_ui_material_operation(
                physical_host,
                "Etch",
                6,
                depth_um=3.5,
                mask=physical_mask,
                etch_mode=physical_mode,
            )
        )
        self.assertTrue(
            execute_replay_material_operation(
                physical_host,
                "Lift-off",
                7,
            )
        )
        self.assertEqual(physical_xsec.calls[0], ("deposit_physical", 4, 1.25))
        self.assertEqual(
            physical_xsec.calls[1][:2],
            ("pattern_pr_lithography", 8),
        )
        self.assertIs(physical_xsec.calls[1][2], physical_windows)
        self.assertEqual(physical_xsec.calls[1][3], "negative")
        self.assertEqual(physical_xsec.calls[2], ("spincoat_physical", 5, 2.5))
        self.assertEqual(physical_xsec.calls[3][:3], ("etch_physical", 6, 3.5))
        self.assertIs(physical_xsec.calls[3][3], physical_mask)
        self.assertIs(physical_xsec.calls[3][4], physical_mode)
        self.assertEqual(physical_xsec.calls[4], ("liftoff", 7))
        self.assertEqual(physical_converter_calls, [])

        class LegacyCrossSection:
            def __init__(self):
                self.calls = []

            def deposit(self, material_index, *, thickness_px):
                self.calls.append(("deposit", material_index, thickness_px))

            def spincoat(self, material_index, *, thickness_px):
                self.calls.append(("spincoat", material_index, thickness_px))

            def etch(self, material_index, *, depth_px, mask, mode):
                self.calls.append(("etch", material_index, depth_px, mask, mode))

            def liftoff(self, material_index):
                self.calls.append(("liftoff", material_index))

        legacy_xsec = LegacyCrossSection()
        converter_calls = []

        def convert(value_um, *, mode, minimum):
            converter_calls.append((value_um, mode, minimum))
            return 6 + len(converter_calls)

        legacy_host = SimpleNamespace(xsec=legacy_xsec, um_to_process_px=convert)
        self.assertTrue(
            execute_replay_material_operation(
                legacy_host,
                "Deposit",
                4,
                thickness_um=1.25,
            )
        )
        self.assertTrue(
            execute_replay_material_operation(
                legacy_host,
                "Spincoating",
                5,
                thickness_um=2.5,
            )
        )
        legacy_mask = object()
        legacy_mode = object()
        self.assertTrue(
            execute_replay_material_operation(
                legacy_host,
                "Etch",
                6,
                depth_um=3.5,
                mask=legacy_mask,
                etch_mode=legacy_mode,
            )
        )
        self.assertEqual(
            converter_calls,
            [(1.25, "ceil", 1), (2.5, "ceil", 1), (3.5, "ceil", 1)],
        )
        self.assertEqual(
            legacy_xsec.calls[:2],
            [("deposit", 4, 7), ("spincoat", 5, 8)],
        )
        self.assertEqual(legacy_xsec.calls[2][:3], ("etch", 6, 9))
        self.assertIs(legacy_xsec.calls[2][3], legacy_mask)
        self.assertIs(legacy_xsec.calls[2][4], legacy_mode)

        ui_legacy_xsec = LegacyCrossSection()
        ui_converter_calls = []
        ui_legacy_host = SimpleNamespace(
            xsec=ui_legacy_xsec,
            um_to_process_px=lambda *args, **kwargs: ui_converter_calls.append(
                (args, kwargs)
            ),
        )
        with self.assertRaises(AttributeError):
            execute_ui_material_operation(
                ui_legacy_host,
                "Deposit",
                4,
                thickness_um=1.25,
            )
        with self.assertRaises(AttributeError):
            execute_ui_material_operation(
                ui_legacy_host,
                "Spincoating",
                5,
                thickness_um=2.5,
            )
        ui_legacy_mask = object()
        ui_legacy_mode = object()
        with self.assertRaises(AttributeError):
            execute_ui_material_operation(
                ui_legacy_host,
                "Etch",
                6,
                depth_um=3.5,
                mask=ui_legacy_mask,
                etch_mode=ui_legacy_mode,
            )
        self.assertEqual(ui_converter_calls, [])
        self.assertEqual(ui_legacy_xsec.calls, [])

        class LegacyLithographyCrossSection:
            def __init__(self):
                self.calls = []

            def pattern_pr_open_window(self, material_index, start, end):
                self.calls.append((material_index, start, end))

        ui_lithography_xsec = LegacyLithographyCrossSection()
        replay_lithography_xsec = LegacyLithographyCrossSection()
        raw_windows = [["0.2", "0.8"], ["0.35", "0.65"]]
        self.assertTrue(
            execute_ui_material_operation(
                SimpleNamespace(xsec=ui_lithography_xsec),
                "Lithography",
                2,
                windows=raw_windows,
                tone="negative",
            )
        )
        self.assertTrue(
            execute_replay_material_operation(
                SimpleNamespace(xsec=replay_lithography_xsec),
                "Lithography",
                2,
                windows=raw_windows,
                tone="negative",
            )
        )
        self.assertEqual(
            ui_lithography_xsec.calls,
            [(2, "0.2", "0.8"), (2, "0.35", "0.65")],
        )
        self.assertEqual(
            replay_lithography_xsec.calls,
            [(2, 0.2, 0.8), (2, 0.35, 0.65)],
        )

        unchanged = flat_cross_section()
        unchanged_before = unchanged.copy_state()
        unchanged_host = SimpleNamespace(xsec=unchanged)
        self.assertTrue(
            execute_ui_material_operation(
                unchanged_host,
                "Strip material",
                99,
            )
        )
        np.testing.assert_array_equal(unchanged.copy_state(), unchanged_before)
        self.assertTrue(
            execute_ui_material_operation(
                unchanged_host,
                "Clean / Surface prep",
                -1,
            )
        )
        self.assertFalse(
            execute_ui_material_operation(unchanged_host, "Bond", 1)
        )

        for route in (
            execute_ui_material_operation,
            execute_replay_material_operation,
        ):
            missing_thickness_xsec = flat_cross_section()
            missing_thickness_before = missing_thickness_xsec.copy_state()
            missing_thickness_host = SimpleNamespace(xsec=missing_thickness_xsec)
            with self.assertRaises(TypeError):
                route(missing_thickness_host, "Deposit", 2)
            np.testing.assert_array_equal(
                missing_thickness_xsec.copy_state(),
                missing_thickness_before,
            )

    def test_provenance_only_flip_is_recorded(self):
        xsec = CrossSection(
            width_px=4,
            substrate_index=1,
            substrate_thickness_um=100.0,
            top_headroom_um=10.0,
        )
        xsec.bond_physical(1, 100.0)
        before_grid = xsec.copy_state()
        before_roles = xsec.copy_structural_roles()
        recorded = []
        owner = SimpleNamespace(
            xsec=xsec,
            record_process_step=lambda title, step: recorded.append((title, step)),
        )

        MainWindow.flip_active_stack_from_top(owner)

        np.testing.assert_array_equal(xsec.grid, before_grid)
        self.assertFalse(np.array_equal(xsec.structural_role_grid, before_roles))
        self.assertEqual(recorded, [("Flip stack", {"operation": "Flip"})])

        false_recorded = []
        false_owner = SimpleNamespace(
            xsec=SimpleNamespace(flip_active_stack=lambda: False),
            record_process_step=lambda title, step: false_recorded.append((title, step)),
        )
        MainWindow.flip_active_stack_from_top(false_owner)
        self.assertEqual(false_recorded, [])

    def test_filled_etch_does_not_keep_stale_front_overlay(self):
        xsec = CrossSection(
            width_px=12,
            substrate_index=1,
            substrate_thickness_um=100.0,
            top_headroom_um=20.0,
        )
        mask = np.zeros(xsec.width, dtype=bool)
        mask[4:8] = True
        xsec.etch_physical(1, 5.0, mask=mask, mode="isotropic")
        self.assertTrue(
            any(item.get("type") == "isotropic_etch_front" for item in xsec.visual_geometry)
        )
        xsec.spincoat_physical(2, 1.0)
        self.assertFalse(
            any(item.get("type") == "isotropic_etch_front" for item in xsec.visual_geometry)
        )
        self.assertTrue(
            any(item.get("type") == "etch_depth_feature" for item in xsec.visual_geometry)
        )
        self._assert_shared_material_operation_characterization()


class MaterialLibraryTests(unittest.TestCase):
    def test_shipped_library_has_canonical_compatibility_references(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "fabstack"
            / "assets"
            / "materials_library.json"
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertIs(validate_material_library(data), data)

    def test_unknown_material_rule_is_rejected(self):
        data = {
            "materials": {"Substrates": ["Silicon"]},
            "compatibility": {"etchants_by_material": {"Typo": ["HF"]}},
        }
        with self.assertRaises(MaterialLibraryError):
            validate_material_library(data)


class UiCopyTests(unittest.TestCase):
    def test_visualization_controls_are_aligned_and_capped(self):
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        try:
            window.resize(1400, 800)
            window.show()
            app.processEvents()

            zoom_geometry = window.global_scale_overlay_slider.geometry()
            detail_geometry = window.feature_scale_overlay_slider.geometry()
            self.assertEqual(zoom_geometry.x(), detail_geometry.x())
            self.assertEqual(zoom_geometry.width(), detail_geometry.width())
            self.assertEqual(window.feature_scale_overlay_value.text(), "0.50")
            self.assertNotIn("Balanced", window.feature_scale_overlay_value.text())
            overlay_layout = window.canvas_scale_overlay.layout()
            self.assertEqual(overlay_layout.horizontalSpacing(), 6)
            self.assertEqual(overlay_layout.columnMinimumWidth(2), 44)
            window.feature_scale_factor = 0.75
            window._sync_feature_scale_controls_from_state()
            self.assertEqual(window.feature_scale_overlay_value.text(), "0.75")
            window.feature_scale_factor = 0.50
            window._sync_feature_scale_controls_from_state()
            self.assertEqual(window.global_scale_overlay_slider.maximum(), 150)
            self.assertEqual(window._clamped_visual_scale(4.0), 1.5)

            window.set_deposition_thickness_step()
            app.processEvents()
            self.assertEqual(window.global_scale_spin.maximum(), 1.5)
        finally:
            if window.scale_dialog is not None:
                window.scale_dialog.close()
            window.close()
            window.deleteLater()
            app.processEvents()

    def test_assistant_reports_a_noop_as_action_needed(self):
        app = QApplication.instance() or QApplication([])
        with patch.object(QMessageBox, "warning", return_value=QMessageBox.Ok):
            window = MainWindow()
            before = len(window.history)
            succeeded, message = window._execute_assistant_command(
                {
                    "action": "operation",
                    "operation": "Strip material",
                    "material": "Gold",
                }
            )

        self.assertFalse(succeeded)
        self.assertEqual(len(window.history), before)
        self.assertIn("made no change", message)
        self.assertIn("try again", message)
        window.deleteLater()
        app.processEvents()

    def test_display_labels_do_not_change_persisted_operation_keys(self):
        from typing import Any, get_args, get_type_hints

        from fabstack.assistant.planner import ALLOWED_OPERATIONS
        from fabstack.operations.contracts import (
            AssistantOperationCommand,
            OPERATION_KEYS,
            OperationKey,
            OperationSource,
            PersistedOperationStep,
            UI_OPERATION_KEYS,
            UNSET,
        )
        from fabstack.presentation.mixins import ui_layout_mixin

        persisted_keys = (
            "Deposit",
            "Spincoating",
            "Bond",
            "Etch",
            "Strip material",
            "Lift-off",
            "Clean / Surface prep",
            "Bake/Annealing",
            "Lithography",
            "Flip",
        )
        self.assertEqual(tuple(OPERATION_LABELS), persisted_keys)
        self.assertEqual(get_args(OperationKey), persisted_keys)
        self.assertIs(ALLOWED_OPERATIONS, OPERATION_KEYS)
        self.assertEqual(OPERATION_KEYS, persisted_keys)
        self.assertEqual(UI_OPERATION_KEYS, persisted_keys[:-1])
        self.assertIs(ui_layout_mixin.UI_OPERATION_KEYS, UI_OPERATION_KEYS)
        self.assertEqual(
            get_args(OperationSource),
            ("interactive", "assistant", "replay"),
        )

        ui_layout_path = (
            Path(__file__).resolve().parents[2]
            / "fabstack"
            / "presentation"
            / "mixins"
            / "ui_layout_mixin.py"
        )
        ui_tree = ast.parse(
            ui_layout_path.read_text(encoding="utf-8"),
            filename=str(ui_layout_path),
        )
        add_items_call = next(
            node
            for node in ast.walk(ui_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "addItems"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "op_combo"
        )
        self.assertEqual(
            ast.dump(add_items_call.args[0], include_attributes=False),
            ast.dump(
                ast.parse("list(UI_OPERATION_KEYS)", mode="eval").body,
                include_attributes=False,
            ),
        )

        persisted_step_keys = {
            "operation",
            "material_name",
            "title",
            "thickness_um",
            "depth_um",
            "method",
            "etch_mode",
            "bond_source_info",
            "windows",
            "window_start",
            "window_end",
            "invert_lithography",
        }
        assistant_command_keys = {
            "action",
            "operation",
            "material",
            "thickness_um",
            "depth_um",
            "temperature_c",
            "time_s",
            "method",
            "etch_mode",
            "invert_lithography",
            "windowing_mode",
            "windows",
        }
        self.assertFalse(PersistedOperationStep.__total__)
        self.assertFalse(AssistantOperationCommand.__total__)
        self.assertEqual(set(PersistedOperationStep.__annotations__), persisted_step_keys)
        self.assertEqual(
            set(AssistantOperationCommand.__annotations__),
            assistant_command_keys,
        )
        self.assertEqual(PersistedOperationStep.__required_keys__, frozenset())
        self.assertEqual(AssistantOperationCommand.__required_keys__, frozenset())
        self.assertEqual(
            get_type_hints(PersistedOperationStep)["bond_source_info"],
            dict[str, Any],
        )

        missing = {}
        explicit_unset = {"value": UNSET}
        explicit_none = {"value": None}
        self.assertNotIn("value", missing)
        self.assertIs(explicit_unset["value"], UNSET)
        self.assertIsNot(explicit_unset["value"], explicit_none["value"])
        self.assertEqual(repr(UNSET), "UNSET")
        self.assertEqual(operation_label("Spincoating"), "Spin coat")
        self.assertEqual(operation_label("Bake/Annealing"), "Bake / anneal")
        self.assertEqual(operation_label("Unknown operation"), "Unknown operation")

    def test_generated_step_titles_use_the_same_display_vocabulary(self):
        owner = SimpleNamespace(
            _format_step_um=lambda value: f"{float(value):g}",
        )
        step = {
            "operation": "Spincoating",
            "material_name": "Photoresist",
            "thickness_um": 1.5,
        }
        title = MainWindow._generated_process_step_title(owner, step)
        self.assertEqual(title, "Spin coat Photoresist (1.5 µm)")
        self.assertEqual(step["operation"], "Spincoating")

    def test_user_facing_copy_does_not_expose_implementation_terms(self):
        root = Path(__file__).resolve().parents[2]
        ui_files = [
            root / "fabstack" / "presentation" / "main_window.py",
            root / "fabstack" / "presentation" / "dialogs" / "material_library.py",
            *(root / "fabstack" / "presentation" / "mixins").glob("*.py"),
        ]
        visible_calls = {
            "QAction",
            "QLabel",
            "QPushButton",
            "getItem",
            "getOpenFileName",
            "getSaveFileName",
            "getText",
            "information",
            "critical",
            "question",
            "setPlaceholderText",
            "setText",
            "setToolTip",
            "setWindowTitle",
            "showMessage",
            "warning",
        }
        blocked_terms = {
            "ascii string",
            "depth_um",
            "in-memory",
            "internal indexing",
            "llama-cpp",
            "local mistral",
            "pip install",
            "python-docx",
            "thickness_um",
        }

        def literal_text(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return node.value
            if isinstance(node, ast.JoinedStr):
                return "".join(
                    value.value
                    for value in node.values
                    if isinstance(value, ast.Constant) and isinstance(value.value, str)
                )
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                return literal_text(node.left) + literal_text(node.right)
            return ""

        violations = []
        for path in ui_files:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                call_name = ""
                if isinstance(node.func, ast.Name):
                    call_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    call_name = node.func.attr
                if call_name not in visible_calls:
                    continue
                if call_name in {"critical", "information", "question", "warning"}:
                    owner = node.func.value if isinstance(node.func, ast.Attribute) else None
                    if not isinstance(owner, ast.Name) or owner.id != "QMessageBox":
                        continue
                if call_name in {"getItem", "getText"}:
                    owner = node.func.value if isinstance(node.func, ast.Attribute) else None
                    if not isinstance(owner, ast.Name) or owner.id != "QInputDialog":
                        continue
                if call_name in {"getOpenFileName", "getSaveFileName"}:
                    owner = node.func.value if isinstance(node.func, ast.Attribute) else None
                    if not isinstance(owner, ast.Name) or owner.id != "QFileDialog":
                        continue
                copy = " ".join(literal_text(arg) for arg in node.args).lower()
                for term in blocked_terms:
                    if term in copy:
                        violations.append(f"{path.name}:{node.lineno}: {term}")
        self.assertEqual(violations, [])


class ArchitectureContractTests(unittest.TestCase):
    def test_lower_layers_do_not_depend_on_qt_presentation(self):
        package_root = Path(__file__).resolve().parents[2] / "fabstack"
        lower_layer_paths = [
            package_root / "assistant",
            package_root / "config",
            package_root / "domain",
            package_root / "materials",
            package_root / "operations",
            package_root / "persistence",
            package_root / "rendering",
        ]
        lower_layer_files = [
            package_root / "resource_paths.py",
            package_root / "terminology.py",
            *(
                path
                for directory in lower_layer_paths
                for path in directory.glob("*.py")
            ),
        ]

        violations = []
        for path in lower_layer_files:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.ImportFrom):
                    modules.append(node.module or "")
                elif isinstance(node, ast.Import):
                    modules.extend(alias.name for alias in node.names)
                for module in modules:
                    if module == "PyQt5" or module.startswith("PyQt5."):
                        violations.append(f"{path.relative_to(package_root)}: {module}")
                    if module == "fabstack.presentation" or module.startswith(
                        "fabstack.presentation."
                    ):
                        violations.append(f"{path.relative_to(package_root)}: {module}")
                    if path.parent.name == "operations" and (
                        module in {"fabstack.assistant", "fabstack.persistence"}
                        or module.startswith("fabstack.assistant.")
                        or module.startswith("fabstack.persistence.")
                    ):
                        violations.append(f"{path.relative_to(package_root)}: {module}")

        self.assertEqual(violations, [])

        expected_service_imports = {
            "scene_constraints.py": {
                "math",
                "numpy",
                "fabstack.rendering.constants",
                "fabstack.rendering.contracts",
            },
            "segment_targets.py": {
                "numpy",
                "fabstack.rendering.constants",
                "fabstack.rendering.contracts",
            },
        }
        expected_constant_imports = {
            "scene_constraints.py": {
                "MAX_CONSTRAINT_PASSES",
                "MIN_SUBSTRATE_CORE_PX",
                "MIN_VOID_PX",
            },
            "segment_targets.py": {"MIN_VOID_PX"},
        }
        for filename, expected_modules in expected_service_imports.items():
            path = package_root / "rendering" / filename
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported_modules = set()
            imported_constants = set()
            for node in tree.body:
                if isinstance(node, ast.Import):
                    imported_modules.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imported_modules.add(node.module or "")
                    if node.module == "fabstack.rendering.constants":
                        imported_constants.update(alias.name for alias in node.names)
            with self.subTest(service=filename):
                self.assertEqual(imported_modules, expected_modules)
                self.assertEqual(
                    imported_constants,
                    expected_constant_imports[filename],
                )

        expected_operation_imports = {
            "__init__.py": {
                "fabstack.operations.contracts",
                "fabstack.operations.execution",
            },
            "contracts.py": {"typing", "fabstack.terminology"},
            "execution.py": {
                "dataclasses",
                "typing",
                "numpy",
                "fabstack.operations.contracts",
            },
        }
        for filename, expected_modules in expected_operation_imports.items():
            path = package_root / "operations" / filename
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported_modules = set()
            for node in tree.body:
                if isinstance(node, ast.Import):
                    imported_modules.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imported_modules.add(node.module or "")
            with self.subTest(operation_module=filename):
                self.assertEqual(imported_modules, expected_modules)

    def test_legacy_facades_delegate_to_packaged_implementations(self):
        import dimension_preservation_validation as legacy_regression
        import library_manager as legacy_library
        import llm_planner_stress_test as legacy_stress
        import llm_step_planner as legacy_planner
        import main_window as legacy_window
        import process_core as legacy_core
        import process_file as legacy_process_file
        import resources as legacy_resources
        import visualization_config as legacy_visualization
        from main_window_mixins import plotting_export_mixin as legacy_plotting
        from main_window_mixins import process_persistence_mixin as legacy_persistence

        from fabstack.assistant import measurements, normalization, planner, prompts
        from fabstack.config import visualization
        from fabstack.domain.cross_section import CrossSection
        from fabstack.persistence import process_file
        from fabstack.presentation.dialogs.material_library import (
            DEFAULT_DEPOSITION_METHODS,
            LibraryManagerDialog,
        )
        from fabstack.presentation.main_window import MainWindow
        from fabstack.presentation.mixins import (
            CanvasRenderingMixin,
            DisplayProfileMixin,
            ExportActionsMixin,
            PlottingExportMixin,
            VisualAdjustmentMixin,
            plotting_export_mixin,
        )
        from fabstack.presentation.mixins import process_persistence_mixin
        from fabstack.resource_paths import resource_path
        from fabstack import operations, rendering
        from fabstack.operations.contracts import (
            AssistantOperationCommand,
            OPERATION_KEYS,
            OperationKey,
            OperationSource,
            PersistedOperationStep,
            UI_OPERATION_KEYS,
            UNSET,
        )
        from fabstack.operations.execution import (
            BakeOperationHost,
            BondExternalResult,
            BondOperationHost,
            FlipOperationHost,
            MaterialOperationHost,
            ReplayExecutionHost,
            apply_process_step_to_current_xsec,
            execute_replay_bake_operation,
            execute_replay_bond_operation,
            execute_replay_flip_operation,
            execute_replay_material_operation,
            execute_ui_bake_operation,
            execute_ui_external_bond_operation,
            execute_ui_flip_operation,
            execute_ui_material_operation,
            execute_ui_simple_bond_operation,
        )
        from fabstack.rendering.contracts import (
            SceneConstraintHost,
            SegmentTargetHost,
        )
        from fabstack.rendering.scaling import (
            _raw_feature_scale_um,
            _visual_height_to_display_rows,
            physical_span_to_raw_visual_px,
            thickness_to_raw_visual_px,
        )
        from fabstack.rendering.scene_constraints import (
            build_scene_constraint_profile,
        )
        from fabstack.rendering.segment_targets import (
            build_material_column_segments_with_targets,
        )
        from tests.regression import dimension_preservation
        from tests.stress import planner_stress

        self.assertIs(legacy_window.MainWindow, MainWindow)
        self.assertIs(legacy_core.CrossSection, CrossSection)
        self.assertIs(legacy_process_file.load_process_file, process_file.load_process_file)
        self.assertIs(legacy_process_file.save_process_file, process_file.save_process_file)
        self.assertIs(legacy_planner.LocalStepPlanner, planner.LocalStepPlanner)
        self.assertIs(legacy_planner._canonical_material, planner._canonical_material)
        self.assertIs(legacy_planner._coerce_bool, planner._coerce_bool)
        self.assertIs(planner._coerce_bool, normalization._coerce_bool)
        self.assertIs(planner._coerce_float, normalization._coerce_float)
        self.assertIs(
            planner._measure_was_requested,
            normalization._measure_was_requested,
        )
        self.assertIs(
            planner._normalize_user_request,
            normalization._normalize_user_request,
        )
        self.assertIs(planner._unique, normalization._unique)
        self.assertIs(
            planner._user_facing_detail,
            normalization._user_facing_detail,
        )
        self.assertIs(planner.MODEL_FILENAME, prompts.MODEL_FILENAME)
        self.assertIs(planner.SYSTEM_PROMPT, prompts.SYSTEM_PROMPT)
        self.assertIs(
            planner.COMMAND_SYSTEM_PROMPT,
            prompts.COMMAND_SYSTEM_PROMPT,
        )
        self.assertIs(legacy_planner.MODEL_FILENAME, prompts.MODEL_FILENAME)
        self.assertIs(legacy_planner.SYSTEM_PROMPT, prompts.SYSTEM_PROMPT)
        self.assertIs(
            legacy_planner.COMMAND_SYSTEM_PROMPT,
            prompts.COMMAND_SYSTEM_PROMPT,
        )
        measurement_names = (
            "_NUMBER_PATTERN",
            "_UNIT_PATTERN",
            "_extract_all_measure_um",
            "_extract_measure",
            "_extract_measure_mentions_um",
            "_extract_measure_um",
            "_extract_substrate_like_unitless_um",
            "_extract_unitless_measure_um",
            "_first_explicit_unit",
            "_last_explicit_unit",
            "_number_span_is_identifier",
            "_unit_value_to_um",
        )
        for name in measurement_names:
            with self.subTest(planner_measurement_identity=name):
                self.assertIs(getattr(planner, name), getattr(measurements, name))
        self.assertIs(
            legacy_persistence.ProcessPersistenceMixin,
            process_persistence_mixin.ProcessPersistenceMixin,
        )
        self.assertIs(
            legacy_plotting.PlottingExportMixin,
            plotting_export_mixin.PlottingExportMixin,
        )
        self.assertIs(
            legacy_plotting.thickness_to_raw_visual_px,
            thickness_to_raw_visual_px,
        )
        self.assertIs(legacy_plotting._raw_feature_scale_um, _raw_feature_scale_um)
        self.assertIs(
            legacy_plotting.physical_span_to_raw_visual_px,
            physical_span_to_raw_visual_px,
        )
        self.assertIs(
            legacy_plotting._visual_height_to_display_rows,
            _visual_height_to_display_rows,
        )
        self.assertIs(rendering.SceneConstraintHost, SceneConstraintHost)
        self.assertIs(rendering.SegmentTargetHost, SegmentTargetHost)
        self.assertIs(
            rendering.build_scene_constraint_profile,
            build_scene_constraint_profile,
        )
        self.assertIs(
            rendering.build_material_column_segments_with_targets,
            build_material_column_segments_with_targets,
        )
        self.assertIs(operations.AssistantOperationCommand, AssistantOperationCommand)
        self.assertIs(operations.BakeOperationHost, BakeOperationHost)
        self.assertIs(operations.BondExternalResult, BondExternalResult)
        self.assertIs(operations.BondOperationHost, BondOperationHost)
        self.assertIs(operations.FlipOperationHost, FlipOperationHost)
        self.assertIs(operations.MaterialOperationHost, MaterialOperationHost)
        self.assertIs(operations.OPERATION_KEYS, OPERATION_KEYS)
        self.assertIs(operations.OperationKey, OperationKey)
        self.assertIs(operations.OperationSource, OperationSource)
        self.assertIs(operations.PersistedOperationStep, PersistedOperationStep)
        self.assertIs(operations.ReplayExecutionHost, ReplayExecutionHost)
        self.assertIs(operations.UI_OPERATION_KEYS, UI_OPERATION_KEYS)
        self.assertIs(operations.UNSET, UNSET)
        self.assertIs(
            operations.apply_process_step_to_current_xsec,
            apply_process_step_to_current_xsec,
        )
        self.assertIs(
            operations.execute_replay_material_operation,
            execute_replay_material_operation,
        )
        self.assertIs(
            operations.execute_replay_bake_operation,
            execute_replay_bake_operation,
        )
        self.assertIs(
            operations.execute_replay_bond_operation,
            execute_replay_bond_operation,
        )
        self.assertIs(
            operations.execute_replay_flip_operation,
            execute_replay_flip_operation,
        )
        self.assertIs(operations.execute_ui_bake_operation, execute_ui_bake_operation)
        self.assertIs(
            operations.execute_ui_external_bond_operation,
            execute_ui_external_bond_operation,
        )
        self.assertIs(operations.execute_ui_flip_operation, execute_ui_flip_operation)
        self.assertIs(
            operations.execute_ui_material_operation,
            execute_ui_material_operation,
        )
        self.assertIs(
            operations.execute_ui_simple_bond_operation,
            execute_ui_simple_bond_operation,
        )

        package_root = Path(__file__).resolve().parents[2] / "fabstack"
        mixin_path = package_root / "presentation" / "mixins" / "display_profile_mixin.py"
        mixin_tree = ast.parse(
            mixin_path.read_text(encoding="utf-8"),
            filename=str(mixin_path),
        )

        def function_node(tree, name):
            return next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == name
            )

        planner_path = package_root / "assistant" / "planner.py"
        planner_source = planner_path.read_text(encoding="utf-8")
        planner_tree = ast.parse(planner_source, filename=str(planner_path))
        planner_lines = planner_source.splitlines(keepends=True)
        planner_class = next(
            node
            for node in planner_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "LocalStepPlanner"
        )
        planner_class_source = "".join(
            planner_lines[planner_class.lineno - 1:planner_class.end_lineno]
        )
        self.assertEqual(
            hashlib.sha256(planner_class_source.encode("utf-8")).hexdigest(),
            "9c96ff69f329d3fac4a08708b4bd1bef16d2e15a56ec5dd4cae30e345857095b",
        )
        self.assertEqual(
            hashlib.sha256(
                ast.dump(planner_class, include_attributes=False).encode("utf-8")
            ).hexdigest(),
            "5f1f3011356a60bfb9926cc677a6e9387f04ce2a7a4aefd3a9c240056e0f4279",
        )

        prompt_path = package_root / "assistant" / "prompts.py"
        prompt_source = prompt_path.read_text(encoding="utf-8")
        prompt_tree = ast.parse(prompt_source, filename=str(prompt_path))
        prompt_lines = prompt_source.splitlines(keepends=True)
        expected_prompt_hashes = {
            "MODEL_FILENAME": (
                "c70f7a30a3c4fc0f4bd6cdc00030008361ff33d66ba63ecb8ae34a78260f582f",
                "831b89a54b5f837ba296417a9d851216596a7f2d02e9c667afe8d98162f5d400",
                "38ca78c2ac76f1c5f8f1c929ca4f26e8c1c991bb7159a41f43d7f37b13cce95a",
            ),
            "SYSTEM_PROMPT": (
                "52c30f242cc46f78dd5d7dca4d253fdf12aa3613586274d15f7a22e05d49d807",
                "9f3f88cbc52253fbf9e8d1ec0e29dc9b99f97df48561085cecaf40b80b7dffca",
                "c9f707199def5ece8c82362ab1926910b79ea0c25d3a4169720f324154d82bcf",
            ),
            "COMMAND_SYSTEM_PROMPT": (
                "1d863d5e0c8b105ae5a4d5fda18e3cf476fe0e40c967c5e7dc59a4660ac0dfdc",
                "789b10e046d67c5faec340455a63adef0558c4eeb85106597bd38b17f88cf25e",
                "2fcc1dc2ed70c49598a020027c24902933c9cde5450ae64333bb88a693d8fd45",
            ),
        }
        prompt_assignments = {
            node.targets[0].id: node
            for node in prompt_tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        }
        for name, (
            expected_source_hash,
            expected_ast_hash,
            expected_value_hash,
        ) in expected_prompt_hashes.items():
            assignment = prompt_assignments[name]
            assignment_source = "".join(
                prompt_lines[assignment.lineno - 1:assignment.end_lineno]
            )
            with self.subTest(planner_prompt=name):
                self.assertEqual(
                    hashlib.sha256(assignment_source.encode("utf-8")).hexdigest(),
                    expected_source_hash,
                )
                self.assertEqual(
                    hashlib.sha256(
                        ast.dump(assignment, include_attributes=False).encode("utf-8")
                    ).hexdigest(),
                    expected_ast_hash,
                )
                self.assertEqual(
                    hashlib.sha256(getattr(prompts, name).encode("utf-8")).hexdigest(),
                    expected_value_hash,
                )

        normalization_path = package_root / "assistant" / "normalization.py"
        normalization_source = normalization_path.read_text(encoding="utf-8")
        normalization_tree = ast.parse(
            normalization_source,
            filename=str(normalization_path),
        )
        normalization_lines = normalization_source.splitlines(keepends=True)
        expected_normalization_hashes = {
            "_user_facing_detail": (
                "cc744f2f00c88ba6e83ee3160e5fc5a7824b98691c078884b2d09b81079d970c",
                "5a6ae364216cc7ebb3b6be93cd63b88a0bf9659d68b71b1c745c9faa674198d5",
                "346016658023c3d1e4525f805a8ce84b7e23e9aee7b2f0d2a2781de183021a78",
            ),
            "_normalize_user_request": (
                "630aa88f44e363dfd44f658993d62e6e7018d5d7d187a5c71bcd7be4a7df84ef",
                "a5b7b28b081dba63b13d6470cf1e803764c64de9eef53edc2a634cb969975d3b",
                "0bb252ec0f9fae0e0f70507960eb545835db2fb51691662d797473988112c84e",
            ),
            "_coerce_float": (
                "34ee75b28b8caa10c2182821fac91e2bd9f4afdf376534aa5ac326b2ae624a8d",
                "252fa0e9c5bcdf4a2e7537aa63fd160db67a62f133ff932272376279b83c0467",
                "2eb0252166b30877c3e8183456c68798a97d2a322ae553cef6c0435580b1ebc4",
            ),
            "_coerce_bool": (
                "03fa8ea11f1ab32a336be9506f216748957505ffd5bf4b8acc11804326dc8740",
                "9ab1d293c3a073e3aec4fa8c26f84d17aa900a459f12362bfed1ff41491e5d23",
                "2d42a5803052c5c24837d96bebe8ee1deca264d611e93a9b751ce20abeb40265",
            ),
            "_unique": (
                "6a1847be145a938ba18a73a1322cf4166717f7a6a51488651e4ea4d880ea37bb",
                "6551b1caec50f721ba33e946afca79311c7efba66ada9ab21f008b74605f38eb",
                "6d0cccf720de3e5f21d2d42d57db055d7807a7f55fa2169a9493b63299d0a29c",
            ),
            "_measure_was_requested": (
                "3201a28292e9e8e6cfb0172fc05369d524f7cf620f972a73b8d937bd97048e1a",
                "d7daf0c9604ab76b007153fcffd2a68a3b7ee0ba6a17ca69bc3e17282341c984",
                "0d7dbaa37573571355a3f4ddc162b3364e5cb3cef19e543720c7f150ac06a67f",
            ),
        }
        expected_leaf_globals = {
            "_user_facing_detail": {"re"},
            "_normalize_user_request": {"re"},
            "_coerce_float": set(),
            "_coerce_bool": set(),
            "_unique": {"Iterable", "List"},
            "_measure_was_requested": {"List"},
        }
        import builtins

        builtin_names = set(dir(builtins))
        for name, (
            expected_source_hash,
            expected_ast_hash,
            expected_body_hash,
        ) in expected_normalization_hashes.items():
            function = function_node(normalization_tree, name)
            function_source = "".join(
                normalization_lines[function.lineno - 1:function.end_lineno]
            )
            function_body = ast.Module(body=function.body, type_ignores=[])
            parameters = {
                argument.arg for argument in ast.walk(function)
                if isinstance(argument, ast.arg)
            }
            assigned = {
                node.id
                for node in ast.walk(function)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            }
            loaded = {
                node.id
                for node in ast.walk(function)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            }
            with self.subTest(planner_normalization=name):
                self.assertEqual(
                    hashlib.sha256(function_source.encode("utf-8")).hexdigest(),
                    expected_source_hash,
                )
                self.assertEqual(
                    hashlib.sha256(
                        ast.dump(function, include_attributes=False).encode("utf-8")
                    ).hexdigest(),
                    expected_ast_hash,
                )
                self.assertEqual(
                    hashlib.sha256(
                        ast.dump(function_body, include_attributes=False).encode("utf-8")
                    ).hexdigest(),
                    expected_body_hash,
                )
                self.assertEqual(
                    loaded - parameters - assigned - builtin_names,
                    expected_leaf_globals[name],
                )

        measurements_path = package_root / "assistant" / "measurements.py"
        measurements_source = measurements_path.read_text(encoding="utf-8")
        measurements_tree = ast.parse(
            measurements_source,
            filename=str(measurements_path),
        )
        measurements_lines = measurements_source.splitlines(keepends=True)
        expected_measurement_constant_hashes = {
            "_UNIT_PATTERN": (
                "7f3eb52038de813072af6a549c18796bf49546db2dd923ee9aef8b8075088de3",
                "199c3dc7bcb509048891f31dfb5314b83d400c3cc92b34c0339ffb061c177511",
                "d8775faa78a544668f677fb4ccb40567c5f3cc0f1125de5f74a03c4d50801bdc",
            ),
            "_NUMBER_PATTERN": (
                "75f62a5b4af2af6f74590a2e3e7ef6c82d02e376b3f614b60db4416f06b33c37",
                "5fac885479ca3887aac7d782810a44de46c87f8d4edf980f795be8029cfb8a5d",
                "43e4235edf4071a7fbf96aab9f858095417222eb0933529266141d2c24432a4c",
            ),
        }
        measurement_assignments = {
            node.targets[0].id: node
            for node in measurements_tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        }
        for name, (
            expected_source_hash,
            expected_ast_hash,
            expected_value_hash,
        ) in expected_measurement_constant_hashes.items():
            assignment = measurement_assignments[name]
            assignment_source = "".join(
                measurements_lines[assignment.lineno - 1:assignment.end_lineno]
            )
            with self.subTest(measurement_constant=name):
                self.assertEqual(
                    hashlib.sha256(assignment_source.encode("utf-8")).hexdigest(),
                    expected_source_hash,
                )
                self.assertEqual(
                    hashlib.sha256(
                        ast.dump(assignment, include_attributes=False).encode("utf-8")
                    ).hexdigest(),
                    expected_ast_hash,
                )
                self.assertEqual(
                    hashlib.sha256(
                        getattr(measurements, name).encode("utf-8")
                    ).hexdigest(),
                    expected_value_hash,
                )

        expected_measurement_function_hashes = {
            "_first_explicit_unit": ("5d2e47da525050f12d8c4820f2457dbc249d52a61d4ca1b57ce018c12b68aaa0", "955bb6f07d2cf996ce8b21e5958e16475a680f5f48c9db3b88006ba348e34c52", "749e0c011320fade264a0ee7f7ece6c68fc189dc182557472348c279bbe90c02"),
            "_last_explicit_unit": ("838e25a4cd267e6f61a7f948f45b2afc00c0b3b648028c19b52f05a6dd8ee6db", "6415f578703f0e129d6389d4dffff538ce6bc42edee22a02b95fa38e07d6ea9f", "556daf2a88c672e65f3c91408ab82c0939b0e5b8f9b9902fef9a347b69c1927e"),
            "_unit_value_to_um": ("b8c25153ca4248a751cfd5778d280fc728d9ec3de3d914ee33a6d523b8181689", "c11fe94da66cf67605f1406fb03d4327e022b7dabb7caafb2341730720ace5fb", "964b9b913864d322d2b0e4cc204a534e215ec186d25a20ea8e764a03a7be4a2b"),
            "_extract_measure_mentions_um": ("05ef0de5e7780429cb841bf6ced34a85d9546ea47dbf0976c68ffef89ec2f5f8", "6fd2bd2e916c714885ba34c6418fc99caa428033501f9acea809f8ebdc171ab1", "db28cfc2f49e937ee1399ab2ef2b9478556a5976bc574299dc6b46aab10543d4"),
            "_extract_unitless_measure_um": ("e10b294db03944cc4dcad9410fe483f7efa4a9201c2e9bdce455b8d29848c1a4", "e3b1ea8f8ee5cb3531f091b2112e70b24131bf089d5644fe53d60d296d4068e9", "1186b30212ce3bc3a14e8f3815ae5141890fa84896475ef544fe9986c591cb0f"),
            "_number_span_is_identifier": ("f57471b1a50174e673c0e8931d37fbd2888b3511e2d9135512f65c620aeb00e3", "4cc24ca7b88ddbe356882d577701abd07484afc580919c25104503ae92ea1659", "da1c44238c970089704a7c0158c84d181f06b5b1768d9c84be72fed7378e074e"),
            "_extract_measure_um": ("611f50611092003246680692b46fb46903d6c57a8f7d6b261352c427d95b271a", "bbe3ccbb819697cb280255397e8d2ba9c25c9ad5d123e2a6dd9c777c37721ab4", "032f641fd80702527111d5328c4783ae41b4d071ebc7a4ed80d749e6f059c6d6"),
            "_extract_all_measure_um": ("a33a4e22e0a3849858d46e03a90e45fe2f04a872b1a549dab6076e1ba93b7e98", "cc195bf146de483f9e7088f82e2b5f455e5235f4693fa6f15151e222ef5e94f9", "84e7265c0afb013e809685be4dc3410a9eb2da13aa29548e2827f663ef0bfec0"),
            "_extract_substrate_like_unitless_um": ("2d30f96d4cc3f2ec4e5e2bf4f7f58c9f7d92128b5691c39f9c35b3d11758930b", "89f3f0595ee1579918065f746461f130a0ca26684de8545f8da9b3890f1f8b65", "bf1fd0c4eacd2b9c7a9a3c77178ed5eebbb926e44790e639fada50f66b134fcb"),
            "_extract_measure": ("c33713ee1aacfd70239ced247faeb3d70ea30214b227e59634371cf34aba7b35", "1cdfb57c5318883f8e929dccc4221f789bed4fdff76d86f1a196b98a0e7dab95", "4e70e7ce4efa8d5662883acc5334b3bf0e54b438428e2704c48e6862755b32ed"),
        }
        expected_measurement_globals = {
            "_first_explicit_unit": {"_NUMBER_PATTERN", "_UNIT_PATTERN", "_normalize_user_request", "re"},
            "_last_explicit_unit": {"_NUMBER_PATTERN", "_UNIT_PATTERN", "_normalize_user_request", "re"},
            "_unit_value_to_um": set(),
            "_extract_measure_mentions_um": {"List", "Tuple", "_NUMBER_PATTERN", "_UNIT_PATTERN", "_last_explicit_unit", "_normalize_user_request", "_number_span_is_identifier", "_unit_value_to_um", "re"},
            "_extract_unitless_measure_um": {"_NUMBER_PATTERN", "_UNIT_PATTERN", "_normalize_user_request", "_number_span_is_identifier", "_unit_value_to_um", "re"},
            "_number_span_is_identifier": {"re"},
            "_extract_measure_um": {"_extract_measure_mentions_um"},
            "_extract_all_measure_um": {"List", "_extract_measure_mentions_um"},
            "_extract_substrate_like_unitless_um": {"_NUMBER_PATTERN", "_UNIT_PATTERN", "_coerce_float", "_normalize_user_request", "re"},
            "_extract_measure": {"_normalize_user_request", "re"},
        }
        for name, (
            expected_source_hash,
            expected_ast_hash,
            expected_body_hash,
        ) in expected_measurement_function_hashes.items():
            function = function_node(measurements_tree, name)
            function_source = "".join(
                measurements_lines[function.lineno - 1:function.end_lineno]
            )
            parameters = {
                argument.arg for argument in ast.walk(function)
                if isinstance(argument, ast.arg)
            }
            assigned = {
                node.id for node in ast.walk(function)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            }
            loaded = {
                node.id for node in ast.walk(function)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            }
            with self.subTest(measurement_function=name):
                self.assertEqual(hashlib.sha256(function_source.encode("utf-8")).hexdigest(), expected_source_hash)
                self.assertEqual(hashlib.sha256(ast.dump(function, include_attributes=False).encode("utf-8")).hexdigest(), expected_ast_hash)
                self.assertEqual(hashlib.sha256(ast.dump(ast.Module(body=function.body, type_ignores=[]), include_attributes=False).encode("utf-8")).hexdigest(), expected_body_hash)
                self.assertEqual(loaded - parameters - assigned - builtin_names, expected_measurement_globals[name])

        remaining_function_hashes = {}
        for function in (
            node for node in planner_tree.body if isinstance(node, ast.FunctionDef)
        ):
            source = "".join(
                planner_lines[function.lineno - 1:function.end_lineno]
            )
            remaining_function_hashes[function.name] = {
                "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "ast_sha256": hashlib.sha256(ast.dump(function, include_attributes=False).encode("utf-8")).hexdigest(),
                "body_sha256": hashlib.sha256(ast.dump(ast.Module(body=function.body, type_ignores=[]), include_attributes=False).encode("utf-8")).hexdigest(),
            }
        self.assertEqual(len(remaining_function_hashes), 87)
        self.assertEqual(
            hashlib.sha256(json.dumps(remaining_function_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "04a3bb54b9dd845d1d248b050211e3e0364ee231e334c163aad14d0a89317a60",
        )

        primitive_cases = {
            "0,25 µm": 0.25,
            "250 nm": 0.25,
            "2 µm": 2.0,
            "3 um": 3.0,
            "0.004 mm": 4.0,
        }
        for text, expected in primitive_cases.items():
            self.assertEqual(planner._extract_measure_um(text), expected)
            self.assertEqual(measurements._extract_measure_um(text), expected)
        self.assertEqual(
            measurements._extract_all_measure_um("0.1 µm gold and 0.01 chrome"),
            [0.1, 0.01],
        )
        self.assertEqual(
            measurements._extract_all_measure_um("SU-8 photoresist, 2 µm"),
            [2.0],
        )
        self.assertTrue(measurements._number_span_is_identifier("SU-8", 3, 4))
        self.assertEqual(
            measurements._extract_substrate_like_unitless_um("silicon substrate 200"),
            200.0,
        )
        self.assertIsNone(
            measurements._extract_substrate_like_unitless_um("deposit 200")
        )
        self.assertTrue(np.isnan(measurements._unit_value_to_um(float("nan"), "mm")))
        self.assertTrue(np.isinf(measurements._unit_value_to_um(float("inf"), "nm")))
        self.assertIsNone(measurements._extract_measure_um("nan um"))
        self.assertEqual(measurements._extract_measure("500 nm"), "0.5 µm (500 nm)")

        planner_imports = {
            node.module
            for node in planner_tree.body
            if isinstance(node, ast.ImportFrom)
        }
        self.assertIn("fabstack.assistant.normalization", planner_imports)
        self.assertIn("fabstack.assistant.measurements", planner_imports)
        self.assertIn("fabstack.assistant.prompts", planner_imports)
        for leaf_name, leaf_tree in (
            ("measurements", measurements_tree),
            ("normalization", normalization_tree),
            ("prompts", prompt_tree),
        ):
            reverse_imports = {
                node.module
                for node in leaf_tree.body
                if isinstance(node, ast.ImportFrom)
            }
            reverse_imports.update(
                alias.name
                for node in leaf_tree.body
                if isinstance(node, ast.Import)
                for alias in node.names
            )
            with self.subTest(planner_leaf_dependencies=leaf_name):
                self.assertNotIn("fabstack.assistant.planner", reverse_imports)
                self.assertFalse(
                    any(
                        module and module.startswith("fabstack.assistant.planner.")
                        for module in reverse_imports
                    )
                )

        moved_names = (
            set(expected_prompt_hashes)
            | set(expected_normalization_hashes)
            | set(expected_measurement_constant_hashes)
            | set(expected_measurement_function_hashes)
        )
        planner_definitions = {
            node.name
            for node in planner_tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        }
        planner_assignments = {
            target.id
            for node in planner_tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertTrue(moved_names.isdisjoint(planner_definitions))
        self.assertTrue(moved_names.isdisjoint(planner_assignments))

        patched_model_filename = "stage5a-runtime-patched.gguf"
        with patch.object(planner, "MODEL_FILENAME", patched_model_filename):
            self.assertEqual(
                planner.LocalStepPlanner().model_path,
                resource_path(patched_model_filename),
            )

        from types import MethodType

        monkeypatched_planner = planner_stress.make_planner(mock_qwen=True)
        self.assertIsInstance(
            monkeypatched_planner._generate_command_json_with_model,
            MethodType,
        )
        self.assertIs(
            monkeypatched_planner._generate_command_json_with_model.__self__,
            monkeypatched_planner,
        )

        expected_wrappers = {
            "_display_scene_constraint_profile": """
def _display_scene_constraint_profile(self, sources, profile):
    return build_scene_constraint_profile(self, sources, profile)
""",
            "_display_material_column_segments_with_targets": """
def _display_material_column_segments_with_targets(
    self,
    active,
    active_heights,
    *,
    history_index=None,
    process_profile=None,
    active_structural_roles=None,
    active_planarizer_roles=None,
):
    return build_material_column_segments_with_targets(
        self,
        active,
        active_heights,
        history_index=history_index,
        process_profile=process_profile,
        active_structural_roles=active_structural_roles,
        active_planarizer_roles=active_planarizer_roles,
    )
""",
        }
        for method_name, expected_source in expected_wrappers.items():
            actual = function_node(mixin_tree, method_name)
            expected = ast.parse(expected_source).body[0]
            with self.subTest(delegate=method_name):
                self.assertEqual(
                    ast.dump(actual, include_attributes=False),
                    ast.dump(expected, include_attributes=False),
                )

        main_window_path = package_root / "presentation" / "main_window.py"
        main_window_tree = ast.parse(
            main_window_path.read_text(encoding="utf-8"),
            filename=str(main_window_path),
        )
        replay_delegate = function_node(
            main_window_tree,
            "_apply_process_step_to_current_xsec",
        )
        expected_replay_delegate = ast.parse(
            """
def _apply_process_step_to_current_xsec(self, step: dict) -> bool:
    return apply_process_step_to_current_xsec(self, step)
"""
        ).body[0]
        self.assertEqual(
            ast.dump(replay_delegate, include_attributes=False),
            ast.dump(expected_replay_delegate, include_attributes=False),
        )
        restored_flip = copy.deepcopy(
            function_node(main_window_tree, "flip_active_stack_from_top")
        )
        routed_flip_call = next(
            node
            for node in ast.walk(restored_flip)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "execute_ui_flip_operation"
        )
        old_flip_call = ast.parse(
            "self.xsec.flip_active_stack()",
            mode="eval",
        ).body
        routed_flip_call.func = old_flip_call.func
        routed_flip_call.args = []
        self.assertEqual(
            hashlib.sha256(
                ast.dump(restored_flip, include_attributes=False).encode("utf-8")
            ).hexdigest(),
            "b0f619cdebe0e0f6918160a179c00e4c2acfd88019cad2391064e8962a746e8a",
        )

        def operation_branch(function, operation):
            matches = []
            for node in ast.walk(function):
                test = getattr(node, "test", None)
                if (
                    isinstance(node, ast.If)
                    and isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == "op"
                    and len(test.ops) == 1
                    and isinstance(test.ops[0], ast.Eq)
                    and len(test.comparators) == 1
                    and isinstance(test.comparators[0], ast.Constant)
                    and test.comparators[0].value == operation
                ):
                    matches.append(node)
            if not matches:
                raise AssertionError(f"Missing operation branch: {operation}")
            return max(matches, key=lambda node: len(node.body))

        def routed_call_index_in_statements(statements, function_name):
            return next(
                index
                for index, node in enumerate(statements)
                if isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == function_name
            )

        def routed_call_index(branch, function_name):
            return routed_call_index_in_statements(branch.body, function_name)

        operations_mixin_path = (
            package_root / "presentation" / "mixins" / "operations_mixin.py"
        )
        operations_mixin_source = operations_mixin_path.read_text(encoding="utf-8")
        operations_mixin_tree = ast.parse(
            operations_mixin_source,
            filename=str(operations_mixin_path),
        )
        operations_mixin_lines = operations_mixin_source.splitlines(keepends=True)
        frozen_bond_helpers = {
            "_copy_bond_source_info": (
                "fff7c1712b27054c001d33da646b3f8a7c56bb84aeecafe72215c94e24a4cf27",
                "4db7b8253775a6a7c0e22e5f651c968776719e68a532bd2d2d92cc0f8d02b68a",
                "69c024e70a572160d5bed813b269de5d7ed70f10e2ae311f044d131130c09e4c",
            ),
            "_apply_bond_with_external_stack": (
                "690072425e85c7d46bdafad0ab0de460206879f2f7510f078a89f7813ff71a12",
                "a2e1ead78020084d7e0e95f5db42793c2393e75d5a4d8f6f6c3e56ddfaf16dd5",
                "6dcbbc428efcb759851426db50f16bac3842a4dd2e0d60c2c477178d1e7d2126",
            ),
        }
        for helper_name, (
            expected_source_hash,
            expected_ast_hash,
            expected_body_hash,
        ) in frozen_bond_helpers.items():
            helper = function_node(operations_mixin_tree, helper_name)
            helper_source = "".join(
                operations_mixin_lines[helper.lineno - 1:helper.end_lineno]
            )
            normalized_body = ast.Module(
                body=copy.deepcopy(helper.body),
                type_ignores=[],
            )
            for node in ast.walk(normalized_body):
                if isinstance(node, ast.Name) and node.id in {"self", "host"}:
                    node.id = "receiver"
            with self.subTest(frozen_bond_helper=helper_name):
                self.assertEqual(
                    hashlib.sha256(helper_source.encode("utf-8")).hexdigest(),
                    expected_source_hash,
                )
                self.assertEqual(
                    hashlib.sha256(
                        ast.dump(helper, include_attributes=False).encode("utf-8")
                    ).hexdigest(),
                    expected_ast_hash,
                )
                self.assertEqual(
                    hashlib.sha256(
                        ast.dump(normalized_body, include_attributes=False).encode(
                            "utf-8"
                        )
                    ).hexdigest(),
                    expected_body_hash,
                )
        restored_apply_operation = copy.deepcopy(
            function_node(operations_mixin_tree, "apply_operation")
        )
        external_bond_call = next(
            node
            for node in ast.walk(restored_apply_operation)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "execute_ui_external_bond_operation"
        )
        old_external_bond_call = ast.parse(
            "self._apply_bond_with_external_stack(mat_index, float(thick_um))",
            mode="eval",
        ).body
        external_bond_call.func = old_external_bond_call.func
        external_bond_call.args = old_external_bond_call.args
        simple_bond_call = next(
            node
            for node in ast.walk(restored_apply_operation)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "execute_ui_simple_bond_operation"
        )
        old_simple_bond_call = ast.parse(
            "self.xsec.bond_physical(mat_index, float(thick_um))",
            mode="eval",
        ).body
        simple_bond_call.func = old_simple_bond_call.func
        simple_bond_call.args = old_simple_bond_call.args
        self.assertEqual(
            hashlib.sha256(
                ast.dump(restored_apply_operation, include_attributes=False).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "99981661fd268f4c9394293c1815135fb0bc264b232a64dfa25e8cae5d30cc23",
        )
        lithography_branch = operation_branch(
            restored_apply_operation,
            "Lithography",
        )
        lithography_call_index = routed_call_index(
            lithography_branch,
            "execute_ui_material_operation",
        )
        self.assertIsInstance(
            lithography_branch.body[lithography_call_index + 1],
            ast.Expr,
        )
        self.assertEqual(
            ast.unparse(lithography_branch.body[lithography_call_index + 1]),
            "self.clear_litho_windows(refresh_plot=False)",
        )
        lithography_branch.body[
            lithography_call_index:lithography_call_index + 1
        ] = ast.parse(
            """
if hasattr(self.xsec, "pattern_pr_lithography"):
    self.xsec.pattern_pr_lithography(
        mat_index,
        windows,
        tone="negative" if invert_lithography else "positive",
    )
else:
    for s, e in windows:
        self.xsec.pattern_pr_open_window(mat_index, s, e)
"""
        ).body
        self.assertEqual(
            hashlib.sha256(
                ast.dump(restored_apply_operation, include_attributes=False).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "444c4ba11cac9a739e66614e8aacd378128d188e4b07fd20d9a2c9aad6110ada",
        )
        bake_branch = operation_branch(restored_apply_operation, "Bake/Annealing")
        bake_assignment_index = next(
            index
            for index, node in enumerate(bake_branch.body)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "execute_ui_bake_operation"
        )
        bake_branch.body[bake_assignment_index:bake_assignment_index + 1] = ast.parse(
            "baked_idx = self._bake_material(mat_index)"
        ).body
        bake_guard = next(
            node
            for node in bake_branch.body
            if isinstance(node, ast.If)
            and "baked_idx != mat_index" in ast.unparse(node.test)
        )
        bake_guard.body[0:0] = ast.parse(
            """
grid = self.xsec.grid
h, w = grid.shape
for x in range(w):
    col = grid[:, x]
    nz = np.where(col != self.VOID_INDEX)[0]
    if nz.size == 0:
        continue
    y0 = nz.min()
    top_mat = col[y0]
    if top_mat == mat_index:
        y = y0
        while y < h and col[y] == mat_index:
            col[y] = baked_idx
            y += 1
self.xsec.grid = grid
"""
        ).body
        self.assertEqual(
            hashlib.sha256(
                ast.dump(restored_apply_operation, include_attributes=False).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "0289187753841c3aafa65de6e276012b233318f833dadf34a3d7ebcdfcc71465",
        )
        etch_branch = operation_branch(restored_apply_operation, "Etch")
        positive_depth_branch = next(
            node
            for node in etch_branch.body
            if isinstance(node, ast.If)
            and node.orelse
            and any(
                isinstance(item, ast.Expr)
                and isinstance(item.value, ast.Call)
                and isinstance(item.value.func, ast.Name)
                and item.value.func.id == "execute_ui_material_operation"
                for item in node.orelse
            )
        )
        etch_call_index = routed_call_index_in_statements(
            positive_depth_branch.orelse,
            "execute_ui_material_operation",
        )
        positive_depth_branch.orelse[etch_call_index:etch_call_index + 1] = ast.parse(
            "self.xsec.etch_physical("
            "mat_index, depth_um=float(depth_um), mask=mask, mode=etch_mode)"
        ).body
        lift_off_branch = operation_branch(restored_apply_operation, "Lift-off")
        lift_off_try = next(
            node for node in lift_off_branch.body if isinstance(node, ast.Try)
        )
        lift_off_call_index = routed_call_index_in_statements(
            lift_off_try.body,
            "execute_ui_material_operation",
        )
        lift_off_try.body[lift_off_call_index:lift_off_call_index + 1] = ast.parse(
            "self.xsec.liftoff(mat_index)"
        ).body
        self.assertEqual(
            hashlib.sha256(
                ast.dump(restored_apply_operation, include_attributes=False).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "3e1f47149018f896245a8ba8d9b87bd361beba61c6c52e5004ddf2d46fdfca53",
        )
        ui_old_equivalents = {
            "Deposit": "self.xsec.deposit_physical(mat_index, float(thick_um))",
            "Spincoating": "self.xsec.spincoat_physical(mat_index, float(thick_um))",
            "Strip material": "self.xsec.strip_top_only(mat_index)",
        }
        for operation, old_source in ui_old_equivalents.items():
            branch = operation_branch(restored_apply_operation, operation)
            call_index = routed_call_index(branch, "execute_ui_material_operation")
            branch.body[call_index:call_index + 1] = ast.parse(old_source).body
        clean_branch = operation_branch(
            restored_apply_operation,
            "Clean / Surface prep",
        )
        clean_call_index = routed_call_index(
            clean_branch,
            "execute_ui_material_operation",
        )
        del clean_branch.body[clean_call_index]
        self.assertEqual(
            hashlib.sha256(
                ast.dump(restored_apply_operation, include_attributes=False).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "86d6351b012c7de59a754e7ddf05165bfb76c662e54f15bf1178605205084fd4",
        )

        execution_path = package_root / "operations" / "execution.py"
        execution_tree = ast.parse(
            execution_path.read_text(encoding="utf-8"),
            filename=str(execution_path),
        )
        restored_replay = copy.deepcopy(
            function_node(execution_tree, "apply_process_step_to_current_xsec")
        )
        bond_branch = operation_branch(restored_replay, "Bond")
        bond_call_index = routed_call_index(
            bond_branch,
            "execute_replay_bond_operation",
        )
        bond_branch.body[bond_call_index:bond_call_index + 1] = ast.parse(
            """
if isinstance(source_info, dict):
    previous_source = host.bond_source_info
    try:
        host.bond_source_info = source_info
        host._apply_bond_with_external_stack(mat_index, thickness_um)
    finally:
        host.bond_source_info = previous_source
else:
    if hasattr(host.xsec, "bond_physical"):
        host.xsec.bond_physical(mat_index, thickness_um)
    else:
        host.xsec.bond(mat_index, thickness_px=thickness_px)
"""
        ).body
        self.assertEqual(
            hashlib.sha256(
                ast.dump(restored_replay, include_attributes=False).encode("utf-8")
            ).hexdigest(),
            "ed610136a2830864d14932efac95bfdb7c245575b85f6901389bd228a83bca39",
        )
        lithography_branch = operation_branch(restored_replay, "Lithography")
        lithography_call_index = routed_call_index(
            lithography_branch,
            "execute_replay_material_operation",
        )
        lithography_branch.body[
            lithography_call_index:lithography_call_index + 1
        ] = ast.parse(
            """
if hasattr(host.xsec, "pattern_pr_lithography"):
    host.xsec.pattern_pr_lithography(
        mat_index,
        windows,
        tone="negative" if invert_lithography else "positive",
    )
else:
    for s, e in windows:
        host.xsec.pattern_pr_open_window(mat_index, float(s), float(e))
"""
        ).body
        self.assertEqual(
            hashlib.sha256(
                ast.dump(restored_replay, include_attributes=False).encode("utf-8")
            ).hexdigest(),
            "553160cd4018c5a2bb811b12c314f4eec27ca28e7aae041fa06cfd78100dbbf6",
        )
        bake_branch = operation_branch(restored_replay, "Bake/Annealing")
        bake_call_index = routed_call_index(
            bake_branch,
            "execute_replay_bake_operation",
        )
        bake_branch.body[bake_call_index:bake_call_index + 1] = ast.parse(
            """
baked_idx = host._bake_material(mat_index)
if baked_idx != mat_index:
    grid = host.xsec.grid
    h, w = grid.shape
    for x in range(w):
        col = grid[:, x]
        nz = np.where(col != host.VOID_INDEX)[0]
        if nz.size == 0:
            continue
        y0 = nz.min()
        if col[y0] != mat_index:
            continue
        y = y0
        while y < h and col[y] == mat_index:
            col[y] = baked_idx
            y += 1
    host.xsec.grid = grid
"""
        ).body
        flip_branch = operation_branch(restored_replay, "Flip")
        flip_call_index = routed_call_index(
            flip_branch,
            "execute_replay_flip_operation",
        )
        flip_branch.body[flip_call_index:flip_call_index + 1] = ast.parse(
            "host.xsec.flip_active_stack()"
        ).body
        self.assertEqual(
            hashlib.sha256(
                ast.dump(restored_replay, include_attributes=False).encode("utf-8")
            ).hexdigest(),
            "802c83322572db0562fd5d8428e487414654b523ab83285310ba1edaec7b52c9",
        )
        etch_branch = operation_branch(restored_replay, "Etch")
        etch_call_index = routed_call_index(
            etch_branch,
            "execute_replay_material_operation",
        )
        etch_branch.body[etch_call_index:etch_call_index + 1] = ast.parse(
            """
if hasattr(host.xsec, "etch_physical"):
    host.xsec.etch_physical(
        mat_index,
        depth_um=depth_um,
        mask=mask,
        mode=etch_mode,
    )
else:
    depth_px = host.um_to_process_px(depth_um, mode="ceil", minimum=1)
    host.xsec.etch(
        mat_index,
        depth_px=depth_px,
        mask=mask,
        mode=etch_mode,
    )
"""
        ).body
        lift_off_branch = operation_branch(restored_replay, "Lift-off")
        lift_off_call_index = routed_call_index(
            lift_off_branch,
            "execute_replay_material_operation",
        )
        lift_off_branch.body[
            lift_off_call_index:lift_off_call_index + 1
        ] = ast.parse("host.xsec.liftoff(mat_index)").body
        self.assertEqual(
            hashlib.sha256(
                ast.dump(restored_replay, include_attributes=False).encode("utf-8")
            ).hexdigest(),
            "2725ea1f2a1c6228c39cce25689ee42246898426b2dd19de8d6a2a33c18b8565",
        )
        replay_old_equivalents = {
            "Deposit": """
if hasattr(host.xsec, "deposit_physical"):
    host.xsec.deposit_physical(mat_index, thickness_um)
else:
    thickness_px = host.um_to_process_px(thickness_um, mode="ceil", minimum=1)
    host.xsec.deposit(mat_index, thickness_px=thickness_px)
""",
            "Spincoating": """
if hasattr(host.xsec, "spincoat_physical"):
    host.xsec.spincoat_physical(mat_index, thickness_um)
else:
    thickness_px = host.um_to_process_px(thickness_um, mode="ceil", minimum=1)
    host.xsec.spincoat(mat_index, thickness_px=thickness_px)
""",
        }
        for operation, old_source in replay_old_equivalents.items():
            branch = operation_branch(restored_replay, operation)
            call_index = routed_call_index(
                branch,
                "execute_replay_material_operation",
            )
            branch.body[call_index:call_index + 1] = ast.parse(old_source).body
        strip_branch = operation_branch(restored_replay, "Strip material")
        strip_call_index = routed_call_index(
            strip_branch,
            "execute_replay_material_operation",
        )
        strip_branch.body[strip_call_index:strip_call_index + 1] = ast.parse(
            "host.xsec.strip_top_only(mat_index)"
        ).body
        clean_branch = operation_branch(restored_replay, "Clean / Surface prep")
        clean_call_index = routed_call_index(
            clean_branch,
            "execute_replay_material_operation",
        )
        del clean_branch.body[clean_call_index]
        self.assertEqual(
            hashlib.sha256(
                ast.dump(restored_replay, include_attributes=False).encode("utf-8")
            ).hexdigest(),
            "708d41a559a5363da07dbea0f42b859982560dbe55bf72243d8c353be74d4703",
        )

        assistant_path = (
            package_root / "presentation" / "mixins" / "llm_assistant_mixin.py"
        )
        assistant_source = assistant_path.read_text(encoding="utf-8")
        assistant_tree = ast.parse(assistant_source, filename=str(assistant_path))
        assistant_lines = assistant_source.splitlines(keepends=True)
        frozen_assistant_functions = {
            "_run_assistant_command_sequence": (
                "3bd53882841b974e8ee8547ae4e45effaa3e7847d7aec9c5e41382ce973ee1d6",
                "ac1859fd0042707aee9ef69bba75ae777f6a71cbbdc2f05f646afe3c304892db",
            ),
            "_assistant_command_is_manual_lithography": (
                "9cab21e0fb579ca1e7a73ac7c6f39703eacf6138e29c0b5eb004dde00c23c53d",
                "a1599d3139d8cd7d8f97f1587a8ab1218c53cbef882b15b9b077220095b13f5e",
            ),
            "_assistant_is_paused_for_manual_lithography": (
                "bc557d0ac3bb55aa72feb14cb1b4f50e9bb943f10b126e46e1cb17b5d91dfade",
                "c31b4a71cee33634ddd53935b228adff67664b1cd8f0db6d0f5e48be5c8ed79f",
            ),
            "_pause_assistant_for_manual_lithography": (
                "eb01f3dc7366f9cf04cde4531a25ca82f82fa2251d88e732058c2bb3c4eccc90",
                "2f832d4e7d2e7a4d4bb8182e468d629691a4050facdf6fbc20f5eb7bd0b04bd8",
            ),
            "_prepare_manual_lithography": (
                "08e5cd86dde20f6a80b79f11f8597b36bf07f1e11ebb40616c294742ca379693",
                "48cfa6d01a1d3ee06dcc0f592421d67b56d1f0859263bcfc6ac62c0e2614c802",
            ),
            "_continue_assistant_manual_lithography": (
                "b385c40b213baf68ec712fc2d9f238d73a0cbe891b600e17e4111caf89a3c17b",
                "818b098777896d3690ac087b772255e2af3f0b208431ee325a73ab1ce764192d",
            ),
            "_execute_assistant_command": (
                "713d637ad128b74e79cd9b7b0370ff6ac135f68726414f585adf311a23bf3b70",
                "82fa040deb8eae0dfe331a2664fb49afaa0a120e74fb770c24fab8a4df9bc026",
            ),
            "_validate_assistant_commands": (
                "3ce23d27fa0bb412f5bed3aae23ec0725d1ae8dc1266b327e04f6e7de02532f8",
                "1109818c6ce64aee5b9010a189b88fcc67204517222fe7c3ed4ce2154ba61a35",
            ),
            "_assistant_runtime_material_for_operation": (
                "621eb3512a40a3074ffeafd3f8b8880b8a5752edb1b007b882e7dccd0a7cd730",
                "fbd6b67d22c32d3b5c975a4c1ff2e2127a88b1b41ffea775eae7bf8e5a287c1f",
            ),
        }
        for function_name, (
            expected_source_hash,
            expected_ast_hash,
        ) in frozen_assistant_functions.items():
            function = function_node(assistant_tree, function_name)
            function_source = "".join(
                assistant_lines[function.lineno - 1:function.end_lineno]
            )
            with self.subTest(frozen_assistant_function=function_name):
                self.assertEqual(
                    hashlib.sha256(function_source.encode("utf-8")).hexdigest(),
                    expected_source_hash,
                )
                self.assertEqual(
                    hashlib.sha256(
                        ast.dump(function, include_attributes=False).encode("utf-8")
                    ).hexdigest(),
                    expected_ast_hash,
                )

        persistence_path = package_root / "persistence" / "process_file.py"
        persistence_source = persistence_path.read_text(encoding="utf-8")
        persistence_tree = ast.parse(
            persistence_source,
            filename=str(persistence_path),
        )
        persistence_validator = function_node(
            persistence_tree,
            "_validate_process_step",
        )
        persistence_lines = persistence_source.splitlines(keepends=True)
        persistence_validator_source = "".join(
            persistence_lines[
                persistence_validator.lineno - 1:persistence_validator.end_lineno
            ]
        )
        self.assertEqual(
            hashlib.sha256(
                persistence_validator_source.encode("utf-8")
            ).hexdigest(),
            "f9f77e7092e17580d50619c436f90ec711d1a1321c2478d5d69e57ad74025437",
        )
        self.assertEqual(
            hashlib.sha256(
                ast.dump(persistence_validator, include_attributes=False).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "ab6d6dbc96455b1776d9e327e1af2ec1d7604356cf0a5b04bbf95aa98c898670",
        )

        class NormalizeReceiver(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id in {"self", "host"}:
                    node.id = "receiver"
                return node

        expected_body_hashes = {
            Path("rendering/scene_constraints.py"): (
                "build_scene_constraint_profile",
                "c637eaeea5bd7b7b793417c4602a4d3e7f9f509c99fb96d6f355d3bea44d85e4",
            ),
            Path("rendering/segment_targets.py"): (
                "build_material_column_segments_with_targets",
                "eaa4866345bab5bd55be8ac5e27da5f4dfbd4a105b7b0c081db8b747065b9002",
            ),
        }
        for relative_path, (function_name, expected_hash) in expected_body_hashes.items():
            path = package_root / relative_path
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            body = ast.Module(body=function_node(tree, function_name).body, type_ignores=[])
            normalized_body = NormalizeReceiver().visit(body)
            body_dump = ast.dump(normalized_body, include_attributes=False)
            with self.subTest(service_body=str(relative_path)):
                self.assertEqual(
                    hashlib.sha256(body_dump.encode("utf-8")).hexdigest(),
                    expected_hash,
                )
        self.assertEqual(
            PlottingExportMixin.__bases__,
            (
                DisplayProfileMixin,
                VisualAdjustmentMixin,
                CanvasRenderingMixin,
                ExportActionsMixin,
            ),
        )
        expected_plotting_methods = {
            "_active_history_index_for_view",
            "_add_export_logo_to_axis",
            "_apportion_display_rows",
            "_bond_source_export_lines",
            "_clear_visual_thickness_highlight",
            "_clear_visual_thickness_override_from_pick",
            "_collect_dynamic_visual_overrides",
            "_configure_word_export_page",
            "_current_material_column_total_um",
            "_display_active_band_data",
            "_display_adjustment_sources",
            "_display_base_feature_rows",
            "_display_contiguous_column_segments",
            "_display_contiguous_solid_stack_keys",
            "_display_feature_profile_key",
            "_display_grid_auto",
            "_display_grid_row_bands",
            "_display_height_for_current_view",
            "_display_material_column_segments_with_targets",
            "_display_material_count_profile_key",
            "_display_nominal_substrate_um",
            "_display_partial_core_reserve_rows",
            "_display_photoresist_indices",
            "_display_physical_order_margin",
            "_display_physical_target_height",
            "_display_profile_feature_rows",
            "_display_raw_material_columns",
            "_display_row_edges_um",
            "_display_row_from_plot_y",
            "_display_row_heights_array",
            "_display_scene_constraint_profile",
            "_display_segment_target_height",
            "_display_structural_body_nominals",
            "_display_structural_body_rows",
            "_display_substrate_core_target_height",
            "_display_substrate_material_indices",
            "_display_substrate_profile",
            "_display_y_for_physical_y_in_column",
            "_draw_visual_geometry_overlays",
            "_etch_fill_intervals",
            "_feature_scale_factor_clamped",
            "_format_visual_thickness_um",
            "_handle_visual_thickness_canvas_pick",
            "_highlight_visual_layer_pick",
            "_image_array_to_uint8_rgb",
            "_include_export_logo",
            "_normalised_visualisation_mode",
            "_open_etch_void_intervals",
            "_planarizer_roles_for_plot_grid",
            "_plot_y_for_visual_scale",
            "_preview_main_subplots_adjust",
            "_process_display_adjustment_profile",
            "_render_material_columns_from_targets",
            "_render_word_step_figure_payload",
            "_resize_uint8_rgb_width",
            "_row_heights_for_plot_grid",
            "_run_export_action",
            "_save_cropped_figure",
            "_save_uint8_rgb_image",
            "_save_word_step_figure",
            "_save_word_step_images_with_consistent_stack_width",
            "_scaled_stack_vertical_span",
            "_scroll_figure_view_to_bottom",
            "_set_process_visual_override",
            "_set_visual_thickness_override_from_pick",
            "_set_visualisation_mode",
            "_show_export_status",
            "_show_nothing_to_export",
            "_show_visual_thickness_pick_dialog",
            "_structural_roles_for_plot_grid",
            "_update_visualisation_controls_state",
            "_visual_geometry_for_plot_grid",
            "_visual_layer_pick_from_event",
            "_visual_override_canvas_height_scale",
            "_visual_override_value_for_key",
            "_visual_overrides_enabled",
            "_visual_render_scale_factor",
            "_visual_scale_factor_clamped",
            "_visual_step_override_values",
            "_visual_thickness_overrides",
            "_visual_thickness_step_overrides",
            "_word_export_image_width_inches",
            "accept_visualisation",
            "export_current_view",
            "export_history_image",
            "export_process_word",
            "export_step_list_txt",
            "export_step_list_word",
            "finalise_visualisation",
            "on_canvas_mouse_move",
            "on_canvas_mouse_press",
            "on_canvas_mouse_release",
            "open_visual_thickness_dialog",
            "plot_grid",
            "reject_visualisation",
            "reset_visualisation_for_process_change",
            "update_plot",
        }
        actual_plotting_methods = {
            name
            for base in PlottingExportMixin.__bases__
            for name, value in base.__dict__.items()
            if callable(value) and not name.startswith("__")
        }
        self.assertEqual(len(expected_plotting_methods), 97)
        self.assertEqual(actual_plotting_methods, expected_plotting_methods)
        self.assertIs(legacy_library.LibraryManagerDialog, LibraryManagerDialog)
        self.assertEqual(
            legacy_library.DEFAULT_DEPOSITION_METHODS,
            DEFAULT_DEPOSITION_METHODS,
        )
        self.assertIs(legacy_resources.resource_path, resource_path)
        self.assertEqual(
            legacy_visualization.VISUAL_SCALE_MAX,
            visualization.VISUAL_SCALE_MAX,
        )
        self.assertIs(legacy_regression.main, dimension_preservation.main)
        self.assertIs(legacy_stress.main, planner_stress.main)


class ReleaseMetadataTests(unittest.TestCase):
    def test_project_metadata_matches_runtime_version(self):
        path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        metadata = path.read_text(encoding="utf-8")
        self.assertIn(f'version = "{__version__}"', metadata)


if __name__ == "__main__":
    unittest.main()
