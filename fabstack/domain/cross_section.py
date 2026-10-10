import copy

import numpy as np


_EPS_UM = 1e-12


class CrossSection:
    """Physically dimensioned 2D material cross-section.

    Grid cells contain material indices; the application owns their names and
    colors. Row heights retain exact micrometer dimensions independently from
    the adaptive display grid.
    """

    def __init__(
        self,
        width_px: int = 200,
        height_px: int = 400,
        substrate_index: int = 1,
        void_index: int = 0,
        substrate_thickness_px: int = 200,
        substrate_thickness_um: float = None,
        top_headroom_um: float = None,
    ):
        self.width = int(width_px)
        self.height = int(height_px)
        self.void_index = void_index
        # Keep the identity of the structural body with the geometry.  The
        # renderer and validators must not have to infer the substrate from a
        # mutable UI combo box, especially after replay, flip, or bonding.
        self.substrate_index = int(substrate_index)
        self.visual_geometry = []

        if substrate_thickness_um is None:
            substrate_um = float(max(1, int(substrate_thickness_px)))
        else:
            substrate_um = max(0.0, float(substrate_thickness_um))
        self.nominal_substrate_thickness_um = float(substrate_um)

        if top_headroom_um is None:
            top_headroom = max(0.0, float(height_px) - substrate_um)
        else:
            top_headroom = max(0.0, float(top_headroom_um))

        rows = []
        heights = []
        if top_headroom > _EPS_UM:
            rows.append(np.full(self.width, void_index, dtype=np.int16))
            heights.append(top_headroom)
        if substrate_um > _EPS_UM:
            rows.append(np.full(self.width, substrate_index, dtype=np.int16))
            heights.append(substrate_um)
        if not rows:
            rows.append(np.full(self.width, void_index, dtype=np.int16))
            heights.append(1.0)

        self.grid = np.vstack(rows).astype(np.int16, copy=False)
        self.row_heights_um = np.asarray(heights, dtype=float)
        # Structural provenance is independent of material identity.  This is
        # essential for bonding: two silicon wafers (and a deposited silicon
        # film) must remain distinguishable even when their material indices
        # are identical.
        self.structural_role_grid = np.zeros_like(self.grid, dtype=np.int16)
        if substrate_um > _EPS_UM:
            substrate_row = len(rows) - 1
            self.structural_role_grid[substrate_row, :] = 1
        self.structural_bodies = {
            1: {
                "body_id": 1,
                "material_index": int(substrate_index),
                "nominal_thickness_um": float(substrate_um),
                "kind": "substrate",
            }
        }
        self._next_structural_body_id = 2
        # Spin-coated material has two persistent semantic regions: the
        # requested overburden above the highest pre-coat surface and the
        # additional fill over lower topography.  Positive ids mark
        # overburden and negative ids mark fill for the same coating event.
        # Keeping this provenance with the cells prevents later lithography,
        # bonding, or deposition steps from reinterpreting a planar film as
        # one generic nonlinear-thickness run.
        self.planarizer_role_grid = np.zeros_like(self.grid, dtype=np.int32)
        self.planarizer_roles = {}
        self._next_planarizer_role_id = 1
        self.height, self.width = self.grid.shape

    # ---------- helpers ----------

    def _ensure_row_heights(self):
        if (
            not hasattr(self, "row_heights_um")
            or self.row_heights_um is None
            or len(self.row_heights_um) != self.grid.shape[0]
        ):
            self.row_heights_um = np.ones(self.grid.shape[0], dtype=float)

    def _ensure_structural_roles(self):
        if (
            not hasattr(self, "structural_role_grid")
            or self.structural_role_grid is None
            or np.asarray(self.structural_role_grid).shape != self.grid.shape
        ):
            roles = np.zeros_like(self.grid, dtype=np.int16)
            substrate_index = int(getattr(self, "substrate_index", -1))
            candidates = []
            for col in range(self.grid.shape[1]):
                row = 0
                while row < self.grid.shape[0]:
                    material = int(self.grid[row, col])
                    end = row + 1
                    while end < self.grid.shape[0] and int(self.grid[end, col]) == material:
                        end += 1
                    if material == substrate_index:
                        height = float(np.sum(self.row_heights_um[row:end]))
                        candidates.append((height, row, end, col))
                    row = end
            if candidates:
                _height, start, end, _col = max(candidates)
                material_mask = self.grid[start:end, :] == substrate_index
                role_block = roles[start:end, :]
                role_block[material_mask] = 1
                roles[start:end, :] = role_block
            self.structural_role_grid = roles
        else:
            self.structural_role_grid = np.asarray(
                self.structural_role_grid,
                dtype=np.int16,
            ).copy()

        if not isinstance(getattr(self, "structural_bodies", None), dict):
            nominal = float(getattr(self, "nominal_substrate_thickness_um", 0.0) or 0.0)
            self.structural_bodies = {
                1: {
                    "body_id": 1,
                    "material_index": int(getattr(self, "substrate_index", -1)),
                    "nominal_thickness_um": nominal,
                    "kind": "substrate",
                }
            }
        # Be defensive with partially upgraded snapshots: role ids are the
        # authoritative geometry identity, so synthesize missing metadata and
        # never reuse an id that is already present in the role grid.
        role_ids = sorted(int(value) for value in np.unique(self.structural_role_grid) if int(value) > 0)
        for body_id in role_ids:
            if body_id in self.structural_bodies or str(body_id) in self.structural_bodies:
                continue
            cells = np.where(self.structural_role_grid == body_id)
            material_index = (
                int(self.grid[cells[0][0], cells[1][0]])
                if cells[0].size else int(getattr(self, "substrate_index", -1))
            )
            column_heights = []
            for col in range(self.grid.shape[1]):
                mask = self.structural_role_grid[:, col] == body_id
                if np.any(mask):
                    column_heights.append(float(np.sum(self.row_heights_um[mask])))
            self.structural_bodies[body_id] = {
                "body_id": int(body_id),
                "material_index": material_index,
                "nominal_thickness_um": max(column_heights, default=0.0),
                "kind": "substrate" if body_id == 1 else "bonded_substrate",
            }
        existing = set(role_ids)
        existing.update(int(key) for key in self.structural_bodies.keys() if int(key) > 0)
        self._next_structural_body_id = max(
            int(getattr(self, "_next_structural_body_id", 2) or 2),
            (max(existing) + 1) if existing else 2,
        )

    def _new_structural_body(self, material_index: int, thickness_um: float, *, kind="bond"):
        self._ensure_structural_roles()
        self._ensure_planarizer_roles()
        body_id = int(self._next_structural_body_id)
        self._next_structural_body_id += 1
        self.structural_bodies[body_id] = {
            "body_id": body_id,
            "material_index": int(material_index),
            "nominal_thickness_um": float(max(0.0, thickness_um)),
            "kind": str(kind or "bond"),
        }
        return body_id

    def _ensure_planarizer_roles(self):
        if (
            not hasattr(self, "planarizer_role_grid")
            or self.planarizer_role_grid is None
            or np.asarray(self.planarizer_role_grid).shape != self.grid.shape
        ):
            self.planarizer_role_grid = np.zeros_like(self.grid, dtype=np.int32)
        else:
            self.planarizer_role_grid = np.asarray(
                self.planarizer_role_grid,
                dtype=np.int32,
            ).copy()
        if not isinstance(getattr(self, "planarizer_roles", None), dict):
            self.planarizer_roles = {}
        existing = {
            abs(int(value))
            for value in np.unique(self.planarizer_role_grid)
            if int(value) != 0
        }
        existing.update(
            int(key) for key in self.planarizer_roles
            if str(key).lstrip("-").isdigit() and int(key) > 0
        )
        self._next_planarizer_role_id = max(
            int(getattr(self, "_next_planarizer_role_id", 1) or 1),
            (max(existing) + 1) if existing else 1,
        )

    def _new_planarizer_role(
        self,
        material_index: int,
        thickness_um: float,
        *,
        fill_thickness_um_by_column=None,
        orientation=1,
    ):
        self._ensure_planarizer_roles()
        role_id = int(self._next_planarizer_role_id)
        self._next_planarizer_role_id += 1
        self.planarizer_roles[role_id] = {
            "role_id": role_id,
            "material_index": int(material_index),
            "requested_thickness_um": float(max(0.0, thickness_um)),
            "fill_thickness_um_by_column": [
                float(max(0.0, value)) if np.isfinite(value) else 0.0
                for value in (
                    fill_thickness_um_by_column
                    if fill_thickness_um_by_column is not None
                    else np.zeros(self.width, dtype=float)
                )
            ],
            "orientation": 1 if int(orientation or 1) >= 0 else -1,
        }
        return role_id

    def _row_edges_um(self):
        self._ensure_row_heights()
        return np.concatenate(([0.0], np.cumsum(self.row_heights_um)))

    def _discard_visual_geometry_type(self, geometry_type: str):
        """Remove display overlays that no longer describe current geometry."""
        self.visual_geometry = [
            item
            for item in (getattr(self, "visual_geometry", []) or [])
            if not isinstance(item, dict) or item.get("type") != geometry_type
        ]

    def _merge_adjacent_equal_rows(self):
        self._ensure_row_heights()
        self._ensure_structural_roles()
        self._ensure_planarizer_roles()
        if self.grid.shape[0] <= 1:
            return

        rows = []
        role_rows = []
        planarizer_rows = []
        heights = []
        current = self.grid[0].copy()
        current_role = self.structural_role_grid[0].copy()
        current_planarizer = self.planarizer_role_grid[0].copy()
        current_height = float(self.row_heights_um[0])

        for i in range(1, self.grid.shape[0]):
            height = float(self.row_heights_um[i])
            if (
                np.array_equal(self.grid[i], current)
                and np.array_equal(self.structural_role_grid[i], current_role)
                and np.array_equal(self.planarizer_role_grid[i], current_planarizer)
            ):
                current_height += height
                continue
            rows.append(current)
            role_rows.append(current_role)
            planarizer_rows.append(current_planarizer)
            heights.append(current_height)
            current = self.grid[i].copy()
            current_role = self.structural_role_grid[i].copy()
            current_planarizer = self.planarizer_role_grid[i].copy()
            current_height = height

        rows.append(current)
        role_rows.append(current_role)
        planarizer_rows.append(current_planarizer)
        heights.append(current_height)
        self.grid = np.vstack(rows).astype(self.grid.dtype, copy=False)
        self.structural_role_grid = np.vstack(role_rows).astype(np.int16, copy=False)
        self.planarizer_role_grid = np.vstack(planarizer_rows).astype(np.int32, copy=False)
        self.row_heights_um = np.asarray(heights, dtype=float)
        self.height, self.width = self.grid.shape

    def _shift_visual_geometry_y(self, shift_um: float):
        try:
            shift_um = float(shift_um)
        except (TypeError, ValueError):
            return
        if not np.isfinite(shift_um) or abs(shift_um) <= _EPS_UM:
            return

        def shift_points(points):
            shifted = []
            for point in points or []:
                try:
                    x, y = point
                    shifted.append((float(x), float(y) + shift_um))
                except (TypeError, ValueError):
                    continue
            return shifted

        for item in self.visual_geometry:
            if not isinstance(item, dict):
                continue
            if "polyline" in item:
                item["polyline"] = shift_points(item.get("polyline"))
            if "polylines" in item:
                item["polylines"] = [
                    shift_points(polyline)
                    for polyline in item.get("polylines", [])
                ]
            if "y_surface_um" in item:
                try:
                    item["y_surface_um"] = float(item["y_surface_um"]) + shift_um
                except (TypeError, ValueError):
                    pass

    def _clear_visual_geometry(self):
        self.visual_geometry = []

    def _mask_segments(self, mask):
        mask = np.asarray(mask, dtype=bool)
        segments = []
        in_segment = False
        start = 0
        for idx, value in enumerate(mask):
            if value and not in_segment:
                start = int(idx)
                in_segment = True
            elif not value and in_segment:
                segments.append((start, int(idx) - 1))
                in_segment = False
        if in_segment:
            segments.append((start, int(mask.shape[0]) - 1))
        return segments

    def _append_etch_depth_feature(
        self,
        material_index: int,
        mode: str,
        removed_intervals,
        requested_depth_um: float,
    ):
        by_column = {}
        for item in removed_intervals or []:
            try:
                x, y0, y1 = item
            except (TypeError, ValueError):
                continue
            y0 = float(y0)
            y1 = float(y1)
            if y1 <= y0 + _EPS_UM:
                continue
            by_column.setdefault(int(x), []).append((y0, y1))

        normalized = []
        for x, intervals in by_column.items():
            intervals = sorted(intervals)
            if not intervals:
                continue
            cur_y0, cur_y1 = intervals[0]
            for y0, y1 in intervals[1:]:
                if y0 <= cur_y1 + _EPS_UM:
                    cur_y1 = max(cur_y1, y1)
                    continue
                normalized.append((x, cur_y0, cur_y1))
                cur_y0, cur_y1 = y0, y1
            normalized.append((x, cur_y0, cur_y1))

        grouped = {}
        for item in normalized:
            try:
                x, y0, y1 = item
            except (TypeError, ValueError):
                continue
            depth = float(y1) - float(y0)
            if depth <= _EPS_UM:
                continue
            key = (round(float(y0), 12), round(float(y1), 12))
            grouped.setdefault(key, []).append(int(x))

        features = []
        for (y0, y1), cols in grouped.items():
            for x_start, x_end in self._contiguous_column_segments(cols):
                features.append(
                    {
                        "x_start": int(x_start),
                        "x_end": int(x_end),
                        "surface_y_um": float(y0),
                        "bottom_y_um": float(y1),
                        "depth_um": float(y1 - y0),
                    }
                )

        if not features:
            return

        self.visual_geometry.append(
            {
                "type": "etch_depth_feature",
                "material_index": int(material_index),
                "mode": str(mode or "anisotropic").lower(),
                "requested_depth_um": float(requested_depth_um),
                "features": features,
            }
        )

    def _contiguous_column_segments(self, cols):
        cols = sorted({int(col) for col in cols})
        if not cols:
            return []
        segments = []
        start = prev = cols[0]
        for col in cols[1:]:
            if col == prev + 1:
                prev = col
                continue
            segments.append((start, prev))
            start = prev = col
        segments.append((start, prev))
        return segments

    def _insert_boundaries_um(self, boundaries_um):
        """
        Split rows at physical y positions measured from the current top.

        Returns the downward shift applied to input coordinates if new top
        headroom was inserted for negative boundaries.
        """
        self._ensure_row_heights()
        boundaries = []
        for value in boundaries_um or []:
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                boundaries.append(value)

        if not boundaries:
            return 0.0

        shift = 0.0
        min_boundary = min(boundaries)
        if min_boundary < -_EPS_UM:
            shift = -min_boundary
            self._ensure_structural_roles()
            self._ensure_planarizer_roles()
            pad = np.full((1, self.width), self.void_index, dtype=self.grid.dtype)
            self.grid = np.vstack((pad, self.grid))
            self.structural_role_grid = np.vstack((
                np.zeros((1, self.width), dtype=np.int16),
                self.structural_role_grid,
            ))
            self.planarizer_role_grid = np.vstack((
                np.zeros((1, self.width), dtype=np.int32),
                self.planarizer_role_grid,
            ))
            self.row_heights_um = np.concatenate(([shift], self.row_heights_um))
            self.height, self.width = self.grid.shape
            self._shift_visual_geometry_y(shift)
            boundaries = [value + shift for value in boundaries]

        edges = self._row_edges_um()
        total = float(edges[-1])
        internal = sorted(
            {
                round(value, 12)
                for value in boundaries
                if _EPS_UM < value < total - _EPS_UM
            }
        )
        if not internal:
            return shift

        new_rows = []
        new_role_rows = []
        new_planarizer_rows = []
        new_heights = []
        self._ensure_structural_roles()
        self._ensure_planarizer_roles()
        boundary_idx = 0
        for row_idx in range(self.grid.shape[0]):
            start = float(edges[row_idx])
            end = float(edges[row_idx + 1])
            cuts = [start]
            while boundary_idx < len(internal) and internal[boundary_idx] <= start + _EPS_UM:
                boundary_idx += 1
            scan_idx = boundary_idx
            while scan_idx < len(internal) and internal[scan_idx] < end - _EPS_UM:
                cuts.append(float(internal[scan_idx]))
                scan_idx += 1
            cuts.append(end)

            for cut_start, cut_end in zip(cuts, cuts[1:]):
                height = float(cut_end - cut_start)
                if height <= _EPS_UM:
                    continue
                new_rows.append(self.grid[row_idx].copy())
                new_role_rows.append(self.structural_role_grid[row_idx].copy())
                new_planarizer_rows.append(self.planarizer_role_grid[row_idx].copy())
                new_heights.append(height)

        self.grid = np.vstack(new_rows).astype(self.grid.dtype, copy=False)
        self.structural_role_grid = np.vstack(new_role_rows).astype(np.int16, copy=False)
        self.planarizer_role_grid = np.vstack(new_planarizer_rows).astype(np.int32, copy=False)
        self.row_heights_um = np.asarray(new_heights, dtype=float)
        self.height, self.width = self.grid.shape
        return shift

    def _row_indices_inside_um(self, y0: float, y1: float):
        if y1 <= y0 + _EPS_UM:
            return np.asarray([], dtype=int)
        edges = self._row_edges_um()
        return np.where((edges[:-1] >= y0 - _EPS_UM) & (edges[1:] <= y1 + _EPS_UM))[0]

    def _surface_tops_um(self):
        edges = self._row_edges_um()
        h, w = self.grid.shape
        # A through-hole has no local support surface. NaN lets coating and
        # deposition skip that column instead of inventing a surface at the
        # bottom of the simulation domain and filling the full wafer depth.
        tops = np.full(w, np.nan, dtype=float)
        for x in range(w):
            non_void = np.where(self.grid[:, x] != self.void_index)[0]
            if non_void.size > 0:
                tops[x] = float(edges[int(non_void.min())])
        return tops

    def _contiguous_exposed_material_depth_um(self, x: int, material_index: int):
        edges = self._row_edges_um()
        col = self.grid[:, int(x)]
        non_void = np.where(col != self.void_index)[0]
        if non_void.size == 0:
            return 0.0
        row = int(non_void.min())
        if col[row] != material_index:
            return 0.0
        start = row
        while row < self.grid.shape[0] and col[row] == material_index:
            row += 1
        return float(edges[row] - edges[start])

    def _pad_top(self, rows: int):
        rows = int(rows)
        if rows <= 0:
            return
        self._ensure_row_heights()
        self._ensure_structural_roles()
        self._ensure_planarizer_roles()
        pad = np.full((rows, self.width), self.void_index, dtype=self.grid.dtype)
        self.grid = np.vstack((pad, self.grid))
        self.structural_role_grid = np.vstack((
            np.zeros((rows, self.width), dtype=np.int16),
            self.structural_role_grid,
        ))
        self.planarizer_role_grid = np.vstack((
            np.zeros((rows, self.width), dtype=np.int32),
            self.planarizer_role_grid,
        ))
        self.row_heights_um = np.concatenate((np.ones(rows, dtype=float), self.row_heights_um))
        self.height, self.width = self.grid.shape

    def _ensure_global_headroom(self, required_px: int):
        required_px = max(0, int(required_px))
        if required_px <= 0:
            return
        top = self._top_index_global()
        if top < required_px:
            self._pad_top(required_px - top)

    def _top_index_global(self) -> int:
        """Return the first row containing material anywhere across the width.

        An empty cross-section returns ``height``. Flat global operations such
        as bonding use this index as their reference surface.
        """
        non_void = np.where(self.grid != self.void_index)[0]
        if non_void.size == 0:
            return self.height
        return int(non_void.min())

    # ---------- process operations ----------

    def deposit_physical(self, material_index: int, thickness_um: float):
        """
        Deposit an exact physical thickness on each local exposed surface.
        Row heights carry the physical scale, so nm films do not require a
        dense nm-resolution substrate grid.
        """
        thickness_um = float(thickness_um)
        if thickness_um <= 0.0:
            return

        tops = self._surface_tops_um()
        intervals = [
            (x, float(top - thickness_um), float(top))
            for x, top in enumerate(tops)
            if np.isfinite(top)
        ]
        if not intervals:
            return
        self._discard_visual_geometry_type("isotropic_etch_front")
        shift = self._insert_boundaries_um([y for _, y0, y1 in intervals for y in (y0, y1)])
        self._ensure_structural_roles()
        self._ensure_planarizer_roles()
        for x, y0, y1 in intervals:
            rows = self._row_indices_inside_um(y0 + shift, y1 + shift)
            if rows.size:
                self.grid[rows, x] = material_index
                self.structural_role_grid[rows, x] = 0
                self.planarizer_role_grid[rows, x] = 0
        self._merge_adjacent_equal_rows()

    def spincoat_physical(self, material_index: int, thickness_um: float):
        """
        Add a planarizing spin-coated layer.

        The requested thickness is measured from the highest exposed surface.
        Lower regions are filled up to the same final top plane, which can make
        the local resist thickness larger inside etched gaps.
        """
        thickness_um = float(thickness_um)
        if thickness_um <= 0.0:
            return

        tops = self._surface_tops_um()
        finite_tops = tops[np.isfinite(tops)]
        if finite_tops.size == 0:
            return

        global_top = float(np.min(finite_tops))
        new_top = global_top - thickness_um
        intervals = [
            (x, new_top, float(top))
            for x, top in enumerate(tops)
            if np.isfinite(top) and float(top) > new_top + _EPS_UM
        ]
        if not intervals:
            return

        self._discard_visual_geometry_type("isotropic_etch_front")
        shift = self._insert_boundaries_um([y for _, y0, y1 in intervals for y in (y0, y1)])
        self._ensure_structural_roles()
        self._ensure_planarizer_roles()
        role_id = self._new_planarizer_role(
            material_index,
            thickness_um,
            fill_thickness_um_by_column=[
                max(0.0, float(top) - float(global_top)) if np.isfinite(top) else 0.0
                for top in tops
            ],
        )
        for x, y0, y1 in intervals:
            rows = self._row_indices_inside_um(y0 + shift, y1 + shift)
            if rows.size:
                self.grid[rows, x] = material_index
                self.structural_role_grid[rows, x] = 0
                row_edges = self._row_edges_um()
                overburden = rows[
                    row_edges[rows + 1] <= global_top + shift + _EPS_UM
                ]
                fill = rows[
                    row_edges[rows] >= global_top + shift - _EPS_UM
                ]
                self.planarizer_role_grid[overburden, x] = int(role_id)
                self.planarizer_role_grid[fill, x] = -int(role_id)
        self._merge_adjacent_equal_rows()

    def bond_physical(self, material_index: int, thickness_um: float):
        """
        Add a flat rigid structural body with exact physical thickness.

        The new body bridges lower topography.  It never fills the cavity or
        moves the existing films beneath it.
        """
        thickness_um = float(thickness_um)
        if thickness_um <= 0.0:
            return

        self._discard_visual_geometry_type("isotropic_etch_front")
        top = float(self._row_edges_um()[self._top_index_global()])
        interval = (top - thickness_um, top)
        shift = self._insert_boundaries_um(interval)
        rows = self._row_indices_inside_um(interval[0] + shift, interval[1] + shift)
        if rows.size:
            body_id = self._new_structural_body(
                material_index,
                thickness_um,
                kind="bonded_substrate",
            )
            self.grid[rows, :] = material_index
            self.structural_role_grid[rows, :] = int(body_id)
            self._ensure_planarizer_roles()
            self.planarizer_role_grid[rows, :] = 0
        self._merge_adjacent_equal_rows()

    def etch_physical(
        self,
        material_index: int,
        depth_um: float,
        mask=None,
        mode: str = "anisotropic",
        lateral_factor: float = 1.0,
    ):
        """
        Etch exact physical depth from exposed selected material.
        """
        depth_um = float(depth_um)
        if depth_um <= 0.0:
            return

        if mask is None:
            mask = np.ones(self.width, dtype=bool)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != self.width:
            raise ValueError("Mask width does not match cross-section width")

        mode_norm = (mode or "anisotropic").lower()
        if mode_norm == "isotropic" and lateral_factor > 0.0:
            self._etch_physical_isotropic(material_index, depth_um, mask, lateral_factor)
            return

        edges = self._row_edges_um()
        intervals = []
        for x in range(self.width):
            if not mask[x]:
                continue
            col = self.grid[:, x]
            non_void = np.where(col != self.void_index)[0]
            if non_void.size == 0:
                continue
            row = int(non_void.min())
            if col[row] != material_index:
                continue

            remaining = depth_um
            while row < self.grid.shape[0] and remaining > _EPS_UM and col[row] == material_index:
                row_start = float(edges[row])
                row_height = float(self.row_heights_um[row])
                remove_height = min(row_height, remaining)
                intervals.append((x, row_start, row_start + remove_height))
                remaining -= remove_height
                if remove_height < row_height - _EPS_UM:
                    break
                row += 1

        if not intervals:
            return
        self._discard_visual_geometry_type("isotropic_etch_front")

        shift = self._insert_boundaries_um([y for _, y0, y1 in intervals for y in (y0, y1)])
        self._ensure_structural_roles()
        self._ensure_planarizer_roles()
        for x, y0, y1 in intervals:
            rows = self._row_indices_inside_um(y0 + shift, y1 + shift)
            if rows.size:
                selected = self.grid[rows, x] == material_index
                self.grid[rows[selected], x] = self.void_index
                self.structural_role_grid[rows[selected], x] = 0
                self.planarizer_role_grid[rows[selected], x] = 0
        if shift:
            intervals = [(x, y0 + shift, y1 + shift) for x, y0, y1 in intervals]
        self._append_etch_depth_feature(
            material_index,
            mode_norm,
            intervals,
            depth_um,
        )
        self._merge_adjacent_equal_rows()

    def _etch_physical_isotropic(self, material_index: int, depth_um: float, mask, lateral_factor: float):
        edges = self._row_edges_um()
        source_points = []
        available_depth_by_x = {}
        boundaries = []
        depth_um = float(depth_um)
        lateral_factor = float(lateral_factor)
        if depth_um <= 0.0 or lateral_factor <= 0.0:
            return
        vertical_slices = max(1, min(96, int(np.ceil(depth_um / max(depth_um / 64.0, 0.001)))))

        for x in range(self.width):
            if not mask[x]:
                continue
            col = self.grid[:, x]
            non_void = np.where(col != self.void_index)[0]
            if non_void.size == 0:
                continue
            row = int(non_void.min())
            if col[row] != material_index:
                continue
            y0 = float(edges[row])
            source_points.append((int(x), y0))
            available_depth_by_x[int(x)] = self._contiguous_exposed_material_depth_um(x, material_index)
            for i in range(vertical_slices + 1):
                boundaries.append(y0 + depth_um * i / vertical_slices)

        if not source_points:
            return

        self._insert_boundaries_um(boundaries)
        edges = self._row_edges_um()
        removal = np.zeros_like(self.grid, dtype=bool)
        radius_y = depth_um
        radius_x = max(0.0, lateral_factor * depth_um)
        if radius_x <= 0.0:
            return

        segments = []
        for x, y0 in source_points:
            effective_depth = min(
                depth_um,
                float(available_depth_by_x.get(int(x), 0.0)),
            )
            if not segments:
                segments.append([x, x, y0, effective_depth])
                continue
            prev = segments[-1]
            if (
                x == prev[1] + 1
                and abs(y0 - prev[2]) <= _EPS_UM
                and abs(effective_depth - prev[3]) <= _EPS_UM
            ):
                prev[1] = x
            else:
                segments.append([x, x, y0, effective_depth])

        visual_entry = {
            "type": "isotropic_etch_front",
            "material_index": int(material_index),
            "depth_um": float(depth_um),
            "lateral_um": float(radius_x),
            "polylines": [],
        }

        for x_start, x_end, y0, effective_depth in segments:
            x_min = max(0, int(np.ceil(float(x_start) - radius_x - _EPS_UM)))
            x_max = min(self.width - 1, int(np.floor(float(x_end) + radius_x + _EPS_UM)))
            if x_max < x_min:
                continue
            if effective_depth > _EPS_UM:
                visual_entry["polylines"].append(
                    self._isotropic_front_polyline(
                        float(x_start),
                        float(x_end),
                        float(y0),
                        float(effective_depth),
                        float(lateral_factor),
                    )
                )
            for row in range(self.grid.shape[0]):
                mid_y = 0.5 * (float(edges[row]) + float(edges[row + 1]))
                dy = mid_y - y0
                if dy < -_EPS_UM or dy > radius_y + _EPS_UM:
                    continue
                normalized_y = dy / radius_y if radius_y > 0.0 else 0.0
                inside = 1.0 - normalized_y ** 2
                if inside < -_EPS_UM:
                    continue
                lateral_reach = radius_x * np.sqrt(max(0.0, inside))
                for x in range(x_min, x_max + 1):
                    if x < x_start:
                        dx = float(x_start - x)
                    elif x > x_end:
                        dx = float(x - x_end)
                    else:
                        dx = 0.0
                    if dx > lateral_reach + _EPS_UM:
                        continue
                    if self.grid[row, x] == material_index:
                        removal[row, x] = True

        removed_intervals = []
        for x in range(self.width):
            rows = np.where(removal[:, x] & (self.grid[:, x] == material_index))[0]
            if rows.size == 0:
                continue
            start = prev = int(rows[0])
            for row in rows[1:]:
                row = int(row)
                if row == prev + 1:
                    prev = row
                    continue
                removed_intervals.append((x, float(edges[start]), float(edges[prev + 1])))
                start = prev = row
            removed_intervals.append((x, float(edges[start]), float(edges[prev + 1])))

        removed_mask = removal & (self.grid == material_index)
        self.grid[removed_mask] = self.void_index
        self._ensure_structural_roles()
        self._ensure_planarizer_roles()
        self.structural_role_grid[removed_mask] = 0
        self.planarizer_role_grid[removed_mask] = 0
        self._discard_visual_geometry_type("isotropic_etch_front")
        if visual_entry["polylines"]:
            self.visual_geometry.append(visual_entry)
        self._append_etch_depth_feature(
            material_index,
            "isotropic",
            removed_intervals,
            depth_um,
        )
        self._merge_adjacent_equal_rows()

    def _isotropic_front_polyline(
        self,
        x_start: float,
        x_end: float,
        y_surface_um: float,
        depth_um: float,
        lateral_factor: float,
    ):
        depth_um = max(0.0, float(depth_um))
        lateral_um = max(0.0, float(lateral_factor)) * depth_um
        if depth_um <= _EPS_UM or lateral_um <= _EPS_UM:
            return [(x_start, y_surface_um), (x_end, y_surface_um + depth_um)]

        samples = max(8, min(64, int(np.ceil(depth_um * 4.0))))
        points = []
        for i in range(samples + 1):
            dy = depth_um * float(i) / float(samples)
            reach = lateral_um * np.sqrt(max(0.0, 1.0 - (dy / depth_um) ** 2))
            points.append((float(x_start) - float(reach), float(y_surface_um) + float(dy)))
        points.append((float(x_end), float(y_surface_um) + depth_um))
        for i in range(samples, -1, -1):
            dy = depth_um * float(i) / float(samples)
            reach = lateral_um * np.sqrt(max(0.0, 1.0 - (dy / depth_um) ** 2))
            points.append((float(x_end) + float(reach), float(y_surface_um) + float(dy)))
        return points

    def deposit(self, material_index: int, thickness_px: int):
        """
        Deposita uno strato di 'material_index' di spessore 'thickness_px'
        seguendo il profilo locale di ogni colonna.
        """
        self.deposit_physical(material_index, float(thickness_px))

    def spincoat(self, material_index: int, thickness_px: int):
        """
        Planarizing spin-coating fallback for pixel-mode callers.
        """
        self.spincoat_physical(material_index, float(thickness_px))

    def strip_top_only(self, material_index: int):
        """
        Remove only the top‑exposed portion of the specified material.

        This will scan each column and delete contiguous pixels of
        `material_index` starting at the surface (the first non‑void
        pixel) until a different material or void is encountered. Deeper
        occurrences of the same material remain untouched. If the
        material is not the first exposed in a column, nothing is
        removed in that column.
        """
        if material_index is None or material_index == self.void_index:
            return
        self._ensure_structural_roles()
        self._ensure_planarizer_roles()
        h, w = self.grid.shape
        changed = False
        for x in range(w):
            col = self.grid[:, x]
            # find topmost non‑void pixel
            non_void = np.where(col != self.void_index)[0]
            if non_void.size == 0:
                continue
            top_y = int(non_void.min())
            # only remove if the exposed material matches
            if col[top_y] != material_index:
                continue
            # remove contiguous run of this material starting at top_y
            y = top_y
            while y < h and col[y] == material_index:
                changed = True
                col[y] = self.void_index
                self.structural_role_grid[y, x] = 0
                self.planarizer_role_grid[y, x] = 0
                y += 1
            self.grid[:, x] = col
        if changed:
            self._discard_visual_geometry_type("isotropic_etch_front")

    def flip_active_stack(self) -> bool:
        """
        Flip only the non‑void portion of the stack vertically.

        Empty rows above and below the stack are left untouched so the
        "air" region does not move while processing focus changes to the
        opposite face.
        """
        non_void_rows = np.where(np.any(self.grid != self.void_index, axis=1))[0]
        if non_void_rows.size == 0:
            return False
        top = int(non_void_rows.min())
        bottom = int(non_void_rows.max())
        self.grid[top:bottom + 1, :] = np.flipud(self.grid[top:bottom + 1, :])
        self._ensure_row_heights()
        self._ensure_structural_roles()
        self._ensure_planarizer_roles()
        self.row_heights_um[top:bottom + 1] = self.row_heights_um[top:bottom + 1][::-1]
        self.structural_role_grid[top:bottom + 1, :] = np.flipud(
            self.structural_role_grid[top:bottom + 1, :]
        )
        self.planarizer_role_grid[top:bottom + 1, :] = np.flipud(
            self.planarizer_role_grid[top:bottom + 1, :]
        )
        for metadata in self.planarizer_roles.values():
            if not isinstance(metadata, dict):
                continue
            metadata["orientation"] = -int(metadata.get("orientation", 1) or 1)
        self._clear_visual_geometry()
        return True

    def bond(self, material_index: int, thickness_px: int):
        """
        'Bonding' ideale: deposita un layer PIATTO globale
        a partire dal top GLOBALE, ignorando i canali.
        """
        self.bond_physical(material_index, float(thickness_px))

    def etch(
        self,
        material_index: int,
        depth_px: int,
        mask=None,
        mode: str = "anisotropic",
        lateral_factor: float = 1.0,
    ):
        """Compatibility wrapper for an etch depth expressed in process rows.

        ``mask`` selects open horizontal columns. Anisotropic mode removes
        vertically; isotropic mode also applies lateral undercut scaled by
        ``lateral_factor``.
        """
        self.etch_physical(
            material_index,
            depth_um=float(depth_px),
            mask=mask,
            mode=mode,
            lateral_factor=lateral_factor,
        )

    def strip(self, material_index: int):
        """Remove every occurrence of ``material_index`` from the cross-section."""
        selected = self.grid == material_index
        if np.any(selected):
            self._discard_visual_geometry_type("isotropic_etch_front")
        self.grid[selected] = self.void_index
        self._ensure_structural_roles()
        self._ensure_planarizer_roles()
        self.structural_role_grid[selected] = 0
        self.planarizer_role_grid[selected] = 0

    def liftoff(self, resist_index: int):
        """
        Perform a lift‑off operation on a patterned photoresist.

        This removes the specified photoresist material and any film that was
        deposited on top of it. Only the topmost resist run in each column is
        removed (plus any material above it), so deeper/bottom occurrences of
        the same resist index are preserved.

        Parameters
        ----------
        resist_index : int
            Index of the photoresist material to lift off. If this is the
            void index or no resist of this index is found, nothing happens.
        """
        if resist_index is None or resist_index == self.void_index:
            # Invalid or void resist index: nothing to do
            return

        self._ensure_structural_roles()
        self._ensure_planarizer_roles()
        h, w = self.grid.shape
        changed = False
        # Iterate over each column independently
        for x in range(w):
            col = self.grid[:, x]
            # Find all positions of the resist in this column
            resist_positions = np.where(col == resist_index)[0]
            if resist_positions.size == 0:
                # No resist in this column – skip
                continue
            # Topmost (smallest y) position of the resist
            top_resist_y = int(resist_positions.min())
            # Remove any material (deposit) above the resist
            if top_resist_y > 0:
                changed = changed or bool(np.any(col[:top_resist_y] != self.void_index))
                col[:top_resist_y] = self.void_index
                self.structural_role_grid[:top_resist_y, x] = 0
                self.planarizer_role_grid[:top_resist_y, x] = 0
            # Remove only the contiguous top run of the selected resist
            y = top_resist_y
            while y < h and col[y] == resist_index:
                changed = True
                col[y] = self.void_index
                self.structural_role_grid[y, x] = 0
                self.planarizer_role_grid[y, x] = 0
                y += 1
            # Write back updated column
            self.grid[:, x] = col
        if changed:
            self._discard_visual_geometry_type("isotropic_etch_front")

    def _remove_top_exposed_pr_columns(self, pr_index: int, column_mask):
        mask = np.asarray(column_mask, dtype=bool)
        if mask.shape[0] != self.width:
            raise ValueError("column_mask length must match cross-section width")

        self._ensure_structural_roles()
        self._ensure_planarizer_roles()
        h, _ = self.grid.shape
        for x in np.where(mask)[0]:
            col = self.grid[:, x]
            non_void = np.where(col != self.void_index)[0]
            if non_void.size == 0:
                continue
            top_y = int(non_void.min())
            if col[top_y] != pr_index:
                continue
            y = top_y
            while y < h and col[y] == pr_index:
                col[y] = self.void_index
                self.structural_role_grid[y, x] = 0
                self.planarizer_role_grid[y, x] = 0
                y += 1
            self.grid[:, x] = col

    def pattern_pr_lithography(self, pr_index: int, windows, tone: str = "positive"):
        """
        Develop a patterned photoresist on the active top surface.

        Positive tone removes photoresist inside the window union.
        Negative tone keeps photoresist inside the window union and removes
        top-exposed photoresist outside it.
        """
        window_mask = np.zeros(self.width, dtype=bool)
        for entry in windows or []:
            try:
                start_frac, end_frac = entry
            except (TypeError, ValueError):
                continue
            window_mask |= self.window_mask(float(start_frac), float(end_frac))

        tone_text = str(tone or "positive").strip().lower()
        if tone_text.startswith("neg"):
            removal_mask = ~window_mask
        else:
            if not np.any(window_mask):
                return
            removal_mask = window_mask
        if np.any(removal_mask):
            self._discard_visual_geometry_type("isotropic_etch_front")
        self._remove_top_exposed_pr_columns(pr_index, removal_mask)

    def pattern_pr_open_window(self, pr_index: int, start_frac: float, end_frac: float):
        """
        Positive lithography development on the active top surface.

        Removes only the top-exposed contiguous run of `pr_index` in each
        column inside the [start_frac, end_frac] window.
        """
        self.pattern_pr_lithography(pr_index, [(start_frac, end_frac)], tone="positive")

    def window_mask(self, start_frac: float, end_frac: float):
        """Return the columns covered by normalized interval ``[start, end)``."""
        mask = np.zeros(self.width, dtype=bool)
        start = int(self.width * start_frac)
        end = int(self.width * end_frac)
        if start < 0:
            start = 0
        if end > self.width:
            end = self.width
        if end <= start:
            return mask
        mask[start:end] = True
        return mask

    # ---------- state snapshots (undo/history) ----------

    def copy_state(self):
        """Return an independent copy of the material grid."""
        return self.grid.copy()

    def copy_row_heights(self):
        self._ensure_row_heights()
        return self.row_heights_um.copy()

    def copy_visual_geometry(self):
        return copy.deepcopy(getattr(self, "visual_geometry", []) or [])

    def copy_structural_roles(self):
        self._ensure_structural_roles()
        return self.structural_role_grid.copy()

    def copy_structural_bodies(self):
        self._ensure_structural_roles()
        return copy.deepcopy(self.structural_bodies)

    def copy_planarizer_roles(self):
        self._ensure_planarizer_roles()
        return self.planarizer_role_grid.copy()

    def copy_planarizer_metadata(self):
        self._ensure_planarizer_roles()
        return copy.deepcopy(self.planarizer_roles)

    def restore_state(
        self,
        grid,
        row_heights_um=None,
        visual_geometry=None,
        structural_role_grid=None,
        structural_bodies=None,
        planarizer_role_grid=None,
        planarizer_roles=None,
    ):
        """Restore a previously captured cross-section state."""
        self.grid = grid.copy()
        if row_heights_um is not None and len(row_heights_um) == self.grid.shape[0]:
            self.row_heights_um = np.asarray(row_heights_um, dtype=float).copy()
        else:
            self.row_heights_um = np.ones(self.grid.shape[0], dtype=float)
        self.visual_geometry = copy.deepcopy(visual_geometry or [])
        if structural_role_grid is not None and np.asarray(structural_role_grid).shape == self.grid.shape:
            self.structural_role_grid = np.asarray(structural_role_grid, dtype=np.int16).copy()
        else:
            self.structural_role_grid = None
        self.structural_bodies = copy.deepcopy(structural_bodies) if isinstance(structural_bodies, dict) else None
        self._ensure_structural_roles()
        if (
            planarizer_role_grid is not None
            and np.asarray(planarizer_role_grid).shape == self.grid.shape
        ):
            self.planarizer_role_grid = np.asarray(
                planarizer_role_grid,
                dtype=np.int32,
            ).copy()
        else:
            self.planarizer_role_grid = None
        self.planarizer_roles = (
            copy.deepcopy(planarizer_roles)
            if isinstance(planarizer_roles, dict)
            else None
        )
        self._ensure_planarizer_roles()
        # Keep dimensions consistent with restored history state.
        self.height, self.width = self.grid.shape
