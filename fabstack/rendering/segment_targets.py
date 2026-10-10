"""Per-column display target construction without presentation imports."""

import numpy as np

from fabstack.rendering.constants import MIN_VOID_PX
from fabstack.rendering.contracts import SegmentTargetHost


def build_material_column_segments_with_targets(
    host: SegmentTargetHost,
    active,
    active_heights,
    *,
    history_index=None,
    process_profile=None,
    active_structural_roles=None,
    active_planarizer_roles=None,
):
    active = np.asarray(active)
    active_heights = np.asarray(active_heights, dtype=float)
    if active.ndim != 2 or active.size == 0:
        return [], 0
    if len(active_heights) != active.shape[0]:
        active_heights = np.ones(active.shape[0], dtype=float)

    columns = host._display_raw_material_columns(
        active,
        active_heights,
        process_profile=process_profile,
        active_structural_roles=active_structural_roles,
        active_planarizer_roles=active_planarizer_roles,
    )
    operation = ""
    operation_material_idx = None
    planarizer_metadata = {}
    try:
        state = host.history[int(history_index)]
    except (TypeError, ValueError, IndexError, AttributeError):
        state = None
    if isinstance(state, dict):
        operation = str(state.get("operation", "") or "").strip()
        material_name = str(state.get("material_name", "") or "").strip()
        if material_name:
            try:
                operation_material_idx = int(host.material_names.index(material_name))
            except (ValueError, AttributeError):
                operation_material_idx = None
        raw_planarizer_metadata = state.get("planarizer_roles", {})
        if isinstance(raw_planarizer_metadata, dict):
            planarizer_metadata = raw_planarizer_metadata
    elif getattr(host, "xsec", None) is not None:
        raw_planarizer_metadata = getattr(host.xsec, "planarizer_roles", {})
        if isinstance(raw_planarizer_metadata, dict):
            planarizer_metadata = raw_planarizer_metadata

    # A spin coat is two different physical features even when both are
    # one contiguous run of the same material in the compact grid:
    #
    #   * the requested overburden above the highest previous surface;
    #   * the extra local fill required by lower topography.
    #
    # Keep those portions separate for display constraints.  Otherwise a
    # planar-alignment correction can accidentally inflate an older film
    # of the same material (or the constant overburden) after grid rows
    # merge.  The previous state's exposed-surface offsets recover the
    # exact local fill height without changing the physical model.
    spin_requested_um = 0.0
    spin_surface_offsets = None
    has_persistent_planarizer = bool(
        active_planarizer_roles is not None
        and np.any(np.asarray(active_planarizer_roles) != 0)
    )
    if (
        not has_persistent_planarizer
        and operation == "Spincoating"
        and operation_material_idx is not None
    ):
        try:
            spin_requested_um = max(0.0, float(state.get("thickness_um", 0.0) or 0.0))
            state_index = int(history_index)
            previous = host.history[state_index - 1] if state_index > 0 else None
        except (TypeError, ValueError, IndexError, AttributeError):
            previous = None
        if spin_requested_um > 1e-12 and isinstance(previous, dict):
            previous_grid = np.asarray(previous.get("grid"))
            previous_heights = np.asarray(
                previous.get("row_heights_um", []),
                dtype=float,
            )
            if (
                previous_grid.ndim == 2
                and previous_grid.shape[1] == active.shape[1]
                and previous_heights.shape[0] == previous_grid.shape[0]
            ):
                previous_edges = np.concatenate(([0.0], np.cumsum(previous_heights)))
                previous_tops = np.full(
                    previous_grid.shape[1],
                    float(previous_edges[-1]),
                    dtype=float,
                )
                for col in range(previous_grid.shape[1]):
                    non_void = np.where(previous_grid[:, col] != host.VOID_INDEX)[0]
                    if non_void.size:
                        previous_tops[col] = float(previous_edges[int(non_void.min())])
                spin_surface_offsets = previous_tops - float(np.min(previous_tops))

    if spin_surface_offsets is not None:
        for col, segments in enumerate(columns):
            fill_end_um = spin_requested_um + max(0.0, float(spin_surface_offsets[col]))
            rewritten = []
            for segment in segments:
                y0 = float(segment.get("y0_local_um", 0.0))
                y1 = float(segment.get("y1_local_um", 0.0))
                material_idx = int(segment.get("material", host.VOID_INDEX))
                cuts = [y0]
                if material_idx == int(operation_material_idx):
                    for cut in (spin_requested_um, fill_end_um):
                        if y0 < cut < y1:
                            cuts.append(float(cut))
                cuts.append(y1)
                cuts = sorted(set(cuts))
                for part_y0, part_y1 in zip(cuts, cuts[1:]):
                    if part_y1 <= part_y0 + 1e-12:
                        continue
                    part = dict(segment)
                    part["y0_local_um"] = float(part_y0)
                    part["y1_local_um"] = float(part_y1)
                    part["height_um"] = float(part_y1 - part_y0)
                    is_current_material = material_idx == int(operation_material_idx)
                    is_overburden = bool(
                        is_current_material
                        and part_y0 >= -1e-9
                        and part_y1 <= spin_requested_um + 1e-9
                    )
                    is_fill = bool(
                        is_current_material
                        and fill_end_um > spin_requested_um + 1e-12
                        and part_y0 >= spin_requested_um - 1e-9
                        and part_y1 <= fill_end_um + 1e-9
                    )
                    part["is_planarizer_overburden"] = is_overburden
                    part["is_planarizer_fill"] = is_fill
                    if is_overburden:
                        semantic_suffix = "spin-overburden"
                    elif is_fill:
                        semantic_suffix = "spin-fill"
                    else:
                        semantic_suffix = "pre-spin"
                    part["source_segment_id"] = (
                        f"{segment.get('source_segment_id', col)}:{semantic_suffix}"
                    )
                    part["source_height_um"] = float(part["height_um"])
                    rewritten.append(part)
            segments[:] = rewritten

    active_planarizer_ids = {
        abs(int(value))
        for value in np.unique(np.asarray(active_planarizer_roles))
        if int(value) != 0
    } if active_planarizer_roles is not None else set()

    for column_index, segments in enumerate(columns):
        core_segment_idx = next(
            (idx for idx, segment in enumerate(segments)
             if bool(segment.get("is_substrate_core", False))),
            None,
        )

        source_groups = {}
        for segment_idx, segment in enumerate(segments):
            if bool(segment.get("is_substrate_core", False)):
                continue
            source_groups.setdefault(
                str(segment.get("source_segment_id", segment_idx)),
                [],
            ).append(segment)

        for group in source_groups.values():
            material_idx = int(group[0]["material"])
            source_height_um = float(group[0].get(
                "source_height_um",
                sum(float(segment.get("height_um", 0.0)) for segment in group),
            ))
            override_key = host._display_material_count_profile_key(
                material_idx,
                source_height_um,
            )
            if bool(group[0].get("is_structural_body", False)):
                body_id = int(group[0].get("structural_body_id", 0) or 0)
                body_nominal_um = float(
                    group[0].get("structural_nominal_um", 0.0) or 0.0
                )
                if body_id > 0 and body_nominal_um > 1e-12:
                    total_target = host._display_substrate_core_target_height(
                        source_height_um,
                        process_profile,
                        structural_body_id=body_id,
                        structural_nominal_um=body_nominal_um,
                    )
                else:
                    total_target = host._display_structural_body_rows(source_height_um)
            else:
                total_target = host._display_segment_target_height(
                    source_height_um,
                    material_idx=material_idx,
                    override_key=override_key,
                    history_index=history_index,
                    process_profile=process_profile,
                    is_substrate_core=False,
                )
            if len(group) == 1:
                group[0]["target_rows"] = int(total_target)
                continue
            apportioned = host._apportion_display_rows(
                int(total_target),
                [float(segment.get("height_um", 0.0)) for segment in group],
            )
            for segment, target in zip(group, apportioned):
                # A merged physical run can be split at the substrate
                # reference plane (for example one void run becomes
                # above-surface headroom plus the actual etched recess).
                # Finalised/manual controls address those visible pieces,
                # so a piece-specific override must take precedence over
                # the normal apportionment of the complete source run.
                piece_key = host._display_material_count_profile_key(
                    material_idx,
                    float(segment.get("height_um", 0.0)),
                )
                piece_override = host._visual_override_value_for_key(
                    piece_key,
                    history_index=history_index,
                )
                segment["target_rows"] = (
                    max(1, int(piece_override))
                    if piece_override is not None
                    else int(target)
                )

        if core_segment_idx is not None:
            core_segment = segments[core_segment_idx]
            try:
                nominal_um = float((process_profile or {}).get("nominal_substrate_um", 0.0) or 0.0)
            except (TypeError, ValueError):
                nominal_um = 0.0

            core_material_idx = int(core_segment["material"])
            core_height_um = float(core_segment["height_um"])
            core_override_key = host._display_material_count_profile_key(
                core_material_idx,
                core_height_um,
            )
            core_segment["target_rows"] = host._display_segment_target_height(
                core_height_um,
                material_idx=core_material_idx,
                override_key=core_override_key,
                history_index=history_index,
                process_profile=process_profile,
                is_substrate_core=True,
                structural_body_id=int(
                    core_segment.get("structural_body_id", 1) or 1
                ),
                structural_nominal_um=float(
                    core_segment.get("structural_nominal_um", 0.0) or 0.0
                ),
            )

            core_nominal_um = float(
                core_segment.get("structural_nominal_um", 0.0) or 0.0
            )
            if core_nominal_um <= 1e-12:
                core_nominal_um = nominal_um
            removed_um = max(0.0, core_nominal_um - core_height_um)
            # The etched depth is global physical geometry, not the
            # difference between two user-editable display overrides.
            # This keeps the open trench stable when a finalised substrate
            # band is adjusted manually.
            actual_removed_rows = (
                host._display_profile_feature_rows(removed_um, process_profile)
                if removed_um > 1e-12 else 0
            )
            if core_segment_idx >= len(segments) / 2.0:
                process_side = segments[:core_segment_idx]
            else:
                process_side = segments[core_segment_idx + 1:]
            process_voids = [
                segment for segment in process_side
                if int(segment.get("material", host.VOID_INDEX)) == int(host.VOID_INDEX)
            ]
            recess_segments = [
                segment for segment in process_side
                if bool(segment.get("in_substrate_recess", False))
            ]
            recess_voids = [
                segment for segment in recess_segments
                if int(segment.get("material", host.VOID_INDEX)) == int(host.VOID_INDEX)
            ]
            recess_solids = [
                segment for segment in recess_segments
                if int(segment.get("material", host.VOID_INDEX)) != int(host.VOID_INDEX)
                and not bool(segment.get("is_structural_body", False))
            ]
            has_recess_override = any(
                host._visual_override_value_for_key(
                    host._display_material_count_profile_key(
                        int(segment.get("material", host.VOID_INDEX)),
                        float(segment.get("height_um", 0.0)),
                    ),
                    history_index=history_index,
                ) is not None
                for segment in recess_segments
            )

            # Conserve one exact recess budget. Solids deposited or spun
            # into the trench replace void rows; they are never stacked on
            # top of a second full-depth void allowance.
            if (
                removed_um > 1e-12
                and recess_segments
                and not has_recess_override
                and actual_removed_rows > 0
            ):
                flexible_spin = [
                    segment for segment in recess_solids
                    if bool(segment.get("is_planarizer_fill", False))
                    and (
                        int(segment.get("planarizer_role", 0)) != 0
                        or (
                            operation == "Spincoating"
                            and operation_material_idx is not None
                            and int(segment.get("material", -1))
                            == int(operation_material_idx)
                        )
                    )
                ]
                fixed_solids = [
                    segment for segment in recess_solids
                    if segment not in flexible_spin
                ]
                fixed_rows = int(sum(
                    max(0, int(segment.get("target_rows", 0)))
                    for segment in fixed_solids

                ))
                removed_planarizer_fill_rows = 0
                for role_id in active_planarizer_ids:
                    metadata = planarizer_metadata.get(
                        int(role_id),
                        planarizer_metadata.get(str(int(role_id)), {}),
                    )
                    if not isinstance(metadata, dict):
                        continue
                    raw_fill = metadata.get("fill_thickness_um_by_column", [])
                    fill_by_column = (
                        list(raw_fill)
                        if isinstance(raw_fill, (list, tuple, np.ndarray))
                        else []
                    )
                    try:
                        original_fill_um = max(
                            0.0,
                            float(fill_by_column[column_index])
                            if column_index < len(fill_by_column)
                            else 0.0,
                        )
                    except (TypeError, ValueError):
                        original_fill_um = 0.0
                    remaining_fill_um = float(sum(
                        max(0.0, float(segment.get("height_um", 0.0)))
                        for segment in segments
                        if int(segment.get("planarizer_role", 0)) == -int(role_id)
                    ))
                    removed_fill_um = max(
                        0.0,
                        original_fill_um - remaining_fill_um,
                    )
                    if removed_fill_um > max(1e-9, original_fill_um * 1e-8):
                        removed_planarizer_fill_rows += int(
                            host._display_profile_feature_rows(
                                removed_fill_um,
                                process_profile,
                            )
                        )

                minimum_void_rows = max(
                    len(recess_voids) * int(MIN_VOID_PX),
                    int(removed_planarizer_fill_rows),
                )
                remaining = max(0, int(actual_removed_rows) - fixed_rows)

                if flexible_spin:
                    spin_budget = max(0, remaining - minimum_void_rows)
                    desired_spin_targets = [
                        int(host._display_profile_feature_rows(
                            float(segment.get("height_um", 0.0)),
                            process_profile,
                        ))
                        for segment in flexible_spin
                    ]
                    desired_spin_total = int(sum(desired_spin_targets))
                    if desired_spin_total <= spin_budget:
                        spin_targets = desired_spin_targets
                    else:
                        spin_targets = host._apportion_display_rows(
                            spin_budget,
                            [float(segment.get("height_um", 0.0)) for segment in flexible_spin],
                        ) if spin_budget > 0 else [0] * len(flexible_spin)
                    for segment, target in zip(flexible_spin, spin_targets):
                        segment["target_rows"] = int(target)
                    remaining -= int(sum(spin_targets))

                if recess_voids:
                    void_total = max(0, remaining)
                    if void_total < minimum_void_rows:
                        void_total = minimum_void_rows
                    assigned = [int(MIN_VOID_PX)] * len(recess_voids)
                    remainder = void_total - sum(assigned)
                    if remainder > 0:
                        extra = host._apportion_display_rows(
                            remainder,
                            [float(segment.get("height_um", 0.0)) for segment in recess_voids],
                        )
                        assigned = [base + add for base, add in zip(assigned, extra)]
                    for segment, target in zip(recess_voids, assigned):
                        segment["target_rows"] = int(target)
                elif not flexible_spin and recess_solids:
                    # A completely solid-filled recess still occupies the
                    # resolved gap budget. Give any raster remainder to the
                    # thickest fill segment without moving its bottom.
                    solid_rows = int(sum(
                        max(0, int(segment.get("target_rows", 0)))
                        for segment in recess_solids
                    ))
                    if solid_rows < int(actual_removed_rows):
                        deficit = int(actual_removed_rows) - solid_rows
                        # If the last fill crosses the original substrate
                        # plane, move display rows from its above-plane
                        # portion into its in-recess portion. The complete
                        # conformal layer keeps the same total thickness as
                        # outside the trench.
                        crossing_groups = []
                        for source_id, group in source_groups.items():
                            below = [
                                segment for segment in group
                                if bool(segment.get("in_substrate_recess", False))
                                and int(segment.get("material", host.VOID_INDEX)) != int(host.VOID_INDEX)
                            ]
                            above = [
                                segment for segment in group
                                if not bool(segment.get("in_substrate_recess", False))
                                and int(segment.get("material", host.VOID_INDEX)) != int(host.VOID_INDEX)
                            ]
                            available = int(sum(
                                max(0, int(segment.get("target_rows", 0)) - 1)
                                for segment in above
                            ))
                            if below and above and available > 0:
                                crossing_groups.append((source_id, below, above, available))
                        for _source_id, below, above, available in crossing_groups:
                            if deficit <= 0:
                                break
                            transfer = min(deficit, available)
                            remaining_transfer = transfer
                            for segment in sorted(
                                above,
                                key=lambda item: int(item.get("target_rows", 0)),
                                reverse=True,
                            ):
                                take = min(
                                    remaining_transfer,
                                    max(0, int(segment.get("target_rows", 0)) - 1),
                                )
                                segment["target_rows"] -= take
                                remaining_transfer -= take
                                if remaining_transfer <= 0:
                                    break
                            anchor = max(
                                below,
                                key=lambda segment: float(segment.get("height_um", 0.0)),
                            )
                            anchor["target_rows"] += transfer
                            deficit -= transfer
                        if deficit > 0:
                            anchor = max(
                                recess_solids,
                                key=lambda segment: float(segment.get("height_um", 0.0)),
                            )
                            anchor["target_rows"] += deficit

    # Semantic interface constraints persist with the geometry, not with
    # the label of the latest operation. A developed spin coat is still a
    # planarizer in a later Bond/Lithography state, and a bonded wafer is
    # still rigid in every descendant state.
    persistent_planarizer_ids = sorted({
        abs(int(segment.get("planarizer_role", 0)))
        for segments in columns
        for segment in segments
        if int(segment.get("planarizer_role", 0)) != 0
    })
    for role_id in persistent_planarizer_ids:
        metadata = planarizer_metadata.get(
            int(role_id),
            planarizer_metadata.get(str(int(role_id)), {}),
        )
        if not isinstance(metadata, dict):
            metadata = {}
        try:
            requested_um = max(
                0.0,
                float(metadata.get("requested_thickness_um", 0.0) or 0.0),
            )
        except (TypeError, ValueError):
            requested_um = 0.0
        try:
            orientation = 1 if int(metadata.get("orientation", 1) or 1) >= 0 else -1
        except (TypeError, ValueError):
            orientation = 1

        # Only the surviving, unetched overburden still owns the original
        # planar free-surface constraint.  A partially etched column must
        # keep its real step instead of being silently re-planarized.
        surviving_overburdens = []
        for segments in columns:
            positive_indices = [
                idx for idx, segment in enumerate(segments)
                if int(segment.get("planarizer_role", 0)) == int(role_id)
            ]
            if not positive_indices:
                continue
            positive_um = float(sum(
                max(0.0, float(segments[idx].get("height_um", 0.0)))
                for idx in positive_indices
            ))
            surviving_overburdens.append((segments, positive_indices, positive_um))

        # Legacy snapshots do not carry requested thickness metadata.  In
        # that case the thickest surviving positive band is the safest
        # reconstruction of the original, intact overburden.
        intact_um = requested_um
        if intact_um <= 1e-12:
            intact_um = max(
                (entry[2] for entry in surviving_overburdens),
                default=0.0,
            )
        intact_tolerance_um = max(1e-9, intact_um * 1e-8)
        role_columns = []
        for segments, positive_indices, positive_um in surviving_overburdens:
            if (
                intact_um > 1e-12
                and abs(float(positive_um) - float(intact_um)) > intact_tolerance_um
            ):
                continue

            if orientation > 0:
                # Normal orientation: the outer free surface is the top
                # edge of the positive overburden.
                plane_index = min(positive_indices)
                plane_from_bottom = int(sum(
                    max(0, int(segment.get("target_rows", 0)))
                    for segment in segments[plane_index:]
                ))
                all_role_indices = [
                    idx for idx, segment in enumerate(segments)
                    if abs(int(segment.get("planarizer_role", 0))) == int(role_id)
                ]
                support_void = next(
                    (
                        segment for segment in segments[
                            max(all_role_indices) + 1:
                        ]
                        if int(segment.get("material", host.VOID_INDEX))
                        == int(host.VOID_INDEX)
                    ),
                    None,
                ) if all_role_indices else None
                fill = next(
                    (
                        segment for segment in segments
                        if int(segment.get("planarizer_role", 0)) == -int(role_id)
                        and not bool(segment.get("is_structural_body", False))
                    ),
                    None,
                )
                # A real support-side opening owns the reconciliation.
                # Growing the unchanged buried fill would invert its
                # scale against the much deeper support void.  Filled
                # recesses with no clearance retain the legacy fill
                # correction that keeps their planar top exact.
                correction = support_void if support_void is not None else fill
                correction_is_support_void = support_void is not None
            else:
                # After a flip, the same outer surface is the bottom edge
                # of the positive overburden.  Negative fill is now above
                # it and must retain its own physical depth (20 and 40 um
                # fills must never be equalised).  Any raster correction
                # belongs only to actual clearance below the outer plane.
                plane_index = max(positive_indices)
                plane_from_bottom = int(sum(
                    max(0, int(segment.get("target_rows", 0)))
                    for segment in segments[plane_index + 1:]
                ))
                correction = next(
                    (
                        segment for segment in segments[plane_index + 1:]
                        if int(segment.get("material", host.VOID_INDEX))
                        == int(host.VOID_INDEX)
                    ),
                    None,
                )
                correction_is_support_void = correction is not None
            role_columns.append((
                correction,
                plane_from_bottom,
                correction_is_support_void,
            ))

        supported_reference_planes = [
            entry[1] for entry in role_columns if not bool(entry[2])
        ]
        common_plane = max(
            supported_reference_planes
            or [entry[1] for entry in role_columns],
            default=0,
        )
        for correction, plane_from_bottom, correction_is_support_void in role_columns:
            deficit = int(common_plane) - int(plane_from_bottom)
            if deficit > 0 and correction is not None:
                correction["target_rows"] = (
                    int(correction.get("target_rows", 0)) + deficit
                )
            elif (
                deficit < 0
                and correction is not None
                and bool(correction_is_support_void)
            ):
                reducible = max(
                    0,
                    int(correction.get("target_rows", 0)) - int(MIN_VOID_PX),
                )
                correction["target_rows"] = int(
                    int(correction.get("target_rows", 0))
                    - min(-deficit, reducible)
                )

    # Backward-compatible reconstruction for legacy snapshots made before
    # planarizer provenance was stored explicitly.
    if (
        not persistent_planarizer_ids
        and operation == "Spincoating"
        and operation_material_idx is not None
    ):
        column_heights = [
            int(sum(max(0, int(segment.get("target_rows", 0))) for segment in segments))
            for segments in columns
        ]
        common_height = max(column_heights, default=0)
        for segments, height in zip(columns, column_heights):
            deficit = int(common_height) - int(height)
            if deficit <= 0:
                continue
            fill = next(
                (segment for segment in segments
                 if int(segment.get("material", -1)) == int(operation_material_idx)
                 and bool(segment.get("is_planarizer_fill", False))
                 and not bool(segment.get("is_structural_body", False))),
                None,
            )
            if fill is not None:
                fill["target_rows"] = int(fill.get("target_rows", 0)) + deficit

    max_height = max(
        (int(sum(max(0, int(segment.get("target_rows", 0))) for segment in segments))
         for segments in columns),
        default=0,
    )
    return columns, int(max_height)
# -- extracted service --


__all__ = ["build_material_column_segments_with_targets"]
