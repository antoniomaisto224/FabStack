"""Scene-wide display constraint construction without presentation imports."""

import math

import numpy as np

from fabstack.rendering.constants import (
    MAX_CONSTRAINT_PASSES,
    MIN_SUBSTRATE_CORE_PX,
    MIN_VOID_PX,
)
from fabstack.rendering.contracts import SceneConstraintHost


def build_scene_constraint_profile(host: SceneConstraintHost, sources, profile):
    """
        Resolve layer/stack/gap ordering once for the whole process history.

        Variables are keyed only by physical thickness, so equal features and
        equal etch depths cannot acquire column-dependent display heights.  A
        monotone relaxation raises only the physically larger side of an
        explicit comparison.  Dense scenes are allowed to grow vertically.
        """
    feature_values = {}
    gap_contexts = []
    planarizer_partition_contexts = []
    bond_attachment_variants = []
    ordinary_feature_keys = set()
    default_nominal_um = float((profile or {}).get("nominal_substrate_um", 0.0) or 0.0)

    def remember(height_um):
        try:
            height = float(height_um)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(height) or height <= 1e-12:
            return None
        key = host._display_feature_profile_key(height)
        feature_values[key] = height
        return key

    for source in sources or []:
        if len(source) >= 3:
            _source_index, grid, row_heights = source[:3]
            structural_roles = source[3] if len(source) >= 4 else None
            source_structural_bodies = source[4] if len(source) >= 5 else None
            planarizer_roles = source[5] if len(source) >= 6 else None
            source_planarizer_metadata = source[6] if len(source) >= 7 else None
        else:
            grid, row_heights = source[0], source[1]
            structural_roles = None
            source_structural_bodies = None
            planarizer_roles = None
            source_planarizer_metadata = None
        source_body_nominals = {}
        if isinstance(source_structural_bodies, dict):
            for raw_id, metadata in source_structural_bodies.items():
                if not isinstance(metadata, dict):
                    continue
                try:
                    body_id = int(metadata.get("body_id", raw_id))
                    body_nominal = float(
                        metadata.get("nominal_thickness_um", 0.0) or 0.0
                    )
                except (TypeError, ValueError):
                    continue
                if body_id > 0 and np.isfinite(body_nominal) and body_nominal > 1e-12:
                    source_body_nominals[body_id] = body_nominal
        if not source_body_nominals and structural_roles is not None:
            global_body_nominals = (
                (profile or {}).get("structural_body_nominals_um", {}) or {}
            )
            for raw_id in np.unique(np.asarray(structural_roles)):
                try:
                    body_id = int(raw_id)
                    body_nominal = float(global_body_nominals.get(body_id, 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                if body_id > 0 and body_nominal > 1e-12:
                    source_body_nominals[body_id] = body_nominal
        source_operation = ""
        source_operation_material = None
        try:
            source_state = host.history[int(_source_index)]
        except (TypeError, ValueError, IndexError, AttributeError):
            source_state = None
        if isinstance(source_state, dict):
            source_operation = str(source_state.get("operation", "") or "").strip()
            source_material_name = str(source_state.get("material_name", "") or "").strip()
            if source_material_name:
                try:
                    source_operation_material = int(host.material_names.index(source_material_name))
                except (ValueError, AttributeError):
                    source_operation_material = None
        try:
            src = np.asarray(grid)
            band_data = host._display_active_band_data(
                src,
                row_heights,
                structural_role_grid=structural_roles,
                planarizer_role_grid=planarizer_roles,
            )
        except Exception:
            continue
        if band_data is None:
            continue

        columns = host._display_raw_material_columns(
            band_data["active"],
            band_data["active_heights"],
            process_profile=profile,
            active_structural_roles=band_data.get("active_structural_roles"),
            active_planarizer_roles=band_data.get("active_planarizer_roles"),
        )

        # At creation, an intact bonded wafer has one exact attachment
        # plane even when the structure below reaches it through several
        # different layer/void decompositions.  Record those physical
        # path identities now so the shared nonlinear profile can solve
        # them additively.  This is deliberately process-wide: both the
        # pre-Bond state and Bond state then use the same resolved feature
        # sizes, so bonding cannot move an existing film or substrate.
        if source_operation == "Bond" and isinstance(source_state, dict):
            try:
                source_index = int(_source_index)
            except (TypeError, ValueError):
                source_index = -1
            previous_body_ids = set()
            if 0 < source_index < len(getattr(host, "history", [])):
                previous_state = host.history[source_index - 1]
                previous_roles = np.asarray(
                    previous_state.get("structural_role_grid", [])
                ) if isinstance(previous_state, dict) else np.asarray([])
                if previous_roles.size:
                    previous_body_ids = {
                        int(value) for value in np.unique(previous_roles)
                        if int(value) > 0
                    }
            current_body_ids = {
                int(value)
                for value in np.unique(np.asarray(
                    band_data.get("active_structural_roles")
                ))
                if int(value) > 0
            }
            new_body_ids = sorted(current_body_ids - previous_body_ids)
            for attached_body_id in new_body_ids:
                nominal_um = float(
                    source_body_nominals.get(int(attached_body_id), 0.0) or 0.0
                )
                if nominal_um <= 1e-12:
                    continue
                expected_path_body_ids = tuple(sorted(
                    int(body_id) for body_id in source_body_nominals
                    if int(body_id) != int(attached_body_id)
                ))
                attachment_paths = []
                intact_body = True
                for column_index, segments in enumerate(columns):
                    body_indices = [
                        idx for idx, segment in enumerate(segments)
                        if int(segment.get("structural_body_id", 0))
                        == int(attached_body_id)
                    ]
                    body_um = float(sum(
                        max(0.0, float(segments[idx].get("height_um", 0.0)))
                        for idx in body_indices
                    ))
                    tolerance_um = max(1e-9, nominal_um * 1e-8)
                    if (
                        not body_indices
                        or min(body_indices) != 0
                        or abs(body_um - nominal_um) > tolerance_um
                    ):
                        intact_body = False
                        break

                    path_segments = segments[max(body_indices) + 1:]
                    positive_keys = []
                    path_groups = {}
                    # A completely removed wafer is still part of the
                    # signed attachment path: its zero remaining core and
                    # full removal must cancel the real through-hole void.
                    # Omitting absent bodies split through-hole columns
                    # into another group and left the rigid slab bent.
                    path_body_heights = {
                        int(body_id): 0.0
                        for body_id in expected_path_body_ids
                    }
                    for segment in path_segments:
                        body_id = int(segment.get("structural_body_id", 0))
                        if body_id > 0:
                            path_body_heights[body_id] = (
                                path_body_heights.get(body_id, 0.0)
                                + max(0.0, float(segment.get("height_um", 0.0)))
                            )
                            continue
                        source_id = str(segment.get(
                            "source_segment_id",
                            f"{column_index}:{len(path_groups)}",
                        ))
                        path_groups.setdefault(source_id, []).append(segment)
                    for group in path_groups.values():
                        key = remember(sum(
                            max(0.0, float(segment.get("height_um", 0.0)))
                            for segment in group
                        ))
                        if key is not None:
                            positive_keys.append(key)
                            ordinary_feature_keys.add(key)

                    negative_keys = []
                    for body_id, current_um in sorted(path_body_heights.items()):
                        body_nominal_um = float(
                            source_body_nominals.get(int(body_id), 0.0) or 0.0
                        )
                        if body_nominal_um <= 1e-12:
                            continue
                        removed_um = max(0.0, body_nominal_um - current_um)
                        key = remember(removed_um)
                        if key is not None:
                            negative_keys.append(key)
                            ordinary_feature_keys.add(key)
                    attachment_paths.append({
                        "group": (
                            int(source_index),
                            int(attached_body_id),
                            expected_path_body_ids,
                        ),
                        "positive_keys": tuple(sorted(positive_keys)),
                        "negative_keys": tuple(sorted(negative_keys)),
                    })
                if intact_body and len(attachment_paths) > 1:
                    bond_attachment_variants.extend(attachment_paths)

        # A partial etch of a nonlinear-scaled planarizer must be treated
        # as an explicit feature.  Scaling the original and remaining
        # thickness independently would render a real removal as
        # H(original)-H(remaining), which can be zero at compact scale.
        # Record both positive overburden and negative recess-fill
        # partitions so the shared process profile can enforce:
        #
        #   H(original) = H(remaining) + H(removed)
        #
        # and make the visible surface drop equal H(total removed).
        if isinstance(source_planarizer_metadata, dict):
            active_planarizer = np.asarray(
                band_data.get("active_planarizer_roles"),
                dtype=np.int32,
            )
            present_planarizer_ids = {
                abs(int(value))
                for value in np.unique(active_planarizer)
                if int(value) != 0
            }
            for role_id in sorted(present_planarizer_ids):
                metadata = source_planarizer_metadata.get(
                    int(role_id),
                    source_planarizer_metadata.get(str(int(role_id)), {}),
                )
                if not isinstance(metadata, dict):
                    continue
                try:
                    original_positive_um = max(
                        0.0,
                        float(metadata.get("requested_thickness_um", 0.0) or 0.0),
                    )
                except (TypeError, ValueError):
                    original_positive_um = 0.0
                raw_fill = metadata.get("fill_thickness_um_by_column", [])
                fill_by_column = list(raw_fill) if isinstance(raw_fill, (list, tuple, np.ndarray)) else []
                for column_index, segments in enumerate(columns):
                    try:
                        original_fill_um = max(
                            0.0,
                            float(fill_by_column[column_index])
                            if column_index < len(fill_by_column)
                            else 0.0,
                        )
                    except (TypeError, ValueError):
                        original_fill_um = 0.0
                    remaining_positive_um = float(sum(
                        max(0.0, float(segment.get("height_um", 0.0)))
                        for segment in segments
                        if int(segment.get("planarizer_role", 0)) == int(role_id)
                    ))
                    remaining_fill_um = float(sum(
                        max(0.0, float(segment.get("height_um", 0.0)))
                        for segment in segments
                        if int(segment.get("planarizer_role", 0)) == -int(role_id)
                    ))
                    tolerance_um = max(
                        1e-9,
                        max(original_positive_um, original_fill_um) * 1e-8,
                    )
                    removed_positive_um = max(
                        0.0,
                        original_positive_um - remaining_positive_um,
                    )
                    removed_fill_um = max(
                        0.0,
                        original_fill_um - remaining_fill_um,
                    )
                    total_removed_um = removed_positive_um + removed_fill_um
                    if total_removed_um <= tolerance_um:
                        continue
                    partition_context = {
                        "role_id": int(role_id),
                        "column": int(column_index),
                        "original_positive_key": remember(original_positive_um),
                        "remaining_positive_key": remember(remaining_positive_um),
                        "removed_positive_key": remember(removed_positive_um),
                        "original_fill_key": remember(original_fill_um),
                        "remaining_fill_key": remember(remaining_fill_um),
                        "removed_fill_key": remember(removed_fill_um),
                        "total_original_key": remember(
                            original_positive_um + original_fill_um
                        ),
                        "total_remaining_key": remember(
                            remaining_positive_um + remaining_fill_um
                        ),
                        "total_removed_key": remember(total_removed_um),
                    }
                    planarizer_partition_contexts.append(partition_context)
                    ordinary_feature_keys.update(
                        key for name, key in partition_context.items()
                        if str(name).endswith("_key") and key is not None
                    )
        # Infer wafer order and the physical bands between bodies from
        # the shared row geometry. The inter-body bands are required for
        # connectivity: a surviving solid bonding layer blocks an etch,
        # while an open/removed band contributes to the exposed gap.
        active_roles = np.asarray(band_data.get("active_structural_roles"))
        active_grid = np.asarray(band_data["active"])
        active_heights = np.asarray(band_data["active_heights"], dtype=float)
        source_body_bounds = {}
        for body_id in source_body_nominals:
            body_rows = np.where(np.any(active_roles == int(body_id), axis=1))[0]
            if body_rows.size:
                source_body_bounds[int(body_id)] = (
                    int(body_rows.min()),
                    int(body_rows.max()),
                )
        source_body_order = [
            body_id for body_id, _bounds in sorted(
                source_body_bounds.items(),
                key=lambda item: (item[1][0], item[1][1], item[0]),
            )
        ]
        separator_rows_by_pair = {}
        separator_bands_by_pair_column = {}
        for upper_id, lower_id in zip(source_body_order, source_body_order[1:]):
            upper_end = int(source_body_bounds[upper_id][1])
            lower_start = int(source_body_bounds[lower_id][0])
            pair = (upper_id, lower_id)
            separator_rows = np.arange(
                upper_end + 1,
                max(upper_end + 1, lower_start),
                dtype=int,
            )
            separator_rows_by_pair[pair] = separator_rows

            # Keep distinct interface films distinct, using the nearest
            # stable intact column as each opening's local reference.
            # Rendering merges vertically per column by material/role;
            # using whole-row topology here would let an unrelated mask
            # elsewhere split a locally contiguous 10 um film into
            # H(5)+H(5). One global reference is also insufficient when
            # different lateral regions of the same bonded pair have
            # different interface stacks. Prefer nearby columns containing
            # both bodies and the complete separator, then fall back to
            # the most complete nearby column when no intact interface
            # remains anywhere.
            upper_bounds = source_body_bounds[upper_id]
            lower_bounds = source_body_bounds[lower_id]
            reference_candidates = []
            for reference_column in range(active_grid.shape[1]):
                has_upper = np.any(
                    active_roles[upper_bounds[0]:upper_bounds[1] + 1, reference_column]
                    == int(upper_id)
                )
                has_lower = np.any(
                    active_roles[lower_bounds[0]:lower_bounds[1] + 1, reference_column]
                    == int(lower_id)
                )
                if not has_upper or not has_lower:
                    continue
                solid_count = int(np.count_nonzero(
                    active_grid[separator_rows, reference_column]
                    != int(host.VOID_INDEX)
                ))
                reference_candidates.append({
                    "column": int(reference_column),
                    "complete": bool(solid_count == int(separator_rows.size)),
                    "solid_count": int(solid_count),
                })
            complete_candidates = [
                candidate for candidate in reference_candidates
                if bool(candidate["complete"])
            ]
            fallback_solid_count = max(
                (int(candidate["solid_count"]) for candidate in reference_candidates),
                default=-1,
            )
            fallback_candidates = [
                candidate for candidate in reference_candidates
                if int(candidate["solid_count"]) == int(fallback_solid_count)
            ]
            for target_column in range(active_grid.shape[1]):
                candidates = complete_candidates or fallback_candidates
                reference_column = min(
                    candidates,
                    key=lambda candidate: (
                        abs(int(candidate["column"]) - int(target_column)),
                        int(candidate["column"]),
                    ),
                )["column"] if candidates else 0
                separator_bands = []
                current_band = []
                current_signature = None
                for row_index in separator_rows:
                    signature = (
                        int(active_grid[int(row_index), reference_column]),
                        int(active_roles[int(row_index), reference_column]),
                    )
                    if current_band and signature != current_signature:
                        separator_bands.append(np.asarray(current_band, dtype=int))
                        current_band = []
                    current_band.append(int(row_index))
                    current_signature = signature
                if current_band:
                    separator_bands.append(np.asarray(current_band, dtype=int))
                separator_bands_by_pair_column[
                    (int(upper_id), int(lower_id), int(target_column))
                ] = tuple(separator_bands)

        for column_index, segments in enumerate(columns):
            column_atom_keys = set()
            core_idx = next(
                (idx for idx, segment in enumerate(segments)
                 if bool(segment.get("is_substrate_core", False))),
                None,
            )
            source_groups = {}
            for segment in segments:
                source_groups.setdefault(
                    str(segment.get("source_segment_id", id(segment))),
                    [],
                ).append(segment)
            for group in source_groups.values():
                segment = group[0]
                key = remember(segment.get(
                    "source_height_um",
                    sum(float(item.get("height_um", 0.0)) for item in group),
                ))
                if key is None:
                    continue
                material_idx = int(segment.get("material", host.VOID_INDEX))
                if (
                    material_idx != int(host.VOID_INDEX)
                    and not bool(segment.get("is_substrate_core", False))
                    and not bool(segment.get("is_structural_body", False))
                ):
                    column_atom_keys.add(key)
                    ordinary_feature_keys.add(key)

            column_stacks = tuple(
                stack for stack in host._display_contiguous_solid_stack_keys(segments)
                if stack
            )
            atom_keys = tuple(sorted(
                column_atom_keys,
                key=lambda key: feature_values[key],
            ))
            sorted_stacks = tuple(sorted(
                column_stacks,
                key=lambda stack: (

                    sum(feature_values[key] for key in stack),
                    tuple(feature_values[key] for key in stack),
                ),
            ))

            # Resolve every wafer body independently, including buried or
            # imported bodies. A missing body in one column is a true
            # through-etch only when that body already exists in this
            # source state's structural metadata.
            body_removals = []
            for body_id, body_nominal_um in source_body_nominals.items():
                current_um = float(sum(
                    float(segment.get("height_um", 0.0))
                    for segment in segments
                    if int(segment.get("structural_body_id", 0)) == int(body_id)
                ))
                removed_um = max(0.0, float(body_nominal_um) - current_um)
                gap_key = remember(removed_um)
                if gap_key is not None:
                    body_removals.append({
                        "body_id": int(body_id),
                        "nominal_um": float(body_nominal_um),
                        "current_um": float(current_um),
                        "removed_um": float(removed_um),
                        "gap_key": gap_key,
                    })

            active_body_id = None
            if core_idx is not None:
                active_body_id = int(
                    segments[core_idx].get("structural_body_id", 1) or 1
                )
            active_removal = next(
                (entry for entry in body_removals if entry["body_id"] == active_body_id),
                None,
            )

            # Compatibility for legacy/synthetic scenes with no role-body
            # metadata: keep the original material-based substrate path.
            if active_removal is None and core_idx is not None and not source_body_nominals:
                core_nominal_um = float(
                    segments[core_idx].get("structural_nominal_um", 0.0) or 0.0
                )
                if core_nominal_um <= 1e-12:
                    core_nominal_um = default_nominal_um
                core_height_um = float(segments[core_idx].get("height_um", 0.0))
                removed_um = max(0.0, core_nominal_um - core_height_um)
                gap_key = remember(removed_um)
                if gap_key is not None:
                    active_removal = {
                        "body_id": int(active_body_id or 1),
                        "nominal_um": float(core_nominal_um),
                        "current_um": float(core_height_um),
                        "removed_um": float(removed_um),
                        "gap_key": gap_key,
                    }
                    body_removals.append(active_removal)

            if active_removal is not None and core_idx is not None:
                if core_idx >= len(segments) / 2.0:
                    process_side = segments[:core_idx]
                else:
                    process_side = segments[core_idx + 1:]
                void_count = sum(
                    1 for segment in process_side
                    if int(segment.get("material", host.VOID_INDEX)) == int(host.VOID_INDEX)
                    and bool(segment.get("in_substrate_recess", False))
                    and float(segment.get("height_um", 0.0)) > 1e-12
                )
                minimum_recess_rows = 0
                for group in source_groups.values():
                    if bool(group[0].get("is_substrate_core", False)):
                        continue
                    material_idx = int(group[0].get("material", host.VOID_INDEX))
                    in_recess = [
                        segment for segment in group
                        if bool(segment.get("in_substrate_recess", False))
                    ]
                    if not in_recess:
                        continue
                    if material_idx == int(host.VOID_INDEX):
                        minimum_recess_rows += len(in_recess) * int(MIN_VOID_PX)
                        continue
                    if bool(group[0].get("is_structural_body", False)):
                        continue
                    if (
                        bool(group[0].get("is_planarizer_fill", False))
                        or (
                            source_operation == "Spincoating"
                            and source_operation_material is not None
                            and material_idx == int(source_operation_material)
                        )
                    ):
                        minimum_recess_rows += 1
                        continue
                    source_height = float(group[0].get(
                        "source_height_um",
                        sum(float(segment.get("height_um", 0.0)) for segment in group),
                    ))
                    total_target = host._display_base_feature_rows(source_height)
                    apportioned = host._apportion_display_rows(
                        total_target,
                        [float(segment.get("height_um", 0.0)) for segment in group],
                    )
                    minimum_recess_rows += int(sum(
                        target for segment, target in zip(group, apportioned)
                        if bool(segment.get("in_substrate_recess", False))
                    ))
                gap_contexts.append({
                    "gap_key": active_removal["gap_key"],
                    "nominal_um": active_removal["nominal_um"],
                    "structural_body_id": active_removal["body_id"],
                    "atom_keys": atom_keys,
                    "stacks": sorted_stacks,
                    "void_count": int(void_count),
                    "minimum_recess_rows": int(minimum_recess_rows),
                })

            for entry in body_removals:
                if active_removal is not None and entry["body_id"] == active_removal["body_id"]:
                    continue
                gap_contexts.append({
                    "gap_key": entry["gap_key"],
                    "nominal_um": entry["nominal_um"],
                    "structural_body_id": entry["body_id"],
                    "atom_keys": (),
                    "stacks": (),
                    "void_count": 0,
                    "minimum_recess_rows": 0,
                })

            # A surface step may traverse several bonded wafers. Its
            # displayed size is the sum of those independently truthful
            # body removals, and layers/stacks must be compared against
            # that complete step rather than only the last body reached.
            removal_by_body = {
                int(entry["body_id"]): entry for entry in body_removals
            }
            connected_removals = []
            previous_body_id = None
            for body_id in source_body_order:
                if previous_body_id is not None:
                    separator_rows = separator_rows_by_pair.get(
                        (int(previous_body_id), int(body_id)),
                        np.asarray([], dtype=int),
                    )
                    if separator_rows.size:
                        separator_solid = np.any(
                            active_grid[separator_rows, int(column_index)]
                            != int(host.VOID_INDEX)
                        )
                        if separator_solid:
                            break
                        for separator_band in separator_bands_by_pair_column.get(
                            (
                                int(previous_body_id),
                                int(body_id),
                                int(column_index),
                            ),
                            (),
                        ):
                            separator_um = float(np.sum(active_heights[separator_band]))
                            separator_key = remember(separator_um)
                            if separator_key is None:
                                continue
                            connected_removals.append({
                                "body_id": 0,
                                "nominal_um": float(separator_um),
                                "current_um": 0.0,
                                "removed_um": float(separator_um),
                                "gap_key": separator_key,
                            })
                entry = removal_by_body.get(int(body_id))
                if entry is None:
                    # The first unetched wafer encountered from the
                    # process side blocks every deeper cavity.
                    break
                connected_removals.append(entry)
                if float(entry.get("current_um", 0.0)) > 1e-12:
                    # A partial core remains, so the exposed path ends in
                    # this body and cannot include buried removals.
                    break
                previous_body_id = int(body_id)

            path_starts_above_active_core = bool(
                connected_removals
                and int(connected_removals[0].get("body_id", 0))
                != int(active_body_id or 0)
            )
            if len(connected_removals) > 1 or path_starts_above_active_core:
                cumulative_um = float(sum(
                    entry["removed_um"] for entry in connected_removals
                ))
                cumulative_key = remember(cumulative_um)
                if cumulative_key is not None:
                    gap_contexts.append({
                        "gap_key": cumulative_key,
                        "component_gap_keys": tuple(
                            entry["gap_key"] for entry in connected_removals
                        ),
                        "nominal_um": float(sum(
                            entry["nominal_um"] for entry in connected_removals
                        )),
                        "structural_body_id": 0,
                        "atom_keys": atom_keys,
                        "stacks": sorted_stacks,
                        "void_count": 0,
                        "minimum_recess_rows": 0,
                        "is_cumulative": True,
                    })

    bond_attachment_equations = []
    attachment_groups = {}
    for variant in bond_attachment_variants:
        attachment_groups.setdefault(variant["group"], []).append(variant)
    for group, variants in attachment_groups.items():
        unique_variants = {}
        for variant in variants:
            signature = (
                tuple(variant.get("positive_keys", ())),
                tuple(variant.get("negative_keys", ())),
            )
            unique_variants[signature] = variant
        ordered_variants = sorted(
            unique_variants.values(),
            key=lambda variant: (
                len(variant.get("positive_keys", ()))
                + len(variant.get("negative_keys", ())),
                tuple(variant.get("positive_keys", ())),
                tuple(variant.get("negative_keys", ())),
            ),
        )
        if len(ordered_variants) < 2:
            continue
        reference = ordered_variants[0]
        for variant in ordered_variants[1:]:
            # Equality of two signed paths:
            #   ref_pos - ref_removed == var_pos - var_removed
            # becomes an equality between two positive composites:
            #   ref_pos + var_removed == var_pos + ref_removed.
            lhs_counts = {}
            rhs_counts = {}
            for key in (
                tuple(reference.get("positive_keys", ()))
                + tuple(variant.get("negative_keys", ()))
            ):
                lhs_counts[key] = lhs_counts.get(key, 0) + 1
            for key in (
                tuple(variant.get("positive_keys", ()))
                + tuple(reference.get("negative_keys", ()))
            ):
                rhs_counts[key] = rhs_counts.get(key, 0) + 1
            for key in set(lhs_counts) & set(rhs_counts):
                cancel = min(lhs_counts[key], rhs_counts[key])
                lhs_counts[key] -= cancel
                rhs_counts[key] -= cancel
            lhs_keys = tuple(sorted(
                key for key, count in lhs_counts.items() for _ in range(count)
            ))
            rhs_keys = tuple(sorted(
                key for key, count in rhs_counts.items() for _ in range(count)
            ))
            if not lhs_keys and not rhs_keys:
                continue
            lhs_um = float(sum(feature_values[key] for key in lhs_keys))
            rhs_um = float(sum(feature_values[key] for key in rhs_keys))
            tolerance_um = max(1e-9, max(lhs_um, rhs_um) * 1e-8)
            if abs(lhs_um - rhs_um) > tolerance_um:
                # Ambiguous legacy/external geometry: do not invent an
                # attachment identity that is not physically exact.
                continue
            equation = {
                "group": group,
                "lhs_keys": lhs_keys,
                "rhs_keys": rhs_keys,
                "physical_um": float(max(lhs_um, rhs_um)),
            }
            bond_attachment_equations.append(equation)
            ordinary_feature_keys.update(lhs_keys)
            ordinary_feature_keys.update(rhs_keys)

    unique_attachment_equations = {}
    for equation in bond_attachment_equations:
        signature = (
            equation["group"],
            equation["lhs_keys"],
            equation["rhs_keys"],
        )
        unique_attachment_equations[signature] = equation
    bond_attachment_equations = [
        unique_attachment_equations[key]
        for key in sorted(unique_attachment_equations, key=str)
    ]

    # Repeated columns normally share identical topology.  Keeping only
    # unique comparison signatures makes the relaxation independent of
    # feature width while retaining different stacks at the same depth.
    unique_contexts = {}
    for context in gap_contexts:
        signature = (
            context["gap_key"],
            round(float(context.get("nominal_um", default_nominal_um)), 9),
            int(context.get("structural_body_id", 1) or 1),
            tuple(context.get("component_gap_keys", ()) or ()),
            bool(context.get("is_cumulative", False)),
            tuple(context["atom_keys"]),
            tuple(context["stacks"]),
            int(context.get("void_count", 0)),
            int(context.get("minimum_recess_rows", 0)),
        )
        unique_contexts[signature] = context
    gap_contexts = [unique_contexts[key] for key in sorted(unique_contexts)]

    unique_planarizer_partitions = {}
    for context in planarizer_partition_contexts:
        signature = (
            int(context.get("role_id", 0)),
            context.get("original_positive_key"),
            context.get("remaining_positive_key"),
            context.get("removed_positive_key"),
            context.get("original_fill_key"),
            context.get("remaining_fill_key"),
            context.get("removed_fill_key"),
            context.get("total_original_key"),
            context.get("total_remaining_key"),
            context.get("total_removed_key"),
        )
        unique_planarizer_partitions[signature] = context
    planarizer_partition_contexts = [
        unique_planarizer_partitions[key]
        for key in sorted(unique_planarizer_partitions, key=str)
    ]
    for context in gap_contexts:
        ordinary_feature_keys.add(context["gap_key"])
        ordinary_feature_keys.update(context.get("component_gap_keys", ()) or ())
        ordinary_feature_keys.update(context.get("atom_keys", ()) or ())
        for stack in context.get("stacks", ()) or ():
            ordinary_feature_keys.update(stack)
    ordered_ordinary_feature_keys = sorted(
        ordinary_feature_keys,
        key=lambda key: (feature_values[key], key),
    )

    resolved = {
        key: host._display_base_feature_rows(height)
        for key, height in feature_values.items()
    }
    compact_nominal_rows = (
        host._display_structural_body_rows(default_nominal_um)
        if default_nominal_um > 1e-12 else int(MIN_SUBSTRATE_CORE_PX)
    )
    body_nominals = {
        int(body_id): float(value)
        for body_id, value in (
            (profile or {}).get("structural_body_nominals_um", {}) or {}
        ).items()
    }
    ordered_gap_keys = sorted(
        {
            context["gap_key"] for context in gap_contexts
            if not bool(context.get("is_cumulative", False))
        },
        key=lambda key: feature_values[key],
    )

    def context_component_keys(context):
        components = tuple(context.get("component_gap_keys", ()) or ())
        return components or (context["gap_key"],)

    def context_display_rows(context):
        return int(sum(resolved[key] for key in context_component_keys(context)))

    def promote_context_to(context, required_rows):
        current_rows = context_display_rows(context)
        deficit = int(required_rows) - int(current_rows)
        if deficit <= 0:
            return False
        components = context_component_keys(context)
        component_counts = {}
        for key in components:
            component_counts[key] = component_counts.get(key, 0) + 1
        promote_key = max(
            component_counts,
            key=lambda key: (feature_values[key], key),
        )
        resolved[promote_key] += int(math.ceil(
            float(deficit) / float(component_counts[promote_key])
        ))
        return True

    cumulative_variants = {}
    for context in gap_contexts:
        if not bool(context.get("is_cumulative", False)):
            continue
        cumulative_variants.setdefault(context["gap_key"], []).append(context)

    def promote_variant_towards(smaller_context, larger_context):
        """Equalise two rendered decompositions of one physical depth."""
        deficit = context_display_rows(larger_context) - context_display_rows(
            smaller_context
        )
        if deficit <= 0:
            return False
        smaller_counts = {}
        larger_counts = {}
        for key in context_component_keys(smaller_context):
            smaller_counts[key] = smaller_counts.get(key, 0) + 1
        for key in context_component_keys(larger_context):
            larger_counts[key] = larger_counts.get(key, 0) + 1
        differentiating = {
            key: int(count) - int(larger_counts.get(key, 0))
            for key, count in smaller_counts.items()
            if int(count) > int(larger_counts.get(key, 0))
        }
        if not differentiating:
            return promote_context_to(
                smaller_context,
                context_display_rows(larger_context),
            )
        promote_key = max(
            differentiating,
            key=lambda key: (
                differentiating[key],
                feature_values[key],
                key,
            ),
        )
        resolved[promote_key] += int(math.ceil(
            float(deficit) / float(differentiating[promote_key])
        ))
        return True

    def structural_body_nominal_groups():
        """Group bodies that are physically equal within order tolerance."""
        groups = []
        ordered = sorted(
            (
                (int(body_id), float(nominal_um))
                for body_id, nominal_um in body_nominals.items()
                if float(nominal_um) > 1e-12
            ),
            key=lambda item: (item[1], item[0]),
        )
        for body_id, nominal_um in ordered:
            if (
                groups
                and host._display_physical_order_margin(
                    nominal_um,
                    groups[-1]["nominal_um"],
                ) == 0
            ):
                groups[-1]["body_ids"].append(int(body_id))
                groups[-1]["nominal_um"] = max(
                    float(groups[-1]["nominal_um"]),
                    float(nominal_um),
                )
            else:
                groups.append({
                    "nominal_um": float(nominal_um),
                    "body_ids": [int(body_id)],
                })
        return groups

    def resolved_structural_body_rows():
        """Return final body budgets for the current feature fixed point."""
        rows_by_body = {
            int(body_id): host._display_structural_body_rows(value)
            for body_id, value in body_nominals.items()
            if float(value) > 1e-12
        }
        if not rows_by_body and default_nominal_um > 1e-12:
            rows_by_body[1] = int(compact_nominal_rows)
        for context in gap_contexts:
            if bool(context.get("is_cumulative", False)):
                continue
            depth_um = feature_values[context["gap_key"]]
            gap_rows = int(context_display_rows(context))
            nominal_um = float(
                context.get("nominal_um", default_nominal_um) or 0.0
            )
            body_id = int(context.get("structural_body_id", 1) or 1)
            if depth_um >= nominal_um - max(1e-12, nominal_um * 1e-9):
                core_requirement = 0
            else:
                core_requirement = host._display_partial_core_reserve_rows(
                    nominal_um,
                    depth_um,
                )
                remaining_um = max(0.0, nominal_um - depth_um)
                remaining_margin = host._display_physical_order_margin(
                    remaining_um,
                    depth_um,
                )
                if remaining_margin > 0:
                    core_requirement = max(
                        int(core_requirement),
                        int(gap_rows) + int(remaining_margin),
                    )
            rows_by_body[body_id] = max(
                int(rows_by_body.get(

                    body_id,
                    host._display_structural_body_rows(nominal_um),
                )),
                int(gap_rows) + int(core_requirement),
            )

        # Order nominal-thickness groups rather than individual bodies.
        # A later local etch may grow one member's core budget; the whole
        # equal group must inherit that growth, including growth entering
        # from a smaller nominal group.
        nominal_groups = structural_body_nominal_groups()
        for group in nominal_groups:
            group["rows"] = max(
                int(rows_by_body.get(body_id, 0))
                for body_id in group["body_ids"]
            )
        for smaller_group, larger_group in zip(
            nominal_groups,
            nominal_groups[1:],
        ):
            margin = host._display_physical_order_margin(
                larger_group["nominal_um"],
                smaller_group["nominal_um"],
            )
            if margin > 0:
                larger_group["rows"] = max(
                    int(larger_group["rows"]),
                    int(smaller_group["rows"]) + int(margin),
                )
        for group in nominal_groups:
            for body_id in group["body_ids"]:
                rows_by_body[body_id] = int(group["rows"])
        return rows_by_body

    converged = False
    for _pass in range(int(MAX_CONSTRAINT_PASSES)):
        changed = False
        for context in planarizer_partition_contexts:
            partition_pairs = (
                (
                    context.get("original_positive_key"),
                    context.get("remaining_positive_key"),
                    context.get("removed_positive_key"),
                ),
                (
                    context.get("original_fill_key"),
                    context.get("remaining_fill_key"),
                    context.get("removed_fill_key"),
                ),
            )
            for original_key, remaining_key, removed_key in partition_pairs:
                if original_key is None or removed_key is None:
                    continue
                required_original = int(resolved[removed_key]) + (
                    int(resolved[remaining_key]) if remaining_key is not None else 0
                )
                if int(resolved[original_key]) < required_original:
                    resolved[original_key] = int(required_original)
                    changed = True

            actual_removed_rows = 0
            promotable_originals = []
            for original_key, remaining_key, removed_key in partition_pairs:
                if original_key is None or removed_key is None:
                    continue
                remaining_rows = (
                    int(resolved[remaining_key]) if remaining_key is not None else 0
                )
                actual_removed_rows += max(
                    0,
                    int(resolved[original_key]) - remaining_rows,
                )
                promotable_originals.append(original_key)

            total_removed_key = context.get("total_removed_key")
            if total_removed_key is None or not promotable_originals:
                continue
            total_removed_rows = int(resolved[total_removed_key])
            if total_removed_rows < actual_removed_rows:
                resolved[total_removed_key] = int(actual_removed_rows)
                changed = True
            elif actual_removed_rows < total_removed_rows:
                # Preserve the calibrated total removal exactly by
                # promoting an original partition, never by shrinking a
                # still-visible remaining layer.
                promote_key = max(
                    promotable_originals,
                    key=lambda key: (feature_values[key], key),
                )
                resolved[promote_key] += int(total_removed_rows - actual_removed_rows)
                changed = True

            for total_name, component_names in (
                (
                    "total_original_key",
                    ("original_positive_key", "original_fill_key"),
                ),
                (
                    "total_remaining_key",
                    ("remaining_positive_key", "remaining_fill_key"),
                ),
            ):
                total_key = context.get(total_name)
                component_keys = [
                    context.get(name) for name in component_names
                    if context.get(name) is not None
                ]
                if total_key is None or not component_keys:
                    continue
                actual_component_rows = int(sum(
                    int(resolved[key]) for key in component_keys
                ))
                total_rows = int(resolved[total_key])
                if total_rows < actual_component_rows:
                    resolved[total_key] = int(actual_component_rows)
                    changed = True
                elif actual_component_rows < total_rows:
                    # Composite keys are comparison variables; the actual
                    # render is their component sum.  Grow a real visible
                    # component so the two representations stay equal.
                    promote_key = max(
                        component_keys,
                        key=lambda key: (feature_values[key], key),
                    )
                    resolved[promote_key] += int(total_rows - actual_component_rows)
                    changed = True

        # Preserve every exact Bond attachment plane additively.  This is
        # the nonlinear counterpart of dimensional bookkeeping:
        # H(etch) must equal H(void) + H(films in that etch), and equal
        # physical contact stacks (for example H(10) and H(5)+H(5)) must
        # resolve to equal totals.  Only global feature variables grow;
        # no individual rendered column is moved or squeezed afterward.
        for equation in bond_attachment_equations:
            lhs_counts = {}
            rhs_counts = {}
            for key in equation.get("lhs_keys", ()):
                lhs_counts[key] = lhs_counts.get(key, 0) + 1
            for key in equation.get("rhs_keys", ()):
                rhs_counts[key] = rhs_counts.get(key, 0) + 1
            lhs_rows = int(sum(
                int(resolved[key]) * int(count)
                for key, count in lhs_counts.items()
            ))
            rhs_rows = int(sum(
                int(resolved[key]) * int(count)
                for key, count in rhs_counts.items()
            ))
            if lhs_rows == rhs_rows:
                continue
            smaller_counts = lhs_counts if lhs_rows < rhs_rows else rhs_counts
            if not smaller_counts:
                continue
            deficit = abs(int(lhs_rows) - int(rhs_rows))
            promote_key = max(
                smaller_counts,
                key=lambda key: (feature_values[key], key),
            )
            resolved[promote_key] += int(math.ceil(
                float(deficit) / float(smaller_counts[promote_key])
            ))
            changed = True

        # Different lateral interface regions can decompose the same
        # physical connected depth differently (for example H(10) versus
        # H(5)+H(5)). They must resolve to the same displayed step before
        # either variant is compared with layers and stacks.
        for variants in cumulative_variants.values():
            if len(variants) < 2:
                continue
            largest_variant = max(variants, key=context_display_rows)
            largest_rows = context_display_rows(largest_variant)
            for variant in variants:
                if context_display_rows(variant) < largest_rows:
                    if promote_variant_towards(variant, largest_variant):
                        changed = True
        for context in gap_contexts:
            gap_key = context["gap_key"]
            depth_um = feature_values[gap_key]
            nominal_um = float(context.get("nominal_um", default_nominal_um) or 0.0)
            compact_body_rows = host._display_structural_body_rows(nominal_um)
            minimum_void_rows = int(context.get("void_count", 0)) * int(MIN_VOID_PX)
            minimum_recess_rows = max(
                minimum_void_rows,
                int(context.get("minimum_recess_rows", 0)),
            )
            if promote_context_to(context, minimum_recess_rows):
                changed = True

            # Give a complete through-removal the compact body baseline
            # immediately.  A second equality pass below promotes it to
            # the final body budget after every body's etch/core and
            # physical-order constraints have been resolved.
            full_removal_tolerance = max(1e-12, nominal_um * 1e-9)
            if (
                not bool(context.get("is_cumulative", False))
                and nominal_um > 1e-12
                and depth_um >= nominal_um - full_removal_tolerance
                and promote_context_to(context, compact_body_rows)
            ):
                changed = True

            # The substrate itself is an explicit comparison: a deeper
            # etched portion must not display smaller than the positive
            # remnant it physically exceeds. The converse is satisfied by
            # growing the protected core/body budget below.
            remaining_um = max(0.0, nominal_um - depth_um)
            if (
                not bool(context.get("is_cumulative", False))
                and depth_um > remaining_um + max(1e-12, nominal_um * 1e-9)
            ):
                margin = host._display_physical_order_margin(depth_um, remaining_um)
                reserve = host._display_partial_core_reserve_rows(nominal_um, depth_um)
                required_gap = max(
                    int(reserve) + int(margin),
                    int(math.ceil((compact_body_rows + int(margin)) / 2.0)),
                )
                if promote_context_to(context, required_gap):
                    changed = True

            for atom_key in context["atom_keys"]:
                atom_um = feature_values[atom_key]
                forward_margin = host._display_physical_order_margin(depth_um, atom_um)
                reverse_margin = host._display_physical_order_margin(atom_um, depth_um)
                if forward_margin > 0:
                    required = resolved[atom_key] + int(forward_margin)
                    if promote_context_to(context, required):
                        changed = True
                elif reverse_margin > 0:
                    required = context_display_rows(context) + int(reverse_margin)
                    if resolved[atom_key] < required:
                        resolved[atom_key] = required
                        changed = True
                else:
                    gap_rows = context_display_rows(context)
                    atom_rows = int(resolved[atom_key])
                    if gap_rows < atom_rows:
                        if promote_context_to(context, atom_rows):
                            changed = True
                    elif atom_rows < gap_rows:
                        resolved[atom_key] = int(gap_rows)
                        changed = True

            for stack in context["stacks"]:
                stack_um = float(sum(feature_values[key] for key in stack))
                stack_rows = int(sum(resolved[key] for key in stack))
                forward_margin = host._display_physical_order_margin(depth_um, stack_um)
                reverse_margin = host._display_physical_order_margin(stack_um, depth_um)
                if forward_margin > 0:
                    required = stack_rows + int(forward_margin)
                    if promote_context_to(context, required):
                        changed = True
                elif reverse_margin > 0:
                    required = context_display_rows(context) + int(reverse_margin)
                    if stack_rows < required and stack:
                        deficit = required - stack_rows
                        counts = {}
                        for key in stack:
                            counts[key] = counts.get(key, 0) + 1
                        promote_key = max(
                            counts,
                            key=lambda key: (feature_values[key], key),
                        )
                        resolved[promote_key] += int(math.ceil(
                            float(deficit) / float(counts[promote_key])
                        ))
                        changed = True
                else:
                    gap_rows = context_display_rows(context)
                    if gap_rows < stack_rows:
                        if promote_context_to(context, stack_rows):
                            changed = True
                    elif stack_rows < gap_rows and stack:
                        deficit = gap_rows - stack_rows
                        counts = {}
                        for key in stack:
                            counts[key] = counts.get(key, 0) + 1
                        promote_key = max(
                            counts,
                            key=lambda key: (feature_values[key], key),
                        )
                        resolved[promote_key] += int(math.ceil(
                            float(deficit) / float(counts[promote_key])
                        ))
                        changed = True

        # A locally promoted shallower trench must never overtake a
        # physically deeper trench elsewhere in the same process.
        for smaller_key, larger_key in zip(ordered_gap_keys, ordered_gap_keys[1:]):
            margin = host._display_physical_order_margin(
                feature_values[larger_key],
                feature_values[smaller_key],
            )
            required = int(resolved[smaller_key]) + int(margin)
            if margin > 0 and resolved[larger_key] < required:
                resolved[larger_key] = required
                changed = True

        # Any additive promotion (for example a readable 1 nm removal
        # within a 5 um film) must propagate to every physically larger
        # ordinary feature. Structural bodies deliberately use their own
        # compact body scale and are not part of this chain.
        for smaller_key, larger_key in zip(
            ordered_ordinary_feature_keys,
            ordered_ordinary_feature_keys[1:],
        ):
            margin = host._display_physical_order_margin(
                feature_values[larger_key],
                feature_values[smaller_key],
            )
            required = int(resolved[smaller_key]) + int(margin)
            if margin > 0 and int(resolved[larger_key]) < required:
                resolved[larger_key] = int(required)
                changed = True

        # A through-removal and the structural body it replaces are two
        # physical paths to the same attachment plane.  Re-evaluate this
        # equality inside the fixed point: a later etch in another bonded
        # body (or body-order propagation) can enlarge the final body
        # budget after the compact baseline above was applied.
        provisional_body_rows = resolved_structural_body_rows()
        for context in gap_contexts:
            if bool(context.get("is_cumulative", False)):
                continue
            depth_um = feature_values[context["gap_key"]]
            nominal_um = float(
                context.get("nominal_um", default_nominal_um) or 0.0
            )
            full_removal_tolerance = max(1e-12, nominal_um * 1e-9)
            if (
                nominal_um > 1e-12
                and depth_um >= nominal_um - full_removal_tolerance
            ):
                body_id = int(context.get("structural_body_id", 1) or 1)
                target_rows = int(provisional_body_rows.get(
                    body_id,
                    host._display_structural_body_rows(nominal_um),
                ))
                if promote_context_to(context, target_rows):
                    changed = True
        if not changed:
            converged = True
            break

    violations = []
    for context in planarizer_partition_contexts:
        actual_removed_rows = 0
        for original_name, remaining_name, removed_name in (
            ("original_positive_key", "remaining_positive_key", "removed_positive_key"),
            ("original_fill_key", "remaining_fill_key", "removed_fill_key"),
        ):
            original_key = context.get(original_name)
            removed_key = context.get(removed_name)
            if original_key is None or removed_key is None:
                continue
            remaining_key = context.get(remaining_name)
            remaining_rows = int(resolved[remaining_key]) if remaining_key is not None else 0
            partition_removed = int(resolved[original_key]) - remaining_rows
            if partition_removed < int(resolved[removed_key]):
                violations.append((
                    "planarizer-partition",
                    feature_values[original_key],
                    feature_values[removed_key],
                ))
            actual_removed_rows += max(0, partition_removed)
        total_removed_key = context.get("total_removed_key")
        if (
            total_removed_key is not None
            and actual_removed_rows != int(resolved[total_removed_key])
        ):
            violations.append((
                "planarizer-total",
                feature_values[total_removed_key],
                actual_removed_rows,
                int(resolved[total_removed_key]),
            ))
        for total_name, component_names in (
            (
                "total_original_key",
                ("original_positive_key", "original_fill_key"),
            ),
            (
                "total_remaining_key",
                ("remaining_positive_key", "remaining_fill_key"),
            ),
        ):
            total_key = context.get(total_name)
            component_keys = [
                context.get(name) for name in component_names
                if context.get(name) is not None
            ]
            if (
                total_key is not None
                and component_keys
                and int(resolved[total_key])
                != int(sum(int(resolved[key]) for key in component_keys))
            ):
                violations.append((
                    "planarizer-composite",
                    feature_values[total_key],
                ))
    for smaller_key, larger_key in zip(
        ordered_ordinary_feature_keys,
        ordered_ordinary_feature_keys[1:],
    ):
        margin = host._display_physical_order_margin(
            feature_values[larger_key],
            feature_values[smaller_key],
        )
        if (
            margin > 0
            and int(resolved[larger_key])
            < int(resolved[smaller_key]) + int(margin)
        ):
            violations.append((
                "global-feature-order",
                feature_values[smaller_key],
                feature_values[larger_key],
            ))
    for equation in bond_attachment_equations:
        lhs_rows = int(sum(
            int(resolved[key]) for key in equation.get("lhs_keys", ())
        ))
        rhs_rows = int(sum(
            int(resolved[key]) for key in equation.get("rhs_keys", ())
        ))
        if lhs_rows != rhs_rows:
            violations.append((
                "bond-attachment=",
                equation.get("physical_um", 0.0),
                lhs_rows,
                rhs_rows,
            ))
    for variants in cumulative_variants.values():
        variant_rows = {context_display_rows(context) for context in variants}
        if len(variant_rows) > 1:
            violations.append(("cumulative-variant=", tuple(sorted(variant_rows))))
    for context in gap_contexts:
        gap_key = context["gap_key"]
        depth_um = feature_values[gap_key]
        displayed_gap_rows = context_display_rows(context)
        for atom_key in context["atom_keys"]:
            atom_um = feature_values[atom_key]
            forward_margin = host._display_physical_order_margin(depth_um, atom_um)
            reverse_margin = host._display_physical_order_margin(atom_um, depth_um)
            if forward_margin > 0 and displayed_gap_rows < resolved[atom_key] + forward_margin:
                violations.append(("gap>layer", depth_um, atom_um))
            elif reverse_margin > 0 and resolved[atom_key] < displayed_gap_rows + reverse_margin:
                violations.append(("layer>gap", atom_um, depth_um))
            elif forward_margin == 0 and reverse_margin == 0 and resolved[atom_key] != displayed_gap_rows:
                violations.append(("gap=layer", depth_um, atom_um))
        for stack in context["stacks"]:
            stack_um = float(sum(feature_values[key] for key in stack))
            stack_rows = int(sum(resolved[key] for key in stack))
            forward_margin = host._display_physical_order_margin(depth_um, stack_um)
            reverse_margin = host._display_physical_order_margin(stack_um, depth_um)
            if forward_margin > 0 and displayed_gap_rows < stack_rows + forward_margin:
                violations.append(("gap>stack", depth_um, stack_um))
            elif reverse_margin > 0 and stack_rows < displayed_gap_rows + reverse_margin:
                violations.append(("stack>gap", stack_um, depth_um))
            elif forward_margin == 0 and reverse_margin == 0 and stack_rows != displayed_gap_rows:
                violations.append(("gap=stack", depth_um, stack_um))

    body_rows = resolved_structural_body_rows()
    for context in gap_contexts:
        if bool(context.get("is_cumulative", False)):
            continue
        depth_um = feature_values[context["gap_key"]]
        nominal_um = float(context.get("nominal_um", default_nominal_um) or 0.0)
        full_removal_tolerance = max(1e-12, nominal_um * 1e-9)
        if (
            nominal_um > 1e-12
            and depth_um >= nominal_um - full_removal_tolerance
        ):
            body_id = int(context.get("structural_body_id", 1) or 1)
            gap_rows = int(context_display_rows(context))
            target_rows = int(body_rows.get(
                body_id,
                host._display_structural_body_rows(nominal_um),
            ))
            if gap_rows != target_rows:
                violations.append((
                    "full-removal=body",
                    nominal_um,
                    gap_rows,
                    target_rows,
                ))

    audited_nominal_groups = structural_body_nominal_groups()
    for group in audited_nominal_groups:
        group_rows = {
            int(body_rows.get(body_id, 0))
            for body_id in group["body_ids"]
        }
        if len(group_rows) > 1:
            violations.append((
                "body=body",
                float(group["nominal_um"]),
                tuple(sorted(group_rows)),
            ))

    for smaller_group, larger_group in zip(
        audited_nominal_groups,
        audited_nominal_groups[1:],
    ):
        margin = host._display_physical_order_margin(
            larger_group["nominal_um"],
            smaller_group["nominal_um"],
        )
        smaller_rows = max(
            int(body_rows.get(body_id, 0))
            for body_id in smaller_group["body_ids"]
        )
        larger_rows = min(
            int(body_rows.get(body_id, 0))
            for body_id in larger_group["body_ids"]
        )
        if margin > 0 and larger_rows < smaller_rows + int(margin):
            violations.append((
                "body>body",
                float(larger_group["nominal_um"]),
                float(smaller_group["nominal_um"]),
                larger_rows,
                smaller_rows,
            ))

    nominal_rows = int(body_rows.get(1, compact_nominal_rows))

    return {
        "resolved_feature_rows": {key: int(value) for key, value in resolved.items()},
        "feature_values_um": {key: float(value) for key, value in feature_values.items()},
        "gap_contexts": gap_contexts,
        "planarizer_partition_contexts": planarizer_partition_contexts,
        "bond_attachment_equations": bond_attachment_equations,
        "nominal_substrate_rows": int(nominal_rows),
        "structural_body_rows": {
            int(body_id): int(value) for body_id, value in body_rows.items()
        },
        "constraint_converged": bool(converged),
        "constraint_violations": violations,
    }
# -- extracted service --


__all__ = ["build_scene_constraint_profile"]
