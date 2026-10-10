"""Narrow structural contracts consumed by pure rendering services."""

from typing import Any, Protocol


class SceneConstraintHost(Protocol):
    """Display-profile capabilities required by scene constraint solving."""

    VOID_INDEX: int
    history: list[dict[str, Any]]
    material_names: list[str]

    def _apportion_display_rows(self, total_target: int, weights): ...
    def _display_active_band_data(self, src, row_heights_um=None, **kwargs): ...
    def _display_base_feature_rows(self, height_um: float) -> int: ...
    def _display_contiguous_solid_stack_keys(self, segments): ...
    def _display_feature_profile_key(self, height_um: float) -> str: ...
    def _display_partial_core_reserve_rows(
        self,
        nominal_um: float,
        depth_um: float,
    ) -> int: ...
    def _display_physical_order_margin(
        self,
        larger_um: float,
        smaller_um: float,
    ) -> int: ...
    def _display_raw_material_columns(self, active, active_heights, **kwargs): ...
    def _display_structural_body_rows(self, height_um: float) -> int: ...


class SegmentTargetHost(Protocol):
    """Display-profile capabilities required by per-column target building."""

    VOID_INDEX: int
    history: list[dict[str, Any]]
    material_names: list[str]
    xsec: Any

    def _apportion_display_rows(self, total_target: int, weights): ...
    def _display_material_count_profile_key(
        self,
        material_idx,
        thickness_um: float,
    ) -> str: ...
    def _display_profile_feature_rows(self, height_um: float, profile=None) -> int: ...
    def _display_raw_material_columns(self, active, active_heights, **kwargs): ...
    def _display_segment_target_height(self, height_um: float, **kwargs): ...
    def _display_structural_body_rows(self, height_um: float) -> int: ...
    def _display_substrate_core_target_height(
        self,
        height_um: float,
        profile=None,
        **kwargs,
    ) -> int: ...
    def _visual_override_value_for_key(self, key: str, history_index=None): ...


__all__ = ["SceneConstraintHost", "SegmentTargetHost"]
