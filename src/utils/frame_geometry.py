"""Scale legacy game-frame pixel settings for a native capture frame.

The project historically expressed screen-space coordinates against a
``700 x 1296`` (height x width) working frame.  Native PotPlayer capture keeps
the source raster, so only settings that address the *game image* should be
scaled.  Minimap/route coordinates deliberately remain untouched because they
live in their own coordinate system.
"""

from __future__ import annotations

from copy import deepcopy
from numbers import Real
from typing import Any, Mapping, MutableMapping, Sequence


LEGACY_FRAME_SIZE = (700, 1296)


def _frame_size(value: Sequence[Real], *, name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or len(value) < 2:
        raise ValueError(f"{name} must be [height, width]")
    height, width = value[:2]
    if (
        isinstance(height, bool)
        or isinstance(width, bool)
        or not isinstance(height, Real)
        or not isinstance(width, Real)
        or height <= 0
        or width <= 0
    ):
        raise ValueError(f"{name} must contain positive numeric dimensions")
    return float(height), float(width)


def _mapping_at(
    cfg: MutableMapping[str, Any], path: tuple[str, ...]
) -> MutableMapping[str, Any] | None:
    current: Any = cfg
    for key in path:
        if not isinstance(current, MutableMapping) or key not in current:
            return None
        current = current[key]
    return current if isinstance(current, MutableMapping) else None


def _parent_and_key(
    cfg: MutableMapping[str, Any], path: tuple[str, ...]
) -> tuple[MutableMapping[str, Any], str] | None:
    parent = _mapping_at(cfg, path[:-1]) if len(path) > 1 else cfg
    if parent is None or path[-1] not in parent:
        return None
    return parent, path[-1]


def _scale_number(value: Any, factor: float) -> Any:
    """Scale pixel numbers while preserving integer-vs-float intent."""

    if isinstance(value, bool) or not isinstance(value, Real):
        return value
    scaled = float(value) * factor
    if isinstance(value, int):
        return int(round(scaled))
    return scaled


def _scale_scalar(
    cfg: MutableMapping[str, Any], path: tuple[str, ...], factor: float
) -> None:
    target = _parent_and_key(cfg, path)
    if target is None:
        return
    parent, key = target
    parent[key] = _scale_number(parent[key], factor)


def _scale_xy_value(value: Any, scale_x: float, scale_y: float) -> Any:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        return value
    scaled = (
        _scale_number(value[0], scale_x),
        _scale_number(value[1], scale_y),
    )
    return scaled if isinstance(value, tuple) else list(scaled)


def _scale_xy(
    cfg: MutableMapping[str, Any], path: tuple[str, ...], scale_x: float, scale_y: float
) -> None:
    target = _parent_and_key(cfg, path)
    if target is None:
        return
    parent, key = target
    parent[key] = _scale_xy_value(parent[key], scale_x, scale_y)


def scale_runtime_pixel_config(
    cfg: Mapping[str, Any], output_size: Sequence[Real]
) -> dict[str, Any]:
    """Return a deep-copied config scaled for ``output_size``.

    ``output_size`` and ``game_window.coordinate_reference_size`` use
    ``[height, width]`` order.  Missing sections are accepted.  The input is
    never mutated, so callers can retain one legacy config and derive a fresh
    runtime copy whenever the capture size changes without compounding scale.

    Scalar radii/distances are multiplied by the larger axis scale.  This is a
    conservative representation of a legacy circular search area when the
    native frame uses unequal x/y scales.
    """

    if not isinstance(cfg, Mapping):
        raise TypeError("cfg must be a mapping")

    result: dict[str, Any] = deepcopy(dict(cfg))
    game_window = result.get("game_window")
    reference = LEGACY_FRAME_SIZE
    if isinstance(game_window, Mapping):
        reference = game_window.get("coordinate_reference_size", reference)

    ref_h, ref_w = _frame_size(reference, name="coordinate_reference_size")
    output_h, output_w = _frame_size(output_size, name="output_size")
    scale_x = output_w / ref_w
    scale_y = output_h / ref_h
    scale_radius = max(scale_x, scale_y)

    # UI points and rectangles are all in full game-frame coordinates.  Only
    # two-number vectors are coordinates; thresholds such as
    # ``login_button_thres`` remain unchanged.
    ui_coords = _mapping_at(result, ("ui_coords",))
    if ui_coords is not None:
        _scale_scalar(result, ("ui_coords", "ui_y_start"), scale_y)
        for key, value in list(ui_coords.items()):
            if key == "ui_y_start":
                continue
            ui_coords[key] = _scale_xy_value(value, scale_x, scale_y)

    # Rune message ROIs.  Iterate language suffixes so new localizations get
    # the same behavior without expanding a hard-coded language list.
    for section_name in list(result):
        if not (
            section_name.startswith("rune_warning_")
            or section_name.startswith("rune_enable_msg_")
        ):
            continue
        _scale_xy(result, (section_name, "top_left"), scale_x, scale_y)
        _scale_xy(result, (section_name, "bottom_right"), scale_x, scale_y)

    _scale_scalar(result, ("rune_detect", "box_width"), scale_x)
    _scale_scalar(result, ("rune_detect", "box_height"), scale_y)
    _scale_scalar(result, ("rune_find", "rune_trigger_distance_x"), scale_x)
    _scale_scalar(result, ("rune_find", "rune_trigger_distance_y"), scale_y)
    _scale_xy(result, ("rune_solver", "arrow_box_coord"), scale_x, scale_y)
    _scale_scalar(result, ("rune_solver", "arrow_box_interval"), scale_x)
    _scale_scalar(result, ("rune_solver", "arrow_box_size"), scale_radius)

    # Combat and game-camera geometry.
    for section in ("aoe_skill", "directional_attack"):
        _scale_scalar(result, (section, "range_x"), scale_x)
        _scale_scalar(result, (section, "range_y"), scale_y)
    _scale_scalar(result, ("monster_detect", "search_box_margin"), scale_radius)
    _scale_scalar(result, ("character", "width"), scale_x)
    _scale_scalar(result, ("character", "height"), scale_y)
    _scale_scalar(result, ("edge_teleport", "trigger_box_width"), scale_x)
    _scale_scalar(result, ("edge_teleport", "trigger_box_height"), scale_y)
    _scale_xy(result, ("party_red_bar", "offset"), scale_x, scale_y)

    # Name/medal/pet/appearance anchors all refer to the full game frame.
    _scale_xy(result, ("nametag", "offset"), scale_x, scale_y)
    _scale_scalar(result, ("nametag", "split_width"), scale_x)
    _scale_scalar(
        result, ("nametag", "jump_confirm_distance"), scale_radius
    )
    _scale_scalar(result, ("nametag", "jump_confirm_radius"), scale_radius)

    _scale_xy(
        result,
        ("nametag", "overhead_marker", "player_offset"),
        scale_x,
        scale_y,
    )
    _scale_scalar(
        result, ("nametag", "overhead_marker", "component_width"), scale_x
    )
    _scale_scalar(
        result, ("nametag", "overhead_marker", "component_height"), scale_y
    )
    for key in (
        "match_search_tolerance",
        "local_search_radius",
    ):
        _scale_scalar(
            result, ("nametag", "overhead_marker", key), scale_radius
        )

    _scale_scalar(result, ("nametag", "medal", "id_fragment_width"), scale_x)
    _scale_scalar(result, ("nametag", "medal", "id_fragment_stride"), scale_x)
    _scale_scalar(result, ("nametag", "medal", "center_offset_x"), scale_x)
    _scale_scalar(result, ("nametag", "medal", "vertical_gap"), scale_y)
    _scale_xy(
        result, ("nametag", "medal", "search_tolerance"), scale_x, scale_y
    )

    _scale_xy(result, ("nametag", "pet", "medal_offset"), scale_x, scale_y)
    _scale_xy(
        result,
        ("nametag", "pet", "medal_search_tolerance"),
        scale_x,
        scale_y,
    )
    _scale_scalar(
        result, ("nametag", "pet", "yolo_name_vertical_gap"), scale_y
    )
    _scale_xy(
        result,
        ("nametag", "pet", "yolo_name_search_tolerance"),
        scale_x,
        scale_y,
    )
    _scale_scalar(result, ("nametag", "pet", "yolo_name_max_gap"), scale_y)

    for key in (
        "local_search_radius",
        "validation_distance",
        "climb_validation_distance",
        "global_confirm_radius",
    ):
        _scale_scalar(result, ("nametag", "appearance", key), scale_radius)

    appearance = _mapping_at(result, ("nametag", "appearance"))
    if appearance is not None:
        templates = appearance.get("templates")
        # load_yaml() normalizes YAML sequences to tuples.  Accept both raw
        # dict/list configs and the tuple form used by the real application.
        if isinstance(templates, (list, tuple)):
            for template in templates:
                if isinstance(template, MutableMapping) and "player_offset" in template:
                    template["player_offset"] = _scale_xy_value(
                        template["player_offset"], scale_x, scale_y
                    )

    return result


__all__ = ["LEGACY_FRAME_SIZE", "scale_runtime_pixel_config"]
