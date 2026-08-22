"""Render WZ geometry and graph transitions into existing route-map commands."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np

from .hunt_planner import (
    Action,
    CombatCheckpoint,
    Edge,
    NavigationGraph,
    Node,
    PatrolPlan,
)
from .recorded_route_anchors import (
    NavigationClimbAnchor,
    ProjectedRecordedRouteAnchors,
)
from .wz_catalog import load_wz_canvas
from .wz_geometry import Point, WzMap


@dataclass(frozen=True, slots=True)
class NavigationProjection:
    wz_map: WzMap
    canvas_scale: float

    @property
    def size(self) -> tuple[int, int]:
        variant = self.wz_map.selected_variant
        return (
            max(1, int(round(variant.pixel_width * self.canvas_scale))),
            max(1, int(round(variant.pixel_height * self.canvas_scale))),
        )

    @property
    def world_scale(self) -> tuple[float, float]:
        scale_x, scale_y = self.wz_map.world_to_canvas_scale
        return scale_x * self.canvas_scale, scale_y * self.canvas_scale

    def world_to_navigation(self, point: Point) -> tuple[int, int]:
        canonical = self.wz_map.world_to_canvas(point)
        return (
            int(round(canonical.x * self.canvas_scale)),
            int(round(canonical.y * self.canvas_scale)),
        )

    def canonical_to_navigation(
        self, point: tuple[float, float]
    ) -> tuple[float, float]:
        return point[0] * self.canvas_scale, point[1] * self.canvas_scale

    def navigation_to_world(self, point: tuple[float, float]) -> Point:
        """Convert a generated route-map coordinate back to WZ world space."""
        canonical = Point(
            float(point[0]) / self.canvas_scale,
            float(point[1]) / self.canvas_scale,
        )
        return self.wz_map.canvas_to_world(canonical)


@dataclass(frozen=True, slots=True)
class RenderedRouteLeg:
    """Geometry retained alongside one generated route image."""

    action: Action
    source: tuple[int, int]
    target: tuple[int, int]
    edge_ids: tuple[str, ...]
    recovery_path: int | None = None
    combat_checkpoint: CombatCheckpoint | None = None
    rope_mount: "RopeMountPlan | None" = None
    combat_checkpoint_position: tuple[int, int] | None = None
    jump_source: tuple[int, int] | None = None
    jump_trigger_bounds: tuple[int, int, int, int] | None = None
    recorded_x_anchor: int | None = None


@dataclass(frozen=True, slots=True)
class RopeMountMotion:
    """Actual minimap-pixel motion used to place a rope jump."""

    walk_speed_px_per_sec: float
    jump_height_px: float
    jump_distance_px: float
    runup_seconds: float


@dataclass(frozen=True, slots=True)
class RopeMountPlan:
    """Two-stage rope approach: stage for momentum, then jump at launch."""

    contact: tuple[int, int]
    ground_y: int
    vertical_gap_px: float
    jump_height_px: float
    jump_distance_px: float
    launch_offset_px: int
    staging_offset_px: int
    predicted_contact_height_px: float
    contact_clearance_px: float
    reachable_at_contact: bool
    approach_direction: str | None = None


@dataclass(frozen=True, slots=True)
class _RouteRenderSpec:
    edge: Edge
    source: Node
    target: Node
    metadata: RenderedRouteLeg
    approach_target: Node | None = None
    jump_source: Node | None = None


MINIMUM_WALK_LEG_PIXELS = 4
MAX_DIRECTIONAL_JUMP_TRIGGER_LEAD_PIXELS = 2


def _directional_jump_trigger_pixel(
    source_pixel: tuple[int, int],
    target_pixel: tuple[int, int],
    action_half_width: int,
) -> tuple[int, int]:
    """Move a horizontal jump trigger inward without changing WZ geometry."""
    delta_x = target_pixel[0] - source_pixel[0]
    if delta_x == 0:
        return source_pixel
    lead = min(
        MAX_DIRECTIONAL_JUMP_TRIGGER_LEAD_PIXELS,
        max(1, int(action_half_width) - 1),
    )
    direction = 1 if delta_x > 0 else -1
    return source_pixel[0] - direction * lead, source_pixel[1]


def _directional_jump_trigger_bounds(
    source_pixel: tuple[int, int],
    target_pixel: tuple[int, int],
    action_half_width: int,
    action_half_height: int,
    jump_distance_px: float | None,
) -> tuple[int, int, int, int]:
    """Return the physically reachable launch band nearest the source edge."""
    source_x, source_y = source_pixel
    target_x, _ = target_pixel
    maximum_span = max(0, int(action_half_width) * 2)
    measured_distance = (
        float(jump_distance_px)
        if jump_distance_px is not None
        and math.isfinite(float(jump_distance_px))
        and float(jump_distance_px) > 0
        else None
    )

    if measured_distance is None:
        trigger_x, _ = _directional_jump_trigger_pixel(
            source_pixel, target_pixel, action_half_width
        )
        left = trigger_x - action_half_width
        right = trigger_x + action_half_width
    elif target_x > source_x:
        # A rightward launch must occur late enough that the measured airborne
        # distance reaches the first target-platform foothold pixel. Keep the
        # outer bound at the source edge so no trigger is painted over empty
        # space.
        left = max(
            int(math.ceil(target_x - measured_distance)),
            source_x - maximum_span,
        )
        right = source_x
        if left > right:
            left = right
    else:
        left = source_x
        right = min(
            int(math.floor(target_x + measured_distance)),
            source_x + maximum_span,
        )
        if right < left:
            right = left

    return (
        int(left),
        int(source_y - action_half_height),
        int(right),
        int(source_y + action_half_height),
    )


def _recorded_jump_trigger(
    anchors: ProjectedRecordedRouteAnchors,
    metadata: RenderedRouteLeg,
    source_pixel: tuple[int, int],
    target_pixel: tuple[int, int],
    default_bounds: tuple[int, int, int, int],
    action_half_width: int,
    action_half_height: int,
) -> tuple[
    tuple[int, int], tuple[int, int, int, int]
] | None:
    """Return an exact recorded X trigger for a normal patrol jump."""
    if metadata.recovery_path is not None or \
            target_pixel[0] == source_pixel[0]:
        return None
    direction = 1 if target_pixel[0] > source_pixel[0] else -1
    expected_x = int(round((default_bounds[0] + default_bounds[2]) / 2.0))
    anchor = anchors.jump_for(
        source_y=source_pixel[1],
        expected_x=expected_x,
        direction=direction,
    )
    if anchor is None:
        return None
    trigger_pixel = (anchor.x, source_pixel[1])
    return trigger_pixel, (
        anchor.x - action_half_width,
        source_pixel[1] - action_half_height,
        anchor.x + action_half_width,
        source_pixel[1] + action_half_height,
    )


def _build_rope_mount_plan(
    source_pixel: tuple[int, int],
    contact: tuple[int, int],
    motion: RopeMountMotion,
    *,
    recorded_launch_offset_px: int | None = None,
    approach_direction: str | None = None,
) -> RopeMountPlan:
    """Lead the jump before the rope and reserve measured run-up room."""
    values = (
        motion.walk_speed_px_per_sec,
        motion.jump_height_px,
        motion.jump_distance_px,
    )
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("rope mount motion values must be finite and positive")
    if not math.isfinite(motion.runup_seconds) or motion.runup_seconds < 0:
        raise ValueError("rope mount run-up time must be finite and non-negative")
    if recorded_launch_offset_px is not None and \
            int(recorded_launch_offset_px) <= 0:
        raise ValueError("recorded rope launch offset must be positive")
    if approach_direction not in {None, "left", "right"}:
        raise ValueError("rope approach direction must be left or right")

    vertical_gap = max(0.0, float(source_pixel[1] - contact[1]))
    # The runtime now triggers directly from the 60 Hz Hero position. The
    # full-speed spatial point therefore targets the measured apex itself;
    # no frame/HID lead distance is baked into route geometry.
    desired_launch_offset = max(
        1,
        int(round(motion.jump_distance_px / 2.0)),
    )

    # Do not move the launch so far out that the descending arc reaches the
    # rope below its WZ bottom. Keep one pixel of vertical capture margin when
    # the measured jump height permits it.
    required_height = vertical_gap + 1.0
    if recorded_launch_offset_px is not None:
        launch_offset = int(recorded_launch_offset_px)
    elif required_height < motion.jump_height_px:
        descending_fraction = (
            1.0
            + math.sqrt(1.0 - required_height / motion.jump_height_px)
        ) / 2.0
        maximum_safe_offset = max(
            1,
            int(math.floor(
                motion.jump_distance_px * descending_fraction
            )),
        )
        launch_offset = min(desired_launch_offset, maximum_safe_offset)
    else:
        launch_offset = max(
            1,
            int(round(motion.jump_distance_px / 2.0)),
        )

    flight_fraction = min(
        1.0,
        max(0.0, launch_offset / motion.jump_distance_px),
    )
    contact_height = (
        4.0
        * motion.jump_height_px
        * flight_fraction
        * (1.0 - flight_fraction)
    )
    contact_clearance = contact_height - vertical_gap
    momentum_distance = max(
        0,
        int(math.ceil(
            motion.walk_speed_px_per_sec * motion.runup_seconds
        )),
    )
    staging_offset = launch_offset + momentum_distance
    return RopeMountPlan(
        contact=contact,
        ground_y=int(source_pixel[1]),
        vertical_gap_px=vertical_gap,
        jump_height_px=float(motion.jump_height_px),
        jump_distance_px=float(motion.jump_distance_px),
        launch_offset_px=launch_offset,
        staging_offset_px=staging_offset,
        predicted_contact_height_px=contact_height,
        contact_clearance_px=contact_clearance,
        reachable_at_contact=contact_clearance >= 1.0,
        approach_direction=approach_direction,
    )


def _horizontal_direction(source: Node, target: Node) -> int:
    if target.x > source.x:
        return 1
    if target.x < source.x:
        return -1
    return 0


def _route_sequence_specs(
    graph: NavigationGraph,
    edge_ids: tuple[str, ...],
    projection: NavigationProjection,
    *,
    recovery_path: int | None,
    combat_checkpoints: dict[int, CombatCheckpoint] | None = None,
) -> tuple[_RouteRenderSpec, ...]:
    """Merge continuous walks and discard sub-pixel route fragments."""
    combat_checkpoints = combat_checkpoints or {}
    nodes = graph.node_by_id
    edges = graph.edge_by_id
    planned_edges = [
        (edge_index, edge)
        for edge_index, edge_id in enumerate(edge_ids)
        if (edge := edges.get(edge_id)) is not None
    ]
    specs: list[_RouteRenderSpec] = []
    index = 0
    while index < len(planned_edges):
        first_plan_index, first_edge = planned_edges[index]
        last_plan_index = first_plan_index
        source = nodes[first_edge.source]
        target = nodes[first_edge.target]
        merged_edge_ids = [first_edge.id]

        if first_edge.action is Action.WALK:
            direction = _horizontal_direction(source, target)
            while index + 1 < len(planned_edges):
                next_plan_index, next_edge = planned_edges[index + 1]
                if next_edge.action is not Action.WALK or \
                        next_edge.source != planned_edges[index][1].target:
                    break
                next_source = nodes[next_edge.source]
                next_target = nodes[next_edge.target]
                if source.surface_id != next_source.surface_id or \
                        source.surface_id != next_target.surface_id or \
                        _horizontal_direction(next_source, next_target) != direction:
                    break
                if last_plan_index in combat_checkpoints:
                    break
                index += 1
                target = next_target
                last_plan_index = next_plan_index
                merged_edge_ids.append(next_edge.id)

            source_pixel = projection.world_to_navigation(
                Point(source.x, source.y)
            )
            target_pixel = projection.world_to_navigation(
                Point(target.x, target.y)
            )
            if max(
                abs(target_pixel[0] - source_pixel[0]),
                abs(target_pixel[1] - source_pixel[1]),
            ) < MINIMUM_WALK_LEG_PIXELS:
                index += 1
                continue
        else:
            source_pixel = projection.world_to_navigation(
                Point(source.x, source.y)
            )
            target_pixel = projection.world_to_navigation(
                Point(target.x, target.y)
            )

        metadata = RenderedRouteLeg(
            action=first_edge.action,
            source=source_pixel,
            target=target_pixel,
            edge_ids=tuple(merged_edge_ids),
            recovery_path=recovery_path,
            combat_checkpoint=combat_checkpoints.get(last_plan_index),
        )
        specs.append(_RouteRenderSpec(first_edge, source, target, metadata))
        index += 1
    return tuple(specs)


def _combine_horizontal_walk_jumps(
    specs: tuple[_RouteRenderSpec, ...],
) -> tuple[_RouteRenderSpec, ...]:
    """Keep a same-direction platform approach and edge jump in one route."""
    combined: list[_RouteRenderSpec] = []
    index = 0
    while index < len(specs):
        walk = specs[index]
        if index + 1 >= len(specs):
            combined.append(walk)
            break

        jump = specs[index + 1]
        walk_direction = _horizontal_direction(walk.source, walk.target)
        jump_direction = _horizontal_direction(jump.source, jump.target)
        same_source_surface = (
            walk.source.surface_id == walk.target.surface_id
            and walk.target.surface_id == jump.source.surface_id
        )
        approaches_jump_source = (
            jump_direction != 0
            and walk_direction == jump_direction
            and jump_direction * (jump.source.x - walk.target.x) >= 0
        )
        can_combine = (
            walk.edge.action is Action.WALK
            and jump.edge.action is Action.JUMP
            and same_source_surface
            and approaches_jump_source
            and walk.metadata.recovery_path == jump.metadata.recovery_path
            and jump.metadata.combat_checkpoint is None
        )
        if not can_combine:
            combined.append(walk)
            index += 1
            continue

        metadata = RenderedRouteLeg(
            action=Action.JUMP,
            source=walk.metadata.source,
            target=jump.metadata.target,
            edge_ids=walk.metadata.edge_ids + jump.metadata.edge_ids,
            recovery_path=walk.metadata.recovery_path,
            combat_checkpoint=walk.metadata.combat_checkpoint,
            combat_checkpoint_position=(
                walk.metadata.target
                if walk.metadata.combat_checkpoint is not None
                else None
            ),
            jump_source=jump.metadata.source,
        )
        combined.append(_RouteRenderSpec(
            edge=jump.edge,
            source=walk.source,
            target=jump.target,
            metadata=metadata,
            approach_target=walk.target,
            jump_source=jump.source,
        ))
        index += 2
    return tuple(combined)


def _route_render_specs(
    graph: NavigationGraph,
    plan: PatrolPlan,
    projection: NavigationProjection,
) -> tuple[_RouteRenderSpec, ...]:
    combat_checkpoints = {
        checkpoint.after_edge_index: checkpoint
        for checkpoint in plan.combat_checkpoints
        if checkpoint.after_edge_index >= 0
    }
    specs = list(_combine_horizontal_walk_jumps(_route_sequence_specs(
        graph,
        plan.edge_ids,
        projection,
        recovery_path=None,
        combat_checkpoints=combat_checkpoints,
    )))
    for recovery_path, edge_ids in enumerate(plan.recovery_edge_paths):
        specs.extend(_combine_horizontal_walk_jumps(_route_sequence_specs(
            graph,
            edge_ids,
            projection,
            recovery_path=recovery_path,
            combat_checkpoints=None,
        )))
    return tuple(specs)


def _reverse_color_map(values: dict[str, Any]) -> dict[str, tuple[int, int, int]]:
    result: dict[str, tuple[int, int, int]] = {}
    for text, command in values.items():
        color = tuple(int(component) for component in str(text).split(","))
        if len(color) == 3:
            result[str(command)] = color
    return result


def _clamped_rectangle(
    image: np.ndarray,
    center: tuple[int, int],
    half_width: int,
    half_height: int,
    color: tuple[int, int, int],
) -> None:
    height, width = image.shape[:2]
    x, y = center
    left = max(0, x - half_width)
    right = min(width - 1, x + half_width)
    top = max(0, y - half_height)
    bottom = min(height - 1, y + half_height)
    if left <= right and top <= bottom:
        cv2.rectangle(image, (left, top), (right, bottom), color, -1)


def _clamped_bounds(
    image: np.ndarray,
    bounds: tuple[int, int, int, int],
    color: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    """Paint an explicit inclusive rectangle and return its clipped bounds."""
    height, width = image.shape[:2]
    left, top, right, bottom = bounds
    clipped = (
        max(0, min(width - 1, int(left))),
        max(0, min(height - 1, int(top))),
        max(0, min(width - 1, int(right))),
        max(0, min(height - 1, int(bottom))),
    )
    left, top, right, bottom = clipped
    if left <= right and top <= bottom:
        cv2.rectangle(image, (left, top), (right, bottom), color, -1)
    return clipped


def _route_background(
    map_rgb: np.ndarray,
    colors: tuple[tuple[int, int, int], ...],
) -> np.ndarray:
    result = map_rgb.copy()
    for color in colors:
        result[np.all(result == color, axis=2)] = (0, 0, 0)
    return result


def render_navigation_assets(
    wz_map: WzMap,
    graph: NavigationGraph,
    plan: PatrolPlan,
    projection: NavigationProjection,
    route_config: dict[str, Any],
    rope_mount_motion: RopeMountMotion | None = None,
    recorded_route_anchors: ProjectedRecordedRouteAnchors | None = None,
    recorded_climb_edges: dict[
        str, NavigationClimbAnchor
    ] | None = None,
) -> tuple[
    np.ndarray,
    tuple[np.ndarray, ...],
    np.ndarray,
    tuple[RenderedRouteLeg, ...],
]:
    """Return map/routes/overlay plus geometry for each rendered route."""
    recorded_route_anchors = recorded_route_anchors or \
        ProjectedRecordedRouteAnchors()
    recorded_climb_edges = recorded_climb_edges or {}
    canonical_bgr, _ = load_wz_canvas(wz_map.canvas_path)
    width, height = projection.size
    map_bgr = cv2.resize(
        canonical_bgr,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    map_rgb = cv2.cvtColor(map_bgr, cv2.COLOR_BGR2RGB)
    normal_colors = _reverse_color_map(route_config.get("color_code", {}))
    vertical_colors = _reverse_color_map(
        route_config.get("color_code_up_down", {})
    )
    required = {
        "left none none",
        "right none none",
        "left none jump",
        "right none jump",
        "none down jump",
        "none none jump",
        "none none goal",
        "none up portal",
        "none up climb",
    }
    missing = sorted(required.difference(normal_colors))
    if missing or "none down none" not in vertical_colors:
        raise ValueError(
            "route colors required by WZ navigation are missing: "
            f"{missing or ['none down none']}"
        )
    all_colors = tuple(normal_colors.values()) + tuple(vertical_colors.values())
    background = _route_background(map_rgb, all_colors)
    ropes = {rope.id: rope for rope in wz_map.ropes}
    routes: list[np.ndarray] = []
    route_legs: list[RenderedRouteLeg] = []
    line_thickness = max(1, int(route_config.get("auto_line_thickness", 3)))
    action_half_width = max(
        1, int(route_config.get("auto_action_half_width", 3))
    )
    action_half_height = max(
        1, int(route_config.get("auto_action_half_height", 2))
    )
    goal_half_width = max(1, int(route_config.get("auto_goal_half_width", 3)))
    goal_half_height = max(1, int(route_config.get("auto_goal_half_height", 2)))
    runup = max(3, int(route_config.get("rope_climb_runup_distance", 8)))
    rope_tip = tuple(
        int(value)
        for value in route_config.get(
            "rope_climb_target_color", (0, 191, 255)
        )
    )

    for spec in _route_render_specs(graph, plan, projection):
        edge = spec.edge
        source = spec.source
        target = spec.target
        source_pixel = spec.metadata.source
        target_pixel = spec.metadata.target
        metadata = spec.metadata
        route = background.copy()
        if spec.approach_target is not None and \
                spec.jump_source is not None:
            jump_source_pixel = projection.world_to_navigation(
                Point(spec.jump_source.x, spec.jump_source.y)
            )
            command = (
                "right none jump"
                if target.x > spec.jump_source.x
                else "left none jump"
            )
            walk_command = (
                "right none none"
                if spec.jump_source.x > source.x
                else "left none none"
            )
            trigger_bounds = _directional_jump_trigger_bounds(
                jump_source_pixel,
                target_pixel,
                action_half_width,
                action_half_height,
                (
                    None
                    if rope_mount_motion is None
                    else rope_mount_motion.jump_distance_px
                ),
            )
            recorded_trigger = _recorded_jump_trigger(
                recorded_route_anchors,
                metadata,
                jump_source_pixel,
                target_pixel,
                trigger_bounds,
                action_half_width,
                action_half_height,
            )
            recorded_x = None
            if recorded_trigger is not None:
                jump_source_pixel, trigger_bounds = recorded_trigger
                recorded_x = jump_source_pixel[0]
            cv2.line(
                route,
                source_pixel,
                jump_source_pixel,
                normal_colors[walk_command],
                line_thickness,
            )
            trigger_bounds = _clamped_bounds(
                route,
                trigger_bounds,
                normal_colors[command],
            )
            metadata = replace(
                metadata,
                jump_source=jump_source_pixel,
                jump_trigger_bounds=trigger_bounds,
                recorded_x_anchor=recorded_x,
            )
        elif edge.action is Action.WALK:
            command = (
                "right none none" if target.x > source.x else "left none none"
            )
            cv2.line(
                route,
                source_pixel,
                target_pixel,
                normal_colors[command],
                line_thickness,
            )
        elif edge.action is Action.JUMP:
            recorded_x = None
            if abs(target.x - source.x) <= 1.0:
                command = "none none jump"
                trigger_pixel = source_pixel
                trigger_bounds = (
                    source_pixel[0] - action_half_width,
                    source_pixel[1] - action_half_height,
                    source_pixel[0] + action_half_width,
                    source_pixel[1] + action_half_height,
                )
            elif target.x > source.x:
                command = "right none jump"
                trigger_bounds = _directional_jump_trigger_bounds(
                    source_pixel,
                    target_pixel,
                    action_half_width,
                    action_half_height,
                    (
                        None
                        if rope_mount_motion is None
                        else rope_mount_motion.jump_distance_px
                    ),
                )
            else:
                command = "left none jump"
                trigger_bounds = _directional_jump_trigger_bounds(
                    source_pixel,
                    target_pixel,
                    action_half_width,
                    action_half_height,
                    (
                        None
                        if rope_mount_motion is None
                        else rope_mount_motion.jump_distance_px
                    ),
                )
            if abs(target.x - source.x) > 1.0:
                recorded_trigger = _recorded_jump_trigger(
                    recorded_route_anchors,
                    metadata,
                    source_pixel,
                    target_pixel,
                    trigger_bounds,
                    action_half_width,
                    action_half_height,
                )
                if recorded_trigger is not None:
                    source_pixel, trigger_bounds = recorded_trigger
                    recorded_x = source_pixel[0]
            trigger_bounds = _clamped_bounds(
                route,
                trigger_bounds,
                normal_colors[command],
            )
            metadata = replace(
                metadata,
                jump_source=source_pixel,
                jump_trigger_bounds=trigger_bounds,
                recorded_x_anchor=recorded_x,
            )
        elif edge.action is Action.DROP:
            _clamped_rectangle(
                route,
                source_pixel,
                action_half_width,
                action_half_height,
                normal_colors["none down jump"],
            )
        elif edge.action is Action.PORTAL:
            _clamped_rectangle(
                route,
                source_pixel,
                max(action_half_width, 5),
                max(action_half_height, 2),
                normal_colors["none up portal"],
            )
        elif edge.action is Action.CLIMB:
            if target.y < source.y:
                rope = ropes.get(edge.detail_id or "")
                contact_y = source.y if rope is None else rope.y2
                contact = projection.world_to_navigation(
                    Point(source.x, contact_y)
                )
                recorded_climb = (
                    recorded_climb_edges.get(edge.id)
                    if metadata.recovery_path is None
                    else None
                )
                recorded_launch_offset = None
                recorded_approach = None
                if recorded_climb is not None:
                    recorded_launch_offset = abs(
                        recorded_climb.launch_x
                        - contact[0]
                    )
                    recorded_approach = recorded_climb.approach_name
                    metadata = replace(
                        metadata,
                        recorded_x_anchor=recorded_climb.launch_x,
                    )
                rope_mount = None
                mount_runup = runup
                if rope_mount_motion is not None:
                    rope_mount = _build_rope_mount_plan(
                        source_pixel,
                        contact,
                        rope_mount_motion,
                        recorded_launch_offset_px=recorded_launch_offset,
                        approach_direction=recorded_approach,
                    )
                    mount_runup = rope_mount.staging_offset_px
                    metadata = replace(metadata, rope_mount=rope_mount)
                left_room = contact[0]
                right_room = width - 1 - contact[0]
                if recorded_approach == "right":
                    side = -1
                elif recorded_approach == "left":
                    side = 1
                else:
                    side = (
                        -1
                        if left_room >= mount_runup
                        or left_room >= right_room
                        else 1
                    )
                approach = (
                    max(
                        0,
                        min(
                            width - 1,
                            contact[0] + side * mount_runup,
                        ),
                    ),
                    source_pixel[1],
                )
                cv2.line(
                    route,
                    approach,
                    contact,
                    normal_colors["none up climb"],
                    max(1, line_thickness - 1),
                )
                if 0 <= contact[0] < width and 0 <= contact[1] < height:
                    route[contact[1], contact[0]] = rope_tip
            else:
                cv2.line(
                    route,
                    source_pixel,
                    target_pixel,
                    vertical_colors["none down none"],
                    max(1, line_thickness - 1),
                )

        _clamped_rectangle(
            route,
            target_pixel,
            goal_half_width,
            goal_half_height,
            normal_colors["none none goal"],
        )
        routes.append(route)
        route_legs.append(metadata)

    overlay = map_bgr.copy()
    for surface in wz_map.surfaces:
        points = np.array(
            [
                projection.world_to_navigation(point)
                for point in surface.points
            ],
            dtype=np.int32,
        )
        if len(points) >= 2:
            cv2.polylines(overlay, [points], False, (40, 230, 40), 1)
    for rope in wz_map.ropes:
        cv2.line(
            overlay,
            projection.world_to_navigation(Point(rope.x, rope.y1)),
            projection.world_to_navigation(Point(rope.x, rope.y2)),
            (255, 190, 0),
            1,
        )
    for portal in wz_map.portals:
        cv2.circle(
            overlay,
            projection.world_to_navigation(Point(portal.x, portal.y)),
            3,
            (255, 0, 255),
            -1,
        )
    for spawn in wz_map.monster_spawns:
        cv2.circle(
            overlay,
            projection.world_to_navigation(Point(spawn.x, spawn.y)),
            2,
            (0, 0, 255),
            -1,
        )
    for node_id in graph.safe_firing_targets:
        node = graph.node_by_id[node_id]
        cv2.circle(
            overlay,
            projection.world_to_navigation(Point(node.x, node.y)),
            4,
            (0, 255, 255),
            1,
        )
    cv2.putText(
        overlay,
        f"WZ {wz_map.map_id}  P:{len(wz_map.surfaces)} "
        f"R:{len(wz_map.ropes)} T:{len(wz_map.portals)}",
        (4, max(12, min(height - 3, 14))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.33,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return map_bgr, tuple(routes), overlay, tuple(route_legs)
