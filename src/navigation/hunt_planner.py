"""Build a conservative action graph and a monster-spawn coverage tour."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Any

from .wz_geometry import EPSILON, Point, Surface, WzMap


class Action(StrEnum):
    WALK = "WALK"
    JUMP = "JUMP"
    DROP = "DROP"
    CLIMB = "CLIMB"
    PORTAL = "PORTAL"


@dataclass(frozen=True, slots=True)
class MotionProfile:
    walk_speed_wz_per_sec: float
    climb_speed_wz_per_sec: float
    gravity_wz_per_sec2: float
    fall_speed_wz_per_sec: float
    jump_speed_wz_per_sec: float
    air_speed_wz_per_sec: float
    jump_height_wz: float
    jump_distance_wz: float
    character_half_width_wz: float
    source: str

    @classmethod
    def from_wz(
        cls,
        wz_map: WzMap,
        config: dict[str, Any],
        *,
        world_to_navigation_scale: tuple[float, float],
        observed_walk_speed_px_per_sec: float | None = None,
        observed_jump_height_px: float | None = None,
        observed_jump_distance_px: float | None = None,
    ) -> "MotionProfile":
        physics = wz_map.physics
        walk_percent = float(config.get("move_speed_percent", 100.0)) / 100.0
        jump_percent = float(config.get("jump_percent", 100.0)) / 100.0
        walk_speed = float(physics.get("walkSpeed", 125.0)) * walk_percent
        jump_speed = abs(float(physics.get("jumpSpeed", 555.0))) * jump_percent
        gravity = float(physics.get("gravityAcc", 2000.0))
        fall_speed = float(physics.get("fallSpeed", 670.0))
        air_speed = float(
            config.get("air_speed_wz_per_sec", walk_speed)
        )
        climb_speed = float(config.get("climb_speed_wz_per_sec", 100.0))
        safety = min(1.0, max(0.1, float(config.get("jump_safety_factor", 0.82))))
        jump_height = jump_speed * jump_speed / (2.0 * gravity) * safety
        jump_distance = air_speed * (2.0 * jump_speed / gravity) * safety
        source_parts = ["wz-physics"]
        scale_x, scale_y = world_to_navigation_scale
        configured_walk = float(config.get("walk_speed_px_per_sec", 0.0) or 0.0)
        configured_height = float(config.get("jump_height_px", 0.0) or 0.0)
        configured_distance = float(config.get("jump_distance_px", 0.0) or 0.0)
        if observed_walk_speed_px_per_sec is not None:
            configured_walk = observed_walk_speed_px_per_sec
            source_parts.append("observed-walk")
        if observed_jump_height_px is not None:
            configured_height = observed_jump_height_px
            source_parts.append("observed-jump-height")
        if observed_jump_distance_px is not None:
            configured_distance = observed_jump_distance_px
            source_parts.append("observed-jump-distance")
        if configured_walk > 0 and scale_x > 0:
            walk_speed = configured_walk / scale_x
            source_parts.append("pixel-walk-override")
        if configured_height > 0 and scale_y > 0:
            jump_height = configured_height / scale_y * safety
            source_parts.append("pixel-jump-height-override")
        if configured_distance > 0 and scale_x > 0:
            jump_distance = configured_distance / scale_x * safety
            source_parts.append("pixel-jump-distance-override")
        values = (
            walk_speed,
            climb_speed,
            gravity,
            fall_speed,
            jump_speed,
            air_speed,
            jump_height,
            jump_distance,
        )
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("Hero motion profile values must be finite and positive")
        return cls(
            walk_speed_wz_per_sec=walk_speed,
            climb_speed_wz_per_sec=climb_speed,
            gravity_wz_per_sec2=gravity,
            fall_speed_wz_per_sec=fall_speed,
            jump_speed_wz_per_sec=jump_speed,
            air_speed_wz_per_sec=air_speed,
            jump_height_wz=jump_height,
            jump_distance_wz=jump_distance,
            character_half_width_wz=float(
                config.get("character_half_width_wz", 15.0)
            ),
            source="+".join(source_parts),
        )

    def flight_time(self, delta_y: float) -> float | None:
        discriminant = (
            self.jump_speed_wz_per_sec ** 2
            + 2.0 * self.gravity_wz_per_sec2 * delta_y
        )
        if discriminant < 0:
            return None
        return (
            self.jump_speed_wz_per_sec + math.sqrt(discriminant)
        ) / self.gravity_wz_per_sec2

    def can_jump(self, delta_x: float, delta_y: float) -> bool:
        if delta_y < -self.jump_height_wz - EPSILON:
            return False
        flight_time = self.flight_time(delta_y)
        if flight_time is None or flight_time <= 0:
            return False
        predicted_distance = self.air_speed_wz_per_sec * flight_time
        maximum_distance = min(self.jump_distance_wz, predicted_distance)
        return abs(delta_x) <= maximum_distance + EPSILON


@dataclass(frozen=True, slots=True)
class RangedAttackProfile:
    horizontal_range_wz: float
    vertical_tolerance_wz: float
    projectile_height_wz: float
    projectile_clearance_wz: float
    origin_margin_wz: float


@dataclass(frozen=True, slots=True)
class Node:
    id: str
    surface_id: str
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class Edge:
    id: str
    action: Action
    source: str
    target: str
    expected_time_sec: float
    detail_id: str | None = None


@dataclass(frozen=True, slots=True)
class NavigationGraph:
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    coverage_targets: tuple[str, ...]
    monster_surface_ids: tuple[str, ...] = ()
    safe_firing_targets: tuple[str, ...] = ()
    safe_covered_monster_surface_ids: tuple[str, ...] = ()

    @property
    def node_by_id(self) -> dict[str, Node]:
        return {node.id: node for node in self.nodes}

    @property
    def edge_by_id(self) -> dict[str, Edge]:
        return {edge.id: edge for edge in self.edges}


@dataclass(frozen=True, slots=True)
class CombatCheckpoint:
    """A route endpoint that must be cleared before patrol may continue."""

    node_id: str
    facing: str
    label: str
    after_edge_index: int = -1


@dataclass(frozen=True, slots=True)
class PatrolPlan:
    edge_ids: tuple[str, ...]
    visited_targets: tuple[str, ...]
    unreachable_targets: tuple[str, ...]
    recovery_edge_paths: tuple[tuple[str, ...], ...] = ()
    combat_checkpoints: tuple[CombatCheckpoint, ...] = ()


def _number(value: float) -> str:
    return format(value, ".9g")


def _node_id(surface_id: str, x: float) -> str:
    return f"node:{surface_id}:{_number(x)}"


def _surface_point(surface: Surface, x: float) -> Point:
    return Point(x, surface.y_at(x))


def _clamp_to_surface(surface: Surface, x: float, margin: float = 0.0) -> float:
    low = surface.min_x + margin
    high = surface.max_x - margin
    if low > high:
        return (surface.min_x + surface.max_x) / 2.0
    return min(max(float(x), low), high)


def _first_surface_below(
    surfaces: tuple[Surface, ...], source: Surface, x: float
) -> Surface | None:
    try:
        source_y = source.y_at(x)
    except ValueError:
        return None
    candidates: list[tuple[float, str, Surface]] = []
    for candidate in surfaces:
        if candidate.id == source.id or not candidate.contains_x(x):
            continue
        try:
            target_y = candidate.y_at(x)
        except ValueError:
            continue
        if target_y > source_y + EPSILON:
            candidates.append((target_y, candidate.id, candidate))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def _near_endpoint_surface(
    surfaces: tuple[Surface, ...],
    x: float,
    y: float,
    tolerance: float,
    *,
    exclude: str | None = None,
) -> Surface | None:
    candidates: list[tuple[float, str, Surface]] = []
    for surface in surfaces:
        if surface.id == exclude or not surface.supports_default_motion or \
                not surface.contains_x(x):
            continue
        try:
            distance = abs(surface.y_at(x) - y)
        except ValueError:
            continue
        if distance <= tolerance:
            candidates.append((distance, surface.id, surface))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def _jump_candidate(
    source: Surface,
    target: Surface,
    profile: MotionProfile,
    launch_margin: float,
    landing_margin: float,
) -> tuple[float, float] | None:
    overlap_left = max(source.min_x, target.min_x)
    overlap_right = min(source.max_x, target.max_x)
    if overlap_right - overlap_left >= profile.character_half_width_wz * 2.0:
        x = (overlap_left + overlap_right) / 2.0
        source_y = source.y_at(x)
        target_y = target.y_at(x)
        if target_y < source_y and profile.can_jump(0.0, target_y - source_y):
            return x, x

    if target.min_x >= source.max_x:
        source_x = _clamp_to_surface(
            source, source.max_x, launch_margin
        )
        target_x = _clamp_to_surface(
            target, target.min_x, landing_margin
        )
    elif target.max_x <= source.min_x:
        source_x = _clamp_to_surface(
            source, source.min_x, launch_margin
        )
        target_x = _clamp_to_surface(
            target, target.max_x, landing_margin
        )
    else:
        return None
    source_y = source.y_at(source_x)
    target_y = target.y_at(target_x)
    if not profile.can_jump(target_x - source_x, target_y - source_y):
        return None
    return source_x, target_x


def _jump_hits_wall(
    wz_map: WzMap,
    source: Point,
    target: Point,
    profile: MotionProfile,
    clearance: float,
) -> bool:
    """Conservatively reject a ballistic arc crossing a vertical foothold."""
    delta_x = target.x - source.x
    if abs(delta_x) <= EPSILON:
        return False
    flight_time = profile.flight_time(target.y - source.y)
    if flight_time is None:
        return True
    low_x, high_x = sorted((source.x, target.x))
    for wall in wz_map.walls:
        if not low_x + EPSILON < wall.x < high_x - EPSILON:
            continue
        fraction = (wall.x - source.x) / delta_x
        elapsed = flight_time * fraction
        arc_y = (
            source.y
            - profile.jump_speed_wz_per_sec * elapsed
            + 0.5 * profile.gravity_wz_per_sec2 * elapsed * elapsed
        )
        if wall.y1 - clearance <= arc_y <= wall.y2 + clearance:
            return True
    return False


def _walk_time(distance: float, speed: float) -> float:
    return max(0.05, abs(distance) / speed)


def _fall_time(distance: float, gravity: float, fall_speed: float) -> float:
    distance = max(0.0, distance)
    time_to_cap = fall_speed / gravity
    distance_to_cap = 0.5 * gravity * time_to_cap * time_to_cap
    if distance <= distance_to_cap:
        return math.sqrt(2.0 * distance / gravity)
    return time_to_cap + (distance - distance_to_cap) / fall_speed


def _safe_firing_coverage(
    wz_map: WzMap,
    surfaces: dict[str, Surface],
    monster_surface_ids: set[str],
    edge_margin: float,
    ranged_attack: RangedAttackProfile,
) -> tuple[dict[str, set[float]], set[str]]:
    """Find spawn-free platform edges with a clear horizontal firing lane."""
    coverage: dict[str, set[float]] = {}
    safely_covered: set[str] = set()
    safe_surfaces = tuple(
        surface for surface in surfaces.values()
        if surface.id not in monster_surface_ids
    )
    for monster_surface_id in sorted(monster_surface_ids):
        monster_surface = surfaces[monster_surface_id]
        candidates_by_direction: dict[int, list[tuple[float, str, float]]] = {
            -1: [],
            1: [],
        }
        for safe_surface in safe_surfaces:
            if safe_surface.max_x < monster_surface.min_x - EPSILON:
                firing_x = _clamp_to_surface(
                    safe_surface, safe_surface.max_x, edge_margin
                )
                target_x = monster_surface.min_x
                direction = 1
            elif safe_surface.min_x > monster_surface.max_x + EPSILON:
                firing_x = _clamp_to_surface(
                    safe_surface, safe_surface.min_x, edge_margin
                )
                target_x = monster_surface.max_x
                direction = -1
            else:
                continue

            firing_y = safe_surface.y_at(firing_x)
            target_y = monster_surface.y_at(target_x)
            distance = abs(target_x - firing_x)
            if distance > ranged_attack.horizontal_range_wz + EPSILON or \
                    abs(target_y - firing_y) > \
                    ranged_attack.vertical_tolerance_wz + EPSILON:
                continue
            blocker = wz_map.first_horizontal_projectile_blocker(
                Point(
                    firing_x,
                    firing_y - ranged_attack.projectile_height_wz,
                ),
                target_x,
                clearance=ranged_attack.projectile_clearance_wz,
                origin_margin=ranged_attack.origin_margin_wz,
            )
            if blocker is not None:
                continue
            candidates_by_direction[direction].append(
                (distance, safe_surface.id, firing_x)
            )

        selected = []
        for direction in (-1, 1):
            candidates = candidates_by_direction[direction]
            if candidates:
                selected.append(min(candidates))
        if not selected:
            continue
        safely_covered.add(monster_surface_id)
        for _, safe_surface_id, firing_x in selected:
            coverage.setdefault(safe_surface_id, set()).add(firing_x)
    return coverage, safely_covered


def build_navigation_graph(
    wz_map: WzMap,
    profile: MotionProfile,
    config: dict[str, Any],
    *,
    ranged_attack: RangedAttackProfile | None = None,
) -> NavigationGraph:
    """Build WALK/JUMP/DROP/CLIMB/PORTAL edges from WZ geometry."""
    surfaces = {
        surface.id: surface
        for surface in wz_map.surfaces
        if surface.supports_default_motion
    }
    surface_values = tuple(surfaces.values())
    if not surface_values:
        raise ValueError("WZ map has no default-motion surfaces")
    edge_margin = max(
        profile.character_half_width_wz,
        float(config.get("platform_edge_margin_wz", 18.0)),
    )
    # Hero may run almost to the foothold edge before jumping, while the
    # landing still needs enough room for the character body. Applying the
    # patrol edge margin to both ends incorrectly turns Forest Floor's
    # 34-WZ gaps into 70-WZ jumps and disconnects every adjacent platform.
    jump_launch_margin = max(
        0.0, float(config.get("jump_launch_margin_wz", 2.0))
    )
    jump_landing_margin = max(
        profile.character_half_width_wz,
        float(config.get(
            "jump_landing_margin_wz",
            profile.character_half_width_wz,
        )),
    )
    rope_tolerance = float(config.get("rope_attach_tolerance_wz", 90.0))
    spawn_snap = float(config.get("spawn_snap_distance_wz", 140.0))
    portal_snap = float(config.get("portal_snap_distance_wz", 160.0))
    minimum_drop_overlap = float(config.get("minimum_drop_overlap_wz", 30.0))
    jump_wall_clearance = max(
        0.0, float(config.get("jump_wall_clearance_wz", 8.0))
    )
    anchors: dict[str, set[float]] = {surface.id: set() for surface in surface_values}
    coverage: dict[str, set[float]] = {}

    for surface in surface_values:
        anchors[surface.id].update(
            {
                _clamp_to_surface(surface, surface.min_x, edge_margin),
                _clamp_to_surface(surface, surface.max_x, edge_margin),
            }
        )

    spawn_x_by_surface: dict[str, list[float]] = {}
    for spawn in wz_map.monster_spawns:
        snapped = wz_map.nearest_surface(
            Point(spawn.x, spawn.y),
            maximum_vertical_distance=spawn_snap,
        )
        if snapped is None or snapped[0].id not in surfaces:
            continue
        spawn_x_by_surface.setdefault(snapped[0].id, []).append(spawn.x)
    safe_firing_keys: set[tuple[str, float]] = set()
    safely_covered: set[str] = set()
    if ranged_attack is not None:
        safe_coverage, safely_covered = _safe_firing_coverage(
            wz_map,
            surfaces,
            set(spawn_x_by_surface),
            edge_margin,
            ranged_attack,
        )
        for surface_id, points in safe_coverage.items():
            anchors[surface_id].update(points)
            coverage.setdefault(surface_id, set()).update(points)
            safe_firing_keys.update((surface_id, x) for x in points)

    sweep_padding = float(config.get("spawn_sweep_padding_wz", 55.0))
    for surface_id, spawn_values in spawn_x_by_surface.items():
        if surface_id in safely_covered:
            continue
        surface = surfaces[surface_id]
        left = _clamp_to_surface(
            surface, min(spawn_values) - sweep_padding, edge_margin
        )
        right = _clamp_to_surface(
            surface, max(spawn_values) + sweep_padding, edge_margin
        )
        points = {left, right}
        anchors[surface_id].update(points)
        coverage[surface_id] = points

    climb_pairs: list[tuple[str, str, float, str]] = []
    for rope in wz_map.ropes:
        top = _near_endpoint_surface(
            surface_values, rope.x, rope.y1, rope_tolerance
        )
        bottom = _near_endpoint_surface(
            surface_values,
            rope.x,
            rope.y2,
            rope_tolerance,
            exclude=None if top is None else top.id,
        )
        if top is None or bottom is None or top.id == bottom.id:
            continue
        if top.y_at(rope.x) > bottom.y_at(rope.x):
            top, bottom = bottom, top
        anchors[top.id].add(rope.x)
        anchors[bottom.id].add(rope.x)
        climb_pairs.append((bottom.id, top.id, rope.x, rope.id))

    portal_pairs: list[tuple[str, str, float, float, str]] = []
    portal_by_name = {portal.name: portal for portal in wz_map.portals}
    for portal in wz_map.portals:
        if portal.target_map_id != wz_map.map_id or not portal.target_name:
            continue
        target = portal_by_name.get(portal.target_name)
        if target is None:
            continue
        source_snap = wz_map.nearest_surface(
            Point(portal.x, portal.y),
            maximum_vertical_distance=portal_snap,
        )
        target_snap = wz_map.nearest_surface(
            Point(target.x, target.y),
            maximum_vertical_distance=portal_snap,
        )
        if source_snap is None or target_snap is None:
            continue
        source_surface, source_point = source_snap
        target_surface, target_point = target_snap
        if source_surface.id not in surfaces or target_surface.id not in surfaces:
            continue
        source_x = _clamp_to_surface(source_surface, source_point.x)
        target_x = _clamp_to_surface(target_surface, target_point.x)
        anchors[source_surface.id].add(source_x)
        anchors[target_surface.id].add(target_x)
        portal_pairs.append(
            (
                source_surface.id,
                target_surface.id,
                source_x,
                target_x,
                portal.id,
            )
        )

    drop_pairs: list[tuple[str, str, float]] = []
    for source in surface_values:
        if source.cant_through or source.forbid_fall_down:
            continue
        for target in surface_values:
            if target.id == source.id:
                continue
            overlap_left = max(source.min_x, target.min_x)
            overlap_right = min(source.max_x, target.max_x)
            if overlap_right - overlap_left < minimum_drop_overlap:
                continue
            x = (overlap_left + overlap_right) / 2.0
            if _first_surface_below(surface_values, source, x) != target:
                continue
            anchors[source.id].add(x)
            anchors[target.id].add(x)
            drop_pairs.append((source.id, target.id, x))

    jump_pairs: list[tuple[str, str, float, float]] = []
    for source in surface_values:
        for target in surface_values:
            if source.id == target.id:
                continue
            candidate = _jump_candidate(
                source,
                target,
                profile,
                jump_launch_margin,
                jump_landing_margin,
            )
            if candidate is None:
                continue
            source_x, target_x = candidate
            if _jump_hits_wall(
                wz_map,
                _surface_point(source, source_x),
                _surface_point(target, target_x),
                profile,
                jump_wall_clearance,
            ):
                continue
            anchors[source.id].add(source_x)
            anchors[target.id].add(target_x)
            jump_pairs.append((source.id, target.id, source_x, target_x))

    nodes: list[Node] = []
    node_lookup: dict[tuple[str, float], Node] = {}
    for surface_id in sorted(anchors):
        surface = surfaces[surface_id]
        for x in sorted(anchors[surface_id]):
            try:
                y = surface.y_at(x)
            except ValueError:
                continue
            node = Node(_node_id(surface_id, x), surface_id, x, y)
            nodes.append(node)
            node_lookup[(surface_id, x)] = node

    edges: list[Edge] = []
    for surface_id in sorted(anchors):
        surface_nodes = sorted(
            (node for node in nodes if node.surface_id == surface_id),
            key=lambda item: item.x,
        )
        for left, right in pairwise(surface_nodes):
            duration = _walk_time(
                right.x - left.x,
                profile.walk_speed_wz_per_sec,
            )
            for source, target in ((left, right), (right, left)):
                edges.append(
                    Edge(
                        id=f"WALK:{source.id}->{target.id}",
                        action=Action.WALK,
                        source=source.id,
                        target=target.id,
                        expected_time_sec=duration,
                    )
                )

    for source_id, target_id, x in sorted(set(drop_pairs)):
        source = node_lookup[(source_id, x)]
        target = node_lookup[(target_id, x)]
        edges.append(
            Edge(
                id=f"DROP:{source.id}->{target.id}",
                action=Action.DROP,
                source=source.id,
                target=target.id,
                expected_time_sec=(
                    _fall_time(
                        target.y - source.y,
                        profile.gravity_wz_per_sec2,
                        profile.fall_speed_wz_per_sec,
                    )
                    + 0.15
                ),
            )
        )

    for bottom_id, top_id, x, rope_id in sorted(set(climb_pairs)):
        bottom = node_lookup[(bottom_id, x)]
        top = node_lookup[(top_id, x)]
        duration = abs(bottom.y - top.y) / profile.climb_speed_wz_per_sec + 0.35
        for source, target in ((bottom, top), (top, bottom)):
            edges.append(
                Edge(
                    id=f"CLIMB:{rope_id}:{source.id}->{target.id}",
                    action=Action.CLIMB,
                    source=source.id,
                    target=target.id,
                    expected_time_sec=duration,
                    detail_id=rope_id,
                )
            )

    for source_id, target_id, source_x, target_x, portal_id in sorted(
        set(portal_pairs)
    ):
        source = node_lookup[(source_id, source_x)]
        target = node_lookup[(target_id, target_x)]
        edges.append(
            Edge(
                id=f"PORTAL:{portal_id}:{source.id}->{target.id}",
                action=Action.PORTAL,
                source=source.id,
                target=target.id,
                expected_time_sec=1.0,
                detail_id=portal_id,
            )
        )

    for source_id, target_id, source_x, target_x in sorted(set(jump_pairs)):
        source = node_lookup[(source_id, source_x)]
        target = node_lookup[(target_id, target_x)]
        flight_time = profile.flight_time(target.y - source.y)
        if flight_time is None:
            continue
        edges.append(
            Edge(
                id=f"JUMP:{source.id}->{target.id}",
                action=Action.JUMP,
                source=source.id,
                target=target.id,
                expected_time_sec=flight_time + 0.12,
            )
        )

    target_ids = tuple(
        sorted(
            {
                node_lookup[(surface_id, x)].id
                for surface_id, values in coverage.items()
                for x in values
                if (surface_id, x) in node_lookup
            }
        )
    )
    safe_firing_target_ids = tuple(sorted(
        node_lookup[(surface_id, x)].id
        for surface_id, x in safe_firing_keys
        if (surface_id, x) in node_lookup
    ))
    unique_edges = {edge.id: edge for edge in edges}
    return NavigationGraph(
        nodes=tuple(sorted(nodes, key=lambda item: item.id)),
        edges=tuple(unique_edges[key] for key in sorted(unique_edges)),
        coverage_targets=target_ids,
        monster_surface_ids=tuple(sorted(spawn_x_by_surface)),
        safe_firing_targets=safe_firing_target_ids,
        safe_covered_monster_surface_ids=tuple(sorted(safely_covered)),
    )


def shortest_path(
    graph: NavigationGraph,
    source: str,
    target: str,
    *,
    excluded_actions: frozenset[Action] = frozenset(),
) -> tuple[float, tuple[str, ...]] | None:
    result = _shortest_path_to_any(
        graph,
        source,
        frozenset({target}),
        excluded_actions=excluded_actions,
    )
    if result is None:
        return None
    return result[0], result[2]


def shortest_path_to_any(
    graph: NavigationGraph,
    source: str,
    targets: frozenset[str],
    *,
    excluded_actions: frozenset[Action] = frozenset(),
) -> tuple[float, str, tuple[str, ...]] | None:
    """Return the deterministic shortest path to any target node."""
    return _shortest_path_to_any(
        graph,
        source,
        targets,
        excluded_actions=excluded_actions,
    )


_ACTION_PENALTY_SECONDS = {
    Action.WALK: 0.0,
    Action.JUMP: 0.35,
    Action.DROP: 0.25,
    Action.CLIMB: 0.20,
    Action.PORTAL: 0.15,
}


def _edge_cost(edge: Edge) -> float:
    return edge.expected_time_sec + _ACTION_PENALTY_SECONDS[edge.action]


def _shortest_path_to_any(
    graph: NavigationGraph,
    source: str,
    targets: frozenset[str],
    *,
    excluded_actions: frozenset[Action] = frozenset(),
) -> tuple[float, str, tuple[str, ...]] | None:
    """Find one deterministic shortest path to any allowed target."""
    if not targets:
        return None
    if source in targets:
        return 0.0, source, ()
    outgoing: dict[str, list[Edge]] = {}
    for edge in graph.edges:
        if edge.action in excluded_actions:
            continue
        outgoing.setdefault(edge.source, []).append(edge)
    for values in outgoing.values():
        values.sort(key=lambda item: item.id)
    distances = {source: 0.0}
    signatures: dict[str, tuple[str, ...]] = {source: ()}
    queue: list[tuple[float, tuple[str, ...], str]] = [(0.0, (), source)]
    while queue:
        cost, signature, node_id = heapq.heappop(queue)
        if cost != distances.get(node_id) or signature != signatures.get(node_id):
            continue
        if node_id in targets:
            return cost, node_id, signature
        for edge in outgoing.get(node_id, ()):
            candidate_cost = cost + _edge_cost(edge)
            candidate_signature = (*signature, edge.id)
            current_cost = distances.get(edge.target)
            current_signature = signatures.get(edge.target)
            better = current_cost is None or candidate_cost < current_cost - 1e-9
            stable_tie = current_cost is not None and \
                abs(candidate_cost - current_cost) <= 1e-9 and \
                (current_signature is None or candidate_signature < current_signature)
            if better or stable_tie:
                distances[edge.target] = candidate_cost
                signatures[edge.target] = candidate_signature
                heapq.heappush(
                    queue,
                    (candidate_cost, candidate_signature, edge.target),
                )
    return None


def _local_walk_loop(
    graph: NavigationGraph, node_id: str
) -> tuple[str, ...]:
    """Create a two-way sweep when one target is its own graph component."""
    nodes = graph.node_by_id
    source = nodes[node_id]
    candidates = sorted(
        (
            edge for edge in graph.edges
            if edge.source == node_id
            and edge.action is Action.WALK
            and nodes[edge.target].surface_id == source.surface_id
        ),
        key=lambda edge: (
            -abs(nodes[edge.target].x - source.x),
            edge.id,
        ),
    )
    for outward in candidates:
        reverse = next(
            (
                edge for edge in graph.edges
                if edge.action is Action.WALK
                and edge.source == outward.target
                and edge.target == outward.source
            ),
            None,
        )
        if reverse is not None:
            return outward.id, reverse.id
    return ()


def _monster_platform_recovery_paths(
    graph: NavigationGraph,
) -> tuple[tuple[str, ...], ...]:
    """Sweep each safely-covered monster platform once, then exit to safety."""
    if not graph.safe_firing_targets:
        return ()
    nodes = graph.node_by_id
    walk_edges = {
        (edge.source, edge.target): edge.id
        for edge in graph.edges
        if edge.action is Action.WALK
    }
    recovery_paths: list[tuple[str, ...]] = []
    for surface_id in graph.safe_covered_monster_surface_ids:
        surface_nodes = sorted(
            (node for node in graph.nodes if node.surface_id == surface_id),
            key=lambda node: (node.x, node.id),
        )
        if len(surface_nodes) < 2:
            continue
        left = surface_nodes[0]
        right = surface_nodes[-1]
        endpoint_candidates = []
        for endpoint in (left, right):
            for target_id in graph.safe_firing_targets:
                result = shortest_path(graph, endpoint.id, target_id)
                if result is not None:
                    endpoint_candidates.append(
                        (result[0], endpoint.x, endpoint.id, result[1])
                    )
        if not endpoint_candidates:
            continue
        _, _, endpoint_id, exit_path = min(endpoint_candidates)

        ordered_nodes = (
            surface_nodes
            if endpoint_id == right.id
            else list(reversed(surface_nodes))
        )
        sweep = []
        for source, target in pairwise(ordered_nodes):
            edge_id = walk_edges.get((source.id, target.id))
            if edge_id is None:
                sweep = []
                break
            sweep.append(edge_id)
        if sweep:
            recovery_paths.append(tuple((*sweep, *exit_path)))
    return tuple(recovery_paths)


def _all_platform_recovery_paths(
    graph: NavigationGraph,
    patrol_edge_ids: tuple[str, ...],
    *,
    excluded_surface_ids: frozenset[str] = frozenset(),
) -> tuple[tuple[str, ...], ...]:
    """Converge both ends of every platform into the normal patrol.

    Recovery paths deliberately reject portal edges: after relogin Hero must
    leave the platform using visible geometry, not an invisible or scripted
    portal.  Each side gets its own inward path so spawning near either edge
    never commands Hero farther into the wall.
    """
    edge_by_id = graph.edge_by_id
    node_by_id = graph.node_by_id
    patrol_node_ids = frozenset(
        node_id
        for edge_id in patrol_edge_ids
        for node_id in (
            edge_by_id[edge_id].source,
            edge_by_id[edge_id].target,
        )
    )
    if not patrol_node_ids:
        return ()

    walk_edges = {
        (edge.source, edge.target): edge
        for edge in graph.edges
        if edge.action is Action.WALK
    }
    cross_edges: dict[str, list[Edge]] = {}
    for edge in graph.edges:
        if edge.action in (Action.WALK, Action.PORTAL):
            continue
        source = node_by_id[edge.source]
        target = node_by_id[edge.target]
        if source.surface_id != target.surface_id:
            cross_edges.setdefault(edge.source, []).append(edge)
    for edges in cross_edges.values():
        edges.sort(key=lambda edge: edge.id)

    nodes_by_surface: dict[str, list[Node]] = {}
    for node in graph.nodes:
        nodes_by_surface.setdefault(node.surface_id, []).append(node)

    def walk_path(
        surface_nodes: list[Node],
        source_index: int,
        target_index: int,
    ) -> tuple[str, ...] | None:
        if source_index == target_index:
            return ()
        step = 1 if target_index > source_index else -1
        path: list[str] = []
        for index in range(source_index, target_index, step):
            edge = walk_edges.get((
                surface_nodes[index].id,
                surface_nodes[index + step].id,
            ))
            if edge is None:
                return None
            path.append(edge.id)
        return tuple(path)

    def path_cost(edge_ids: tuple[str, ...]) -> float:
        return sum(_edge_cost(edge_by_id[edge_id]) for edge_id in edge_ids)

    recovery_paths: list[tuple[str, ...]] = []
    seen_paths: set[tuple[str, ...]] = set()
    ordered_surfaces = sorted(
        (
            (surface_id, sorted(nodes, key=lambda node: (node.x, node.id)))
            for surface_id, nodes in nodes_by_surface.items()
            if surface_id not in excluded_surface_ids
        ),
        key=lambda item: (
            min(node.y for node in item[1]),
            item[1][0].x,
            item[0],
        ),
    )
    for surface_id, surface_nodes in ordered_surfaces:
        if len(surface_nodes) < 2:
            raise ValueError(
                f"Platform {surface_id} has too few navigation nodes for "
                "two-sided recovery"
            )
        last_index = len(surface_nodes) - 1
        center_x = (surface_nodes[0].x + surface_nodes[-1].x) / 2.0
        candidates: list[
            tuple[
                float,
                float,
                float,
                float,
                str,
                tuple[str, ...],
                tuple[str, ...],
                tuple[str, ...],
            ]
        ] = []

        patrol_junctions = [
            (index, node, (), 0.0)
            for index, node in enumerate(surface_nodes)
            if node.id in patrol_node_ids
        ]
        junctions = patrol_junctions
        if not junctions:
            junctions = []
            for index, node in enumerate(surface_nodes):
                for cross_edge in cross_edges.get(node.id, ()):
                    tail = _shortest_path_to_any(
                        graph,
                        cross_edge.target,
                        patrol_node_ids,
                        excluded_actions=frozenset({Action.PORTAL}),
                    )
                    if tail is None:
                        continue
                    escape_path = (cross_edge.id, *tail[2])
                    junctions.append((
                        index,
                        node,
                        escape_path,
                        _edge_cost(cross_edge) + tail[0],
                    ))

        for index, node, escape_path, escape_cost in junctions:
            left_path = walk_path(surface_nodes, 0, index)
            right_path = walk_path(surface_nodes, last_index, index)
            if left_path is None or right_path is None:
                continue
            candidates.append((
                max(path_cost(left_path), path_cost(right_path))
                + escape_cost,
                escape_cost,
                abs(node.x - center_x),
                node.x,
                node.id,
                escape_path,
                left_path,
                right_path,
            ))

        if not candidates:
            raise ValueError(
                f"Platform {surface_id} has no non-portal recovery path "
                "into the Forest Floor patrol"
            )
        (
            _, _, _, _, _, escape_path, left_path, right_path
        ) = min(candidates)
        for approach_path in (left_path, right_path):
            path = (*approach_path, *escape_path)
            if path and path not in seen_paths:
                recovery_paths.append(path)
                seen_paths.add(path)

    return tuple(recovery_paths)


def build_patrol_plan(graph: NavigationGraph) -> PatrolPlan:
    """Greedily cover configured spawn sweeps or safe firing positions."""
    node_by_id = graph.node_by_id
    remaining = set(graph.coverage_targets)
    if not remaining:
        return PatrolPlan((), (), ())
    ordered_targets = sorted(
        remaining,
        key=lambda node_id: (
            -node_by_id[node_id].y,
            node_by_id[node_id].x,
            node_id,
        ),
    )
    all_edges: list[str] = []
    visited: list[str] = []
    unreachable: list[str] = []

    while remaining:
        start = next(
            (node_id for node_id in ordered_targets if node_id in remaining),
            min(remaining),
        )
        component_edge_start = len(all_edges)
        component_start = start
        current = start
        remaining.remove(start)
        visited.append(start)
        while remaining:
            candidates: list[tuple[float, str, tuple[str, ...]]] = []
            for target in sorted(remaining):
                result = shortest_path(graph, current, target)
                if result is not None:
                    candidates.append((result[0], target, result[1]))
            if not candidates:
                break
            _, target, path = min(candidates, key=lambda item: (item[0], item[1]))
            all_edges.extend(path)
            current = target
            remaining.remove(target)
            visited.append(target)
        closing = shortest_path(graph, current, component_start)
        if closing is not None:
            all_edges.extend(closing[1])
        else:
            unreachable.append(component_start)
        if len(all_edges) == component_edge_start:
            local_loop = _local_walk_loop(graph, component_start)
            if local_loop:
                all_edges.extend(local_loop)
            else:
                unreachable.append(component_start)

        if remaining:
            # A new disconnected component gets its own loop. Runtime route
            # recovery can select the locally overlapping loop after a map
            # spawn or portal places Hero there.
            reachable_from_any = False
            for source in visited:
                if any(
                    shortest_path(graph, source, target) is not None
                    for target in remaining
                ):
                    reachable_from_any = True
                    break
            if reachable_from_any:
                continue

    truly_unreachable = tuple(sorted(set(unreachable)))
    return PatrolPlan(
        edge_ids=tuple(all_edges),
        visited_targets=tuple(visited),
        unreachable_targets=truly_unreachable,
        recovery_edge_paths=_monster_platform_recovery_paths(graph),
    )


def number_forest_floor_platforms(
    wz_map: WzMap,
    graph: NavigationGraph,
) -> dict[int, Surface]:
    """Recover the visible P1-P13 labels from Forest Floor geometry.

    Platform numbers are recovered from geometry instead of foothold IDs:
    P1 is the wide ground platform, then each row is numbered left-to-right
    from bottom to top. This keeps the route tied to the visible map layout
    while still failing loudly if the WZ export no longer has that layout.
    """
    if wz_map.map_id != "100040110":
        raise ValueError(
            "Forest Floor patrol is only valid for WZ map 100040110"
        )

    surfaces = {
        surface.id: surface
        for surface in wz_map.surfaces
        if surface.supports_default_motion
    }
    monster_surfaces = [
        surfaces[surface_id]
        for surface_id in graph.monster_surface_ids
        if surface_id in surfaces
    ]
    if len(monster_surfaces) != 7:
        raise ValueError(
            "Forest Floor patrol requires seven monster platforms"
        )

    def level_y(surface: Surface) -> float:
        x = (surface.min_x + surface.max_x) / 2.0
        return surface.y_at(x)

    ground = max(
        monster_surfaces,
        key=lambda surface: (
            surface.max_x - surface.min_x,
            level_y(surface),
            surface.id,
        ),
    )
    numbered_monster_surfaces = [
        surface for surface in monster_surfaces if surface.id != ground.id
    ]
    monster_row_y = sorted(
        {
            round(level_y(surface), 3)
            for surface in numbered_monster_surfaces
        },
        reverse=True,
    )
    if len(monster_row_y) != 4:
        raise ValueError(
            "Forest Floor patrol requires four numbered platform rows"
        )

    numbered_rows: list[list[Surface]] = []
    for row_y in monster_row_y:
        row = sorted(
            (
                surface for surface in surfaces.values()
                if surface.max_x - surface.min_x >= 300.0
                and abs(level_y(surface) - row_y) <= 3.0
            ),
            key=lambda surface: (surface.min_x, surface.id),
        )
        if len(row) != 3:
            raise ValueError(
                "Forest Floor patrol expected three platforms at "
                f"WZ y={row_y:g}, found {len(row)}"
            )
        numbered_rows.append(row)

    platform_list = [ground]
    for row in numbered_rows:
        platform_list.extend(row)
    if len(platform_list) != 13:
        raise ValueError("Forest Floor patrol platform numbering failed")
    return {
        number: surface for number, surface in enumerate(platform_list, 1)
    }


def forest_floor_platform_patrol_anchors(
    wz_map: WzMap,
    graph: NavigationGraph,
    platforms: dict[int, Surface] | None = None,
) -> dict[int, tuple[str, ...]]:
    """Derive safe patrol/firing anchors without encoding route edges."""
    platforms = platforms or number_forest_floor_platforms(wz_map, graph)
    node_by_id = graph.node_by_id
    result: dict[int, tuple[str, ...]] = {}

    for number, surface in platforms.items():
        safe_nodes = sorted(
            (
                node_by_id[node_id]
                for node_id in graph.safe_firing_targets
                if node_id in node_by_id
                and node_by_id[node_id].surface_id == surface.id
            ),
            key=lambda node: (node.x, node.id),
        )
        if safe_nodes:
            result[number] = (
                (safe_nodes[0].id,)
                if len(safe_nodes) == 1
                else (safe_nodes[0].id, safe_nodes[-1].id)
            )

    # P1's visible safe sweep is bounded by the two lowest tree-hole portals,
    # rather than the physical foothold ends hidden behind solid tree trunks.
    ground = platforms[1]
    ground_nodes = sorted(
        (node for node in graph.nodes if node.surface_id == ground.id),
        key=lambda node: (node.x, node.id),
    )
    hidden_portals = []
    for portal in wz_map.portals:
        if portal.type != 1 or not ground.contains_x(portal.x):
            continue
        vertical_gap = ground.y_at(portal.x) - portal.y
        if vertical_gap >= -EPSILON:
            hidden_portals.append((vertical_gap, portal.x, portal.id, portal))
    if len(hidden_portals) < 2 or not ground_nodes:
        raise ValueError("Forest Floor P1 requires two tree-hole patrol anchors")
    lowest_portals = [
        candidate[-1] for candidate in sorted(hidden_portals)[:2]
    ]
    p1_nodes = sorted(
        (
            min(
                ground_nodes,
                key=lambda node: (
                    abs(node.x - portal.x), node.x, node.id
                ),
            )
            for portal in lowest_portals
        ),
        key=lambda node: (node.x, node.id),
    )
    if p1_nodes[0].id == p1_nodes[1].id:
        raise ValueError("Forest Floor P1 tree holes resolved to one anchor")
    result[1] = (p1_nodes[0].id, p1_nodes[1].id)
    return result


def build_forest_floor_patrol_plan(
    wz_map: WzMap,
    graph: NavigationGraph,
) -> PatrolPlan:
    """Build the requested 1-13 platform loop for Forest Floor."""
    platform = number_forest_floor_platforms(wz_map, graph)
    platform_patrol_anchors = forest_floor_platform_patrol_anchors(
        wz_map, graph, platform
    )
    surfaces = {
        surface.id: surface
        for surface in wz_map.surfaces
        if surface.supports_default_motion
    }
    ground = platform[1]

    def level_y(surface: Surface) -> float:
        x = (surface.min_x + surface.max_x) / 2.0
        return surface.y_at(x)

    node_by_id = graph.node_by_id

    def nodes_from(ids: tuple[str, ...], number: int) -> list[Node]:
        surface_id = platform[number].id
        return sorted(
            (
                node_by_id[node_id]
                for node_id in ids
                if node_by_id[node_id].surface_id == surface_id
            ),
            key=lambda node: (node.x, node.id),
        )

    def firing_node(number: int, side: str) -> Node:
        candidates = nodes_from(graph.safe_firing_targets, number)
        if not candidates:
            raise ValueError(
                f"Forest Floor P{number} has no safe firing node"
            )
        return candidates[0] if side == "left" else candidates[-1]

    def transition_source(
        source_number: int,
        target_number: int,
        action: Action,
        side: str,
    ) -> Node:
        candidates = sorted(
            (
                node_by_id[edge.source]
                for edge in graph.edges
                if edge.action is action
                and node_by_id[edge.source].surface_id ==
                    platform[source_number].id
                and node_by_id[edge.target].surface_id ==
                    platform[target_number].id
            ),
            key=lambda node: (node.x, node.id),
        )
        if not candidates:
            raise ValueError(
                f"Forest Floor has no {action.value} transition "
                f"P{source_number}->P{target_number}"
            )
        return candidates[0] if side == "left" else candidates[-1]

    def outgoing_drop_near_x(number: int, target_x: float) -> Node:
        surface = platform[number]
        candidates = {
            edge.source: node_by_id[edge.source]
            for edge in graph.edges
            if edge.action is Action.DROP
            and node_by_id[edge.source].surface_id == surface.id
        }
        if not candidates:
            raise ValueError(
                f"Forest Floor P{number} has no downward transition"
            )
        return min(
            candidates.values(),
            key=lambda node: (abs(node.x - target_x), node.x, node.id),
        )

    p1_rope = transition_source(1, 3, Action.CLIMB, "left")
    p3_left = firing_node(3, "left")
    p3_right = firing_node(3, "right")
    p4_rope = transition_source(4, 7, Action.CLIMB, "left")
    p7_left = firing_node(7, "left")
    p6_rope = transition_source(6, 9, Action.CLIMB, "right")
    p9_left = firing_node(9, "left")
    p9_right = firing_node(9, "right")
    p10_rope = transition_source(10, 13, Action.CLIMB, "left")
    p13_left = firing_node(13, "left")
    p12_drop = transition_source(12, 9, Action.DROP, "left")
    p8_drop = transition_source(8, 5, Action.DROP, "left")
    p5_right = firing_node(5, "right")
    p6_drop = transition_source(6, 3, Action.DROP, "left")
    p1_left, p1_right = (
        node_by_id[node_id] for node_id in platform_patrol_anchors[1]
    )
    # Descend from P2 directly above P1's left tree-hole stop.  Choosing the
    # geometric middle made Hero patrol 17 navigation pixels farther left and
    # then walk back on P1; this WZ-derived alignment keeps the P2 boundary
    # conservative and turns the whole descent into one vertical chain.
    p2_drop = outgoing_drop_near_x(2, p1_left.x)
    p1_surface_nodes = nodes_from(tuple(node_by_id), 1)

    checkpoint_templates = (
        CombatCheckpoint(p3_left.id, "left", "P3 clear P2"),
        CombatCheckpoint(p3_right.id, "right", "P3 clear P4"),
        CombatCheckpoint(p7_left.id, "left", "P7 clear P6"),
        CombatCheckpoint(p9_left.id, "left", "P9 clear P8"),
        CombatCheckpoint(p9_right.id, "right", "P9 clear P10"),
        CombatCheckpoint(p13_left.id, "left", "P13 clear P12"),
        CombatCheckpoint(p5_right.id, "right", "P5 clear P6"),
        CombatCheckpoint(p1_left.id, "both", "P1 clear left outside"),
        CombatCheckpoint(p1_right.id, "both", "P1 clear right outside"),
    )
    checkpoint_by_node = {
        checkpoint.node_id: checkpoint
        for checkpoint in checkpoint_templates
    }

    # The order mirrors the user's numbered route. Repeated P3/P9 firing
    # nodes deliberately become repeated clear-before-move stops on descent.
    stops = (
        p1_rope,
        p3_left,
        p3_right,
        p4_rope,
        p7_left,
        p6_rope,
        p9_left,
        p9_right,
        p10_rope,
        p13_left,
        p12_drop,
        p9_right,
        p9_left,
        p8_drop,
        p5_right,
        p6_drop,
        p3_left,
        p3_right,
        p2_drop,
        p1_left,
        p1_right,
        p1_rope,
    )

    edge_ids: list[str] = []
    checkpoint_events: list[CombatCheckpoint] = []
    edge_by_id = graph.edge_by_id

    def wall_recovery_path(source: Node, target: Node) -> tuple[str, ...]:
        result = shortest_path(graph, source.id, target.id)
        if result is None or not result[1]:
            raise ValueError(
                "Forest Floor P1 wall recovery cannot reach its tree hole"
            )
        path = result[1]
        if any(
            edge_by_id[edge_id].action is not Action.WALK
            or node_by_id[edge_by_id[edge_id].source].surface_id != ground.id
            or node_by_id[edge_by_id[edge_id].target].surface_id != ground.id
            for edge_id in path
        ):
            raise ValueError(
                "Forest Floor P1 wall recovery must stay on the ground"
            )
        return path

    p1_wall_recovery_paths = (
        wall_recovery_path(p1_surface_nodes[0], p1_left),
        wall_recovery_path(p1_surface_nodes[-1], p1_right),
    )
    top_y = level_y(platform[13])
    for source, target in pairwise(stops):
        result = shortest_path(graph, source.id, target.id)
        if result is None:
            raise ValueError(
                "Forest Floor patrol cannot connect "
                f"{source.id} to {target.id}"
            )
        for edge_id in result[1]:
            edge = edge_by_id[edge_id]
            if edge.action is Action.PORTAL:
                raise ValueError(
                    "Forest Floor patrol unexpectedly selected a portal"
                )
            if node_by_id[edge.target].y < top_y - EPSILON:
                raise ValueError(
                    "Forest Floor patrol escaped above platform 13 while "
                    f"connecting {source.id} to {target.id}: {edge.id}"
                )
        edge_ids.extend(result[1])
        checkpoint = checkpoint_by_node.get(target.id)
        if checkpoint is not None and result[1]:
            checkpoint_events.append(CombatCheckpoint(
                node_id=checkpoint.node_id,
                facing=checkpoint.facing,
                label=checkpoint.label,
                after_edge_index=len(edge_ids) - 1,
            ))

    # P1 keeps its short, tree-hole-bounded wall recoveries. Every other WZ
    # platform receives two inward branches so relogin/knockback can rejoin
    # the numbered patrol from either side, including the unnumbered top
    # spawn platform above P11-P13.
    platform_recovery_paths = _all_platform_recovery_paths(
        graph,
        tuple(edge_ids),
        excluded_surface_ids=frozenset({ground.id}),
    )

    return PatrolPlan(
        edge_ids=tuple(edge_ids),
        visited_targets=tuple(stop.id for stop in stops),
        unreachable_targets=(),
        recovery_edge_paths=(
            *p1_wall_recovery_paths,
            *platform_recovery_paths,
        ),
        combat_checkpoints=tuple(checkpoint_events),
    )
