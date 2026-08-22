"""Extract stable X-coordinate anchors from recorded route images.

Recorded routes are not executed while WZ navigation is active.  Their
compact jump markers and successful rope-mount trajectories are still useful
calibration evidence: WZ owns topology/Y coordinates, while the recording
provides the X coordinate at which this Hero actually performed the action.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from .hunt_planner import Action, NavigationGraph
from .wz_geometry import Point

if TYPE_CHECKING:
    from .route_renderer import NavigationProjection


JUMP_SOURCE_Y_TOLERANCE_PX = 12
CLIMB_SOURCE_Y_TOLERANCE_PX = 20
CLIMB_SOURCE_X_TOLERANCE_PX = 24


@dataclass(frozen=True, slots=True)
class CanonicalJumpAnchor:
    point: tuple[float, float]
    direction: int
    live_point: tuple[int, int]
    route_name: str


@dataclass(frozen=True, slots=True)
class CanonicalClimbAnchor:
    contact_point: tuple[float, float]
    source_point: tuple[float, float]
    launch_point: tuple[float, float]
    approach_direction: int
    live_contact_x: int
    live_launch_x: int
    live_source_y: int
    route_name: str


@dataclass(frozen=True, slots=True)
class NavigationJumpAnchor:
    x: int
    y: int
    direction: int
    live_point: tuple[int, int]
    route_name: str


@dataclass(frozen=True, slots=True)
class NavigationClimbAnchor:
    contact_x: int
    source_y: int
    launch_x: int
    approach_direction: int
    live_contact_x: int
    live_launch_x: int
    live_source_y: int
    route_name: str

    @property
    def approach_name(self) -> str:
        return "right" if self.approach_direction > 0 else "left"


@dataclass(frozen=True, slots=True)
class ProjectedRecordedRouteAnchors:
    jumps: tuple[NavigationJumpAnchor, ...] = ()
    climbs: tuple[NavigationClimbAnchor, ...] = ()

    def jump_for(
        self,
        *,
        source_y: int,
        expected_x: int,
        direction: int,
    ) -> NavigationJumpAnchor | None:
        candidates = [
            anchor
            for anchor in self.jumps
            if anchor.direction == direction
            and abs(anchor.y - source_y) <= JUMP_SOURCE_Y_TOLERANCE_PX
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda anchor: (
                abs(anchor.x - expected_x),
                abs(anchor.y - source_y),
                anchor.route_name,
                anchor.live_point,
            ),
        )

    def match_climb_edges(
        self,
        graph: NavigationGraph,
        projection: NavigationProjection,
    ) -> dict[str, NavigationClimbAnchor]:
        """Pair each recorded ascent with the nearest upward CLIMB edge."""
        nodes = graph.node_by_id
        available = []
        for edge in graph.edges:
            if edge.action is not Action.CLIMB:
                continue
            source = nodes[edge.source]
            target = nodes[edge.target]
            if target.y >= source.y:
                continue
            source_pixel = projection.world_to_navigation(
                Point(source.x, source.y)
            )
            available.append((edge, source_pixel))

        matched: dict[str, NavigationClimbAnchor] = {}
        used_edges: set[str] = set()
        for anchor in sorted(
            self.climbs,
            key=lambda item: (
                item.source_y,
                item.contact_x,
                item.route_name,
            ),
        ):
            candidates = []
            for edge, source_pixel in available:
                if edge.id in used_edges:
                    continue
                dx = abs(source_pixel[0] - anchor.contact_x)
                dy = abs(source_pixel[1] - anchor.source_y)
                if dy > CLIMB_SOURCE_Y_TOLERANCE_PX or \
                        dx > CLIMB_SOURCE_X_TOLERANCE_PX:
                    continue
                candidates.append((dy, dx, edge.id, edge))
            if not candidates:
                continue
            edge = min(candidates)[-1]
            matched[edge.id] = anchor
            used_edges.add(edge.id)
        return matched


def prefer_recorded_climb_edges(
    graph: NavigationGraph,
    matched: dict[str, NavigationClimbAnchor],
) -> NavigationGraph:
    """Keep the recorded rope when one surface pair has parallel climbs."""
    if not matched:
        return graph
    nodes = graph.node_by_id
    preferred_by_transition = {}
    for edge_id in matched:
        edge = graph.edge_by_id[edge_id]
        source = nodes[edge.source]
        target = nodes[edge.target]
        preferred_by_transition[(
            source.surface_id,
            target.surface_id,
            edge.action,
        )] = edge.id

    edges = tuple(
        edge for edge in graph.edges
        if edge.action is not Action.CLIMB
        or preferred_by_transition.get((
            nodes[edge.source].surface_id,
            nodes[edge.target].surface_id,
            edge.action,
        ), edge.id) == edge.id
    )
    if len(edges) == len(graph.edges):
        return graph
    return NavigationGraph(
        nodes=graph.nodes,
        edges=edges,
        coverage_targets=graph.coverage_targets,
        monster_surface_ids=graph.monster_surface_ids,
        safe_firing_targets=graph.safe_firing_targets,
        safe_covered_monster_surface_ids=(
            graph.safe_covered_monster_surface_ids
        ),
    )


@dataclass(frozen=True, slots=True)
class RecordedRouteAnchors:
    jumps: tuple[CanonicalJumpAnchor, ...] = ()
    climbs: tuple[CanonicalClimbAnchor, ...] = ()
    route_files: tuple[str, ...] = ()

    def project(
        self, projection: NavigationProjection
    ) -> ProjectedRecordedRouteAnchors:
        jumps = []
        for anchor in self.jumps:
            x, y = projection.canonical_to_navigation(anchor.point)
            jumps.append(NavigationJumpAnchor(
                x=int(round(x)),
                y=int(round(y)),
                direction=anchor.direction,
                live_point=anchor.live_point,
                route_name=anchor.route_name,
            ))

        climbs = []
        for anchor in self.climbs:
            contact_x, _ = projection.canonical_to_navigation(
                anchor.contact_point
            )
            _, source_y = projection.canonical_to_navigation(
                anchor.source_point
            )
            launch_x, _ = projection.canonical_to_navigation(
                anchor.launch_point
            )
            climbs.append(NavigationClimbAnchor(
                contact_x=int(round(contact_x)),
                source_y=int(round(source_y)),
                launch_x=int(round(launch_x)),
                approach_direction=anchor.approach_direction,
                live_contact_x=anchor.live_contact_x,
                live_launch_x=anchor.live_launch_x,
                live_source_y=anchor.live_source_y,
                route_name=anchor.route_name,
            ))
        return ProjectedRecordedRouteAnchors(
            jumps=tuple(jumps),
            climbs=tuple(climbs),
        )


@dataclass(frozen=True, slots=True)
class _LiveJump:
    x: int
    y: int
    direction: int
    route_name: str


@dataclass(frozen=True, slots=True)
class _LiveClimb:
    contact_x: int
    contact_y: int
    source_y: int
    launch_x: int
    launch_y: int
    approach_direction: int
    route_name: str


def _color_maps(route_config: dict[str, Any]):
    normal = {
        tuple(map(int, key.split(","))): str(command)
        for key, command in route_config.get("color_code", {}).items()
    }
    vertical = {
        tuple(map(int, key.split(","))): str(command)
        for key, command in route_config.get(
            "color_code_up_down", {}
        ).items()
    }
    return normal, vertical


def _mask_map_route_colors(
    reference_map_bgr: np.ndarray,
    route_rgb: np.ndarray,
    colors: tuple[tuple[int, int, int], ...],
) -> np.ndarray:
    if reference_map_bgr.shape[:2] != route_rgb.shape[:2]:
        raise ValueError(
            "Recorded route/reference map canvas mismatch: "
            f"map={reference_map_bgr.shape[:2]}, "
            f"route={route_rgb.shape[:2]}"
        )
    mask = np.zeros(reference_map_bgr.shape[:2], dtype=np.bool_)
    for color in colors:
        mask |= np.all(reference_map_bgr == color, axis=2)
    result = route_rgb.copy()
    result[mask] = (0, 0, 0)
    return result


def _compact_directional_jumps(
    route_rgb: np.ndarray,
    normal_colors: dict[tuple[int, int, int], str],
    route_name: str,
) -> list[_LiveJump]:
    result = []
    for color, command in normal_colors.items():
        parts = command.split()
        if len(parts) != 3 or parts[0] not in {"left", "right"} or \
                parts[2] != "jump":
            continue
        mask = np.all(route_rgb == color, axis=2).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        for component in range(1, count):
            x, y, width, height, area = map(int, stats[component])
            if area < 5 or width > 7 or height > 7:
                continue
            result.append(_LiveJump(
                x=x + width // 2,
                y=y + height // 2,
                direction=-1 if parts[0] == "left" else 1,
                route_name=route_name,
            ))
    return result


def _successful_climbs(
    route_rgb: np.ndarray,
    vertical_colors: dict[tuple[int, int, int], str],
    jumps: list[_LiveJump],
    route_name: str,
) -> tuple[list[_LiveClimb], set[_LiveJump]]:
    up_color = next(
        (
            color for color, command in vertical_colors.items()
            if command == "none up none"
        ),
        None,
    )
    if up_color is None:
        return [], set()

    mask = np.all(route_rgb == up_color, axis=2).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    climbs = []
    paired: set[_LiveJump] = set()
    for component in range(1, count):
        x, y, width, height, area = map(int, stats[component])
        if height < 12 or width > 20 or area < height - 2:
            continue
        ys, xs = np.where(labels == component)
        top_limit = y + max(2, height // 3)
        upper_xs = [int(value) for value in xs[ys < top_limit]]
        if not upper_xs:
            continue
        frequencies = Counter(upper_xs)
        contact_x = min(
            frequencies,
            key=lambda value: (-frequencies[value], value),
        )
        bottom_y = y + height - 1

        candidates = []
        for jump in jumps:
            if jump in paired:
                continue
            dy = abs(jump.y - bottom_y)
            dx = abs(jump.x - contact_x)
            if dy > 3 or not 1 <= dx <= 24:
                continue
            expected_direction = 1 if contact_x > jump.x else -1
            if jump.direction != expected_direction:
                continue
            candidates.append((dy, dx, jump.x, jump.y, jump))
        if not candidates:
            continue
        jump = min(candidates)[-1]
        paired.add(jump)
        climbs.append(_LiveClimb(
            contact_x=contact_x,
            contact_y=y,
            source_y=jump.y,
            launch_x=jump.x,
            launch_y=jump.y,
            approach_direction=jump.direction,
            route_name=route_name,
        ))
    return climbs, paired


def load_recorded_route_anchors(
    route_directory: str | Path | None,
    reference_map_bgr: np.ndarray,
    route_config: dict[str, Any],
    registration: Any,
) -> RecordedRouteAnchors:
    """Load human-recorded action evidence and convert it to canvas space."""
    if route_directory is None:
        return RecordedRouteAnchors()
    directory = Path(route_directory)
    route_paths = tuple(sorted(
        path for path in directory.glob("route*.png")
        if path.name != "route_rest.png"
    ))
    if not route_paths:
        return RecordedRouteAnchors()

    normal_colors, vertical_colors = _color_maps(route_config)
    all_colors = tuple((*normal_colors, *vertical_colors))
    live_jumps = []
    live_climbs = []
    for route_path in route_paths:
        route_bgr = cv2.imread(str(route_path), cv2.IMREAD_COLOR)
        if route_bgr is None:
            raise ValueError(f"Unable to load recorded route: {route_path}")
        route_rgb = cv2.cvtColor(route_bgr, cv2.COLOR_BGR2RGB)
        route_rgb = _mask_map_route_colors(
            reference_map_bgr, route_rgb, all_colors
        )
        jumps = _compact_directional_jumps(
            route_rgb, normal_colors, route_path.name
        )
        climbs, paired = _successful_climbs(
            route_rgb, vertical_colors, jumps, route_path.name
        )
        live_climbs.extend(climbs)
        live_jumps.extend(jump for jump in jumps if jump not in paired)

    canonical_jumps = []
    seen_jumps = set()
    for anchor in sorted(
        live_jumps,
        key=lambda item: (
            item.y, item.x, item.direction, item.route_name
        ),
    ):
        key = (anchor.x, anchor.y, anchor.direction)
        if key in seen_jumps:
            continue
        seen_jumps.add(key)
        canonical_jumps.append(CanonicalJumpAnchor(
            point=registration.live_to_canonical((anchor.x, anchor.y)),
            direction=anchor.direction,
            live_point=(anchor.x, anchor.y),
            route_name=anchor.route_name,
        ))

    canonical_climbs = []
    for anchor in sorted(
        live_climbs,
        key=lambda item: (
            item.source_y, item.contact_x, item.route_name
        ),
    ):
        canonical_climbs.append(CanonicalClimbAnchor(
            contact_point=registration.live_to_canonical((
                anchor.contact_x, anchor.contact_y
            )),
            source_point=registration.live_to_canonical((
                anchor.launch_x, anchor.source_y
            )),
            launch_point=registration.live_to_canonical((
                anchor.launch_x, anchor.launch_y
            )),
            approach_direction=anchor.approach_direction,
            live_contact_x=anchor.contact_x,
            live_launch_x=anchor.launch_x,
            live_source_y=anchor.source_y,
            route_name=anchor.route_name,
        ))

    return RecordedRouteAnchors(
        jumps=tuple(canonical_jumps),
        climbs=tuple(canonical_climbs),
        route_files=tuple(path.name for path in route_paths),
    )
