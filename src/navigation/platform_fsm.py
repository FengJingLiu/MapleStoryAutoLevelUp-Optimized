"""Platform-level patrol state machine backed by the WZ action graph.

The state machine stores only a cyclic list of platform numbers.  It creates
one short-lived path from Hero's latest grounded position to the current
platform target, so callers never need to pre-paint or pre-render a complete
route loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable

from .hunt_planner import (
    Action,
    Edge,
    MotionProfile,
    NavigationGraph,
    Node,
    shortest_path_to_any,
)
from .wz_geometry import EPSILON, Point, Surface


class PlatformPhase(StrEnum):
    LOCALIZING = "LOCALIZING"
    TRAVELING = "TRAVELING"
    DWELLING = "DWELLING"
    BLOCKED = "BLOCKED"
    SUSPENDED = "SUSPENDED"


class PlatformPathPurpose(StrEnum):
    TRANSIT = "TRANSIT"
    PATROL = "PATROL"
    RECOVERY = "RECOVERY"


@dataclass(frozen=True, slots=True)
class PlatformPath:
    """One temporary path for the existing low-level action executor."""

    graph: NavigationGraph
    edge_ids: tuple[str, ...]
    purpose: PlatformPathPurpose
    source_platform: int | None
    target_platform: int
    target_node_id: str
    label: str


@dataclass(frozen=True, slots=True)
class PlatformStateSnapshot:
    phase: PlatformPhase
    state_index: int | None
    current_platform: int | None
    target_platform: int | None
    sequence: tuple[int, ...]
    dwell_elapsed_seconds: float
    active_path_label: str | None
    path_revision: int
    blocked_reason: str | None


class PlatformPatrolStateMachine:
    """Run a platform sequence and calculate only the currently needed path."""

    def __init__(
        self,
        graph: NavigationGraph,
        motion_profile: MotionProfile,
        platforms: dict[int, Surface],
        sequence: tuple[int, ...],
        *,
        patrol_anchors: dict[int, tuple[str, ...]] | None = None,
        dwell_seconds: float,
        combat_quiet_seconds: float,
        maximum_dwell_seconds: float,
        travel_combat_budget_seconds: float = 6.0,
        stable_surface_frames: int = 2,
        arrival_tolerance_wz: float = 14.0,
        blocked_retry_seconds: float = 1.0,
        maximum_recovery_drop_height_wz: float = 300.0,
        exclude_portals: bool = True,
        patrol_enabled: bool = True,
        allowed_platforms: frozenset[int] | None = None,
        log: Callable[[str], None] | None = None,
    ):
        if not sequence:
            raise ValueError("platform state machine sequence cannot be empty")
        missing = sorted(set(sequence).difference(platforms))
        if missing:
            raise ValueError(
                f"platform state machine references unknown platforms: {missing}"
            )
        normalized_allowed = (
            None
            if allowed_platforms is None
            else frozenset(int(value) for value in allowed_platforms)
        )
        if normalized_allowed is not None:
            missing_allowed = sorted(normalized_allowed.difference(platforms))
            if missing_allowed:
                raise ValueError(
                    "platform state machine allowed_platforms references "
                    f"unknown platforms: {missing_allowed}"
                )
            excluded_targets = sorted(
                set(sequence).difference(normalized_allowed)
            )
            if excluded_targets:
                raise ValueError(
                    "platform state targets must be included in "
                    f"allowed_platforms: {excluded_targets}"
                )
        numeric_values = (
            dwell_seconds,
            combat_quiet_seconds,
            maximum_dwell_seconds,
            travel_combat_budget_seconds,
            arrival_tolerance_wz,
            blocked_retry_seconds,
            maximum_recovery_drop_height_wz,
        )
        if any(not math.isfinite(float(value)) for value in numeric_values):
            raise ValueError("platform state machine values must be finite")
        if dwell_seconds < 0 or combat_quiet_seconds < 0:
            raise ValueError("platform dwell/quiet seconds cannot be negative")
        if travel_combat_budget_seconds < 0:
            raise ValueError("platform travel combat budget cannot be negative")
        if maximum_dwell_seconds <= 0 or \
                maximum_dwell_seconds < dwell_seconds:
            raise ValueError(
                "platform maximum_dwell_seconds must be >= dwell_seconds"
            )
        if arrival_tolerance_wz <= 0 or blocked_retry_seconds <= 0:
            raise ValueError("platform tolerances and retry time must be positive")
        if maximum_recovery_drop_height_wz <= 0:
            raise ValueError("maximum recovery drop height must be positive")

        self.graph = graph
        self.motion_profile = motion_profile
        self.platforms = dict(platforms)
        self.allowed_platforms = normalized_allowed
        self.patrol_anchors = dict(patrol_anchors or {})
        self.sequence = tuple(int(value) for value in sequence)
        self.dwell_seconds = float(dwell_seconds)
        self.combat_quiet_seconds = float(combat_quiet_seconds)
        self.maximum_dwell_seconds = float(maximum_dwell_seconds)
        self.travel_combat_budget_seconds = float(
            travel_combat_budget_seconds
        )
        self.stable_surface_frames = max(1, int(stable_surface_frames))
        self.arrival_tolerance_wz = float(arrival_tolerance_wz)
        self.blocked_retry_seconds = float(blocked_retry_seconds)
        self.maximum_recovery_drop_height_wz = float(
            maximum_recovery_drop_height_wz
        )
        self.patrol_enabled = bool(patrol_enabled)
        self.excluded_actions = (
            frozenset({Action.PORTAL}) if exclude_portals else frozenset()
        )
        self.log = log

        self.phase = PlatformPhase.LOCALIZING
        self.state_index: int | None = None
        self.current_platform: int | None = None
        self.active_path: PlatformPath | None = None
        self.path_revision = 0
        self.blocked_reason: str | None = None
        self._blocked_retry_at = 0.0

        self._surface_number = {
            surface.id: number for number, surface in self.platforms.items()
        }
        self._candidate_surface_id: str | None = None
        self._candidate_surface_frames = 0
        self._last_surface_id: str | None = None
        self._last_point: Point | None = None
        self._last_position_at: float | None = None
        self._dwell_started_at: float | None = None
        self._paused_dwell_elapsed = 0.0
        self._last_attackable_at: float | None = None
        self._last_combat_observation: bool | None = None
        self._last_combat_observation_at: float | None = None
        self._travel_attackable_since: float | None = None
        self._travel_combat_exhausted = False
        self._patrol_target_node_id: str | None = None
        self._last_replan_at = -math.inf
        self._suspended_phase: PlatformPhase | None = None

    def _log(self, message: str) -> None:
        if self.log is not None:
            self.log(f"[platform-fsm] {message}")

    @property
    def target_platform(self) -> int | None:
        if self.state_index is None:
            return None
        return self.sequence[self.state_index]

    @property
    def intentional_idle(self) -> bool:
        return self.phase is PlatformPhase.DWELLING and self.active_path is None

    @property
    def clear_before_move(self) -> bool:
        return self.phase not in {
            PlatformPhase.LOCALIZING,
            PlatformPhase.SUSPENDED,
        }

    @property
    def combat_priority(self) -> bool:
        if self.phase is PlatformPhase.DWELLING:
            return True
        if self.phase is PlatformPhase.TRAVELING:
            return not self._travel_combat_exhausted
        return False

    @property
    def safe_to_buff(self) -> bool:
        if not self.intentional_idle or self._last_combat_observation is not False:
            return False
        if self._last_combat_observation_at is None:
            return False
        if self._last_attackable_at is None:
            return True
        return self._last_combat_observation_at - self._last_attackable_at >= \
            min(0.25, self.combat_quiet_seconds)

    def snapshot(self, timestamp: float | None = None) -> PlatformStateSnapshot:
        now = self._last_position_at if timestamp is None else float(timestamp)
        elapsed = 0.0
        if self._dwell_started_at is not None and now is not None:
            elapsed = max(0.0, now - self._dwell_started_at)
        elif self._paused_dwell_elapsed > 0:
            elapsed = self._paused_dwell_elapsed
        return PlatformStateSnapshot(
            phase=self.phase,
            state_index=self.state_index,
            current_platform=self.current_platform,
            target_platform=self.target_platform,
            sequence=self.sequence,
            dwell_elapsed_seconds=elapsed,
            active_path_label=(
                None if self.active_path is None else self.active_path.label
            ),
            path_revision=self.path_revision,
            blocked_reason=self.blocked_reason,
        )

    def rebind_graph(
        self,
        graph: NavigationGraph,
        motion_profile: MotionProfile,
        platforms: dict[int, Surface],
        *,
        patrol_anchors: dict[int, tuple[str, ...]] | None = None,
    ) -> None:
        """Replace learned graph geometry without resetting the patrol state."""
        missing = sorted(set(self.sequence).difference(platforms))
        if missing:
            raise ValueError(
                f"rebuilt graph lost platform definitions: {missing}"
            )
        if self.allowed_platforms is not None:
            missing_allowed = sorted(
                self.allowed_platforms.difference(platforms)
            )
            if missing_allowed:
                raise ValueError(
                    "rebuilt graph lost allowed platform definitions: "
                    f"{missing_allowed}"
                )
        self.graph = graph
        self.motion_profile = motion_profile
        self.platforms = dict(platforms)
        self.patrol_anchors = dict(patrol_anchors or {})
        self._surface_number = {
            surface.id: number for number, surface in self.platforms.items()
        }
        self._replace_path(None)
        self._patrol_target_node_id = None
        self.blocked_reason = None
        if self.phase is PlatformPhase.BLOCKED:
            self.phase = PlatformPhase.TRAVELING
        self._log("Graph rebuilt; current platform state retained and path invalidated")

    def _replace_path(self, path: PlatformPath | None) -> None:
        if path == self.active_path:
            return
        self.active_path = path
        self.path_revision += 1

    def _nodes_on_surface(
        self,
        surface_id: str,
        *,
        graph: NavigationGraph | None = None,
    ) -> list[Node]:
        active_graph = self.graph if graph is None else graph
        return sorted(
            (
                node for node in active_graph.nodes
                if node.surface_id == surface_id
            ),
            key=lambda node: (node.x, node.id),
        )

    def _attach_player_node(
        self,
        surface: Surface,
        point: Point,
    ) -> tuple[NavigationGraph, str]:
        """Add a virtual WALK source at Hero's exact grounded x coordinate."""
        surface_nodes = self._nodes_on_surface(surface.id)
        if not surface_nodes:
            raise ValueError(f"platform surface {surface.id} has no graph nodes")
        x = min(max(float(point.x), surface.min_x), surface.max_x)
        closest = min(
            surface_nodes,
            key=lambda node: (abs(node.x - x), node.x, node.id),
        )
        if abs(closest.x - x) <= 1e-6:
            return self.graph, closest.id

        y = surface.y_at(x)
        node_id = f"runtime-player:{surface.id}:{x:.6f}"
        source = Node(node_id, surface.id, x, y)
        left = [node for node in surface_nodes if node.x < x]
        right = [node for node in surface_nodes if node.x > x]
        neighbors = []
        if left:
            neighbors.append(left[-1])
        if right:
            neighbors.append(right[0])
        if not neighbors:
            neighbors.append(closest)

        dynamic_edges = tuple(
            Edge(
                id=f"runtime-walk:{node_id}->{neighbor.id}",
                action=Action.WALK,
                source=node_id,
                target=neighbor.id,
                expected_time_sec=(
                    abs(neighbor.x - x)
                    / self.motion_profile.walk_speed_wz_per_sec
                ),
            )
            for neighbor in neighbors
        )
        return NavigationGraph(
            nodes=(*self.graph.nodes, source),
            edges=(*self.graph.edges, *dynamic_edges),
            coverage_targets=self.graph.coverage_targets,
            monster_surface_ids=self.graph.monster_surface_ids,
            safe_firing_targets=self.graph.safe_firing_targets,
            safe_covered_monster_surface_ids=(
                self.graph.safe_covered_monster_surface_ids
            ),
        ), node_id

    def _candidate_target_nodes(
        self,
        target_platform: int,
        *,
        specific_node_id: str | None = None,
    ) -> frozenset[str]:
        surface_id = self.platforms[target_platform].id
        if specific_node_id is not None:
            node = self.graph.node_by_id.get(specific_node_id)
            if node is None or node.surface_id != surface_id:
                raise ValueError(
                    f"target node {specific_node_id} is not on P{target_platform}"
                )
            return frozenset({specific_node_id})
        return frozenset(
            node.id for node in self.graph.nodes
            if node.surface_id == surface_id
        )

    def _calculate_path(
        self,
        target_platform: int,
        purpose: PlatformPathPurpose,
        *,
        specific_node_id: str | None = None,
    ) -> PlatformPath | None:
        if self._last_surface_id is None or self._last_point is None:
            return None
        source_surface = next(
            (
                surface for surface in self.platforms.values()
                if surface.id == self._last_surface_id
            ),
            None,
        )
        if source_surface is None:
            # Hero can land on an unnumbered WZ surface. It still belongs to
            # the static graph, so recover the actual Surface from its nodes'
            # identifier via the latest point's containing platform geometry.
            source_nodes = self._nodes_on_surface(self._last_surface_id)
            if not source_nodes:
                return None
            # A lightweight Surface lookup is supplied by observe_position
            # through this cache even for unnumbered platforms.
            source_surface = getattr(self, "_last_surface", None)
            if source_surface is None or source_surface.id != self._last_surface_id:
                return None

        execution_graph, source_node_id = self._attach_player_node(
            source_surface,
            self._last_point,
        )
        if self.allowed_platforms is not None:
            allowed_surface_ids = {
                self.platforms[number].id
                for number in self.allowed_platforms
            }
            if source_surface.id not in allowed_surface_ids:
                return None
            node_by_id = execution_graph.node_by_id
            restricted_edges = tuple(
                edge for edge in execution_graph.edges
                if node_by_id[edge.source].surface_id in allowed_surface_ids
                and node_by_id[edge.target].surface_id in allowed_surface_ids
            )
            execution_graph = NavigationGraph(
                nodes=execution_graph.nodes,
                edges=restricted_edges,
                coverage_targets=execution_graph.coverage_targets,
                monster_surface_ids=execution_graph.monster_surface_ids,
                safe_firing_targets=execution_graph.safe_firing_targets,
                safe_covered_monster_surface_ids=(
                    execution_graph.safe_covered_monster_surface_ids
                ),
            )
        if purpose is PlatformPathPurpose.RECOVERY:
            # A down+jump is reliable between adjacent hunting tiers, but the
            # isolated upper platforms must descend on their WZ rope instead.
            node_by_id = execution_graph.node_by_id
            recovery_edges = tuple(
                edge for edge in execution_graph.edges
                if edge.action is not Action.DROP
                or node_by_id[edge.target].y - node_by_id[edge.source].y
                <= self.maximum_recovery_drop_height_wz + EPSILON
            )
            if len(recovery_edges) != len(execution_graph.edges):
                execution_graph = NavigationGraph(
                    nodes=execution_graph.nodes,
                    edges=recovery_edges,
                    coverage_targets=execution_graph.coverage_targets,
                    monster_surface_ids=execution_graph.monster_surface_ids,
                    safe_firing_targets=execution_graph.safe_firing_targets,
                    safe_covered_monster_surface_ids=(
                        execution_graph.safe_covered_monster_surface_ids
                    ),
                )
        target_nodes = self._candidate_target_nodes(
            target_platform,
            specific_node_id=specific_node_id,
        )
        result = shortest_path_to_any(
            execution_graph,
            source_node_id,
            target_nodes,
            excluded_actions=self.excluded_actions,
        )
        if result is None:
            return None
        _, target_node_id, edge_ids = result
        if not edge_ids:
            return None
        source_platform = self._surface_number.get(self._last_surface_id)
        label = (
            f"{purpose.value} "
            f"P{source_platform if source_platform is not None else '?'}"
            f"->P{target_platform}"
        )
        return PlatformPath(
            graph=execution_graph,
            edge_ids=edge_ids,
            purpose=purpose,
            source_platform=source_platform,
            target_platform=target_platform,
            target_node_id=target_node_id,
            label=label,
        )

    def _set_blocked(self, reason: str, timestamp: float) -> None:
        self.phase = PlatformPhase.BLOCKED
        self.blocked_reason = reason
        self._blocked_retry_at = timestamp + self.blocked_retry_seconds
        self._replace_path(None)
        self._log(reason)

    def _plan_to_platform(
        self,
        target_platform: int,
        timestamp: float,
        *,
        purpose: PlatformPathPurpose = PlatformPathPurpose.TRANSIT,
    ) -> None:
        path = self._calculate_path(target_platform, purpose)
        if path is None:
            if self.current_platform == target_platform:
                self._enter_dwell(timestamp)
                return
            self._set_blocked(
                f"No WZ path from current surface to P{target_platform}; "
                f"retry in {self.blocked_retry_seconds:g}s",
                timestamp,
            )
            return
        self.phase = PlatformPhase.TRAVELING
        self.blocked_reason = None
        self._replace_path(path)
        actions = [
            path.graph.edge_by_id[edge_id].action.value
            for edge_id in path.edge_ids
        ]
        self._log(
            f"Planned {path.label}: edges={len(path.edge_ids)}, "
            f"actions={actions}"
        )

    def _patrol_anchor_ids(self, platform_number: int) -> tuple[str, ...]:
        surface_id = self.platforms[platform_number].id

        def on_surface(node_id: str) -> bool:
            node = self.graph.node_by_id.get(node_id)
            return node is not None and node.surface_id == surface_id

        configured = tuple(
            node_id
            for node_id in self.patrol_anchors.get(platform_number, ())
            if on_surface(node_id)
        )
        if configured:
            return configured

        candidates = [
            node_id for node_id in self.graph.safe_firing_targets
            if on_surface(node_id)
        ]
        if not candidates:
            candidates = [
                node_id for node_id in self.graph.coverage_targets
                if on_surface(node_id)
            ]
        if not candidates:
            candidates = [
                node.id for node in self.graph.nodes
                if node.surface_id == surface_id
            ]
        ordered = sorted(
            set(candidates),
            key=lambda node_id: (
                self.graph.node_by_id[node_id].x,
                node_id,
            ),
        )
        if len(ordered) <= 1:
            return tuple(ordered)
        return ordered[0], ordered[-1]

    def _plan_patrol(self, timestamp: float) -> None:
        if not self.patrol_enabled:
            self._replace_path(None)
            return
        platform_number = self.target_platform
        if platform_number is None or self.current_platform != platform_number:
            return
        anchors = self._patrol_anchor_ids(platform_number)
        if not anchors or self._last_point is None:
            self._replace_path(None)
            return

        if self._patrol_target_node_id in anchors and len(anchors) > 1:
            index = anchors.index(self._patrol_target_node_id)
            target_node_id = anchors[(index + 1) % len(anchors)]
        else:
            target_node_id = min(
                anchors,
                key=lambda node_id: (
                    abs(self.graph.node_by_id[node_id].x - self._last_point.x),
                    node_id,
                ),
            )
            if len(anchors) > 1 and abs(
                    self.graph.node_by_id[target_node_id].x
                    - self._last_point.x) <= self.arrival_tolerance_wz:
                target_node_id = max(
                    anchors,
                    key=lambda node_id: (
                        abs(
                            self.graph.node_by_id[node_id].x
                            - self._last_point.x
                        ),
                        node_id,
                    ),
                )

        target = self.graph.node_by_id[target_node_id]
        if abs(target.x - self._last_point.x) <= self.arrival_tolerance_wz:
            self._patrol_target_node_id = target_node_id
            self._replace_path(None)
            return
        path = self._calculate_path(
            platform_number,
            PlatformPathPurpose.PATROL,
            specific_node_id=target_node_id,
        )
        if path is None:
            self._replace_path(None)
            return
        self._patrol_target_node_id = target_node_id
        self._replace_path(path)
        self._log(
            f"Patrol P{platform_number} toward x={target.x:.1f}; "
            f"edges={len(path.edge_ids)}"
        )

    def _enter_dwell(
        self,
        timestamp: float,
        *,
        resumed_elapsed: float = 0.0,
    ) -> None:
        platform_number = self.target_platform
        if platform_number is None:
            return
        self.phase = PlatformPhase.DWELLING
        self.current_platform = platform_number
        self.blocked_reason = None
        self._dwell_started_at = timestamp - max(0.0, resumed_elapsed)
        self._paused_dwell_elapsed = 0.0
        self._last_attackable_at = timestamp
        self._last_combat_observation = None
        self._last_combat_observation_at = None
        self._travel_attackable_since = None
        self._travel_combat_exhausted = False
        self._patrol_target_node_id = None
        self._replace_path(None)
        dwell_action = "patrol/shoot" if self.patrol_enabled else "hold"
        self._log(
            f"Enter P{platform_number}; {dwell_action} for at least "
            f"{self.dwell_seconds:g}s (hard limit "
            f"{self.maximum_dwell_seconds:g}s)"
        )
        self._plan_patrol(timestamp)

    def _choose_initial_state(self, timestamp: float) -> None:
        if self.current_platform in self.sequence:
            self.state_index = self.sequence.index(self.current_platform)
            self._enter_dwell(timestamp)
            return

        candidates: list[tuple[float, int, PlatformPath]] = []
        for index, platform_number in enumerate(self.sequence):
            path = self._calculate_path(
                platform_number,
                PlatformPathPurpose.RECOVERY,
            )
            if path is None:
                continue
            cost = sum(
                path.graph.edge_by_id[edge_id].expected_time_sec
                for edge_id in path.edge_ids
            )
            candidates.append((cost, index, path))
        if not candidates:
            self._set_blocked(
                "No configured platform state is reachable from Hero",
                timestamp,
            )
            return
        _, self.state_index, path = min(
            candidates,
            key=lambda item: (item[0], item[1], item[2].edge_ids),
        )
        self.phase = PlatformPhase.TRAVELING
        self.blocked_reason = None
        self._replace_path(path)
        self._log(
            f"Localized outside patrol states; recover to "
            f"P{self.target_platform}"
        )

    def _advance_state(self, timestamp: float, reason: str) -> None:
        if self.state_index is None:
            self.phase = PlatformPhase.LOCALIZING
            self._replace_path(None)
            return
        previous = self.target_platform
        self.state_index = (self.state_index + 1) % len(self.sequence)
        target = self.target_platform
        self.phase = PlatformPhase.TRAVELING
        self._dwell_started_at = None
        self._paused_dwell_elapsed = 0.0
        self._patrol_target_node_id = None
        self._travel_attackable_since = None
        self._travel_combat_exhausted = False
        self._replace_path(None)
        self._log(f"Advance P{previous}->P{target}: {reason}")
        if self.current_platform == target:
            self._enter_dwell(timestamp)
        elif target is not None:
            self._plan_to_platform(target, timestamp)

    def observe_position(
        self,
        surface: Surface,
        point: Point,
        timestamp: float,
        *,
        grounded: bool,
    ) -> None:
        """Consume a WZ-space Hero fix and update the active path if needed."""
        timestamp = float(timestamp)
        if not grounded:
            return
        if surface.id == self._candidate_surface_id:
            self._candidate_surface_frames += 1
        else:
            self._candidate_surface_id = surface.id
            self._candidate_surface_frames = 1
        if self._candidate_surface_frames < self.stable_surface_frames:
            return

        self._last_surface = surface
        self._last_surface_id = surface.id
        self._last_point = Point(
            min(max(float(point.x), surface.min_x), surface.max_x),
            surface.y_at(min(max(float(point.x), surface.min_x), surface.max_x)),
        )
        self._last_position_at = timestamp
        self.current_platform = self._surface_number.get(surface.id)

        if self.phase is PlatformPhase.SUSPENDED:
            self.resume(timestamp)

        if self.phase is PlatformPhase.LOCALIZING:
            self._choose_initial_state(timestamp)
            return

        target = self.target_platform
        if target is None:
            self.phase = PlatformPhase.LOCALIZING
            return

        if self.phase is PlatformPhase.BLOCKED:
            if timestamp >= self._blocked_retry_at:
                self.phase = PlatformPhase.TRAVELING
                self.blocked_reason = None
                self._plan_to_platform(
                    target,
                    timestamp,
                    purpose=PlatformPathPurpose.RECOVERY,
                )
            return

        if self.phase is PlatformPhase.TRAVELING:
            if self.current_platform == target:
                self._enter_dwell(
                    timestamp,
                    resumed_elapsed=self._paused_dwell_elapsed,
                )
                return
            if self.active_path is None:
                self._plan_to_platform(target, timestamp)
                return
            allowed_surfaces = {
                self.active_path.graph.node_by_id[node_id].surface_id
                for edge_id in self.active_path.edge_ids
                for node_id in (
                    self.active_path.graph.edge_by_id[edge_id].source,
                    self.active_path.graph.edge_by_id[edge_id].target,
                )
            }
            if surface.id not in allowed_surfaces:
                self._log(
                    f"Hero deviated onto {surface.id}; replan to P{target}"
                )
                self._replace_path(None)
                self._plan_to_platform(
                    target,
                    timestamp,
                    purpose=PlatformPathPurpose.RECOVERY,
                )
            return

        if self.phase is PlatformPhase.DWELLING:
            if self.current_platform != target:
                if self._dwell_started_at is not None:
                    self._paused_dwell_elapsed = max(
                        0.0,
                        timestamp - self._dwell_started_at,
                    )
                self._replace_path(None)
                self.phase = PlatformPhase.TRAVELING
                self._log(
                    f"Hero left P{target}; return without advancing state"
                )
                self._plan_to_platform(
                    target,
                    timestamp,
                    purpose=PlatformPathPurpose.RECOVERY,
                )
            elif self.active_path is None:
                self._plan_patrol(timestamp)

    def observe_combat(
        self,
        attackable: bool | None,
        timestamp: float,
    ) -> None:
        """Advance a dwell only after this frame's combat arbitration."""
        if attackable is None:
            return
        timestamp = float(timestamp)
        if self.phase is PlatformPhase.TRAVELING:
            if self._travel_combat_exhausted:
                return
            if attackable:
                if self._travel_attackable_since is None:
                    self._travel_attackable_since = timestamp
                elif timestamp - self._travel_attackable_since >= \
                        self.travel_combat_budget_seconds:
                    self._travel_combat_exhausted = True
                    self._log(
                        "Travel combat budget exhausted; preserve the WZ "
                        "path until the target platform"
                    )
            else:
                self._travel_attackable_since = None
            return
        if self.phase is not PlatformPhase.DWELLING:
            return
        self._last_combat_observation = bool(attackable)
        self._last_combat_observation_at = timestamp
        if attackable:
            self._last_attackable_at = timestamp

        if self._dwell_started_at is None:
            self._dwell_started_at = timestamp
        elapsed = max(0.0, timestamp - self._dwell_started_at)
        if elapsed >= self.maximum_dwell_seconds:
            self._advance_state(
                timestamp,
                "maximum dwell reached; release possible persistent detection",
            )
            return
        quiet_elapsed = (
            math.inf
            if self._last_attackable_at is None
            else timestamp - self._last_attackable_at
        )
        if not attackable and elapsed >= self.dwell_seconds and \
                quiet_elapsed >= self.combat_quiet_seconds:
            self._advance_state(
                timestamp,
                f"dwell={elapsed:.2f}s and combat quiet={quiet_elapsed:.2f}s",
            )

    def route_goal_reached(
        self,
        route_index: int,
        route_count: int,
        timestamp: float,
    ) -> bool:
        """Handle only the final leg; intermediate legs remain deterministic."""
        if self.active_path is None or route_count <= 0 or \
                int(route_index) != int(route_count) - 1:
            return False
        completed = self.active_path
        self._replace_path(None)
        if completed.purpose is PlatformPathPurpose.PATROL:
            self._patrol_target_node_id = completed.target_node_id
            if self.phase is PlatformPhase.DWELLING:
                self._plan_patrol(float(timestamp))
        self._log(f"Completed {completed.label}; verify live platform")
        return True

    def request_replan(self, reason: str, timestamp: float) -> bool:
        """Invalidate a lost temporary path and deterministically calculate again."""
        timestamp = float(timestamp)
        if self.phase in {
                PlatformPhase.LOCALIZING, PlatformPhase.SUSPENDED}:
            return False
        if timestamp - self._last_replan_at < 0.25:
            return False
        self._last_replan_at = timestamp
        self._replace_path(None)
        self._log(f"Replan requested: {reason}")
        target = self.target_platform
        if target is None:
            self.phase = PlatformPhase.LOCALIZING
        elif self.phase is PlatformPhase.DWELLING and \
                self.current_platform == target:
            self._plan_patrol(timestamp)
        else:
            self.phase = PlatformPhase.TRAVELING
            self._plan_to_platform(
                target,
                timestamp,
                purpose=PlatformPathPurpose.RECOVERY,
            )
        return True

    def suspend(self, timestamp: float) -> None:
        if self.phase is PlatformPhase.SUSPENDED:
            return
        timestamp = float(timestamp)
        self._suspended_phase = self.phase
        if self.phase is PlatformPhase.DWELLING and \
                self._dwell_started_at is not None:
            self._paused_dwell_elapsed = max(
                0.0,
                timestamp - self._dwell_started_at,
            )
        self.phase = PlatformPhase.SUSPENDED
        self._replace_path(None)
        self._log("Suspended; dwell clock frozen and active path discarded")

    def resume(self, timestamp: float) -> None:
        if self.phase is not PlatformPhase.SUSPENDED:
            return
        timestamp = float(timestamp)
        target = self.target_platform
        previous_phase = self._suspended_phase
        self._suspended_phase = None
        if self._last_surface_id is None or target is None:
            self.phase = PlatformPhase.LOCALIZING
        elif self.current_platform == target and \
                previous_phase is PlatformPhase.DWELLING:
            self._enter_dwell(
                timestamp,
                resumed_elapsed=self._paused_dwell_elapsed,
            )
        else:
            self.phase = PlatformPhase.TRAVELING
            self._plan_to_platform(
                target,
                timestamp,
                purpose=PlatformPathPurpose.RECOVERY,
            )
        self._log("Resumed from a fresh grounded Hero fix")
