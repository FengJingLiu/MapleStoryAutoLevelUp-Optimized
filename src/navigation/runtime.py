"""Runtime coordinator for automatic WZ recognition and generated routes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .hunt_planner import (
    Action,
    MotionProfile,
    NavigationGraph,
    PatrolPlan,
    RangedAttackProfile,
    build_forest_floor_patrol_plan,
    build_navigation_graph,
    build_patrol_plan,
    forest_floor_platform_patrol_anchors,
    number_forest_floor_platforms,
)
from .motion_estimator import HeroMotionEstimator, MotionObservation
from .platform_fsm import PlatformPatrolStateMachine
from .recorded_route_anchors import (
    ProjectedRecordedRouteAnchors,
    RecordedRouteAnchors,
    load_recorded_route_anchors,
    prefer_recorded_climb_edges,
)
from .route_renderer import (
    MINIMUM_WALK_LEG_PIXELS,
    NavigationProjection,
    RenderedRouteLeg,
    RopeMountMotion,
    render_navigation_assets,
)
from .wz_catalog import (
    CanvasFeatures,
    CanvasRegistration,
    WzMapCatalog,
    WzMapRecognitionError,
)
from .wz_geometry import Point, WzMap, load_wz_map


@dataclass(frozen=True, slots=True)
class NavigationUpdate:
    map_id: str
    player_navigation: tuple[float, float]
    player_navigation_raw: tuple[float, float]
    registration: CanvasRegistration
    resource_generation: int
    motion_observation: MotionObservation


class WzNavigationRuntime:
    """Recognize a WZ map, project Hero, and expose generated route images."""

    def __init__(
        self,
        config: dict[str, Any],
        route_config: dict[str, Any],
        *,
        directional_attack_config: dict[str, Any] | None = None,
        selected_map_name: str | None = None,
        recorded_route_directory: str | Path | None = None,
        log: Callable[[str], None] | None = None,
    ):
        if not isinstance(config, dict) or not isinstance(route_config, dict):
            raise ValueError("WZ navigation and route configuration must be mappings")
        self.config = config
        self.route_config = route_config
        self.directional_attack_config = directional_attack_config
        self.recorded_route_directory = (
            None
            if recorded_route_directory is None
            else Path(recorded_route_directory)
        )
        self.log = log
        geometry_directory = Path(str(config.get("geometry_dir", "")))
        if not str(geometry_directory) or str(geometry_directory) == ".":
            raise ValueError("wz_navigation.geometry_dir is required")
        matcher = config.get("matcher", {})
        if not isinstance(matcher, dict):
            raise ValueError("wz_navigation.matcher must be a mapping")
        self.catalog = WzMapCatalog(
            geometry_directory,
            minimum_inliers=int(matcher.get("minimum_inliers", 6)),
            minimum_inlier_ratio=float(
                matcher.get("minimum_inlier_ratio", 0.55)
            ),
            minimum_inlier_gap=int(matcher.get("minimum_inlier_gap", 2)),
            descriptor_ratio=float(matcher.get("descriptor_ratio", 0.75)),
            minimum_scale=float(matcher.get("minimum_scale", 0.5)),
            maximum_scale=float(matcher.get("maximum_scale", 8.0)),
            maximum_axis_ratio=float(matcher.get("maximum_axis_ratio", 1.2)),
            maximum_rotation_degrees=float(
                matcher.get("maximum_rotation_degrees", 3.0)
            ),
            maximum_shear_cosine=float(
                matcher.get("maximum_shear_cosine", 0.12)
            ),
            progress=self._log,
        )
        configured_ids = config.get("candidate_map_ids", ())
        if configured_ids is None:
            configured_ids = ()
        if isinstance(configured_ids, (str, bytes)) or not isinstance(
            configured_ids, (list, tuple, set)
        ):
            raise ValueError("wz_navigation.candidate_map_ids must be a list")
        self.candidate_map_ids = {
            str(value).zfill(9) for value in configured_ids
        } or None
        configured_bindings = config.get("map_bindings", {})
        if not isinstance(configured_bindings, dict):
            raise ValueError("wz_navigation.map_bindings must be a mapping")
        self.map_bindings: dict[str, str] = {}
        for configured_name, configured_map_id in configured_bindings.items():
            if not isinstance(configured_name, str) or not configured_name.strip():
                raise ValueError(
                    "wz_navigation.map_bindings keys must be map names"
                )
            map_id = str(configured_map_id).strip()
            if isinstance(configured_map_id, bool) or not map_id.isdigit() or \
                    len(map_id) > 9:
                raise ValueError(
                    "wz_navigation.map_bindings values must be WZ map IDs"
                )
            self.map_bindings[configured_name.strip()] = map_id.zfill(9)

        bound_map_id = self.map_bindings.get(selected_map_name or "")
        self.bound_map_id = bound_map_id
        if bound_map_id is not None:
            self.candidate_map_ids = {bound_map_id}
            self._log(
                f"Using configured map binding: {selected_map_name} -> "
                f"{bound_map_id}"
            )
        self.registration_failure_limit = max(
            1, int(config.get("registration_failure_limit", 3))
        )
        self.recognition_retry_interval = max(
            1.0, float(config.get("recognition_retry_interval", 10.0))
        )
        self.registration_failures = 0
        self._has_live_registration = False
        self._next_recognition_at = 0.0
        self.wz_map: WzMap | None = None
        self.features: CanvasFeatures | None = None
        self.registration: CanvasRegistration | None = None
        self.projection: NavigationProjection | None = None
        self.motion_profile: MotionProfile | None = None
        self.ranged_attack_profile: RangedAttackProfile | None = None
        self.graph: NavigationGraph | None = None
        self.plan: PatrolPlan | None = None
        self.platform_state_machine: PlatformPatrolStateMachine | None = None
        self._rendered_platform_path_revision = -1
        self.map_bgr: np.ndarray | None = None
        self.routes_rgb: tuple[np.ndarray, ...] = ()
        self.route_legs: tuple[RenderedRouteLeg, ...] = ()
        self.recorded_route_anchors = RecordedRouteAnchors()
        self._projected_recorded_route_anchors = \
            ProjectedRecordedRouteAnchors()
        self._recorded_climb_edges = {}
        self.geometry_overlay_bgr: np.ndarray | None = None
        self.resource_generation = 0
        self.motion_estimator = HeroMotionEstimator(
            minimum_walk_samples=int(config.get("minimum_walk_samples", 12)),
            minimum_jump_samples=int(config.get("minimum_jump_samples", 2)),
        )
        self._used_observation = MotionObservation(None, None, None, 0, 0)
        if not isinstance(config.get("motion", {}), dict):
            raise ValueError("wz_navigation.motion must be a mapping")
        ranged_safe_platforms = config.get(
            "projectile_terrain_check", False
        ) is True
        if ranged_safe_platforms and not isinstance(
                directional_attack_config, dict):
            raise ValueError(
                "projectile terrain navigation requires directional attack "
                "config"
            )
        self._configured_patrol_strategy = (
            "ranged_safe_platforms"
            if ranged_safe_platforms
            else "spawn_sweep"
        )
        self.patrol_strategy = self._configured_patrol_strategy

    def _log(self, message: str) -> None:
        if self.log is not None:
            self.log(f"[wz-navigation] {message}")

    @property
    def active(self) -> bool:
        return self.wz_map is not None and self.registration is not None and \
            self.projection is not None and self.graph is not None and (
                self.platform_state_machine is not None
                or bool(self.routes_rgb)
            )

    @property
    def platform_state_machine_active(self) -> bool:
        return self.platform_state_machine is not None

    @property
    def platform_intentional_idle(self) -> bool:
        state_machine = self.platform_state_machine
        return bool(
            state_machine is not None and state_machine.intentional_idle
        )

    @property
    def platform_clear_before_move(self) -> bool:
        state_machine = self.platform_state_machine
        return bool(
            state_machine is not None and state_machine.clear_before_move
        )

    @property
    def platform_combat_priority(self) -> bool:
        state_machine = self.platform_state_machine
        return bool(state_machine is not None and state_machine.combat_priority)

    @property
    def platform_safe_to_buff(self) -> bool:
        state_machine = self.platform_state_machine
        return bool(state_machine is not None and state_machine.safe_to_buff)

    @property
    def map_id(self) -> str | None:
        return None if self.wz_map is None else self.wz_map.map_id

    @property
    def jump_active(self) -> bool:
        return self.motion_estimator.jump_active

    @staticmethod
    def _point_to_segment_distance(
        point: tuple[float, float],
        source: tuple[int, int],
        target: tuple[int, int],
    ) -> float:
        point_vector = np.asarray(point, dtype=np.float64)
        source_vector = np.asarray(source, dtype=np.float64)
        segment = np.asarray(target, dtype=np.float64) - source_vector
        length_squared = float(segment @ segment)
        if length_squared <= 0.0:
            return float(np.linalg.norm(point_vector - source_vector))
        ratio = float((point_vector - source_vector) @ segment) / length_squared
        closest = source_vector + min(1.0, max(0.0, ratio)) * segment
        return float(np.linalg.norm(point_vector - closest))

    def nearest_route_index(
        self, player_navigation: tuple[float, float]
    ) -> int | None:
        """Select the executable route geometry nearest to Hero."""
        if not self.route_legs:
            return None
        distances = []
        for index, leg in enumerate(self.route_legs):
            if leg.action is Action.WALK:
                distance = self._point_to_segment_distance(
                    player_navigation, leg.source, leg.target
                )
            elif leg.jump_source is not None and \
                    leg.source != leg.jump_source:
                distance = self._point_to_segment_distance(
                    player_navigation, leg.source, leg.jump_source
                )
            else:
                distance = float(np.linalg.norm(
                    np.asarray(player_navigation, dtype=np.float64)
                    - np.asarray(leg.source, dtype=np.float64)
                ))
            distances.append(
                (distance, leg.recovery_path is not None, index)
            )
        return min(distances)[2]

    def walk_target_crossed(
        self,
        route_index: int,
        player_navigation: tuple[float, float],
        *,
        tolerance: int,
    ) -> bool:
        """Recognize a horizontal WALK endpoint after a one-frame overshoot."""
        if not (0 <= int(route_index) < len(self.route_legs)):
            return False
        leg = self.route_legs[int(route_index)]
        if leg.action is not Action.WALK:
            return False
        source_x, _ = leg.source
        target_x, target_y = leg.target
        player_x, player_y = player_navigation
        tolerance = max(0, int(tolerance))
        if abs(float(player_y) - target_y) > tolerance:
            return False
        if target_x > source_x:
            return target_x <= float(player_x) <= target_x + tolerance
        if target_x < source_x:
            return target_x - tolerance <= float(player_x) <= target_x
        return False

    def combat_checkpoint_reached(
        self,
        route_index: int,
        player_navigation: tuple[float, float],
        *,
        braking_distance: int = 0,
    ) -> bool:
        """Recognize an intermediate checkpoint before a compound jump."""
        if not (0 <= int(route_index) < len(self.route_legs)):
            return False
        leg = self.route_legs[int(route_index)]
        checkpoint = leg.combat_checkpoint
        position = leg.combat_checkpoint_position
        trigger_bounds = leg.jump_trigger_bounds
        jump_source = leg.jump_source
        if checkpoint is None or position is None or \
                trigger_bounds is None or jump_source is None:
            return False

        player_x, player_y = map(float, player_navigation)
        checkpoint_x, checkpoint_y = position
        left, top, right, bottom = trigger_bounds
        vertical_tolerance = max(
            2.0,
            float(max(abs(top - checkpoint_y), abs(bottom - checkpoint_y))),
        )
        if abs(player_y - float(checkpoint_y)) > vertical_tolerance:
            return False

        braking_distance = max(0, int(braking_distance))
        if leg.target[0] > jump_source[0]:
            return (
                float(checkpoint_x - braking_distance)
                <= player_x <= float(right)
            )
        if leg.target[0] < jump_source[0]:
            return (
                float(left)
                <= player_x <= float(checkpoint_x + braking_distance)
            )
        return False

    def _fallback_plan(self, graph: NavigationGraph) -> PatrolPlan:
        walk_edges = tuple(
            edge.id for edge in graph.edges if edge.action is Action.WALK
        )
        return PatrolPlan(walk_edges, graph.coverage_targets, ())

    def _ranged_attack_profile(
        self, motion_profile: MotionProfile
    ) -> RangedAttackProfile | None:
        if self.patrol_strategy != "ranged_safe_platforms":
            return None
        if self.registration is None:
            raise RuntimeError("ranged patrol requires an active registration")
        attack_config = self.directional_attack_config or {}
        try:
            range_x = float(attack_config["range_x"])
            range_y = float(attack_config["range_y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "directional attack range_x/range_y must be numeric"
            ) from exc
        if not all(
                np.isfinite(value) and value > 0
                for value in (range_x, range_y)):
            raise ValueError(
                "directional attack range_x/range_y must be positive"
            )
        return RangedAttackProfile(
            horizontal_range_wz=range_x / self.registration.scale_x,
            vertical_tolerance_wz=(
                range_y / (2.0 * self.registration.scale_y)
            ),
            projectile_height_wz=float(
                self.config.get("projectile_height_wz", 30.0)
            ),
            projectile_clearance_wz=float(
                self.config.get("projectile_clearance_wz", 8.0)
            ),
            origin_margin_wz=motion_profile.character_half_width_wz,
        )

    def _patrol_strategy_for_map(self, map_id: str) -> str:
        """Keep Forest Floor's numbered loop independent of attack filters."""
        if str(map_id) == "100040110":
            return "ranged_safe_platforms"
        return self._configured_patrol_strategy

    def _platform_state_config_for_map(
        self, map_id: str
    ) -> dict[str, Any] | None:
        section = self.config.get("platform_state_machine", {})
        if section is None:
            return None
        if not isinstance(section, dict):
            raise ValueError(
                "wz_navigation.platform_state_machine must be a mapping"
            )
        enabled = section.get("enable", False)
        if not isinstance(enabled, bool):
            raise ValueError(
                "wz_navigation.platform_state_machine.enable must be boolean"
            )
        if not enabled:
            return None
        maps = section.get("maps", {})
        if not isinstance(maps, dict):
            raise ValueError(
                "wz_navigation.platform_state_machine.maps must be a mapping"
            )
        map_config = maps.get(str(map_id))
        if map_config is None:
            map_config = maps.get(int(map_id)) if str(map_id).isdigit() else None
        if map_config is None:
            return None
        if isinstance(map_config, (list, tuple)):
            map_config = {"sequence": map_config}
        if not isinstance(map_config, dict):
            raise ValueError(
                f"platform state configuration for {map_id} must be a mapping"
            )
        return {**section, **map_config}

    @staticmethod
    def _platform_sequence(config: dict[str, Any]) -> tuple[int, ...]:
        raw_sequence = config.get("sequence", ())
        if isinstance(raw_sequence, (str, bytes)) or not isinstance(
                raw_sequence, (list, tuple)):
            raise ValueError("platform state machine sequence must be a list")
        sequence: list[int] = []
        for raw_value in raw_sequence:
            value = raw_value
            if isinstance(value, str):
                value = value.strip().upper()
                if value.startswith("P"):
                    value = value[1:]
            if isinstance(value, bool):
                raise ValueError(
                    "platform sequence values must be positive platform numbers"
                )
            try:
                number = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid platform sequence value: {raw_value!r}"
                ) from exc
            if number <= 0:
                raise ValueError(
                    "platform sequence values must be positive platform numbers"
                )
            sequence.append(number)
        if not sequence:
            raise ValueError("platform state machine sequence cannot be empty")
        return tuple(sequence)

    def _build_platform_state_machine(
        self,
        graph: NavigationGraph,
        profile: MotionProfile,
        map_config: dict[str, Any],
    ) -> PlatformPatrolStateMachine:
        if self.wz_map is None:
            raise RuntimeError("platform state machine requires a WZ map")
        if self.wz_map.map_id != "100040110":
            raise ValueError(
                "numbered platform state machine currently supports "
                "Forest Floor WZ map 100040110"
            )
        platforms = number_forest_floor_platforms(self.wz_map, graph)
        patrol_anchors = forest_floor_platform_patrol_anchors(
            self.wz_map, graph, platforms
        )
        sequence = self._platform_sequence(map_config)
        dwell_seconds = float(map_config.get("dwell_seconds", 8.0))
        quiet_seconds = float(map_config.get(
            "combat_quiet_seconds",
            self.config.get("combat_clear_quiet_seconds", 0.8),
        ))
        maximum_dwell_seconds = float(map_config.get(
            "maximum_dwell_seconds",
            max(20.0, dwell_seconds * 2.5),
        ))
        configured_arrival_tolerance = float(
            map_config.get("arrival_tolerance_wz", 14.0)
        )
        if self.projection is None:
            raise RuntimeError(
                "platform state machine requires an active projection"
            )
        navigation_scale_x = abs(float(self.projection.world_scale[0]))
        renderable_arrival_tolerance = (
            configured_arrival_tolerance
            if navigation_scale_x <= 0
            else MINIMUM_WALK_LEG_PIXELS / navigation_scale_x
        )
        arrival_tolerance = max(
            configured_arrival_tolerance,
            renderable_arrival_tolerance,
        )
        maximum_recovery_drop_height = float(
            map_config.get("maximum_recovery_drop_height_wz", 300.0)
        )
        existing = self.platform_state_machine
        if existing is not None and existing.sequence == sequence:
            existing.dwell_seconds = dwell_seconds
            existing.combat_quiet_seconds = quiet_seconds
            existing.maximum_dwell_seconds = maximum_dwell_seconds
            existing.travel_combat_budget_seconds = float(
                map_config.get("travel_combat_budget_seconds", 6.0)
            )
            existing.stable_surface_frames = max(
                1, int(map_config.get("stable_surface_frames", 2))
            )
            existing.arrival_tolerance_wz = arrival_tolerance
            existing.blocked_retry_seconds = float(
                map_config.get("blocked_retry_seconds", 1.0)
            )
            existing.maximum_recovery_drop_height_wz = (
                maximum_recovery_drop_height
            )
            existing.excluded_actions = (
                frozenset({Action.PORTAL})
                if map_config.get("exclude_portals", True) is True
                else frozenset()
            )
            existing.rebind_graph(
                graph,
                profile,
                platforms,
                patrol_anchors=patrol_anchors,
            )
            return existing
        return PlatformPatrolStateMachine(
            graph,
            profile,
            platforms,
            sequence,
            patrol_anchors=patrol_anchors,
            dwell_seconds=dwell_seconds,
            combat_quiet_seconds=quiet_seconds,
            maximum_dwell_seconds=maximum_dwell_seconds,
            travel_combat_budget_seconds=float(
                map_config.get("travel_combat_budget_seconds", 6.0)
            ),
            stable_surface_frames=int(
                map_config.get("stable_surface_frames", 2)
            ),
            arrival_tolerance_wz=arrival_tolerance,
            blocked_retry_seconds=float(
                map_config.get("blocked_retry_seconds", 1.0)
            ),
            maximum_recovery_drop_height_wz=maximum_recovery_drop_height,
            exclude_portals=(map_config.get("exclude_portals", True) is True),
            log=self._log,
        )

    def _rope_mount_motion(
        self,
        motion_profile: MotionProfile,
        observation: MotionObservation,
    ) -> RopeMountMotion:
        """Return unsmoothed, actual pixel motion for rope-jump timing."""
        if self.projection is None:
            raise RuntimeError("rope mount motion requires an active projection")
        motion_config = self.config.get("motion", {})
        scale_x, scale_y = self.projection.world_scale

        def actual_value(observed, configured_name, fallback):
            if observed is not None and float(observed) > 0:
                return float(observed)
            configured = float(
                motion_config.get(configured_name, 0.0) or 0.0
            )
            return configured if configured > 0 else float(fallback)

        theoretical_height = (
            motion_profile.jump_speed_wz_per_sec ** 2
            / (2.0 * motion_profile.gravity_wz_per_sec2)
            * scale_y
        )
        theoretical_distance = (
            motion_profile.air_speed_wz_per_sec
            * (2.0 * motion_profile.jump_speed_wz_per_sec)
            / motion_profile.gravity_wz_per_sec2
            * scale_x
        )
        return RopeMountMotion(
            walk_speed_px_per_sec=actual_value(
                observation.walk_speed_px_per_sec,
                "walk_speed_px_per_sec",
                motion_profile.walk_speed_wz_per_sec * scale_x,
            ),
            jump_height_px=actual_value(
                observation.jump_height_px,
                "jump_height_px",
                theoretical_height,
            ),
            jump_distance_px=actual_value(
                observation.jump_distance_px,
                "jump_distance_px",
                theoretical_distance,
            ),
            runup_seconds=max(
                0.0,
                float(self.route_config.get("rope_climb_runup_ms", 180))
                / 1000.0,
            ),
        )

    def _build_assets(
        self,
        *,
        observation: MotionObservation | None = None,
    ) -> None:
        if self.wz_map is None or self.projection is None:
            raise RuntimeError("cannot build navigation assets before map activation")
        observation = observation or MotionObservation(None, None, None, 0, 0)
        self.patrol_strategy = self._patrol_strategy_for_map(
            self.wz_map.map_id
        )
        motion_config = self.config.get("motion", {})
        profile = MotionProfile.from_wz(
            self.wz_map,
            motion_config,
            world_to_navigation_scale=self.projection.world_scale,
            observed_walk_speed_px_per_sec=observation.walk_speed_px_per_sec,
            observed_jump_height_px=observation.jump_height_px,
            observed_jump_distance_px=observation.jump_distance_px,
        )
        ranged_attack_profile = self._ranged_attack_profile(profile)
        graph = build_navigation_graph(
            self.wz_map,
            profile,
            self.config,
            ranged_attack=ranged_attack_profile,
        )
        projected_anchors = self.recorded_route_anchors.project(
            self.projection
        )
        recorded_climb_edges = projected_anchors.match_climb_edges(
            graph, self.projection
        )
        graph = prefer_recorded_climb_edges(
            graph, recorded_climb_edges
        )
        self._projected_recorded_route_anchors = projected_anchors
        self._recorded_climb_edges = recorded_climb_edges
        platform_config = self._platform_state_config_for_map(
            self.wz_map.map_id
        )
        if platform_config is not None:
            state_machine = self._build_platform_state_machine(
                graph,
                profile,
                platform_config,
            )
            plan = PatrolPlan((), (), ())
        else:
            self.platform_state_machine = None
            state_machine = None
            if self.wz_map.map_id == "100040110":
                plan = build_forest_floor_patrol_plan(self.wz_map, graph)
            else:
                plan = build_patrol_plan(graph)
            if not plan.edge_ids:
                plan = self._fallback_plan(graph)
        rope_mount_motion = self._rope_mount_motion(profile, observation)
        map_bgr, routes, overlay, route_legs = render_navigation_assets(
            self.wz_map,
            graph,
            plan,
            self.projection,
            self.route_config,
            rope_mount_motion,
            projected_anchors,
            recorded_climb_edges,
        )
        if not routes and state_machine is None:
            raise ValueError(
                f"WZ map {self.wz_map.map_id} produced no executable routes"
            )
        self.motion_profile = profile
        self.ranged_attack_profile = ranged_attack_profile
        self.graph = graph
        self.plan = plan
        self.platform_state_machine = state_machine
        self._rendered_platform_path_revision = (
            -1 if state_machine is None else state_machine.path_revision
        )
        self.map_bgr = map_bgr
        self.routes_rgb = routes
        self.route_legs = route_legs
        self.geometry_overlay_bgr = overlay
        self._used_observation = observation
        self.resource_generation += 1
        action_counts = {
            action.value: sum(1 for edge in graph.edges if edge.action is action)
            for action in Action
        }
        self._log(
            f"Built {len(graph.nodes)} nodes, {len(graph.edges)} edges, "
            f"{len(routes)} active legs from {len(plan.edge_ids)} planned "
            f"edges plus {len(plan.recovery_edge_paths)} recovery paths; "
            f"strategy={self.patrol_strategy}; "
            f"platform-fsm={state_machine is not None}; "
            f"combat-checkpoints={len(plan.combat_checkpoints)}; "
            f"safe-firing-targets={len(graph.safe_firing_targets)}, "
            f"monster-platforms={len(graph.monster_surface_ids)}; "
            f"actions={action_counts}; "
            f"motion={profile.source}"
        )
        if projected_anchors.jumps or projected_anchors.climbs:
            applied_jump_legs = sum(
                1 for leg in route_legs
                if leg.action is Action.JUMP
                and leg.recorded_x_anchor is not None
            )
            applied_climb_legs = sum(
                1 for leg in route_legs
                if leg.action is Action.CLIMB
                and leg.recorded_x_anchor is not None
            )
            self._log(
                "Recorded route X anchors: "
                f"jumps={len(projected_anchors.jumps)} "
                f"(applied legs={applied_jump_legs}), "
                f"climbs={len(projected_anchors.climbs)} "
                f"(matched ropes={len(recorded_climb_edges)}, "
                f"applied legs={applied_climb_legs})"
            )
        if state_machine is not None:
            self._log(
                "Platform state machine ready without a pre-rendered route "
                f"loop: sequence={list(state_machine.sequence)}, "
                f"dwell={state_machine.dwell_seconds:g}s"
            )

    def _sync_platform_path_resources(self) -> bool:
        """Render only the temporary path selected by the platform FSM."""
        state_machine = self.platform_state_machine
        if state_machine is None:
            return False
        if state_machine.path_revision == \
                self._rendered_platform_path_revision:
            return False
        if self.wz_map is None or self.projection is None or \
                self.motion_profile is None:
            raise RuntimeError("platform path rendering requires an active map")

        active_path = state_machine.active_path
        if active_path is None:
            self.plan = PatrolPlan((), (), ())
            self.routes_rgb = ()
            self.route_legs = ()
            self._rendered_platform_path_revision = \
                state_machine.path_revision
            self.resource_generation += 1
            self._log(
                f"Platform state {state_machine.phase.value}: no movement "
                "path; combat/patrol timer owns the frame"
            )
            return True

        plan = PatrolPlan(
            edge_ids=active_path.edge_ids,
            visited_targets=(active_path.target_node_id,),
            unreachable_targets=(),
        )
        rope_mount_motion = self._rope_mount_motion(
            self.motion_profile,
            self._used_observation,
        )
        map_bgr, routes, overlay, route_legs = render_navigation_assets(
            self.wz_map,
            active_path.graph,
            plan,
            self.projection,
            self.route_config,
            rope_mount_motion,
            self._projected_recorded_route_anchors,
            self._recorded_climb_edges,
        )
        if not routes:
            raise ValueError(
                f"temporary platform path {active_path.label} rendered no legs"
            )
        self.plan = plan
        self.map_bgr = map_bgr
        self.routes_rgb = routes
        self.route_legs = route_legs
        self.geometry_overlay_bgr = overlay
        self._rendered_platform_path_revision = state_machine.path_revision
        self.resource_generation += 1
        self._log(
            f"Rendered temporary {active_path.label}: "
            f"{len(active_path.edge_ids)} graph edges -> "
            f"{len(routes)} executable legs"
        )
        return True

    def _activate(
        self,
        registration: CanvasRegistration,
        features: CanvasFeatures,
        recorded_route_reference_bgr: np.ndarray | None = None,
    ) -> None:
        wz_map = load_wz_map(
            registration.entry.geometry_path,
            canvas_path=registration.entry.canvas_path,
        )
        self.wz_map = wz_map
        self.features = features
        self.registration = registration
        self.projection = NavigationProjection(
            wz_map=wz_map,
            canvas_scale=registration.effective_scale,
        )
        self.recorded_route_anchors = RecordedRouteAnchors()
        if wz_map.map_id == "100040110" and \
                recorded_route_reference_bgr is not None:
            self.recorded_route_anchors = load_recorded_route_anchors(
                self.recorded_route_directory,
                recorded_route_reference_bgr,
                self.route_config,
                registration,
            )
            if self.recorded_route_anchors.route_files:
                self._log(
                    "Loaded recorded route X evidence from "
                    f"{list(self.recorded_route_anchors.route_files)}: "
                    f"jumps={len(self.recorded_route_anchors.jumps)}, "
                    f"climbs={len(self.recorded_route_anchors.climbs)}"
                )
        self.registration_failures = 0
        self._has_live_registration = False
        self.motion_estimator = HeroMotionEstimator(
            minimum_walk_samples=int(self.config.get("minimum_walk_samples", 12)),
            minimum_jump_samples=int(self.config.get("minimum_jump_samples", 2)),
        )
        self._used_observation = MotionObservation(None, None, None, 0, 0)
        try:
            self._build_assets()
        except Exception:
            # Activation is transactional: a malformed map must never leave a
            # selected registration paired with absent or stale route assets.
            self._clear_active_map()
            raise
        self._log(
            f"Activated map {wz_map.map_id}: "
            f"foothold-surfaces={len(wz_map.surfaces)}, "
            f"walls={len(wz_map.walls)}, "
            f"ropes={len(wz_map.ropes)}, portals={len(wz_map.portals)}, "
            f"monster-spawns={len(wz_map.monster_spawns)}"
        )

    def bootstrap(self, minimap_or_stitched_map_bgr: np.ndarray) -> str:
        registration, features = self.catalog.recognize(
            minimap_or_stitched_map_bgr,
            map_ids=self.candidate_map_ids,
        )
        self._activate(
            registration,
            features,
            recorded_route_reference_bgr=minimap_or_stitched_map_bgr,
        )
        self._next_recognition_at = 0.0
        return registration.entry.map_id

    def _clear_active_map(self) -> None:
        previous = self.map_id
        self.catalog.clear_selection()
        self.wz_map = None
        self.features = None
        self.registration = None
        self.projection = None
        self.motion_profile = None
        self.ranged_attack_profile = None
        self.graph = None
        self.plan = None
        self.platform_state_machine = None
        self._rendered_platform_path_revision = -1
        self.map_bgr = None
        self.routes_rgb = ()
        self.route_legs = ()
        self.recorded_route_anchors = RecordedRouteAnchors()
        self._projected_recorded_route_anchors = \
            ProjectedRecordedRouteAnchors()
        self._recorded_climb_edges = {}
        self.geometry_overlay_bgr = None
        self.registration_failures = 0
        self._has_live_registration = False
        self._next_recognition_at = 0.0
        if previous is not None:
            self._log(f"Released map {previous}; waiting for a new WZ match")

    def _recognize_unknown_map(
        self,
        live_minimap_bgr: np.ndarray,
        timestamp: float,
    ) -> bool:
        """Run the expensive catalog scan at a bounded retry cadence."""
        if timestamp < self._next_recognition_at:
            return False
        try:
            registration, features = self.catalog.recognize(
                live_minimap_bgr,
                map_ids=self.candidate_map_ids,
            )
            self._activate(registration, features)
            self._has_live_registration = True
            self._next_recognition_at = 0.0
            return True
        except (OSError, ValueError, WzMapRecognitionError) as exc:
            # Start the cooldown after the scan finishes. A full catalog scan
            # can itself exceed the interval; measuring from the input-frame
            # timestamp would otherwise trigger another scan immediately.
            self._next_recognition_at = (
                time.monotonic() + self.recognition_retry_interval
            )
            self._log(f"Map recognition unavailable: {exc}")
            return False

    def invalidate_live_registration(self, reason: str) -> bool:
        """Request exactly one new live SIFT registration on the next update."""
        if self.wz_map is None or self.features is None:
            return False
        was_registered = self._has_live_registration
        self._has_live_registration = False
        self.registration_failures = 0
        self._log(f"Live registration invalidated: {reason}")
        return was_registered

    @staticmethod
    def _changed_enough(previous: float | None, current: float | None) -> bool:
        if current is None:
            return False
        if previous is None:
            return True
        return abs(current - previous) / max(abs(previous), 1.0) >= 0.08

    def _maybe_rebuild_from_motion(
        self,
        observation: MotionObservation,
        *,
        allow_rebuild: bool,
    ) -> None:
        if not allow_rebuild or not self.config.get("learn_motion", True):
            return
        previous = self._used_observation
        changed = any(
            (
                self._changed_enough(
                    previous.walk_speed_px_per_sec,
                    observation.walk_speed_px_per_sec,
                ),
                self._changed_enough(
                    previous.jump_height_px,
                    observation.jump_height_px,
                ),
                self._changed_enough(
                    previous.jump_distance_px,
                    observation.jump_distance_px,
                ),
            )
        )
        if changed:
            self._log(
                "Rebuilding route graph from observed Hero motion: "
                f"walk={observation.walk_speed_px_per_sec}, "
                f"jumpHeight={observation.jump_height_px}, "
                f"jumpDistance={observation.jump_distance_px}, "
                f"jumpRunup={observation.jump_runup_distance_px}"
            )
            self._build_assets(observation=observation)

    def platform_route_goal_reached(
        self,
        route_index: int,
        *,
        timestamp: float | None = None,
    ) -> bool:
        state_machine = self.platform_state_machine
        if state_machine is None:
            return False
        handled = state_machine.route_goal_reached(
            int(route_index),
            len(self.route_legs),
            time.monotonic() if timestamp is None else float(timestamp),
        )
        if handled:
            self._sync_platform_path_resources()
        return handled

    def request_platform_replan(
        self,
        reason: str,
        *,
        timestamp: float | None = None,
    ) -> bool:
        state_machine = self.platform_state_machine
        if state_machine is None:
            return False
        changed = state_machine.request_replan(
            str(reason),
            time.monotonic() if timestamp is None else float(timestamp),
        )
        if changed:
            self._sync_platform_path_resources()
        return changed

    def observe_platform_combat(
        self,
        attackable: bool | None,
        *,
        timestamp: float | None = None,
    ) -> bool:
        state_machine = self.platform_state_machine
        if state_machine is None:
            return False
        previous_revision = state_machine.path_revision
        state_machine.observe_combat(
            attackable,
            time.monotonic() if timestamp is None else float(timestamp),
        )
        changed = state_machine.path_revision != previous_revision
        if changed:
            self._sync_platform_path_resources()
        return changed

    def suspend_platform_navigation(
        self,
        *,
        timestamp: float | None = None,
    ) -> bool:
        state_machine = self.platform_state_machine
        if state_machine is None:
            return False
        previous_revision = state_machine.path_revision
        state_machine.suspend(
            time.monotonic() if timestamp is None else float(timestamp)
        )
        changed = state_machine.path_revision != previous_revision
        if changed:
            self._sync_platform_path_resources()
        return changed

    def update(
        self,
        live_minimap_bgr: np.ndarray,
        player_live: tuple[int, int] | None,
        *,
        on_ladder: bool,
        previous_command: str | tuple[str, str, str],
        allow_motion_rebuild: bool,
        timestamp: float | None = None,
    ) -> NavigationUpdate | None:
        """Update registration; return a stable navigation-space Hero point."""
        timestamp = time.monotonic() if timestamp is None else float(timestamp)
        if self.wz_map is None:
            if not self._recognize_unknown_map(live_minimap_bgr, timestamp):
                return None
        elif not self._has_live_registration:
            registration = self.catalog.register_selected(live_minimap_bgr)
            if registration is None:
                self.registration_failures += 1
                if self.registration_failures < \
                        self.registration_failure_limit:
                    return None
                if self.bound_map_id is None:
                    self._clear_active_map()
                    # Return first so the engine releases held input. The next
                    # frame performs the potentially expensive all-map scan
                    # while Hero is known to be idle.
                    return None
                if self.registration_failures == \
                        self.registration_failure_limit:
                    self._log(
                        "Live registration unavailable; retaining configured "
                        f"map binding {self.bound_map_id} and input suspension"
                    )
                return None
            else:
                self.registration = registration
                self.registration_failures = 0
                self._has_live_registration = True
                self._log(
                    "Cached live registration for high-rate navigation: "
                    f"{registration.inlier_count}/"
                    f"{registration.good_match_count} inliers, "
                    f"scale={registration.effective_scale:.3f}"
                )

        if player_live is None or self.registration is None or \
                self.projection is None or self.wz_map is None:
            return None
        canonical = self.registration.live_to_canonical(player_live)
        raw_navigation = self.projection.canonical_to_navigation(canonical)
        navigation = raw_navigation
        jump_interval = self.motion_estimator.jump_active or \
            HeroMotionEstimator.is_jump_command(previous_command)
        if not on_ladder and not jump_interval:
            world = self.wz_map.canvas_to_world(Point(*canonical))
            snapped = self.wz_map.nearest_surface(
                world,
                maximum_vertical_distance=float(
                    self.config.get("hero_snap_distance_wz", 120.0)
                ),
            )
            if snapped is not None:
                snapped_pixel = self.projection.world_to_navigation(snapped[1])
                navigation = (raw_navigation[0], float(snapped_pixel[1]))

        width, height = self.projection.size
        navigation = (
            min(max(0.0, navigation[0]), float(width - 1)),
            min(max(0.0, navigation[1]), float(height - 1)),
        )
        observation = self.motion_estimator.observe(
            raw_navigation,
            timestamp,
            previous_command,
            on_ladder=on_ladder,
        )
        self._maybe_rebuild_from_motion(
            observation,
            allow_rebuild=allow_motion_rebuild,
        )
        state_machine = self.platform_state_machine
        if state_machine is not None and self.wz_map is not None and \
                self.projection is not None:
            world = self.projection.navigation_to_world(navigation)
            snapped_surface = self.wz_map.nearest_surface(
                world,
                maximum_vertical_distance=float(
                    self.config.get("hero_snap_distance_wz", 120.0)
                ),
            )
            if snapped_surface is not None:
                state_machine.observe_position(
                    snapped_surface[0],
                    snapped_surface[1],
                    timestamp,
                    grounded=(
                        not on_ladder
                        and not self.motion_estimator.jump_active
                    ),
                )
                self._sync_platform_path_resources()
        return NavigationUpdate(
            map_id=self.wz_map.map_id,
            player_navigation=navigation,
            player_navigation_raw=raw_navigation,
            registration=self.registration,
            resource_generation=self.resource_generation,
            motion_observation=observation,
        )

    def summary(self) -> dict[str, Any]:
        platform_snapshot = (
            None
            if self.platform_state_machine is None
            else self.platform_state_machine.snapshot()
        )
        return {
            "mapId": self.map_id,
            "surfaces": 0 if self.wz_map is None else len(self.wz_map.surfaces),
            "walls": 0 if self.wz_map is None else len(self.wz_map.walls),
            "ropes": 0 if self.wz_map is None else len(self.wz_map.ropes),
            "portals": 0 if self.wz_map is None else len(self.wz_map.portals),
            "monsterSpawns": (
                0 if self.wz_map is None else len(self.wz_map.monster_spawns)
            ),
            "nodes": 0 if self.graph is None else len(self.graph.nodes),
            "edges": 0 if self.graph is None else len(self.graph.edges),
            "routes": len(self.routes_rgb),
            "platformStateMachine": self.platform_state_machine is not None,
            "platformState": None if platform_snapshot is None else {
                "phase": platform_snapshot.phase.value,
                "stateIndex": platform_snapshot.state_index,
                "currentPlatform": platform_snapshot.current_platform,
                "targetPlatform": platform_snapshot.target_platform,
                "sequence": list(platform_snapshot.sequence),
                "dwellElapsedSeconds": (
                    platform_snapshot.dwell_elapsed_seconds
                ),
                "activePath": platform_snapshot.active_path_label,
                "pathRevision": platform_snapshot.path_revision,
                "blockedReason": platform_snapshot.blocked_reason,
            },
            "patrolStrategy": self.patrol_strategy,
            "monsterPlatforms": (
                0 if self.graph is None else len(self.graph.monster_surface_ids)
            ),
            "safeFiringTargets": (
                0 if self.graph is None else len(self.graph.safe_firing_targets)
            ),
            "recoveryPaths": (
                0 if self.plan is None else len(self.plan.recovery_edge_paths)
            ),
            "combatCheckpoints": (
                0 if self.plan is None else len(self.plan.combat_checkpoints)
            ),
            "rangedAttackRangeWz": (
                None if self.ranged_attack_profile is None
                else self.ranged_attack_profile.horizontal_range_wz
            ),
            "motionSource": (
                None if self.motion_profile is None else self.motion_profile.source
            ),
            "motion": None if self.motion_profile is None else {
                "walkSpeedWzPerSec": self.motion_profile.walk_speed_wz_per_sec,
                "jumpHeightWz": self.motion_profile.jump_height_wz,
                "jumpDistanceWz": self.motion_profile.jump_distance_wz,
            },
            "recordedXAnchors": {
                "routeFiles": list(self.recorded_route_anchors.route_files),
                "jumps": len(self.recorded_route_anchors.jumps),
                "climbs": len(self.recorded_route_anchors.climbs),
                "matchedRopes": len(self._recorded_climb_edges),
                "appliedJumpLegs": sum(
                    1 for leg in self.route_legs
                    if leg.action is Action.JUMP
                    and leg.recorded_x_anchor is not None
                ),
                "appliedClimbLegs": sum(
                    1 for leg in self.route_legs
                    if leg.action is Action.CLIMB
                    and leg.recorded_x_anchor is not None
                ),
            },
        }
