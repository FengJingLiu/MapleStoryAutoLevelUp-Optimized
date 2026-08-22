from pathlib import Path
from copy import deepcopy

import cv2
import pytest
import yaml

from src.navigation.hunt_planner import (
    Action,
    MotionProfile,
    RangedAttackProfile,
    build_navigation_graph,
    forest_floor_platform_patrol_anchors,
    number_forest_floor_platforms,
)
from src.navigation.platform_fsm import (
    PlatformPathPurpose,
    PlatformPatrolStateMachine,
    PlatformPhase,
)
from src.navigation.route_renderer import (
    MINIMUM_WALK_LEG_PIXELS,
    NavigationProjection,
)
from src.navigation.runtime import WzNavigationRuntime
from src.navigation.wz_geometry import Point, load_wz_map


def _forest_floor_navigation():
    export_path = (
        Path(__file__).resolve().parents[2]
        / "SuperHumanMapleStory"
        / "var"
        / "all-maps-with-canvases"
        / "100040110.json"
    )
    if not export_path.is_file():
        pytest.skip("Forest Floor WZ export is not installed")
    wz_map = load_wz_map(export_path)
    projection = NavigationProjection(wz_map, 2.806)
    profile = MotionProfile.from_wz(
        wz_map,
        {
            "walk_speed_px_per_sec": 21.357549228124483,
            "jump_height_px": 13.96834524683104,
            "jump_distance_px": 11.025092940349396,
        },
        world_to_navigation_scale=projection.world_scale,
    )
    graph = build_navigation_graph(
        wz_map,
        profile,
        {},
        ranged_attack=RangedAttackProfile(
            horizontal_range_wz=400 / 2.806,
            vertical_tolerance_wz=60 / (2 * 2.806),
            projectile_height_wz=30,
            projectile_clearance_wz=8,
            origin_margin_wz=15,
        ),
    )
    platforms = number_forest_floor_platforms(wz_map, graph)
    anchors = forest_floor_platform_patrol_anchors(
        wz_map, graph, platforms
    )
    return wz_map, profile, graph, platforms, anchors


def _platform_center(platform):
    x = (platform.min_x + platform.max_x) / 2.0
    return Point(x, platform.y_at(x))


def _route_config():
    return {
        "color_code": {
            "255,0,0": "left none none",
            "0,255,0": "right none none",
            "255,128,0": "left none jump",
            "0,255,128": "right none jump",
            "0,0,255": "none down jump",
            "255,0,255": "none none jump",
            "255,255,0": "none none goal",
            "127,255,255": "none up portal",
            "0,127,255": "none up climb",
        },
        "color_code_up_down": {
            "127,127,127": "none up none",
            "255,255,127": "none down none",
        },
        "rope_climb_target_color": (0, 191, 255),
        "rope_climb_runup_distance": 8,
    }


def _state_machine(sequence=(1, 3, 7, 9, 13, 9, 5, 3)):
    _, profile, graph, platforms, anchors = _forest_floor_navigation()
    return PlatformPatrolStateMachine(
        graph,
        profile,
        platforms,
        tuple(sequence),
        patrol_anchors=anchors,
        dwell_seconds=8.0,
        combat_quiet_seconds=0.8,
        maximum_dwell_seconds=24.0,
        stable_surface_frames=1,
        exclude_portals=True,
    )


def test_platform_fsm_starts_on_detected_platform_and_plans_only_next_target():
    state_machine = _state_machine()
    p1 = state_machine.platforms[1]
    state_machine.observe_position(
        p1, _platform_center(p1), 0.0, grounded=True
    )

    assert state_machine.phase is PlatformPhase.DWELLING
    assert state_machine.current_platform == 1
    assert state_machine.target_platform == 1
    assert state_machine.active_path is not None
    assert state_machine.active_path.purpose is PlatformPathPurpose.PATROL
    assert len(state_machine._patrol_anchor_ids(1)) == 2

    state_machine.observe_combat(False, 8.1)

    assert state_machine.phase is PlatformPhase.TRAVELING
    assert state_machine.target_platform == 3
    assert state_machine.active_path is not None
    actions = [
        state_machine.active_path.graph.edge_by_id[edge_id].action
        for edge_id in state_machine.active_path.edge_ids
    ]
    assert actions[-1] is Action.CLIMB
    assert Action.PORTAL not in actions


def test_upper_right_recovery_uses_rope_instead_of_deep_drop():
    wz_map, profile, graph, platforms, anchors = _forest_floor_navigation()
    state_machine = PlatformPatrolStateMachine(
        graph,
        profile,
        platforms,
        (1, 3, 7, 9, 13, 9, 5, 3),
        patrol_anchors=anchors,
        dwell_seconds=8.0,
        combat_quiet_seconds=0.8,
        maximum_dwell_seconds=24.0,
        stable_surface_frames=1,
        maximum_recovery_drop_height_wz=300.0,
        exclude_portals=True,
    )
    upper_right = next(
        surface for surface in wz_map.surfaces
        if surface.id == "surface:1:50:126:135"
    )
    point = Point(900.0, upper_right.y_at(900.0))

    state_machine.observe_position(upper_right, point, 0.0, grounded=True)

    assert state_machine.phase is PlatformPhase.TRAVELING
    assert state_machine.target_platform == 13
    assert state_machine.active_path is not None
    path = state_machine.active_path
    cross_edges = [
        path.graph.edge_by_id[edge_id]
        for edge_id in path.edge_ids
        if path.graph.node_by_id[path.graph.edge_by_id[edge_id].source].surface_id
        != path.graph.node_by_id[path.graph.edge_by_id[edge_id].target].surface_id
    ]
    assert cross_edges[0].action is Action.CLIMB
    assert all(
        edge.action is not Action.DROP
        or path.graph.node_by_id[edge.target].y
        - path.graph.node_by_id[edge.source].y <= 300.0
        for edge in cross_edges
    )


def test_runtime_bootstrap_builds_graph_without_prerendering_full_route_loop():
    export_path = (
        Path(__file__).resolve().parents[2]
        / "SuperHumanMapleStory"
        / "var"
        / "all-maps-with-canvases"
        / "100040110.json"
    )
    map_path = (
        Path(__file__).resolve().parents[1]
        / "minimaps"
        / "forest_floor"
        / "map.png"
    )
    if not export_path.is_file() or not map_path.is_file():
        pytest.skip("Forest Floor WZ/runtime assets are not installed")
    runtime = WzNavigationRuntime(
        {
            "geometry_dir": str(export_path.parent),
            "map_bindings": {"forest_floor": "100040110"},
            "learn_motion": False,
            "platform_state_machine": {
                "enable": True,
                "maps": {
                    "100040110": {
                        "sequence": [1, 3, 7, 9, 13, 9, 5, 3],
                        "dwell_seconds": 8.0,
                        "maximum_dwell_seconds": 24.0,
                    }
                },
            },
        },
        _route_config(),
        directional_attack_config={"range_x": 400, "range_y": 60},
        selected_map_name="forest_floor",
    )
    image = cv2.imread(str(map_path), cv2.IMREAD_COLOR)

    assert runtime.bootstrap(image) == "100040110"
    assert runtime.active
    assert runtime.platform_state_machine_active
    assert runtime.graph is not None
    assert runtime.routes_rgb == ()
    assert runtime.route_legs == ()
    assert runtime.platform_state_machine is not None
    assert runtime.platform_state_machine.maximum_recovery_drop_height_wz == 300
    assert runtime.summary()["platformState"]["phase"] == "LOCALIZING"


def test_p13_subpixel_patrol_anchor_is_treated_as_arrived():
    project_root = Path(__file__).resolve().parents[1]
    export_path = (
        project_root.parent
        / "SuperHumanMapleStory"
        / "var"
        / "all-maps-with-canvases"
        / "100040110.json"
    )
    map_path = project_root / "minimaps" / "forest_floor" / "map.png"
    if not export_path.is_file() or not map_path.is_file():
        pytest.skip("Forest Floor WZ/runtime assets are not installed")
    with (project_root / "config" / "config_default.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        config = yaml.safe_load(stream)
    navigation_config = deepcopy(config["wz_navigation"])
    navigation_config.update({
        "geometry_dir": str(export_path.parent),
        "candidate_map_ids": ["100040110"],
        "learn_motion": False,
    })
    navigation_config["platform_state_machine"] = {
        "enable": True,
        "maps": {
            "100040110": {
                "sequence": [13],
                "dwell_seconds": 8.0,
                "maximum_dwell_seconds": 24.0,
                "stable_surface_frames": 2,
                "arrival_tolerance_wz": 14.0,
            }
        },
    }
    runtime = WzNavigationRuntime(
        navigation_config,
        config["route"],
        directional_attack_config=config["directional_attack"],
        selected_map_name="forest_floor",
    )
    runtime.bootstrap(cv2.imread(str(map_path), cv2.IMREAD_COLOR))
    state_machine = runtime.platform_state_machine
    assert state_machine is not None
    assert runtime.projection is not None
    p13 = state_machine.platforms[13]
    point = Point(p13.min_x, p13.y_at(p13.min_x))

    state_machine.observe_position(p13, point, 0.0, grounded=True)
    state_machine.observe_position(p13, point, 0.1, grounded=True)

    minimum_renderable_wz = (
        MINIMUM_WALK_LEG_PIXELS / runtime.projection.world_scale[0]
    )
    assert state_machine.arrival_tolerance_wz == pytest.approx(
        minimum_renderable_wz
    )
    assert state_machine.phase is PlatformPhase.DWELLING
    assert state_machine.active_path is None
    assert state_machine.intentional_idle
    assert runtime._sync_platform_path_resources() is False


def test_platform_fsm_temporary_path_uses_recorded_jump_and_rope_x():
    project_root = Path(__file__).resolve().parents[1]
    export_path = (
        project_root.parent
        / "SuperHumanMapleStory"
        / "var"
        / "all-maps-with-canvases"
        / "100040110.json"
    )
    map_path = project_root / "minimaps" / "forest_floor" / "map.png"
    if not export_path.is_file() or not map_path.is_file():
        pytest.skip("Forest Floor WZ/runtime assets are not installed")
    with (project_root / "config" / "config_default.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        config = yaml.safe_load(stream)
    navigation_config = deepcopy(config["wz_navigation"])
    navigation_config.update({
        "geometry_dir": str(export_path.parent),
        "candidate_map_ids": ["100040110"],
        "learn_motion": False,
    })
    navigation_config["platform_state_machine"] = {
        "enable": True,
        "maps": {
            "100040110": {
                "sequence": [1, 3, 7, 9, 13, 9, 5, 3],
                "dwell_seconds": 8.0,
                "maximum_dwell_seconds": 24.0,
                "stable_surface_frames": 2,
            }
        },
    }
    navigation_config["motion"].update({
        "walk_speed_px_per_sec": 21.3575,
        "jump_height_px": 13.9683,
        "jump_distance_px": 11.0251,
    })
    runtime = WzNavigationRuntime(
        navigation_config,
        config["route"],
        directional_attack_config=config["directional_attack"],
        selected_map_name="forest_floor",
        recorded_route_directory=map_path.parent,
    )
    image = cv2.imread(str(map_path), cv2.IMREAD_COLOR)
    runtime.bootstrap(image)
    state_machine = runtime.platform_state_machine
    assert state_machine is not None
    p3 = state_machine.platforms[3]
    p3_center = _platform_center(p3)
    state_machine.observe_position(p3, p3_center, 0.0, grounded=True)
    state_machine.observe_position(p3, p3_center, 0.1, grounded=True)
    state_machine.observe_combat(False, 9.0)

    assert state_machine.target_platform == 7
    assert runtime._sync_platform_path_resources()
    assert any(
        leg.action is Action.JUMP and leg.recorded_x_anchor == 142
        for leg in runtime.route_legs
    )
    recorded_climb = next(
        leg for leg in runtime.route_legs
        if leg.action is Action.CLIMB
    )
    assert recorded_climb.recorded_x_anchor == 220
    assert recorded_climb.rope_mount.contact[0] == 212
    assert recorded_climb.rope_mount.approach_direction == "left"


def test_platform_fsm_hard_limit_prevents_permanent_false_positive_hold():
    state_machine = _state_machine()
    p1 = state_machine.platforms[1]
    state_machine.observe_position(
        p1, _platform_center(p1), 0.0, grounded=True
    )

    state_machine.observe_combat(True, 12.0)
    assert state_machine.target_platform == 1
    state_machine.observe_combat(True, 24.1)

    assert state_machine.target_platform == 3
    assert state_machine.phase is PlatformPhase.TRAVELING


def test_platform_fsm_bounds_combat_starvation_while_traveling():
    state_machine = _state_machine()
    p1 = state_machine.platforms[1]
    state_machine.observe_position(
        p1, _platform_center(p1), 0.0, grounded=True
    )
    state_machine.observe_combat(False, 8.1)
    assert state_machine.phase is PlatformPhase.TRAVELING

    state_machine.observe_combat(True, 9.0)
    state_machine.observe_combat(True, 15.1)
    assert not state_machine.combat_priority
    state_machine.observe_combat(False, 15.2)
    assert not state_machine.combat_priority

    p3 = state_machine.platforms[3]
    state_machine.observe_position(
        p3, _platform_center(p3), 16.0, grounded=True
    )
    assert state_machine.phase is PlatformPhase.DWELLING
    assert state_machine.combat_priority


def test_platform_fsm_suspension_freezes_dwell_clock_and_relocalizes():
    state_machine = _state_machine(sequence=(3, 7))
    p3 = state_machine.platforms[3]
    point = _platform_center(p3)
    state_machine.observe_position(p3, point, 0.0, grounded=True)
    state_machine.suspend(3.0)

    assert state_machine.phase is PlatformPhase.SUSPENDED
    state_machine.observe_position(p3, point, 100.0, grounded=True)
    assert state_machine.phase is PlatformPhase.DWELLING
    assert state_machine.snapshot(100.0).dwell_elapsed_seconds == \
        pytest.approx(3.0)

    state_machine.observe_combat(False, 104.9)
    assert state_machine.target_platform == 3
    state_machine.observe_combat(False, 105.1)
    assert state_machine.target_platform == 7


def test_platform_only_p4_to_p5_uses_live_detour_without_portal():
    state_machine = _state_machine(sequence=(4, 5, 6))
    p4 = state_machine.platforms[4]
    state_machine.observe_position(
        p4, _platform_center(p4), 0.0, grounded=True
    )
    state_machine.observe_combat(False, 8.1)

    assert state_machine.phase is PlatformPhase.TRAVELING
    assert state_machine.target_platform == 5
    assert state_machine.active_path is not None
    actions = [
        state_machine.active_path.graph.edge_by_id[edge_id].action
        for edge_id in state_machine.active_path.edge_ids
    ]
    assert Action.PORTAL not in actions
    assert any(action in {Action.JUMP, Action.CLIMB, Action.DROP} for action in actions)
