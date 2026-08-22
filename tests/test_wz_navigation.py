import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from src.navigation.hunt_planner import (
    Action,
    CombatCheckpoint,
    Edge,
    MotionProfile,
    NavigationGraph,
    Node,
    PatrolPlan,
    RangedAttackProfile,
    build_forest_floor_patrol_plan,
    build_navigation_graph,
    build_patrol_plan,
)
from src.navigation.motion_estimator import HeroMotionEstimator, MotionObservation
from src.navigation.route_renderer import (
    NavigationProjection,
    RenderedRouteLeg,
    RopeMountMotion,
    render_navigation_assets,
)
from src.navigation.runtime import WzNavigationRuntime
from src.navigation.wz_catalog import WzMapCatalog, WzMapRecognitionError
from src.navigation.wz_geometry import Point, load_wz_map


MAP_ID = "100000001"


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


def _write_export(tmp_path: Path) -> tuple[Path, Path, np.ndarray]:
    canvas_dir = tmp_path / "canvases" / MAP_ID
    canvas_dir.mkdir(parents=True)
    rng = np.random.default_rng(20260821)
    canvas = rng.integers(20, 235, (120, 200, 4), dtype=np.uint8)
    canvas[:, :, 3] = 255
    for x in range(10, 195, 20):
        cv2.line(canvas, (x, 5), (200 - x // 2, 115), (20, 20, 20, 255), 2)
    cv2.putText(
        canvas,
        "WZ",
        (55, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    canvas_path = canvas_dir / "canvas-test.png"
    assert cv2.imwrite(str(canvas_path), canvas)

    footholds = [
        {
            "id": "1", "layer": 0, "group": 0,
            "x1": 0, "y1": 100, "x2": 100, "y2": 100,
            "prev": None, "next": "2",
        },
        {
            "id": "2", "layer": 0, "group": 0,
            "x1": 100, "y1": 100, "x2": 200, "y2": 100,
            "prev": "1", "next": None,
        },
        {
            "id": "3", "layer": 0, "group": 1,
            "x1": 0, "y1": 35, "x2": 200, "y2": 35,
            "prev": None, "next": None,
        },
        {
            "id": "4", "layer": 0, "group": 2,
            "x1": 0, "y1": 0, "x2": 0, "y2": 100,
            "prev": None, "next": None,
        },
    ]
    payload = {
        "schemaVersion": "1.0.0",
        "source": {
            "requestedMapId": MAP_ID,
            "geometryMapId": MAP_ID,
        },
        "bounds": {"left": 0, "top": 0, "right": 200, "bottom": 120},
        "minimap": {
            "width": 200,
            "height": 120,
            "centerX": 0,
            "centerY": 0,
            "magnification": 1,
            "canvasVariants": [
                {
                    "stableCanvasVariantId": "canvas:test",
                    "pixelWidth": 200,
                    "pixelHeight": 120,
                    "contentFingerprint": "test",
                    "cacheReference": f"canvases/{MAP_ID}/canvas-test.png",
                }
            ],
        },
        "footholds": footholds,
        "ropes": [
            {
                "id": "r1", "x": 100, "y1": 35, "y2": 100,
                "layer": 0, "ladder": False, "upperFoothold": True,
            }
        ],
        "portals": [
            {
                "id": "p1", "name": "a", "type": 2,
                "x": 20, "y": 100,
                "targetMapId": MAP_ID, "targetName": "b", "script": None,
            },
            {
                "id": "p2", "name": "b", "type": 2,
                "x": 180, "y": 35,
                "targetMapId": MAP_ID, "targetName": "a", "script": None,
            },
        ],
        "lifeSpawns": [
            {"id": "m1", "type": "m", "x": 40, "y": 95},
            {"id": "m2", "type": "m", "x": 160, "y": 30},
        ],
        "clientPhysics": {
            "fields": [
                {"name": "walkSpeed", "value": 125},
                {"name": "gravityAcc", "value": 2000},
                {"name": "fallSpeed", "value": 670},
                {"name": "jumpSpeed", "value": 555},
            ]
        },
    }
    geometry_path = tmp_path / f"{MAP_ID}.json"
    geometry_path.write_text(json.dumps(payload), encoding="utf-8")
    return geometry_path, canvas_path, canvas[:, :, :3]


def test_wz_loader_normalizes_platforms_and_vertical_walls(tmp_path):
    geometry_path, canvas_path, _ = _write_export(tmp_path)
    wz_map = load_wz_map(geometry_path, canvas_path=canvas_path)

    assert wz_map.map_id == MAP_ID
    assert len(wz_map.surfaces) == 2
    assert len(wz_map.walls) == 1
    bottom = next(surface for surface in wz_map.surfaces if surface.group == 0)
    assert bottom.foothold_ids == ("1", "2")
    assert bottom.y_at(150) == 100
    assert len(wz_map.ropes) == 1
    assert len(wz_map.portals) == 2
    assert len(wz_map.monster_spawns) == 2


def test_horizontal_projectile_detects_raised_foothold_and_wall(tmp_path):
    geometry_path, canvas_path, _ = _write_export(tmp_path)
    payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    payload["footholds"].extend([
        {
            "id": "5", "layer": 0, "group": 3,
            "x1": 70, "y1": 70, "x2": 130, "y2": 70,
            "prev": None, "next": None,
        },
        {
            "id": "6", "layer": 0, "group": 4,
            "x1": 160, "y1": 60, "x2": 160, "y2": 100,
            "prev": None, "next": None,
        },
    ])
    geometry_path.write_text(json.dumps(payload), encoding="utf-8")
    wz_map = load_wz_map(geometry_path, canvas_path=canvas_path)

    blocker = wz_map.first_horizontal_projectile_blocker(
        Point(40, 70),
        140,
        clearance=5,
        origin_margin=15,
    )
    assert blocker is not None
    assert blocker.kind == "surface"
    assert blocker.geometry_id.startswith("surface:0:3:")
    assert blocker.point == Point(70, 70)
    assert wz_map.first_horizontal_projectile_blocker(
        Point(40, 70),
        60,
        clearance=5,
        origin_margin=15,
    ) is None

    wall = wz_map.first_horizontal_projectile_blocker(
        Point(140, 70),
        180,
        clearance=0,
        origin_margin=5,
    )
    assert wall is not None
    assert wall.kind == "wall"
    assert wall.geometry_id == "wall:0:4:6"


def test_graph_and_renderer_cover_all_navigation_actions(tmp_path):
    geometry_path, canvas_path, _ = _write_export(tmp_path)
    wz_map = load_wz_map(geometry_path, canvas_path=canvas_path)
    projection = NavigationProjection(wz_map, canvas_scale=2.0)
    projected = projection.world_to_navigation(Point(55, 75))
    restored = projection.navigation_to_world(projected)
    assert abs(restored.x - 55) < 1
    assert abs(restored.y - 75) < 1
    profile = MotionProfile.from_wz(
        wz_map,
        {"jump_safety_factor": 0.9},
        world_to_navigation_scale=projection.world_scale,
    )
    graph = build_navigation_graph(wz_map, profile, {})
    actions = {edge.action for edge in graph.edges}
    assert {
        Action.WALK,
        Action.JUMP,
        Action.DROP,
        Action.CLIMB,
        Action.PORTAL,
    }.issubset(actions)

    selected_edges = tuple(
        next(edge.id for edge in graph.edges if edge.action is action)
        for action in Action
    )
    map_bgr, routes, overlay, route_legs = render_navigation_assets(
        wz_map,
        graph,
        PatrolPlan(selected_edges, (), ()),
        projection,
        _route_config(),
    )
    assert map_bgr.shape == (240, 400, 3)
    assert overlay.shape == map_bgr.shape
    assert len(routes) == len(Action)
    assert len(route_legs) == len(routes)
    assert all(route.shape == map_bgr.shape for route in routes)


@pytest.mark.parametrize(
    ("source_x", "target_x", "command", "expected_x_bounds"),
    (
        (80, 100, "right none jump", (75, 81)),
        (120, 100, "left none jump", (119, 125)),
    ),
)
def test_renderer_leads_directional_jump_trigger_inside_source_platform(
    tmp_path,
    source_x,
    target_x,
    command,
    expected_x_bounds,
):
    geometry_path, canvas_path, _ = _write_export(tmp_path)
    wz_map = load_wz_map(geometry_path, canvas_path=canvas_path)
    projection = NavigationProjection(wz_map, canvas_scale=1.0)
    graph = NavigationGraph(
        nodes=(
            Node("source", "source-platform", source_x, 80),
            Node("target", "target-platform", target_x, 80),
        ),
        edges=(Edge("jump", Action.JUMP, "source", "target", 1.0),),
        coverage_targets=(),
    )

    _, routes, _, route_legs = render_navigation_assets(
        wz_map,
        graph,
        PatrolPlan(("jump",), (), ()),
        projection,
        _route_config(),
    )

    color = {
        "right none jump": (0, 255, 128),
        "left none jump": (255, 128, 0),
    }[command]
    ys, xs = np.where(np.all(routes[0] == color, axis=2))
    assert (xs.min(), xs.max()) == expected_x_bounds
    assert (ys.min(), ys.max()) == (78, 82)
    assert route_legs[0].source == (source_x, 80)


def test_renderer_places_rope_staging_and_launch_from_actual_motion(tmp_path):
    geometry_path, canvas_path, _ = _write_export(tmp_path)
    payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    payload["ropes"][0]["y2"] = 96
    geometry_path.write_text(json.dumps(payload), encoding="utf-8")
    wz_map = load_wz_map(geometry_path, canvas_path=canvas_path)
    projection = NavigationProjection(wz_map, canvas_scale=1.0)
    graph = NavigationGraph(
        nodes=(
            Node("bottom", "bottom", 100, 100),
            Node("top", "top", 100, 35),
        ),
        edges=(
            Edge(
                "climb",
                Action.CLIMB,
                "bottom",
                "top",
                1.0,
                detail_id="r1",
            ),
        ),
        coverage_targets=(),
    )
    motion = RopeMountMotion(
        walk_speed_px_per_sec=21.357549228124483,
        jump_height_px=13.96834524683104,
        jump_distance_px=11.025092940349396,
        runup_seconds=0.18,
    )

    _, routes, _, route_legs = render_navigation_assets(
        wz_map,
        graph,
        PatrolPlan(("climb",), (), ()),
        projection,
        _route_config(),
        motion,
    )

    mount = route_legs[0].rope_mount
    assert mount is not None
    assert mount.contact == (100, 96)
    assert mount.ground_y == 100
    assert mount.vertical_gap_px == 4.0
    assert mount.launch_lead_px == 2
    assert mount.launch_offset_px == 8
    assert mount.staging_offset_px == 12
    assert mount.predicted_contact_height_px == pytest.approx(
        11.10, abs=0.05
    )
    assert mount.contact_clearance_px > 7.0
    assert mount.reachable_at_contact
    assert tuple(routes[0][100, 88]) == (0, 127, 255)


def test_rope_mount_uses_raw_measurements_not_graph_safety_discount(tmp_path):
    geometry_path, canvas_path, _ = _write_export(tmp_path)
    wz_map = load_wz_map(geometry_path, canvas_path=canvas_path)
    projection = NavigationProjection(wz_map, canvas_scale=1.0)
    measured = {
        "walk_speed_px_per_sec": 21.357549228124483,
        "jump_height_px": 13.96834524683104,
        "jump_distance_px": 11.025092940349396,
        "jump_safety_factor": 0.82,
    }
    profile = MotionProfile.from_wz(
        wz_map,
        measured,
        world_to_navigation_scale=projection.world_scale,
    )
    runtime = WzNavigationRuntime.__new__(WzNavigationRuntime)
    runtime.config = {"motion": measured}
    runtime.route_config = {"rope_climb_runup_ms": 180}
    runtime.projection = projection

    rope_motion = runtime._rope_mount_motion(
        profile,
        MotionObservation(None, None, None, 0, 0),
    )

    assert rope_motion.walk_speed_px_per_sec == measured[
        "walk_speed_px_per_sec"
    ]
    assert rope_motion.jump_height_px == measured["jump_height_px"]
    assert rope_motion.jump_distance_px == measured["jump_distance_px"]
    assert rope_motion.launch_lead_seconds == 0.10
    assert profile.jump_height_wz * projection.world_scale[1] == pytest.approx(
        measured["jump_height_px"] * 0.82
    )


def test_graph_can_jump_forest_floor_sized_adjacent_platform_gap(tmp_path):
    geometry_path, canvas_path, _ = _write_export(tmp_path)
    payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    payload["footholds"] = [
        {
            "id": "left", "layer": 0, "group": 0,
            "x1": 0, "y1": 100, "x2": 100, "y2": 100,
            "prev": None, "next": None,
        },
        {
            "id": "right", "layer": 0, "group": 1,
            "x1": 134, "y1": 100, "x2": 234, "y2": 100,
            "prev": None, "next": None,
        },
    ]
    payload["ropes"] = []
    payload["portals"] = []
    payload["lifeSpawns"] = []
    payload["bounds"]["right"] = 234
    payload["minimap"]["width"] = 234
    geometry_path.write_text(json.dumps(payload), encoding="utf-8")
    wz_map = load_wz_map(geometry_path, canvas_path=canvas_path)
    profile = MotionProfile(
        walk_speed_wz_per_sec=125,
        climb_speed_wz_per_sec=100,
        gravity_wz_per_sec2=2000,
        fall_speed_wz_per_sec=670,
        jump_speed_wz_per_sec=555,
        air_speed_wz_per_sec=125,
        jump_height_wz=65,
        jump_distance_wz=52.04,
        character_half_width_wz=15,
        source="measured",
    )

    graph = build_navigation_graph(wz_map, profile, {})
    cross_platform_jumps = [
        edge for edge in graph.edges
        if edge.action is Action.JUMP
        and graph.node_by_id[edge.source].surface_id !=
            graph.node_by_id[edge.target].surface_id
    ]

    assert len(cross_platform_jumps) == 2
    right_jump = max(
        cross_platform_jumps,
        key=lambda edge: graph.node_by_id[edge.target].x,
    )
    assert graph.node_by_id[right_jump.source].x == 98
    assert graph.node_by_id[right_jump.target].x == 149


def test_ranged_patrol_uses_clear_spawn_free_platform_and_builds_local_loop(
        tmp_path):
    geometry_path, canvas_path, _ = _write_export(tmp_path)
    payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    payload["footholds"].append({
        "id": "safe", "layer": 0, "group": 3,
        "x1": 210, "y1": 100, "x2": 330, "y2": 100,
        "prev": None, "next": None,
    })
    payload["lifeSpawns"] = [
        {"id": "m1", "type": "m", "x": 40, "y": 95}
    ]
    payload["bounds"]["right"] = 330
    payload["minimap"]["width"] = 330
    geometry_path.write_text(json.dumps(payload), encoding="utf-8")
    wz_map = load_wz_map(geometry_path, canvas_path=canvas_path)
    projection = NavigationProjection(wz_map, canvas_scale=1.0)
    motion = MotionProfile.from_wz(
        wz_map,
        {"jump_safety_factor": 0.9},
        world_to_navigation_scale=projection.world_scale,
    )

    graph = build_navigation_graph(
        wz_map,
        motion,
        {},
        ranged_attack=RangedAttackProfile(
            horizontal_range_wz=100,
            vertical_tolerance_wz=10,
            projectile_height_wz=30,
            projectile_clearance_wz=0,
            origin_margin_wz=15,
        ),
    )
    plan = build_patrol_plan(graph)

    assert len(graph.monster_surface_ids) == 1
    assert len(graph.safe_firing_targets) == 1
    firing_node = graph.node_by_id[graph.safe_firing_targets[0]]
    assert firing_node.surface_id not in graph.monster_surface_ids
    assert graph.coverage_targets == graph.safe_firing_targets
    assert plan.edge_ids
    assert all(
        graph.edge_by_id[edge_id].action is Action.WALK
        and graph.node_by_id[
            graph.edge_by_id[edge_id].source
        ].surface_id == firing_node.surface_id
        for edge_id in plan.edge_ids
    )
    assert len(plan.recovery_edge_paths) == 1
    _, _, _, rendered_legs = render_navigation_assets(
        wz_map,
        graph,
        plan,
        projection,
        _route_config(),
    )
    assert any(leg.recovery_path == 0 for leg in rendered_legs)


def test_renderer_merges_continuous_walks_and_drops_walks_under_four_pixels(
        tmp_path):
    geometry_path, canvas_path, _ = _write_export(tmp_path)
    wz_map = load_wz_map(geometry_path, canvas_path=canvas_path)
    projection = NavigationProjection(wz_map, canvas_scale=2.0)
    nodes = (
        Node("n0", "surface-a", 0, 100),
        Node("n1", "surface-a", 10, 100),
        Node("n2", "surface-a", 100, 100),
        Node("n3", "surface-a", 150, 100),
        Node("n4", "surface-a", 151, 100),
    )
    edges = (
        Edge("e1", Action.WALK, "n0", "n1", 1.0),
        Edge("e2", Action.WALK, "n1", "n2", 1.0),
        Edge("tiny", Action.WALK, "n3", "n4", 1.0),
    )
    graph = NavigationGraph(nodes, edges, ())

    _, routes, _, route_legs = render_navigation_assets(
        wz_map,
        graph,
        PatrolPlan(("e1", "e2", "tiny"), (), ()),
        projection,
        _route_config(),
    )

    assert len(routes) == 1
    assert route_legs == (
        RenderedRouteLeg(
            Action.WALK,
            projection.world_to_navigation(Point(0, 100)),
            projection.world_to_navigation(Point(100, 100)),
            ("e1", "e2"),
        ),
    )


def test_renderer_stops_walk_merge_at_one_checkpoint_event(tmp_path):
    geometry_path, canvas_path, _ = _write_export(tmp_path)
    wz_map = load_wz_map(geometry_path, canvas_path=canvas_path)
    projection = NavigationProjection(wz_map, canvas_scale=2.0)
    graph = NavigationGraph(
        nodes=(
            Node("n0", "surface-a", 0, 100),
            Node("n1", "surface-a", 50, 100),
            Node("n2", "surface-a", 100, 100),
        ),
        edges=(
            Edge("e1", Action.WALK, "n0", "n1", 1.0),
            Edge("e2", Action.WALK, "n1", "n2", 1.0),
        ),
        coverage_targets=(),
    )
    checkpoint = CombatCheckpoint(
        "n1", "right", "clear right", after_edge_index=0
    )

    _, routes, _, route_legs = render_navigation_assets(
        wz_map,
        graph,
        PatrolPlan(
            ("e1", "e2"),
            (),
            (),
            combat_checkpoints=(checkpoint,),
        ),
        projection,
        _route_config(),
    )

    assert len(routes) == 2
    assert route_legs[0].edge_ids == ("e1",)
    assert route_legs[0].combat_checkpoint == checkpoint
    assert route_legs[1].edge_ids == ("e2",)
    assert route_legs[1].combat_checkpoint is None


def test_renderer_keeps_walk_and_horizontal_jump_in_one_route(tmp_path):
    geometry_path, canvas_path, _ = _write_export(tmp_path)
    wz_map = load_wz_map(geometry_path, canvas_path=canvas_path)
    projection = NavigationProjection(wz_map, canvas_scale=1.0)
    graph = NavigationGraph(
        nodes=(
            Node("runway", "surface-a", 10, 80),
            Node("checkpoint", "surface-a", 48, 80),
            Node("takeoff", "surface-a", 55, 80),
            Node("landing", "surface-b", 65, 80),
        ),
        edges=(
            Edge("walk", Action.WALK, "runway", "checkpoint", 1.0),
            Edge("jump", Action.JUMP, "takeoff", "landing", 1.0),
        ),
        coverage_targets=(),
    )
    checkpoint = CombatCheckpoint(
        "checkpoint", "right", "clear landing", after_edge_index=0
    )

    _, routes, _, route_legs = render_navigation_assets(
        wz_map,
        graph,
        PatrolPlan(
            ("walk", "jump"),
            (),
            (),
            combat_checkpoints=(checkpoint,),
        ),
        projection,
        _route_config(),
        RopeMountMotion(20.0, 14.0, 12.0, 0.18),
    )

    assert len(routes) == 1
    assert route_legs == (
        RenderedRouteLeg(
            Action.JUMP,
            (10, 80),
            (65, 80),
            ("walk", "jump"),
            combat_checkpoint=checkpoint,
            combat_checkpoint_position=(48, 80),
            jump_source=(55, 80),
            jump_trigger_bounds=(53, 78, 55, 82),
        ),
    )
    assert tuple(routes[0][80, 48]) == (0, 255, 0)
    assert tuple(routes[0][80, 53]) == (0, 255, 128)
    assert tuple(routes[0][80, 65]) == (255, 255, 0)
    assert not np.any(np.all(
        routes[0][78:83, 45:52] == (255, 255, 0),
        axis=2,
    ))


def test_real_forest_floor_plan_follows_numbered_platform_loop():
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
    registration_scale = 2.806
    projection = NavigationProjection(wz_map, registration_scale)
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
            horizontal_range_wz=400 / registration_scale,
            vertical_tolerance_wz=60 / (2 * registration_scale),
            projectile_height_wz=30,
            projectile_clearance_wz=8,
            origin_margin_wz=15,
        ),
    )

    plan = build_forest_floor_patrol_plan(wz_map, graph)
    checkpoint_labels = [
        checkpoint.label for checkpoint in plan.combat_checkpoints
    ]
    p1_checkpoints = [
        (projection.world_to_navigation(Point(node.x, node.y)), checkpoint.facing)
        for checkpoint in plan.combat_checkpoints
        if checkpoint.label.startswith("P1 clear")
        for node in (graph.node_by_id[checkpoint.node_id],)
    ]
    cross_platform_actions = []
    first_climb_navigation = None
    for edge_id in plan.edge_ids:
        edge = graph.edge_by_id[edge_id]
        if graph.node_by_id[edge.source].surface_id != \
                graph.node_by_id[edge.target].surface_id:
            cross_platform_actions.append(edge.action)
            if edge.action is Action.CLIMB and \
                    first_climb_navigation is None:
                source = graph.node_by_id[edge.source]
                target = graph.node_by_id[edge.target]
                first_climb_navigation = (
                    projection.world_to_navigation(
                        Point(source.x, source.y)
                    ),
                    projection.world_to_navigation(
                        Point(target.x, target.y)
                    ),
                )

    assert checkpoint_labels == [
        "P3 clear P2",
        "P3 clear P4",
        "P7 clear P6",
        "P9 clear P8",
        "P9 clear P10",
        "P13 clear P12",
        "P9 clear P10",
        "P9 clear P8",
        "P5 clear P6",
        "P3 clear P2",
        "P3 clear P4",
        "P1 clear left outside",
        "P1 clear right outside",
    ]
    assert p1_checkpoints == [
        ((67, 333), "both"),
        ((178, 333), "both"),
    ]
    patrol_node_ids = {
        node_id
        for edge_id in plan.edge_ids
        for node_id in (
            graph.edge_by_id[edge_id].source,
            graph.edge_by_id[edge_id].target,
        )
    }
    graph_surface_ids = {node.surface_id for node in graph.nodes}
    recovery_start_surfaces = [
        graph.node_by_id[graph.edge_by_id[path[0]].source].surface_id
        for path in plan.recovery_edge_paths
    ]
    assert len(plan.recovery_edge_paths) == 2 * len(graph_surface_ids)
    assert Counter(recovery_start_surfaces) == Counter({
        surface_id: 2 for surface_id in graph_surface_ids
    })
    assert all(
        graph.edge_by_id[path[-1]].target in patrol_node_ids
        for path in plan.recovery_edge_paths
    )
    assert first_climb_navigation == ((102, 333), (102, 291))
    assert all(
        graph.edge_by_id[edge_id].action is not Action.PORTAL
        for path in plan.recovery_edge_paths
        for edge_id in path
    )
    top_surface_id = min(
        graph_surface_ids,
        key=lambda surface_id: min(
            node.y for node in graph.nodes
            if node.surface_id == surface_id
        ),
    )
    top_recovery_paths = [
        path
        for path, surface_id in zip(
            plan.recovery_edge_paths,
            recovery_start_surfaces,
        )
        if surface_id == top_surface_id
    ]
    assert len(top_recovery_paths) == 2
    assert all(
        any(
            graph.edge_by_id[edge_id].action is Action.DROP
            for edge_id in path
        )
        for path in top_recovery_paths
    )
    assert cross_platform_actions == [
        Action.CLIMB,
        Action.JUMP,
        Action.CLIMB,
        Action.JUMP,
        Action.CLIMB,
        Action.JUMP,
        Action.CLIMB,
        Action.JUMP,
        Action.DROP,
        Action.JUMP,
        Action.DROP,
        Action.JUMP,
        Action.DROP,
        Action.JUMP,
        Action.DROP,
        Action.DROP,
    ]
    assert all(
        graph.edge_by_id[edge_id].action is not Action.PORTAL
        for edge_id in plan.edge_ids
    )

    first_climb = next(
        graph.edge_by_id[edge_id]
        for edge_id in plan.edge_ids
        if graph.edge_by_id[edge_id].action is Action.CLIMB
    )
    first_source = graph.node_by_id[first_climb.source]
    first_target = graph.node_by_id[first_climb.target]
    competing_climbs = [
        edge for edge in graph.edges
        if edge.action is Action.CLIMB
        and graph.node_by_id[edge.source].surface_id
        == first_source.surface_id
        and graph.node_by_id[edge.target].surface_id
        == first_target.surface_id
    ]
    assert first_source.x == min(
        graph.node_by_id[edge.source].x for edge in competing_climbs
    )

    _, routes, _, route_legs = render_navigation_assets(
        wz_map,
        graph,
        plan,
        projection,
        _route_config(),
        RopeMountMotion(
            walk_speed_px_per_sec=21.357549228124483,
            jump_height_px=13.96834524683104,
            jump_distance_px=11.025092940349396,
            runup_seconds=0.18,
        ),
    )
    p3_clear_index, p3_to_p4_route = next(
        (index, leg)
        for index, leg in enumerate(route_legs)
        if leg.combat_checkpoint is not None
        and leg.combat_checkpoint.label == "P3 clear P4"
        and leg.jump_source is not None
    )
    assert p3_to_p4_route.action is Action.JUMP
    assert p3_to_p4_route.source == (96, 291)
    assert p3_to_p4_route.combat_checkpoint_position == (146, 291)
    assert p3_to_p4_route.jump_source == (149, 291)
    assert p3_to_p4_route.jump_trigger_bounds == (147, 289, 149, 293)
    assert p3_to_p4_route.target == (158, 291)
    assert tuple(routes[p3_clear_index][291, 146]) == (0, 255, 0)
    assert tuple(routes[p3_clear_index][291, 147]) == (0, 255, 128)
    assert tuple(routes[p3_clear_index][291, 158]) == (255, 255, 0)

    p3_left_index, p3_left_route = next(
        (index, leg)
        for index, leg in enumerate(route_legs)
        if leg.source == (146, 291)
        and leg.jump_source == (93, 291)
        and leg.target == (84, 291)
    )
    assert p3_left_route.action is Action.JUMP
    assert p3_left_route.jump_trigger_bounds == (93, 289, 95, 293)
    assert tuple(routes[p3_left_index][291, 93]) == (255, 128, 0)
    assert tuple(routes[p3_left_index][291, 84]) == (255, 255, 0)
    assert not np.any(np.all(
        routes[p3_left_index][289:294, 90:97] == (255, 255, 0),
        axis=2,
    ))

    compound_jump_routes = [
        leg for leg in route_legs
        if leg.action is Action.JUMP and leg.jump_source is not None
        and leg.recovery_path is None
    ]
    assert len(compound_jump_routes) == 7
    assert all(
        leg.source != leg.jump_source
        and len(leg.edge_ids) >= 2
        and leg.jump_trigger_bounds is not None
        for leg in compound_jump_routes
    )
    assert [
        leg.combat_checkpoint.label
        for leg in compound_jump_routes
        if leg.combat_checkpoint is not None
    ] == [
        "P3 clear P4",
        "P7 clear P6",
        "P9 clear P10",
        "P13 clear P12",
        "P9 clear P8",
        "P5 clear P6",
    ]
    first_mount = next(
        leg.rope_mount for leg in route_legs if leg.rope_mount is not None
    )
    recovery_legs = [
        (index, leg)
        for index, leg in enumerate(route_legs)
        if leg.recovery_path is not None
    ]
    # P3 recovery is deliberately inward from both physical edges. Keep this
    # independent from the normal P3->P2/P4 jump legs so a knockback cannot
    # select an outward recovery command and walk Hero off the foothold.
    assert any(
        leg.source == (93, 291)
        and leg.target == (121, 291)
        and leg.action is Action.WALK
        for _, leg in recovery_legs
    )
    assert any(
        leg.source == (149, 291)
        and leg.target == (121, 291)
        and leg.action is Action.WALK
        for _, leg in recovery_legs
    )
    p1_left_index, p1_left_leg = next(
        (index, leg)
        for index, leg in enumerate(route_legs)
        if leg.combat_checkpoint is not None
        and leg.combat_checkpoint.label == "P1 clear left outside"
    )
    p2_descent_walk = route_legs[p1_left_index - 2]
    assert p1_left_leg.action is Action.DROP
    assert p2_descent_walk.action is Action.WALK
    assert p2_descent_walk.target[0] == p1_left_leg.target[0]
    assert p1_left_leg.source[0] == p1_left_leg.target[0]
    assert [
        (leg.source, leg.target, leg.recovery_path)
        for _, leg in recovery_legs[:2]
    ] == [
        ((17, 333), (67, 333), 0),
        ((224, 333), (178, 333), 1),
    ]
    top_left_index, top_left_leg = next(
        (index, leg)
        for index, leg in recovery_legs
        if leg.source == (17, 29)
        and leg.target == (121, 29)
    )
    top_right_index, top_right_leg = next(
        (index, leg)
        for index, leg in recovery_legs
        if leg.source == (224, 29)
        and leg.target == (121, 29)
    )
    assert top_left_leg.action is Action.WALK
    assert top_right_leg.action is Action.WALK
    assert any(
        leg.recovery_path == top_left_leg.recovery_path
        and leg.action is Action.DROP
        and leg.source == (121, 29)
        and leg.target == (121, 186)
        for _, leg in recovery_legs
    )
    assert any(
        leg.recovery_path == top_right_leg.recovery_path
        and leg.action is Action.DROP
        and leg.source == (121, 29)
        and leg.target == (121, 186)
        for _, leg in recovery_legs
    )
    runtime = WzNavigationRuntime.__new__(WzNavigationRuntime)
    runtime.route_legs = route_legs
    assert not runtime.combat_checkpoint_reached(
        p3_clear_index, (145, 291)
    )
    assert not runtime.combat_checkpoint_reached(
        p3_clear_index, (140, 291), braking_distance=5
    )
    assert runtime.combat_checkpoint_reached(
        p3_clear_index, (141, 291), braking_distance=5
    )
    assert runtime.combat_checkpoint_reached(
        p3_clear_index, (146, 291)
    )
    assert runtime.combat_checkpoint_reached(
        p3_clear_index, (149, 291)
    )
    assert not runtime.combat_checkpoint_reached(
        p3_clear_index, (150, 291)
    )
    assert runtime.nearest_route_index((14, 333)) == recovery_legs[0][0]
    assert runtime.nearest_route_index((227, 333)) == recovery_legs[1][0]
    assert runtime.nearest_route_index((30, 29)) == top_left_index
    assert runtime.nearest_route_index((210, 29)) == top_right_index
    assert first_mount.contact == (102, 329)
    assert first_mount.vertical_gap_px == 4.0
    assert first_mount.launch_lead_px == 2
    assert first_mount.launch_offset_px == 8
    assert first_mount.staging_offset_px == 12
    assert first_mount.predicted_contact_height_px == pytest.approx(
        11.10, abs=0.05
    )
    assert first_mount.contact_clearance_px > 7.0
    assert first_mount.reachable_at_contact


def test_recorded_forest_floor_x_anchors_drive_wz_jump_and_rope_actions():
    project_root = Path(__file__).resolve().parents[1]
    export_path = (
        project_root.parent
        / "SuperHumanMapleStory"
        / "var"
        / "all-maps-with-canvases"
        / "100040110.json"
    )
    map_path = project_root / "minimaps" / "forest_floor" / "map.png"
    route_directory = map_path.parent
    if not export_path.is_file() or not map_path.is_file():
        pytest.skip("Forest Floor WZ/recorded route assets are not installed")

    with (project_root / "config" / "config_default.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        config = yaml.safe_load(stream)
    navigation_config = dict(config["wz_navigation"])
    navigation_config["geometry_dir"] = str(export_path.parent)
    navigation_config["candidate_map_ids"] = ["100040110"]
    navigation_config["learn_motion"] = False
    navigation_config["platform_state_machine"] = {"enable": False}
    navigation_config["motion"] = {
        **navigation_config["motion"],
        "walk_speed_px_per_sec": 21.3575,
        "jump_height_px": 13.9683,
        "jump_distance_px": 11.0251,
    }
    runtime = WzNavigationRuntime(
        navigation_config,
        config["route"],
        directional_attack_config=config["directional_attack"],
        selected_map_name="forest_floor",
        recorded_route_directory=route_directory,
    )
    image = cv2.imread(str(map_path), cv2.IMREAD_COLOR)

    assert runtime.bootstrap(image) == "100040110"
    assert runtime.summary()["recordedXAnchors"] == {
        "routeFiles": ["route1.png", "route2.png"],
        "jumps": 7,
        "climbs": 3,
        "matchedRopes": 3,
        "appliedJumpLegs": 7,
        "appliedClimbLegs": 3,
    }
    normal_jump_legs = [
        leg for leg in runtime.route_legs
        if leg.recovery_path is None and leg.action is Action.JUMP
    ]
    assert [leg.recorded_x_anchor for leg in normal_jump_legs] == [
        142, 176, 142, 173, 97, 64, 100,
    ]
    assert all(
        leg.jump_source[0] == leg.recorded_x_anchor
        and leg.jump_trigger_bounds[0] <= leg.recorded_x_anchor
        <= leg.jump_trigger_bounds[2]
        for leg in normal_jump_legs
    )

    normal_climb_legs = [
        leg for leg in runtime.route_legs
        if leg.recovery_path is None and leg.action is Action.CLIMB
    ]
    assert [leg.recorded_x_anchor for leg in normal_climb_legs] == [
        None, 220, 114, 220,
    ]
    anchored_mounts = [
        leg.rope_mount for leg in normal_climb_legs
        if leg.recorded_x_anchor is not None
    ]
    assert [mount.launch_offset_px for mount in anchored_mounts] == [8, 12, 8]
    assert all(mount.approach_direction == "left" for mount in anchored_mounts)
    assert all(
        leg.recorded_x_anchor is None
        for leg in runtime.route_legs
        if leg.recovery_path is not None
    )


def test_runtime_selects_nearest_route_and_detects_walk_overshoot():
    runtime = WzNavigationRuntime.__new__(WzNavigationRuntime)
    runtime.route_legs = (
        RenderedRouteLeg(Action.WALK, (0, 10), (50, 10), ("walk",)),
        RenderedRouteLeg(Action.JUMP, (90, 50), (100, 30), ("jump",)),
        RenderedRouteLeg(Action.WALK, (80, 20), (60, 20), ("left",)),
    )

    assert runtime.nearest_route_index((25, 11)) == 0
    assert runtime.nearest_route_index((89, 49)) == 1
    assert runtime.walk_target_crossed(0, (52, 11), tolerance=3)
    assert not runtime.walk_target_crossed(0, (49, 10), tolerance=3)
    assert not runtime.walk_target_crossed(0, (54, 10), tolerance=3)
    assert runtime.walk_target_crossed(2, (58, 19), tolerance=3)


def test_forest_floor_patrol_does_not_depend_on_projectile_filter():
    runtime = WzNavigationRuntime.__new__(WzNavigationRuntime)
    runtime._configured_patrol_strategy = "spawn_sweep"

    assert runtime._patrol_strategy_for_map("100040110") == \
        "ranged_safe_platforms"
    assert runtime._patrol_strategy_for_map("100000001") == "spawn_sweep"


def test_catalog_and_runtime_match_scaled_live_minimap(tmp_path):
    _, _, canonical = _write_export(tmp_path)
    affine = np.array([[1.6, 0.0, 13.0], [0.0, 1.6, 9.0]], dtype=np.float32)
    live = cv2.warpAffine(canonical, affine, (350, 220), borderValue=(72, 72, 72))
    cv2.circle(live, (173, 105), 3, (136, 255, 255), -1)

    catalog = WzMapCatalog(tmp_path, minimum_inliers=6)
    registration, _ = catalog.recognize(live)
    assert registration.entry.map_id == MAP_ID
    assert registration.inlier_count >= 6
    assert abs(registration.effective_scale - 1.6) < 0.05

    runtime = WzNavigationRuntime(
        {
            "geometry_dir": str(tmp_path),
            "candidate_map_ids": [MAP_ID],
            "learn_motion": False,
        },
        _route_config(),
    )
    assert runtime.bootstrap(live) == MAP_ID
    player_live = (173, 105)
    update = runtime.update(
        live,
        player_live,
        on_ladder=False,
        previous_command="none none none",
        allow_motion_rebuild=False,
        timestamp=1.0,
    )
    assert update is not None
    assert update.map_id == MAP_ID
    assert runtime.routes_rgb
    assert runtime.summary()["surfaces"] == 2


def test_selected_recorded_map_uses_configured_wz_binding(tmp_path):
    _, _, canonical = _write_export(tmp_path)
    messages = []
    runtime = WzNavigationRuntime(
        {
            "geometry_dir": str(tmp_path),
            "candidate_map_ids": ["999999999"],
            "map_bindings": {"forest_floor": MAP_ID},
            "learn_motion": False,
        },
        _route_config(),
        selected_map_name="forest_floor",
        log=messages.append,
    )

    assert runtime.candidate_map_ids == {MAP_ID}
    assert runtime.bootstrap(canonical) == MAP_ID
    assert any(
        f"forest_floor -> {MAP_ID}" in message for message in messages
    )


def test_configured_binding_retains_last_live_registration_on_dropouts(
    tmp_path, monkeypatch
):
    _, _, canonical = _write_export(tmp_path)
    messages = []
    runtime = WzNavigationRuntime(
        {
            "geometry_dir": str(tmp_path),
            "map_bindings": {"forest_floor": MAP_ID},
            "registration_failure_limit": 3,
            "learn_motion": False,
        },
        _route_config(),
        selected_map_name="forest_floor",
        log=messages.append,
    )
    runtime.bootstrap(canonical)
    player_live = (100, 80)

    first_update = runtime.update(
        canonical,
        player_live,
        on_ladder=False,
        previous_command="none none none",
        allow_motion_rebuild=False,
        timestamp=1.0,
    )
    assert first_update is not None
    live_registration = runtime.registration
    monkeypatch.setattr(
        runtime.catalog, "register_selected", lambda _live: None
    )

    for timestamp in range(2, 7):
        update = runtime.update(
            canonical,
            player_live,
            on_ladder=False,
            previous_command="none none none",
            allow_motion_rebuild=False,
            timestamp=float(timestamp),
        )
        assert update is not None
        assert update.map_id == MAP_ID

    assert runtime.registration is live_registration
    assert runtime.registration_failures == 5
    assert runtime.routes_rgb
    assert any(
        "retaining configured map binding" in message
        for message in messages
    )


def test_auto_wz_does_not_apply_recorded_map_binding(tmp_path):
    _write_export(tmp_path)
    runtime = WzNavigationRuntime(
        {
            "geometry_dir": str(tmp_path),
            "map_bindings": {"forest_floor": MAP_ID},
        },
        _route_config(),
        selected_map_name="__auto_wz__",
    )

    assert runtime.candidate_map_ids is None


def test_motion_estimator_uses_the_command_active_during_each_interval():
    estimator = HeroMotionEstimator(
        minimum_walk_samples=3,
        minimum_jump_samples=1,
    )
    estimator.observe((0, 100), 0.0, "right none none", on_ladder=False)
    estimator.observe((2, 100), 0.1, "right none none", on_ladder=False)
    estimator.observe((4, 100), 0.2, "right none none", on_ladder=False)
    walk = estimator.observe(
        (6, 100), 0.3, "right none none", on_ladder=False
    )
    assert walk.walk_speed_px_per_sec == 20

    # The command starts a ground run-up. Distance learning must begin only
    # when Y actually rises, not when the jump command is issued.
    estimator.observe((8, 100), 0.4, "right none jump", on_ladder=False)
    assert estimator.jump_active
    estimator.observe((10, 95), 0.5, "right none none", on_ladder=False)
    estimator.observe((12, 85), 0.6, "right none none", on_ladder=False)
    estimator.observe((14, 98), 0.7, "right none none", on_ladder=False)
    estimator.observe((15, 100), 0.8, "right none none", on_ladder=False)
    jump = estimator.observe(
        (16, 100.5), 0.9, "right none none", on_ladder=False
    )
    assert jump.jump_height_px == 15
    assert jump.jump_distance_px == 7
    assert jump.jump_runup_distance_px == 2
    assert not estimator.jump_active

    drop_estimator = HeroMotionEstimator(minimum_jump_samples=1)
    drop_estimator.observe((0, 10), 0.0, "none none none", on_ladder=False)
    drop = drop_estimator.observe(
        (0, 12), 0.1, "none down jump", on_ladder=False
    )
    assert drop.jump_samples == 0
    assert not drop_estimator.jump_active


def test_motion_estimator_accepts_landing_on_a_different_height():
    estimator = HeroMotionEstimator(minimum_jump_samples=1)
    estimator.observe((0, 100), 0.0, "none none none", on_ladder=False)
    estimator.observe((3, 100), 0.1, "right none jump", on_ladder=False)
    estimator.observe((5, 96), 0.2, "right none none", on_ladder=False)
    estimator.observe((8, 85), 0.3, "right none none", on_ladder=False)
    estimator.observe((11, 90), 0.4, "right none none", on_ladder=False)
    estimator.observe((13, 92), 0.5, "right none none", on_ladder=False)
    observation = estimator.observe(
        (14, 92.5), 0.6, "right none none", on_ladder=False
    )

    assert observation.jump_height_px == 15
    assert observation.jump_distance_px == 10
    assert observation.jump_runup_distance_px == 3
    assert observation.jump_samples == 1
    assert not estimator.jump_active


def test_motion_estimator_discards_jump_command_without_takeoff():
    estimator = HeroMotionEstimator(minimum_jump_samples=1)
    estimator.observe((0, 100), 0.0, "none none none", on_ladder=False)
    estimator.observe((2, 100), 0.1, "right none jump", on_ladder=False)
    assert estimator.jump_active

    observation = estimator.observe(
        (6, 100), 0.9, "right none none", on_ladder=False
    )
    assert observation.jump_samples == 0
    assert observation.jump_height_px is None
    assert observation.jump_distance_px is None
    assert observation.jump_runup_distance_px is None
    assert not estimator.jump_active


def test_failed_full_catalog_scan_respects_retry_cooldown(tmp_path, monkeypatch):
    _write_export(tmp_path)
    runtime = WzNavigationRuntime(
        {
            "geometry_dir": str(tmp_path),
            "recognition_retry_interval": 10,
        },
        _route_config(),
    )

    class MissingCatalog:
        def __init__(self):
            self.calls = 0

        def recognize(self, *_args, **_kwargs):
            self.calls += 1
            raise WzMapRecognitionError("not found")

    catalog = MissingCatalog()
    runtime.catalog = catalog
    clock = {"now": 100.0}
    monkeypatch.setattr(
        "src.navigation.runtime.time.monotonic", lambda: clock["now"]
    )
    minimap = np.zeros((50, 50, 3), dtype=np.uint8)

    def update():
        return runtime.update(
            minimap,
            None,
            on_ladder=False,
            previous_command="none none none",
            allow_motion_rebuild=False,
        )

    assert update() is None
    assert catalog.calls == 1
    clock["now"] = 109.9
    assert update() is None
    assert catalog.calls == 1
    clock["now"] = 110.0
    assert update() is None
    assert catalog.calls == 2
