"""Recognize a minimap and render the WZ navigation geometry for inspection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import cv2
import yaml

from src.navigation.hunt_planner import Action
from src.navigation.runtime import WzNavigationRuntime
from src.utils.common import override_cfg
from src.navigation.wz_geometry import Point


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_default_config() -> dict:
    path = PROJECT_ROOT / "config" / "config_default.yaml"
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _plan_preview(runtime: WzNavigationRuntime):
    assert runtime.geometry_overlay_bgr is not None
    assert runtime.graph is not None
    assert runtime.plan is not None
    assert runtime.projection is not None
    preview = runtime.geometry_overlay_bgr.copy()
    nodes = runtime.graph.node_by_id
    edges = runtime.graph.edge_by_id
    colors = {
        Action.WALK: (255, 255, 0),
        Action.JUMP: (0, 128, 255),
        Action.DROP: (0, 255, 255),
        Action.CLIMB: (255, 180, 0),
        Action.PORTAL: (255, 0, 255),
    }
    for edge_id in runtime.plan.edge_ids:
        edge = edges.get(edge_id)
        if edge is None:
            continue
        source = nodes[edge.source]
        target = nodes[edge.target]
        source_pixel = runtime.projection.world_to_navigation(
            Point(source.x, source.y)
        )
        target_pixel = runtime.projection.world_to_navigation(
            Point(target.x, target.y)
        )
        cv2.arrowedLine(
            preview,
            source_pixel,
            target_pixel,
            colors[edge.action],
            1,
            cv2.LINE_AA,
            tipLength=0.18,
        )
    return preview


def _write_image(path: Path, image) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"unable to write image: {path}")


def inspect(args: argparse.Namespace) -> dict:
    config = _load_default_config()
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as stream:
            override_cfg(config, yaml.safe_load(stream) or {})
    navigation_config = deepcopy(config["wz_navigation"])
    geometry_dir = Path(args.geometry_dir).expanduser()
    if not geometry_dir.is_absolute():
        geometry_dir = (PROJECT_ROOT / geometry_dir).resolve()
    navigation_config["geometry_dir"] = str(geometry_dir)
    if args.map_id:
        navigation_config["candidate_map_ids"] = [args.map_id]
    navigation_config["learn_motion"] = False
    if args.full_patrol:
        navigation_config["platform_state_machine"] = {"enable": False}

    minimap_path = Path(args.minimap).expanduser().resolve()
    minimap = cv2.imread(str(minimap_path), cv2.IMREAD_COLOR)
    if minimap is None:
        raise FileNotFoundError(f"unable to read minimap: {minimap_path}")

    runtime = WzNavigationRuntime(
        navigation_config,
        config["route"],
        directional_attack_config=config["directional_attack"],
        recorded_route_directory=minimap_path.parent,
        log=print,
    )
    map_id = runtime.bootstrap(minimap)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry_path = output_dir / f"{map_id}_geometry.png"
    patrol_path = output_dir / f"{map_id}_patrol.png"
    _write_image(geometry_path, runtime.geometry_overlay_bgr)
    _write_image(patrol_path, _plan_preview(runtime))
    if args.write_route_images:
        route_dir = output_dir / f"{map_id}_routes"
        route_dir.mkdir(parents=True, exist_ok=True)
        for index, route_rgb in enumerate(runtime.routes_rgb, start=1):
            _write_image(
                route_dir / f"route{index:03d}.png",
                cv2.cvtColor(route_rgb, cv2.COLOR_RGB2BGR),
            )

    summary = runtime.summary()
    assert runtime.registration is not None
    assert runtime.graph is not None
    assert runtime.plan is not None
    edge_by_id = runtime.graph.edge_by_id
    summary.update(
        {
            "input": str(minimap_path),
            "geometryCache": str(geometry_dir),
            "match": {
                "inliers": runtime.registration.inlier_count,
                "goodMatches": runtime.registration.good_match_count,
                "inlierRatio": runtime.registration.inlier_ratio,
                "scale": runtime.registration.effective_scale,
                "residualP95Px": runtime.registration.residual_p95_px,
            },
            "planActions": dict(
                sorted(
                    Counter(
                        edge_by_id[edge_id].action.value
                        for edge_id in runtime.plan.edge_ids
                        if edge_id in edge_by_id
                    ).items()
                )
            ),
            "geometryPreview": str(geometry_path),
            "patrolPreview": str(patrol_path),
        }
    )
    summary_path = output_dir / f"{map_id}_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Match a captured/stitched minimap against exported WZ canvases "
            "and render its generated navigation graph."
        )
    )
    parser.add_argument("--minimap", required=True, help="BGR/RGB minimap PNG")
    parser.add_argument(
        "--geometry-dir",
        default="../SuperHumanMapleStory/var/all-maps-with-canvases",
        help="MapleWzExporter export-all directory",
    )
    parser.add_argument(
        "--map-id",
        default="",
        help="optional map ID allow-list shortcut for a known sample",
    )
    parser.add_argument(
        "--config",
        default="",
        help="optional custom YAML merged over config_default.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="screenshot/wz_navigation",
        help="preview output directory",
    )
    parser.add_argument(
        "--write-route-images",
        action="store_true",
        help="also write every generated route leg",
    )
    parser.add_argument(
        "--full-patrol",
        action="store_true",
        help=(
            "disable the live platform state machine only for this preview "
            "and render the complete Forest Floor patrol/recovery plan"
        ),
    )
    return parser


def main() -> int:
    inspect(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
