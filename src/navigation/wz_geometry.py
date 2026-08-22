"""Small, stable reader for the WZ JSON contract used by the navigator.

The binary WZ boundary stays in the .NET exporter from SuperHumanMapleStory.
This module intentionally consumes only its versioned JSON output and does not
import that project's Python package or mutable internal object model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class CanvasVariant:
    variant_id: str
    pixel_width: int
    pixel_height: int
    fingerprint: str
    cache_reference: str


@dataclass(frozen=True, slots=True)
class MinimapMetadata:
    width: float
    height: float
    center_x: float
    center_y: float
    magnification: float | None
    variants: tuple[CanvasVariant, ...]

    @property
    def world_left(self) -> float:
        return -self.center_x

    @property
    def world_top(self) -> float:
        return -self.center_y


@dataclass(frozen=True, slots=True)
class Surface:
    id: str
    layer: int
    group: int
    points: tuple[Point, ...]
    foothold_ids: tuple[str, ...]
    cant_through: bool
    forbid_fall_down: bool
    force_values: tuple[float, ...]

    @property
    def min_x(self) -> float:
        return min(point.x for point in self.points)

    @property
    def max_x(self) -> float:
        return max(point.x for point in self.points)

    @property
    def min_y(self) -> float:
        return min(point.y for point in self.points)

    @property
    def max_y(self) -> float:
        return max(point.y for point in self.points)

    @property
    def supports_default_motion(self) -> bool:
        return not any(abs(value) > EPSILON for value in self.force_values)

    def contains_x(self, x: float) -> bool:
        return self.min_x - EPSILON <= x <= self.max_x + EPSILON

    def y_at(self, x: float) -> float:
        """Interpolate the first non-vertical foothold segment covering x."""
        candidates: list[tuple[int, float]] = []
        for index, (left, right) in enumerate(zip(self.points, self.points[1:])):
            low_x, high_x = sorted((left.x, right.x))
            if low_x - EPSILON <= x <= high_x + EPSILON and \
                    abs(right.x - left.x) > EPSILON:
                ratio = (x - left.x) / (right.x - left.x)
                candidates.append(
                    (index, left.y + ratio * (right.y - left.y))
                )
        if not candidates:
            raise ValueError(f"x={x} is outside surface {self.id}")
        return min(candidates)[1]


@dataclass(frozen=True, slots=True)
class Wall:
    id: str
    x: float
    y1: float
    y2: float


@dataclass(frozen=True, slots=True)
class Rope:
    id: str
    x: float
    y1: float
    y2: float
    layer: int
    ladder: bool
    upper_foothold: bool | None


@dataclass(frozen=True, slots=True)
class Portal:
    id: str
    name: str
    type: int
    x: float
    y: float
    target_map_id: str | None
    target_name: str | None
    script: str | None


@dataclass(frozen=True, slots=True)
class LifeSpawn:
    id: str
    kind: str
    x: float
    y: float
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ProjectileBlocker:
    """First WZ foothold or wall intersecting a horizontal projectile."""

    kind: str
    geometry_id: str
    point: Point


@dataclass(frozen=True, slots=True)
class WzMap:
    path: Path
    map_id: str
    geometry_map_id: str
    bounds: tuple[float, float, float, float]
    minimap: MinimapMetadata
    selected_variant: CanvasVariant
    surfaces: tuple[Surface, ...]
    walls: tuple[Wall, ...]
    ropes: tuple[Rope, ...]
    portals: tuple[Portal, ...]
    life_spawns: tuple[LifeSpawn, ...]
    physics: dict[str, float]

    @property
    def monster_spawns(self) -> tuple[LifeSpawn, ...]:
        return tuple(spawn for spawn in self.life_spawns if spawn.kind == "m")

    @property
    def canvas_path(self) -> Path:
        reference = Path(self.selected_variant.cache_reference)
        if reference.is_absolute() or ".." in reference.parts:
            raise ValueError("WZ canvas reference must stay inside its cache")
        result = (self.path.parent / reference).resolve()
        if self.path.parent.resolve() not in result.parents:
            raise ValueError("WZ canvas reference escaped its cache")
        return result

    @property
    def world_to_canvas_scale(self) -> tuple[float, float]:
        return (
            self.selected_variant.pixel_width / self.minimap.width,
            self.selected_variant.pixel_height / self.minimap.height,
        )

    def world_to_canvas(self, point: Point) -> Point:
        scale_x, scale_y = self.world_to_canvas_scale
        return Point(
            (point.x - self.minimap.world_left) * scale_x,
            (point.y - self.minimap.world_top) * scale_y,
        )

    def canvas_to_world(self, point: Point) -> Point:
        scale_x, scale_y = self.world_to_canvas_scale
        return Point(
            point.x / scale_x + self.minimap.world_left,
            point.y / scale_y + self.minimap.world_top,
        )

    def nearest_surface(
        self,
        point: Point,
        *,
        maximum_vertical_distance: float = 140.0,
        prefer_below: bool = True,
    ) -> tuple[Surface, Point] | None:
        candidates: list[tuple[tuple[float, float, str], Surface, Point]] = []
        for surface in self.surfaces:
            if not surface.contains_x(point.x):
                continue
            try:
                surface_y = surface.y_at(point.x)
            except ValueError:
                continue
            delta = surface_y - point.y
            if abs(delta) > maximum_vertical_distance:
                continue
            below_penalty = 0.0 if not prefer_below or delta >= -10.0 else 1.0
            candidates.append(
                (
                    (below_penalty, abs(delta), surface.id),
                    surface,
                    Point(point.x, surface_y),
                )
            )
        if not candidates:
            return None
        _, surface, snapped = min(candidates, key=lambda item: item[0])
        return surface, snapped

    def first_horizontal_projectile_blocker(
        self,
        origin: Point,
        target_x: float,
        *,
        clearance: float = 0.0,
        origin_margin: float = 0.0,
    ) -> ProjectileBlocker | None:
        """Return the first foothold/wall touching a horizontal shot.

        MapleStory footholds are line geometry rather than filled polygons.
        Expanding those lines by ``clearance`` accounts for the projectile's
        collision thickness and small minimap-registration errors without
        treating an unrelated overhead platform as a solid column.
        """
        if not all(
            math.isfinite(value)
            for value in (origin.x, origin.y, target_x, clearance, origin_margin)
        ):
            raise ValueError("projectile geometry values must be finite")
        clearance = max(0.0, float(clearance))
        origin_margin = max(0.0, float(origin_margin))
        delta_x = float(target_x) - origin.x
        if abs(delta_x) <= origin_margin + EPSILON:
            return None

        direction = 1.0 if delta_x > 0.0 else -1.0
        ray_start_x = origin.x + direction * origin_margin
        ray_end_x = float(target_x) - direction * EPSILON
        ray_min_x, ray_max_x = sorted((ray_start_x, ray_end_x))
        candidates: list[tuple[float, str, str, Point]] = []

        for wall in self.walls:
            if not ray_min_x - EPSILON <= wall.x <= ray_max_x + EPSILON:
                continue
            if not wall.y1 - clearance <= origin.y <= wall.y2 + clearance:
                continue
            hit_y = min(max(origin.y, wall.y1), wall.y2)
            candidates.append(
                (
                    abs(wall.x - origin.x),
                    "wall",
                    wall.id,
                    Point(wall.x, hit_y),
                )
            )

        for surface in self.surfaces:
            if surface.max_x < ray_min_x - EPSILON or \
                    surface.min_x > ray_max_x + EPSILON:
                continue
            for left, right in zip(surface.points, surface.points[1:]):
                segment_min_x, segment_max_x = sorted((left.x, right.x))
                overlap_min_x = max(ray_min_x, segment_min_x)
                overlap_max_x = min(ray_max_x, segment_max_x)
                if overlap_min_x > overlap_max_x + EPSILON:
                    continue
                if abs(right.x - left.x) <= EPSILON:
                    continue

                def y_at(x: float) -> float:
                    ratio = (x - left.x) / (right.x - left.x)
                    return left.y + ratio * (right.y - left.y)

                endpoint_values = (
                    (overlap_min_x, y_at(overlap_min_x)),
                    (overlap_max_x, y_at(overlap_max_x)),
                )
                segment_low_y = min(value[1] for value in endpoint_values)
                segment_high_y = max(value[1] for value in endpoint_values)
                if origin.y < segment_low_y - clearance or \
                        origin.y > segment_high_y + clearance:
                    continue

                if abs(right.y - left.y) <= EPSILON:
                    hit_x = (
                        overlap_min_x if direction > 0 else overlap_max_x
                    )
                else:
                    band_crossings = []
                    for band_y in (
                            origin.y - clearance,
                            origin.y + clearance):
                        ratio = (band_y - left.y) / (right.y - left.y)
                        band_crossings.append(
                            left.x + ratio * (right.x - left.x)
                        )
                    band_min_x, band_max_x = sorted(band_crossings)
                    collision_min_x = max(overlap_min_x, band_min_x)
                    collision_max_x = min(overlap_max_x, band_max_x)
                    if collision_min_x > collision_max_x + EPSILON:
                        continue
                    hit_x = (
                        collision_min_x if direction > 0
                        else collision_max_x
                    )
                hit_y = y_at(hit_x)
                candidates.append(
                    (
                        abs(hit_x - origin.x),
                        "surface",
                        surface.id,
                        Point(hit_x, hit_y),
                    )
                )

        if not candidates:
            return None
        _, kind, geometry_id, point = min(
            candidates,
            key=lambda item: (item[0], item[1], item[2]),
        )
        return ProjectileBlocker(kind, geometry_id, point)


@dataclass(frozen=True, slots=True)
class _RawFoothold:
    id: str
    layer: int
    group: int
    x1: float
    y1: float
    x2: float
    y2: float
    previous: str | None
    next: str | None
    cant_through: int | None
    forbid_fall_down: int | None
    force: float | None


def _point_key(x: float, y: float) -> tuple[float, float]:
    return round(x, 9), round(y, 9)


def _normalize_footholds(
    values: list[dict[str, Any]],
) -> tuple[tuple[Surface, ...], tuple[Wall, ...]]:
    traversable: list[_RawFoothold] = []
    walls: list[Wall] = []
    for value in values:
        foothold = _RawFoothold(
            id=str(value["id"]),
            layer=int(value["layer"]),
            group=int(value["group"]),
            x1=float(value["x1"]),
            y1=float(value["y1"]),
            x2=float(value["x2"]),
            y2=float(value["y2"]),
            previous=None if value.get("prev") is None else str(value["prev"]),
            next=None if value.get("next") is None else str(value["next"]),
            cant_through=value.get("cantThrough"),
            forbid_fall_down=value.get("forbidFallDown"),
            force=None if value.get("force") is None else float(value["force"]),
        )
        length = math.hypot(foothold.x2 - foothold.x1, foothold.y2 - foothold.y1)
        if length <= EPSILON:
            continue
        if abs(foothold.x2 - foothold.x1) <= EPSILON:
            walls.append(
                Wall(
                    f"wall:{foothold.layer}:{foothold.group}:{foothold.id}",
                    foothold.x1,
                    min(foothold.y1, foothold.y2),
                    max(foothold.y1, foothold.y2),
                )
            )
            continue
        traversable.append(foothold)

    unused = {foothold.id: foothold for foothold in traversable}
    if len(unused) != len(traversable):
        raise ValueError("WZ map contains duplicate foothold IDs")
    endpoint_index: dict[tuple[float, float], list[str]] = {}
    for foothold in traversable:
        for key in (
            _point_key(foothold.x1, foothold.y1),
            _point_key(foothold.x2, foothold.y2),
        ):
            endpoint_index.setdefault(key, []).append(foothold.id)

    surfaces: list[Surface] = []
    while unused:
        starts = [
            foothold
            for foothold in unused.values()
            if foothold.previous is None
            or foothold.previous not in unused
            or unused[foothold.previous].next != foothold.id
        ]
        start = min(
            starts or list(unused.values()),
            key=lambda item: (item.layer, item.group, item.id),
        )
        segments = [start]
        points = [Point(start.x1, start.y1), Point(start.x2, start.y2)]
        unused.pop(start.id)
        while True:
            tail = points[-1]
            explicit = unused.get(segments[-1].next or "")
            if explicit is not None and explicit.layer == start.layer and \
                    explicit.group == start.group and _point_key(
                        explicit.x1, explicit.y1
                    ) == _point_key(tail.x, tail.y):
                candidate = explicit
                next_point = Point(candidate.x2, candidate.y2)
            else:
                candidates = [
                    unused[candidate_id]
                    for candidate_id in endpoint_index.get(
                        _point_key(tail.x, tail.y), []
                    )
                    if candidate_id in unused
                    and unused[candidate_id].layer == start.layer
                    and unused[candidate_id].group == start.group
                ]
                if not candidates:
                    break
                candidate = min(candidates, key=lambda item: item.id)
                if _point_key(candidate.x1, candidate.y1) == _point_key(
                        tail.x, tail.y):
                    next_point = Point(candidate.x2, candidate.y2)
                else:
                    next_point = Point(candidate.x1, candidate.y1)
            segments.append(candidate)
            points.append(next_point)
            unused.pop(candidate.id)

        if points[-1].x < points[0].x:
            points.reverse()
            segments.reverse()
        foothold_ids = tuple(segment.id for segment in segments)
        surfaces.append(
            Surface(
                id=(
                    f"surface:{start.layer}:{start.group}:"
                    f"{foothold_ids[0]}:{foothold_ids[-1]}"
                ),
                layer=start.layer,
                group=start.group,
                points=tuple(points),
                foothold_ids=foothold_ids,
                cant_through=any(
                    segment.cant_through not in (None, 0)
                    for segment in segments
                ),
                forbid_fall_down=any(
                    segment.forbid_fall_down not in (None, 0)
                    for segment in segments
                ),
                force_values=tuple(
                    sorted(
                        {
                            segment.force
                            for segment in segments
                            if segment.force is not None
                        }
                    )
                ),
            )
        )

    return (
        tuple(sorted(surfaces, key=lambda item: (item.min_x, item.min_y, item.id))),
        tuple(sorted(walls, key=lambda item: (item.x, item.y1, item.id))),
    )


def _required_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def load_wz_map(path: str | Path, *, canvas_path: str | Path | None = None) -> WzMap:
    """Load one exporter document and normalize its navigation geometry."""
    geometry_path = Path(path).resolve()
    with geometry_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schemaVersion") != "1.0.0":
        raise ValueError("unsupported WZ map JSON schema")

    source = payload["source"]
    minimap_data = payload.get("minimap")
    if not isinstance(minimap_data, dict):
        raise ValueError("WZ map has no minimap metadata")
    variants = tuple(
        CanvasVariant(
            variant_id=str(value["stableCanvasVariantId"]),
            pixel_width=int(value["pixelWidth"]),
            pixel_height=int(value["pixelHeight"]),
            fingerprint=str(value["contentFingerprint"]),
            cache_reference=str(value["cacheReference"]),
        )
        for value in minimap_data.get("canvasVariants", ())
        if value.get("cacheReference")
    )
    if not variants:
        raise ValueError("WZ map has no cached minimap canvas")

    selected_variant = variants[0]
    if canvas_path is not None:
        selected_name = Path(canvas_path).name.casefold()
        matching = [
            variant
            for variant in variants
            if Path(variant.cache_reference).name.casefold() == selected_name
            or variant.fingerprint.startswith(
                Path(canvas_path).stem.removeprefix("canvas-")
            )
        ]
        if len(matching) != 1:
            raise ValueError("matched WZ canvas is not a unique map variant")
        selected_variant = matching[0]

    minimap = MinimapMetadata(
        width=_required_number(minimap_data.get("width"), "minimap.width"),
        height=_required_number(minimap_data.get("height"), "minimap.height"),
        center_x=_required_number(minimap_data.get("centerX"), "minimap.centerX"),
        center_y=_required_number(minimap_data.get("centerY"), "minimap.centerY"),
        magnification=(
            None
            if minimap_data.get("magnification") is None
            else _required_number(
                minimap_data.get("magnification"), "minimap.magnification"
            )
        ),
        variants=variants,
    )
    if minimap.width <= 0 or minimap.height <= 0:
        raise ValueError("WZ minimap world rectangle must be positive")

    surfaces, walls = _normalize_footholds(list(payload.get("footholds", ())))
    ropes = tuple(
        Rope(
            id=str(value["id"]),
            x=float(value["x"]),
            y1=min(float(value["y1"]), float(value["y2"])),
            y2=max(float(value["y1"]), float(value["y2"])),
            layer=int(value["layer"]),
            ladder=bool(value["ladder"]),
            upper_foothold=value.get("upperFoothold"),
        )
        for value in payload.get("ropes", ())
    )
    portals = tuple(
        Portal(
            id=str(value["id"]),
            name=str(value["name"]),
            type=int(value["type"]),
            x=float(value["x"]),
            y=float(value["y"]),
            target_map_id=(
                None
                if value.get("targetMapId") is None
                else str(value["targetMapId"]).zfill(9)
            ),
            target_name=(
                None if value.get("targetName") is None else str(value["targetName"])
            ),
            script=None if value.get("script") is None else str(value["script"]),
        )
        for value in payload.get("portals", ())
    )
    spawns = tuple(
        LifeSpawn(
            id=str(value.get("id", "")),
            kind=str(value.get("type", "")),
            x=float(value["x"]),
            y=float(value["y"]),
            raw=value,
        )
        for value in payload.get("lifeSpawns", ())
        if isinstance(value, dict) and "x" in value and "y" in value
    )
    physics = {
        str(field["name"]): float(field["value"])
        for field in (payload.get("clientPhysics") or {}).get("fields", ())
        if isinstance(field, dict) and "name" in field and "value" in field
    }
    bounds = payload["bounds"]
    return WzMap(
        path=geometry_path,
        map_id=str(source["requestedMapId"]).zfill(9),
        geometry_map_id=str(source["geometryMapId"]).zfill(9),
        bounds=(
            float(bounds["left"]),
            float(bounds["top"]),
            float(bounds["right"]),
            float(bounds["bottom"]),
        ),
        minimap=minimap,
        selected_variant=selected_variant,
        surfaces=surfaces,
        walls=walls,
        ropes=ropes,
        portals=portals,
        life_spawns=spawns,
        physics=physics,
    )
