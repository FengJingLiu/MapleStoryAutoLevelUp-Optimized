"""Rebuild hero templates from a native-resolution game-content frame.

The source templates are only used as visual/mask references.  Matching first
performs a coarse isotropic search and then refines width and height scales
independently around the best candidates.  This makes the tool suitable for
capture-card output whose horizontal and vertical UI scaling differ slightly.

Examples::

    python -m tools.rebuild_native_hero_templates \
        --frame screenshot/potplayer_fullscreen_content.png \
        --pose anchors --dry-run

    python -m tools.rebuild_native_hero_templates \
        --frame screenshot/player_facing_right.png \
        --pose stand_right

Dry runs never replace a template.  They still write a timestamped annotated
preview under ``<output-dir>/previews``.  A real run backs up every destination
that it is about to replace before writing any new template.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil
import sys
from typing import Iterable, Sequence

import cv2
import numpy as np


GREEN = (0, 255, 0)
POSES = ("stand_right", "stand_left", "climb", "anchors")


@dataclass(frozen=True)
class TemplateSpec:
    key: str
    filename: str
    masked: bool


@dataclass(frozen=True)
class MatchResult:
    spec: TemplateSpec
    location: tuple[int, int]
    size: tuple[int, int]
    scale_x: float
    scale_y: float
    score: float
    source_template: np.ndarray
    source_mask: np.ndarray | None

    @property
    def box(self) -> tuple[int, int, int, int]:
        x, y = self.location
        width, height = self.size
        return x, y, x + width, y + height


def read_image(path: Path | str) -> np.ndarray:
    """Read an image without relying on OpenCV's Unicode path handling."""
    path = Path(path)
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def write_image(path: Path | str, image: np.ndarray) -> None:
    """Encode and atomically replace one PNG/JPEG image."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"Unable to encode image: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    encoded.tofile(temporary)
    temporary.replace(path)


def template_mask(template: np.ndarray) -> np.ndarray | None:
    """Return the existing exact-green foreground mask, if one is present."""
    green_pixels = np.all(template == GREEN, axis=2)
    if not np.any(green_pixels):
        return None
    return ((~green_pixels) * 255).astype(np.uint8)


def _scaled_reference(
    template: np.ndarray,
    mask: np.ndarray | None,
    scale_x: float,
    scale_y: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    width = max(2, int(round(template.shape[1] * scale_x)))
    height = max(2, int(round(template.shape[0] * scale_y)))
    scaled = cv2.resize(
        template,
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )
    scaled_mask = None
    if mask is not None:
        scaled_mask = cv2.resize(
            mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    return scaled, scaled_mask


def _match_score_map(
    search_gray: np.ndarray,
    template: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[float, tuple[int, int]] | None:
    if (
        template.shape[0] > search_gray.shape[0]
        or template.shape[1] > search_gray.shape[1]
    ):
        return None
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(
        search_gray,
        template_gray,
        cv2.TM_SQDIFF_NORMED,
        mask=mask,
    )
    result = np.nan_to_num(result, nan=1.0, posinf=1.0, neginf=1.0)
    score, _, location, _ = cv2.minMaxLoc(result)
    return float(score), location


def _float_range(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("Scale step must be positive")
    values = []
    current = start
    while current <= stop + step * 0.25:
        values.append(round(current, 6))
        current += step
    return values


def _clip_search_box(
    frame: np.ndarray,
    search_box: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int]:
    frame_h, frame_w = frame.shape[:2]
    if search_box is None:
        return 0, 0, frame_w, frame_h
    x0, y0, x1, y1 = map(int, search_box)
    x0 = min(frame_w, max(0, x0))
    y0 = min(frame_h, max(0, y0))
    x1 = min(frame_w, max(x0, x1))
    y1 = min(frame_h, max(y0, y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Empty search box after clipping: {search_box}")
    return x0, y0, x1, y1


def find_multiscale_match(
    frame: np.ndarray,
    spec: TemplateSpec,
    source_template: np.ndarray,
    *,
    scale_min: float = 1.0,
    scale_max: float = 4.0,
    coarse_step: float = 0.10,
    refine_radius: float = 0.30,
    refine_step: float = 0.025,
    coarse_max_dimension: int = 1600,
    search_box: tuple[int, int, int, int] | None = None,
    top_candidates: int = 3,
) -> MatchResult:
    """Locate a template with coarse isotropic and refined per-axis scales."""
    if frame is None or frame.ndim != 3 or frame.size == 0:
        raise ValueError("Native frame is empty")
    if scale_min <= 0 or scale_max < scale_min:
        raise ValueError(f"Invalid scale range: {scale_min}..{scale_max}")

    source_mask = template_mask(source_template) if spec.masked else None
    x0, y0, x1, y1 = _clip_search_box(frame, search_box)
    search = frame[y0:y1, x0:x1]

    maximum_dimension = max(search.shape[:2])
    coarse_factor = min(
        1.0,
        max(0.1, float(coarse_max_dimension) / maximum_dimension),
    )
    if coarse_factor < 1.0:
        coarse_search = cv2.resize(
            search,
            (
                max(2, int(round(search.shape[1] * coarse_factor))),
                max(2, int(round(search.shape[0] * coarse_factor))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    else:
        coarse_search = search
    coarse_gray = cv2.cvtColor(coarse_search, cv2.COLOR_BGR2GRAY)

    candidates: list[tuple[float, float, tuple[int, int]]] = []
    for scale in _float_range(scale_min, scale_max, coarse_step):
        scaled, scaled_mask = _scaled_reference(
            source_template,
            source_mask,
            scale * coarse_factor,
            scale * coarse_factor,
        )
        result = _match_score_map(coarse_gray, scaled, scaled_mask)
        if result is None:
            continue
        score, location = result
        native_location = (
            int(round(location[0] / coarse_factor)),
            int(round(location[1] / coarse_factor)),
        )
        candidates.append((score, scale, native_location))

    if not candidates:
        raise ValueError(
            f"Template {spec.filename} does not fit inside the search frame"
        )

    # Adjacent coarse scales usually report the same object.  Keep a few
    # distinct locations so refinement can recover if the lowest coarse score
    # came from a compression/interpolation artefact.
    distinct: list[tuple[float, float, tuple[int, int]]] = []
    for candidate in sorted(candidates):
        _, scale, location = candidate
        duplicate = any(
            abs(location[0] - old[2][0]) <= 8
            and abs(location[1] - old[2][1]) <= 8
            and abs(scale - old[1]) <= coarse_step * 1.5
            for old in distinct
        )
        if not duplicate:
            distinct.append(candidate)
        if len(distinct) >= max(1, int(top_candidates)):
            break
    if not distinct:
        distinct.append(min(candidates))

    best: MatchResult | None = None
    search_gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    for _, coarse_scale, coarse_location in distinct:
        coarse_width = int(round(source_template.shape[1] * coarse_scale))
        coarse_height = int(round(source_template.shape[0] * coarse_scale))
        padding = max(24, int(round(max(coarse_width, coarse_height) * 0.55)))
        local_x0 = max(0, coarse_location[0] - padding)
        local_y0 = max(0, coarse_location[1] - padding)
        local_x1 = min(
            search.shape[1], coarse_location[0] + coarse_width + padding
        )
        local_y1 = min(
            search.shape[0], coarse_location[1] + coarse_height + padding
        )
        local_gray = search_gray[local_y0:local_y1, local_x0:local_x1]

        refine_min = max(scale_min, coarse_scale - refine_radius)
        refine_max = min(scale_max, coarse_scale + refine_radius)
        scale_x_values = _float_range(refine_min, refine_max, refine_step)
        scale_y_values = _float_range(refine_min, refine_max, refine_step)
        for scale_x in scale_x_values:
            for scale_y in scale_y_values:
                scaled, scaled_mask = _scaled_reference(
                    source_template,
                    source_mask,
                    scale_x,
                    scale_y,
                )
                result = _match_score_map(local_gray, scaled, scaled_mask)
                if result is None:
                    continue
                score, location = result
                if best is not None and score >= best.score:
                    continue
                best = MatchResult(
                    spec=spec,
                    location=(
                        x0 + local_x0 + location[0],
                        y0 + local_y0 + location[1],
                    ),
                    size=(scaled.shape[1], scaled.shape[0]),
                    scale_x=scale_x,
                    scale_y=scale_y,
                    score=score,
                    source_template=source_template,
                    source_mask=source_mask,
                )

    if best is None:
        raise ValueError(f"Unable to refine match for {spec.filename}")
    return best


def extract_matched_template(
    frame: np.ndarray,
    match: MatchResult,
) -> np.ndarray:
    """Extract one exact match rectangle and restore its green transparency."""
    x0, y0, x1, y1 = match.box
    if x0 < 0 or y0 < 0 or x1 > frame.shape[1] or y1 > frame.shape[0]:
        raise ValueError(f"Match lies outside frame: {match.box}")
    extracted = frame[y0:y1, x0:x1].copy()
    if not match.spec.masked:
        return extracted

    if match.source_mask is None:
        raise ValueError(f"Masked source has no green mask: {match.spec.filename}")
    scaled_mask = cv2.resize(
        match.source_mask,
        match.size,
        interpolation=cv2.INTER_NEAREST,
    )
    extracted[scaled_mask == 0] = GREEN
    return extracted


def specs_for_pose(name: str, pose: str) -> tuple[TemplateSpec, ...]:
    if pose == "anchors":
        return (
            TemplateSpec("id", f"{name}.png", True),
            TemplateSpec("medal", f"{name}_medal.png", False),
            TemplateSpec("pet", f"{name}_pet.png", False),
        )
    suffix = {
        "stand_right": "appearance_stand_right",
        "stand_left": "appearance_stand_left",
        "climb": "appearance_climb",
    }.get(pose)
    if suffix is None:
        raise ValueError(f"Unsupported pose: {pose}")
    return (TemplateSpec("appearance", f"{name}_{suffix}.png", True),)


def _related_anchor_search_box(
    frame: np.ndarray,
    key: str,
    matches: dict[str, MatchResult],
) -> tuple[int, int, int, int] | None:
    """Restrict generic medal/pet text to the configured hero geometry."""
    frame_h, frame_w = frame.shape[:2]
    if key == "medal" and "id" in matches:
        identity = matches["id"]
        expected_width = int(round(96 * identity.scale_x))
        expected_height = int(round(14 * identity.scale_y))
        expected_x = int(round(
            identity.location[0]
            + identity.size[0] / 2
            - expected_width / 2
            + 3 * identity.scale_x
        ))
        expected_y = identity.location[1] + identity.size[1]
        margin_x = max(80, int(round(30 * identity.scale_x)))
        margin_y = max(45, int(round(18 * identity.scale_y)))
        return (
            max(0, expected_x - margin_x),
            max(0, expected_y - margin_y),
            min(frame_w, expected_x + expected_width + margin_x),
            min(frame_h, expected_y + expected_height + margin_y),
        )
    if key == "pet" and "medal" in matches:
        medal = matches["medal"]
        expected_width = int(round(53 * medal.scale_x))
        expected_height = int(round(14 * medal.scale_y))
        expected_x = int(round(medal.location[0] - 37 * medal.scale_x))
        expected_y = int(round(medal.location[1] - 17 * medal.scale_y))
        margin_x = max(100, int(round(40 * medal.scale_x)))
        margin_y = max(60, int(round(24 * medal.scale_y)))
        return (
            max(0, expected_x - margin_x),
            max(0, expected_y - margin_y),
            min(frame_w, expected_x + expected_width + margin_x),
            min(frame_h, expected_y + expected_height + margin_y),
        )
    return None


def rebuild_from_frame(
    frame: np.ndarray,
    *,
    pose: str,
    name: str,
    source_dir: Path,
    scale_min: float = 1.0,
    scale_max: float = 4.0,
    coarse_step: float = 0.10,
    refine_radius: float = 0.30,
    refine_step: float = 0.025,
    coarse_max_dimension: int = 1600,
    max_score: float = 0.30,
) -> tuple[list[MatchResult], dict[str, np.ndarray]]:
    """Match and extract all templates requested by one pose."""
    matches: dict[str, MatchResult] = {}
    extracted: dict[str, np.ndarray] = {}
    results: list[MatchResult] = []
    for spec in specs_for_pose(name, pose):
        source_path = source_dir / spec.filename
        if not source_path.is_file():
            raise FileNotFoundError(f"Source template not found: {source_path}")
        source_template = read_image(source_path)
        match = find_multiscale_match(
            frame,
            spec,
            source_template,
            scale_min=scale_min,
            scale_max=scale_max,
            coarse_step=coarse_step,
            refine_radius=refine_radius,
            refine_step=refine_step,
            coarse_max_dimension=coarse_max_dimension,
            search_box=_related_anchor_search_box(frame, spec.key, matches),
        )
        if match.score > max_score:
            raise RuntimeError(
                f"Unsafe {spec.key} match score {match.score:.4f} exceeds "
                f"--max-score {max_score:.4f}; no templates were written"
            )
        matches[spec.key] = match
        results.append(match)
        extracted[spec.filename] = extract_matched_template(frame, match)
    return results, extracted


def render_preview(
    frame: np.ndarray,
    matches: Iterable[MatchResult],
) -> np.ndarray:
    preview = frame.copy()
    thickness = max(2, int(round(max(frame.shape[:2]) / 1200)))
    font_scale = max(0.6, max(frame.shape[:2]) / 2200)
    for index, match in enumerate(matches):
        x0, y0, x1, y1 = match.box
        color = ((37 + index * 83) % 255, (220 - index * 47) % 255, 255)
        cv2.rectangle(preview, (x0, y0), (x1, y1), color, thickness)
        label = (
            f"{match.spec.key} score={match.score:.3f} "
            f"scale=({match.scale_x:.3f},{match.scale_y:.3f}) "
            f"size={match.size[0]}x{match.size[1]}"
        )
        text_y = max(24, y0 - 8)
        cv2.putText(
            preview,
            label,
            (x0, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return preview


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def save_preview(
    preview: np.ndarray,
    *,
    frame_path: Path,
    pose: str,
    output_dir: Path,
) -> Path:
    preview_dir = output_dir / "previews"
    preview_path = (
        preview_dir / f"{frame_path.stem}_{pose}_{_timestamp()}_preview.jpg"
    )
    write_image(preview_path, preview)
    return preview_path


def install_templates(
    extracted: dict[str, np.ndarray],
    *,
    output_dir: Path,
    backup_dir: Path | None = None,
) -> tuple[list[Path], Path | None]:
    """Back up all destinations first, then atomically install outputs."""
    destinations = {
        filename: output_dir / filename for filename in extracted
    }
    existing = [path for path in destinations.values() if path.exists()]
    actual_backup_dir = backup_dir
    if existing and actual_backup_dir is None:
        actual_backup_dir = output_dir / "backups" / _timestamp()
    if actual_backup_dir is not None and existing:
        actual_backup_dir.mkdir(parents=True, exist_ok=True)
        for destination in existing:
            backup_path = actual_backup_dir / destination.name
            if backup_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing backup: {backup_path}"
                )
        for destination in existing:
            shutil.copy2(destination, actual_backup_dir / destination.name)

    written = []
    for filename, image in extracted.items():
        destination = destinations[filename]
        write_image(destination, image)
        written.append(destination)
    return written, actual_backup_dir if existing else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild native-resolution hero templates safely",
    )
    parser.add_argument("--frame", required=True, type=Path)
    parser.add_argument("--pose", required=True, choices=POSES)
    parser.add_argument("--name", default="liu_muning")
    parser.add_argument("--source-dir", type=Path, default=Path("nametag"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory; defaults to --source-dir",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        help="Backup directory; defaults to a timestamped output-dir/backups path",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--scale-min", type=float, default=1.0)
    parser.add_argument("--scale-max", type=float, default=4.0)
    parser.add_argument("--coarse-step", type=float, default=0.10)
    parser.add_argument("--refine-radius", type=float, default=0.30)
    parser.add_argument("--refine-step", type=float, default=0.025)
    parser.add_argument("--coarse-max-dimension", type=int, default=1600)
    parser.add_argument(
        "--max-score",
        type=float,
        default=0.30,
        help="Abort without writing if any SQDIFF score exceeds this value",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or args.source_dir
    frame = read_image(args.frame)
    matches, extracted = rebuild_from_frame(
        frame,
        pose=args.pose,
        name=args.name,
        source_dir=args.source_dir,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        coarse_step=args.coarse_step,
        refine_radius=args.refine_radius,
        refine_step=args.refine_step,
        coarse_max_dimension=args.coarse_max_dimension,
        max_score=args.max_score,
    )
    for match in matches:
        print(
            f"MATCH {match.spec.key}: loc={match.location} "
            f"size={match.size} score={match.score:.6f} "
            f"scale=({match.scale_x:.3f},{match.scale_y:.3f})"
        )

    preview_path = save_preview(
        render_preview(frame, matches),
        frame_path=args.frame,
        pose=args.pose,
        output_dir=output_dir,
    )
    print(f"PREVIEW {preview_path}")

    if args.dry_run:
        print("DRY RUN: no hero template was written or replaced")
        return 0

    written, backup_path = install_templates(
        extracted,
        output_dir=output_dir,
        backup_dir=args.backup_dir,
    )
    if backup_path is not None:
        print(f"BACKUP {backup_path}")
    for path in written:
        print(f"SAVED {path}")
    return 0


def main() -> None:
    try:
        raise SystemExit(run())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
