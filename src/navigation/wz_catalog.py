"""Feature-based recognition of a live minimap against exported WZ canvases."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


class WzMapRecognitionError(RuntimeError):
    """Raised when WZ minimap evidence is absent or ambiguous."""


@dataclass(frozen=True, slots=True)
class CanvasEntry:
    map_id: str
    geometry_path: Path
    canvas_path: Path
    fingerprint_hint: str


@dataclass(slots=True)
class CanvasFeatures:
    entry: CanvasEntry
    bgr: np.ndarray
    alpha: np.ndarray
    keypoints: tuple[cv2.KeyPoint, ...]
    descriptors: np.ndarray


@dataclass(frozen=True, slots=True)
class CanvasRegistration:
    entry: CanvasEntry
    canonical_to_live: np.ndarray
    inlier_count: int
    good_match_count: int
    inlier_ratio: float
    coverage_x: float
    coverage_y: float
    residual_p95_px: float
    scale_x: float
    scale_y: float
    rotation_degrees: float

    @property
    def effective_scale(self) -> float:
        return math.sqrt(self.scale_x * self.scale_y)

    @property
    def rank_key(self) -> tuple[float, ...]:
        return (
            float(self.inlier_count),
            min(self.coverage_x, self.coverage_y),
            self.inlier_ratio,
            float(self.good_match_count),
            -self.residual_p95_px,
        )

    def live_to_canonical(self, point: tuple[float, float]) -> tuple[float, float]:
        inverse = cv2.invertAffineTransform(self.canonical_to_live)
        x, y = point
        return (
            float(inverse[0, 0] * x + inverse[0, 1] * y + inverse[0, 2]),
            float(inverse[1, 0] * x + inverse[1, 1] * y + inverse[1, 2]),
        )


def load_wz_canvas(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a WZ canvas and composite transparent pixels like the game UI."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.ndim != 3 or raw.shape[2] not in (3, 4):
        raise ValueError(f"invalid WZ minimap canvas: {path}")
    if raw.shape[2] == 4:
        alpha = raw[:, :, 3].astype(np.float32) / 255.0
        alpha_3 = alpha[:, :, None]
        background = np.full(raw.shape[:2] + (3,), 72.0, dtype=np.float32)
        bgr = np.rint(
            raw[:, :, :3].astype(np.float32) * alpha_3
            + background * (1.0 - alpha_3)
        ).astype(np.uint8)
        return bgr, alpha
    return raw, np.ones(raw.shape[:2], dtype=np.float32)


def _small_component_mask(
    mask: np.ndarray,
    *,
    maximum_area: int = 256,
    maximum_span: int = 24,
) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8) * 255,
        connectivity=8,
    )
    result = np.zeros(mask.shape, dtype=np.bool_)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) <= maximum_area and \
                int(stats[label, cv2.CC_STAT_WIDTH]) <= maximum_span and \
                int(stats[label, cv2.CC_STAT_HEIGHT]) <= maximum_span:
            result[labels == label] = True
    return result


def _static_grayscale(bgr: np.ndarray) -> np.ndarray:
    if bgr is None or bgr.dtype != np.uint8 or bgr.ndim != 3 or \
            bgr.shape[2] != 3 or bgr.size == 0:
        raise ValueError("minimap raster must be nonempty uint8 BGR")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    yellow_delta = np.max(
        np.abs(
            bgr.astype(np.int16)
            - np.array((136, 255, 255), dtype=np.int16)
        ),
        axis=2,
    )
    yellow = yellow_delta <= 36
    red = (
        ((hsv[:, :, 0] <= 8) | (hsv[:, :, 0] >= 172))
        & (hsv[:, :, 1] >= 150)
        & (hsv[:, :, 2] >= 170)
    )
    dynamic = _small_component_mask(yellow | red)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if np.any(dynamic):
        gray = cv2.inpaint(
            gray,
            dynamic.astype(np.uint8) * 255,
            3,
            cv2.INPAINT_TELEA,
        )
    return cv2.GaussianBlur(gray, (3, 3), 0)


class WzMapCatalog:
    """Lazy catalog that scans canonical canvases only when a map is unknown."""

    def __init__(
        self,
        geometry_directory: str | Path,
        *,
        minimum_inliers: int = 6,
        minimum_inlier_ratio: float = 0.55,
        minimum_inlier_gap: int = 2,
        descriptor_ratio: float = 0.75,
        minimum_scale: float = 0.5,
        maximum_scale: float = 8.0,
        maximum_axis_ratio: float = 1.2,
        maximum_rotation_degrees: float = 3.0,
        maximum_shear_cosine: float = 0.12,
        progress: Callable[[str], None] | None = None,
    ):
        self.geometry_directory = Path(geometry_directory).resolve()
        self.minimum_inliers = max(3, int(minimum_inliers))
        self.minimum_inlier_ratio = float(minimum_inlier_ratio)
        self.minimum_inlier_gap = max(0, int(minimum_inlier_gap))
        self.descriptor_ratio = float(descriptor_ratio)
        self.minimum_scale = float(minimum_scale)
        self.maximum_scale = float(maximum_scale)
        self.maximum_axis_ratio = float(maximum_axis_ratio)
        self.maximum_rotation_degrees = float(maximum_rotation_degrees)
        self.maximum_shear_cosine = float(maximum_shear_cosine)
        self.progress = progress
        self._sift = cv2.SIFT_create(
            nfeatures=700,
            contrastThreshold=0.01,
            edgeThreshold=12,
        )
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        self._selected_features: CanvasFeatures | None = None

    def entries(self, map_ids: set[str] | None = None) -> tuple[CanvasEntry, ...]:
        canvas_root = self.geometry_directory / "canvases"
        if not self.geometry_directory.is_dir() or not canvas_root.is_dir():
            raise WzMapRecognitionError(
                f"WZ geometry cache is missing: {self.geometry_directory}"
            )
        normalized_ids = (
            None if map_ids is None else {str(value).zfill(9) for value in map_ids}
        )
        values: list[CanvasEntry] = []
        for canvas_path in canvas_root.glob("*/*.png"):
            map_id = canvas_path.parent.name
            if len(map_id) != 9 or not map_id.isascii() or not map_id.isdigit():
                continue
            if normalized_ids is not None and map_id not in normalized_ids:
                continue
            geometry_path = self.geometry_directory / f"{map_id}.json"
            if not geometry_path.is_file():
                continue
            values.append(
                CanvasEntry(
                    map_id=map_id,
                    geometry_path=geometry_path,
                    canvas_path=canvas_path.resolve(),
                    fingerprint_hint=canvas_path.stem.removeprefix("canvas-"),
                )
            )
        values.sort(key=lambda item: (item.map_id, item.canvas_path.name))
        if not values:
            raise WzMapRecognitionError("WZ geometry cache contains no canvases")
        return tuple(values)

    def _features(self, entry: CanvasEntry) -> CanvasFeatures | None:
        bgr, alpha = load_wz_canvas(entry.canvas_path)
        keypoints, descriptors = self._sift.detectAndCompute(
            _static_grayscale(bgr), None
        )
        if descriptors is None or len(keypoints) < 3:
            return None
        return CanvasFeatures(
            entry=entry,
            bgr=bgr,
            alpha=alpha,
            keypoints=tuple(keypoints),
            descriptors=descriptors,
        )

    def _registration(
        self,
        canonical: CanvasFeatures,
        live_keypoints: tuple[cv2.KeyPoint, ...],
        live_descriptors: np.ndarray,
    ) -> CanvasRegistration | None:
        matches = self._matcher.knnMatch(
            canonical.descriptors,
            live_descriptors,
            k=2,
        )
        good = [
            first
            for first, second in matches
            if first.distance < self.descriptor_ratio * second.distance
        ]
        if len(good) < 3:
            return None
        source = np.float32(
            [canonical.keypoints[item.queryIdx].pt for item in good]
        )
        target = np.float32(
            [live_keypoints[item.trainIdx].pt for item in good]
        )
        affine, inlier_mask = cv2.estimateAffine2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=4.0,
            maxIters=1000,
            confidence=0.995,
            refineIters=10,
        )
        if affine is None or inlier_mask is None:
            return None
        if not np.all(np.isfinite(affine)):
            return None
        inliers = inlier_mask.reshape(-1).astype(bool)
        inlier_count = int(np.count_nonzero(inliers))
        inlier_ratio = inlier_count / len(good)
        if inlier_count < 3 or inlier_ratio < self.minimum_inlier_ratio:
            return None

        a, b, _ = affine[0]
        c, d, _ = affine[1]
        determinant = a * d - b * c
        scale_x = math.hypot(a, c)
        scale_y = math.hypot(b, d)
        if determinant <= 0.0 or min(scale_x, scale_y) <= 0.0:
            return None
        axis_ratio = max(scale_x, scale_y) / min(scale_x, scale_y)
        shear_denominator = scale_x * scale_y
        shear_cosine = abs(a * b + c * d) / shear_denominator
        rotation = math.degrees(math.atan2(c, a))
        if not (
            self.minimum_scale <= min(scale_x, scale_y)
            and max(scale_x, scale_y) <= self.maximum_scale
            and axis_ratio <= self.maximum_axis_ratio
            and shear_cosine <= self.maximum_shear_cosine
            and abs(rotation) <= self.maximum_rotation_degrees
        ):
            return None

        inlier_source = source[inliers]
        predicted = cv2.transform(inlier_source[None, :, :], affine)[0]
        residuals = np.linalg.norm(predicted - target[inliers], axis=1)
        height, width = canonical.bgr.shape[:2]
        coverage_x = (
            float(np.ptp(inlier_source[:, 0])) / max(1.0, float(width - 1))
        )
        coverage_y = (
            float(np.ptp(inlier_source[:, 1])) / max(1.0, float(height - 1))
        )
        return CanvasRegistration(
            entry=canonical.entry,
            canonical_to_live=affine.astype(np.float64),
            inlier_count=inlier_count,
            good_match_count=len(good),
            inlier_ratio=inlier_ratio,
            coverage_x=coverage_x,
            coverage_y=coverage_y,
            residual_p95_px=float(np.percentile(residuals, 95)),
            scale_x=scale_x,
            scale_y=scale_y,
            rotation_degrees=rotation,
        )

    def _live_features(
        self, live_bgr: np.ndarray
    ) -> tuple[tuple[cv2.KeyPoint, ...], np.ndarray]:
        keypoints, descriptors = self._sift.detectAndCompute(
            _static_grayscale(live_bgr), None
        )
        if descriptors is None or len(keypoints) < 3:
            raise WzMapRecognitionError(
                "current minimap has insufficient static features"
            )
        return tuple(keypoints), descriptors

    def recognize(
        self,
        live_bgr: np.ndarray,
        *,
        map_ids: set[str] | None = None,
    ) -> tuple[CanvasRegistration, CanvasFeatures]:
        """Find one unique map ID using SIFT matches and a sane affine."""
        live_keypoints, live_descriptors = self._live_features(live_bgr)
        entries = self.entries(map_ids)
        if self.progress is not None:
            self.progress(
                f"Scanning {len(entries)} WZ minimap canvas(es)"
            )
        candidates: list[tuple[CanvasRegistration, CanvasFeatures]] = []
        for entry in entries:
            try:
                canonical = self._features(entry)
            except (OSError, ValueError):
                continue
            if canonical is None:
                continue
            registration = self._registration(
                canonical,
                live_keypoints,
                live_descriptors,
            )
            if registration is not None:
                candidates.append((registration, canonical))
        candidates.sort(key=lambda item: item[0].rank_key, reverse=True)
        if not candidates or candidates[0][0].inlier_count < self.minimum_inliers:
            raise WzMapRecognitionError(
                "no WZ minimap passed the feature/inlier threshold"
            )
        best, features = candidates[0]
        if len(candidates) > 1:
            second = candidates[1][0]
            same_raster = (
                best.entry.fingerprint_hint == second.entry.fingerprint_hint
            )
            too_close = (
                best.inlier_count - second.inlier_count
                < self.minimum_inlier_gap
            )
            if same_raster or too_close:
                raise WzMapRecognitionError(
                    "WZ minimap match is ambiguous between "
                    f"{best.entry.map_id} and {second.entry.map_id}"
                )
        self._selected_features = features
        if self.progress is not None:
            self.progress(
                f"Matched WZ map {best.entry.map_id}: "
                f"{best.inlier_count}/{best.good_match_count} inliers, "
                f"scale={best.effective_scale:.3f}"
            )
        return best, features

    def register_selected(
        self,
        live_bgr: np.ndarray,
    ) -> CanvasRegistration | None:
        """Re-register the selected canvas to a possibly moving viewport."""
        canonical = self._selected_features
        if canonical is None:
            return None
        try:
            live_keypoints, live_descriptors = self._live_features(live_bgr)
        except WzMapRecognitionError:
            return None
        registration = self._registration(
            canonical,
            live_keypoints,
            live_descriptors,
        )
        if registration is None or registration.inlier_count < self.minimum_inliers:
            return None
        return registration

    def select(self, features: CanvasFeatures) -> None:
        self._selected_features = features

    def clear_selection(self) -> None:
        self._selected_features = None

    @property
    def selected_features(self) -> CanvasFeatures | None:
        return self._selected_features
