"""Fail-closed Chinese RapidOCR page and target localization for recovery.

Every recovery page is classified from configured Chinese text.  A click
point is returned only when exactly one configured target is present.
"""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata
from typing import Any, Optional, Sequence, Tuple

import numpy as np


Point = Tuple[int, int]
Box = Tuple[Point, Point, Point, Point]
Region = Tuple[int, int, int, int]


class RapidOcrError(RuntimeError):
    """Raised when the configured OCR backend cannot run reliably."""


def normalize_ocr_text(value: str) -> str:
    """Normalize harmless Unicode/spacing differences without fuzzy matching."""
    normalized = unicodedata.normalize("NFKC", str(value))
    return "".join(char for char in normalized if not char.isspace())


def is_chinese_ocr_target(value: str) -> bool:
    """Accept Chinese UI targets and reject Korean/Japanese script aliases.

    Digits and punctuation are allowed because labels such as ``4.漂漂猪``
    are valid targets.  At least one Han ideograph must remain after Unicode
    normalization, while Hangul and kana are rejected explicitly.
    """
    text = normalize_ocr_text(value)
    has_han = any(
        "\u3400" <= char <= "\u4dbf"
        or "\u4e00" <= char <= "\u9fff"
        or "\uf900" <= char <= "\ufaff"
        for char in text
    )
    has_hangul_or_kana = any(
        "\u1100" <= char <= "\u11ff"
        or "\u3040" <= char <= "\u30ff"
        or "\u3130" <= char <= "\u318f"
        or "\uac00" <= char <= "\ud7af"
        for char in text
    )
    return has_han and not has_hangul_or_kana


def matches_ocr_target(
        text: str, targets: Sequence[str], match_mode: str) -> bool:
    """Match normalized OCR text, including a safe unique partial mode."""
    if match_mode == "exact":
        return text in targets
    if match_mode == "contains":
        return any(target in text for target in targets)
    if match_mode == "partial":
        # Recognition can lose characters hidden by the mouse cursor.  Accept
        # any shared meaningful character, including just ``4``, ``漂``, or
        # ``猪``.  This also covers corrupted output such as ``4.漂画猪``.
        # Punctuation alone is never enough, and the caller still requires
        # exactly one spatial candidate.
        return any(
            target in text or any(
                char.isalnum() and char in text for char in target
            )
            for target in targets
        )
    return False


@dataclass(frozen=True)
class OcrTextMatch:
    """One OCR result expressed in current full-frame coordinates."""

    text: str
    normalized_text: str
    score: float
    box: Box
    center: Point


class RapidOcrTextLocator:
    """Locate a unique configured text target inside one bounded frame ROI.

    ``engine`` is injectable for deterministic tests.  In production the
    RapidOCR models are initialized lazily on the first actual login target,
    keeping ordinary bot startup and gameplay free of OCR work.
    """

    def __init__(self, engine: Optional[Any] = None) -> None:
        self._engine = engine

    def _get_engine(self):
        if self._engine is not None:
            return self._engine
        try:
            from rapidocr import RapidOCR

            # Pin every OCR stage to RapidOCR's Chinese model family.  The
            # package defaults are currently Chinese too, but spelling this
            # out prevents a user/global RapidOCR config from switching the
            # recovery flow to a Korean, Japanese, or Latin recognizer.
            self._engine = RapidOCR(params={
                "Det.lang_type": "ch",
                "Cls.lang_type": "ch",
                "Rec.lang_type": "ch",
                "Global.log_level": "warning",
            })
        except Exception as exc:
            raise RapidOcrError(
                "RapidOCR could not initialize its local ONNX models"
            ) from exc
        return self._engine

    @staticmethod
    def _validated_region(frame: np.ndarray, region: Sequence[int]) -> Region:
        if not isinstance(frame, np.ndarray) or frame.size == 0 or \
                frame.ndim not in (2, 3):
            raise ValueError("OCR frame must be a non-empty image array")
        if not isinstance(region, (list, tuple)) or len(region) != 4 or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in region):
            raise ValueError("OCR region must contain four integer values")
        x0, y0, x1, y1 = map(int, region)
        frame_h, frame_w = frame.shape[:2]
        if not (0 <= x0 < x1 <= frame_w and 0 <= y0 < y1 <= frame_h):
            raise ValueError("OCR region must lie inside the current frame")
        return x0, y0, x1, y1

    @staticmethod
    def _box_in_full_frame(
            box: Any, region: Region, roi_shape: Sequence[int]) -> Optional[Box]:
        try:
            points = np.asarray(box, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if points.shape != (4, 2) or not np.all(np.isfinite(points)):
            return None

        roi_h, roi_w = map(int, roi_shape[:2])
        xs = points[:, 0]
        ys = points[:, 1]
        if np.any(xs < 0) or np.any(xs > roi_w - 1) or \
                np.any(ys < 0) or np.any(ys > roi_h - 1):
            return None
        if float(np.max(xs) - np.min(xs)) < 1.0 or \
                float(np.max(ys) - np.min(ys)) < 1.0:
            return None

        # Reject zero-area/self-degenerate quadrilaterals instead of deriving
        # a plausible-looking click point from corrupted backend output.
        area = 0.5 * abs(float(
            np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))
        ))
        if area < 1.0:
            return None

        x0, y0, _, _ = region
        return tuple(
            (
                int(round(float(x) + x0)),
                int(round(float(y) + y0)),
            )
            for x, y in points
        )

    @staticmethod
    def _matches_target(
            text: str, targets: Sequence[str], match_mode: str) -> bool:
        return matches_ocr_target(text, targets, match_mode)

    @staticmethod
    def _validated_thresholds(min_score, box_threshold) -> Tuple[float, float]:
        """Validate inference thresholds shared by scan and target lookup."""
        try:
            checked_min_score = float(min_score)
            checked_box_threshold = float(box_threshold)
        except (TypeError, ValueError):
            raise ValueError("OCR thresholds must be numeric") from None
        if isinstance(min_score, bool) or not np.isfinite(
                checked_min_score) or not 0.0 < checked_min_score <= 1.0:
            raise ValueError("OCR min_score must be in (0, 1]")
        if isinstance(box_threshold, bool) or not np.isfinite(
                checked_box_threshold
                ) or not 0.0 < checked_box_threshold <= 1.0:
            raise ValueError("OCR box_threshold must be in (0, 1]")
        return checked_min_score, checked_box_threshold

    def recognize(
        self,
        frame: np.ndarray,
        region: Sequence[int],
        *,
        min_score: float,
        box_threshold: float = 0.3,
    ) -> Optional[Tuple[OcrTextMatch, ...]]:
        """Recognize validated text once inside a bounded frame ROI."""
        checked_region = self._validated_region(frame, region)
        checked_min_score, checked_box_threshold = \
            self._validated_thresholds(min_score, box_threshold)

        x0, y0, x1, y1 = checked_region
        roi = np.ascontiguousarray(frame[y0:y1, x0:x1])
        try:
            result = self._get_engine()(
                roi,
                use_cls=False,
                return_word_box=False,
                text_score=checked_min_score,
                box_thresh=checked_box_threshold,
            )
        except RapidOcrError:
            raise
        except Exception as exc:
            raise RapidOcrError("RapidOCR inference failed") from exc

        boxes = getattr(result, "boxes", None)
        texts = getattr(result, "txts", None)
        scores = getattr(result, "scores", None)
        if boxes is None or texts is None or scores is None:
            return None
        try:
            item_count = len(boxes)
            text_count = len(texts)
            score_count = len(scores)
        except TypeError:
            return None
        if text_count != item_count or score_count != item_count:
            return None

        matches = []
        for box, raw_text, raw_score in zip(boxes, texts, scores):
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(score):
                return None
            full_box = self._box_in_full_frame(
                box, checked_region, roi.shape
            )
            if full_box is None:
                return None
            text = str(raw_text)
            normalized_text = normalize_ocr_text(text)
            # Keep numeric fragments such as the ``4`` in ``4.漂漂猪``.  Click
            # targets themselves remain Chinese-validated, so unrelated Latin
            # or numeric OCR results cannot authorize an action.
            if not normalized_text:
                continue
            if score < checked_min_score:
                continue
            center = (
                int(round(sum(point[0] for point in full_box) / 4.0)),
                int(round(sum(point[1] for point in full_box) / 4.0)),
            )
            matches.append(OcrTextMatch(
                text=text,
                normalized_text=normalized_text,
                score=score,
                box=full_box,
                center=center,
            ))

        return tuple(matches)

    def locate(
        self,
        frame: np.ndarray,
        region: Sequence[int],
        targets: Sequence[str],
        *,
        min_score: float,
        match_mode: str = "exact",
        box_threshold: float = 0.3,
    ) -> Optional[OcrTextMatch]:
        """Return exactly one accepted target, otherwise return ``None``."""
        if not isinstance(targets, (list, tuple)) or not targets:
            raise ValueError("OCR targets must be a non-empty sequence")
        normalized_targets = tuple(
            normalize_ocr_text(target) for target in targets
        )
        if any(not target for target in normalized_targets):
            raise ValueError("OCR targets must not normalize to empty text")
        if any(
                not is_chinese_ocr_target(target)
                for target in normalized_targets):
            raise ValueError(
                "OCR targets must contain Chinese Han text and no Hangul/kana"
            )
        checked_mode = str(match_mode).strip().lower()
        if checked_mode not in {"exact", "contains", "partial"}:
            raise ValueError(
                "OCR match_mode must be exact, contains, or partial"
            )

        recognized = self.recognize(
            frame,
            region,
            min_score=min_score,
            box_threshold=box_threshold,
        )
        if recognized is None:
            return None
        matches = [
            match for match in recognized
            if self._matches_target(
                match.normalized_text, normalized_targets, checked_mode
            )
        ]

        # Never choose a highest score among duplicate clickable labels.  The
        # state machine must get one spatially unambiguous target or no point.
        return matches[0] if len(matches) == 1 else None


class StableOcrTargetGate:
    """Require a unique OCR target to remain stable on fresh capture frames."""

    def __init__(
        self,
        *,
        confirm_frames: int = 2,
        max_center_drift: Sequence[int] = (24, 24),
    ) -> None:
        if isinstance(confirm_frames, bool) or not isinstance(
                confirm_frames, int) or confirm_frames < 1:
            raise ValueError("confirm_frames must be a positive integer")
        if not isinstance(max_center_drift, (list, tuple)) or \
                len(max_center_drift) != 2 or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in max_center_drift
                ):
            raise ValueError(
                "max_center_drift must contain two non-negative integers"
            )
        self.confirm_frames = confirm_frames
        self.max_center_drift = tuple(map(int, max_center_drift))
        self.reset()

    def reset(self) -> None:
        self._page = None
        self._last_frame_token = None
        self._last_center = None
        self._confirm_count = 0

    def observe(
        self,
        page: str,
        frame_token: Any,
        match: Optional[OcrTextMatch],
    ) -> Optional[Point]:
        """Return a click point only on a fresh, sufficiently stable match."""
        if frame_token == self._last_frame_token:
            # A duplicate capture cannot add positive evidence. A contradictory
            # observation on that token is still allowed to revoke an earlier
            # candidate, while the token remains consumed.
            if match is None or page != self._page:
                self._page = None
                self._last_center = None
                self._confirm_count = 0
            return None
        self._last_frame_token = frame_token

        if match is None:
            self._page = None
            self._last_center = None
            self._confirm_count = 0
            return None

        center = tuple(map(int, match.center))
        same_page = page == self._page
        stable_center = self._last_center is not None and all(
            abs(center[index] - self._last_center[index])
            <= self.max_center_drift[index]
            for index in (0, 1)
        )
        if same_page and stable_center:
            self._confirm_count += 1
        else:
            self._confirm_count = 1
        self._page = page
        self._last_center = center

        if self._confirm_count < self.confirm_frames:
            return None
        return center
