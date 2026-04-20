# esafe/core/tracker.py
"""다중 객체 추적(기본 ByteTrack) 래퍼 모듈입니다."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import supervision as sv
except Exception:  # pragma: no cover - 환경 의존성 예외 보호
    sv = None


def _bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """바운딩 박스 중심 좌표를 계산합니다."""
    x1, y1, x2, y2 = bbox
    return (float((x1 + x2) * 0.5), float((y1 + y2) * 0.5))


class ByteTrackerWrapper:
    """supervision ByteTrack 기반 추적 결과를 통합 포맷으로 제공합니다."""

    def __init__(self, fps: float = 30.0) -> None:
        """ByteTrack(또는 대체 추적기)를 초기화합니다."""
        self.fps = int(max(1.0, fps))
        self.tracker = None
        self._fallback_next_id = 1
        self._fallback_tracks: Dict[int, Dict[str, Any]] = {}

        if sv is not None:
            try:
                self.tracker = sv.ByteTrack(frame_rate=self.fps)
            except Exception:
                self.tracker = None

    def _empty_detections(self) -> Any:
        """빈 supervision Detections 객체를 생성합니다."""
        if sv is None:
            return None
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.array([], dtype=np.float32),
            class_id=np.array([], dtype=np.int32),
        )

    def _fallback_track(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """supervision 미사용 시 간단한 중심점 매칭 추적을 수행합니다."""
        if not detections:
            # 오래된 트랙은 자연스럽게 제거합니다.
            stale_ids = []
            for track_id, track in self._fallback_tracks.items():
                track["missing"] = int(track.get("missing", 0)) + 1
                if track["missing"] > 30:
                    stale_ids.append(track_id)
            for stale_id in stale_ids:
                self._fallback_tracks.pop(stale_id, None)
            return []

        # 이번 프레임에서 아직 매칭되지 않은 트랙은 먼저 누락 카운트를 증가시킵니다.
        for track in self._fallback_tracks.values():
            track["missing"] = int(track.get("missing", 0)) + 1

        assigned_tracks = set()
        tracked_output: List[Dict[str, Any]] = []

        for det in detections:
            bbox = tuple(float(v) for v in det["bbox"])
            center = _bbox_center(bbox)
            class_id = int(det.get("class_id", -1))

            best_track_id = None
            best_dist = float("inf")
            for track_id, track in self._fallback_tracks.items():
                if track_id in assigned_tracks:
                    continue
                if int(track.get("class_id", -1)) != class_id:
                    continue
                dist = math.dist(center, track.get("center", center))
                if dist < best_dist and dist <= 60.0:
                    best_dist = dist
                    best_track_id = track_id

            if best_track_id is None:
                best_track_id = self._fallback_next_id
                self._fallback_next_id += 1

            self._fallback_tracks[best_track_id] = {
                "center": center,
                "class_id": class_id,
                "missing": 0,
            }
            assigned_tracks.add(best_track_id)

            tracked_output.append(
                {
                    "track_id": best_track_id,
                    "bbox": bbox,
                    "confidence": float(det.get("confidence", 0.0)),
                    "class_id": class_id,
                    "class_name": det.get("class_name", "unknown"),
                }
            )

        stale_ids = [
            track_id
            for track_id in self._fallback_tracks
            if track_id not in assigned_tracks and self._fallback_tracks[track_id].get("missing", 0) > 30
        ]
        for stale_id in stale_ids:
            self._fallback_tracks.pop(stale_id, None)

        return tracked_output

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """현재 프레임 검출 결과를 입력받아 트래킹 ID가 부여된 결과를 반환합니다."""
        if self.tracker is None or sv is None:
            return self._fallback_track(detections)

        if not detections:
            try:
                self.tracker.update_with_detections(self._empty_detections())
            except Exception:
                pass
            return []

        xyxy = np.array([det["bbox"] for det in detections], dtype=np.float32)
        confidence = np.array([det.get("confidence", 0.0) for det in detections], dtype=np.float32)
        class_id = np.array([det.get("class_id", -1) for det in detections], dtype=np.int32)

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

        try:
            tracked = self.tracker.update_with_detections(sv_detections)
        except Exception:
            return self._fallback_track(detections)

        tracker_ids = tracked.tracker_id
        if tracker_ids is None:
            tracker_ids = np.full((len(tracked.xyxy),), -1, dtype=np.int32)

        output: List[Dict[str, Any]] = []
        for idx in range(len(tracked.xyxy)):
            cid = int(tracked.class_id[idx]) if tracked.class_id is not None else -1
            cname = "person" if cid == 0 else "suitcase" if cid == 28 else f"class_{cid}"
            output.append(
                {
                    "track_id": int(tracker_ids[idx]),
                    "bbox": tuple(float(v) for v in tracked.xyxy[idx]),
                    "confidence": float(tracked.confidence[idx]) if tracked.confidence is not None else 0.0,
                    "class_id": cid,
                    "class_name": cname,
                }
            )

        return output
