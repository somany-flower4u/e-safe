# esafe/detectors/direction_detector.py
"""유형 3(역방향 진입) 감지 모듈입니다."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Tuple


def _bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """바운딩 박스 중심 좌표를 반환합니다."""
    x1, y1, x2, y2 = bbox
    return (float((x1 + x2) * 0.5), float((y1 + y2) * 0.5))


def _point_in_roi(point: Tuple[float, float], roi: Tuple[int, int, int, int]) -> bool:
    """점이 ROI 내부인지 확인합니다."""
    px, py = point
    rx1, ry1, rx2, ry2 = roi
    return rx1 <= px <= rx2 and ry1 <= py <= ry2


class WrongDirectionDetector:
    """사람 이동 벡터를 기반으로 역방향 진입을 감지합니다."""

    def __init__(
        self,
        fps: float = 30.0,
        min_speed_px_per_frame: float = 2.0,
        hold_seconds: float = 1.5,
        required_consecutive_frames: int = 3,
    ) -> None:
        """감지기 동작 파라미터를 초기화합니다."""
        self.fps = max(1.0, fps)
        self.min_speed_px_per_frame = min_speed_px_per_frame
        self.hold_frames = int(hold_seconds * self.fps)
        self.required_consecutive_frames = max(1, required_consecutive_frames)

        self.frame_count = 0
        self._confirm_counter = 0
        self._paths: Dict[int, Deque[Tuple[float, float]]] = {}
        self._wrong_frames: Dict[int, int] = {}

    def _update_confirm(self, detected: bool) -> bool:
        """역방향 후보의 연속 검출 카운터를 갱신합니다."""
        self._confirm_counter = self._confirm_counter + 1 if detected else 0
        return self._confirm_counter >= self.required_consecutive_frames

    def detect(
        self,
        persons: List[Dict[str, Any]],
        roi: Tuple[int, int, int, int],
        escalator_direction: str,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """현재 프레임의 역방향 위험 여부(bool)와 신뢰도(float)를 반환합니다."""
        self.frame_count += 1

        expected_sign = 1 if escalator_direction == "down" else -1
        candidate = False
        confidence = 0.0
        location: Tuple[int, int] | None = None
        offender_id: int | None = None

        seen_ids = set()
        for person in persons:
            if "bbox" not in person:
                continue

            track_id = int(person.get("track_id", -1))
            if track_id < 0:
                continue

            center = _bbox_center(tuple(float(v) for v in person["bbox"]))
            if not _point_in_roi(center, roi):
                continue

            seen_ids.add(track_id)
            path = self._paths.setdefault(track_id, deque(maxlen=3))
            path.append(center)
            self._wrong_frames.setdefault(track_id, 0)

            if len(path) < 3:
                continue

            vy1 = path[1][1] - path[0][1]
            vy2 = path[2][1] - path[1][1]
            avg_vy = (vy1 + vy2) * 0.5

            if abs(avg_vy) < self.min_speed_px_per_frame:
                self._wrong_frames[track_id] = 0
                continue

            moving_opposite = (avg_vy > 0 and expected_sign < 0) or (avg_vy < 0 and expected_sign > 0)
            if moving_opposite:
                self._wrong_frames[track_id] += 1
            else:
                self._wrong_frames[track_id] = 0

            if self._wrong_frames[track_id] >= self.hold_frames:
                candidate = True
                offender_id = track_id
                location = (int(center[0]), int(center[1]))
                confidence = max(
                    confidence,
                    min(
                        1.0,
                        0.55
                        + abs(avg_vy) / max(1.0, self.min_speed_px_per_frame * 6.0)
                        + self._wrong_frames[track_id] / max(1, self.hold_frames * 2),
                    ),
                )

        stale_ids = [track_id for track_id in self._paths if track_id not in seen_ids]
        for stale_id in stale_ids:
            self._paths.pop(stale_id, None)
            self._wrong_frames.pop(stale_id, None)

        confirmed = self._update_confirm(candidate)
        if confirmed:
            return True, max(0.6, confidence), {
                "hazard_type": "wrong_direction",
                "label": "역방향 진입",
                "location": location,
                "track_id": offender_id,
            }

        return False, 0.0, {"hazard_type": None, "label": "정상", "location": None, "track_id": None}
