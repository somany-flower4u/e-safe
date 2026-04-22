# esafe/detectors/direction_detector.py
"""유형 3(역방향 진입) 감지 모듈입니다."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Tuple


def _bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """바운딩 박스 중심 좌표를 반환합니다."""
    x1, y1, x2, y2 = bbox
    return (float((x1 + x2) * 0.5), float((y1 + y2) * 0.5))


def _calculate_direction_angle(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """두 점 사이의 방향 각도를 계산합니다 (라디안)."""
    import math
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.atan2(dy, dx)


def _is_moving_opposite(
    angle: float, expected_sign: int, angle_tolerance: float = 0.5
) -> bool:
    """계산된 각도가 반대 방향인지 확인합니다."""
    import math
    # expected_sign > 0: 아래쪽(+Y), expected_sign < 0: 위쪽(-Y)
    if expected_sign > 0:
        # 아래로 가야 함 (각도 ~90도 또는 -90도)
        # 반대는 위쪽 (각도 ~-90도 또는 90도의 반대)
        opposite_range_lower = math.pi - angle_tolerance
        opposite_range_upper = -math.pi + angle_tolerance
        return angle > opposite_range_lower or angle < opposite_range_upper
    else:
        # 위로 가야 함 (각도 ~-90도 또는 90도)
        # 반대는 아래쪽 (각도 ~90도)
        opposite_range_lower = -math.pi / 2 - angle_tolerance
        opposite_range_upper = -math.pi / 2 + angle_tolerance
        return opposite_range_lower < angle < opposite_range_upper or angle > math.pi / 2 - angle_tolerance


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
        min_speed_px_per_frame: float = 1.0,
        hold_seconds: float = 1.5,
        required_consecutive_frames: int = 3,
        angle_tolerance: float = 0.6,
    ) -> None:
        """감지기 동작 파라미터를 초기화합니다."""
        self.fps = max(1.0, fps)
        self.min_speed_px_per_frame = min_speed_px_per_frame
        self.hold_frames = int(hold_seconds * self.fps)
        self.required_consecutive_frames = max(1, required_consecutive_frames)
        self.angle_tolerance = angle_tolerance

        self.frame_count = 0
        self._confirm_counter = 0
        self._paths: Dict[int, Deque[Tuple[float, float]]] = {}
        self._wrong_frames: Dict[int, int] = {}
        self._direction_history: Dict[int, Deque[bool]] = {}

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
        import math
        
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
            path = self._paths.setdefault(track_id, deque(maxlen=8))
            path.append(center)
            self._wrong_frames.setdefault(track_id, 0)
            self._direction_history.setdefault(track_id, deque(maxlen=5))

            if len(path) < 2:
                continue

            # 최근 2점 사이의 거리로 속도 계산
            dx = path[-1][0] - path[-2][0]
            dy = path[-1][1] - path[-2][1]
            speed = math.sqrt(dx * dx + dy * dy)

            # 속도가 너무 낮으면 스킵 (정지 상태)
            if speed < self.min_speed_px_per_frame * 0.5:
                self._wrong_frames[track_id] = 0
                self._direction_history[track_id].append(False)
                continue

            # 최근 이동 방향 각도 계산
            angle = _calculate_direction_angle(path[-2], path[-1])
            
            # 반대 방향 판정
            moving_opposite = _is_moving_opposite(angle, expected_sign, self.angle_tolerance)
            self._direction_history[track_id].append(moving_opposite)

            # 일관성 검증: 최근 방향 이력의 일관성 확인
            if len(self._direction_history[track_id]) >= 3:
                consistency = sum(self._direction_history[track_id]) / len(self._direction_history[track_id])
            else:
                consistency = 1.0 if moving_opposite else 0.0

            # 일관성이 높으면 카운트 증가
            if consistency >= 0.6:
                self._wrong_frames[track_id] += 1
            else:
                self._wrong_frames[track_id] = max(0, self._wrong_frames[track_id] - 1)

            if self._wrong_frames[track_id] >= self.hold_frames:
                candidate = True
                offender_id = track_id
                location = (int(center[0]), int(center[1]))
                # 신뢰도: 속도 + 일관성 + 지속 시간
                speed_factor = min(1.0, speed / max(1.0, self.min_speed_px_per_frame * 10.0)) * 0.25
                consistency_factor = consistency * 0.35
                duration_factor = min(1.0, self._wrong_frames[track_id] / max(1.0, self.hold_frames * 2.0)) * 0.4
                confidence = max(
                    confidence,
                    min(0.95, 0.65 + speed_factor + consistency_factor + duration_factor),
                )

        stale_ids = [track_id for track_id in self._paths if track_id not in seen_ids]
        for stale_id in stale_ids:
            self._paths.pop(stale_id, None)
            self._wrong_frames.pop(stale_id, None)
            self._direction_history.pop(stale_id, None)

        confirmed = self._update_confirm(candidate)
        if confirmed:
            return True, max(0.70, confidence), {
                "hazard_type": "wrong_direction",
                "label": "역방향 진입",
                "location": location,
                "track_id": offender_id,
            }

        return False, 0.0, {"hazard_type": None, "label": "정상", "location": None, "track_id": None}
