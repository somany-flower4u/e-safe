# esafe/detectors/carrier_detector.py
"""유형 1(캐리어 낙하·방치) 감지 모듈입니다."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


def _bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """바운딩 박스 중심 좌표를 계산합니다."""
    x1, y1, x2, y2 = bbox
    return (float((x1 + x2) * 0.5), float((y1 + y2) * 0.5))


def _point_in_roi(point: Tuple[float, float], roi: Tuple[int, int, int, int]) -> bool:
    """점이 ROI 내부에 있는지 확인합니다."""
    px, py = point
    rx1, ry1, rx2, ry2 = roi
    return rx1 <= px <= rx2 and ry1 <= py <= ry2


class CarrierHazardDetector:
    """캐리어 낙하 및 방치 위험을 프레임 단위로 감지합니다."""

    def __init__(
        self,
        fps: float = 30.0,
        stationary_seconds: float = 3.0,
        drop_speed_threshold_px: float = 16.0,
        stationary_pixel_threshold: float = 8.0,
        required_consecutive_frames: int = 3,
    ) -> None:
        """감지기 파라미터를 초기화합니다."""
        self.fps = max(1.0, fps)
        self.stationary_required_frames = int(stationary_seconds * self.fps)
        self.drop_speed_threshold_px = drop_speed_threshold_px
        self.stationary_pixel_threshold = stationary_pixel_threshold
        self.required_consecutive_frames = max(1, required_consecutive_frames)
        self.no_person_suspicion_frames = int(1.0 * self.fps)
        self.owner_distance_threshold_px = 120.0

        self.frame_count = 0
        self._abandon_counter = 0
        self._drop_counter = 0
        self._states: Dict[int, Dict[str, Any]] = {}

    def _update_counter(self, counter_name: str, detected: bool) -> bool:
        """연속 감지 카운터를 갱신하고 확정 여부를 반환합니다."""
        if counter_name == "abandon":
            self._abandon_counter = self._abandon_counter + 1 if detected else 0
            return self._abandon_counter >= self.required_consecutive_frames

        self._drop_counter = self._drop_counter + 1 if detected else 0
        return self._drop_counter >= self.required_consecutive_frames

    def _nearest_person_distance(
        self, center: Tuple[float, float], person_centers: List[Tuple[float, float]]
    ) -> float:
        """캐리어 중심과 가장 가까운 사람 중심 거리(픽셀)를 계산합니다."""
        if not person_centers:
            return float("inf")
        return min(math.dist(center, p_center) for p_center in person_centers)

    def detect(
        self,
        suitcases: List[Dict[str, Any]],
        persons: List[Dict[str, Any]],
        roi: Tuple[int, int, int, int],
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """현재 프레임에서 캐리어 위험 여부(bool)와 신뢰도(float)를 반환합니다."""
        self.frame_count += 1

        person_centers = [
            _bbox_center(tuple(person["bbox"]))
            for person in persons
            if "bbox" in person and person.get("track_id", -1) >= 0
        ]

        abandon_candidate = False
        drop_candidate = False
        abandon_confidence = 0.0
        drop_confidence = 0.0
        abandon_location: Tuple[int, int] | None = None
        drop_location: Tuple[int, int] | None = None

        seen_ids = set()
        for suitcase in suitcases:
            if "bbox" not in suitcase:
                continue

            track_id = int(suitcase.get("track_id", -1))
            if track_id < 0:
                continue

            bbox = tuple(float(v) for v in suitcase["bbox"])
            center = _bbox_center(bbox)
            if not _point_in_roi(center, roi):
                continue

            seen_ids.add(track_id)
            state = self._states.setdefault(
                track_id,
                {
                    "last_center": center,
                    "last_y": center[1],
                    "stationary_frames": 0,
                    "drop_frames": 0,
                    "no_person_frames": 0,
                    "missing_frames": 0,
                },
            )

            move_dist = math.dist(center, state["last_center"])
            if move_dist <= self.stationary_pixel_threshold:
                state["stationary_frames"] += 1
            else:
                state["stationary_frames"] = 0

            delta_y = center[1] - float(state["last_y"])
            if delta_y > self.drop_speed_threshold_px:
                state["drop_frames"] += 1
            else:
                state["drop_frames"] = 0

            near_person_dist = self._nearest_person_distance(center, person_centers)
            if near_person_dist > self.owner_distance_threshold_px:
                state["no_person_frames"] += 1
            else:
                state["no_person_frames"] = 0

            is_abandon_suspected = (
                state["stationary_frames"] >= self.stationary_required_frames
                or state["no_person_frames"] >= self.no_person_suspicion_frames
            )
            if is_abandon_suspected:
                abandon_candidate = True
                abandon_location = (int(center[0]), int(center[1]))
                conf_stationary = min(
                    1.0, state["stationary_frames"] / max(1, self.stationary_required_frames)
                )
                conf_no_person = min(
                    1.0, state["no_person_frames"] / max(1, self.no_person_suspicion_frames)
                )
                abandon_confidence = max(abandon_confidence, 0.4 * conf_stationary + 0.6 * conf_no_person)

            if state["drop_frames"] > 0:
                drop_candidate = True
                drop_location = (int(center[0]), int(center[1]))
                drop_confidence = max(
                    drop_confidence,
                    min(1.0, delta_y / max(1.0, self.drop_speed_threshold_px * 2.0)),
                )

            state["last_center"] = center
            state["last_y"] = center[1]
            state["missing_frames"] = 0

        # 일정 시간 이상 사라진 트랙 상태는 메모리에서 제거합니다.
        stale_ids = []
        for track_id, state in self._states.items():
            if track_id not in seen_ids:
                state["missing_frames"] += 1
                if state["missing_frames"] > int(self.fps * 5):
                    stale_ids.append(track_id)
        for stale_id in stale_ids:
            self._states.pop(stale_id, None)

        drop_confirmed = self._update_counter("drop", drop_candidate)
        abandon_confirmed = self._update_counter("abandon", abandon_candidate)

        if drop_confirmed:
            return True, max(0.55, drop_confidence), {
                "hazard_type": "carrier_drop",
                "label": "캐리어 낙하",
                "location": drop_location,
            }

        if abandon_confirmed:
            return True, max(0.5, abandon_confidence), {
                "hazard_type": "carrier_abandon",
                "label": "캐리어 방치",
                "location": abandon_location,
            }

        return False, 0.0, {"hazard_type": None, "label": "정상", "location": None}
