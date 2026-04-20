# esafe/detectors/fall_detector.py
"""유형 2(사람 쓰러짐·발작) 감지 모듈입니다."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Tuple


def _avg_visible_y(
    keypoints: Dict[str, Dict[str, float]],
    names: List[str],
    min_visibility: float,
) -> float | None:
    """주어진 키포인트 집합에서 가시성 조건을 만족하는 평균 Y 값을 계산합니다."""
    values: List[float] = []
    for name in names:
        point = keypoints.get(name)
        if point is None:
            continue
        if float(point.get("visibility", 0.0)) >= min_visibility:
            values.append(float(point.get("y", 0.0)))

    if not values:
        return None
    return float(sum(values) / len(values))


class FallDetector:
    """Pose 키포인트 기반으로 쓰러짐/의식불명 의심 상태를 감지합니다."""

    def __init__(
        self,
        fps: float = 30.0,
        min_visibility: float = 0.6,
        unconscious_seconds: float = 2.0,
        stable_pixel_threshold: float = 10.0,
        required_consecutive_frames: int = 3,
    ) -> None:
        """감지기에 필요한 임계값을 초기화합니다."""
        self.fps = max(1.0, fps)
        self.min_visibility = min_visibility
        self.unconscious_frames = int(unconscious_seconds * self.fps)
        self.stable_pixel_threshold = stable_pixel_threshold
        self.required_consecutive_frames = max(1, required_consecutive_frames)

        self.frame_count = 0
        self._fall_counter = 0
        self._unconscious_counter = 0
        self._nose_histories: Dict[int, Deque[Tuple[int, float, float]]] = {}

    def _update_counter(self, counter_name: str, detected: bool) -> bool:
        """연속 프레임 카운터를 갱신하고 확정 여부를 반환합니다."""
        if counter_name == "fall":
            self._fall_counter = self._fall_counter + 1 if detected else 0
            return self._fall_counter >= self.required_consecutive_frames

        self._unconscious_counter = self._unconscious_counter + 1 if detected else 0
        return self._unconscious_counter >= self.required_consecutive_frames

    def detect(
        self,
        pose_results: List[Dict[str, Any]],
        frame_shape: Tuple[int, int, int],
        roi: Tuple[int, int, int, int],
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """현재 프레임의 쓰러짐 위험 여부(bool)와 신뢰도(float)를 반환합니다."""
        self.frame_count += 1

        frame_height = int(frame_shape[0])
        rx1, ry1, rx2, ry2 = roi
        floor_level_y = min(frame_height - 1, int(ry2 - (ry2 - ry1) * 0.05))

        fall_candidate = False
        unconscious_candidate = False
        fall_confidence = 0.0
        unconscious_confidence = 0.0
        fall_location: Tuple[int, int] | None = None
        unconscious_location: Tuple[int, int] | None = None

        seen_ids = set()
        for result in pose_results:
            track_id = int(result.get("track_id", -1))
            keypoints = result.get("keypoints", {})
            if track_id < 0 or not keypoints:
                continue

            seen_ids.add(track_id)

            shoulder_y = _avg_visible_y(
                keypoints,
                ["left_shoulder", "right_shoulder"],
                self.min_visibility,
            )
            hip_y = _avg_visible_y(
                keypoints,
                ["left_hip", "right_hip"],
                self.min_visibility,
            )
            knee_y = _avg_visible_y(
                keypoints,
                ["left_knee", "right_knee"],
                self.min_visibility,
            )

            if shoulder_y is not None and hip_y is not None and knee_y is not None:
                # 정상 자세(어깨/엉덩이 < 무릎)와 반대 순서가 나타나면 쓰러짐으로 간주합니다.
                inverted = shoulder_y > knee_y or hip_y > knee_y
                if inverted:
                    fall_candidate = True
                    fall_confidence = max(
                        fall_confidence,
                        min(1.0, ((shoulder_y - knee_y) + (hip_y - knee_y)) / 100.0 + 0.6),
                    )
                    anchor = keypoints.get("nose") or keypoints.get("left_shoulder")
                    if anchor:
                        fall_location = (int(anchor.get("x", 0.0)), int(anchor.get("y", 0.0)))

            nose = keypoints.get("nose")
            if nose and float(nose.get("visibility", 0.0)) >= self.min_visibility:
                nose_x = float(nose.get("x", 0.0))
                nose_y = float(nose.get("y", 0.0))
                history = self._nose_histories.setdefault(track_id, deque(maxlen=int(self.fps * 3)))
                history.append((self.frame_count, nose_x, nose_y))

                if nose_y >= floor_level_y and len(history) >= self.unconscious_frames:
                    duration_frames = history[-1][0] - history[0][0] + 1
                    xs = [row[1] for row in history]
                    ys = [row[2] for row in history]
                    drift = max(max(xs) - min(xs), max(ys) - min(ys))
                    if duration_frames >= self.unconscious_frames and drift <= self.stable_pixel_threshold:
                        unconscious_candidate = True
                        unconscious_location = (int(nose_x), int(nose_y))
                        unconscious_confidence = max(
                            unconscious_confidence,
                            min(1.0, 0.65 + (self.unconscious_frames / max(1.0, duration_frames))),
                        )

        stale_ids = [track_id for track_id in self._nose_histories if track_id not in seen_ids]
        for stale_id in stale_ids:
            self._nose_histories.pop(stale_id, None)

        unconscious_confirmed = self._update_counter("unconscious", unconscious_candidate)
        fall_confirmed = self._update_counter("fall", fall_candidate)

        if unconscious_confirmed:
            return True, max(0.7, unconscious_confidence), {
                "hazard_type": "unconscious",
                "label": "의식 불명 의심",
                "location": unconscious_location,
            }

        if fall_confirmed:
            return True, max(0.6, fall_confidence), {
                "hazard_type": "fall_down",
                "label": "쓰러짐 감지",
                "location": fall_location,
            }

        return False, 0.0, {"hazard_type": None, "label": "정상", "location": None}
