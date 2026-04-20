# esafe/detectors/crowd_detector.py
"""유형 5(과밀 혼잡) 감지 모듈입니다."""

from __future__ import annotations

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


class CrowdingDetector:
    """입구 ROI 내 인원 수를 기반으로 과밀도를 감지합니다."""

    def __init__(self, required_consecutive_frames: int = 3) -> None:
        """감지기 카운터를 초기화합니다."""
        self.required_consecutive_frames = max(1, required_consecutive_frames)
        self.frame_count = 0
        self._crowding_counter = 0

    def detect(
        self,
        persons: List[Dict[str, Any]],
        entrance_roi: Tuple[int, int, int, int],
        threshold: int,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """현재 프레임의 과밀 위험 여부(bool)와 신뢰도(float)를 반환합니다."""
        self.frame_count += 1

        person_count = 0
        for person in persons:
            if "bbox" not in person:
                continue
            center = _bbox_center(tuple(float(v) for v in person["bbox"]))
            if _point_in_roi(center, entrance_roi):
                person_count += 1

        crowded = person_count > threshold
        self._crowding_counter = self._crowding_counter + 1 if crowded else 0
        confirmed = self._crowding_counter >= self.required_consecutive_frames

        confidence = 0.0
        if crowded:
            overflow = person_count - threshold
            confidence = min(1.0, 0.5 + overflow / max(1, threshold))

        return (
            confirmed,
            confidence,
            {
                "hazard_type": "crowding" if confirmed else None,
                "label": "과밀 혼잡" if confirmed else "정상",
                "count": person_count,
                "threshold": threshold,
            },
        )
