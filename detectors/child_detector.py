# esafe/detectors/child_detector.py
"""유형 4(영유아 단독 탑승·캐리어 탑승) 감지 모듈입니다."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np


def _bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """바운딩 박스 중심 좌표를 반환합니다."""
    x1, y1, x2, y2 = bbox
    return (float((x1 + x2) * 0.5), float((y1 + y2) * 0.5))


def _bbox_height(bbox: Tuple[float, float, float, float]) -> float:
    """바운딩 박스 높이를 계산합니다."""
    return float(max(0.0, bbox[3] - bbox[1]))


def _bbox_width(bbox: Tuple[float, float, float, float]) -> float:
    """바운딩 박스 폭을 계산합니다."""
    return float(max(0.0, bbox[2] - bbox[0]))


def _bbox_aspect_ratio(bbox: Tuple[float, float, float, float]) -> float:
    """바운딩 박스 종횡비(높이/폭)를 계산합니다."""
    height = _bbox_height(bbox)
    width = _bbox_width(bbox)
    return height / max(0.01, width)


def _point_in_roi(point: Tuple[float, float], roi: Tuple[int, int, int, int]) -> bool:
    """점이 ROI 내부인지 확인합니다."""
    px, py = point
    rx1, ry1, rx2, ry2 = roi
    return rx1 <= px <= rx2 and ry1 <= py <= ry2


def _point_in_bbox(point: Tuple[float, float], bbox: Tuple[float, float, float, float]) -> bool:
    """점이 바운딩 박스 내부인지 확인합니다."""
    px, py = point
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2


def _bbox_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """두 바운딩 박스의 IoU를 계산합니다."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union_area = max(1e-6, area_a + area_b - inter_area)
    return inter_area / union_area


class ChildRiskDetector:
    """사람/캐리어 검출 결과를 바탕으로 영유아 위험 상황을 감지합니다."""

    def __init__(
        self,
        child_height_ratio: float = 0.65,
        required_consecutive_frames: int = 3,
        min_adult_ratio: float = 0.65,
    ) -> None:
        """감지기 파라미터를 초기화합니다."""
        self.child_height_ratio = child_height_ratio
        self.min_adult_ratio = min_adult_ratio
        self.required_consecutive_frames = max(1, required_consecutive_frames)

        self.frame_count = 0
        self._alone_counter = 0
        self._carrier_counter = 0

    def _update_counter(self, counter_name: str, detected: bool) -> bool:
        """연속 감지 카운터를 업데이트합니다."""
        if counter_name == "alone":
            self._alone_counter = self._alone_counter + 1 if detected else 0
            return self._alone_counter >= self.required_consecutive_frames

        self._carrier_counter = self._carrier_counter + 1 if detected else 0
        return self._carrier_counter >= self.required_consecutive_frames

    def _meter_to_pixels(
        self,
        frame_shape: Tuple[int, int, int],
        roi: Tuple[int, int, int, int],
        meter: float = 1.0,
    ) -> float:
        """근사 픽셀-미터 변환값을 계산합니다."""
        _, frame_width, _ = frame_shape
        rx1, _, rx2, _ = roi
        roi_width = max(1, rx2 - rx1)
        base_scale = max(40.0, min(160.0, roi_width * 0.18))
        frame_scale = max(0.8, min(1.2, frame_width / 640.0))
        return base_scale * frame_scale * meter

    def detect(
        self,
        persons: List[Dict[str, Any]],
        suitcases: List[Dict[str, Any]],
        roi: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """현재 프레임의 영유아 위험 여부(bool)와 신뢰도(float)를 반환합니다."""
        self.frame_count += 1

        persons_in_roi: List[Dict[str, Any]] = []
        for person in persons:
            bbox = tuple(float(v) for v in person.get("bbox", (0, 0, 0, 0)))
            center = _bbox_center(bbox)
            if _point_in_roi(center, roi):
                enriched = dict(person)
                enriched["bbox"] = bbox
                enriched["center"] = center
                enriched["height"] = _bbox_height(bbox)
                persons_in_roi.append(enriched)

        if not persons_in_roi:
            self._update_counter("alone", False)
            self._update_counter("carrier", False)
            return False, 0.0, {"hazard_type": None, "label": "정상", "child_ids": [], "location": None}

        heights = [person["height"] for person in persons_in_roi if person["height"] > 0]
        if not heights:
            return False, 0.0, {"hazard_type": None, "label": "정상", "child_ids": [], "location": None}

        max_height = max(heights)
        # 성인 기준: 최대 높이의 65% 이상인 사람들의 평균값 사용
        adult_candidates = [h for h in heights if h >= max_height * self.min_adult_ratio]
        adult_mean_height = float(np.mean(adult_candidates)) if adult_candidates else max_height

        child_people: List[Dict[str, Any]] = []
        adult_people: List[Dict[str, Any]] = []
        
        for person in persons_in_roi:
            person_height = person["height"]
            # 영유아 판정: 성인 평균 높이의 65% 이하이고, 절대 높이도 검증
            if person_height <= adult_mean_height * self.child_height_ratio:
                child_people.append(person)
            else:
                adult_people.append(person)

        if not child_people:
            self._update_counter("alone", False)
            self._update_counter("carrier", False)
            return False, 0.0, {"hazard_type": None, "label": "정상", "child_ids": [], "location": None}

        radius_px = self._meter_to_pixels(frame_shape, roi, meter=1.0)

        alone_candidate = False
        carrier_candidate = False
        alone_confidence = 0.0
        carrier_confidence = 0.0
        location: Tuple[int, int] | None = None
        child_ids = [int(child.get("track_id", -1)) for child in child_people]

        for child in child_people:
            child_center = child["center"]
            child_bbox = child["bbox"]
            child_height_ratio = child["height"] / max(0.01, adult_mean_height)

            if not adult_people:
                # 성인이 없는 경우: 높이 비율이 낮을수록 신뢰도 증가
                alone_candidate = True
                alone_confidence = max(alone_confidence, min(0.95, 0.75 + (1.0 - child_height_ratio) * 0.2))
                location = (int(child_center[0]), int(child_center[1]))
            else:
                # 성인이 있는 경우: 거리 기반 판정
                nearest_dist = min(
                    math.dist(child_center, adult["center"])
                    for adult in adult_people
                    if int(adult.get("track_id", -1)) != int(child.get("track_id", -1))
                )
                if nearest_dist > radius_px:
                    alone_candidate = True
                    # 거리가 멀수록 신뢰도 증가, 높이 비율도 반영
                    distance_factor = min(1.0, (nearest_dist - radius_px) / max(1.0, radius_px * 2.5))
                    height_factor = max(0.0, 1.0 - child_height_ratio) * 0.15
                    alone_confidence = max(
                        alone_confidence,
                        min(0.95, 0.65 + distance_factor * 0.25 + height_factor),
                    )
                    location = (int(child_center[0]), int(child_center[1]))

            for suitcase in suitcases:
                suitcase_bbox = tuple(float(v) for v in suitcase.get("bbox", (0, 0, 0, 0)))
                overlap = _bbox_iou(child_bbox, suitcase_bbox)
                if _point_in_bbox(child_center, suitcase_bbox) or overlap >= 0.1:
                    carrier_candidate = True
                    carrier_confidence = max(carrier_confidence, min(0.95, 0.65 + overlap * 0.3))
                    location = (int(child_center[0]), int(child_center[1]))
                    break

        carrier_confirmed = self._update_counter("carrier", carrier_candidate)
        alone_confirmed = self._update_counter("alone", alone_candidate)

        if carrier_confirmed:
            return True, max(0.70, carrier_confidence), {
                "hazard_type": "child_on_carrier",
                "label": "영유아 캐리어 탑승",
                "child_ids": child_ids,
                "location": location,
            }

        if alone_confirmed:
            return True, max(0.65, alone_confidence), {
                "hazard_type": "child_alone",
                "label": "영유아 단독 탑승",
                "child_ids": child_ids,
                "location": location,
            }

        return False, 0.0, {"hazard_type": None, "label": "정상", "child_ids": child_ids, "location": None}
