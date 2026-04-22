# esafe/detectors/child_detector.py
"""
영유아 위험 감지기
- 어린이 분류: 박스 높이 < 중간 성인 높이의 50%
- 단독 탑승: 어린이 근처 200px 이내 성인 없음
- 캐리어 탑승: 어린이와 캐리어 IoU > 0.3
"""

from typing import Tuple, Optional
import numpy as np


class ChildRiskDetector:
    """영유아 위험(단독 탑승/캐리어 탑승) 감지 클래스"""

    def __init__(self, adult_height_ratio: float = 0.5, nearby_radius: int = 200):
        """
        초기화
        
        Args:
            adult_height_ratio: 어린이 판정 기준 (기본: 중간 성인 높이의 50%)
            nearby_radius: 보호자 근처 거리 (default: 200px)
        """
        self.adult_height_ratio = adult_height_ratio
        self.nearby_radius = nearby_radius
        self.adult_heights = []

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        두 박스 사이 IoU(Intersection over Union) 계산
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
            
        Returns:
            IoU 값 (0~1)
        """
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        return inter_area / union_area

    def _get_median_adult_height(self, person_boxes: np.ndarray) -> float:
        """
        현재 프레임에서 성인의 중간 높이 계산
        
        Args:
            person_boxes: 사람 박스 [[x1, y1, x2, y2], ...]
            
        Returns:
            중간 높이 값
        """
        if len(person_boxes) == 0:
            return 100  # 기본값
        
        heights = person_boxes[:, 3] - person_boxes[:, 1]
        return np.median(heights)

    def _is_child(self, person_box: np.ndarray, median_adult_height: float) -> bool:
        """
        사람이 어린이인지 판정
        
        Args:
            person_box: [x1, y1, x2, y2]
            median_adult_height: 프레임 내 중간 성인 높이
            
        Returns:
            True if child
        """
        height = person_box[3] - person_box[1]
        threshold = median_adult_height * self.adult_height_ratio
        return height < threshold

    def update(self, person_boxes: np.ndarray, carrier_boxes: Optional[np.ndarray] = None) -> Tuple[bool, str, float]:
        """
        영유아 위험 상태 업데이트
        
        Args:
            person_boxes: 사람 박스 [[x1, y1, x2, y2], ...] (Nx4)
            carrier_boxes: 캐리어 박스 [[x1, y1, x2, y2], ...] (Mx4), 옵션
            
        Returns:
            (is_danger: bool, label: str, confidence: float)
        """
        is_danger = False
        danger_label = ""
        confidence = 0.0
        
        if len(person_boxes) == 0:
            return is_danger, danger_label, confidence
        
        # 중간 성인 높이 계산
        median_adult_height = self._get_median_adult_height(person_boxes)
        
        children = []
        adults = []
        
        # 어린이/성인 분류
        for idx, person_box in enumerate(person_boxes):
            if self._is_child(person_box, median_adult_height):
                children.append((idx, person_box))
            else:
                adults.append(person_box)
        
        # ===== 각 어린이 검사 =====
        for child_idx, child_box in children:
            cx = (child_box[0] + child_box[2]) / 2
            cy = (child_box[1] + child_box[3]) / 2
            
            # ===== 단독 탑승: 근처 성인 없음 =====
            has_nearby_adult = False
            for adult_box in adults:
                ax = (adult_box[0] + adult_box[2]) / 2
                ay = (adult_box[1] + adult_box[3]) / 2
                dist = np.sqrt((cx - ax) ** 2 + (cy - ay) ** 2)
                
                if dist < self.nearby_radius:
                    has_nearby_adult = True
                    break
            
            if not has_nearby_adult:
                is_danger = True
                danger_label = "단독탑승"
                confidence = 0.9
                return is_danger, danger_label, confidence
            
            # ===== 캐리어 탑승: 어린이와 캐리어 IoU > 0.3 =====
            if carrier_boxes is not None and len(carrier_boxes) > 0:
                for carrier_box in carrier_boxes:
                    iou = self._compute_iou(child_box, carrier_box)
                    
                    if iou > 0.3:
                        is_danger = True
                        danger_label = "캐리어탑승"
                        confidence = 0.95
                        return is_danger, danger_label, confidence
        
        return is_danger, danger_label, confidence

    def reset(self):
        """상태 초기화"""
        self.adult_heights.clear()
