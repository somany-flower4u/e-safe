# esafe/utils/roi_manager.py
"""
ROI (관심 영역) 관리
"""

from typing import Tuple


class ROIManager:
    """ROI 관리 클래스"""

    def __init__(self, default_roi: Tuple[float, float, float, float] = (0.15, 0.1, 0.85, 0.9)):
        """
        초기화
        
        Args:
            default_roi: 기본 ROI 영역 (x_min, y_min, x_max, y_max) - 정규화 좌표 (0~1)
                        기본: 중앙 70% (x: 15%~85%, y: 10%~90%)
        """
        self.roi = default_roi

    def set_roi(self, roi: Tuple[float, float, float, float]):
        """
        ROI 설정
        
        Args:
            roi: (x_min, y_min, x_max, y_max) - 정규화 좌표
        """
        self.roi = roi

    def get_roi(self) -> Tuple[float, float, float, float]:
        """
        현재 ROI 반환
        
        Returns:
            (x_min, y_min, x_max, y_max)
        """
        return self.roi

    def point_in_roi(self, x: float, y: float, normalized: bool = False) -> bool:
        """
        점이 ROI 내에 있는지 확인
        
        Args:
            x, y: 좌표
            normalized: True면 정규화 좌표 (0~1), False면 픽셀 좌표
            
        Returns:
            True if inside ROI
        """
        x_min, y_min, x_max, y_max = self.roi
        
        if normalized:
            return x_min <= x <= x_max and y_min <= y <= y_max
        else:
            # 픽셀 좌표는 정규화 필요 (요청자가 해야함)
            return x_min <= x <= x_max and y_min <= y <= y_max

    def box_in_roi(self, box: Tuple[float, float, float, float],
                   threshold: float = 0.5) -> bool:
        """
        박스가 ROI 내에 있는지 확인 (교집합 비율)
        
        Args:
            box: [x1, y1, x2, y2] - 픽셀 좌표
            threshold: 교집합 비율 임계값 (default: 0.5 = 50% 이상)
            
        Returns:
            True if box is mostly inside ROI
        """
        # 이 함수는 실제로는 프레임 크기를 알아야 하므로
        # 호출자가 정규화된 좌표로 제공해야 함
        x1, y1, x2, y2 = box
        x_min, y_min, x_max, y_max = self.roi
        
        # 교집합 계산
        inter_x1 = max(x1, x_min)
        inter_y1 = max(y1, y_min)
        inter_x2 = min(x2, x_max)
        inter_y2 = min(y2, y_max)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return False
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        box_area = (x2 - x1) * (y2 - y1)
        
        if box_area == 0:
            return False
        
        overlap_ratio = inter_area / box_area
        return overlap_ratio >= threshold

    def reset(self):
        """ROI를 기본값으로 초기화"""
        self.roi = (0.15, 0.1, 0.85, 0.9)
