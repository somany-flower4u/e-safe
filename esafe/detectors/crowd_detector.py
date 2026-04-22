# esafe/detectors/crowd_detector.py
"""
과밀 혼잡 감지기
- ROI 내 사람 수 추적
- 사용자 설정 임계값 초과 시 위험 판정
- 실시간 게이지 표시용 카운트 히스토리 유지
"""

from collections import deque
from typing import Tuple


class CrowdingDetector:
    """과밀 혼잡 감지 클래스"""

    def __init__(self, default_threshold: int = 5, history_size: int = 100):
        """
        초기화
        
        Args:
            default_threshold: 기본 혼잡 임계값 (default: 5명)
            history_size: 히스토리 크기 (default: 100 프레임)
        """
        self.threshold = default_threshold
        self.count_history = deque(maxlen=history_size)

    def update(self, person_count: int, threshold: int = None) -> Tuple[bool, int, int]:
        """
        혼잡도 업데이트
        
        Args:
            person_count: ROI 내 사람 수
            threshold: 혼잡 임계값 (옵션, 기본값 사용시 None)
            
        Returns:
            (is_danger: bool, count: int, threshold: int)
        """
        if threshold is not None:
            self.threshold = threshold
        
        # 카운트 히스토리 저장
        self.count_history.append(person_count)
        
        # 임계값 초과 판정
        is_danger = person_count > self.threshold
        
        return is_danger, person_count, self.threshold

    def get_history(self):
        """
        카운트 히스토리 반환 (게이지 표시용)
        
        Returns:
            deque of int: 과거 카운트 값들
        """
        return list(self.count_history)

    def reset(self):
        """상태 초기화"""
        self.count_history.clear()
