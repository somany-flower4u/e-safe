# esafe/detectors/carrier_detector.py
"""
캐리어/수하물 위험 감지기
- 방치: 3초 이상 정지된 캐리어
- 낙하: 빠른 하향 움직임
- 미감시: 근처에 사람이 없는 캐리어
"""

from collections import deque
from typing import Tuple, Optional
import numpy as np


class CarrierHazardDetector:
    """캐리어 위험(방치/낙하/미감시) 감지 클래스"""

    def __init__(self, stationary_threshold: int = 10, stationary_frames: int = 3,
                 velocity_threshold: float = 15.0, person_radius: int = 150):
        """
        초기화
        
        Args:
            stationary_threshold: 정지 판정 픽셀 이동 거리 (default: 10px)
            stationary_frames: 방치 판정 필요 프레임 수 (default: 3 = ~0.1초 at 30fps -> 3초 = 90frames)
            velocity_threshold: 낙하 판정 Y 속도 임계값 (default: 15.0)
            person_radius: 미감시 판정 거리 (default: 150px)
        """
        self.stationary_threshold = stationary_threshold
        self.stationary_frames = stationary_frames * 30  # 3초를 프레임으로 변환 (30fps 기준)
        self.velocity_threshold = velocity_threshold
        self.person_radius = person_radius
        
        # 캐리어별 추적: {track_id: deque of (x, y)}
        self.carrier_history = {}
        self.stationary_count = {}

    def update(self, carrier_boxes: np.ndarray, carrier_ids: np.ndarray,
               person_boxes: Optional[np.ndarray] = None) -> Tuple[bool, str, float]:
        """
        캐리어 위험 상태 업데이트
        
        Args:
            carrier_boxes: 캐리어 박스 [[x1, y1, x2, y2], ...] 형태 (Nx4)
            carrier_ids: 각 캐리어의 추적 ID (N,)
            person_boxes: 사람 박스 [[x1, y1, x2, y2], ...] 형태 (Mx4), 옵션
            
        Returns:
            (is_danger: bool, label: str, confidence: float)
        """
        is_danger = False
        danger_label = ""
        confidence = 0.0
        
        # 현재 프레임의 캐리어 ID 집합
        current_ids = set(carrier_ids)
        
        # 사라진 캐리어 정리
        for cid in list(self.carrier_history.keys()):
            if cid not in current_ids:
                del self.carrier_history[cid]
                if cid in self.stationary_count:
                    del self.stationary_count[cid]
        
        # 각 캐리어 처리
        for idx, carrier_id in enumerate(carrier_ids):
            x1, y1, x2, y2 = carrier_boxes[idx]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            # 이력 초기화
            if carrier_id not in self.carrier_history:
                self.carrier_history[carrier_id] = deque(maxlen=5)
                self.stationary_count[carrier_id] = 0
            
            history = self.carrier_history[carrier_id]
            history.append((cx, cy))
            
            # ===== 낙하 감지: 빠른 하향 움직임 =====
            if len(history) >= 2:
                prev_cy = history[-2][1]
                curr_cy = cy
                y_velocity = curr_cy - prev_cy  # 양수 = 하향
                
                if y_velocity > self.velocity_threshold:
                    is_danger = True
                    danger_label = "낙하"
                    confidence = min(0.95, y_velocity / 30.0)
                    return is_danger, danger_label, confidence
            
            # ===== 방치 감지: 3초 이상 정지 =====
            movement = 0
            if len(history) >= 2:
                prev_cx, prev_cy = history[-2]
                movement = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
            
            if movement < self.stationary_threshold:
                self.stationary_count[carrier_id] += 1
            else:
                self.stationary_count[carrier_id] = 0
            
            if self.stationary_count[carrier_id] >= self.stationary_frames:
                is_danger = True
                danger_label = "방치"
                confidence = 0.9
                return is_danger, danger_label, confidence
            
            # ===== 미감시 감지: 근처 사람 없음 =====
            if person_boxes is not None and len(person_boxes) > 0:
                min_dist = float('inf')
                for px1, py1, px2, py2 in person_boxes:
                    pcx = (px1 + px2) / 2
                    pcy = (py1 + py2) / 2
                    dist = np.sqrt((cx - pcx) ** 2 + (cy - pcy) ** 2)
                    min_dist = min(min_dist, dist)
                
                # 미감시는 자동 위험은 아니지만 상태 반환 (별도 처리 가능)
                if min_dist > self.person_radius:
                    # UNATTENDED 상태 - 별도 레벨링 미적용, 정보용
                    pass
        
        return is_danger, danger_label, confidence

    def reset(self):
        """상태 초기화"""
        self.carrier_history.clear()
        self.stationary_count.clear()
