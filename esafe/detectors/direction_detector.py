# esafe/detectors/direction_detector.py
"""
에스컬레이터 역방향 진입 감지기
- 5프레임 슬라이딩 윈도우로 사람 이동 방향 추적
- 에스컬레이터 방향 반대 방향 1.5초 이상 진입 감지
"""

from collections import deque
from typing import Tuple, Optional
import numpy as np


class WrongDirectionDetector:
    """역방향 진입 감지 클래스"""

    def __init__(self, window_frames: int = 5, min_duration_frames: int = 45,
                 speed_threshold: float = 3.0):
        """
        초기화
        
        Args:
            window_frames: 슬라이딩 윈도우 프레임 수 (default: 5)
            min_duration_frames: 역방향 판정 필요 프레임 수 (default: 45 = 1.5초 at 30fps)
            speed_threshold: 정지 판정 속도 임계값 (default: 3.0 px/frame)
        """
        self.window_frames = window_frames
        self.min_duration_frames = min_duration_frames
        self.speed_threshold = speed_threshold
        
        # 사람별 추적: {track_id: deque of (cx, cy)}
        self.person_trajectory = {}
        self.wrong_direction_count = {}

    def update(self, person_boxes: np.ndarray, person_ids: Optional[np.ndarray] = None,
               escalator_direction: str = "down") -> Tuple[bool, str, float]:
        """
        역방향 진입 상태 업데이트
        
        Args:
            person_boxes: 사람 박스 [[x1, y1, x2, y2], ...] (Nx4)
            person_ids: 각 사람의 추적 ID (N,), 옵션
            escalator_direction: 에스컬레이터 방향 "down" 또는 "up"
            
        Returns:
            (is_danger: bool, label: str, confidence: float)
        """
        is_danger = False
        danger_label = ""
        confidence = 0.0
        
        if len(person_boxes) == 0:
            return is_danger, danger_label, confidence
        
        # person_ids가 없으면 인덱스 사용
        if person_ids is None:
            person_ids = np.arange(len(person_boxes))
        
        current_ids = set(person_ids)
        
        # 사라진 사람 정리
        for pid in list(self.person_trajectory.keys()):
            if pid not in current_ids:
                del self.person_trajectory[pid]
                if pid in self.wrong_direction_count:
                    del self.wrong_direction_count[pid]
        
        # 각 사람 처리
        for idx, person_id in enumerate(person_ids):
            x1, y1, x2, y2 = person_boxes[idx]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            # 초기화
            if person_id not in self.person_trajectory:
                self.person_trajectory[person_id] = deque(maxlen=self.window_frames)
                self.wrong_direction_count[person_id] = 0
            
            trajectory = self.person_trajectory[person_id]
            trajectory.append((cx, cy))
            
            # 슬라이딩 윈도우에서 평균 움직임 벡터 계산
            if len(trajectory) >= 2:
                # 첫 번째와 마지막 위치로 벡터 계산
                start_x, start_y = trajectory[0]
                end_x, end_y = trajectory[-1]
                
                move_x = end_x - start_x
                move_y = end_y - start_y
                
                speed = np.sqrt(move_x ** 2 + move_y ** 2) / len(trajectory)
                
                # 정지 상태 건너뛰기
                if speed < self.speed_threshold:
                    self.wrong_direction_count[person_id] = 0
                    continue
                
                # ===== 방향 판정 =====
                # escalator_direction = "down": 위→아래 (Y 증가) 방향이 정상
                # 역방향: 아래→위 (Y 감소) 방향
                
                if escalator_direction == "down":
                    # 정상: move_y > 0 (아래로)
                    # 역방향: move_y < 0 (위로)
                    is_wrong = move_y < 0
                else:  # "up"
                    # 정상: move_y < 0 (위로)
                    # 역방향: move_y > 0 (아래로)
                    is_wrong = move_y > 0
                
                if is_wrong:
                    self.wrong_direction_count[person_id] += 1
                else:
                    self.wrong_direction_count[person_id] = 0
                
                # ===== 역방향 확정 =====
                if self.wrong_direction_count[person_id] >= self.min_duration_frames:
                    is_danger = True
                    danger_label = "역방향"
                    confidence = 0.9
                    return is_danger, danger_label, confidence
        
        return is_danger, danger_label, confidence

    def reset(self):
        """상태 초기화"""
        self.person_trajectory.clear()
        self.wrong_direction_count.clear()
