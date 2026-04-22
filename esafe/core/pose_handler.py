# esafe/core/pose_handler.py
"""
MediaPipe Pose 핸들러
- 자세 추정 (선택적 - 미사용시 앱 계속 실행)
"""

from typing import Optional, List
import numpy as np

HAS_MEDIAPIPE = False
mp_pose = None

# MediaPipe 임포트 시도
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
        mp_pose = mp.solutions.pose
        HAS_MEDIAPIPE = True
except (ImportError, AttributeError):
    pass  # MediaPipe 사용 불가


class PoseHandler:
    """MediaPipe Pose 핸들러 클래스 (선택적)"""

    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        초기화
        
        Args:
            min_detection_confidence: 최소 감지 신뢰도 (default: 0.5)
            min_tracking_confidence: 최소 추적 신뢰도 (default: 0.5)
        """
        self.pose = None
        self.available = False
        
        if not HAS_MEDIAPIPE or mp_pose is None:
            return
        
        try:
            self.pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.available = True
        except Exception as e:
            print(f"Pose 초기화 실패: {e}")
            self.available = False

    def estimate(self, frame: np.ndarray):
        """
        프레임에서 자세 추정
        
        Args:
            frame: 입력 프레임 (HxWx3 RGB)
            
        Returns:
            results: MediaPipe pose results 또는 None
        """
        if not self.available or self.pose is None:
            return None
        
        try:
            results = self.pose.process(frame)
            return results
        except Exception as e:
            print(f"자세 추정 실패: {e}")
            return None

    def get_landmarks(self, results) -> Optional[List]:
        """
        결과에서 랜드마크 추출
        
        Args:
            results: MediaPipe pose results
            
        Returns:
            랜드마크 리스트 또는 None
        """
        if results is None or not hasattr(results, 'pose_landmarks') or results.pose_landmarks is None:
            return None
        
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append({
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': lm.visibility
            })
        
        return landmarks
