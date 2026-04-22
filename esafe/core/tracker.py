# esafe/core/tracker.py
"""
객체 추적 시스템
- supervision ByteTracker (v0.21+ API)
- 사람 및 캐리어 추적
"""

from typing import Tuple, Optional
import numpy as np

try:
    import supervision as sv
    HAS_SUPERVISION = True
except ImportError:
    HAS_SUPERVISION = False


class ObjectTracker:
    """객체 추적 클래스 (supervision ByteTrack 사용)"""

    def __init__(self):
        """초기화"""
        self.tracker = None
        
        if HAS_SUPERVISION:
            # supervision ByteTrack 사용 (최신 API)
            self.tracker = sv.ByteTrack()

    def track(self, boxes: np.ndarray, confidences: np.ndarray,
              class_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        객체 추적 수행
        
        Args:
            boxes: 감지된 박스 [[x1, y1, x2, y2], ...] (Nx4)
            confidences: 신뢰도 (N,)
            class_ids: 클래스 ID (N,)
            
        Returns:
            (tracked_boxes, tracked_ids, confidences, class_ids)
        """
        if self.tracker is None or len(boxes) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        try:
            # supervision Detections 객체 생성
            detections = sv.Detections(
                xyxy=boxes,
                confidence=confidences,
                class_id=class_ids
            )
            
            # 추적 업데이트 (v0.21+ API)
            detections = self.tracker.update_with_detections(detections)
            
            # 추적 ID 추출
            tracker_ids = detections.tracker_id
            
            # ID가 None인 경우 처리
            if tracker_ids is None:
                tracker_ids = np.arange(len(detections.xyxy))
            else:
                tracker_ids = tracker_ids.astype(int)
            
            return detections.xyxy, tracker_ids, detections.confidence, detections.class_id
        
        except Exception as e:
            print(f"추적 실패: {e}")
            return boxes, np.arange(len(boxes)), confidences, class_ids

    def reset(self):
        """추적기 초기화"""
        if HAS_SUPERVISION:
            self.tracker = sv.ByteTrack()
