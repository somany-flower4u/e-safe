# esafe/core/yolo_handler.py
"""
YOLO v8 nano 모델 핸들러
- CPU 전용 추론
- 사람, 캐리어 객체 감지
- 신뢰도 필터링
"""

from typing import Tuple, Optional
import numpy as np

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


class YOLOHandler:
    """YOLOv8n 모델 핸들러 클래스"""

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.5):
        """
        초기화
        
        Args:
            model_name: 모델 이름 (default: "yolov8n.pt" - nano 모델)
            confidence: 신뢰도 임계값 (default: 0.5)
        """
        self.confidence = confidence
        self.model = None
        self.device = "cpu"
        
        if HAS_ULTRALYTICS:
            try:
                # CPU 전용 모델 로드
                self.model = YOLO(model_name)
                self.model.to(self.device)
            except Exception as e:
                print(f"모델 로드 실패: {e}")
                self.model = None

    def detect(self, frame: np.ndarray, conf: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        프레임에서 객체 감지
        
        Args:
            frame: 입력 프레임 (HxWx3 BGR)
            conf: 신뢰도 임계값 (옵션, 기본값 사용시 None)
            
        Returns:
            (boxes, confidences, class_ids)
            - boxes: [[x1, y1, x2, y2], ...] (Nx4)
            - confidences: 신뢰도 (N,)
            - class_ids: 클래스 ID (N,) - 0=person, 24=suitcase
        """
        if self.model is None:
            return np.array([]), np.array([]), np.array([])
        
        if conf is None:
            conf = self.confidence
        
        try:
            # 추론 수행
            results = self.model.predict(
                source=frame,
                conf=conf,
                device=self.device,
                verbose=False
            )
            
            result = results[0]
            
            # 결과 추출
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            return boxes, confidences, class_ids
        
        except Exception as e:
            print(f"추론 실패: {e}")
            return np.array([]), np.array([]), np.array([])

    def filter_by_class(self, boxes: np.ndarray, confidences: np.ndarray,
                        class_ids: np.ndarray, target_class_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        특정 클래스만 필터링
        
        Args:
            boxes: 모든 박스 [[x1, y1, x2, y2], ...]
            confidences: 모든 신뢰도
            class_ids: 모든 클래스 ID
            target_class_id: 대상 클래스 ID (0=person, 24=suitcase)
            
        Returns:
            (filtered_boxes, filtered_confidences)
        """
        mask = class_ids == target_class_id
        filtered_boxes = boxes[mask]
        filtered_confidences = confidences[mask]
        
        return filtered_boxes, filtered_confidences

    def set_confidence(self, confidence: float):
        """신뢰도 임계값 설정"""
        self.confidence = np.clip(confidence, 0.0, 1.0)
