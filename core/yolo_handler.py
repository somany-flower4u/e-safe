# esafe/core/yolo_handler.py
"""YOLOv8 객체 감지 래퍼 모듈입니다."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - 환경 의존성 예외 보호
    YOLO = None


class YoloHandler:
    """YOLOv8n 모델 초기화 및 추론 결과 정규화를 담당합니다."""

    PERSON_CLASS_ID = 0
    SUITCASE_CLASS_ID = 28

    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.5) -> None:
        """YOLO 모델을 초기화합니다."""
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = 0.45
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """모델 로딩을 시도하고 실패 시 안전하게 비활성화합니다."""
        if YOLO is None:
            self.model = None
            return

        try:
            self.model = YOLO(self.model_name)
        except Exception:
            self.model = None

    def set_conf_threshold(self, conf_threshold: float) -> None:
        """추론 신뢰도 임계값을 업데이트합니다."""
        self.conf_threshold = float(max(0.0, min(1.0, conf_threshold)))

    def _class_name(self, class_id: int) -> str:
        """클래스 ID를 사람이 읽기 쉬운 이름으로 변환합니다."""
        if class_id == self.PERSON_CLASS_ID:
            return "person"
        if class_id == self.SUITCASE_CLASS_ID:
            return "suitcase"
        return f"class_{class_id}"

    def infer(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """입력 프레임에서 사람/캐리어 객체를 감지해 반환합니다."""
        if frame is None or self.model is None:
            return []

        try:
            # CPU 환경에서도 안정적으로 동작하도록 기본 device를 auto로 둡니다.
            results = self.model.predict(
                source=frame,
                imgsz=640,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )
        except Exception:
            return []

        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        detections: List[Dict[str, Any]] = []
        try:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
        except Exception:
            return []

        for bbox, score, class_id in zip(xyxy, conf, cls):
            cid = int(class_id)
            if cid not in (self.PERSON_CLASS_ID, self.SUITCASE_CLASS_ID):
                continue

            x1, y1, x2, y2 = [float(v) for v in bbox]
            detections.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(score),
                    "class_id": cid,
                    "class_name": self._class_name(cid),
                }
            )

        return detections
