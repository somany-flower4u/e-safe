# esafe/core/pose_handler.py
"""MediaPipe Pose 추론 래퍼 모듈입니다."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:  # pragma: no cover - 환경 의존성 예외 보호
    mp = None


class PoseHandler:
    """사람 박스 크롭 이미지에 대해 Pose 키포인트를 추론합니다."""

    # MediaPipe PoseLandmark 인덱스 중 17개 핵심 관절만 사용합니다.
    SELECTED_KEYPOINTS = {
        "nose": 0,
        "left_eye": 2,
        "right_eye": 5,
        "left_ear": 7,
        "right_ear": 8,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
    }

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """MediaPipe Pose 모델을 초기화합니다."""
        self.pose = None
        if mp is None:
            return

        try:
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        except Exception:
            self.pose = None

    def _sanitize_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[int, int, int, int]:
        """프레임 경계 안으로 바운딩 박스를 보정합니다."""
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = int(max(0, min(width - 1, x1)))
        y1 = int(max(0, min(height - 1, y1)))
        x2 = int(max(0, min(width, x2)))
        y2 = int(max(0, min(height, y2)))
        return x1, y1, x2, y2

    def infer_person(
        self,
        frame: np.ndarray,
        person_bbox: Tuple[float, float, float, float],
        track_id: int,
    ) -> Dict[str, Any] | None:
        """단일 사람 박스에서 17개 핵심 키포인트를 추론합니다."""
        if self.pose is None:
            return None

        x1, y1, x2, y2 = self._sanitize_bbox(person_bbox, frame.shape)
        if x2 - x1 < 20 or y2 - y1 < 40:
            return None

        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return None

        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        landmarks = getattr(result, "pose_landmarks", None)
        if landmarks is None:
            return None

        crop_h, crop_w = cropped.shape[:2]
        keypoints: Dict[str, Dict[str, float]] = {}
        for name, index in self.SELECTED_KEYPOINTS.items():
            if index >= len(landmarks.landmark):
                continue
            point = landmarks.landmark[index]
            abs_x = float(x1 + point.x * crop_w)
            abs_y = float(y1 + point.y * crop_h)
            keypoints[name] = {
                "x": abs_x,
                "y": abs_y,
                "visibility": float(point.visibility),
            }

        return {"track_id": int(track_id), "keypoints": keypoints}

    def extract_for_people(
        self,
        frame: np.ndarray,
        people: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """사람 목록에 대해 Pose 추론을 수행하고 결과 리스트를 반환합니다."""
        pose_results: List[Dict[str, Any]] = []
        for person in people:
            track_id = int(person.get("track_id", -1))
            bbox = person.get("bbox")
            if track_id < 0 or bbox is None:
                continue
            pose = self.infer_person(frame, tuple(float(v) for v in bbox), track_id)
            if pose is not None:
                pose_results.append(pose)
        return pose_results
