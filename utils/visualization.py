# esafe/utils/visualization.py
"""OpenCV 기반 시각화 유틸리티 모듈입니다."""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np


def draw_dashed_rectangle(
    frame: np.ndarray,
    roi: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash: int = 10,
    gap: int = 6,
) -> None:
    """점선 형태의 사각형 ROI를 그립니다."""
    x1, y1, x2, y2 = roi

    for x in range(x1, x2, dash + gap):
        cv2.line(frame, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(frame, (x, y2), (min(x + dash, x2), y2), color, thickness)

    for y in range(y1, y2, dash + gap):
        cv2.line(frame, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(frame, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def draw_objects(
    frame: np.ndarray,
    tracked_objects: List[Dict[str, object]],
    child_ids: List[int],
) -> None:
    """사람/캐리어 객체 박스와 ID 라벨을 색상 규칙에 맞춰 그립니다."""
    for obj in tracked_objects:
        bbox = obj.get("bbox")
        if bbox is None:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        class_name = str(obj.get("class_name", "unknown"))
        track_id = int(obj.get("track_id", -1))

        if class_name == "suitcase":
            color = (0, 0, 255)  # 빨강
            prefix = "S"
        elif class_name == "person" and track_id in child_ids:
            color = (255, 0, 255)  # 보라
            prefix = "C"
        else:
            color = (255, 0, 0)  # 파랑
            prefix = "P"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{prefix}-{track_id}"
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_risk_texts(frame: np.ndarray, risk_texts: List[str]) -> None:
    """활성 위험 유형 텍스트를 프레임 상단에 표시합니다."""
    if not risk_texts:
        cv2.putText(
            frame,
            "위험 상태: 정상",
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (30, 200, 30),
            2,
            cv2.LINE_AA,
        )
        return

    joined = " | ".join(risk_texts)
    cv2.putText(
        frame,
        f"위험 상태: {joined}",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (30, 30, 235),
        2,
        cv2.LINE_AA,
    )


def draw_emergency_border(frame: np.ndarray, timestamp: float, period_sec: float = 0.5) -> None:
    """긴급 단계에서 빨간 점멸 테두리를 그립니다."""
    cycle = int(timestamp / period_sec) % 2
    if cycle == 0:
        return

    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)


def render_overlay(
    frame: np.ndarray,
    tracked_objects: List[Dict[str, object]],
    child_ids: List[int],
    roi: Tuple[int, int, int, int],
    entrance_roi: Tuple[int, int, int, int],
    risk_texts: List[str],
    response_level: int,
    timestamp: float,
) -> np.ndarray:
    """프레임 오버레이(객체/ROI/위험 텍스트/긴급 점멸)를 적용한 이미지를 반환합니다."""
    output = frame.copy()

    # 메인 ROI는 녹색 점선, 입구 ROI는 노란색 점선으로 구분합니다.
    draw_dashed_rectangle(output, roi, color=(0, 220, 0), thickness=2)
    draw_dashed_rectangle(output, entrance_roi, color=(0, 220, 220), thickness=2)

    draw_objects(output, tracked_objects, child_ids)
    draw_risk_texts(output, risk_texts)

    if response_level == 3:
        draw_emergency_border(output, timestamp=timestamp, period_sec=0.5)

    return output
