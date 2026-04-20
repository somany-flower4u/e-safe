# esafe/utils/roi_manager.py
"""ROI(관심영역) 계산 및 판별 유틸리티 모듈입니다."""

from __future__ import annotations

from typing import Tuple


def compute_default_roi(frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    """프레임 중앙 70% 영역을 기본 ROI로 계산합니다."""
    frame_h, frame_w = frame_shape[:2]
    x1 = int(frame_w * 0.15)
    x2 = int(frame_w * 0.85)
    y1 = int(frame_h * 0.10)
    y2 = int(frame_h * 0.90)
    return x1, y1, x2, y2


def compute_entrance_roi(
    main_roi: Tuple[int, int, int, int],
    escalator_direction: str,
) -> Tuple[int, int, int, int]:
    """에스컬레이터 진행 방향에 맞는 입구 ROI를 계산합니다."""
    x1, y1, x2, y2 = main_roi
    height = max(1, y2 - y1)
    segment = int(height * 0.25)

    if escalator_direction == "down":
        # 하행(위->아래)에서는 상단 부분이 입구입니다.
        return (x1, y1, x2, y1 + segment)

    # 상행(아래->위)에서는 하단 부분이 입구입니다.
    return (x1, y2 - segment, x2, y2)


def is_point_in_roi(point: Tuple[float, float], roi: Tuple[int, int, int, int]) -> bool:
    """점이 ROI 내부인지 여부를 반환합니다."""
    px, py = point
    x1, y1, x2, y2 = roi
    return x1 <= px <= x2 and y1 <= py <= y2


def clamp_roi(roi: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    """ROI 좌표를 프레임 범위 안으로 보정합니다."""
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = roi
    x1 = max(0, min(frame_w - 1, x1))
    x2 = max(0, min(frame_w - 1, x2))
    y1 = max(0, min(frame_h - 1, y1))
    y2 = max(0, min(frame_h - 1, y2))
    return x1, y1, x2, y2
