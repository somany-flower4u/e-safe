# esafe/utils/visualization.py
"""
프레임 시각화 (박스, 라벨, ROI 등 그리기)
"""

from typing import Tuple, Optional, List
import numpy as np
import cv2


class Visualizer:
    """프레임 시각화 클래스"""

    # 색상 정의 (BGR)
    COLOR_PERSON = (255, 0, 0)  # 파랑: 사람
    COLOR_CARRIER = (0, 0, 255)  # 빨강: 캐리어
    COLOR_CHILD = (255, 0, 255)  # 보라: 어린이
    COLOR_ROI = (0, 255, 0)  # 초록: ROI
    COLOR_DANGER = (0, 0, 255)  # 빨강: 위험

    def __init__(self, thickness: int = 2, font_scale: float = 0.6):
        """
        초기화
        
        Args:
            thickness: 박스 두께 (default: 2)
            font_scale: 폰트 크기 (default: 0.6)
        """
        self.thickness = thickness
        self.font_scale = font_scale
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_boxes(self, frame: np.ndarray, boxes: np.ndarray,
                   color: Tuple[int, int, int] = (255, 0, 0),
                   label: str = "", ids: Optional[np.ndarray] = None) -> np.ndarray:
        """
        박스 그리기
        
        Args:
            frame: 입력 프레임
            boxes: 박스 좌표 [[x1, y1, x2, y2], ...]
            color: 색상 (BGR)
            label: 라벨 문자열
            ids: 추적 ID (옵션)
            
        Returns:
            수정된 프레임
        """
        frame = frame.copy()
        
        for idx, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)
            
            # 라벨 그리기
            if label or ids is not None:
                text = label
                if ids is not None:
                    text = f"ID:{ids[idx]}"
                    if label:
                        text = f"{label} {text}"
                
                cv2.putText(frame, text, (x1, y1 - 5),
                           self.font, self.font_scale, color, 1)
        
        return frame

    def draw_roi(self, frame: np.ndarray, roi: Tuple[float, float, float, float],
                 color: Tuple[int, int, int] = (0, 255, 0),
                 dash_length: int = 20) -> np.ndarray:
        """
        ROI 영역 그리기 (점선)
        
        Args:
            frame: 입력 프레임
            roi: ROI 영역 (x_min, y_min, x_max, y_max) - 정규화 좌표 (0~1)
            color: 색상 (BGR)
            dash_length: 점선 길이
            
        Returns:
            수정된 프레임
        """
        frame = frame.copy()
        h, w = frame.shape[:2]
        
        x_min = int(roi[0] * w)
        y_min = int(roi[1] * h)
        x_max = int(roi[2] * w)
        y_max = int(roi[3] * h)
        
        # 상단 좌측
        cv2.line(frame, (x_min, y_min), (x_min + dash_length, y_min), color, 2)
        cv2.line(frame, (x_min, y_min), (x_min, y_min + dash_length), color, 2)
        
        # 상단 우측
        cv2.line(frame, (x_max, y_min), (x_max - dash_length, y_min), color, 2)
        cv2.line(frame, (x_max, y_min), (x_max, y_min + dash_length), color, 2)
        
        # 하단 좌측
        cv2.line(frame, (x_min, y_max), (x_min + dash_length, y_max), color, 2)
        cv2.line(frame, (x_min, y_max), (x_min, y_max - dash_length), color, 2)
        
        # 하단 우측
        cv2.line(frame, (x_max, y_max), (x_max - dash_length, y_max), color, 2)
        cv2.line(frame, (x_max, y_max), (x_max, y_max - dash_length), color, 2)
        
        return frame

    def draw_text(self, frame: np.ndarray, text: str,
                  position: Tuple[int, int],
                  color: Tuple[int, int, int] = (255, 255, 255),
                  bg_color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        텍스트 그리기 (배경 옵션)
        
        Args:
            frame: 입력 프레임
            text: 텍스트 내용
            position: 위치 (x, y)
            color: 텍스트 색상 (BGR)
            bg_color: 배경 색상 (옵션)
            
        Returns:
            수정된 프레임
        """
        frame = frame.copy()
        
        if bg_color is not None:
            # 배경 그리기
            text_size = cv2.getTextSize(text, self.font, self.font_scale, 1)[0]
            x, y = position
            cv2.rectangle(frame, (x - 2, y - text_size[1] - 2),
                         (x + text_size[0] + 2, y + 2), bg_color, -1)
        
        cv2.putText(frame, text, position, self.font, self.font_scale, color, 1)
        
        return frame

    def draw_danger_overlay(self, frame: np.ndarray, danger_label: str,
                           confidence: float) -> np.ndarray:
        """
        위험 감지 오버레이 그리기
        
        Args:
            frame: 입력 프레임
            danger_label: 위험 유형 라벨
            confidence: 신뢰도
            
        Returns:
            수정된 프레임
        """
        frame = frame.copy()
        h, w = frame.shape[:2]
        
        text = f"⚠ {danger_label} ({confidence:.0%})"
        
        # 상단 중앙에 표시
        text_size = cv2.getTextSize(text, self.font, self.font_scale + 0.2, 2)[0]
        x = (w - text_size[0]) // 2
        y = 30
        
        # 빨간 배경
        cv2.rectangle(frame, (x - 10, y - text_size[1] - 5),
                     (x + text_size[0] + 10, y + 5), (0, 0, 255), -1)
        
        # 텍스트
        cv2.putText(frame, text, (x, y), self.font,
                   self.font_scale + 0.2, (0, 255, 255), 2)
        
        return frame

    def draw_count_text(self, frame: np.ndarray, count: int,
                       position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """
        사람 수 표시
        
        Args:
            frame: 입력 프레임
            count: 사람 수
            position: 위치 (x, y)
            
        Returns:
            수정된 프레임
        """
        frame = frame.copy()
        text = f"인원: {count}"
        
        text_size = cv2.getTextSize(text, self.font, self.font_scale, 1)[0]
        x, y = position
        
        # 반투명 배경
        cv2.rectangle(frame, (x - 5, y - text_size[1] - 5),
                     (x + text_size[0] + 5, y + 5), (200, 200, 200), -1)
        
        cv2.putText(frame, text, (x, y), self.font, self.font_scale, (0, 0, 0), 1)
        
        return frame
