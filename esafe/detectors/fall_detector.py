# esafe/detectors/fall_detector.py
"""
사람 낙상 및 발작 감지기
- 방식 1 (우선순위): MediaPipe 자세 추정 (정확도 높음)
  * 낙상: 어깨/엉덩이가 무릎보다 아래로 내려감 (Y 좌표)
  * 의식불명: 코가 프레임 높이 75% 아래에 있고 2초 이상 정지
  
- 방식 2 (대체): 높이 변화 기반 감지 (MediaPipe 없을 때)
  * 낙상: 사람 박스 높이가 50% 이하로 급격히 감소
"""

from collections import deque
from typing import Tuple, Optional
import numpy as np

# MediaPipe는 선택적 - 사용 가능하면 사용, 아니면 스킵
HAS_MEDIAPIPE = False
mp_pose = None

try:
    import mediapipe as mp
    # 새 버전에서 solutions API 확인
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
        mp_pose = mp.solutions.pose
        HAS_MEDIAPIPE = True
except (ImportError, AttributeError) as e:
    pass  # MediaPipe 사용 불가 - 앱은 계속 실행됨 (높이 기반 감지 사용)


class FallDetector:
    """낙상 및 의식불명 감지 클래스 (MediaPipe 선택적 + 높이 기반 대체)"""

    def __init__(self, visibility_threshold: float = 0.6, unconscious_frames: int = 2,
                 height_drop_threshold: float = 0.5, height_history_size: int = 10):
        """
        초기화
        
        Args:
            visibility_threshold: 랜드마크 신뢰도 임계값 (default: 0.6)
            unconscious_frames: 의식불명 판정 필요 프레임 수 (default: 2 = ~2초 at 30fps)
            height_drop_threshold: 높이 감소 임계값 (default: 0.5 = 50%)
            height_history_size: 높이 이력 크기 (default: 10 프레임)
        """
        self.visibility_threshold = visibility_threshold
        self.unconscious_frames = unconscious_frames * 30  # 초를 프레임으로 변환
        self.height_drop_threshold = height_drop_threshold
        self.height_history_size = height_history_size
        
        self.pose = None
        self.available = False
        
        # MediaPipe Pose 초기화 (레거시 solutions API)
        if HAS_MEDIAPIPE and mp_pose is not None:
            try:
                self.pose = mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.available = True
                print("✓ FallDetector: MediaPipe Pose 초기화됨")
            except Exception as e:
                print(f"⚠ FallDetector: MediaPipe Pose 초기화 실패 ({e}) - 높이 기반 감지 사용")
                self.available = False
                self.pose = None
        else:
            if not HAS_MEDIAPIPE:
                print("⚠ FallDetector: MediaPipe 라이브러리 없음 - 높이 기반 감지 사용")
        
        # 사람별 추적: {track_id: deque of landmarks_history}
        self.person_history = {}
        self.unconscious_count = {}
        
        # 높이 기반 감지: {track_id: deque of heights}
        self.height_history = {}
        self.prev_heights = {}

    def update(self, frame: np.ndarray, person_boxes: np.ndarray,
               person_ids: Optional[np.ndarray] = None) -> Tuple[bool, str, float]:
        """
        낙상 상태 업데이트
        
        Args:
            frame: 비디오 프레임 (RGB, HxWx3)
            person_boxes: 사람 박스 [[x1, y1, x2, y2], ...] (Nx4)
            person_ids: 각 사람의 추적 ID (N,), 옵션
            
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
        
        # ===== 방식 1: MediaPipe 포즈 기반 (가용시) =====
        if self.available and self.pose is not None:
            is_danger, danger_label, confidence = self._detect_with_pose(
                frame, person_boxes, person_ids
            )
            if is_danger:
                return is_danger, danger_label, confidence
        
        # ===== 방식 2: 높이 변화 기반 (항상 적용 - 대체 및 보조) =====
        is_danger, danger_label, confidence = self._detect_with_height(
            person_boxes, person_ids
        )
        
        return is_danger, danger_label, confidence
    
    def _detect_with_pose(self, frame: np.ndarray, person_boxes: np.ndarray,
                          person_ids: np.ndarray) -> Tuple[bool, str, float]:
        """MediaPipe 포즈 기반 낙상 감지"""
        h, w = frame.shape[:2]
        
        current_ids = set(person_ids)
        
        # 사라진 사람 정리
        for pid in list(self.person_history.keys()):
            if pid not in current_ids:
                del self.person_history[pid]
                if pid in self.unconscious_count:
                    del self.unconscious_count[pid]
        
        # 각 사람 처리
        for idx, person_id in enumerate(person_ids):
            x1, y1, x2, y2 = person_boxes[idx]
            
            # 박스 내부 영역 추출
            person_roi = frame[int(y1):int(y2), int(x1):int(x2)]
            
            if person_roi.shape[0] < 10 or person_roi.shape[1] < 10:
                continue
            
            # MediaPipe Pose 추론 (RGB 이미지 필요)
            try:
                results = self.pose.process(person_roi)
            except Exception as e:
                continue
            
            if not hasattr(results, 'pose_landmarks') or results.pose_landmarks is None:
                continue
            
            # 초기화
            if person_id not in self.person_history:
                self.person_history[person_id] = deque(maxlen=5)
                self.unconscious_count[person_id] = 0
            
            history = self.person_history[person_id]
            
            # 랜드마크 추출 (정규화 좌표)
            landmarks = results.pose_landmarks.landmark
            
            # 주요 랜드마크: NOSE=0, LEFT_SHOULDER=11, RIGHT_SHOULDER=12,
            # LEFT_HIP=23, RIGHT_HIP=24, LEFT_KNEE=25, RIGHT_KNEE=26
            nose_lm = landmarks[0]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            left_knee = landmarks[25]
            right_knee = landmarks[26]
            
            # 신뢰도 필터링
            valid = (nose_lm.visibility > self.visibility_threshold and
                     left_shoulder.visibility > self.visibility_threshold and
                     right_shoulder.visibility > self.visibility_threshold and
                     left_knee.visibility > self.visibility_threshold and
                     right_knee.visibility > self.visibility_threshold)
            
            if not valid:
                continue
            
            # 임시 저장
            lm_dict = {
                'nose_y': nose_lm.y,
                'left_shoulder_y': left_shoulder.y,
                'right_shoulder_y': right_shoulder.y,
                'left_hip_y': left_hip.y,
                'right_hip_y': right_hip.y,
                'left_knee_y': left_knee.y,
                'right_knee_y': right_knee.y,
            }
            history.append(lm_dict)
            
            # 낙상: 어깨 또는 엉덩이가 무릎보다 아래
            shoulder_y = min(lm_dict['left_shoulder_y'], lm_dict['right_shoulder_y'])
            hip_y = min(lm_dict['left_hip_y'], lm_dict['right_hip_y'])
            knee_y = min(lm_dict['left_knee_y'], lm_dict['right_knee_y'])
            
            if shoulder_y > knee_y or hip_y > knee_y:
                return True, "쓰러짐", 0.95
            
            # 의식불명: 코가 프레임 75% 아래 + 2초
            if lm_dict['nose_y'] > 0.75:
                self.unconscious_count[person_id] += 1
            else:
                self.unconscious_count[person_id] = 0
            
            if self.unconscious_count[person_id] >= self.unconscious_frames:
                return True, "의식불명", 0.85
        
        return False, "", 0.0
    
    def _detect_with_height(self, person_boxes: np.ndarray,
                            person_ids: np.ndarray) -> Tuple[bool, str, float]:
        """높이 변화 기반 낙상 감지 (간단하고 빠른 대체 방식)"""
        
        current_ids = set(person_ids)
        
        # 사라진 사람 정리
        for pid in list(self.height_history.keys()):
            if pid not in current_ids:
                del self.height_history[pid]
                if pid in self.prev_heights:
                    del self.prev_heights[pid]
        
        # 각 사람 처리
        for idx, person_id in enumerate(person_ids):
            x1, y1, x2, y2 = person_boxes[idx]
            
            # 현재 높이 계산
            current_height = y2 - y1
            
            # 초기화
            if person_id not in self.height_history:
                self.height_history[person_id] = deque(maxlen=self.height_history_size)
                self.prev_heights[person_id] = current_height
            
            hist = self.height_history[person_id]
            hist.append(current_height)
            
            # 이전 높이의 평균 (최근 5프레임)
            if len(hist) >= 5:
                prev_avg_height = np.mean(list(hist)[-5:-1])
                
                # 높이가 50% 이하로 급감 = 낙상 가능성
                height_ratio = current_height / (prev_avg_height + 1e-5)
                
                if height_ratio < self.height_drop_threshold:
                    return True, "쓰러짐", 0.7
        
        return False, "", 0.0

    def reset(self):
        """상태 초기화"""
        self.person_history.clear()
        self.unconscious_count.clear()
        self.height_history.clear()
        self.prev_heights.clear()
