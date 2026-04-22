# esafe/core/response_system.py
"""
자동 응답 시스템
- 3단계 위험 레벨 (주의/경고/긴급)
- 사운드 알림 및 시각적 표시
- 이벤트 로깅
"""

from collections import deque
from typing import Tuple, Optional
from datetime import datetime
import time


class ResponseSystem:
    """자동 응답 시스템 클래스"""

    # 위험 레벨 정의
    LEVEL_NORMAL = "정상"
    LEVEL_CAUTION = "주의"
    LEVEL_WARNING = "경고"
    LEVEL_EMERGENCY = "긴급"
    
    # 위험 유형별 레벨 매핑
    DANGER_TO_LEVEL = {
        "방치": "CAUTION",
        "낙하": "EMERGENCY",
        "단독탑승": "WARNING",
        "캐리어탑승": "WARNING",
        "역방향": "WARNING",
        "과밀": "CAUTION",
        "쓰러짐": "EMERGENCY",
        "의식불명": "EMERGENCY",
    }

    def __init__(self, cooldown_seconds: int = 10):
        """
        초기화
        
        Args:
            cooldown_seconds: 같은 유형 이벤트 쿨다운 (default: 10초)
        """
        self.cooldown_seconds = cooldown_seconds
        self.last_event_time = {}  # {event_type: timestamp}
        self.event_log = deque(maxlen=1000)
        self.current_level = self.LEVEL_NORMAL
        self.level_start_time = time.time()

    def should_trigger(self, danger_type: str) -> bool:
        """
        이벤트 트리거 여부 판정 (쿨다운 확인)
        
        Args:
            danger_type: 위험 유형 문자열
            
        Returns:
            True if should trigger
        """
        current_time = time.time()
        
        if danger_type not in self.last_event_time:
            self.last_event_time[danger_type] = current_time
            return True
        
        elapsed = current_time - self.last_event_time[danger_type]
        
        if elapsed >= self.cooldown_seconds:
            self.last_event_time[danger_type] = current_time
            return True
        
        return False

    def process_danger(self, danger_type: str, confidence: float,
                       location: Optional[Tuple[float, float]] = None) -> str:
        """
        위험 정보 처리 및 응답 레벨 결정
        
        Args:
            danger_type: 위험 유형 ("방치", "낙하", "역방향", 등)
            confidence: 신뢰도 (0~1)
            location: 위치 (x, y), 옵션
            
        Returns:
            응답 레벨 ("정상", "주의", "경고", "긴급")
        """
        if not self.should_trigger(danger_type):
            return self.current_level
        
        # 위험 레벨 결정
        level_key = self.DANGER_TO_LEVEL.get(danger_type, "WARNING")
        
        if level_key == "CAUTION":
            self.current_level = self.LEVEL_CAUTION
        elif level_key == "WARNING":
            self.current_level = self.LEVEL_WARNING
        elif level_key == "EMERGENCY":
            self.current_level = self.LEVEL_EMERGENCY
        
        self.level_start_time = time.time()
        
        # 로그 기록
        self.log_event(danger_type, level_key, confidence, location)
        
        return self.current_level

    def log_event(self, danger_type: str, level_key: str, confidence: float,
                  location: Optional[Tuple[float, float]] = None):
        """
        이벤트 로그 기록
        
        Args:
            danger_type: 위험 유형
            level_key: 레벨 키 ("CAUTION", "WARNING", "EMERGENCY")
            confidence: 신뢰도
            location: 위치 좌표
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'danger_type': danger_type,
            'level': level_key,
            'confidence': confidence,
            'location': location
        }
        self.event_log.append(event)

    def get_event_log(self):
        """이벤트 로그 반환"""
        return list(self.event_log)

    def reset_level(self, timeout_seconds: float = 5.0):
        """
        레벨 초기화 (타임아웃)
        
        Args:
            timeout_seconds: 레벨 유지 시간 (default: 5초)
        """
        if time.time() - self.level_start_time > timeout_seconds:
            self.current_level = self.LEVEL_NORMAL

    def get_current_level(self) -> str:
        """현재 응답 레벨 반환"""
        self.reset_level()
        return self.current_level

    def clear_logs(self):
        """로그 초기화"""
        self.event_log.clear()
