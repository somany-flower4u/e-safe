# esafe/utils/event_logger.py
"""
이벤트 로그 관리 및 저장
"""

from typing import List, Dict
from datetime import datetime
import csv
import os


class EventLogger:
    """이벤트 로거 클래스"""

    def __init__(self, log_file: str = "esafe_events.csv"):
        """
        초기화
        
        Args:
            log_file: 로그 파일 경로 (default: "esafe_events.csv")
        """
        self.log_file = log_file
        self.events = []
        
        # 파일이 없으면 헤더 작성
        if not os.path.exists(log_file):
            self._write_header()

    def _write_header(self):
        """CSV 파일 헤더 작성"""
        try:
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Danger Type', 'Level', 'Confidence', 'Location X', 'Location Y'])
        except Exception as e:
            print(f"헤더 작성 실패: {e}")

    def add_event(self, danger_type: str, level: str, confidence: float,
                  location: tuple = None):
        """
        이벤트 추가
        
        Args:
            danger_type: 위험 유형
            level: 레벨 (CAUTION, WARNING, EMERGENCY)
            confidence: 신뢰도
            location: 위치 (x, y)
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'danger_type': danger_type,
            'level': level,
            'confidence': confidence,
            'location': location
        }
        self.events.append(event)
        
        # 즉시 파일에 저장
        self._save_event(event)

    def _save_event(self, event: Dict):
        """이벤트를 파일에 저장"""
        try:
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                loc_x, loc_y = event['location'] if event['location'] else (None, None)
                writer.writerow([
                    event['timestamp'],
                    event['danger_type'],
                    event['level'],
                    f"{event['confidence']:.3f}",
                    loc_x,
                    loc_y
                ])
        except Exception as e:
            print(f"이벤트 저장 실패: {e}")

    def get_events(self) -> List[Dict]:
        """이벤트 목록 반환"""
        return self.events.copy()

    def get_event_count_by_type(self) -> Dict[str, int]:
        """위험 유형별 이벤트 개수 반환"""
        counts = {}
        for event in self.events:
            danger_type = event['danger_type']
            counts[danger_type] = counts.get(danger_type, 0) + 1
        return counts

    def clear_events(self):
        """메모리 이벤트 초기화"""
        self.events.clear()

    def get_recent_events(self, limit: int = 10) -> List[Dict]:
        """최근 이벤트 반환"""
        return self.events[-limit:] if limit > 0 else self.events.copy()
