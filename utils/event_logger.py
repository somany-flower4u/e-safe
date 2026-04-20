# esafe/utils/event_logger.py
"""이벤트 로그 저장/조회 유틸리티 모듈입니다."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


class EventLogger:
    """위험 이벤트를 메모리에 누적하고 CSV로 내보내는 로거 클래스입니다."""

    def __init__(self, output_csv_path: str = "events_log.csv") -> None:
        """로그 저장 경로와 내부 버퍼를 초기화합니다."""
        self.output_csv_path = Path(output_csv_path)
        self.events: List[Dict[str, object]] = []

    def add_event(
        self,
        risk_type: str,
        level: str,
        response_action: str,
        location: Optional[Tuple[int, int]] = None,
        confidence: float = 0.0,
        timestamp: Optional[str] = None,
    ) -> Dict[str, object]:
        """단일 이벤트를 로그에 추가합니다."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        x = location[0] if location else None
        y = location[1] if location else None
        row: Dict[str, object] = {
            "타임스탬프": timestamp,
            "위험 유형": risk_type,
            "단계": level,
            "대응 내용": response_action,
            "위치 X": x,
            "위치 Y": y,
            "신뢰도": round(float(confidence), 3),
        }
        self.events.append(row)
        return row

    def to_dataframe(self, max_rows: int = 300) -> pd.DataFrame:
        """현재 로그를 DataFrame으로 변환합니다."""
        if not self.events:
            return pd.DataFrame(
                columns=[
                    "타임스탬프",
                    "위험 유형",
                    "단계",
                    "대응 내용",
                    "위치 X",
                    "위치 Y",
                    "신뢰도",
                ]
            )

        df = pd.DataFrame(self.events)
        if len(df) > max_rows:
            df = df.tail(max_rows).reset_index(drop=True)
        return df

    def event_frequency_dataframe(self) -> pd.DataFrame:
        """위험 유형별 이벤트 발생 횟수 DataFrame을 반환합니다."""
        if not self.events:
            return pd.DataFrame(columns=["위험 유형", "발생 횟수"])

        df = pd.DataFrame(self.events)
        freq = (
            df.groupby("위험 유형", dropna=False)
            .size()
            .reset_index(name="발생 횟수")
            .sort_values("발생 횟수", ascending=False)
            .reset_index(drop=True)
        )
        return freq

    def save_csv(self) -> Path:
        """이벤트 로그를 CSV 파일로 저장합니다."""
        df = self.to_dataframe(max_rows=1_000_000)
        self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_csv_path, index=False, encoding="utf-8-sig")
        return self.output_csv_path

    def clear(self) -> None:
        """메모리상의 이벤트 로그를 초기화합니다."""
        self.events.clear()
