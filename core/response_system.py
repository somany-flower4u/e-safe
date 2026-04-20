# esafe/core/response_system.py
"""위험 감지 결과에 대한 3단계 자동 대응 시스템 모듈입니다."""

from __future__ import annotations

import time
import wave
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import pygame
except Exception:  # pragma: no cover - 환경 의존성 예외 보호
    pygame = None


class AutoResponseSystem:
    """위험 유형을 대응 단계로 매핑하고 오디오 경보를 실행합니다."""

    LEVEL_META = {
        0: {
            "stage": "NORMAL",
            "label": "정상",
            "color": "#1E8E3E",
            "action": "정상 운영 상태",
        },
        1: {
            "stage": "CAUTION",
            "label": "주의",
            "color": "#F2C94C",
            "action": "노란 경고 배너 표시 및 경고음 1회 재생",
        },
        2: {
            "stage": "WARNING",
            "label": "경고",
            "color": "#F2994A",
            "action": "주황 경고 배너 표시 및 경고음 2회 재생",
        },
        3: {
            "stage": "EMERGENCY",
            "label": "긴급",
            "color": "#D62828",
            "action": "적색 점멸 및 경보음 3회 재생, 자동 정지 신호",
        },
    }

    HAZARD_LEVEL = {
        "carrier_abandon": 1,
        "crowding": 1,
        "wrong_direction": 2,
        "child_alone": 2,
        "child_on_carrier": 2,
        "carrier_drop": 3,
        "fall_down": 3,
        "unconscious": 3,
    }

    def __init__(self, cooldown_seconds: float = 10.0) -> None:
        """응답 시스템 상태 및 오디오 장치를 초기화합니다."""
        self.cooldown_seconds = cooldown_seconds
        self.last_trigger_time: Dict[str, float] = {}
        self.audio_enabled = False
        self.warning_beep_path = self._ensure_reference_beep_file()
        self._init_audio()

    def _ensure_reference_beep_file(self) -> str:
        """assets 폴더에 기본 경고음 파일이 없으면 자동 생성합니다."""
        assets_dir = Path(__file__).resolve().parents[1] / "assets"
        beep_path = assets_dir / "warning_beep.wav"

        if beep_path.exists():
            return str(beep_path)

        try:
            assets_dir.mkdir(parents=True, exist_ok=True)
            waveform = self._build_sine_wave(800.0, 0.3, volume=0.35)
            mono_wave = waveform[:, 0].astype(np.int16)

            with wave.open(str(beep_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(44100)
                wav_file.writeframes(mono_wave.tobytes())
        except Exception:
            return ""

        return str(beep_path)

    def _init_audio(self) -> None:
        """pygame 오디오 믹서를 초기화합니다."""
        if pygame is None:
            self.audio_enabled = False
            return

        try:
            if not pygame.get_init():
                pygame.init()
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2)
            self.audio_enabled = True
        except Exception:
            self.audio_enabled = False

    def _build_sine_wave(
        self,
        frequency_hz: float,
        duration_sec: float,
        volume: float = 0.35,
        sample_rate: int = 44100,
    ) -> np.ndarray:
        """numpy 기반 사인파 비프음을 생성합니다."""
        t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
        mono_wave = np.sin(2.0 * np.pi * frequency_hz * t) * volume
        stereo_wave = np.column_stack((mono_wave, mono_wave))
        return (stereo_wave * 32767).astype(np.int16)

    def _play_beep(self, frequency_hz: float, duration_sec: float) -> None:
        """단일 비프음을 재생합니다."""
        if not self.audio_enabled or pygame is None:
            return

        try:
            wave = self._build_sine_wave(frequency_hz, duration_sec)
            sound = pygame.sndarray.make_sound(wave)
            channel = sound.play()
            if channel is None:
                return
            while channel.get_busy():
                pygame.time.wait(5)
        except Exception:
            self.audio_enabled = False

    def _play_pattern(self, level: int) -> None:
        """대응 단계별 경고음 패턴을 재생합니다."""
        if level == 1:
            pattern = [(800.0, 0.3)]
        elif level == 2:
            pattern = [(800.0, 0.3), (800.0, 0.3)]
        elif level == 3:
            pattern = [(1200.0, 0.28), (1200.0, 0.28), (1200.0, 0.28)]
        else:
            pattern = []

        for idx, (freq, duration) in enumerate(pattern):
            self._play_beep(freq, duration)
            if idx < len(pattern) - 1 and pygame is not None:
                pygame.time.wait(70)

    def _is_cooldown(self, hazard_type: str, now_ts: float) -> bool:
        """동일 이벤트의 쿨다운 여부를 확인합니다."""
        last_time = self.last_trigger_time.get(hazard_type)
        if last_time is None:
            return False
        return (now_ts - last_time) < self.cooldown_seconds

    def level_of_hazard(self, hazard_type: str | None) -> int:
        """위험 유형의 대응 단계를 반환합니다."""
        if hazard_type is None:
            return 0
        return int(self.HAZARD_LEVEL.get(hazard_type, 0))

    def resolve_response(self, active_events: List[Dict[str, Any]], now_ts: float | None = None) -> Dict[str, Any]:
        """활성 위험 목록으로부터 현재 대응 상태를 계산하고 필요 시 경고음을 재생합니다."""
        if now_ts is None:
            now_ts = time.time()

        highest_active_level = 0
        for event in active_events:
            highest_active_level = max(highest_active_level, self.level_of_hazard(event.get("hazard_type")))

        triggered_events: List[Dict[str, Any]] = []
        for event in active_events:
            hazard_type = event.get("hazard_type")
            if not hazard_type:
                continue
            if self._is_cooldown(hazard_type, now_ts):
                continue

            self.last_trigger_time[hazard_type] = now_ts
            event_with_level = dict(event)
            event_with_level["level"] = self.level_of_hazard(hazard_type)
            triggered_events.append(event_with_level)

        highest_triggered_level = max((event["level"] for event in triggered_events), default=0)
        if highest_triggered_level > 0:
            self._play_pattern(highest_triggered_level)

        meta = self.LEVEL_META.get(highest_active_level, self.LEVEL_META[0])
        return {
            "level": highest_active_level,
            "stage": meta["stage"],
            "label": meta["label"],
            "color": meta["color"],
            "action": meta["action"],
            "triggered_events": triggered_events,
            "emergency_popup": highest_active_level == 3,
            "blink": highest_active_level == 3,
        }
