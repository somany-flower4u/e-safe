# esafe/app.py
"""
E-SAFE (Escalator Safety AI For Everyone)
인천국제공항 에스컬레이터 안전 실시간 모니터링 시스템
"""

import sys
import os
from pathlib import Path

# 현재 스크립트 디렉토리를 sys.path에 추가 (상대 임포트 호환성)
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from collections import deque
import time

# E-SAFE 모듈 임포트
from core.yolo_handler import YOLOHandler
from core.tracker import ObjectTracker
from core.response_system import ResponseSystem
from detectors.carrier_detector import CarrierHazardDetector
from detectors.fall_detector import FallDetector
from detectors.direction_detector import WrongDirectionDetector
from detectors.child_detector import ChildRiskDetector
from detectors.crowd_detector import CrowdingDetector
from utils.visualization import Visualizer
from utils.roi_manager import ROIManager
from utils.event_logger import EventLogger

# 플롯 라이브러리
import plotly.graph_objects as go
import plotly.express as px

# 사운드 생성 (pygame)
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


# ===== 음성 알림 생성 (Pygame) =====
def generate_beep(frequency: float, duration: float, volume: float = 0.3) -> 'pygame.mixer.Sound':
    """
    Pygame을 사용한 사인파 비프 생성
    
    Args:
        frequency: 주파수 (Hz)
        duration: 지속 시간 (초)
        volume: 음량 (0~1)
        
    Returns:
        pygame.mixer.Sound 객체
    """
    if not HAS_PYGAME:
        return None
    
    try:
        sample_rate = 22050
        num_samples = int(sample_rate * duration)
        
        # 사인파 생성
        t = np.linspace(0, duration, num_samples, False)
        wave = np.sin(2 * np.pi * frequency * t)
        
        # 음량 조절 및 16-bit PCM 변환
        wave = (wave * volume * 32767).astype(np.int16)
        
        # 스테레오로 변환
        stereo_wave = np.repeat(wave[np.newaxis, :], 2, axis=0)
        
        # Pygame Sound 객체 생성
        sound = pygame.mixer.Sound(buffer=stereo_wave.tobytes())
        return sound
    except Exception as e:
        print(f"비프 생성 실패: {e}")
        return None


def play_alert(frequency: float, duration: float, count: int = 1):
    """
    음성 알림 재생
    
    Args:
        frequency: 주파수
        duration: 지속 시간
        count: 반복 횟수
    """
    if not HAS_PYGAME:
        return
    
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        
        sound = generate_beep(frequency, duration)
        if sound:
            for _ in range(count):
                sound.play()
                time.sleep(duration + 0.1)
    except Exception as e:
        print(f"알림 재생 실패: {e}")


# ===== 합성 프레임 생성 (웹캠 사용 불가시) =====
def generate_demo_frame(frame_num: int, frame_h: int = 480, frame_w: int = 640) -> np.ndarray:
    """
    데모 모드: 움직이는 사각형 시뮬레이션
    
    Args:
        frame_num: 프레임 번호
        frame_h: 프레임 높이
        frame_w: 프레임 너비
        
    Returns:
        합성 프레임 (BGR)
    """
    frame = np.ones((frame_h, frame_w, 3), dtype=np.uint8) * 200
    
    # 움직이는 인물/캐리어 박스 그리기
    offset = (frame_num * 5) % frame_w
    
    # 인물 박스
    x1 = offset
    y1 = 100
    cv2.rectangle(frame, (x1, y1), (x1 + 50, y1 + 100), (255, 0, 0), 2)
    cv2.putText(frame, "P", (x1 + 15, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # 캐리어 박스
    x2 = (offset + 100) % frame_w
    y2 = 200
    cv2.rectangle(frame, (x2, y2), (x2 + 40, y2 + 80), (0, 0, 255), 2)
    cv2.putText(frame, "C", (x2 + 10, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # 타이틀
    cv2.putText(frame, "DEMO MODE", (frame_w // 2 - 100, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    return frame


# ===== Streamlit 페이지 설정 =====
st.set_page_config(
    page_title="E-SAFE 관제 대시보드",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== 세션 상태 초기화 =====
if 'yolo_handler' not in st.session_state:
    st.session_state.yolo_handler = YOLOHandler()

if 'tracker' not in st.session_state:
    st.session_state.tracker = ObjectTracker()

if 'response_system' not in st.session_state:
    st.session_state.response_system = ResponseSystem()

if 'carrier_detector' not in st.session_state:
    st.session_state.carrier_detector = CarrierHazardDetector()

if 'fall_detector' not in st.session_state:
    st.session_state.fall_detector = FallDetector()

if 'direction_detector' not in st.session_state:
    st.session_state.direction_detector = WrongDirectionDetector()

if 'child_detector' not in st.session_state:
    st.session_state.child_detector = ChildRiskDetector()

if 'crowd_detector' not in st.session_state:
    st.session_state.crowd_detector = CrowdingDetector()

if 'visualizer' not in st.session_state:
    st.session_state.visualizer = Visualizer()

if 'roi_manager' not in st.session_state:
    st.session_state.roi_manager = ROIManager()

if 'event_logger' not in st.session_state:
    st.session_state.event_logger = EventLogger()

if 'is_running' not in st.session_state:
    st.session_state.is_running = False

if 'danger_history' not in st.session_state:
    st.session_state.danger_history = deque(maxlen=4)

if 'frame_num' not in st.session_state:
    st.session_state.frame_num = 0

if 'status_placeholder' not in st.session_state:
    st.session_state.status_placeholder = None

if 'video_frame_placeholder' not in st.session_state:
    st.session_state.video_frame_placeholder = None


# ===== 사이드바 설정 =====
st.sidebar.title("🛡️ E-SAFE 관제 대시보드 — 인천공항")

# 입력 모드 선택
input_mode = st.sidebar.radio(
    "입력 모드 선택",
    ["웹캠 (실시간)", "동영상 파일 업로드"]
)

# 동영상 파일 업로더 (파일 모드일 때만 표시)
uploaded_video = None
if input_mode == "동영상 파일 업로드":
    uploaded_video = st.sidebar.file_uploader(
        "MP4, AVI, MOV 파일 선택",
        type=["mp4", "avi", "mov"]
    )

# 에스컬레이터 방향
escalator_direction = st.sidebar.selectbox(
    "에스컬레이터 방향",
    ["하행 (위→아래)", "상행 (아래→위)"],
    help="역방향 감지의 기준 방향"
)
direction_value = "down" if escalator_direction == "하행 (위→아래)" else "up"

# 혼잡도 임계값
crowding_threshold = st.sidebar.slider(
    "혼잡 임계값 (명)",
    min_value=2,
    max_value=10,
    value=5,
    help="ROI 내 사람 수가 이 값을 초과하면 경고"
)

# YOLO 신뢰도
yolo_confidence = st.sidebar.slider(
    "YOLO 신뢰도",
    min_value=0.3,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="객체 감지 신뢰도 임계값"
)
st.session_state.yolo_handler.set_confidence(yolo_confidence)

# 시작/정지 버튼
st.sidebar.markdown("---")
is_running = st.sidebar.toggle(
    "▶️ 시작 / ⏹️ 정지",
    value=st.session_state.is_running
)
st.session_state.is_running = is_running

# 개인정보 보호 안내
st.sidebar.markdown("---")
st.sidebar.info("🔒 **개인정보 보호**: 영상 30초 내 자동 파기")


# ===== 메인 콘텐츠 =====
st.title("E-SAFE: 에스컬레이터 안전 실시간 모니터링 시스템")

# 상태 배너 (동적 갱신용)
status_placeholder = st.empty()

# 좌우 컬럼 레이아웃
col_video, col_status = st.columns([2, 1])

with col_video:
    st.subheader("📹 실시간 피드")
    video_frame_placeholder = st.empty()

with col_status:
    st.subheader("📊 상태 패널")
    person_count_metric = st.empty()
    
    # 5개 위험 유형 상태
    st.markdown("**위험 감지 상태:**")
    danger_row1 = st.columns(2)
    danger_row2 = st.columns(2)
    danger_row3 = st.columns(1)
    
    danger_indicators = {
        "캐리어": danger_row1[0].empty(),
        "낙상": danger_row1[1].empty(),
        "역방향": danger_row2[0].empty(),
        "영유아": danger_row2[1].empty(),
        "혼잡": danger_row3[0].empty(),
    }
    
    # 혼잡도 게이지
    st.markdown("**혼잡도 레벨:**")
    crowding_progress = st.empty()
    
    # 현재 레벨
    st.markdown("**현재 응답 레벨:**")
    level_display = st.empty()


# ===== 하단 로그 섹션 =====
st.markdown("---")
col_log1, col_log2 = st.columns([2, 1])

with col_log1:
    st.subheader("📋 이벤트 로그")
    log_table = st.empty()

with col_log2:
    st.subheader("📈 위험 통계")
    stat_chart = st.empty()


# ===== 메인 처리 루프 =====
def process_frame(frame: np.ndarray, frame_index: int = 0):
    """
    단일 프레임 처리
    
    Args:
        frame: 입력 프레임 (BGR)
        frame_index: 프레임 인덱스
        
    Returns:
        (annotated_frame, current_level, event_info)
    """
    h, w = frame.shape[:2]
    
    # 프레임 리사이징 (추론 속도 개선)
    inference_h, inference_w = 480, 640
    frame_resized = cv2.resize(frame, (inference_w, inference_h))
    
    # RGB 변환 (YOLO는 BGR 지만 mediapipe는 RGB 필요)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    # ===== 객체 감지 =====
    boxes, confidences, class_ids = st.session_state.yolo_handler.detect(frame_resized)
    
    if len(boxes) == 0:
        # 감지 실패 - 빈 상태로 반환
        annotated_frame = frame_resized.copy()
        event_info = {
            'carrier': None,
            'fall': None,
            'direction': None,
            'child': None,
            'crowd': (0, 0),  # 인원 0명, 임계값 0
            'current_level': '정상'
        }
        return annotated_frame, "정상", event_info
    
    # ===== 객체 추적 =====
    tracked_boxes, tracked_ids, tracked_conf, tracked_class_ids = st.session_state.tracker.track(
        boxes, confidences, class_ids
    )
    
    # 클래스별 필터링 (0=person, 24=suitcase)
    person_boxes, person_conf = st.session_state.yolo_handler.filter_by_class(
        tracked_boxes, tracked_conf, tracked_class_ids, 0
    )
    carrier_boxes, carrier_conf = st.session_state.yolo_handler.filter_by_class(
        tracked_boxes, tracked_conf, tracked_class_ids, 24
    )
    
    person_ids = np.array([tracked_ids[i] for i in range(len(tracked_class_ids)) 
                          if tracked_class_ids[i] == 0])
    carrier_ids = np.array([tracked_ids[i] for i in range(len(tracked_class_ids)) 
                           if tracked_class_ids[i] == 24])
    
    # ===== 위험 감지 =====
    event_info = {
        'carrier': None,
        'fall': None,
        'direction': None,
        'child': None,
        'crowd': None,
        'current_level': '정상'
    }
    
    current_level = '정상'
    max_level_priority = 0  # 0=정상, 1=주의, 2=경고, 3=긴급
    
    # 1. 캐리어 위험 감지
    is_danger, label, confidence = st.session_state.carrier_detector.update(
        carrier_boxes, carrier_ids, person_boxes
    )
    if is_danger:
        event_info['carrier'] = (label, confidence)
        level = st.session_state.response_system.process_danger(label, confidence)
        level_priority = {'정상': 0, '주의': 1, '경고': 2, '긴급': 3}.get(level, 0)
        if level_priority > max_level_priority:
            current_level = level
            max_level_priority = level_priority
    
    # 2. 낙상 감지 (pose 추정 필요)
    if len(person_boxes) > 0:
        is_danger, label, confidence = st.session_state.fall_detector.update(
            frame_rgb, person_boxes, person_ids
        )
        if is_danger:
            event_info['fall'] = (label, confidence)
            level = st.session_state.response_system.process_danger(label, confidence)
            level_priority = {'정상': 0, '주의': 1, '경고': 2, '긴급': 3}.get(level, 0)
            if level_priority > max_level_priority:
                current_level = level
                max_level_priority = level_priority
    
    # 3. 역방향 감지
    if len(person_boxes) > 0:
        is_danger, label, confidence = st.session_state.direction_detector.update(
            person_boxes, person_ids, direction_value
        )
        if is_danger:
            event_info['direction'] = (label, confidence)
            level = st.session_state.response_system.process_danger(label, confidence)
            level_priority = {'정상': 0, '주의': 1, '경고': 2, '긴급': 3}.get(level, 0)
            if level_priority > max_level_priority:
                current_level = level
                max_level_priority = level_priority
    
    # 4. 영유아 위험 감지
    if len(person_boxes) > 0:
        is_danger, label, confidence = st.session_state.child_detector.update(
            person_boxes, carrier_boxes
        )
        if is_danger:
            event_info['child'] = (label, confidence)
            level = st.session_state.response_system.process_danger(label, confidence)
            level_priority = {'정상': 0, '주의': 1, '경고': 2, '긴급': 3}.get(level, 0)
            if level_priority > max_level_priority:
                current_level = level
                max_level_priority = level_priority
    
    # 5. 혼잡 감지
    is_danger, person_count, threshold = st.session_state.crowd_detector.update(
        len(person_boxes), crowding_threshold
    )
    event_info['crowd'] = (person_count, threshold)
    if is_danger:
        level = st.session_state.response_system.process_danger("과밀", 0.8, location=None)
        level_priority = {'정상': 0, '주의': 1, '경고': 2, '긴급': 3}.get(level, 0)
        if level_priority > max_level_priority:
            current_level = level
            max_level_priority = level_priority
    
    # ===== 프레임 시각화 =====
    annotated_frame = frame_resized.copy()
    
    # 박스 그리기
    if len(person_boxes) > 0:
        annotated_frame = st.session_state.visualizer.draw_boxes(
            annotated_frame, person_boxes, (255, 0, 0), "P", person_ids
        )
    
    if len(carrier_boxes) > 0:
        annotated_frame = st.session_state.visualizer.draw_boxes(
            annotated_frame, carrier_boxes, (0, 0, 255), "C", carrier_ids
        )
    
    # ROI 그리기
    roi = st.session_state.roi_manager.get_roi()
    annotated_frame = st.session_state.visualizer.draw_roi(annotated_frame, roi)
    
    # 인원 수 표시
    annotated_frame = st.session_state.visualizer.draw_count_text(annotated_frame, len(person_boxes))
    
    # 위험 표시
    if event_info['carrier']:
        label, conf = event_info['carrier']
        annotated_frame = st.session_state.visualizer.draw_danger_overlay(annotated_frame, label, conf)
    elif event_info['fall']:
        label, conf = event_info['fall']
        annotated_frame = st.session_state.visualizer.draw_danger_overlay(annotated_frame, label, conf)
    elif event_info['direction']:
        label, conf = event_info['direction']
        annotated_frame = st.session_state.visualizer.draw_danger_overlay(annotated_frame, label, conf)
    elif event_info['child']:
        label, conf = event_info['child']
        annotated_frame = st.session_state.visualizer.draw_danger_overlay(annotated_frame, label, conf)
    
    event_info['current_level'] = current_level
    
    return annotated_frame, current_level, event_info


def update_ui(annotated_frame, current_level, event_info):
    """
    UI 업데이트
    """
    # 안전성 확인
    if event_info is None:
        event_info = {
            'carrier': None,
            'fall': None,
            'direction': None,
            'child': None,
            'crowd': (0, 0),
            'current_level': '정상'
        }
    
    # 상태 배너 색상
    if current_level == '정상':
        status_color = '🟢'
        status_bg = '#90EE90'
    elif current_level == '주의':
        status_color = '🟡'
        status_bg = '#FFFFE0'
    elif current_level == '경고':
        status_color = '🟠'
        status_bg = '#FFE4B5'
    else:  # 긴급
        status_color = '🔴'
        status_bg = '#FFB6C1'
    
    status_text = f"{status_color} **현재 상태: {current_level}**"
    
    # 상태 배너 업데이트 (st.empty 사용)
    with status_placeholder.container():
        st.markdown(f'<div style="background-color: {status_bg}; padding: 15px; border-radius: 5px; text-align: center;">{status_text}</div>',
                   unsafe_allow_html=True)
    
    # 비디오 프레임 업데이트
    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    video_frame_placeholder.image(frame_rgb, use_column_width=True)
    
    # 우측 패널 업데이트
    person_count = event_info['crowd'][0]
    person_count_metric.metric("현재 인원", person_count)
    
    # 위험 상태 표시
    danger_icons = {}
    danger_labels = ["캐리어", "낙상", "역방향", "영유아", "혼잡"]
    danger_checks = [
        event_info['carrier'] is not None,
        event_info['fall'] is not None,
        event_info['direction'] is not None,
        event_info['child'] is not None,
        event_info['crowd'][0] > event_info['crowd'][1]
    ]
    
    danger_indicators["캐리어"].metric("캐리어 방치/낙하", "🔴" if danger_checks[0] else "⚪")
    danger_indicators["낙상"].metric("쓰러짐/의식불명", "🔴" if danger_checks[1] else "⚪")
    danger_indicators["역방향"].metric("역방향 진입", "🔴" if danger_checks[2] else "⚪")
    danger_indicators["영유아"].metric("영유아 탑승", "🔴" if danger_checks[3] else "⚪")
    danger_indicators["혼잡"].metric("과밀 혼잡", "🔴" if danger_checks[4] else "⚪")
    
    # 혼잡도 게이지
    crowding_threshold = event_info['crowd'][1] if event_info['crowd'][1] > 0 else 1
    crowding_ratio = person_count / crowding_threshold
    crowding_progress.progress(min(1.0, crowding_ratio))
    
    # 레벨 표시
    level_display.metric("응답 레벨", current_level)
    
    # 이벤트 로그 업데이트
    events = st.session_state.event_logger.get_recent_events(10)
    if events:
        df_log = pd.DataFrame([
            {
                'Timestamp': e['timestamp'][:19],
                'Type': e['danger_type'],
                'Level': e['level'],
                'Confidence': f"{e['confidence']:.1%}"
            }
            for e in events
        ])
        log_table.dataframe(df_log, use_container_width=True, hide_index=True)
    
    # 통계 차트
    event_counts = st.session_state.event_logger.get_event_count_by_type()
    if event_counts:
        df_stat = pd.DataFrame(list(event_counts.items()), columns=['위험유형', '발생횟수'])
        fig = px.bar(df_stat, x='위험유형', y='발생횟수', color='발생횟수',
                    color_continuous_scale='Reds')
        fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
        stat_chart.plotly_chart(fig, use_container_width=True)
    
    # 사운드 알림
    if current_level == '주의':
        play_alert(800, 0.3, 1)
    elif current_level == '경고':
        play_alert(1000, 0.3, 2)
    elif current_level == '긴급':
        play_alert(1200, 0.5, 3)


# ===== 메인 루프 =====
if st.session_state.is_running:
    if input_mode == "웹캠 (실시간)":
        # 웹캠 캡처
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.warning("⚠️ 웹캠을 사용할 수 없습니다. 데모 모드로 전환합니다.")
            use_demo = True
        else:
            use_demo = False
        
        # 프레임 처리 루프
        frame_count = 0
        while st.session_state.is_running:
            if use_demo:
                # 데모 프레임
                frame = generate_demo_frame(frame_count)
            else:
                ret, frame = cap.read()
                if not ret:
                    st.error("프레임 캡처 실패")
                    break
                
                # 리사이징
                frame = cv2.resize(frame, (640, 480))
            
            frame_count += 1
            
            # 프레임 처리
            annotated_frame, current_level, event_info = process_frame(frame, frame_count)
            
            # UI 업데이트
            update_ui(annotated_frame, current_level, event_info)
            
            # 이벤트 로깅
            if event_info['carrier']:
                label, conf = event_info['carrier']
                st.session_state.event_logger.add_event(label, current_level, conf)
            elif event_info['fall']:
                label, conf = event_info['fall']
                st.session_state.event_logger.add_event(label, current_level, conf)
            elif event_info['direction']:
                label, conf = event_info['direction']
                st.session_state.event_logger.add_event(label, current_level, conf)
            elif event_info['child']:
                label, conf = event_info['child']
                st.session_state.event_logger.add_event(label, current_level, conf)
            
            time.sleep(0.05)  # ~20 FPS
        
        if not use_demo:
            cap.release()
    
    else:
        # 동영상 파일 재생
        if uploaded_video is None:
            st.info("📁 동영상 파일을 업로드하세요.")
        else:
            # 임시 파일 저장
            temp_file = Path("temp_video.mp4")
            temp_file.write_bytes(uploaded_video.read())
            
            # 동영상 캡처
            cap = cv2.VideoCapture(str(temp_file))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_delay = 1 / fps if fps > 0 else 0.033
            
            frame_count = 0
            while st.session_state.is_running:
                ret, frame = cap.read()
                if not ret:
                    st.success("✅ 동영상 재생 완료")
                    break
                
                frame = cv2.resize(frame, (640, 480))
                frame_count += 1
                
                # 프레임 처리
                annotated_frame, current_level, event_info = process_frame(frame, frame_count)
                
                # UI 업데이트
                update_ui(annotated_frame, current_level, event_info)
                
                # 이벤트 로깅
                if event_info['carrier']:
                    label, conf = event_info['carrier']
                    st.session_state.event_logger.add_event(label, current_level, conf)
                elif event_info['fall']:
                    label, conf = event_info['fall']
                    st.session_state.event_logger.add_event(label, current_level, conf)
                elif event_info['direction']:
                    label, conf = event_info['direction']
                    st.session_state.event_logger.add_event(label, current_level, conf)
                elif event_info['child']:
                    label, conf = event_info['child']
                    st.session_state.event_logger.add_event(label, current_level, conf)
                
                time.sleep(frame_delay)
            
            cap.release()
            if temp_file.exists():
                temp_file.unlink()

else:
    # 실행 대기 상태
    st.info("▶️ **시작 버튼을 눌러 모니터링을 시작하세요.**")
    
    # 기본 정보 표시
    annotated_frame = np.ones((480, 640, 3), dtype=np.uint8) * 50
    cv2.putText(annotated_frame, "E-SAFE", (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
    cv2.putText(annotated_frame, "Escalator Safety AI For Everyone", (80, 250), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(annotated_frame, "Incheon International Airport", (120, 350),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
    
    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    with col_video:
        st.image(frame_rgb, use_column_width=True)
