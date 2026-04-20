# esafe/app.py
"""E-SAFE Streamlit 메인 애플리케이션입니다."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import plotly.express as px
import streamlit as st

from core.pose_handler import PoseHandler
from core.response_system import AutoResponseSystem
from core.tracker import ByteTrackerWrapper
from core.yolo_handler import YoloHandler
from detectors.carrier_detector import CarrierHazardDetector
from detectors.child_detector import ChildRiskDetector
from detectors.crowd_detector import CrowdingDetector
from detectors.direction_detector import WrongDirectionDetector
from detectors.fall_detector import FallDetector
from utils.event_logger import EventLogger
from utils.roi_manager import compute_default_roi, compute_entrance_roi
from utils.visualization import render_overlay


HAZARD_LABELS = {
	"carrier_abandon": "캐리어 방치",
	"carrier_drop": "캐리어 낙하",
	"fall_down": "쓰러짐 감지",
	"unconscious": "의식 불명 의심",
	"wrong_direction": "역방향 진입",
	"child_alone": "영유아 단독 탑승",
	"child_on_carrier": "영유아 캐리어 탑승",
	"crowding": "과밀 혼잡",
}

RISK_TYPE_DISPLAY = {
	"carrier": "유형 1 - 캐리어 낙하·방치",
	"fall": "유형 2 - 사람 쓰러짐·발작",
	"direction": "유형 3 - 역방향 진입",
	"child": "유형 4 - 영유아 위험",
	"crowding": "유형 5 - 과밀 혼잡",
}


def init_page() -> None:
	"""Streamlit 페이지 기본 설정을 적용합니다."""
	st.set_page_config(
		page_title="E-SAFE 관제 대시보드",
		page_icon="⚠️",
		layout="wide",
	)


def init_session_state() -> None:
	"""세션 상태 기본값을 초기화합니다."""
	defaults: Dict[str, Any] = {
		"running": False,
		"runtime_initialized": False,
		"capture": None,
		"capture_key": None,
		"source_choice": None,
		"uploaded_file_key": None,
		"uploaded_video_path": None,
		"demo_mode": False,
		"source_message": "",
		"synthetic_frame_index": 0,
		"last_frame_rgb": None,
		"last_metrics": None,
	}

	for key, value in defaults.items():
		if key not in st.session_state:
			st.session_state[key] = value


def release_capture() -> None:
	"""열려 있는 비디오 캡처 자원을 안전하게 해제합니다."""
	capture = st.session_state.get("capture")
	if capture is not None:
		try:
			capture.release()
		except Exception:
			pass
	st.session_state["capture"] = None


def ensure_runtime(conf_threshold: float) -> None:
	"""검출/추적/응답 객체를 세션에 초기화합니다."""
	if not st.session_state.get("runtime_initialized", False):
		st.session_state["yolo_handler"] = YoloHandler(model_name="yolov8n.pt", conf_threshold=conf_threshold)
		st.session_state["pose_handler"] = PoseHandler(
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5,
		)
		st.session_state["tracker"] = ByteTrackerWrapper(fps=30.0)

		st.session_state["carrier_detector"] = CarrierHazardDetector(
			fps=30.0,
			stationary_seconds=3.0,
			required_consecutive_frames=3,
		)
		st.session_state["fall_detector"] = FallDetector(
			fps=30.0,
			min_visibility=0.6,
			unconscious_seconds=2.0,
			required_consecutive_frames=3,
		)
		st.session_state["direction_detector"] = WrongDirectionDetector(
			fps=30.0,
			hold_seconds=1.5,
			required_consecutive_frames=3,
		)
		st.session_state["child_detector"] = ChildRiskDetector(required_consecutive_frames=3)
		st.session_state["crowd_detector"] = CrowdingDetector(required_consecutive_frames=3)
		st.session_state["response_system"] = AutoResponseSystem(cooldown_seconds=10.0)
		st.session_state["event_logger"] = EventLogger(output_csv_path="events_log.csv")
		st.session_state["runtime_initialized"] = True

	st.session_state["yolo_handler"].set_conf_threshold(conf_threshold)


def persist_uploaded_video(uploaded_file: Any) -> str:
	"""업로드 파일을 임시 경로에 저장하고 파일 경로를 반환합니다."""
	file_key = f"{uploaded_file.name}-{uploaded_file.size}"
	cached_key = st.session_state.get("uploaded_file_key")
	cached_path = st.session_state.get("uploaded_video_path")

	if cached_key == file_key and cached_path and Path(cached_path).exists():
		return str(cached_path)

	suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
	with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
		tmp.write(uploaded_file.getbuffer())
		path = tmp.name

	st.session_state["uploaded_file_key"] = file_key
	st.session_state["uploaded_video_path"] = path
	return path


def draw_person_silhouette(
	frame: np.ndarray,
	center_x: int,
	center_y: int,
	color: Tuple[int, int, int],
	scale: float = 1.0,
) -> None:
	"""합성 데모 모드용 사람 실루엣을 그립니다."""
	head_r = int(10 * scale)
	body_w = int(20 * scale)
	body_h = int(38 * scale)
	cv2.circle(frame, (center_x, center_y - body_h // 2), head_r, color, -1)
	cv2.rectangle(
		frame,
		(center_x - body_w // 2, center_y - body_h // 2 + head_r),
		(center_x + body_w // 2, center_y + body_h // 2),
		color,
		-1,
	)


def generate_synthetic_frame(frame_index: int, width: int = 640, height: int = 480) -> np.ndarray:
	"""웹캠/영상이 없는 환경에서 동작 시연용 합성 프레임을 생성합니다."""
	frame = np.zeros((height, width, 3), dtype=np.uint8)

	# 단색 배경 대신 그라데이션을 적용해 영역 구분을 쉽게 만듭니다.
	for y in range(height):
		tone = 20 + int((y / height) * 40)
		frame[y, :] = (tone, tone + 10, tone + 18)

	cv2.putText(
		frame,
		"E-SAFE DEMO MODE",
		(18, 34),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.9,
		(220, 220, 220),
		2,
		cv2.LINE_AA,
	)

	# 에스컬레이터 방향감을 주기 위해 사선 안내선을 그립니다.
	for offset in range(-80, 560, 45):
		cv2.line(
			frame,
			(max(0, offset), 460),
			(min(width - 1, offset + 140), 80),
			(80, 80, 80),
			1,
		)

	y_base = 420 - (frame_index * 3 % 360)
	x_wave = int(320 + 70 * np.sin(frame_index * 0.05))
	draw_person_silhouette(frame, x_wave - 40, y_base, (200, 200, 200), scale=1.0)
	draw_person_silhouette(frame, x_wave + 55, y_base + 80, (180, 180, 200), scale=0.78)

	suitcase_y = 90 + (frame_index * 6 % 300)
	cv2.rectangle(frame, (x_wave + 85, suitcase_y), (x_wave + 125, suitcase_y + 34), (40, 40, 220), -1)

	return frame


def ensure_capture(selected_source: str, uploaded_file: Any) -> None:
	"""현재 입력 설정에 맞는 캡처 소스를 준비합니다."""
	source_changed = st.session_state.get("source_choice") != selected_source
	if source_changed:
		release_capture()
		st.session_state["capture_key"] = None
		st.session_state["demo_mode"] = False
		st.session_state["source_choice"] = selected_source

	if selected_source == "웹캠 (실시간)":
		if st.session_state.get("capture") is None and not st.session_state.get("demo_mode", False):
			webcam = cv2.VideoCapture(0)
			if webcam.isOpened():
				st.session_state["capture"] = webcam
				st.session_state["capture_key"] = "webcam-0"
				st.session_state["source_message"] = "웹캠 실시간 모드"
				return

			webcam.release()
			st.session_state["demo_mode"] = True

			# 웹캠 실패 시 업로드 영상이 있으면 자동으로 데모 입력으로 전환합니다.
			if uploaded_file is not None:
				video_path = persist_uploaded_video(uploaded_file)
				fallback = cv2.VideoCapture(video_path)
				if fallback.isOpened():
					st.session_state["capture"] = fallback
					st.session_state["capture_key"] = video_path
					st.session_state["source_message"] = "웹캠 실패: 업로드 영상 데모 모드"
					return

			st.session_state["source_message"] = "웹캠 실패: 합성 시뮬레이션 데모 모드"
		return

	if uploaded_file is None:
		release_capture()
		st.session_state["capture_key"] = None
		st.session_state["source_message"] = "동영상을 업로드하면 재생이 시작됩니다"
		return

	video_path = persist_uploaded_video(uploaded_file)
	if st.session_state.get("capture_key") != video_path or st.session_state.get("capture") is None:
		release_capture()
		file_cap = cv2.VideoCapture(video_path)
		if file_cap.isOpened():
			st.session_state["capture"] = file_cap
			st.session_state["capture_key"] = video_path
			st.session_state["source_message"] = "업로드 영상 재생 모드"
		else:
			st.session_state["capture"] = None
			st.session_state["capture_key"] = None
			st.session_state["source_message"] = "영상 파일 열기 실패: 합성 시뮬레이션 모드"


def read_next_frame(selected_source: str, uploaded_file: Any) -> Tuple[np.ndarray, str]:
	"""현재 소스에서 다음 프레임을 읽어 반환합니다."""
	ensure_capture(selected_source, uploaded_file)
	capture = st.session_state.get("capture")
	source_message = st.session_state.get("source_message", "")

	if capture is not None:
		success, frame = capture.read()
		if success and frame is not None:
			return frame, source_message

		# 파일 입력 또는 데모 입력은 끝에 도달하면 루프 재생합니다.
		if st.session_state.get("capture_key") and st.session_state.get("capture_key") != "webcam-0":
			capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
			success, frame = capture.read()
			if success and frame is not None:
				return frame, source_message

		if selected_source == "웹캠 (실시간)":
			release_capture()
			st.session_state["demo_mode"] = True
			st.session_state["source_message"] = "웹캠 연결 해제: 합성 시뮬레이션 데모 모드"

	# 어떤 입력도 사용할 수 없으면 합성 시뮬레이션 프레임을 생성합니다.
	frame_index = int(st.session_state.get("synthetic_frame_index", 0))
	st.session_state["synthetic_frame_index"] = frame_index + 1
	demo_frame = generate_synthetic_frame(frame_index)
	return demo_frame, st.session_state.get("source_message", "합성 시뮬레이션 데모 모드")


def process_frame(
	frame_bgr: np.ndarray,
	escalator_direction: str,
	crowd_threshold: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
	"""단일 프레임에 대해 감지-추적-대응-오버레이 파이프라인을 실행합니다."""
	frame = cv2.resize(frame_bgr, (640, 480), interpolation=cv2.INTER_LINEAR)

	roi = compute_default_roi(frame.shape)
	entrance_roi = compute_entrance_roi(roi, escalator_direction)

	yolo_handler: YoloHandler = st.session_state["yolo_handler"]
	tracker: ByteTrackerWrapper = st.session_state["tracker"]
	pose_handler: PoseHandler = st.session_state["pose_handler"]

	detections = yolo_handler.infer(frame)
	tracked_objects = tracker.update(detections)

	persons = [obj for obj in tracked_objects if obj.get("class_name") == "person"]
	suitcases = [obj for obj in tracked_objects if obj.get("class_name") == "suitcase"]
	pose_results = pose_handler.extract_for_people(frame, persons)

	carrier_detector: CarrierHazardDetector = st.session_state["carrier_detector"]
	fall_detector: FallDetector = st.session_state["fall_detector"]
	direction_detector: WrongDirectionDetector = st.session_state["direction_detector"]
	child_detector: ChildRiskDetector = st.session_state["child_detector"]
	crowd_detector: CrowdingDetector = st.session_state["crowd_detector"]

	carrier_hit, carrier_conf, carrier_detail = carrier_detector.detect(suitcases, persons, roi)
	fall_hit, fall_conf, fall_detail = fall_detector.detect(pose_results, frame.shape, roi)
	direction_hit, direction_conf, direction_detail = direction_detector.detect(
		persons,
		roi,
		escalator_direction,
	)
	child_hit, child_conf, child_detail = child_detector.detect(persons, suitcases, roi, frame.shape)
	crowd_hit, crowd_conf, crowd_detail = crowd_detector.detect(persons, entrance_roi, crowd_threshold)

	active_events: List[Dict[str, Any]] = []
	risk_texts: List[str] = []

	if carrier_hit:
		hazard = carrier_detail.get("hazard_type")
		active_events.append(
			{
				"hazard_type": hazard,
				"confidence": carrier_conf,
				"location": carrier_detail.get("location"),
			}
		)
		risk_texts.append(HAZARD_LABELS.get(hazard, "캐리어 위험"))

	if fall_hit:
		hazard = fall_detail.get("hazard_type")
		active_events.append(
			{
				"hazard_type": hazard,
				"confidence": fall_conf,
				"location": fall_detail.get("location"),
			}
		)
		risk_texts.append(HAZARD_LABELS.get(hazard, "사람 위험"))

	if direction_hit:
		hazard = direction_detail.get("hazard_type")
		active_events.append(
			{
				"hazard_type": hazard,
				"confidence": direction_conf,
				"location": direction_detail.get("location"),
			}
		)
		risk_texts.append(HAZARD_LABELS.get(hazard, "역방향 위험"))

	if child_hit:
		hazard = child_detail.get("hazard_type")
		active_events.append(
			{
				"hazard_type": hazard,
				"confidence": child_conf,
				"location": child_detail.get("location"),
			}
		)
		risk_texts.append(HAZARD_LABELS.get(hazard, "영유아 위험"))

	if crowd_hit:
		hazard = crowd_detail.get("hazard_type")
		active_events.append(
			{
				"hazard_type": hazard,
				"confidence": crowd_conf,
				"location": None,
			}
		)
		risk_texts.append(HAZARD_LABELS.get(hazard, "과밀 위험"))

	response_system: AutoResponseSystem = st.session_state["response_system"]
	response = response_system.resolve_response(active_events, now_ts=time.time())

	event_logger: EventLogger = st.session_state["event_logger"]
	for event in response.get("triggered_events", []):
		event_level = int(event.get("level", 0))
		level_meta = response_system.LEVEL_META.get(event_level, response_system.LEVEL_META[0])
		hazard_type = event.get("hazard_type")
		event_logger.add_event(
			risk_type=HAZARD_LABELS.get(hazard_type, str(hazard_type)),
			level=level_meta["stage"],
			response_action=level_meta["action"],
			location=event.get("location"),
			confidence=float(event.get("confidence", 0.0)),
		)

	child_ids = child_detail.get("child_ids", []) if isinstance(child_detail, dict) else []
	overlay_bgr = render_overlay(
		frame=frame,
		tracked_objects=tracked_objects,
		child_ids=child_ids,
		roi=roi,
		entrance_roi=entrance_roi,
		risk_texts=risk_texts,
		response_level=int(response.get("level", 0)),
		timestamp=time.time(),
	)
	overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

	metrics = {
		"person_count": len(persons),
		"crowd_count": int(crowd_detail.get("count", 0)),
		"crowd_threshold": int(crowd_detail.get("threshold", crowd_threshold)),
		"response": response,
		"risk_status": {
			"carrier": carrier_hit,
			"fall": fall_hit,
			"direction": direction_hit,
			"child": child_hit,
			"crowding": crowd_hit,
		},
		"active_risk_labels": risk_texts,
	}
	return overlay_rgb, metrics


def render_banner(response: Dict[str, Any], container: Any | None = None) -> None:
	"""현재 위험 단계 배너를 화면 상단에 렌더링합니다."""
	target = container if container is not None else st
	color = response.get("color", "#1E8E3E")
	stage = response.get("stage", "NORMAL")
	label = response.get("label", "정상")
	action = response.get("action", "정상 운영 상태")
	text_color = "#111111" if stage == "CAUTION" else "#FFFFFF"

	target.markdown(
		f"""
		<div style="
			width:100%;
			padding:14px 16px;
			border-radius:10px;
			background:{color};
			color:{text_color};
			font-weight:700;
			text-align:center;
			margin-bottom:8px;
			letter-spacing:0.2px;
		">
			현재 위험 상태: {stage} ({label}) | {action}
		</div>
		""",
		unsafe_allow_html=True,
	)


def render_status_panel(metrics: Dict[str, Any]) -> None:
	"""우측 실시간 감지 현황 패널을 렌더링합니다."""
	st.subheader("실시간 감지 현황")

	st.metric("감지 인원 수", metrics.get("person_count", 0))

	risk_status: Dict[str, bool] = metrics.get("risk_status", {})
	for key, label in RISK_TYPE_DISPLAY.items():
		status_text = "🔴 감지" if risk_status.get(key, False) else "⚪ 정상"
		st.write(f"{label}: {status_text}")

	response = metrics.get("response", {})
	st.write(f"현재 대응 단계: {response.get('stage', 'NORMAL')} ({response.get('label', '정상')})")

	crowd_count = int(metrics.get("crowd_count", 0))
	crowd_threshold = max(1, int(metrics.get("crowd_threshold", 5)))
	gauge_value = min(1.0, crowd_count / crowd_threshold)
	st.progress(gauge_value)
	st.caption(f"과밀도 게이지: {crowd_count} / {crowd_threshold}")


def render_event_views() -> None:
	"""이벤트 로그 테이블과 실시간 빈도 차트를 렌더링합니다."""
	event_logger: EventLogger = st.session_state["event_logger"]

	st.subheader("이벤트 로그")
	events_df = event_logger.to_dataframe(max_rows=300)
	st.dataframe(events_df, use_container_width=True, height=240)

	st.subheader("실시간 이벤트 발생 빈도")
	freq_df = event_logger.event_frequency_dataframe()
	if freq_df.empty:
		st.info("아직 기록된 이벤트가 없습니다.")
		return

	fig = px.bar(
		freq_df,
		x="위험 유형",
		y="발생 횟수",
		color="위험 유형",
		text="발생 횟수",
		title="위험 유형별 누적 이벤트",
	)
	fig.update_layout(height=320, margin=dict(l=12, r=12, t=60, b=12), showlegend=False)
	st.plotly_chart(fig, use_container_width=True)


def run_dashboard() -> None:
	"""대시보드 메인 UI와 프레임 처리 루프를 실행합니다."""
	st.sidebar.title("E-SAFE 관제 대시보드 — 인천공항")

	source_choice = st.sidebar.radio(
		"입력 소스 선택",
		["웹캠 (실시간)", "동영상 파일 업로드"],
		index=0,
	)
	uploaded_file = st.sidebar.file_uploader(
		"동영상 파일 업로드",
		type=["mp4", "avi", "mov"],
		accept_multiple_files=False,
	)

	direction_text = st.sidebar.selectbox(
		"에스컬레이터 방향 설정",
		["하행(위→아래)", "상행(아래→위)"],
		index=0,
	)
	escalator_direction = "down" if direction_text.startswith("하행") else "up"

	crowd_threshold = st.sidebar.slider("과밀 임계 인원 설정", min_value=2, max_value=10, value=5)
	st.sidebar.caption("ROI 설정 안내: 화면 비율로 자동 설정됨")
	conf_threshold = st.sidebar.slider("감지 모델 신뢰도 임계값", 0.3, 0.9, 0.5, 0.05)

	ensure_runtime(conf_threshold)

	start_pressed = st.sidebar.button("시작", type="primary", use_container_width=True)
	stop_pressed = st.sidebar.button("정지", use_container_width=True)
	save_log_pressed = st.sidebar.button("이벤트 로그 CSV 저장", use_container_width=True)

	if start_pressed:
		st.session_state["running"] = True
	if stop_pressed:
		st.session_state["running"] = False
	if save_log_pressed:
		saved_path = st.session_state["event_logger"].save_csv()
		st.sidebar.success(f"저장 완료: {saved_path}")

	response_for_banner = {
		"stage": "NORMAL",
		"label": "정상",
		"color": "#1E8E3E",
		"action": "정상 운영 상태",
	}
	if st.session_state.get("last_metrics"):
		response_for_banner = st.session_state["last_metrics"].get("response", response_for_banner)

	banner_slot = st.empty()

	main_left, main_right = st.columns([2, 1], gap="large")
	with main_left:
		frame_slot = st.empty()
		caption_slot = st.empty()

	if st.session_state.get("running", False):
		loop_start = time.perf_counter()
		frame_bgr, source_message = read_next_frame(source_choice, uploaded_file)

		frame_rgb, metrics = process_frame(
			frame_bgr=frame_bgr,
			escalator_direction=escalator_direction,
			crowd_threshold=crowd_threshold,
		)
		st.session_state["last_frame_rgb"] = frame_rgb
		st.session_state["last_metrics"] = metrics
		response_for_banner = metrics.get("response", response_for_banner)
		render_banner(response_for_banner, banner_slot)

		with main_left:
			frame_slot.image(frame_rgb, channels="RGB", use_container_width=True)
			caption_slot.caption(source_message)

		with main_right:
			render_status_panel(metrics)

		# 긴급 단계에서는 즉시 팝업 메시지를 출력합니다.
		if metrics.get("response", {}).get("emergency_popup", False):
			st.error("에스컬레이터 자동 정지 신호 발송")

		# 목표 30FPS를 위해 프레임 간 대기 시간을 조절합니다.
		elapsed = time.perf_counter() - loop_start
		sleep_time = max(0.0, (1.0 / 30.0) - elapsed)
		if sleep_time > 0:
			time.sleep(sleep_time)

		render_event_views()
		st.rerun()
		return

	render_banner(response_for_banner, banner_slot)

	with main_left:
		last_frame = st.session_state.get("last_frame_rgb")
		if last_frame is None:
			preview = generate_synthetic_frame(0)
			last_frame = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
		frame_slot.image(last_frame, channels="RGB", use_container_width=True)
		caption_slot.caption("대기 상태입니다. 시작 버튼을 누르면 실시간 감지가 시작됩니다.")

	with main_right:
		if st.session_state.get("last_metrics"):
			render_status_panel(st.session_state["last_metrics"])
		else:
			st.subheader("실시간 감지 현황")
			st.info("시작 전 상태입니다.")

	if source_choice == "웹캠 (실시간)" and st.session_state.get("demo_mode", False):
		st.warning("웹캠 연결에 실패하여 데모 모드로 동작합니다. 동영상 업로드 또는 합성 프레임이 사용됩니다.")

	render_event_views()


def main() -> None:
	"""애플리케이션 진입점입니다."""
	init_page()
	init_session_state()
	run_dashboard()


if __name__ == "__main__":
	main()

