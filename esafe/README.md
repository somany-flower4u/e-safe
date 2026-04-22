# E-SAFE: Escalator Safety AI For Everyone
## 인천공항 에스컬레이터 안전 실시간 모니터링 시스템

### 📌 프로젝트 개요

E-SAFE는 AI 기반 실시간 에스컬레이터 안전 모니터링 시스템입니다. 웹캠 또는 동영상 파일로부터 **5가지 위험 상황**을 감지하고, **3단계 자동 응답**을 통해 안전을 보장합니다.

---

### 🎯 주요 기능

#### 5가지 위험 감지
1. **캐리어 위험** (`carrier_detector.py`)
   - 방치: 3초 이상 움직이지 않는 캐리어
   - 낙하: 빠른 하강 움직임 감지
   - 미감시: 근처에 사람이 없는 캐리어

2. **낙상/발작** (`fall_detector.py`)
   - MediaPipe Pose를 활용한 자세 분석
   - 쓰러짐: 어깨/엉덩이가 무릎보다 아래
   - 의식불명: 코가 프레임 75% 아래에 2초 이상 정지

3. **역방향 진입** (`direction_detector.py`)
   - 5프레임 슬라이딩 윈도우로 이동 방향 추적
   - 에스컬레이터 방향과 반대 방향 1.5초 이상 이동 감지

4. **영유아 위험** (`child_detector.py`)
   - 박스 높이 기반 어린이 분류
   - 단독 탑승: 보호자 없이 탑승 (200px 이내 성인 없음)
   - 캐리어 탑승: 어린이가 캐리어 위에 있음 (IoU > 0.3)

5. **과밀 혼잡** (`crowd_detector.py`)
   - ROI 내 사람 수 실시간 모니터링
   - 사용자 설정 임계값(2~10명) 초과 시 경고

#### 3단계 자동 응답

| 레벨 | 유형 | 시각 신호 | 음성 신호 | 로그 |
|------|------|---------|---------|------|
| **주의** (CAUTION) | 방치, 과밀 | 🟡 노란색 배너 | 800Hz, 0.3초, 1회 | ✓ |
| **경고** (WARNING) | 역방향, 영유아 | 🟠 주황색 배너 | 1000Hz, 0.3초, 2회 | ✓ |
| **긴급** (EMERGENCY) | 낙상, 낙하 | 🔴 빨간색 배너 + 깜빡임 | 1200Hz, 0.5초, 3회 | ✓ CSV 저장 |

---

### 🛠 기술 스택

| 항목 | 라이브러리 | 용도 |
|------|----------|------|
| **UI** | Streamlit 1.35.0+ | 웹 대시보드 |
| **객체 감지** | YOLOv8 nano (ultralytics 8.2.0+) | 사람/캐리어 감지 |
| **자세 추정** | MediaPipe 0.10.14+ | 낙상 감지 |
| **추적** | Supervision 0.21.0+ (ByteTracker) | 객체 ID 추적 |
| **비디오** | OpenCV (headless) 4.9.0+ | 영상 처리 |
| **음성** | Pygame 2.6.0+ | 음성 알림 |
| **시각화** | Plotly 5.22.0+ | 실시간 그래프 |
| **데이터** | Pandas 2.2.0+ | 로그 관리 |

**환경**: CPU 전용 (GPU 불필요), Python 3.10+

---

### 📁 프로젝트 구조

```
esafe/
├── app.py                    # 메인 Streamlit 애플리케이션
├── requirements.txt          # 의존성 목록
├── README.md                 # 이 파일
│
├── detectors/                # 위험 감지 모듈
│   ├── __init__.py
│   ├── carrier_detector.py   # 캐리어 위험 감지
│   ├── fall_detector.py      # 낙상/발작 감지
│   ├── direction_detector.py # 역방향 감지
│   ├── child_detector.py     # 영유아 위험 감지
│   └── crowd_detector.py     # 과밀 혼잡 감지
│
├── core/                     # 핵심 시스템 모듈
│   ├── __init__.py
│   ├── yolo_handler.py       # YOLOv8 래퍼
│   ├── pose_handler.py       # MediaPipe Pose 래퍼
│   ├── tracker.py            # ByteTracker 래퍼
│   └── response_system.py    # 자동 응답 시스템
│
└── utils/                    # 유틸리티 모듈
    ├── __init__.py
    ├── visualization.py      # 프레임 시각화
    ├── roi_manager.py        # ROI 관리
    └── event_logger.py       # 이벤트 로깅
```

---

### 🚀 설치 및 실행

#### 1. 환경 설정

```bash
# 저장소 클론 또는 파일 복사
cd esafe

# 가상 환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

#### 2. 모델 다운로드 (자동)

첫 실행 시 YOLOv8 nano 모델이 자동으로 다운로드됩니다 (~100MB).

```bash
# 선택사항: 사전 다운로드
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

#### 3. 애플리케이션 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 자동 열림

---

### 📊 UI 레이아웃

```
┌─────────────────────────────────────────────────────────────────┐
│ 🛡️ E-SAFE 관제 대시보드 — 인천공항                               │ ← 사이드바
├─────────────────────────────────────────────────────────────────┤
│ 🔴 **현재 상태: 긴급**                                           │ ← 상태 배너
├───────────────────────────────────────────┬─────────────────────┤
│                                           │ 현재 인원: 12        │
│  📹 실시간 피드                           ├─────────────────────┤
│  (640×480 비디오)                         │ 📊 상태 패널:        │
│  - 사람 박스 (파랑)                       │ ⚪ 캐리어 방치/낙하  │
│  - 캐리어 박스 (빨강)                     │ 🔴 쓰러짐/의식불명  │
│  - 어린이 박스 (보라)                     │ ⚪ 역방향 진입       │
│  - ROI 영역 (초록 점선)                   │ ⚪ 영유아 탑승       │
│  - 인원 수 표시                           │ 🔴 과밀 혼잡        │
│                                           │                     │
│                                           │ 혼잡도: [████░]     │
│                                           │ 응답 레벨: 긴급     │
├───────────────────────────────────────────┴─────────────────────┤
│ 📋 이벤트 로그             │ 📈 위험 통계                         │
│ [테이블]                  │ [막대 그래프]                        │
└───────────────────────────┴────────────────────────────────────┘
```

#### 사이드바 컨트롤
- **입력 모드**: 웹캠(실시간) / 동영상 파일
- **에스컬레이터 방향**: 하행(위→아래) / 상행(아래→위)
- **혼잡 임계값**: 슬라이더 (2~10명, 기본값 5명)
- **YOLO 신뢰도**: 슬라이더 (0.3~0.9, 기본값 0.5)
- **시작/정지**: 토글 버튼

---

### 🔧 주요 시스템 설계

#### 거짓 양성 방지 (False Positive Prevention)
모든 위험 판정은 4개 연속 프레임 이상에서 확인 후 트리거:
```python
from collections import deque
detector_history = deque(maxlen=4)  # 각 감지기마다
```

#### 이벤트 쿨다운
같은 유형의 위험은 10초 후에만 다시 트리거:
```python
cooldown_seconds = 10
last_event_time = {}  # {event_type: timestamp}
```

#### 성능 최적화
- 프레임 리사이징: 640×480 (추론 전)
- Pose 추정: 사람 ROI에만 적용 (전체 프레임 X)
- st.empty() 사용: 페이지 재로드 없이 동적 갱신
- 단일 스레드: Streamlit 호환성

**목표 FPS**: 15 FPS 이상 (최신 노트북 CPU)

#### ROI (관심 영역)
기본 설정: 프레임 중앙 70% (x: 15%~85%, y: 10%~90%)
```python
roi = (0.15, 0.1, 0.85, 0.9)  # 정규화 좌표
```

---

### 🔊 음성 알림 (Pygame 기반)

외부 WAV 파일 없이 NumPy로 사인파 생성:

```python
def generate_beep(frequency: float, duration: float):
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = np.sin(2 * np.pi * frequency * t)
    # → Pygame Sound로 변환 후 재생
```

**주파수별 알림**:
- 🟡 주의: 800Hz × 1회
- 🟠 경고: 1000Hz × 2회
- 🔴 긴급: 1200Hz × 3회

---

### 📹 입력 모드

#### 웹캠 (실시간)
- `cv2.VideoCapture(0)` 사용
- 캡처 실패 시 자동으로 **데모 모드**로 전환
- 데모: 합성 프레임에 움직이는 박스 시뮬레이션

#### 동영상 파일
- 지원 형식: MP4, AVI, MOV
- 자동 리사이징 및 FPS 조절
- 재생 완료 후 자동 정리

---

### 📋 이벤트 로그

CSV 파일로 자동 저장 (`esafe_events.csv`):
```
Timestamp,Danger Type,Level,Confidence,Location X,Location Y
2026-04-22T15:30:45.123456,낙화,긴급,0.950,320,240
2026-04-22T15:30:48.456789,역방향,경고,0.850,180,150
```

---

### 🎓 사용 예제

#### 실시간 모니터링 시작

1. Streamlit 실행: `streamlit run app.py`
2. 사이드바에서 **"웹캠 (실시간)" 선택**
3. **▶️ 시작 버튼 클릭**
4. 실시간 피드 및 알림 확인

#### 동영상 분석

1. **"동영상 파일 업로드" 선택**
2. MP4 파일 업로드
3. **▶️ 시작 버튼 클릭**
4. 프레임별 분석 및 로그 생성

---

### ⚙️ 설정 커스터마이징

각 감지기의 파라미터를 조정하여 민감도 조절:

#### `carrier_detector.py`
```python
CarrierHazardDetector(
    stationary_threshold=10,      # 정지 판정 픽셀 거리
    stationary_frames=3,          # 방치 판정 시간 (×30 프레임)
    velocity_threshold=15.0,      # 낙하 판정 Y 속도
    person_radius=150             # 미감시 판정 거리
)
```

#### `fall_detector.py`
```python
FallDetector(
    visibility_threshold=0.6,     # 랜드마크 신뢰도
    unconscious_frames=2          # 의식불명 판정 시간 (×30 프레임)
)
```

#### `crowd_detector.py`
```python
CrowdingDetector(
    default_threshold=5           # 기본 혼잡 인원 (UI 슬라이더로도 변경)
)
```

---

### 🐛 문제 해결

#### 웹캠 인식 불가
```
⚠️ 웹캠을 사용할 수 없습니다. 데모 모드로 전환합니다.
```
→ 자동으로 합성 프레임 사용

#### GPU 사용 원치 않음
자동으로 CPU 전용 모드 실행 (설정 필요 없음)

#### 느린 성능 (< 15 FPS)
- YOLO 신뢰도 증가 (0.5 → 0.7)
- 프레임 해상도 감소 (app.py의 `inference_h, inference_w` 수정)
- 불필요한 감지기 비활성화

#### Pygame 사운드 오류
자동으로 무음 처리 (try/except 래핑됨)

---

### 📝 라이선스 및 기여

이 프로젝트는 인천국제공항의 안전 개선을 위해 개발되었습니다.

**의존성 라이선스**:
- YOLOv8: AGPL-3.0
- MediaPipe: Apache 2.0
- Streamlit: Apache 2.0
- OpenCV: BSD 3-Clause

---

### 📞 지원

문제 발생 시:
1. 콘솔 로그 확인
2. 모든 의존성 최신 버전으로 업데이트
3. 가상 환경 재생성

---

### 🔮 향후 개선 사항

- [ ] GPU 지원 (CUDA/ROCm)
- [ ] 영상 저장 기능
- [ ] 웹훅 연동 (즉시 알림)
- [ ] 다중 카메라 지원
- [ ] 클라우드 분석
- [ ] 이상 탐지 (Anomaly Detection)

---

**E-SAFE v1.0** | 2026-04-22 | 인천공항 안전 프로젝트
