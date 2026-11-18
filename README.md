# E-SAFE: 에스컬레이터·무빙워크 위험 감지 및 자동 대응

인천국제공항 **AI-PORT 대국민 아이디어 공모전** 시연용 프로토타입입니다. E-SAFE(Escalator Safety AI For Everyone)는 에스컬레이터와 무빙워크 영상에서 위험 상황을 찾고, 위험 단계에 따라 화면 표시와 경고음으로 대응하는 관제 대시보드입니다.

영상 한 프레임마다 사람과 캐리어를 감지하고 추적한 뒤, 5가지 위험 유형을 규칙으로 판정합니다. 결과는 Streamlit 대시보드에 실시간으로 보여 줍니다.

> English summary: a prototype for the Incheon International Airport AI-PORT idea contest. It detects people and suitcases with YOLOv8n, tracks them with ByteTrack, estimates poses with MediaPipe, and applies rule-based detectors for five escalator hazards. A three-level response (caution, warning, emergency) is shown on a Streamlit dashboard with alert sounds and a CSV event log.

## 주요 기능

- **객체 감지:** YOLOv8n으로 사람(`person`)과 캐리어(`suitcase`)를 찾습니다.
- **다중 객체 추적:** supervision의 ByteTrack으로 프레임 간 ID를 유지합니다.
- **자세 추정:** MediaPipe Pose로 쓰러짐과 의식 불명 의심을 판정합니다.
- **위험 유형 5가지:** 아래 표를 보세요.
- **3단계 자동 대응:** 주의, 경고, 긴급 단계별로 배너와 경고음을 냅니다.
- **대시보드:** 영상 위에 감지 결과를 겹쳐 그리고, 이벤트 로그를 표로, 이벤트 발생 빈도를 막대 그래프로 보여 주고, 로그는 CSV로 저장합니다.

### 위험 유형과 대응 단계

| 유형 | 감지 항목 | 코드 이름 | 단계 |
|---|---|---|---|
| 1. 캐리어 | 캐리어 방치 | `carrier_abandon` | 1 (주의) |
| 1. 캐리어 | 캐리어 낙하 | `carrier_drop` | 3 (긴급) |
| 2. 사람 쓰러짐 | 쓰러짐 | `fall_down` | 3 (긴급) |
| 2. 사람 쓰러짐 | 의식 불명 의심 | `unconscious` | 3 (긴급) |
| 3. 역방향 진입 | 역방향 진입 | `wrong_direction` | 2 (경고) |
| 4. 영유아 | 영유아 단독 탑승 | `child_alone` | 2 (경고) |
| 4. 영유아 | 영유아 캐리어 탑승 | `child_on_carrier` | 2 (경고) |
| 5. 과밀 혼잡 | 입구 과밀 | `crowding` | 1 (주의) |

단계 매핑은 `core/response_system.py`의 `HAZARD_LEVEL`에 있습니다. 각 단계의 동작은 이렇습니다.

| 단계 | 이름 | 동작 |
|---|---|---|
| 1 | 주의 | 노란 경고 배너, 경고음 1회 |
| 2 | 경고 | 주황 경고 배너, 경고음 2회 |
| 3 | 긴급 | 적색 점멸, 경보음 3회, 자동 정지 신호 표시 |

같은 유형의 이벤트는 10초 동안 다시 울리지 않습니다(쿨다운). 긴급 단계의 "자동 정지 신호"는 화면 표시입니다. 실제 에스컬레이터를 제어하는 코드는 없습니다.

### 판정 방식

판정은 학습한 분류 모델이 아니라 규칙입니다. 감지 박스의 위치, 이동 속도, 정지 시간, 자세 관절 위치를 임계값과 비교합니다. 잘못된 경보를 줄이려고 같은 조건이 연속 3프레임 이어질 때만 위험으로 확정합니다.

기본 설정은 다음과 같습니다(30 FPS 가정).

| 항목 | 값 |
|---|---|
| 관심 영역(ROI) | 프레임 중앙 가로 15~85%, 세로 10~90% |
| 캐리어 방치 | 정지 3초 |
| 의식 불명 의심 | 자세가 2초 이상 움직이지 않음, 관절 가시성 0.6 이상 |
| 역방향 진입 | 역방향 이동 1.5초 유지 |
| 영유아 판별 | 같은 ROI 안 성인 평균 키의 0.5배 이하(`child_height_ratio`)인 사람 |
| 과밀 | 입구 영역 인원이 사이드바 설정값(2~10명, 기본 5명) 이상 |

이 값들은 시연용으로 정한 것입니다. 실제 공항 영상으로 검증하거나 튜닝하지 않았습니다.

## 폴더 구조

```
app.py                  Streamlit 앱: 입력 선택, 프레임 처리 루프, 대시보드
core/
  yolo_handler.py       YOLOv8n 추론 (사람·캐리어)
  tracker.py            ByteTrack 추적 래퍼
  pose_handler.py       MediaPipe Pose 추출
  response_system.py    위험 유형 → 3단계 대응, 경고음
detectors/              위험 유형별 감지기
  carrier_detector.py   캐리어 방치·낙하
  fall_detector.py      쓰러짐·의식 불명 의심
  direction_detector.py 역방향 진입
  child_detector.py     영유아 단독·캐리어 탑승
  crowd_detector.py     과밀 혼잡
utils/                  ROI 계산, 오버레이 그리기, 이벤트 로그
assets/warning_beep.wav 기본 경고음
yolov8n.pt              YOLOv8n 사전학습 가중치
```

## 설치와 실행

Python 환경에서 의존성을 설치하고 Streamlit을 실행합니다.

```bash
pip install -r requirements.txt
streamlit run app.py
```

사이드바에서 입력을 고릅니다.
- **웹캠(실시간):** 장치 번호를 정합니다. 열지 못하면 합성 데모 화면으로 바뀝니다.
- **영상 업로드:** 에스컬레이터가 보이는 영상 파일을 올립니다. 끝까지 재생하면 처음부터 반복합니다.
- 에스컬레이터 진행 방향, 과밀 임계 인원, 감지 신뢰도(0.3~0.9), 화면 FPS를 조절할 수 있습니다.
- "이벤트 로그 CSV 저장" 버튼은 실행 폴더에 `events_log.csv`를 만듭니다.

참고할 점이 있습니다.
- 경고음은 `pygame`으로 재생합니다. `pygame`을 설치하지 못한 환경에서는 소리 없이 동작합니다.
- 이 저장소에는 테스트 영상이 없습니다. 저작권 문제가 없는 영상(예: Pixabay의 에스컬레이터·공항 영상)을 직접 받아 쓰세요.

## 한계

- 공모전 시연용이며 실제 설비와 연결되어 있지 않습니다.
- 단일 카메라, 고정된 ROI(화면 비율)를 가정합니다.
- 정확도, 지연 시간 같은 성능 수치는 측정하지 않았습니다.
- 영유아는 사람 박스의 크기로 추정하므로 카메라 각도와 거리에 영향을 받습니다.

## 라이선스

코드는 [GNU AGPL-3.0](LICENSE)을 따릅니다. Ultralytics YOLOv8(AGPL-3.0)을 사용하고 그 가중치(`yolov8n.pt`)를 포함하기 때문입니다. MediaPipe, supervision, Streamlit 등 다른 의존성은 각자의 라이선스를 따릅니다.
