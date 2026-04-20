<!-- esafe/README.md -->

## E-SAFE 소개

E-SAFE (Escalator Safety AI For Everyone)는 인천국제공항 에스컬레이터·무빙워크의 위험 상황을 실시간으로 감지하고,
감지 단계에 따라 자동 대응(주의/경고/긴급)을 수행하는 AI 안전 관제 데모입니다.
본 프로젝트는 인천공항 AI-PORT 대국민 아이디어 공모전 시연을 위한 프로토타입으로 설계되었습니다.

## 설치 방법

```bash
pip install -r requirements.txt
```

## 실행 방법

```bash
streamlit run app.py
```

## 주요 기능

- YOLOv8n 기반 사람/캐리어 감지
- MediaPipe Pose 기반 쓰러짐/의식 불명 의심 감지
- ByteTrack 기반 다중 객체 추적
- 5가지 위험 유형 감지 및 3단계 자동 대응
- Streamlit 대시보드 실시간 시각화 및 이벤트 로그

## 테스트용 동영상 추천

- Pixabay 검색 (저작권 무료): https://pixabay.com/videos/search/escalator/
- Pixabay 검색 (저작권 무료): https://pixabay.com/videos/search/airport/

## 주의사항

본 프로그램은 인천공항 AI-PORT 아이디어 공모전 시연용 프로토타입입니다.

