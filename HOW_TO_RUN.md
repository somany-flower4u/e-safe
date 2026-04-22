# 🚀 E-SAFE 앱 실행 방법

## Windows 사용자

### 옵션 1: 배치 파일 (추천)
```bash
run.bat
```
더블클릭하면 자동으로 실행됩니다.

### 옵션 2: PowerShell
```powershell
.\run.ps1
```

### 옵션 3: 명령어
```bash
cd esafe
streamlit run app.py
```

---

## Mac/Linux 사용자

```bash
cd esafe
python -m streamlit run app.py
```

---

## 앱이 실행되면

1. **자동으로 브라우저 열림**
   - http://localhost:8501 에 접속됩니다

2. **만약 자동으로 열리지 않으면**
   - 수동으로 다음 주소 방문:
   - http://localhost:8501

3. **또는 로컬 네트워크에서 접속**
   - http://192.168.XXX.XXX:8501
   - (터미널에서 나오는 정확한 IP 사용)

---

## 앱 종료

- 터미널에서 `Ctrl+C` 입력
- 또는 Streamlit 우측 상단 "⋮ → Close app" 클릭

---

## 트러블슈팅

### "streamlit: command not found"
```bash
# Python 환경 확인
python -m streamlit --version

# 설치 필요시
pip install streamlit==1.35.0
```

### "ModuleNotFoundError"
```bash
# 모든 의존성 설치
pip install -r requirements.txt
```

### 포트 8501이 이미 사용 중
```bash
streamlit run app.py --server.port 8502
```

---

## 시스템 요구사항

- **Python**: 3.10+
- **메모리**: 4GB+
- **디스크**: 3GB+ (YOLO 모델)
- **웹캠**: 있어야 실시간 감지 가능

---

## 모든 기능

✅ 5가지 위험 감지
✅ 3단계 자동화 응답
✅ 실시간 비디오 처리
✅ 이벤트 로깅
✅ 음성/시각 알림

**지금 바로 실행하세요!** 🎉
