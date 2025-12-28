# 🎤 헤이 은석! v3.0

교회 코미디용 AI 성경 봇 - "헤이 은석!"이라고 부르면 성경 구절을 읽어주는 AI

## ✨ 특징

- **STT**: whisper-large-v3-turbo (한국어 95%+ 정확도)
- **TTS**: XTTS v2 (자연스러운 음성 복제)
- **화자인식**: SpeechBrain ECAPA-TDNN
- **성경**: 개역한글 66권 전체 지원
- **GPU**: RTX 4090 최적화

## 📁 프로젝트 구조

```
bible_ai_runpod/
├── server.py              # Runpod 서버 코드
├── client.py              # 로컬 클라이언트 코드
├── install.sh             # Runpod 설치 스크립트
├── start.sh               # 서버 시작 스크립트
├── requirements_server.txt # 서버 의존성
├── requirements_client.txt # 클라이언트 의존성
└── README.md              # 이 파일
```

## 🚀 Runpod 서버 설정

### Step 1: Pod 생성

1. [Runpod](https://runpod.io) 접속
2. **Deploy** → **GPU Pod**
3. 설정:
   - **GPU**: `RTX 4090` (24GB) - $0.44/hr
   - **Container Disk**: `50GB`
   - **Template**: `RunPod Pytorch 2.4.0`
   - **Expose HTTP Ports**: `8000`

### Step 2: 설치

Pod 터미널에서:

```bash
cd /app

# GitHub에서 코드 다운로드
git clone https://github.com/cosmosapjw-quantum/eunseok_AI.git
cd eunseok_AI/eunseok_AI/bible_ai_runpod

# 설치 스크립트 실행
chmod +x install.sh
./install.sh
```

### Step 3: 음성 파일 복사

```bash
# 음성 파일을 voice_samples 폴더로 복사
cp /app/eunseok_AI/eunseok_AI/*.m4a /app/voice_samples/
cp /app/eunseok_AI/eunseok_AI/*.mp3 /app/voice_samples/

# WAV 변환 (TTS 참조용)
ffmpeg -i /app/voice_samples/insuk.m4a /app/voice_samples/insuk.wav
```

### Step 4: 서버 실행

```bash
cd /app
./start.sh
```

예상 출력:
```
============================================================
  🎤 헤이 은석! v3.0
  📊 STT: whisper-large-v3-turbo
  🔊 TTS: XTTS v2
============================================================

[GPU] NVIDIA GeForce RTX 4090 (24.0GB)
[MODEL] Whisper 로딩: large-v3-turbo
[MODEL] Whisper 로드 완료!
[MODEL] 화자 인식 모델 로딩...
  ✓ jiwon: me.mp3
  ✓ moksa: moksa.mp3
[MODEL] XTTS v2 로딩...
  ✓ 참조 음성: insuk.wav
[MODEL] XTTS 로드 완료!
[DATA] 성경 로딩: /app/bible_ko.json
[DATA] 66권 로드

============================================================
  ✅ 서버 준비 완료!
  📖 성경: 66권
  👥 화자: 2명
  🎙️ TTS: 준비됨
============================================================

INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 5: 서버 URL 확인

Runpod 대시보드에서:
1. Pod 클릭
2. **Connect** 버튼
3. **HTTP Service [Port 8000]** URL 복사

URL 형식: `https://[POD-ID]-8000.proxy.runpod.net`

---

## 💻 로컬 클라이언트 설정

### Step 1: 의존성 설치

```bash
pip install requests sounddevice soundfile numpy
```

macOS의 경우:
```bash
brew install portaudio
pip install pyaudio
```

### Step 2: 클라이언트 실행

```bash
python client.py --server https://YOUR-POD-ID-8000.proxy.runpod.net
```

### 사용법

1. 프로그램이 시작되면 **"헤이 은석!"**이라고 말합니다
2. 인사말이 나오면 **성경 구절**을 말합니다
   - 예: "요한복음 3장 16절"
   - 예: "창세기 1장 1절"
   - 예: "시편 23편 1절"
3. AI가 해당 구절을 읽어줍니다

---

## 🔧 API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/` | GET | 서버 정보 |
| `/health` | GET | 상태 확인 |
| `/voices` | GET | 음성 파일 목록 |
| `/upload` | POST | 음성 파일 업로드 |
| `/process_wake` | POST | 웨이크워드 처리 |
| `/process_bible` | POST | 성경 구절 처리 |
| `/tts` | POST | 텍스트 → 음성 |
| `/test` | GET | 구절 테스트 |
| `/reload` | POST | 음성 리로드 |

### 테스트 예시

```bash
# 서버 상태 확인
curl https://YOUR-URL/health

# 성경 구절 테스트
curl "https://YOUR-URL/test?book=요한복음&chapter=3&verse=16"
```

---

## 📂 음성 파일

### 필요한 파일

| 파일명 | 용도 | 필수 |
|--------|------|------|
| `insuk.wav` | TTS 참조 음성 (은석 목소리) | ⭕ |
| `me.mp3` | 지원 화자 인식용 | ❌ |
| `moksa.mp3` | 목사님 화자 인식용 | ❌ |
| `hyanguk.mp3` | 향욱 화자 인식용 | ❌ |

### WAV 변환

```bash
# m4a → wav
ffmpeg -i insuk.m4a insuk.wav

# mp3 → wav
ffmpeg -i insuk.mp3 insuk.wav
```

---

## ⚙️ 설정 변경

`server.py`의 `Config` 클래스에서:

```python
@dataclass
class Config:
    # STT 모델 (large-v3-turbo 권장)
    whisper_model: str = "large-v3-turbo"
    
    # 화자인식 임계값 (0.0~1.0, 낮을수록 관대)
    speaker_threshold: float = 0.18
    
    # TTS 언어
    tts_language: str = "ko"
```

### Whisper 모델 옵션

| 모델 | VRAM | 속도 | 정확도 |
|------|------|------|--------|
| `small` | 2GB | ⚡⚡⚡ | ⭐⭐ |
| `medium` | 5GB | ⚡⚡ | ⭐⭐⭐ |
| `large-v3-turbo` | 6GB | ⚡⚡ | ⭐⭐⭐⭐ |
| `large-v3` | 10GB | ⚡ | ⭐⭐⭐⭐⭐ |

---

## 🐛 문제 해결

### cuDNN 오류

```
Unable to load libcudnn_ops.so.9.1.0
```

해결:
```bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```

또는 `./start.sh` 사용

### 음성 파일 인식 안됨

```bash
# 파일 확인
ls -la /app/voice_samples/

# 리로드
curl -X POST https://YOUR-URL/reload
```

### STT 정확도 낮음

1. 마이크 품질 확인
2. 조용한 환경에서 녹음
3. 천천히 또박또박 말하기

---

## 📊 성능

RTX 4090 기준:

| 항목 | 시간 |
|------|------|
| STT (3초 오디오) | 0.3~0.5초 |
| TTS (짧은 문장) | 2~3초 |
| TTS (긴 구절) | 5~8초 |

---

## 📜 라이선스

MIT License

## 🙏 감사

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Coqui TTS](https://github.com/coqui-ai/TTS)
- [SpeechBrain](https://github.com/speechbrain/speechbrain)
- 성경 데이터: [thiagobodruk/bible](https://github.com/thiagobodruk/bible)
