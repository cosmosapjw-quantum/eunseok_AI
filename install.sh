#!/bin/bash
# ============================================================
# 헤이 은석! v3.0 - Runpod 설치 스크립트
# RTX 4090 + whisper-large-v3-turbo + XTTS v2
# ============================================================

set -e

echo ""
echo "========================================================"
echo "  🎤 헤이 은석! v3.0 설치"
echo "  RTX 4090 최적화 버전"
echo "========================================================"
echo ""

cd /app

# 1. 시스템 패키지
echo "[1/7] 시스템 패키지 설치..."
apt-get update -qq
apt-get install -y -qq ffmpeg jq > /dev/null 2>&1
echo "  ✓ ffmpeg, jq 설치됨"

# 2. Python 기본 의존성
echo "[2/7] Python 기본 패키지 설치..."
pip install --quiet --ignore-installed blinker
pip install --quiet torch==2.4.0 torchaudio==2.4.0
echo "  ✓ PyTorch 2.4.0"

# 3. STT (faster-whisper)
echo "[3/7] STT 모델 설치 (faster-whisper)..."
pip install --quiet faster-whisper
echo "  ✓ faster-whisper (large-v3-turbo 지원)"

# 4. 화자 인식 (SpeechBrain)
echo "[4/7] 화자 인식 설치..."
pip install --quiet speechbrain
echo "  ✓ SpeechBrain"

# 5. TTS (XTTS v2)
echo "[5/7] TTS 설치 (XTTS v2)..."
pip install --quiet TTS
pip install --quiet transformers==4.40.0
echo "  ✓ XTTS v2"

# 6. 웹 서버
echo "[6/7] 웹 서버 설치..."
pip install --quiet fastapi uvicorn python-multipart soundfile
echo "  ✓ FastAPI"

# 디렉토리 생성
mkdir -p /app/voice_samples
mkdir -p /app/output
mkdir -p /app/pretrained_models

# 7. 성경 JSON 다운로드 및 검증
echo "[7/7] 성경 데이터 다운로드..."
if [ ! -f /app/bible_ko.json ]; then
    wget -q https://raw.githubusercontent.com/thiagobodruk/bible/master/json/ko_ko.json -O /app/bible_ko.json
fi

# 성경 데이터 검증
if [ -f /app/bible_ko.json ]; then
    BOOK_COUNT=$(jq 'length' /app/bible_ko.json 2>/dev/null || echo "0")
    if [ "$BOOK_COUNT" -ge 66 ]; then
        # 창세기 1:1 테스트
        GENESIS_1_1=$(jq -r '.[0].chapters[0][0]' /app/bible_ko.json 2>/dev/null | head -c 30)
        # 요한복음 3:16 테스트
        JOHN_3_16=$(jq -r '.[42].chapters[2][15]' /app/bible_ko.json 2>/dev/null | head -c 30)
        
        echo "  ✓ ${BOOK_COUNT}권 성경 다운로드 완료"
        echo "  ✓ 창세기 1:1 = '${GENESIS_1_1}...'"
        echo "  ✓ 요한복음 3:16 = '${JOHN_3_16}...'"
    else
        echo "  ⚠ 성경 데이터 불완전: ${BOOK_COUNT}권"
    fi
else
    echo "  ⚠ 성경 다운로드 실패"
fi

# 시작 스크립트 생성
cat > /app/start.sh << 'STARTSCRIPT'
#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
cd /app
python server.py
STARTSCRIPT
chmod +x /app/start.sh

# server.py 복사 (GitHub에서 클론한 경우)
if [ -f /app/eunseok_AI/eunseok_AI/bible_ai_runpod/server.py ]; then
    cp /app/eunseok_AI/eunseok_AI/bible_ai_runpod/server.py /app/server.py
    echo ""
    echo "  ✓ server.py 복사됨"
fi

echo ""
echo "========================================================"
echo "  ✅ 설치 완료!"
echo "========================================================"
echo ""
echo "다음 단계:"
echo ""
echo "1. 음성 파일 복사:"
echo "   cp /app/eunseok_AI/eunseok_AI/*.m4a /app/voice_samples/"
echo "   cp /app/eunseok_AI/eunseok_AI/*.mp3 /app/voice_samples/"
echo ""
echo "2. WAV 변환 (TTS용 - 필수!):"
echo "   ffmpeg -i /app/voice_samples/insuk.m4a /app/voice_samples/insuk.wav"
echo ""
echo "3. 서버 실행:"
echo "   ./start.sh"
echo ""
echo "4. 테스트 (브라우저에서):"
echo "   https://YOUR-URL/test?book=요한복음&chapter=3&verse=16"
echo ""
