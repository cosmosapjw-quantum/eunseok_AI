"""
헤이 은석! - 클라이언트 v3.0
============================

로컬 컴퓨터에서 실행합니다.
마이크로 음성을 녹음하고 Runpod 서버로 전송합니다.

설치:
    pip install requests sounddevice soundfile numpy pyaudio

사용:
    python client.py --server https://YOUR-POD-ID-8000.proxy.runpod.net
"""

import os
import sys
import time
import wave
import argparse
import base64
import tempfile
from pathlib import Path

import requests
import sounddevice as sd
import soundfile as sf
import numpy as np

# 설정
SAMPLE_RATE = 16000
CHANNELS = 1
WAKE_DURATION = 3.0      # 웨이크워드 녹음 시간
BIBLE_DURATION = 5.0     # 성경 구절 녹음 시간


class AudioPlayer:
    """오디오 재생"""
    
    @staticmethod
    def play_base64(audio_b64: str):
        """Base64 오디오 재생"""
        try:
            audio_data = base64.b64decode(audio_b64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name
            
            data, sr = sf.read(tmp_path)
            sd.play(data, sr)
            sd.wait()
            os.unlink(tmp_path)
        except Exception as e:
            print(f"[ERROR] 재생 실패: {e}")
    
    @staticmethod
    def play_file(path: str):
        """파일 재생"""
        try:
            data, sr = sf.read(path)
            sd.play(data, sr)
            sd.wait()
        except Exception as e:
            print(f"[ERROR] 재생 실패: {e}")


class AudioRecorder:
    """오디오 녹음"""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS):
        self.sample_rate = sample_rate
        self.channels = channels
        
    def record(self, duration: float, message: str = "녹음 중...") -> str:
        """녹음 후 임시 파일 경로 반환"""
        print(f"🎤 {message} ({duration}초)")
        
        frames = int(duration * self.sample_rate)
        audio = sd.rec(frames, samplerate=self.sample_rate, channels=self.channels, dtype='int16')
        sd.wait()
        
        # 임시 파일 저장
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio, self.sample_rate)
        return tmp.name


class EunseokClient:
    """헤이 은석! 클라이언트"""
    
    def __init__(self, server_url: str):
        self.server = server_url.rstrip('/')
        self.recorder = AudioRecorder()
        self.player = AudioPlayer()
        
        # 연결 확인
        self._check_connection()
        
    def _check_connection(self):
        """서버 연결 확인"""
        try:
            resp = requests.get(f"{self.server}/health", timeout=10)
            if resp.status_code == 200:
                info = resp.json()
                print(f"\n✅ 서버 연결됨")
                print(f"   GPU: {info.get('gpu', 'N/A')}")
                print(f"   STT: {info.get('stt', 'N/A')}")
                print(f"   TTS: {'준비됨' if info.get('tts_ready') else '참조음성 없음'}")
                print(f"   성경: {info.get('bible', 0)}권")
                print(f"   화자: {info.get('speakers', 0)}명\n")
            else:
                print(f"⚠️ 서버 응답 오류: {resp.status_code}")
        except Exception as e:
            print(f"❌ 서버 연결 실패: {e}")
            print(f"   URL: {self.server}")
            sys.exit(1)
            
    def process_wake(self, audio_path: str) -> dict:
        """웨이크워드 처리"""
        with open(audio_path, 'rb') as f:
            files = {'audio': ('audio.wav', f, 'audio/wav')}
            resp = requests.post(f"{self.server}/process_wake", files=files, timeout=30)
        return resp.json()
    
    def process_bible(self, audio_path: str) -> dict:
        """성경 구절 처리"""
        with open(audio_path, 'rb') as f:
            files = {'audio': ('audio.wav', f, 'audio/wav')}
            resp = requests.post(f"{self.server}/process_bible", files=files, timeout=60)
        return resp.json()
    
    def run(self):
        """메인 루프"""
        print("=" * 50)
        print("  🎤 헤이 은석! v3.0 클라이언트")
        print("  '헤이 은석!'이라고 말하면 시작합니다")
        print("  Ctrl+C로 종료")
        print("=" * 50)
        
        while True:
            try:
                print("\n" + "-" * 40)
                print("'헤이 은석!'을 기다리는 중...")
                
                # 웨이크워드 녹음
                audio_path = self.recorder.record(WAKE_DURATION, "녹음 중")
                
                # 서버로 전송
                print("📤 서버 전송 중...")
                result = self.process_wake(audio_path)
                os.unlink(audio_path)
                
                transcript = result.get('transcript', '')
                wake_detected = result.get('wake_word', False)
                speaker = result.get('speaker', 'unknown')
                confidence = result.get('confidence', 0)
                action = result.get('action', 'none')
                
                print(f"📝 인식: '{transcript}'")
                
                if not wake_detected:
                    continue
                    
                print(f"✨ 웨이크워드 감지!")
                print(f"👤 화자: {speaker} (신뢰도: {confidence:.0%})")
                
                # 향욱 특별 처리
                if action == "hyanguk_1":
                    print("🚫 (향욱 1차 무시)")
                    continue
                elif action == "hyanguk_2":
                    print("🔇 (향욱 2차 무시 - 카운터 리셋)")
                    continue
                
                # 인사 재생
                if result.get('audio'):
                    print(f"🤖 {result.get('text', '')}")
                    self.player.play_base64(result['audio'])
                
                # 성경 구절 녹음
                print("\n📖 성경 구절을 말씀해주세요...")
                audio_path = self.recorder.record(BIBLE_DURATION, "녹음 중")
                
                # 서버로 전송
                print("📤 서버 전송 중...")
                result = self.process_bible(audio_path)
                os.unlink(audio_path)
                
                transcript = result.get('transcript', '')
                print(f"📝 인식: '{transcript}'")
                
                # 성경 구절 재생
                if result.get('audio'):
                    verse = result.get('text', '')
                    print(f"📖 {verse[:80]}{'...' if len(verse) > 80 else ''}")
                    self.player.play_base64(result['audio'])
                else:
                    print(f"⚠️ {result.get('text', '오류')}")
                    
            except KeyboardInterrupt:
                print("\n\n👋 종료합니다!")
                break
            except Exception as e:
                print(f"❌ 오류: {e}")
                time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="헤이 은석! 클라이언트")
    parser.add_argument('--server', '-s', required=True, help='서버 URL (예: https://xxx-8000.proxy.runpod.net)')
    args = parser.parse_args()
    
    client = EunseokClient(args.server)
    client.run()


if __name__ == "__main__":
    main()
