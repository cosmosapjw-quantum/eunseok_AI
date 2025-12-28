"""
헤이 은석! - Runpod API 서버 v3.0
================================

최고 품질 버전:
- STT: whisper-large-v3-turbo (한국어 95%+ 정확도)
- TTS: XTTS v2 (자연스러운 음성 복제)
- 화자인식: SpeechBrain ECAPA-TDNN
- 성경: 오프라인 JSON (66권 완벽 지원)

실행:
    ./start.sh
"""

import os
import json
import time
import tempfile
import base64
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum

import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import Response as FastAPIResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import soundfile as sf
from faster_whisper import WhisperModel
from speechbrain.inference.speaker import SpeakerRecognition
from TTS.api import TTS


# ============================================================================
# 설정
# ============================================================================

@dataclass
class Config:
    """서버 설정"""
    host: str = "0.0.0.0"
    port: int = 8000
    
    voice_dir: str = "/app/voice_samples"
    output_dir: str = "/app/output"
    model_dir: str = "/app/pretrained_models"
    bible_path: str = "/app/bible_ko.json"
    
    # STT - large-v3-turbo가 가장 좋은 균형
    whisper_model: str = "large-v3-turbo"
    whisper_device: str = "cuda"
    whisper_compute: str = "float16"
    
    # 화자인식 임계값
    speaker_threshold: float = 0.18
    
    # TTS
    tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts_language: str = "ko"


class Speaker(str, Enum):
    JIWON = "jiwon"
    MOKSA = "moksa"
    HYANGUK = "hyanguk"
    UNKNOWN = "unknown"


SPEAKER_FILES = {
    Speaker.JIWON: ["me", "jiwon"],
    Speaker.MOKSA: ["moksa"],
    Speaker.HYANGUK: ["hyanguk"],
}


# ============================================================================
# 화자 인식
# ============================================================================

class SpeakerRecognizer:
    """SpeechBrain 기반 화자 인식"""
    
    def __init__(self, config: Config):
        self.config = config
        self.samples: dict[Speaker, str] = {}
        
        print("[MODEL] 화자 인식 모델 로딩...")
        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=f"{config.model_dir}/spkrec",
            run_opts={"device": "cuda"}
        )
        self._load_samples()
        
    def _load_samples(self):
        for speaker, filenames in SPEAKER_FILES.items():
            for name in filenames:
                for ext in [".wav", ".mp3", ".m4a"]:
                    path = f"{self.config.voice_dir}/{name}{ext}"
                    if os.path.exists(path):
                        self.samples[speaker] = path
                        print(f"  ✓ {speaker.value}: {name}{ext}")
                        break
                if speaker in self.samples:
                    break
                    
    def identify(self, audio_path: str) -> Tuple[Speaker, float]:
        best = (Speaker.UNKNOWN, -1.0)
        for speaker, sample in self.samples.items():
            try:
                score, _ = self.model.verify_files(sample, audio_path)
                score = float(score.item())
                if score > best[1]:
                    best = (speaker, score)
            except Exception as e:
                print(f"[ERROR] 화자 비교 실패: {e}")
        if best[1] >= self.config.speaker_threshold:
            return best
        return (Speaker.UNKNOWN, best[1])
    
    def reload(self):
        self.samples.clear()
        self._load_samples()
        
    def list_speakers(self) -> List[dict]:
        return [{"speaker": k.value, "file": v} for k, v in self.samples.items()]


# ============================================================================
# 음성 인식 (STT)
# ============================================================================

class STT:
    """Whisper 기반 음성 인식"""
    
    WAKE_WORDS = [
        "헤이 은석", "헤이은석", "hey 은석", "헤이 은서", "에이 은석",
        "애이 은석", "헤이 응석", "헤이은서", "헤이 은숙", "헤이 인석",
        "이 은석", "헤이 윤석", "hey inseok", "hey insuk"
    ]
    
    def __init__(self, config: Config):
        print(f"[MODEL] Whisper 로딩: {config.whisper_model}")
        self.model = WhisperModel(
            config.whisper_model,
            device=config.whisper_device,
            compute_type=config.whisper_compute
        )
        print("[MODEL] Whisper 로드 완료!")
        
    def transcribe(self, audio_path: str) -> str:
        segments, info = self.model.transcribe(
            audio_path,
            language="ko",
            beam_size=5,
            best_of=5,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=200)
        )
        text = " ".join([s.text for s in segments]).strip()
        return text
        
    def is_wake_word(self, text: str) -> bool:
        norm = text.lower().replace(" ", "").replace("?", "").replace("!", "")
        return any(w.replace(" ", "").lower() in norm for w in self.WAKE_WORDS)


# ============================================================================
# 음성 합성 (TTS)
# ============================================================================

class TTSEngine:
    """XTTS v2 음성 합성"""
    
    def __init__(self, config: Config):
        self.config = config
        self.reference = None
        
        print("[MODEL] XTTS v2 로딩...")
        self.tts = TTS(model_name=config.tts_model, progress_bar=True).to("cuda")
        self._find_reference()
        print("[MODEL] XTTS 로드 완료!")
        
    def _find_reference(self):
        for ext in [".wav", ".mp3", ".m4a"]:
            path = f"{self.config.voice_dir}/insuk{ext}"
            if os.path.exists(path):
                self.reference = path
                print(f"  ✓ 참조 음성: insuk{ext}")
                return
        print("  ⚠ insuk 음성 파일 없음!")
        
    def synthesize(self, text: str, output_path: str) -> bool:
        if not self.reference:
            return False
        try:
            self.tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=self.reference,
                language=self.config.tts_language
            )
            return True
        except Exception as e:
            print(f"[ERROR] TTS 실패: {e}")
            return False
            
    def reload(self):
        self._find_reference()


# ============================================================================
# 성경 데이터 (오프라인 JSON)
# ============================================================================

class Bible:
    """
    JSON 기반 성경 데이터
    
    JSON 구조 (thiagobodruk/bible):
    [
        {
            "abbrev": "gn",
            "book": "창세기",
            "chapters": [
                ["1절 내용", "2절 내용", ...],  # 1장 (index 0)
                ["1절 내용", "2절 내용", ...],  # 2장 (index 1)
                ...
            ]
        },
        ...
    ]
    """
    
    # 한글 숫자 → 아라비아 숫자 변환
    NUM_MAP = {
        # 기본 숫자
        '일': '1', '이': '2', '삼': '3', '사': '4', '오': '5',
        '육': '6', '칠': '7', '팔': '8', '구': '9', '십': '10',
        # 11-19
        '십일': '11', '십이': '12', '십삼': '13', '십사': '14', '십오': '15',
        '십육': '16', '십칠': '17', '십팔': '18', '십구': '19',
        # 20-29
        '이십': '20', '이십일': '21', '이십이': '22', '이십삼': '23', 
        '이십사': '24', '이십오': '25', '이십육': '26', '이십칠': '27',
        '이십팔': '28', '이십구': '29',
        # 30-50
        '삼십': '30', '삼십일': '31', '사십': '40', '오십': '50',
        # 100+
        '백': '100', '백오십': '150',
        # STT 오인식 보정
        '신육': '16', '시육': '16', '심육': '16', '시뉵': '16',
        '신칠': '17', '심칠': '17', '신팔': '18', '신구': '19',
    }
    
    # 책 이름 → 인덱스 매핑 (JSON 배열 순서, 0-based)
    BOOK_MAP = {
        # === 구약 (0-38) ===
        "창세기": 0, "창세": 0, "창색이": 0, "상세기": 0,
        "출애굽기": 1, "출애굽": 1, "출에굽기": 1,
        "레위기": 2, "레위": 2,
        "민수기": 3, "민수": 3,
        "신명기": 4, "신명": 4,
        "여호수아": 5, "여호수아기": 5,
        "사사기": 6, "사사": 6,
        "룻기": 7, "룻": 7,
        "사무엘상": 8, "삼상": 8, "사무엘 상": 8,
        "사무엘하": 9, "삼하": 9, "사무엘 하": 9,
        "열왕기상": 10, "왕상": 10, "열왕기 상": 10,
        "열왕기하": 11, "왕하": 11, "열왕기 하": 11,
        "역대상": 12, "대상": 12, "역대 상": 12,
        "역대하": 13, "대하": 13, "역대 하": 13,
        "에스라": 14, "에즈라": 14,
        "느헤미야": 15, "느헤미아": 15,
        "에스더": 16, "에스더기": 16,
        "욥기": 17, "욥": 17,
        "시편": 18, "시평": 18, "씨편": 18, "싯편": 18,
        "잠언": 19, "자면": 19, "잠원": 19,
        "전도서": 20, "전도": 20,
        "아가": 21, "아가서": 21,
        "이사야": 22, "이사아": 22, "이사야서": 22,
        "예레미야": 23, "예레미아": 23, "예레미야서": 23,
        "예레미야애가": 24, "애가": 24,
        "에스겔": 25, "에제키엘": 25,
        "다니엘": 26, "다니엘서": 26,
        "호세아": 27, "호세아서": 27,
        "요엘": 28, "요엘서": 28,
        "아모스": 29, "아모스서": 29,
        "오바댜": 30, "오바디아": 30,
        "요나": 31, "요나서": 31,
        "미가": 32, "미가서": 32,
        "나훔": 33, "나훔서": 33,
        "하박국": 34, "하바국": 34,
        "스바냐": 35, "스바니아": 35,
        "학개": 36, "학게": 36,
        "스가랴": 37, "스가리아": 37,
        "말라기": 38, "말라키": 38,
        
        # === 신약 (39-65) ===
        "마태복음": 39, "마태복": 39, "마태": 39, "마테복음": 39,
        "마가복음": 40, "마가복": 40, "마가": 40,
        "누가복음": 41, "누가복": 41, "누가": 41,
        "요한복음": 42, "요한복": 42, "요한": 42, "요한복은": 42,
        "요한 보금": 42, "요한보금": 42, "요한 먹은": 42, "요한먹은": 42,
        "요한 버금": 42, "요한버금": 42, "요안복음": 42,
        "사도행전": 43, "사도행": 43, "행전": 43,
        "로마서": 44, "로마": 44, "로마써": 44,
        "고린도전서": 45, "고전": 45, "고린도 전서": 45,
        "고린도후서": 46, "고후": 46, "고린도 후서": 46,
        "갈라디아서": 47, "갈라디아": 47,
        "에베소서": 48, "에베소": 48,
        "빌립보서": 49, "빌립보": 49, "필립보서": 49,
        "골로새서": 50, "골로새": 50,
        "데살로니가전서": 51, "데전": 51, "데살로니가 전서": 51,
        "데살로니가후서": 52, "데후": 52, "데살로니가 후서": 52,
        "디모데전서": 53, "딤전": 53, "디모데 전서": 53,
        "디모데후서": 54, "딤후": 54, "디모데 후서": 54,
        "디도서": 55, "디도": 55,
        "빌레몬서": 56, "빌레몬": 56,
        "히브리서": 57, "히브리": 57,
        "야고보서": 58, "야고보": 58,
        "베드로전서": 59, "벧전": 59, "베드로 전서": 59,
        "베드로후서": 60, "벧후": 60, "베드로 후서": 60,
        "요한일서": 61, "요일": 61, "요한 일서": 61,
        "요한이서": 62, "요이": 62, "요한 이서": 62,
        "요한삼서": 63, "요삼": 63, "요한 삼서": 63,
        "유다서": 64, "유다": 64,
        "요한계시록": 65, "계시록": 65, "요한 계시록": 65,
    }
    
    # 인덱스 → 한글 책 이름 (표시용)
    INDEX_TO_NAME = {
        0: "창세기", 1: "출애굽기", 2: "레위기", 3: "민수기", 4: "신명기",
        5: "여호수아", 6: "사사기", 7: "룻기", 8: "사무엘상", 9: "사무엘하",
        10: "열왕기상", 11: "열왕기하", 12: "역대상", 13: "역대하", 14: "에스라",
        15: "느헤미야", 16: "에스더", 17: "욥기", 18: "시편", 19: "잠언",
        20: "전도서", 21: "아가", 22: "이사야", 23: "예레미야", 24: "예레미야애가",
        25: "에스겔", 26: "다니엘", 27: "호세아", 28: "요엘", 29: "아모스",
        30: "오바댜", 31: "요나", 32: "미가", 33: "나훔", 34: "하박국",
        35: "스바냐", 36: "학개", 37: "스가랴", 38: "말라기",
        39: "마태복음", 40: "마가복음", 41: "누가복음", 42: "요한복음",
        43: "사도행전", 44: "로마서", 45: "고린도전서", 46: "고린도후서",
        47: "갈라디아서", 48: "에베소서", 49: "빌립보서", 50: "골로새서",
        51: "데살로니가전서", 52: "데살로니가후서", 53: "디모데전서", 54: "디모데후서",
        55: "디도서", 56: "빌레몬서", 57: "히브리서", 58: "야고보서",
        59: "베드로전서", 60: "베드로후서", 61: "요한일서", 62: "요한이서",
        63: "요한삼서", 64: "유다서", 65: "요한계시록"
    }
    
    def __init__(self, path: str):
        self.data = []
        self.loaded = False
        
        if not os.path.exists(path):
            print(f"[BIBLE] ⚠ 파일 없음: {path}")
            return
            
        print(f"[BIBLE] 로딩: {path}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            
            # 검증
            if not isinstance(self.data, list):
                print(f"[BIBLE] ⚠ 잘못된 형식: 리스트가 아님")
                return
                
            if len(self.data) < 66:
                print(f"[BIBLE] ⚠ 데이터 부족: {len(self.data)}권 (66권 필요)")
            
            # 첫 번째 책 구조 확인
            first_book = self.data[0]
            if "chapters" not in first_book:
                print(f"[BIBLE] ⚠ chapters 필드 없음")
                return
                
            self.loaded = True
            print(f"[BIBLE] ✓ {len(self.data)}권 로드 완료")
            
            # 샘플 검증 (창세기 1:1)
            test_verse = self._get_verse_internal(0, 1, 1)
            if test_verse:
                print(f"[BIBLE] ✓ 검증: 창1:1 = '{test_verse[:30]}...'")
            else:
                print(f"[BIBLE] ⚠ 검증 실패: 창세기 1:1")
                
        except json.JSONDecodeError as e:
            print(f"[BIBLE] ⚠ JSON 파싱 오류: {e}")
        except Exception as e:
            print(f"[BIBLE] ⚠ 로드 오류: {e}")
    
    def _convert_numbers(self, text: str) -> str:
        """한글 숫자 → 아라비아 숫자"""
        result = text
        for kor, num in sorted(self.NUM_MAP.items(), key=lambda x: -len(x[0])):
            result = result.replace(kor, num)
        return result
    
    def _find_book(self, text: str) -> Optional[int]:
        """텍스트에서 책 이름 찾기"""
        clean = text.replace(" ", "")
        for name, idx in sorted(self.BOOK_MAP.items(), key=lambda x: -len(x[0])):
            if name.replace(" ", "") in clean:
                return idx
        return None
    
    def _get_verse_internal(self, book_idx: int, chapter: int, verse: int) -> Optional[str]:
        """내부용: 구절 가져오기"""
        if not self.data or book_idx >= len(self.data):
            return None
        book = self.data[book_idx]
        if "chapters" not in book:
            return None
        chapters = book["chapters"]
        chap_idx = chapter - 1
        if chap_idx < 0 or chap_idx >= len(chapters):
            return None
        verses = chapters[chap_idx]
        verse_idx = verse - 1
        if verse_idx < 0 or verse_idx >= len(verses):
            return None
        return verses[verse_idx]
    
    def parse(self, text: str) -> Optional[Tuple[int, int, int, Optional[int]]]:
        """텍스트 파싱 → (책_인덱스, 장, 절_시작, 절_끝)"""
        print(f"[PARSE] 입력: '{text}'")
        
        converted = self._convert_numbers(text)
        print(f"[PARSE] 숫자 변환: '{converted}'")
        
        book_idx = self._find_book(text)
        if book_idx is None:
            print(f"[PARSE] ✗ 책을 찾지 못함")
            return None
        print(f"[PARSE] 책: {self.INDEX_TO_NAME.get(book_idx)} (idx={book_idx})")
        
        numbers = re.findall(r'(\d+)', converted)
        print(f"[PARSE] 숫자들: {numbers}")
        
        if len(numbers) < 2:
            print(f"[PARSE] ✗ 장/절 숫자 부족")
            return None
            
        chapter = int(numbers[0])
        verse_start = int(numbers[1])
        verse_end = int(numbers[2]) if len(numbers) > 2 else None
        
        print(f"[PARSE] ✓ {self.INDEX_TO_NAME.get(book_idx)} {chapter}:{verse_start}")
        return (book_idx, chapter, verse_start, verse_end)
    
    def get_verse(self, book_idx: int, chapter: int, 
                  v_start: int, v_end: Optional[int] = None) -> str:
        """구절 텍스트 가져오기"""
        if not self.loaded:
            return "성경 데이터가 로드되지 않았습니다."
            
        book_name = self.INDEX_TO_NAME.get(book_idx, f"책{book_idx}")
        
        if book_idx < 0 or book_idx >= len(self.data):
            return f"잘못된 책 번호입니다: {book_idx}"
        
        book = self.data[book_idx]
        chapters = book.get("chapters", [])
        
        chap_idx = chapter - 1
        if chap_idx < 0 or chap_idx >= len(chapters):
            return f"{book_name}에는 {chapter}장이 없습니다. (총 {len(chapters)}장)"
        
        verses_in_chapter = chapters[chap_idx]
        end = v_end if v_end else v_start
        result = []
        
        for v in range(v_start, end + 1):
            verse_idx = v - 1
            if 0 <= verse_idx < len(verses_in_chapter):
                result.append(f"{v}절. {verses_in_chapter[verse_idx]}")
            elif v == v_start:
                return f"{book_name} {chapter}장에는 {v}절이 없습니다. (총 {len(verses_in_chapter)}절)"
        
        if not result:
            return f"{book_name} {chapter}장 {v_start}절을 찾을 수 없습니다."
        return " ".join(result)
    
    def get_info(self) -> dict:
        """성경 데이터 정보"""
        if not self.loaded:
            return {"loaded": False, "books": 0}
        return {
            "loaded": True,
            "books": len(self.data),
            "test_genesis_1_1": self._get_verse_internal(0, 1, 1)[:50] if self._get_verse_internal(0, 1, 1) else None,
            "test_john_3_16": self._get_verse_internal(42, 3, 16)[:50] if self._get_verse_internal(42, 3, 16) else None,
        }


# ============================================================================
# FastAPI 서버
# ============================================================================

app = FastAPI(title="헤이 은석! API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

config: Config = None
speaker_rec: SpeakerRecognizer = None
stt: STT = None
tts: TTSEngine = None
bible: Bible = None
hyanguk_count = 0


class ProcessResponse(BaseModel):
    speaker: str
    confidence: float
    transcript: str
    wake_word: bool
    text: Optional[str] = None
    audio: Optional[str] = None
    action: str


class TTSRequest(BaseModel):
    text: str


@app.on_event("startup")
async def startup():
    global config, speaker_rec, stt, tts, bible
    
    print("\n" + "=" * 60)
    print("  🎤 헤이 은석! v3.0")
    print("  📊 STT: whisper-large-v3-turbo")
    print("  🔊 TTS: XTTS v2")
    print("=" * 60 + "\n")
    
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.voice_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)
    
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] {gpu} ({mem:.1f}GB)\n")
    
    bible = Bible(config.bible_path)
    stt = STT(config)
    speaker_rec = SpeakerRecognizer(config)
    tts = TTSEngine(config)
    
    print("\n" + "=" * 60)
    print("  ✅ 서버 준비 완료!")
    print(f"  📖 성경: {len(bible.data)}권 {'✓' if bible.loaded else '✗'}")
    print(f"  👥 화자: {len(speaker_rec.samples)}명")
    print(f"  🎙️ TTS: {'준비됨' if tts.reference else '참조음성 없음'}")
    print("=" * 60 + "\n")


@app.get("/")
async def root():
    return {"name": "헤이 은석!", "version": "3.0.0", "status": "running"}


@app.get("/health")
async def health():
    gpu = mem = "N/A"
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        mem = f"{used:.1f}GB / {total:.1f}GB"
    return {
        "status": "ok", "gpu": gpu, "memory": mem,
        "stt": config.whisper_model,
        "tts_ready": tts.reference is not None,
        "speakers": len(speaker_rec.samples),
        "bible": bible.get_info() if bible else None
    }


@app.get("/test")
async def test_verse(book: str, chapter: int, verse: int, verse_end: Optional[int] = None):
    """성경 구절 테스트"""
    if not bible or not bible.loaded:
        return {"error": "성경 데이터 없음"}
    book_idx = bible._find_book(book)
    if book_idx is None:
        return {"error": f"책 없음: {book}"}
    text = bible.get_verse(book_idx, chapter, verse, verse_end)
    return {"book": bible.INDEX_TO_NAME.get(book_idx), "chapter": chapter, "verse": verse, "text": text}


@app.get("/test_parse")
async def test_parse(text: str):
    """STT 결과 파싱 테스트"""
    if not bible:
        return {"error": "성경 데이터 없음"}
    result = bible.parse(text)
    if result:
        book_idx, chapter, v_start, v_end = result
        verse_text = bible.get_verse(book_idx, chapter, v_start, v_end)
        return {"input": text, "book": bible.INDEX_TO_NAME.get(book_idx), 
                "chapter": chapter, "verse": v_start, "text": verse_text}
    return {"input": text, "error": "파싱 실패"}


@app.get("/voices")
async def voices():
    files = []
    if os.path.exists(config.voice_dir):
        for f in os.listdir(config.voice_dir):
            files.append({"name": f, "size": os.path.getsize(f"{config.voice_dir}/{f}")})
    return {"files": files, "speakers": speaker_rec.list_speakers(), "tts_ref": tts.reference}


@app.post("/upload")
async def upload(file: UploadFile = File(...), speaker_type: Optional[str] = Form(None)):
    ext = Path(file.filename).suffix.lower()
    if ext not in [".wav", ".mp3", ".m4a", ".ogg"]:
        return {"success": False, "message": "지원하지 않는 형식"}
    name_map = {"jiwon": "me", "me": "me", "moksa": "moksa", "hyanguk": "hyanguk", "insuk": "insuk"}
    save_name = f"{name_map.get(speaker_type.lower() if speaker_type else '', file.filename.split('.')[0])}{ext}"
    path = f"{config.voice_dir}/{save_name}"
    with open(path, "wb") as f:
        f.write(await file.read())
    speaker_rec.reload()
    if speaker_type and speaker_type.lower() == "insuk":
        tts.reload()
    return {"success": True, "filename": save_name}


@app.post("/process_wake", response_model=ProcessResponse)
async def process_wake(audio: UploadFile = File(...)):
    global hyanguk_count
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
        
    try:
        t0 = time.time()
        transcript = stt.transcribe(tmp_path)
        print(f"[STT] '{transcript}' ({time.time()-t0:.2f}s)")
        
        wake = stt.is_wake_word(transcript)
        speaker, conf = speaker_rec.identify(tmp_path)
        print(f"[SPEAKER] {speaker.value} ({conf:.2f})")
        
        resp = ProcessResponse(speaker=speaker.value, confidence=conf,
                               transcript=transcript, wake_word=wake, action="none")
        
        if not wake:
            return resp
            
        if speaker == Speaker.HYANGUK:
            hyanguk_count += 1
            resp.action = f"hyanguk_{min(hyanguk_count, 2)}"
            if hyanguk_count >= 2:
                hyanguk_count = 0
            return resp
            
        greetings = {
            Speaker.JIWON: "네, 안녕하세요 지원 청년! 찾으시는 성경 구절을 말씀해주세요.",
            Speaker.MOKSA: "네, 안녕하세요 목사님! 찾으시는 성경 구절을 말씀해주세요.",
        }
        greeting = greetings.get(speaker, "네, 안녕하세요! 찾으시는 성경 구절을 말씀해주세요.")
        resp.text = greeting
        resp.action = "greeting"
        
        t0 = time.time()
        out = f"{config.output_dir}/greeting.wav"
        if tts.synthesize(greeting, out):
            print(f"[TTS] {time.time()-t0:.2f}s")
            with open(out, "rb") as f:
                resp.audio = base64.b64encode(f.read()).decode()
        return resp
    finally:
        os.unlink(tmp_path)


@app.post("/process_bible", response_model=ProcessResponse)
async def process_bible(audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
        
    try:
        t0 = time.time()
        transcript = stt.transcribe(tmp_path)
        print(f"[STT] '{transcript}' ({time.time()-t0:.2f}s)")
        
        resp = ProcessResponse(speaker="", confidence=0, transcript=transcript, wake_word=False, action="bible")
        
        ref = bible.parse(transcript)
        if ref:
            book_idx, chap, v_start, v_end = ref
            verse = bible.get_verse(book_idx, chap, v_start, v_end)
            resp.text = verse
            print(f"[BIBLE] ✓ {bible.INDEX_TO_NAME.get(book_idx)} {chap}:{v_start}")
        else:
            resp.text = "죄송합니다, 성경 구절을 인식하지 못했습니다. 다시 한번 말씀해주세요."
            print(f"[BIBLE] ✗ 파싱 실패")
            
        t0 = time.time()
        out = f"{config.output_dir}/bible.wav"
        if tts.synthesize(resp.text, out):
            print(f"[TTS] {time.time()-t0:.2f}s")
            with open(out, "rb") as f:
                resp.audio = base64.b64encode(f.read()).decode()
        return resp
    finally:
        os.unlink(tmp_path)


@app.post("/tts")
async def tts_api(request: TTSRequest):
    out = f"{config.output_dir}/tts.wav"
    if tts.synthesize(request.text, out):
        with open(out, "rb") as f:
            return {"audio": base64.b64encode(f.read()).decode()}
    raise HTTPException(500, "TTS 실패")


@app.get("/reset_hyanguk")
async def reset_hyanguk():
    global hyanguk_count
    hyanguk_count = 0
    return {"message": "리셋됨"}


@app.post("/reload")
async def reload():
    speaker_rec.reload()
    tts.reload()
    return {"message": "리로드됨"}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
