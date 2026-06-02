"""해설/대사 음성에서 카드 식별 + 선택 단서를 추출.

용도:
  - 영상에서 "your card is the queen of hearts" 같은 문장 시점·내용 추출
  - YOLO 시각 검출(`objects.CardTimeline`)과 cross-check해 chosen card를
    높은 신뢰도로 식별

전사는 scripts/auto_label.py와 동일하게 faster-whisper 사용. 같은 영상이
이미 전사돼 있으면 캐시(JSON)를 재사용한다.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


# ---------- 데이터 타입 ----------
@dataclass
class CardMention:
    """전사 텍스트에서 카드 이름이 언급된 한 시점."""
    time_sec: float
    card_id: str       # "QH", "10D" 등 — objects.py 카드 라벨과 동일 형식
    text: str          # 매칭된 원문 세그먼트


@dataclass
class SelectionPhrase:
    """관객 선택을 시사하는 구문이 등장한 시점."""
    time_sec: float
    phrase_kind: str   # 'select' | 'reveal' | 'audience'
    text: str


@dataclass
class NumberMention:
    """음성에서 1~52 범위 숫자가 언급된 시점 (ACAAN 등 숫자-기반 트릭 단서)."""
    time_sec: float
    number: int
    text: str          # 매칭된 원문


@dataclass
class CountingSequence:
    """'1, 2, 3, ..., N' 형태 연속 카운팅이 감지된 구간 (카드 deal 동작 동반 강하게 시사)."""
    start_sec: float
    end_sec: float
    numbers: list[int]  # 카운팅 순서대로 잡힌 숫자들
    text: str           # 원문 (압축)


# ---------- 카드 이름 → ID 매칭 ----------
# 52-class 라벨: rank(A,2-10,J,Q,K) + suit(C,D,H,S)
_RANK_PATTERNS = {
    "A":  [r"\bace\b"],
    "K":  [r"\bking\b"],
    "Q":  [r"\bqueen\b"],
    "J":  [r"\bjack\b"],
    "10": [r"\bten\b", r"\b10\b"],
    "9":  [r"\bnine\b", r"\b9\b"],
    "8":  [r"\beight\b", r"\b8\b"],
    "7":  [r"\bseven\b", r"\b7\b"],
    "6":  [r"\bsix\b", r"\b6\b"],
    "5":  [r"\bfive\b", r"\b5\b"],
    "4":  [r"\bfour\b", r"\b4\b"],
    "3":  [r"\bthree\b", r"\b3\b"],
    "2":  [r"\btwo\b", r"\b2\b"],
}
_SUIT_PATTERNS = {
    "C": [r"\bclubs?\b"],
    "D": [r"\bdiamonds?\b"],
    "H": [r"\bhearts?\b"],
    "S": [r"\bspades?\b"],
}

_RANK_RE = {r: re.compile("|".join(pats), re.I) for r, pats in _RANK_PATTERNS.items()}
_SUIT_RE = {s: re.compile("|".join(pats), re.I) for s, pats in _SUIT_PATTERNS.items()}


# Whisper 흔한 hallucination 보정. 마술 카드명을 잘못 듣는 패턴들.
# (small 모델이 마술 영상의 노이즈/음악 때문에 카드명을 자주 잘못 들음)
_STT_FIXES = [
    # "clubs"의 wild misspellings
    (re.compile(r"\be[\- ]?clover\b", re.I), "of clubs"),
    (re.compile(r"\belf clover\b", re.I), "of clubs"),
    (re.compile(r"\bofclubs?\b", re.I), "of clubs"),
    # "spades" 흔한 오인
    (re.compile(r"\b(spade|spaids?)\b", re.I), "spades"),
    # "diamonds"
    (re.compile(r"\b(daimonds?|dimonds?)\b", re.I), "diamonds"),
    # "hearts"
    (re.compile(r"\b(harts?|hurts?)\b", re.I), "hearts"),
]


def _apply_stt_fixes(text: str) -> str:
    """전사 텍스트에 흔한 카드 misspelling 보정 적용."""
    for pat, repl in _STT_FIXES:
        text = pat.sub(repl, text)
    return text


def _find_card_in_text(text: str) -> str | None:
    """텍스트에 'rank of suit' 패턴이 있으면 'RankSuit'(예: 'QH') 반환."""
    found_rank = None
    rank_pos = -1
    for r, regex in _RANK_RE.items():
        m = regex.search(text)
        if m:
            found_rank = r
            rank_pos = m.start()
            break
    if not found_rank:
        return None
    # rank 다음 가까운 곳에 suit가 있어야 'rank of suit' 패턴(60자 이내 윈도우)
    tail = text[rank_pos: rank_pos + 60]
    for s, regex in _SUIT_RE.items():
        if regex.search(tail):
            return found_rank + s
    return None


# ---------- 선택 구문 패턴 ----------
_SELECTION_PATTERNS = {
    "select": re.compile(
        r"\b(pick a card|choose a card|select a card|take a card"
        r"|remember your card|memorize your card)\b",
        re.I,
    ),
    # reveal: 'card' 단어가 같이 있는 강한 매칭만. 'this is the' 같은 일반
    # 영어 표현 제거(false positive 원인).
    "reveal": re.compile(
        r"\b(your card is|the card you chose|the card was|here.?s your card"
        r"|this card chose|the card chose|chose the card|only one card"
        r"|the chosen card|the selected card|reveal.{0,15}card"
        r"|card.{0,10}is the|card.{0,10}was the)\b",
        re.I,
    ),
    "audience": re.compile(
        r"\b(spectator|audience|volunteer)\b",
        re.I,
    ),
}


def find_card_mentions(segments) -> list[CardMention]:
    """전사 세그먼트(start, end, text)에서 카드 이름 언급 추출.
    STT misspelling 보정 후 다시 시도해서 회수율 ↑."""
    out: list[CardMention] = []
    for s in segments:
        text = (getattr(s, "text", "") or "").strip()
        if not text:
            continue
        cid = _find_card_in_text(text)
        if not cid:
            # 보정 적용 후 재시도
            fixed = _apply_stt_fixes(text)
            if fixed != text:
                cid = _find_card_in_text(fixed)
        if cid:
            mid = (s.start + s.end) / 2
            out.append(CardMention(time_sec=float(mid), card_id=cid, text=text))
    return out


# ---------- 숫자 / 카운팅 패턴 ----------
# 영어 1~52 단어형 + 숫자형. 일부 자주 등장하는 한국어/영어 합성도 보강.
_WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
}
# "twenty-seven" 처럼 합성 숫자
_COMPOUND_NUM_RE = re.compile(
    r"\b(twenty|thirty|forty|fifty)[\- ]?(one|two|three|four|five|six|seven|eight|nine)\b",
    re.I,
)
# 단일 숫자(단어 또는 1~2자리)
_SINGLE_NUM_WORD_RE = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty)\b",
    re.I,
)
_DIGIT_NUM_RE = re.compile(r"\b([1-9]|[1-4][0-9]|5[0-2])\b")


def _extract_numbers_from_text(text: str) -> list[int]:
    """텍스트에서 1~52 범위 숫자(단어/숫자) 추출. 순서 보존, 중복 OK."""
    nums: list[tuple[int, int]] = []  # (text_pos, value)
    # 합성 숫자(긴 패턴) 먼저
    for m in _COMPOUND_NUM_RE.finditer(text):
        tens = _WORD_NUM[m.group(1).lower()]
        ones = _WORD_NUM[m.group(2).lower()]
        v = tens + ones
        if 1 <= v <= 52:
            nums.append((m.start(), v))
    # 단일 숫자 단어 — 합성 매칭 영역과 안 겹치게
    consumed = set()
    for m in _COMPOUND_NUM_RE.finditer(text):
        for i in range(m.start(), m.end()):
            consumed.add(i)
    for m in _SINGLE_NUM_WORD_RE.finditer(text):
        if m.start() in consumed:
            continue
        v = _WORD_NUM[m.group(1).lower()]
        if 1 <= v <= 52:
            nums.append((m.start(), v))
    # 숫자형(arabic)
    for m in _DIGIT_NUM_RE.finditer(text):
        nums.append((m.start(), int(m.group(1))))
    nums.sort()
    return [v for _, v in nums]


def find_number_mentions(segments) -> list[NumberMention]:
    """전사 세그먼트에서 1~52 범위 숫자 언급 시점 추출.

    ACAAN, '관객이 숫자 선택' 같은 트릭의 핵심 단서.
    """
    out: list[NumberMention] = []
    for s in segments:
        text = (getattr(s, "text", "") or "").strip()
        if not text:
            continue
        # 한 번 보정해 카드명 부근에서 숫자만 끄집어내기 쉽게
        fixed = _apply_stt_fixes(text)
        for v in _extract_numbers_from_text(fixed):
            mid = (s.start + s.end) / 2
            out.append(NumberMention(time_sec=float(mid), number=v, text=text))
    return out


def find_counting_sequences(segments, min_len: int = 3,
                            max_gap_sec: float = 2.0) -> list[CountingSequence]:
    """연속 카운팅(1, 2, 3, ...) 시퀀스 감지. min_len 이상 연속 증가하면 후보.

    - 한 세그먼트 안에서: 한 문장에 '1, 2, 3' 같은 나열
    - 인접 세그먼트 사이: max_gap_sec 안에 '연속 카운트 이어짐'
    """
    out: list[CountingSequence] = []
    # 세그먼트별 추출 숫자 시퀀스 (단, 1~52만)
    per_seg = []  # list of (start, end, text, [nums])
    for s in segments:
        text = (getattr(s, "text", "") or "").strip()
        if not text:
            continue
        fixed = _apply_stt_fixes(text)
        nums = _extract_numbers_from_text(fixed)
        if nums:
            per_seg.append((float(s.start), float(s.end), text, nums))

    # 한 세그먼트 내부 카운팅: 연속 증가하는 부분 추출
    for (st, en, text, nums) in per_seg:
        cur = [nums[0]]
        for n in nums[1:]:
            if n == cur[-1] + 1:
                cur.append(n)
            else:
                if len(cur) >= min_len:
                    out.append(CountingSequence(st, en, list(cur), text))
                cur = [n]
        if len(cur) >= min_len:
            out.append(CountingSequence(st, en, list(cur), text))

    # 인접 세그먼트 이어지는 카운팅(예: "...8, 9, 10." → 다음 세그먼트 "11, 12, 13")
    if len(per_seg) >= 2:
        merged_seq: list[int] = []
        merged_start = None
        merged_end = None
        merged_text_parts: list[str] = []
        prev_end = None
        for (st, en, text, nums) in per_seg:
            if not merged_seq:
                # 단일 시퀀스 내부 연속 증가 시작
                if len(nums) >= 1:
                    merged_seq = [nums[0]]
                    for n in nums[1:]:
                        if n == merged_seq[-1] + 1:
                            merged_seq.append(n)
                        else:
                            merged_seq = [n]
                    merged_start = st
                    merged_end = en
                    merged_text_parts = [text]
                    prev_end = en
                continue
            # 다음 세그먼트에 merged_seq 마지막 다음 숫자가 있으면 이어붙임
            if prev_end is not None and st - prev_end <= max_gap_sec:
                ext_started = False
                for n in nums:
                    if n == merged_seq[-1] + 1:
                        merged_seq.append(n)
                        ext_started = True
                if ext_started:
                    merged_end = en
                    merged_text_parts.append(text)
                    prev_end = en
                    continue
            # 이어지지 않음 — flush
            if len(merged_seq) >= min_len:
                out.append(CountingSequence(
                    merged_start, merged_end, list(merged_seq),
                    " | ".join(merged_text_parts)))
            merged_seq = []
            merged_text_parts = []
        if len(merged_seq) >= min_len:
            out.append(CountingSequence(
                merged_start, merged_end, list(merged_seq),
                " | ".join(merged_text_parts)))

    # 중복 제거(같은 start 시각에 같은 시퀀스)
    seen = set()
    dedup = []
    for c in out:
        key = (round(c.start_sec, 1), tuple(c.numbers))
        if key not in seen:
            seen.add(key)
            dedup.append(c)
    return dedup


def find_selection_phrases(segments) -> list[SelectionPhrase]:
    """관객 선택/카드 reveal/관객 언급 구문 추출."""
    out: list[SelectionPhrase] = []
    for s in segments:
        text = (getattr(s, "text", "") or "").strip()
        if not text:
            continue
        mid = (s.start + s.end) / 2
        for kind, regex in _SELECTION_PATTERNS.items():
            if regex.search(text):
                out.append(SelectionPhrase(time_sec=float(mid),
                                           phrase_kind=kind, text=text))
                break  # 한 세그먼트당 한 카테고리
    return out


# ---------- 전사 실행 + 캐시 ----------
def _ffmpeg() -> str:
    from imageio_ffmpeg import get_ffmpeg_exe
    return get_ffmpeg_exe()


def _extract_audio(video: Path, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / (video.stem + ".wav")
    if out.exists() and out.stat().st_size > 0:
        return out
    subprocess.run(
        [_ffmpeg(), "-i", str(video), "-ar", "16000", "-ac", "1",
         "-c:a", "pcm_s16le", "-y", "-loglevel", "error", str(out)],
        check=True,
    )
    return out


def transcribe_and_extract(video_path: str,
                           model_size: str = "small",
                           audio_cache: Path = Path("downloads/.audio"),
                           transcript_cache: Path = Path("downloads/.transcripts"),
                           ) -> dict | None:
    """영상 전사 후 음성 단서 dict 반환.

    반환: {
        'mentions': list[CardMention],
        'phrases':  list[SelectionPhrase],
        'numbers':  list[NumberMention],      # 1~52 범위 숫자 언급
        'counts':   list[CountingSequence],   # '1,2,3,...' 카운팅 시퀀스
    }
    모델/ffmpeg 미설치면 None. 전사 결과 JSON 캐시 재사용.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None

    video = Path(video_path)
    transcript_cache.mkdir(parents=True, exist_ok=True)
    cache_file = transcript_cache / (video.stem + ".json")

    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            segments = [type("S", (), {"start": s["start"], "end": s["end"],
                                        "text": s["text"]})() for s in data]
        except Exception:
            segments = None
    else:
        segments = None

    if segments is None:
        try:
            audio = _extract_audio(video, audio_cache)
        except Exception as e:
            print(f"      [audio_cues] 오디오 추출 실패: {e}")
            return None
        try:
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            seg_iter, info = model.transcribe(
                str(audio), language="en", vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )
            segments = list(seg_iter)
            cache_file.write_text(
                json.dumps([{"start": s.start, "end": s.end, "text": s.text}
                            for s in segments], ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"      [audio_cues] 전사 {len(segments)}세그먼트, "
                  f"언어 {info.language}")
        except Exception as e:
            print(f"      [audio_cues] 전사 실패: {e}")
            return None

    return {
        "mentions": find_card_mentions(segments),
        "phrases":  find_selection_phrases(segments),
        "numbers":  find_number_mentions(segments),
        "counts":   find_counting_sequences(segments),
    }
