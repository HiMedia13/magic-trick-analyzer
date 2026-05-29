"""해설 음성에서 기법명을 추출해 자동으로 라이브러리 라벨링.

핵심 아이디어: '마술 비밀 공개' 류 채널은 해설자가 슬로우모션과 함께 기법 이름을
음성으로 말한다("here's the classic pass", "this is called the double lift").
그 음성 자체가 라벨 데이터이므로 사람 라벨링 없이 자동 누적이 가능하다.

흐름:
  1) 영상 다운로드/해석    (magic_analyzer.fetch.resolve_video_input)
  2) 오디오 추출(WAV 16k)  (imageio-ffmpeg 번들 ffmpeg)
  3) Whisper 전사          (faster-whisper, CPU 최적화)
  4) 기법 언급 검출        (techniques.lookup의 별칭 매칭 재사용, 부정문 제외)
  5) 시그니처 추출 + 등록   (magic_analyzer.library.signature_from_video / save_entry)

사용:
  python scripts/auto_label.py "https://youtu.be/VIDEO_ID"
  python scripts/auto_label.py URL1 URL2 URL3 --model small --offset 0.0

한계:
  - 영어 채널 가정. 한국어/일본어 채널은 별도 모델 필요.
  - 해설자가 시연 전/후로 기법명을 말하면 시점이 어긋남 — --offset으로 조정.
  - 같은 기법을 도입부/마무리에서 언급하면 중복/오라벨 가능 — 라이브러리는
    기법별 max similarity를 쓰므로 노이즈는 일부 자동 흡수됨.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from magic_analyzer.fetch import resolve_video_input  # noqa: E402
from magic_analyzer.library import save_entry, signature_from_video  # noqa: E402
from magic_analyzer.techniques import lookup  # noqa: E402


# 기법 언급이 부정/미래/과거 문맥이면 라벨로 쓰지 않음.
_NEGATION_PAT = re.compile(
    r"\b(not|no|never|without|isn'?t|wasn'?t|aren'?t|won'?t|don'?t|doesn'?t)\b",
    re.IGNORECASE,
)
# 도입/요약 문장은 시연 시점과 어긋날 가능성 큼.
_NON_DEMO_PAT = re.compile(
    r"\b(going to|will show|earlier|previously|today we'll|let me explain"
    r"|now we'll see|coming up|stay tuned)\b",
    re.IGNORECASE,
)


def _ffmpeg() -> str:
    from imageio_ffmpeg import get_ffmpeg_exe
    return get_ffmpeg_exe()


def extract_audio(video: Path, cache_dir: Path) -> Path:
    """mp4 → 16kHz mono PCM WAV. 이미 있으면 재사용."""
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


def transcribe(audio: Path, model_size: str = "small", language: str = "en"):
    """faster-whisper로 세그먼트 단위 전사. 반환: (segments_list, info)."""
    from faster_whisper import WhisperModel

    print(f"      [Whisper] {model_size} 모델 로딩(첫 1회 다운로드 가능)...")
    # int8 quantization → CPU에서 가장 빠르고 메모리 적음
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print(f"      [Whisper] {audio.name} 전사 중...")
    segments, info = model.transcribe(
        str(audio), language=language, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    return list(segments), info


def find_mentions(segments) -> list[dict]:
    """전사 결과에서 기법 언급 후보 추출. 부정/도입부 문장은 제외.

    반환: [{time_sec, en, ko, type, text, reason_dropped?}, ...]
    """
    out: list[dict] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        entry = lookup(text)  # techniques.py의 별칭 매칭이 그대로 작동
        if not entry:
            continue
        if _NEGATION_PAT.search(text):
            out.append({"dropped": "negated", "text": text, "ko": entry["ko"]})
            continue
        if _NON_DEMO_PAT.search(text):
            out.append({"dropped": "intro_or_recap", "text": text,
                        "ko": entry["ko"]})
            continue
        mid = (seg.start + seg.end) / 2
        out.append({
            "time_sec": float(mid),
            "en": entry["en"], "ko": entry["ko"], "type": entry["type"],
            "text": text,
        })
    return out


def auto_label_one(url_or_path: str, model_size: str, offset: float,
                   audio_dir: Path, dry_run: bool) -> int:
    print(f"\n=== {url_or_path} ===")
    print("[1/5] 영상 해석/다운로드")
    video = resolve_video_input(url_or_path)
    print(f"      {video}")

    print("[2/5] 오디오 추출")
    audio = extract_audio(video, audio_dir)

    print("[3/5] 전사")
    segments, info = transcribe(audio, model_size)
    print(f"      언어 {info.language} (conf {info.language_probability:.2f}), "
          f"{len(segments)}세그먼트, 길이 {info.duration:.0f}s")

    print("[4/5] 기법 언급 검출")
    mentions = find_mentions(segments)
    valid = [m for m in mentions if "time_sec" in m]
    dropped = [m for m in mentions if "dropped" in m]
    print(f"      검출 {len(valid)}건, 드롭 {len(dropped)}건"
          f"(부정/도입부)")
    for m in dropped[:5]:
        print(f"      [drop:{m['dropped']}] {m['ko']} ← \"{m['text'][:60]}\"")

    if not valid:
        print("      등록할 후보 없음")
        return 0

    print(f"[5/5] {'(DRY RUN) ' if dry_run else ''}시그니처 추출 + 라이브러리 등록")
    added = 0
    for m in valid:
        t = m["time_sec"] + offset
        if dry_run:
            print(f"      [dry] {m['ko']} ({m['en']}) @ {t:.2f}s "
                  f"← \"{m['text'][:60]}\"")
            continue
        try:
            sig = signature_from_video(str(video), t)
        except Exception as e:
            print(f"      [실패] {m['ko']} @ {t:.2f}s: {e}")
            continue
        if sig is None:
            print(f"      [건너뜀] {m['ko']} @ {t:.2f}s: 손 미검출")
            continue
        save_entry(
            m["en"], m["ko"], sig,
            source=f"stt:{Path(video).stem}@{t:.2f}|{m['text'][:60]}",
        )
        print(f"      [등록] {m['ko']} @ {t:.2f}s")
        added += 1
    return added


def _force_utf8():
    """Windows 콘솔의 cp949에서 한국어 출력이 깨지지 않도록."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main():
    _force_utf8()
    ap = argparse.ArgumentParser(description="음성 자막 기반 자동 라이브러리 라벨링")
    ap.add_argument("videos", nargs="+", help="YouTube URL 또는 로컬 영상 경로")
    ap.add_argument("--model", default="small",
                    choices=["tiny", "base", "small", "medium", "large-v3"],
                    help="Whisper 모델 크기 (기본 small)")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="기법 언급 시점에 더할 초 — 해설자가 시연 전/후로 말하면 조정")
    ap.add_argument("--audio-cache", default="downloads/.audio",
                    help="추출한 WAV 캐시 위치")
    ap.add_argument("--dry-run", action="store_true",
                    help="시그니처 추출/등록 없이 후보만 출력")
    args = ap.parse_args()

    total = 0
    for v in args.videos:
        try:
            total += auto_label_one(v, args.model, args.offset,
                                    Path(args.audio_cache), args.dry_run)
        except Exception as e:
            print(f"\n[오류] {v}: {e}", file=sys.stderr)

    print(f"\n=== 전체 완료: {total}개 라벨 추가 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
