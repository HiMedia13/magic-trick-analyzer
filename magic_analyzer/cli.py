"""명령행 진입점 — 영상 분석 → 리포트/마킹영상/의심프레임 출력."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# MediaPipe/TF 로그 노이즈(XNNPACK/clearcut 텔레메트리 등) 억제.
# 반드시 hands(=mediapipe) 임포트 '전에' 설정해야 효과가 있다.
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("GLOG_logtostderr", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2

from .detect import detect
from .fetch import resolve_video_input
from .hands import HandTracker
from .report import draw_overlay, format_report, to_dict, write_json
from .video import iter_frames, make_writer, open_video


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="magic-analyzer",
        description="카드/동전 마술 영상에서 '수상한 순간'을 찾아주는 분석 도구",
    )
    p.add_argument("video", help="분석할 영상 파일 경로 또는 YouTube URL")
    p.add_argument("--mode", choices=["card", "coin", "auto"], default="card",
                   help="마술 종류. auto면 영상을 보고 카드/동전 자동 판별 (OPENAI_API_KEY 필요)")
    p.add_argument("--out", default="out", help="결과 출력 폴더 (기본: out)")
    p.add_argument("--stride", type=int, default=1,
                   help="N프레임마다 1장만 분석(속도용, 기본 1=전부)")
    p.add_argument("--annotate", action="store_true",
                   help="손 관절+의심구간이 표시된 영상 저장")
    p.add_argument("--save-frames", type=int, default=5, metavar="K",
                   help="상위 K개 의심 구간의 정점 프레임 이미지 저장 (기본 5, 0이면 끔)")
    p.add_argument("--score-thresh", type=float, default=0.9,
                   help="의심 봉우리 점수 임계값 (낮출수록 더 많이 잡음)")
    p.add_argument("--min-gap", type=float, default=1.2,
                   help="의심 순간 사이 최소 간격(초) — 가까운 봉우리는 하나로")
    p.add_argument("--window", type=float, default=0.8,
                   help="봉우리 앞뒤로 구간에 포함할 시간(초)")
    p.add_argument("--max-results", type=int, default=None,
                   help="최대 의심 순간 개수 (기본: 제한 없음)")
    p.add_argument("--llm", action="store_true",
                   help="의심 구간을 OpenAI 비전 모델로 추론 (OPENAI_API_KEY 환경변수 필요)")
    p.add_argument("--llm-model", default="gpt-4o",
                   help="OpenAI 비전 모델명 (기본 gpt-4o)")
    p.add_argument("--llm-top", type=int, default=5,
                   help="LLM으로 추론할 상위 구간 개수 (기본 5)")
    return p


def _load_env() -> None:
    """프로젝트 루트와 현재 디렉토리의 .env를 환경변수로 로드(OPENAI_API_KEY 등).

    LangSmith 추적은 langchain이 임포트/실행되기 '전에' 켜야 하므로, 키가 있으면
    여기서(가장 이른 시점) 추적 환경변수를 설정한다.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        for cand in (Path(__file__).resolve().parent.parent / ".env",
                     Path.cwd() / ".env"):
            if cand.exists():
                load_dotenv(cand)

    if os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"):
        # 양쪽 이름 모두 세팅(버전별 차이 대응)
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", "magic-analyzer")


def _force_utf8() -> None:
    """Windows 콘솔(cp949)에서 한글/이모지 출력이 깨지거나 죽지 않도록 utf-8로 전환."""
    for stream in (sys.stdout, sys.stderr):
        try:
            # line_buffering=True → 줄마다 즉시 flush (로그 파일이 실시간으로 갱신됨)
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8()
    _load_env()
    args = _build_parser().parse_args(argv)
    try:
        video_path = resolve_video_input(args.video)
    except Exception as e:
        print(f"[오류] 입력 처리 실패: {e}", file=sys.stderr)
        return 1
    if not video_path.exists():
        print(f"[오류] 파일이 없습니다: {video_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 0단계: 카드/동전 자동 판별 (--mode auto) ---
    if args.mode == "auto":
        print("[0/3] 카드/동전 자동 판별 중... (영상 프레임 분석)")
        try:
            from .classify import classify_video
            args.mode = classify_video(str(video_path), model=args.llm_model)
            print(f"      → '{args.mode}' 마술로 판별")
        except Exception as e:
            args.mode = "card"
            print(f"      [경고] 자동 판별 실패({e}) → 기본 'card'로 진행",
                  file=sys.stderr)

    # --- 1단계: 손 추적 + 객체(카드/동전) 검출 ---
    print(f"[1/3] 손 추적 + 객체 검출 중... ({video_path.name}, mode={args.mode})")
    # 객체 검출 모델 미리 로드(첫 호출의 다운로드/초기화 비용을 진행률 메시지 전에).
    try:
        from .objects import warmup as obj_warmup, detect_objects
        obj_ready = obj_warmup()
        if obj_ready:
            print("      객체 검출 모델 준비 완료")
        else:
            print("      객체 검출 비활성(YOLO 미사용) — 손 신호만 사용")
    except Exception as e:
        obj_ready = False
        detect_objects = None  # type: ignore
        print(f"      객체 검출 비활성: {e}")
    # YOLO 추론은 프레임당 ~30ms라 매 프레임 돌리면 분 단위 추가됨. 신호는 전이
    # 검출이라 stride=3(약 0.125s)에서도 vanish/appear를 충분히 잡는다.
    obj_stride = 3

    cap, meta = open_video(video_path)
    total = meta.frame_count or 0
    if total:
        print(f"      길이 {meta.duration_sec:.0f}s, 약 {total}프레임 — 진행률 표시")
    frames = []
    objects: list = []  # frames와 같은 길이로 유지(누락 프레임은 None)
    with HandTracker() as tracker:
        for fr in iter_frames(cap, meta.fps, stride=args.stride):
            frames.append(tracker.process(fr.index, fr.time_sec, fr.image))
            if obj_ready and fr.index % obj_stride == 0:
                objects.append(detect_objects(fr.image, fr.index, fr.time_sec,
                                              mode=args.mode))
            else:
                objects.append(None)  # detect.py가 직전 상태로 carry-forward
            if total and fr.index and fr.index % 600 == 0:
                print(f"      ...추적 {fr.index}/{total} ({100 * fr.index / total:.0f}%)")
    cap.release()
    n_card_hits = sum(1 for o in objects if o and o.n_cards > 0)
    n_coin_hits = sum(1 for o in objects if o and o.n_coins > 0)
    print(f"      처리 프레임: {len(frames)}장, fps={meta.fps:.1f}, "
          f"길이={meta.duration_sec:.1f}s")
    if obj_ready:
        print(f"      객체 검출: 카드 hit {n_card_hits}프레임, "
              f"동전 hit {n_coin_hits}프레임 (stride={obj_stride})")

    # --- 2단계: 의심 구간 탐지 ---
    print("[2/3] 의심 구간 분석 중...")
    segments = detect(frames, meta.fps, mode=args.mode,
                      score_thresh=args.score_thresh, min_gap_sec=args.min_gap,
                      window_sec=args.window, max_results=args.max_results,
                      objects=objects)

    report_txt = format_report(segments, meta, args.mode)
    print()
    print(report_txt)

    (out_dir / "report.txt").write_text(report_txt, encoding="utf-8")
    write_json(out_dir / "report.json", to_dict(segments, meta, args.mode))

    # --- 3단계: 영상/프레임 출력 (영상 2차 패스) ---
    llm_segs = segments[:max(args.llm_top, 0)] if args.llm else []
    need_second_pass = args.annotate or args.save_frames > 0
    if need_second_pass and segments:
        print("[3/3] 마킹 영상/프레임 저장 중...")
        obs_by_index = {f.index: f for f in frames}
        peaks = {round(s.peak_sec, 2): s for s in segments[:max(args.save_frames, 0)]}
        cap, meta = open_video(video_path)
        writer = make_writer(out_dir / "annotated.webm", meta) if args.annotate else None
        saved = 0
        try:
            for fr in iter_frames(cap, meta.fps, stride=1):
                obs = obs_by_index.get(fr.index)
                active = next((s for s in segments
                               if s.start_sec <= fr.time_sec <= s.end_sec), None)
                if writer is not None:
                    writer.write(draw_overlay(fr.image, obs, active))
                    if total and fr.index and fr.index % 600 == 0:
                        print(f"      ...영상 인코딩 {fr.index}/{total} "
                              f"({100 * fr.index / total:.0f}%)")
                # 정점 프레임 저장
                if args.save_frames > 0:
                    key = round(fr.time_sec, 2)
                    if key in peaks:
                        s = peaks.pop(key)
                        fn = out_dir / f"suspect_{saved + 1:02d}_{key:.2f}s.jpg"
                        cv2.imwrite(str(fn), draw_overlay(fr.image, obs, s))
                        saved += 1
        finally:
            if writer is not None:
                writer.release()
            cap.release()
        if args.annotate:
            print(f"      → {out_dir / 'annotated.webm'}")
        if saved:
            print(f"      → 의심 프레임 {saved}장 저장")
    else:
        print("[3/3] (저장할 영상/프레임 없음)")

    # --- 4단계: 도구 호출 에이전트 분석 (옵션) ---
    if llm_segs:
        _run_agent(llm_segs, video_path, meta.fps, args, out_dir)

    print(f"\n완료. 결과 폴더: {out_dir.resolve()}")
    return 0


def _fmt_ts(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


def _run_agent(llm_segs, video_path, fps, args, out_dir: Path) -> None:
    """도구 호출 에이전트(ReAct, OpenAI 비전 도구 + LangSmith 추적)로 분석."""
    print(f"[4/4] 도구 호출 에이전트 분석 중... ({args.llm_model}, "
          f"의심 {len(llm_segs)}개 — 에이전트가 직접 inspect)")

    seg_metas = [{"peak_sec": s.peak_sec, "score": s.score,
                  "top_signals": s.top_signals} for s in llm_segs]
    try:
        from .agent import analyze
        result = analyze(str(video_path), fps, seg_metas,
                         mode=args.mode, model=args.llm_model, max_inspect=args.llm_top)
    except Exception as e:
        print(f"      [오류] 에이전트 실행 실패: {e}", file=sys.stderr)
        print("      OPENAI_API_KEY를 설정했는지 확인하세요.", file=sys.stderr)
        return

    analyses = result.get("analyses", [])
    summary = result.get("summary", "")
    techniques = result.get("techniques", [])
    matches = result.get("matches", [])

    lines = ["", "=" * 64, f" 도구 호출 에이전트 분석 (OpenAI {args.llm_model})",
             "=" * 64, "", "[전체 트릭 추정]", summary, ""]
    if techniques:
        lines.append("[사용된 기법 설명]")
        for t in techniques:
            lines += [f"· {t['name_ko']} ({t['name_en']})",
                      f"  {t['desc']}",
                      f"  관찰 단서: {t.get('cues', '')}",
                      f"  참고 영상: {t['reference_url']}", ""]
    lines.append("[에이전트가 들여다본 순간]")
    results = []
    for i, a in enumerate(analyses):
        header = f"[{i + 1}] {_fmt_ts(a['peak_sec'])} (점수 {a.get('score', 0):.2f})"
        hyp = a.get("hypothesis", "")
        print(f"      {header}\n        {hyp}")
        lines += [header, f"  {hyp}", ""]
        results.append({"rank": i + 1, "peak_sec": round(a["peak_sec"], 2),
                        "inference": hyp})
    if matches:
        lines.append("[라이브러리 매칭(데이터 기반 유사도 0~1)]")
        for m in matches:
            top = ", ".join(f"{r['name_ko']} {r['similarity']:.2f}"
                            for r in m["results"])
            lines.append(f"  {_fmt_ts(m['time_sec'])}: {top}")
        lines.append("")
    if techniques:
        print(f"\n      [사용된 기법] " +
              ", ".join(f"{t['name_ko']}" for t in techniques))
    print(f"\n      [전체 트릭 추정]\n        {summary}")

    (out_dir / "llm.txt").write_text("\n".join(lines), encoding="utf-8")
    write_json(out_dir / "llm.json",
               {"model": args.llm_model, "summary": summary,
                "techniques": techniques, "matches": matches, "results": results})
    print(f"      → {out_dir / 'llm.txt'}")


if __name__ == "__main__":
    raise SystemExit(main())
