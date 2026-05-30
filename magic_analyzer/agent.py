"""마술 분석 에이전트 (LangGraph ReAct + 도구 호출, + LangSmith 추적).

고정 파이프라인이 아니라 **도구를 호출하는 진짜 에이전트**다. 에이전트(gpt-4o)는
다음 도구를 스스로 호출하며 어디를 들여다볼지 결정한다:

  - list_suspect_moments() : 자동 탐지된 의심 순간(시각/신호/점수) 목록
  - inspect_moment(t)      : 그 시각의 프레임(직전/정점/직후)을 '비전으로' 분석해
                             손 동작 설명을 돌려줌 (영상을 들여다보는 능력)

에이전트는 목록을 보고 → 의심스러운 순간을 골라 inspect → 종합 결론을 낸다.
이미지는 inspect_moment 안의 비전 서브호출에만 들어가므로 메인 트레이스가 가볍다.

LangSmith 추적: 환경변수로 자동 + 루트에 @traceable. OpenAI 사용(Anthropic 아님).
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field

import cv2
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langsmith import traceable

from .detect import SIGNAL_DESC

MODE_KO = {"card": "카드 마술", "coin": "동전 마술"}

# 에이전트(요약 생성)와 비전 서브호출의 응답 길이 한도.
# 요약은 자세한 결론(여러 단락)이 필요해 1500, 비전 한 컷 설명은 2~3문장이라 350.
AGENT_MAX_TOKENS = 1500
VISION_MAX_TOKENS = 350


@dataclass
class AgentScratch:
    """에이전트 도구들이 공유하는 스크래치 상태.

    도구 호출이 누적 결과를 남기려면 어딘가에 모아야 하는데, 클로저로 흘려놓으면
    공유 관계가 암묵적이 된다. 이 데이터클래스로 한곳에 모아 명시적으로 만든다.
    """
    inspections: list[dict] = field(default_factory=list)
    techniques_found: list[dict] = field(default_factory=list)
    matches: list[dict] = field(default_factory=list)

AGENT_SYSTEM = (
    "당신은 클로즈업 마술(카드·동전)의 기법을 분석하는 전문가 에이전트입니다. "
    "교육·복기 목적으로 한 영상의 '비밀 동작'을 추론하고 '기법 자체를 자세히 설명'합니다.\n\n"
    "**핵심 작업 흐름 — 가설→검증→수정(reflect loop)**\n"
    "선형으로 의심 순간을 한 번씩만 보는 게 아니라, 가설을 세우고 다른 시점의 "
    "증거로 검증해 수정하세요.\n\n"
    "단계:\n"
    "1) **SCAN** — list_suspect_moments로 의심 순간 목록 확인. "
    "track_chosen_card로 관객이 고른 카드 후보 + 그 카드의 등장 타임라인 파악.\n"
    "2) **HYPOTHESIZE per moment** — 각 의심 시점에 대해 inspect_moment(t)로 "
    "비전 관찰 + match_technique(t)로 데이터 매칭을 보고 가설을 세웁니다. "
    "예: '클래식 패스로 chosen card를 컨트롤' / '프렌치 드롭으로 카드 vanish' "
    "/ '단순한 deck cut'.\n"
    "3) **VERIFY** — 가설을 데이터로 검증:\n"
    "   - verify_palm_hypothesis(t, card_id): 그 시점에 chosen card가 사라졌는가? "
    "     영구 사라졌는가(팜 강한 증거) 한참 후 다시 등장(produce 가능)?\n"
    "   - where_did_card_go(card_id, t): 그 카드가 다음에 언제 등장하는가?\n"
    "   - card_timeline_for(card_id): 다른 카드 ID의 등장 패턴과 비교.\n"
    "   - explain_technique(기법명): 그 기법의 작동 원리·관찰 단서가 데이터와 부합하는가?\n"
    "4) **REVISE** — 검증 결과가 가설을 반박하면(예: '팜됐다 했는데 1초 후 다시 등장') "
    "가설을 철회/수정하세요. '프렌치 드롭은 동전 기법인데 카드 영상에 출력'처럼 "
    "type 불일치도 단서.\n"
    "5) **CONCLUDE** — 최종 결론(한국어, 자세히):\n"
    "   - 이 마술이 무엇을 보여주는 트릭인지\n"
    "   - 사용된 각 기법이 '어떻게 작동하는지'\n"
    "   - 어느 순간·어떤 손동작·어떤 카드 이벤트가 근거인지 (시각·card_id 명시)\n"
    "   - 검증에서 반박된 가설도 짧게 언급('처음엔 X로 추정했으나 Y 검증에서 철회')\n\n"
    "사용 한도: inspect_moment 최대 6회. 다른 도구는 자유. 같은 시점 inspect 반복 금지.\n"
    "원칙: 확신 없으면 단정 말고 가능성으로, 데이터가 모순되면 '판단 어려움'이라고 하세요. "
    "특히 mode가 'card'면 coin 전용 기법(프렌치 드롭 등)은 후보에서 제외하세요."
)
VISION_SYSTEM = (
    "마술 분석용입니다. 연속 프레임(직전→정점→직후)에서 두 손의 위치/모양 변화를 "
    "관찰하고, 어떤 슬레이트(팜·프렌치드롭·더블리프트·패스·비밀 전달 등)가 일어났을 "
    "법한지 한국어 2~3문장으로 설명하세요. 단정 금지, 가능성으로."
)


def _data_url(image_bgr, max_edge: int = 512, quality: int = 75) -> str:
    h, w = image_bgr.shape[:2]
    scale = max_edge / max(h, w)
    if scale < 1.0:
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("프레임 인코딩 실패")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _maybe_enable_tracing() -> None:
    if os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"):
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", "magic-analyzer")


def _prebuffer_segment_frames(video_path: str, fps: float,
                              segments: list[dict], off: float = 0.4) -> dict:
    """모든 세그먼트의 직전/정점/직후 프레임을 한 번에 디코드해 메모리에 캐싱.

    에이전트 루프 안에서 VideoCapture를 만지면 LLM API 호출 사이 idle 동안 FFmpeg
    다중스레드 디코더가 깨질 수 있다(libavcodec/pthread_frame.c assertion).
    분석 시작 시 한 cap으로 필요 프레임을 모두 디코드해두면 에이전트는 FFmpeg를
    건드리지 않는다 — 1회 open으로 inspect_moment N회 호출을 처리한다.

    반환: {round(peak_sec, 2): [pre, peak, post]} (피크 키는 canonical 2자리 반올림)
    """
    cache: dict[float, list] = {}
    cap = cv2.VideoCapture(str(video_path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        for s in segments:
            ps = float(s["peak_sec"])
            frames = []
            for dt in (-off, 0.0, off):
                idx = max(0, min(total - 1, int(round((ps + dt) * fps))))
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if ok:
                    frames.append(frame)
            cache[round(ps, 2)] = frames
    finally:
        cap.release()
    return cache


def _make_frame_reader(cache: dict, video_path: str, fps: float):
    """캐시된 프레임을 우선 조회하고, 캐시에 없는 시각은 새 cap을 잠깐 열어 폴백.

    _snap_to_peak로 캐시 키에 정확히 매칭되도록 스냅하므로 폴백 경로는 거의
    타지 않는다. 폴백도 한 번 open + 즉시 close라 idle 상태가 없다.
    """
    def read(time_sec: float, off: float = 0.4):
        key = round(float(time_sec), 2)
        if key in cache:
            return cache[key]
        cap = cv2.VideoCapture(str(video_path))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            out = []
            for dt in (-off, 0.0, off):
                idx = max(0, min(total - 1, int(round((time_sec + dt) * fps))))
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if ok:
                    out.append(frame)
            return out
        finally:
            cap.release()
    return read


def _nearest_signals(segments: list[dict], time_sec: float) -> list[str]:
    if not segments:
        return []
    near = min(segments, key=lambda s: abs(s["peak_sec"] - time_sec))
    return near.get("top_signals", []) if abs(near["peak_sec"] - time_sec) < 1.0 else []


def _snap_to_peak(segments: list[dict], time_sec: float, tol: float = 0.6) -> float:
    """에이전트가 넘긴 time_sec을 세그먼트의 canonical peak_sec(소수 둘째 자리)으로
    스냅. 동일 값으로 round-trip되어야 report.peak_sec과 llm.peak_sec이 정확히
    일치한다(이전에는 LLM이 1자리로 반올림해 ±0.6초 근사 매칭이 필요했음)."""
    if not segments:
        return float(time_sec)
    near = min(segments, key=lambda s: abs(s["peak_sec"] - time_sec))
    if abs(near["peak_sec"] - float(time_sec)) <= tol:
        return float(near["peak_sec"])
    return float(time_sec)


@traceable(name="magic-trick-agent", run_type="chain")
def analyze(video_path: str, fps: float, segments: list[dict],
           mode: str = "card", model: str = "gpt-4o",
           max_inspect: int = 6, card_timeline=None) -> dict:
    """비디오 경로 + 탐지된 의심 순간 메타로 ReAct 에이전트를 돌려 분석.

    segments: [{peak_sec, score, top_signals}, ...]
    반환: {"analyses": [{peak_sec, hypothesis, ...}], "summary": <최종 결론>}
    """
    _maybe_enable_tracing()
    if not segments:
        return {"analyses": [], "summary": "분석할 구간이 없습니다."}

    from .library import load_library, match, signature_from_video
    from .techniques import TECHNIQUES

    # 모드(카드/동전)에 맞는 기법만 매칭 후보로 둔다. 안 그러면 카드 영상이 동전
    # 기법에 매칭되는 교차 오매칭이 생긴다(손 궤적만으로는 카드/동전 구분 불가).
    _type_by_en = {e["en"]: e["type"] for e in TECHNIQUES.values()}

    def _mode_ok(tech_en: str) -> bool:
        t = _type_by_en.get(tech_en, "")
        if not t:
            return True            # 종류 미상은 통과(안전)
        if mode == "card":
            return "card" in t     # 'card' + 'coin/card 겸용'
        if mode == "coin":
            return "coin" in t     # 'coin' + 'coin/card 겸용'
        return True

    library = [e for e in load_library() if _mode_ok(e["technique"])]

    # 분석 시작 시 모든 세그먼트 프레임을 한 cap으로 디코드해 메모리에 캐싱.
    # 에이전트 루프(LLM 호출 사이 수초~수십초 idle)는 FFmpeg를 건드리지 않는다.
    frame_cache = _prebuffer_segment_frames(video_path, fps, segments)
    read_frames = _make_frame_reader(frame_cache, video_path, fps)
    scratch = AgentScratch()

    @tool
    def list_suspect_moments() -> str:
        """자동 탐지된 의심 순간 목록을 (시각/신호/점수) 텍스트로 반환한다."""
        lines = []
        for i, s in enumerate(segments):
            sig = ", ".join(s.get("top_signals", [])) or "신호 없음"
            # 소수 둘째 자리(canonical)까지 보여줘 round-trip이 정확하게 일치하도록.
            lines.append(f"{i + 1}. {s['peak_sec']:.2f}s | 점수 {s.get('score', 0):.2f} "
                         f"| 신호: {sig}")
        return "탐지된 의심 순간:\n" + "\n".join(lines)

    @tool
    def inspect_moment(time_sec: float) -> str:
        """주어진 시각(초)의 프레임(직전/정점/직후)을 비전으로 분석해 손 동작 설명을 반환한다."""
        if len(scratch.inspections) >= max_inspect:
            return "검토 한도에 도달했습니다. 지금까지의 관찰로 결론을 작성하세요."
        t = _snap_to_peak(segments, float(time_sec))
        frames = read_frames(t)
        if not frames:
            return f"{t:.2f}s 프레임을 읽지 못했습니다."
        sig = _nearest_signals(segments, t)
        sig_txt = ("자동 탐지 신호: "
                   + ", ".join(f"{s}({SIGNAL_DESC[s].split(' — ')[0]})" for s in sig)) \
            if sig else "자동 탐지 신호: 특이사항 없음"
        labels = ["직전", "정점", "직후"][:len(frames)]
        content = [{"type": "text",
                    "text": f"{MODE_KO.get(mode, mode)} {t:.2f}초 부근. {sig_txt}.\n"
                            f"프레임 순서: {'/'.join(labels)}."}]
        for fr in frames:
            content.append({"type": "image_url", "image_url": {"url": _data_url(fr)}})
        vision = ChatOpenAI(model=model, max_tokens=VISION_MAX_TOKENS)
        resp = vision.invoke([SystemMessage(content=VISION_SYSTEM),
                              HumanMessage(content=content)])
        desc = (resp.content or "").strip()
        scratch.inspections.append({"peak_sec": t, "hypothesis": desc,
                                    "top_signals": sig})
        return desc

    @tool
    def explain_technique(technique: str) -> str:
        """의심되는 마술 기법의 자세한 설명(작동 원리)과 참고 튜토리얼 영상 링크를 반환한다.
        기법명을 한글 또는 영어로 입력한다. 예: '프렌치 드롭', 'double lift'."""
        from .techniques import entry_to_dict, lookup, search_url
        e = lookup(technique)
        if e:
            d = entry_to_dict(e)
            if not any(t["name_en"] == d["name_en"] for t in scratch.techniques_found):
                scratch.techniques_found.append(d)
            return (f"{d['name_ko']} ({d['name_en']}, {d['type']})\n"
                    f"작동 원리: {d['desc']}\n관찰 단서: {d['cues']}\n"
                    f"참고 영상: {d['reference_url']}")
        url = search_url(technique)
        if not any(t["name_en"] == technique for t in scratch.techniques_found):
            scratch.techniques_found.append({"name_ko": technique, "name_en": technique,
                                             "type": "", "desc": "(용어집에 없음 — 일반 지식 기반)",
                                             "cues": "", "reference_url": url})
        return f"'{technique}'은 용어집에 없습니다. 일반 지식으로 설명하고, 참고 영상: {url}"

    @tool
    def match_technique(time_sec: float) -> str:
        """주어진 시각의 손 궤적을 '기법 예시 라이브러리'와 비교해 가장 닮은 기법과
        유사도(0~1)를 반환한다. 비전 관찰과 별개의 데이터 기반 단서."""
        if not library:
            return "기법 예시 라이브러리가 비어 있습니다(등록된 예시 없음)."
        t = _snap_to_peak(segments, float(time_sec))
        sig = signature_from_video(video_path, t)
        if sig is None:
            return f"{t:.2f}s의 손 궤적을 얻지 못했습니다(손 미검출)."
        res = match(sig, library, k=3)
        if not res:
            return "유사한 기법을 찾지 못했습니다."
        scratch.matches.append({"time_sec": t, "results": res})
        return "라이브러리 매칭(유사도 0~1): " + ", ".join(
            f"{r['name_ko']} {r['similarity']:.2f}" for r in res)

    # ----- Phase 2: 가설 검증 도구 (카드 타임라인 기반) -----
    # 카드 타임라인이 없으면 도구들은 '데이터 없음'으로 응답.
    def _fmt_app(a) -> str:
        return f"{a.start_sec:.2f}~{a.end_sec:.2f}s ({a.duration:.2f}s, area={a.max_area:.0f})"

    @tool
    def track_chosen_card() -> str:
        """관객이 고른 것으로 추정되는 카드(face-up으로 가장 오래·크게 보인 카드)와
        그 카드의 전체 등장 타임라인을 반환한다. 가설 검증의 기준점."""
        if card_timeline is None or not card_timeline.appearances:
            return "카드 검출 데이터 없음(객체 검출 비활성 또는 카드 face-up 없음)."
        chosen = card_timeline.chosen_card()
        if not chosen:
            return "추정 가능한 chosen card 없음."
        apps = card_timeline.timeline_for(chosen)
        total = card_timeline.total_visible(chosen)
        lines = [f"추정 chosen card = {chosen} (총 face-up 노출 {total:.2f}s, "
                 f"{len(apps)}회 등장)"]
        for a in apps:
            lines.append(f"  · {_fmt_app(a)}")
        return "\n".join(lines)

    @tool
    def card_timeline_for(card_id: str) -> str:
        """특정 카드 ID(예: '10D', 'KH')의 등장 타임라인을 반환한다."""
        if card_timeline is None:
            return "카드 검출 데이터 없음."
        apps = card_timeline.timeline_for(card_id.strip().upper())
        if not apps:
            return f"{card_id}: 영상에서 face-up으로 검출된 적 없음."
        lines = [f"{card_id}: {len(apps)}회 등장, 총 {sum(a.duration for a in apps):.2f}s"]
        for a in apps:
            lines.append(f"  · {_fmt_app(a)}")
        return "\n".join(lines)

    @tool
    def where_did_card_go(card_id: str, after_time_sec: float) -> str:
        """특정 카드가 주어진 시각 이후 다음에 face-up으로 등장하는 시점을 반환한다.
        없으면 '영상 끝까지 다시 안 보임' — 팜/덱 영구 은닉 가설을 강하게 지지."""
        if card_timeline is None:
            return "카드 검출 데이터 없음."
        cid = card_id.strip().upper()
        nxt = card_timeline.next_after(cid, float(after_time_sec))
        if nxt is None:
            return (f"{cid}: {after_time_sec:.2f}s 이후 영상 끝까지 다시 face-up "
                    f"검출 안 됨(영구 사라짐 — 팜/덱 깊이 가능성).")
        gap = nxt.start_sec - after_time_sec
        return (f"{cid}: {after_time_sec:.2f}s 이후 다음 등장은 "
                f"{_fmt_app(nxt)} (gap {gap:.2f}s).")

    @tool
    def verify_palm_hypothesis(time_sec: float, card_id: str | None = None) -> str:
        """특정 의심 시점에서 '카드가 팜/은닉됐다'는 가설을 데이터로 검증.

        card_id를 지정하면 그 카드, 아니면 chosen_card에 대해:
        - 그 시점 직전엔 보였는가?
        - 그 시점 직후엔 사라졌는가?
        - 그 후 영상에서 다시 등장하는가? (재등장은 팜 가설 약화 — 단순 위치 이동)
        """
        if card_timeline is None:
            return "카드 검출 데이터 없음."
        t = float(time_sec)
        cid = card_id.strip().upper() if card_id else card_timeline.chosen_card()
        if not cid:
            return "검증할 카드 미지정 + chosen card도 없음."
        apps = card_timeline.timeline_for(cid)
        if not apps:
            return f"{cid}: 영상에서 face-up 검출된 적 없어 가설 검증 불가."
        # 직전 등장(보임)
        before = [a for a in apps if a.end_sec <= t]
        # 직후 등장(다시 보임)
        after = [a for a in apps if a.start_sec >= t]
        last_before = before[-1] if before else None
        next_after = after[0] if after else None
        parts = [f"=== {cid} @ {t:.2f}s 가설 검증 ==="]
        if last_before:
            gap_b = t - last_before.end_sec
            parts.append(f"직전 등장: {_fmt_app(last_before)} (시점까지 {gap_b:.2f}s 전)")
        else:
            parts.append("직전 등장: 없음(시점 이전 한 번도 face-up 안 됨)")
        if next_after:
            gap_a = next_after.start_sec - t
            parts.append(f"직후 등장: {_fmt_app(next_after)} (시점에서 {gap_a:.2f}s 후)")
            if gap_a > 5.0:
                parts.append("→ 한참 후 다시 등장 = 일시 은닉 후 reveal 가능(팜+나중에 produce).")
            else:
                parts.append("→ 짧은 간격 재등장 = 팜보다는 단순 위치 이동/회전 가능성.")
        else:
            parts.append("직후 등장: 없음(영상 끝까지 다시 안 보임)")
            parts.append("→ 영구 사라짐 = 팜/덱 깊이 은닉 강한 증거.")
        return "\n".join(parts)

    tools = [list_suspect_moments, inspect_moment, explain_technique,
             match_technique, track_chosen_card, card_timeline_for,
             where_did_card_go, verify_palm_hypothesis]

    agent = create_react_agent(
        ChatOpenAI(model=model, max_tokens=AGENT_MAX_TOKENS),
        tools,
        prompt=AGENT_SYSTEM,
    )
    task = (f"이 {MODE_KO.get(mode, mode)} 영상의 비밀 기법을 분석하세요. "
            f"SCAN→HYPOTHESIZE→VERIFY→REVISE→CONCLUDE 흐름을 따르세요. "
            f"먼저 list_suspect_moments + track_chosen_card로 전체 그림을 파악한 뒤, "
            f"의심 시점마다 inspect_moment/match_technique로 가설을 세우고, "
            f"verify_palm_hypothesis/where_did_card_go로 그 가설을 다른 시점 데이터로 "
            f"검증하세요. 가설이 반박되면 수정한 뒤 최종 결론을 작성하세요.")
    # reflect 루프가 도구를 더 많이 호출하므로 recursion_limit을 넉넉히.
    result = agent.invoke({"messages": [("user", task)]},
                          config={"recursion_limit": 8 * max_inspect + 24})

    summary = ""
    for m in reversed(result["messages"]):
        if getattr(m, "type", "") == "ai" and m.content:
            summary = m.content if isinstance(m.content, str) else str(m.content)
            break

    analyses = sorted(scratch.inspections, key=lambda a: -_score_for(segments, a["peak_sec"]))
    for a in analyses:
        a["score"] = round(_score_for(segments, a["peak_sec"]), 3)
    return {"analyses": analyses, "summary": summary,
            "techniques": scratch.techniques_found, "matches": scratch.matches}


def _score_for(segments: list[dict], time_sec: float) -> float:
    if not segments:
        return 0.0
    near = min(segments, key=lambda s: abs(s["peak_sec"] - time_sec))
    return float(near.get("score", 0.0)) if abs(near["peak_sec"] - time_sec) < 1.0 else 0.0
