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
from typing import TypedDict

import cv2
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
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
           max_inspect: int = 6, card_timeline=None,
           chosen_evidence=None, audio_cues=None) -> dict:
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
        """관객이 고른 카드를 다중 신호(시각 prominence + 음성 언급 + 관객 손)로
        식별하고 그 카드의 전체 등장 타임라인을 반환. 가설 검증의 기준점."""
        # 1순위: chosen_evidence (D 결합)이 있으면 그쪽 정보 사용
        if chosen_evidence is not None and chosen_evidence.card_id:
            cid = chosen_evidence.card_id
            lines = [f"추정 chosen card = {cid} (신뢰도 {chosen_evidence.confidence})",
                     f"근거: {chosen_evidence.rationale}"]
            if chosen_evidence.selection_event:
                e = chosen_evidence.selection_event
                lines.append(f"  · selection 추정: {e.time_sec:.2f}s (지속 {e.duration:.2f}s)")
            if chosen_evidence.reveal_event:
                e = chosen_evidence.reveal_event
                lines.append(f"  · reveal 추정: {e.time_sec:.2f}s (지속 {e.duration:.2f}s)")
            if chosen_evidence.audio_mention:
                m = chosen_evidence.audio_mention
                lines.append(f"  · 음성 언급 @ {m.time_sec:.2f}s: \"{m.text[:60]}\"")
            if chosen_evidence.audience_hand_event:
                h = chosen_evidence.audience_hand_event
                lines.append(f"  · 관객 손 등장 @ {h.start_sec:.2f}~{h.end_sec:.2f}s "
                             f"(최대 {h.max_n_hands}손)")
            if card_timeline:
                apps = card_timeline.timeline_for(cid)
                if apps:
                    lines.append(f"전체 등장 타임라인 ({len(apps)}회):")
                    for a in apps:
                        lines.append(f"  · {_fmt_app(a)}")
            return "\n".join(lines)
        # 폴백: 단순 휴리스틱
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

    # ----- Phase 4: 명명된 트릭 카탈로그 도구 (tricks.py 기반) -----
    from . import tricks as _tricks_mod

    @tool
    def list_candidate_tricks(effect: str | None = None) -> str:
        """알려진 카드 트릭 목록을 effect 카테고리로 필터링해 반환.

        effect 카테고리: coincidence(우연 일치) / transformation(변화) /
        transposition(위치 교환) / prediction(예측) / ambitious(반복 상승) /
        production(등장) / vanish(사라짐) / restoration(복원) / revelation(드러남)
        / mental(심리). None이면 전체.

        이 도구로 NARRATIVE의 effect와 부합하는 트릭 후보를 좁힌 뒤,
        describe_trick으로 각 후보의 expected beats를 확인하세요.
        """
        items = _tricks_mod.by_effect(effect)
        if not items:
            avail = ", ".join(_tricks_mod.all_effects())
            return f"effect='{effect}'에 해당하는 트릭 없음. 사용 가능: {avail}"
        lines = [f"=== effect={'전체' if not effect else effect}: {len(items)}개 ==="]
        for t in items:
            lines.append(f"  · {t['en']} ({t['ko']}) — {t['desc'][:80]}...")
        return "\n".join(lines)

    @tool
    def describe_trick(trick_name: str) -> str:
        """특정 트릭의 expected beats(시간순)와 사용 기법, 시각 단서를 반환.

        agent는 이 도구로 '만약 이 트릭이라면 이런 beat이 보여야 한다'를 확인한 뒤
        실제 NARRATIVE/SCAN의 관측 beat과 비교해 가설을 검증.
        """
        e = _tricks_mod.lookup(trick_name)
        if not e:
            return f"'{trick_name}' 트릭을 카탈로그에서 찾지 못함. list_candidate_tricks로 확인."
        lines = [
            f"=== {e['en']} ({e['ko']}) | effect={e['effect']} ===",
            f"설명: {e['desc']}",
            "",
            "**Expected beats (이 트릭이라면 보여야 할 시간순 흐름):**",
        ]
        for kind, what in e["beats"]:
            lines.append(f"  - {kind}: {what}")
        lines.append("")
        lines.append(f"**사용 기법**: {', '.join(e['techniques'])}")
        lines.append(f"**시각 단서**: ")
        for cue in e["visual_cues"]:
            lines.append(f"  · {cue}")
        return "\n".join(lines)

    @tool
    def score_trick_match(trick_name: str, observed_beats: str) -> str:
        """이 트릭의 expected beats가 관측된 beat 흐름과 얼마나 부합하는지 분석.

        observed_beats: NARRATIVE 단계에서 추출한 beat 리스트(원문 그대로 넘기면 됨).
        반환: expected vs observed beat 매칭 + 일치/불일치 항목 정리.
        실제 점수 산정은 LLM이 응답으로 판단하도록 데이터만 제공.
        """
        e = _tricks_mod.lookup(trick_name)
        if not e:
            return f"'{trick_name}' 트릭 없음."
        lines = [
            f"=== {e['en']} 매칭 분석 ===",
            f"effect: {e['effect']}",
            "",
            "**이 트릭이라면 보여야 할 beat (expected):**",
        ]
        for kind, what in e["beats"]:
            lines.append(f"  □ {kind}: {what}")
        lines.append("")
        lines.append("**실제 관측 beat (observed, NARRATIVE에서):**")
        for line in observed_beats.split("\n"):
            line = line.strip()
            if line:
                lines.append(f"  ▷ {line}")
        lines.append("")
        lines.append("위 expected vs observed를 비교해 일치/불일치를 판단하세요. "
                     "expected의 핵심 beat(setup/control/reveal 등)이 observed에 있는지가 관건. "
                     f"이 트릭의 시각 단서가 영상에 보이는지도 점검: {'; '.join(e['visual_cues'])}")
        return "\n".join(lines)

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

    # ===== Deep Agent를 LangGraph StateGraph로 구성 =====
    # NARRATIVE → SCAN → HYPOTHESIZE → VERIFY → REVISE → CONCLUDE 6노드.
    # NARRATIVE는 영상 전체를 큰 그림으로 한 번 본 뒤 trick effect와 narrative
    # arc를 추론 — 후속 노드들이 이를 context로 사용해 'spread' false positive 등
    # bottom-up 신호만으론 어려운 판단을 도움.

    mode_ko = MODE_KO.get(mode, mode)

    # NARRATIVE 노드 — 균등 샘플 프레임으로 영상 전체 한 번 분석.
    NARRATIVE_SYSTEM = (
        "당신은 마술 분석 전문가입니다. 균등 간격으로 샘플된 프레임들을 보고 "
        "이 영상의 전체 narrative arc를 추론합니다. 특정 기법을 단정하지 말고, "
        "큰 흐름과 효과(effect)에 집중하세요. 후속 단계에서 세부 분석이 이어집니다.\n\n"
        "**중요**: 영상에 트릭 제목/태그가 텍스트로 표시되어 있더라도 그 텍스트를 "
        "단서로 쓰지 마세요. 손동작·카드 상태·관객 반응 같은 시각적 행위만으로 "
        "트릭을 추론합니다."
    )
    TRICK_HYPOTHESIS_PROMPT = (
        "당신은 명명된 카드 트릭 카탈로그를 보고 후보를 좁히는 분석가입니다.\n\n"
        "**핵심 원칙 1: NARRATIVE의 effect 추정을 의심하라.**\n"
        "NARRATIVE가 'revelation'으로 본 게 사실 'ambitious'(같은 카드 반복 reveal)일 수 "
        "있고, 'transformation'이 사실 'coincidence'(미리 깔린 카드)일 수도 있음. "
        "시각만으론 구별 불가능한 경우 많음.\n\n"
        "**핵심 원칙 2: 무조건 전체 카탈로그를 먼저 본 후 좁혀라.**\n"
        "**필수 첫 단계**: list_candidate_tricks() 를 effect 인자 없이 한 번 호출해서 "
        "전체 112개 카탈로그를 먼저 본 다음, NARRATIVE의 effect 후보로 좁힐 것. "
        "NARRATIVE에 ambitious가 안 들어 있어도 전체 목록에서 같은 카드 반복 reveal "
        "패턴이 보이면 ambitious 후보를 강제로 포함.\n\n"
        "흐름:\n"
        "1) **list_candidate_tricks()** (effect=None) — 전체 카탈로그 한 번 확인 (필수).\n"
        "2) NARRATIVE 효과 후보들로 list_candidate_tricks(effect)도 호출.\n"
        "3) **관측 beat 흐름이 시사하는 트릭 4~6개 선정**:\n"
        "   - **NARRATIVE의 audio_ctx에 1~52 숫자 + 카운팅 시퀀스 → "
        "card_at_any_number(ACAAN) / any_card_at_any_number_classic / "
        "card_at_dealt_number / predicted_position 후보 반드시 포함**.\n"
        "   - 같은 카드 여러 번 reveal → **ambitious_card** 반드시 포함\n"
        "   - 사인 카드 강조 → ambitious / card_to_pocket_signed / 등\n"
        "   - 색 다른 카드 → red_hot_mama / chicago_opener / color_changing_deck\n"
        "   - 두 카드 매칭 → twin_revelation / do_as_i_do\n"
        "   - 카드가 사라짐 → vanish / card_to_pocket / card_to_wallet\n"
        "4) 각 후보 describe_trick(name) + score_trick_match(name, observed_beats).\n"
        "5) 가장 부합하는 1~3개 ranked 후보 출력.\n\n"
        "**중요**: 분기 사고 명시. '만약 X 트릭이라면 [beats]이 관찰돼야 하는데 실제로는 "
        "[...]이 보이므로 부합/불부합' 형식. 단일 단정 X."
    )
    # 영상 총 길이를 비디오 메타에서 정확히 얻기 — segments에는 reveal 부근까지
    # 모두 포함된다는 보장이 없음.
    _cap = cv2.VideoCapture(str(video_path))
    try:
        total_frames = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        duration_sec = total_frames / fps
    finally:
        _cap.release()

    SCAN_PROMPT = (
        f"당신은 {mode_ko} 영상의 **SCAN 단계** 분석가입니다. "
        "list_suspect_moments + track_chosen_card 두 도구를 호출해 "
        "(1) 의심 시점 목록과 (2) 관객이 고른 카드 후보·관련 단서를 한 번에 "
        "파악하세요. 결과를 한국어로 간결히 정리(불릿)해 다음 단계에 넘기세요. "
        "추측·가설은 아직 세우지 마세요 — 사실 수집만."
    )
    HYPOTHESIZE_PROMPT = (
        f"당신은 {mode_ko} 영상의 **HYPOTHESIZE 단계** 분석가입니다. "
        "각 의심 시점에 대해 inspect_moment(time_sec)로 비전 관찰을 하고 "
        "match_technique(time_sec)로 라이브러리 매칭 점수를 확인한 뒤, "
        f"가설(어떤 기법인지)을 한 줄씩 세우세요. inspect_moment는 최대 "
        f"{max_inspect}회. 가설은 '시점 — 후보 기법 — 핵심 근거 한 줄' 형식."
        " 검증은 다음 VERIFY 단계에서 하니까 이 단계에선 가설만."
        f" 카드 영상에선 coin 전용 기법(프렌치 드롭 등)을 후보에서 제외하세요."
    )
    VERIFY_PROMPT = (
        f"당신은 {mode_ko} 영상의 **VERIFY 단계** 분석가입니다. "
        "이전 단계의 가설들을 카드 타임라인 데이터로 검증합니다. "
        "각 가설에 대해 verify_palm_hypothesis(time, card_id?)로 그 시점에 "
        "chosen card가 사라졌는지·얼마나 후 다시 등장하는지 확인하고, "
        "where_did_card_go / card_timeline_for로 보강 단서를 모으세요. "
        "검증 결과: '가설 → 지지(strong/weak) 또는 반박' 형식으로 한 줄씩."
    )
    REVISE_PROMPT = (
        f"당신은 {mode_ko} 영상의 **REVISE 단계** 분석가입니다. "
        "VERIFY에서 반박된 가설을 수정하거나 철회하고, 지지된 가설은 강화하세요. "
        "도구 호출 없이 reasoning만. 최종 정제된 가설 목록을 한 줄씩 출력하세요. "
        "철회된 가설은 별도로 '철회: X (이유)' 로 명시."
    )
    CONCLUDE_PROMPT = (
        f"당신은 {mode_ko} 영상의 **CONCLUDE 단계** 분석가입니다. "
        "정제된 가설을 바탕으로 최종 결론을 작성합니다. 각 채택 기법에 대해 "
        "explain_technique(기법명)을 호출해 작동 원리/관찰 단서/참고 영상을 "
        "확보하세요. 최종 결론은 한국어로 자세히: (1) 이 트릭이 무엇을 "
        "보여주는지, (2) 사용된 각 기법의 작동 원리, (3) 어느 시점·어떤 카드 "
        "이벤트가 근거인지. REVISE에서 철회된 가설도 짧게 언급."
    )

    # 노드별 sub-agent 생성. 각자 자기 도구 subset만 알고 자기 ReAct 루프를 돈다.
    def _llm():
        return ChatOpenAI(model=model, max_tokens=AGENT_MAX_TOKENS)

    trick_hyp_agent = create_react_agent(
        _llm(),
        [list_candidate_tricks, describe_trick, score_trick_match],
        prompt=TRICK_HYPOTHESIS_PROMPT)
    scan_agent = create_react_agent(
        _llm(), [list_suspect_moments, track_chosen_card], prompt=SCAN_PROMPT)
    hyp_agent = create_react_agent(
        _llm(), [inspect_moment, match_technique], prompt=HYPOTHESIZE_PROMPT)
    ver_agent = create_react_agent(
        _llm(), [verify_palm_hypothesis, where_did_card_go, card_timeline_for],
        prompt=VERIFY_PROMPT)
    # REVISE는 도구 없는 순수 추론 — 빈 도구 리스트로 react_agent 만들면 단발 호출
    rev_agent = create_react_agent(_llm(), [], prompt=REVISE_PROMPT)
    conc_agent = create_react_agent(
        _llm(), [explain_technique], prompt=CONCLUDE_PROMPT)

    # ----- State 정의 -----
    class DeepAgentState(TypedDict, total=False):
        narrative: str
        trick_candidates: str
        scan: str
        hypotheses: str
        verifications: str
        revised: str
        conclusion: str

    def _last_ai_text(invoke_result) -> str:
        for m in reversed(invoke_result["messages"]):
            if getattr(m, "type", "") == "ai" and m.content:
                c = m.content
                return c if isinstance(c, str) else str(c)
        return ""

    # ----- 노드 함수 -----
    @traceable(name="deep-narrative", run_type="chain")
    def narrative_node(state: DeepAgentState) -> dict:
        """균등 간격 10프레임을 한 번에 vision LLM에 던져 narrative arc 추론."""
        n_samples = 10
        cap = cv2.VideoCapture(str(video_path))
        frames_at: list[tuple[float, "cv2.typing.MatLike"]] = []
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            # 인덱스로 균등 샘플(처음/끝 제외해서 안정 프레임)
            idxs = [int(total * (i + 0.5) / n_samples) for i in range(n_samples)]
            for i in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, fr = cap.read()
                if ok:
                    frames_at.append((i / fps, fr))
        finally:
            cap.release()
        if not frames_at:
            return {"narrative": "프레임 샘플링 실패 — narrative 생략."}

        # 음성 단서가 있으면 컨텍스트로 추가(전체 전사가 아니라 reveal 구문 정도)
        audio_ctx = ""
        if chosen_evidence is not None:
            if chosen_evidence.audio_mention:
                audio_ctx = (f"\n음성에서 언급된 카드: {chosen_evidence.audio_mention.card_id} "
                             f"(\"{chosen_evidence.audio_mention.text[:80]}\")")
            elif chosen_evidence.card_id:
                audio_ctx = f"\n추정 chosen card(다중 신호): {chosen_evidence.card_id}"
        # 숫자/카운팅 단서 — ACAAN 등 숫자 기반 트릭의 결정적 단서
        if audio_cues:
            nums = audio_cues.get("numbers") or []
            counts = audio_cues.get("counts") or []
            if nums:
                sample = [(round(n.time_sec, 1), n.number) for n in nums[:8]]
                uniq_nums = sorted({n.number for n in nums})
                audio_ctx += (f"\n음성 숫자 언급 {len(nums)}건 — 등장 숫자: "
                              f"{uniq_nums} (시점/숫자 일부: {sample})")
            if counts:
                seq_descs = [f"@{c.start_sec:.1f}s [{','.join(str(n) for n in c.numbers)}]"
                             for c in counts[:3]]
                audio_ctx += (f"\n음성 카운팅 시퀀스 {len(counts)}건: "
                              f"{'; '.join(seq_descs)}")

        content: list[dict] = [{"type": "text", "text": (
            f"이 {mode_ko} 영상의 균등 샘플 {len(frames_at)}프레임 (시각순). "
            f"영상 길이 약 {duration_sec:.1f}초.{audio_ctx}\n\n"
            "다음을 구조화한 한국어 narrative로 답하세요:\n\n"
            "**Effect 후보 (1·2·3순위)**: 단일 단정 X, 2~3개 후보 제시. "
            "**전체 13개 카테고리 모두 검토** — 그 중 부합 후보 선택:\n"
            "  · **ambitious** — 같은 선택 카드가 여러 번 reveal됨(반복 상승). "
            "    제일 흔한 카드 마술 패턴.\n"
            "  · **transformation** — 카드가 다른 카드로 변함\n"
            "  · **transposition** — 두 카드 위치 교환\n"
            "  · **production** — 없던 카드 등장\n"
            "  · **vanish** — 있던 카드 사라짐\n"
            "  · **restoration** — 찢긴/잘린 카드 복원\n"
            "  · **prediction** — 미리 정한 결과 적중\n"
            "  · **coincidence** — 두 카드 우연 일치\n"
            "  · **revelation** — 숨겨진 정보(관객 카드) 드러남\n"
            "  · **sympathy** — 두 묶음 동일 변화\n"
            "  · **mental** — 마음 읽기·예측(mentalism)\n"
            "  · **penetration** — 카드가 물체 관통\n"
            "  · **attraction** — 카드가 천장·이마 등에 부착\n\n"
            "**중요 패턴 단서**:\n"
            "  - **음성에 1~52 숫자 + 카운팅 시퀀스(1,2,3,...) → ACAAN(prediction) "
            "    강력 후보**. 관객이 임의 숫자 호명 → 그 위치 카운트 후 카드 매칭.\n"
            "  - 같은 카드가 영상 중 여러 번 reveal → **ambitious** 강력 후보\n"
            "  - '색 다른 카드' → transformation OR coincidence(미리 깔림) 양쪽\n"
            "  - 사인 카드 강조 + 반복 등장 → ambitious / signed card routine\n"
            "  - 여러 카드 동시 매칭 → coincidence / sympathy\n"
            "  - **'reveal'을 분류할 때 주의**: 카드를 한 장씩 떨어뜨리며 카운트하는 "
            "    동작이면 'count_deal'(ACAAN 단서), 같은 카드가 다시 맨 위에서 나타나면 "
            "    'repeated_reveal'(ambitious 단서). 둘은 시각만으론 비슷해 보이니 "
            "    음성 숫자 단서를 결정적 기준으로 사용.\n"
            "  - 시각만으로 구별 어려운 경우 여러 후보 모두 명시.\n\n"
            "**Narrative beats**: 영상 흐름의 핵심 사건들 (시각순). 형식 "
            "'@<시각>: <사건>'. 가능한 beat 카테고리:\n"
            "  - setup (덱·도구 보여주기 — gimmick 미리 깔기 가능)\n"
            "  - selection (관객이 카드/물건 선택)\n"
            "  - control/manipulation (마술사 비밀 조작)\n"
            "  - vanish/transform 등 핵심 변화\n"
            "  - reveal (결과 공개)\n\n"
            "**시각 단서**: 영상에서 직접 관찰된 특이 요소 (예: '한 카드만 백 색이 "
            "다름', '두 카드 사이에 다른 카드 끼어 있음', '카드 4장만 별도 사용' 등). "
            "이 단서가 트릭 식별에 결정적이므로 자세히.\n\n"
            "특정 시점의 미세 동작은 무시. 큰 흐름·효과가 무엇인지가 목표."
        )}]
        for t, fr in frames_at:
            content.append({"type": "text", "text": f"@ {t:.1f}s:"})
            content.append({"type": "image_url",
                            "image_url": {"url": _data_url(fr)}})

        # 별도 LLM 호출 — sub-agent의 prompt를 거치지 않고 직접 vision
        vision = ChatOpenAI(model=model, max_tokens=1500)
        resp = vision.invoke([SystemMessage(content=NARRATIVE_SYSTEM),
                              HumanMessage(content=content)])
        return {"narrative": (resp.content or "").strip()}

    @traceable(name="deep-trick-hypothesis", run_type="chain")
    def trick_hypothesis_node(state: DeepAgentState) -> dict:
        """NARRATIVE의 effect + beats를 보고 명명된 트릭 카탈로그에서 후보 검색·분기 비교."""
        msg = (f"[NARRATIVE — 영상 전체 흐름]\n{state.get('narrative', '(없음)')}\n\n"
               "위 narrative의 effect를 list_candidate_tricks로 검색해 후보를 좁히고, "
               "각 후보에 describe_trick + score_trick_match로 expected beats vs 실제 "
               "관측 beats를 분기적으로 비교하세요. 단일 트릭 단정 X, ranked 가설을 출력.")
        r = trick_hyp_agent.invoke({"messages": [("user", msg)]},
                                   config={"recursion_limit": 20})
        return {"trick_candidates": _last_ai_text(r)}

    @traceable(name="deep-scan", run_type="chain")
    def scan_node(state: DeepAgentState) -> dict:
        msg = (f"[NARRATIVE 단계 결과 — 영상 전체 흐름]\n"
               f"{state.get('narrative', '(없음)')}\n\n"
               f"[TRICK 후보 가설]\n{state.get('trick_candidates', '(없음)')}\n\n"
               f"위 두 정보를 참고하면서 이 {mode_ko} 영상의 의심 시점과 chosen "
               "card를 파악하세요. 두 도구를 모두 호출한 뒤 결과를 정리해 보고하세요. "
               "narrative beat 시각·trick 후보의 expected beat과 의심 시점이 겹치면 메모.")
        r = scan_agent.invoke({"messages": [("user", msg)]},
                              config={"recursion_limit": 16})
        return {"scan": _last_ai_text(r)}

    @traceable(name="deep-hypothesize", run_type="chain")
    def hypothesize_node(state: DeepAgentState) -> dict:
        msg = (f"[NARRATIVE — 영상 전체 흐름]\n{state.get('narrative', '(없음)')}\n\n"
               f"[TRICK 후보]\n{state.get('trick_candidates', '(없음)')}\n\n"
               f"[SCAN — 의심 시점 + chosen card]\n{state.get('scan', '(없음)')}\n\n"
               "위 정보를 함께 보고 각 의심 시점을 inspect/match해 가설을 세우세요. "
               "TRICK 후보의 expected 기법(예: Red Hot Mama → double_lift/force)에 "
               "부합하는 기법을 우선 고려하세요. trick 후보와 모순되는 기법은 후보에서 "
               f"빼세요. inspect_moment는 {max_inspect}회 한도.")
        r = hyp_agent.invoke({"messages": [("user", msg)]},
                             config={"recursion_limit": 4 * max_inspect + 12})
        return {"hypotheses": _last_ai_text(r)}

    @traceable(name="deep-verify", run_type="chain")
    def verify_node(state: DeepAgentState) -> dict:
        msg = (f"[NARRATIVE]\n{state.get('narrative', '(없음)')}\n\n"
               f"[SCAN]\n{state.get('scan', '(없음)')}\n\n"
               f"[HYPOTHESES]\n{state.get('hypotheses', '(없음)')}\n\n"
               "각 가설을 verify_palm_hypothesis 등으로 검증하세요. narrative beat과 "
               "어긋나는 가설은 의심하세요(예: vanish 효과인데 가설이 단순 위치 이동이면 반박 후보).")
        r = ver_agent.invoke({"messages": [("user", msg)]},
                             config={"recursion_limit": 24})
        return {"verifications": _last_ai_text(r)}

    @traceable(name="deep-revise", run_type="chain")
    def revise_node(state: DeepAgentState) -> dict:
        msg = (f"[NARRATIVE]\n{state.get('narrative', '(없음)')}\n\n"
               f"[HYPOTHESES]\n{state.get('hypotheses', '(없음)')}\n\n"
               f"[VERIFICATIONS]\n{state.get('verifications', '(없음)')}\n\n"
               "반박된 가설은 철회·수정하고, narrative effect 및 verify 결과 모두에 "
               "부합하는 가설은 강화한 최종 가설 목록을 정리하세요.")
        r = rev_agent.invoke({"messages": [("user", msg)]},
                             config={"recursion_limit": 4})
        return {"revised": _last_ai_text(r)}

    @traceable(name="deep-conclude", run_type="chain")
    def conclude_node(state: DeepAgentState) -> dict:
        msg = (f"[NARRATIVE — 큰 흐름]\n{state.get('narrative', '(없음)')}\n\n"
               f"[TRICK 후보]\n{state.get('trick_candidates', '(없음)')}\n\n"
               f"[SCAN — 세부 데이터]\n{state.get('scan', '(없음)')}\n\n"
               f"[REVISED HYPOTHESES — 정제된 가설]\n{state.get('revised', '(없음)')}\n\n"
               "**최종 결론 작성 규칙**:\n"
               "1) TRICK 후보 1순위가 narrative+가설과 부합하면 그 트릭 이름으로 결론 시작 "
               "(예: '이 트릭은 Red Hot Mama입니다').\n"
               "2) 그 트릭의 expected beats를 영상의 실제 시각과 매핑해 설명.\n"
               "3) 채택 기법마다 explain_technique 호출해 작동 원리·참고 영상 확보.\n"
               "4) TRICK 후보 2순위/철회된 가설도 짧게 언급 ('처음엔 Ambitious Card 의심했으나 "
               "X 이유로 철회').\n"
               "5) 한국어, 자세히.")
        r = conc_agent.invoke({"messages": [("user", msg)]},
                              config={"recursion_limit": 16})
        return {"conclusion": _last_ai_text(r)}

    # ----- 그래프 조립 -----
    # NARRATIVE → TRICK_HYPOTHESIS → SCAN → HYPOTHESIZE → VERIFY → REVISE → CONCLUDE
    # TRICK_HYPOTHESIS 노드가 명명된 트릭 카탈로그에서 분기 검증으로 후보를
    # 좁혀 후속 단계들이 일반 카테고리(transformation/coincidence)가 아니라
    # 구체 트릭명(Red Hot Mama / Ambitious Card 등)으로 결론에 도달하게 함.
    g = StateGraph(DeepAgentState)
    g.add_node("NARRATIVE", narrative_node)
    g.add_node("TRICK_HYPOTHESIS", trick_hypothesis_node)
    g.add_node("SCAN", scan_node)
    g.add_node("HYPOTHESIZE", hypothesize_node)
    g.add_node("VERIFY", verify_node)
    g.add_node("REVISE", revise_node)
    g.add_node("CONCLUDE", conclude_node)
    g.add_edge(START, "NARRATIVE")
    g.add_edge("NARRATIVE", "TRICK_HYPOTHESIS")
    g.add_edge("TRICK_HYPOTHESIS", "SCAN")
    g.add_edge("SCAN", "HYPOTHESIZE")
    g.add_edge("HYPOTHESIZE", "VERIFY")
    g.add_edge("VERIFY", "REVISE")
    g.add_edge("REVISE", "CONCLUDE")
    g.add_edge("CONCLUDE", END)

    deep_graph = g.compile()
    final_state = deep_graph.invoke({})
    summary = final_state.get("conclusion", "") or final_state.get("revised", "")

    analyses = sorted(scratch.inspections, key=lambda a: -_score_for(segments, a["peak_sec"]))
    for a in analyses:
        a["score"] = round(_score_for(segments, a["peak_sec"]), 3)
    return {
        "analyses": analyses,
        "summary": summary,
        "techniques": scratch.techniques_found,
        "matches": scratch.matches,
        # 디버그/추적용 단계별 출력
        "deep_stages": {
            "narrative": final_state.get("narrative", ""),
            "trick_candidates": final_state.get("trick_candidates", ""),
            "scan": final_state.get("scan", ""),
            "hypotheses": final_state.get("hypotheses", ""),
            "verifications": final_state.get("verifications", ""),
            "revised": final_state.get("revised", ""),
        },
    }


def _score_for(segments: list[dict], time_sec: float) -> float:
    if not segments:
        return 0.0
    near = min(segments, key=lambda s: abs(s["peak_sec"] - time_sec))
    return float(near.get("score", 0.0)) if abs(near["peak_sec"] - time_sec) < 1.0 else 0.0
