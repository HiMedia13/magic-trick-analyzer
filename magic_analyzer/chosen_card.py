"""관객이 선택한 카드를 다중 신호로 식별하기.

신호 결합:
  A) **시각 prominence** — 카드가 카메라에 face-up + 큰 면적으로 sustained
     노출된 이벤트(CardTimeline.prominence_events). 첫 이벤트=선택, 마지막=reveal.
  B) **음성 cross-check** — 해설/대사에서 카드명 언급(QH, 10D 등) + 선택 구문
     ('your card', 'pick a card')의 시점·내용을 시각과 정렬(audio_cues).
  C) **관객 손 이벤트** — 한 프레임에 손이 2개 초과로 보이는 구간(마술사 외 손
     = 관객 손) — selection moment 강한 단서.

결합 (D): 시각·음성·관객손 합의가 강할수록 confidence 상승.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .audio_cues import CardMention, SelectionPhrase
from .hands import FrameObs
from .objects import CardEvent, CardTimeline


@dataclass
class AudienceHandEvent:
    """한 프레임에 손이 2개 초과로 보인 구간(관객 손 등장 후보)."""
    start_sec: float
    end_sec: float
    max_n_hands: int

    @property
    def mid(self) -> float:
        return (self.start_sec + self.end_sec) / 2


@dataclass
class ChosenCardEvidence:
    card_id: str | None
    confidence: str   # 'high' | 'medium' | 'low' | 'none'
    selection_event: CardEvent | None       # 시각 첫 prominent event
    reveal_event: CardEvent | None          # 시각 마지막 prominent event
    audio_mention: CardMention | None       # 같은 카드를 부른 음성 시점
    audience_hand_event: AudienceHandEvent | None  # 가장 가까운 관객 손 이벤트
    rationale: str    # 사람이 읽는 설명

    def to_dict(self) -> dict:
        def _ev(e):
            if e is None:
                return None
            return {"card_id": getattr(e, "card_id", None),
                    "time_sec": round(e.time_sec, 2),
                    "duration": round(getattr(e, "duration", 0), 2),
                    "max_area": int(getattr(e, "max_area", 0))}
        def _am(m):
            if m is None:
                return None
            return {"time_sec": round(m.time_sec, 2),
                    "card_id": m.card_id,
                    "text": m.text}
        def _ah(h):
            if h is None:
                return None
            return {"start_sec": round(h.start_sec, 2),
                    "end_sec": round(h.end_sec, 2),
                    "max_n_hands": h.max_n_hands}
        return {
            "card_id": self.card_id,
            "confidence": self.confidence,
            "selection_event": _ev(self.selection_event),
            "reveal_event": _ev(self.reveal_event),
            "audio_mention": _am(self.audio_mention),
            "audience_hand_event": _ah(self.audience_hand_event),
            "rationale": self.rationale,
        }


def detect_audience_hand_events(frames: list[FrameObs],
                                min_extra_hands: int = 1,
                                gap_tolerance_sec: float = 0.5,
                                ) -> list[AudienceHandEvent]:
    """프레임에 손이 2개 초과로 보이는 구간을 묶어 이벤트로.

    `min_extra_hands=1` → 3손 이상(=관객 손 1개 등장)부터 후보로 본다.
    `gap_tolerance_sec`: 짧은 검출 누락은 같은 구간으로.
    """
    out: list[AudienceHandEvent] = []
    cur_start = None
    cur_end = None
    cur_max = 0
    for f in frames:
        n = len(f.hands)
        if n >= 2 + min_extra_hands:
            if cur_start is None:
                cur_start = f.time_sec
                cur_end = f.time_sec
                cur_max = n
            else:
                if f.time_sec - cur_end <= gap_tolerance_sec:
                    cur_end = f.time_sec
                    cur_max = max(cur_max, n)
                else:
                    out.append(AudienceHandEvent(cur_start, cur_end, cur_max))
                    cur_start = f.time_sec
                    cur_end = f.time_sec
                    cur_max = n
    if cur_start is not None:
        out.append(AudienceHandEvent(cur_start, cur_end, cur_max))
    return out


def _nearest(events: list, time_sec: float, attr: str = "time_sec",
             max_gap: float = 5.0):
    """events 중 time_sec에 가장 가까운 항목. max_gap 초과면 None."""
    if not events:
        return None
    best = min(events, key=lambda e: abs(getattr(e, attr) - time_sec))
    if abs(getattr(best, attr) - time_sec) <= max_gap:
        return best
    return None


def identify_chosen_card(card_timeline: CardTimeline,
                         frames: list[FrameObs] | None = None,
                         mentions: list[CardMention] | None = None,
                         selection_phrases: list[SelectionPhrase] | None = None,
                         ) -> ChosenCardEvidence:
    """다중 신호로 chosen card 식별 — **우선순위 기반**.

    이전 단순 누적 점수는 '마술사가 자주 보여주는 디스플레이 카드'(예: QH가
    26회 등장)가 '관객이 한 번 선택한 진짜 카드'(예: 2C가 처음 길게 등장 후
    reveal에서 다시 등장)를 이기는 문제가 있었음.

    재설계 의사결정 (먼저 매칭되는 규칙이 이김):
      [0순위] reveal 음성 구문("your card is", "this card chose" 등) ±5초 안에
              prominent 등장 카드 = 관객 카드 (가장 강한 신호 — 마술의 의도된
              reveal 모먼트). 음성 mention까지 일치하면 high, 아니면 medium.
      [1순위] 음성에서 명시적으로 언급된 카드(rank+suit) 중 prominent 등장도
              있는 것. medium.
      [2순위] 관객 손 이벤트 직후(~5초) 첫 prominent 등장 카드. low~medium.
      [폴백]  단순 누적(이전 휴리스틱). low.
    """
    audience_events = detect_audience_hand_events(frames or [])
    ae_first = audience_events[0] if audience_events else None
    events = card_timeline.prominence_events() if card_timeline else []
    mentions = mentions or []
    selection_phrases = selection_phrases or []
    reveal_phrases = [p for p in selection_phrases if p.phrase_kind == "reveal"]

    def _make(card_id, conf, sel_ev, rev_ev, audio_m, ae_m, rationale):
        return ChosenCardEvidence(
            card_id=card_id, confidence=conf,
            selection_event=sel_ev, reveal_event=rev_ev,
            audio_mention=audio_m, audience_hand_event=ae_m,
            rationale=rationale,
        )

    def _card_events_of(cid):
        return [e for e in events if e.card_id == cid]

    # ===== [0순위] reveal 음성 구문 ± 5초 안의 "드문" prominent 카드 =====
    # 핵심 통찰: 관객 카드는 영상 전체에서 적게 등장(selection 1회 + reveal 1회).
    # 마술사 디스플레이 카드(예: QH가 26회 등장)는 자주 보여지므로 reveal 시점에
    # 가장 가까운 게 디스플레이 카드일 수 있음 → '시간 가까움'으로만 뽑으면
    # 진짜 관객 카드를 놓침.
    #
    # 정렬 우선순위:
    #   1) 전체 등장 횟수 적음 (rare = 관객 카드 패턴)
    #   2) 시간 가까움
    #   3) 면적 큼
    # 영상 후반부 reveal 우선(마지막 reveal일수록 진짜 reveal일 가능성 ↑).
    for rp in sorted(reveal_phrases, key=lambda p: -p.time_sec):
        nearby = [e for e in events if abs(e.time_sec - rp.time_sec) <= 5.0]
        if not nearby:
            continue
        nearby.sort(key=lambda e: (
            len(card_timeline.timeline_for(e.card_id)),  # 전체 등장 횟수 적음 우선
            abs(e.time_sec - rp.time_sec),                # 시간 가까움
            -e.max_area,                                   # 면적 큼
        ))
        chosen = nearby[0].card_id
        ce = _card_events_of(chosen)
        sel_ev = ce[0] if ce else None
        rev_ev = ce[-1] if ce else None
        audio_m = next((m for m in mentions if m.card_id == chosen), None)
        conf = "high" if audio_m else "medium"
        total_appearances = len(card_timeline.timeline_for(chosen))
        parts = [
            f"reveal 음성 @ {rp.time_sec:.2f}s ('{rp.text[:60]}') ± 5초 안에 "
            f"{chosen} prominent 등장(@ {nearby[0].time_sec:.2f}s)",
            f"+ {chosen}는 영상 전체 {total_appearances}회 등장(드문 카드 = 관객 카드 패턴)",
        ]
        if audio_m:
            parts.append(f"+ 음성에서 {chosen} 명시 언급 일치 ('{audio_m.text[:50]}')")
        if ae_first:
            parts.append(f"관객 손 이벤트 @ {ae_first.start_sec:.2f}s")
        return _make(chosen, conf, sel_ev, rev_ev, audio_m, ae_first,
                     " | ".join(parts))

    # ===== [1순위] 음성 언급 + prominent 등장 모두 있는 카드 =====
    mention_cards = {m.card_id for m in mentions}
    candidates = [cid for cid in mention_cards if _card_events_of(cid)]
    if candidates:
        # 음성 언급된 카드 중 prominent 노출이 가장 긴 것
        candidates.sort(
            key=lambda c: -sum(e.duration for e in _card_events_of(c)))
        chosen = candidates[0]
        ce = _card_events_of(chosen)
        audio_m = next(m for m in mentions if m.card_id == chosen)
        parts = [f"음성 언급 {chosen} ('{audio_m.text[:60]}') + 시각 "
                 f"{len(ce)}회 prominent 등장"]
        if ae_first:
            parts.append(f"관객 손 이벤트 @ {ae_first.start_sec:.2f}s")
        return _make(chosen, "medium", ce[0], ce[-1], audio_m, ae_first,
                     " | ".join(parts))

    # ===== [2순위] 관객 손 이벤트 직후(5초) 첫 prominent 카드 =====
    if ae_first and events:
        after_ae = [e for e in events if e.time_sec >= ae_first.mid and
                    e.time_sec - ae_first.mid <= 5.0]
        if after_ae:
            after_ae.sort(key=lambda e: e.time_sec)
            chosen = after_ae[0].card_id
            ce = _card_events_of(chosen)
            rationale = (f"관객 손 이벤트 @ {ae_first.start_sec:.2f}s 직후 "
                         f"{after_ae[0].time_sec:.2f}s에 {chosen} prominent 등장")
            return _make(chosen, "low", ce[0], ce[-1], None, ae_first, rationale)

    # ===== [폴백] 단순 누적 휴리스틱 =====
    if card_timeline and card_timeline.appearances:
        chosen = card_timeline.chosen_card()
        ce = _card_events_of(chosen) if chosen else []
        rationale = ("음성/관객손 단서 부족 → 단순 누적(총 노출 × √면적) 휴리스틱. "
                     "마술사가 디스플레이 용도로 자주 보여주는 카드일 수 있어 신뢰도 낮음.")
        return _make(chosen, "low", ce[0] if ce else None,
                     ce[-1] if ce else None, None, ae_first, rationale)

    return _make(None, "none", None, None, None, None, "신호 없음")
