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
    """다중 신호로 chosen card 식별.

    의사결정:
      1) 시각 prominence_events 추출. 후보 = 거기 등장한 카드들.
      2) 후보별 점수 = (총 prominence 점수)
         + 음성 언급 보너스 (그 카드가 텍스트로 언급되면 + 큰 값)
         + 관객 손 이벤트 근접 보너스 (5초 내에 관객 손 있으면 +)
      3) 최고 점수 카드 선택, 신뢰도는 사용된 신호 수로 결정:
         - 시각 + 음성 + 관객손 → high
         - 시각 + 음성 → medium
         - 시각만 → low
         - 후보 없음 → none
    """
    events = card_timeline.prominence_events() if card_timeline else []
    if not events:
        return ChosenCardEvidence(
            card_id=card_timeline.chosen_card() if card_timeline else None,
            confidence="low" if card_timeline and card_timeline.appearances else "none",
            selection_event=None, reveal_event=None,
            audio_mention=None, audience_hand_event=None,
            rationale="시각 prominent 이벤트가 없어 카드 타임라인의 chosen_card 휴리스틱으로 폴백.",
        )

    # 카드별 prominence 점수 집계
    score_by_card: dict[str, float] = {}
    for e in events:
        score_by_card[e.card_id] = score_by_card.get(e.card_id, 0.0) + e.score

    # 음성 언급 보너스
    mention_set = {m.card_id for m in (mentions or [])}
    for cid in list(score_by_card):
        if cid in mention_set:
            score_by_card[cid] *= 2.5  # 음성과 시각이 합의 = 강력 보너스

    # 관객 손 이벤트 (전체에 1개라도 있으면 시각 첫 이벤트 근처 카드에 보너스)
    audience_events = detect_audience_hand_events(frames or [])
    if audience_events:
        # 가장 이른 관객 손 이벤트 근처의 시각 prominence 카드에 보너스
        first_ae = audience_events[0]
        for e in events:
            if abs(e.time_sec - first_ae.mid) <= 5.0:
                score_by_card[e.card_id] = score_by_card.get(e.card_id, 0) * 1.4

    chosen_id = max(score_by_card, key=score_by_card.get) if score_by_card else None

    # 그 카드의 첫/마지막 시각 이벤트
    card_events = [e for e in events if e.card_id == chosen_id]
    sel_ev = card_events[0] if card_events else None
    rev_ev = card_events[-1] if card_events else None

    audio_match = next((m for m in (mentions or []) if m.card_id == chosen_id),
                      None)
    ae_match = audience_events[0] if audience_events else None

    # 신뢰도 등급
    signals = sum([sel_ev is not None, audio_match is not None, ae_match is not None])
    if signals >= 3:
        conf = "high"
    elif signals == 2:
        conf = "medium"
    elif signals == 1:
        conf = "low"
    else:
        conf = "none"

    parts = []
    if sel_ev:
        parts.append(f"시각 prominent {len(card_events)}회 등장 "
                     f"(첫 {sel_ev.time_sec:.2f}s, 마지막 {rev_ev.time_sec:.2f}s)")
    if audio_match:
        parts.append(f"음성 언급 @ {audio_match.time_sec:.2f}s "
                     f"(\"{audio_match.text[:50]}\")")
    if ae_match:
        parts.append(f"관객 손 등장 @ {ae_match.start_sec:.2f}~{ae_match.end_sec:.2f}s "
                     f"(최대 {ae_match.max_n_hands}손)")
    rationale = " | ".join(parts) if parts else "신호 없음"

    return ChosenCardEvidence(
        card_id=chosen_id,
        confidence=conf,
        selection_event=sel_ev,
        reveal_event=rev_ev,
        audio_mention=audio_match,
        audience_hand_event=ae_match,
        rationale=rationale,
    )
