"""카드/동전 객체 검출 — 손 신호와 독립된 시각 단서.

손 키네마틱 신호(detect.py의 FAST/GRAB/VANISH/CONTACT/BORDER)는 매끄러운 슬레이트를
놓친다. 객체 자체의 등장/소실을 보면 보완 신호가 된다.

- 카드(face-up): YOLOv8 52-class 모델 (mustafakemal0146/playing-cards-yolov8)
  * 한계: 카드 페이스가 보일 때만 검출. 페이스 가려지면 0 → 사실은 이 0/non-zero
    전이 자체가 마술 모먼트(카드 사라짐/등장)의 본질이라 신호로 유용.
- 동전: cv2.HoughCircles (학습 모델 부재) — 원형 검출 휴리스틱
  * 한계: 다른 둥근 물체에 false positive 가능. confidence는 검출된 원의 일관성.

모델은 처음 import 시 lazy하게 받아두고, 이후 호출은 캐시 사용.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np


# ---------- 데이터 타입 ----------
@dataclass
class ObjectObs:
    """한 프레임의 객체 검출 관측."""
    frame_idx: int
    time_sec: float
    # 카드: (x1, y1, x2, y2, conf, card_id) — card_id는 "10D"/"KH"/... 52-class 라벨
    cards: list[tuple[float, float, float, float, float, str]]
    coins: list[tuple[float, float, float, float]]  # [(cx,cy,radius,strength), ...]

    @property
    def n_cards(self) -> int:
        return len(self.cards)

    @property
    def n_coins(self) -> int:
        return len(self.coins)

    @property
    def card_area(self) -> float:
        """가장 큰 카드의 면적(픽셀^2). 카드 없음 = 0."""
        if not self.cards:
            return 0.0
        return max((c[2] - c[0]) * (c[3] - c[1]) for c in self.cards)

    @property
    def card_ids(self) -> list[str]:
        """이 프레임에서 검출된 카드 ID 목록 (예: ['10D', 'KH'])."""
        return [c[5] for c in self.cards]


# ---------- 카드 검출(YOLO) ----------
_HF_REPO = "mustafakemal0146/playing-cards-yolov8"
_HF_FILE = "playing_cards_model_0_playing-cards-colab.pt"
_card_model = None  # lazy singleton


def _get_card_model():
    """첫 호출 시 HF에서 모델 받고 로드. 실패 시 None — 시스템은 코인만 탐지."""
    global _card_model
    if _card_model is not None:
        return _card_model
    try:
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO
    except ImportError as e:
        print(f"      [객체 검출 비활성] ultralytics/huggingface_hub 미설치: {e}")
        _card_model = False
        return None
    try:
        weights = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILE)
        _card_model = YOLO(weights)
    except Exception as e:
        print(f"      [객체 검출 비활성] 모델 다운로드 실패: {e}")
        _card_model = False
        return None
    return _card_model


def detect_cards(image_bgr: np.ndarray, conf: float = 0.25) -> list:
    """이미지 한 장에서 카드 bbox + confidence + card_id 리스트 반환.

    반환: [(x1, y1, x2, y2, conf, card_id), ...] — card_id는 52-class 라벨("10D" 등).
    모델 없으면 빈 리스트.
    """
    model = _get_card_model()
    if not model:
        return []
    results = model.predict(image_bgr, conf=conf, verbose=False)
    out = []
    for b in results[0].boxes:
        xyxy = b.xyxy[0].cpu().numpy().tolist()
        c = float(b.conf[0])
        cls = int(b.cls[0])
        card_id = model.names[cls]
        out.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], c, card_id))
    return out


# ---------- 동전 검출(Hough Circles) ----------
def detect_coins(image_bgr: np.ndarray,
                 min_radius_ratio: float = 0.02,
                 max_radius_ratio: float = 0.10) -> list:
    """이미지에서 동전 같은 원형 영역 검출.

    카메라/거리에 무관하도록 반지름은 영상 짧은 변 기준 상대값으로 한정.
    """
    h, w = image_bgr.shape[:2]
    short = min(h, w)
    min_r = max(8, int(short * min_radius_ratio))
    max_r = max(min_r + 4, int(short * max_radius_ratio))

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_r * 2,
        param1=120, param2=40, minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        return []
    circles = np.round(circles[0]).astype(int)
    # HoughCircles는 confidence를 안 주므로 일단 1.0 고정(필요 시 누적 가산 시 가중)
    return [(float(cx), float(cy), float(r), 1.0) for (cx, cy, r) in circles]


# ---------- 통합 ----------
def detect_objects(image_bgr: np.ndarray, frame_idx: int, time_sec: float,
                   mode: str = "card") -> ObjectObs:
    """프레임 1장 → ObjectObs. mode에 따라 카드/동전 중 관련 검출만."""
    cards = detect_cards(image_bgr) if mode in ("card", "auto") else []
    coins = detect_coins(image_bgr) if mode in ("coin", "auto") else []
    return ObjectObs(frame_idx=frame_idx, time_sec=time_sec, cards=cards, coins=coins)


def warmup() -> bool:
    """파이프라인 시작 전 모델 다운로드/로드를 미리 트리거. True = 카드 모델 사용 가능."""
    m = _get_card_model()
    return bool(m)


# ---------- 카드 타임라인 (per-card tracking over time) ----------
@dataclass
class CardAppearance:
    """한 카드 ID가 연속적으로 보인 한 구간."""
    card_id: str        # "10D", "KH" 등
    start_sec: float
    end_sec: float
    max_area: float     # 이 구간 동안 가장 컸던 bbox 면적
    n_frames: int       # 이 구간에서 검출된 프레임 수

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


class CardTimeline:
    """영상 전체의 per-card 등장 타임라인 + 쿼리.

    의도:
      - "관객이 고른 카드"를 추적: chosen_card()
      - 카드별 등장/소실 이력으로 가설 검증("이 카드가 그 후에 다시 보였나?")
      - YOLO가 못 보는 시점(덱·팜)에 대해 '있다 → 없다 → 영영 없다' 같은
        가설을 데이터로 뒷받침
    """

    def __init__(self, appearances: list[CardAppearance], duration_sec: float):
        self.appearances = sorted(appearances, key=lambda a: a.start_sec)
        self.duration_sec = duration_sec

    @property
    def card_ids(self) -> list[str]:
        """검출된 unique 카드 ID 목록."""
        seen = set()
        out = []
        for a in self.appearances:
            if a.card_id not in seen:
                seen.add(a.card_id)
                out.append(a.card_id)
        return out

    def timeline_for(self, card_id: str) -> list[CardAppearance]:
        """특정 카드의 모든 등장 구간."""
        return [a for a in self.appearances if a.card_id == card_id]

    def total_visible(self, card_id: str) -> float:
        """이 카드의 총 화면 노출 시간(초)."""
        return sum(a.duration for a in self.timeline_for(card_id))

    def max_area_for(self, card_id: str) -> float:
        apps = self.timeline_for(card_id)
        return max((a.max_area for a in apps), default=0.0)

    def chosen_card(self) -> str | None:
        """관객 카드 휴리스틱: 총 노출 시간 × 최대 면적이 가장 큰 카드.

        '관객이 골랐다 = 마술사가 카메라로 길게/크게 보여준다' 가정.
        후보 없으면 None.
        """
        if not self.appearances:
            return None
        scored = [
            (cid, self.total_visible(cid) * (self.max_area_for(cid) ** 0.5))
            for cid in self.card_ids
        ]
        scored.sort(key=lambda x: -x[1])
        return scored[0][0] if scored else None

    def next_after(self, card_id: str, time_sec: float) -> CardAppearance | None:
        """time_sec 이후 이 카드가 다음에 등장하는 구간(없으면 None)."""
        for a in self.timeline_for(card_id):
            if a.start_sec >= time_sec:
                return a
        return None

    def visible_at(self, time_sec: float) -> list[str]:
        """time_sec 시점에 화면에 보이는 카드 ID 목록."""
        return [a.card_id for a in self.appearances
                if a.start_sec <= time_sec <= a.end_sec]

    def prominence_events(self, min_duration: float = 0.4,
                          min_area: float = 600.0) -> list["CardEvent"]:
        """카드가 prominent하게 보인 이벤트들(시간순). selection·reveal 후보.

        한 등장 구간이 `min_duration`초 이상 지속되고 `min_area` 이상 면적이면
        prominent로 간주. min_area=600은 360p 영상 기준 카드 한 장이 화면에
        뚜렷이 보이는 기준선(실측치).
        """
        events: list[CardEvent] = []
        for a in self.appearances:
            if a.duration < min_duration or a.max_area < min_area:
                continue
            mid = (a.start_sec + a.end_sec) / 2
            score = a.duration * (a.max_area ** 0.5)
            events.append(CardEvent(card_id=a.card_id, time_sec=mid,
                                    duration=a.duration, max_area=a.max_area,
                                    score=score))
        events.sort(key=lambda e: e.time_sec)
        return events

    def disappeared_around(self, time_sec: float, window: float = 1.0) -> list[str]:
        """time_sec 직전엔 보였지만 직후엔 안 보이는 카드들 ('사라짐' 후보)."""
        before = set()
        after = set()
        for a in self.appearances:
            if a.start_sec <= time_sec - 0.01 and a.end_sec >= time_sec - window:
                before.add(a.card_id)
            if a.end_sec >= time_sec + 0.01 and a.start_sec <= time_sec + window:
                after.add(a.card_id)
        return sorted(before - after)


@dataclass
class CardEvent:
    """카드가 카메라에 prominent하게 보인 한 이벤트(selection/reveal 후보).

    'prominent' = 일정 시간(sustained) 동안 큰 면적으로 face-up 노출.
    영상 진행 순서로 첫 prominent 이벤트 = 선택 후보, 마지막 = reveal 후보.
    """
    card_id: str
    time_sec: float       # 이 이벤트의 중심 시각
    duration: float       # 지속 시간
    max_area: float       # 이 이벤트 동안 최대 bbox 면적
    score: float          # duration * sqrt(area) — 비교용 점수


def build_card_timeline(objects_list: list, duration_sec: float,
                        gap_tolerance_sec: float = 0.3) -> CardTimeline:
    """프레임별 ObjectObs 리스트 → CardTimeline.

    같은 카드 ID의 연속 검출을 한 등장 구간으로 묶는다. 짧은 검출 누락
    (`gap_tolerance_sec` 이하)은 같은 구간으로 본다(YOLO가 한두 프레임
    놓치는 건 무시).
    """
    # card_id → [(time_sec, area), ...] (정렬됨)
    per_card: dict[str, list[tuple[float, float]]] = {}
    for obs in objects_list:
        if obs is None:
            continue
        for c in obs.cards:
            x1, y1, x2, y2 = c[0], c[1], c[2], c[3]
            card_id = c[5]
            area = (x2 - x1) * (y2 - y1)
            per_card.setdefault(card_id, []).append((obs.time_sec, area))

    appearances: list[CardAppearance] = []
    for card_id, samples in per_card.items():
        samples.sort()
        cur_start = samples[0][0]
        cur_end = samples[0][0]
        cur_max = samples[0][1]
        cur_n = 1
        for t, a in samples[1:]:
            if t - cur_end <= gap_tolerance_sec:
                cur_end = t
                cur_max = max(cur_max, a)
                cur_n += 1
            else:
                appearances.append(CardAppearance(card_id, cur_start, cur_end,
                                                  cur_max, cur_n))
                cur_start, cur_end, cur_max, cur_n = t, t, a, 1
        appearances.append(CardAppearance(card_id, cur_start, cur_end,
                                          cur_max, cur_n))

    return CardTimeline(appearances=appearances, duration_sec=duration_sec)
