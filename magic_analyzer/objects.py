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
    cards: list[tuple[float, float, float, float, float]]  # [(x1,y1,x2,y2,conf), ...]
    coins: list[tuple[float, float, float, float]]         # [(cx,cy,radius,strength), ...]

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
        return max((x2 - x1) * (y2 - y1) for (x1, y1, x2, y2, _) in self.cards)


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
    """이미지 한 장에서 카드 bbox + confidence 리스트 반환. 모델 없으면 빈 리스트."""
    model = _get_card_model()
    if not model:
        return []
    results = model.predict(image_bgr, conf=conf, verbose=False)
    out = []
    for b in results[0].boxes:
        xyxy = b.xyxy[0].cpu().numpy().tolist()
        c = float(b.conf[0])
        out.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], c))
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
