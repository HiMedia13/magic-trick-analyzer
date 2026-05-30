"""MediaPipe 손 추적 래퍼 (Tasks API) — 프레임마다 손의 위치/펼침정도/관절을 뽑는다.

좌표는 모두 0~1로 정규화된 값(영상 크기와 무관). 그래야 해상도가 달라도
같은 임계값으로 판단할 수 있다.

이 빌드의 MediaPipe는 레거시 `solutions` API가 없어 Tasks API(HandLandmarker)를
사용한다. 모델 파일(hand_landmarker.task)은 assets.ensure_hand_model()이 확보한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from .assets import ensure_hand_model

# MediaPipe 손 관절 인덱스
WRIST = 0
FINGERTIPS = (4, 8, 12, 16, 20)   # 엄지~새끼 끝


@dataclass
class HandObs:
    """한 프레임에서 검출된 손 하나."""
    label: str               # "Left" / "Right"
    center: np.ndarray       # (x, y) 0~1, 손 전체 중심
    bbox: tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax) 0~1
    openness: float          # 손 펼침정도 (작을수록 주먹에 가까움)
    landmarks: np.ndarray    # (21, 2)


@dataclass
class FrameObs:
    """한 프레임의 손 관측 결과."""
    index: int
    time_sec: float
    hands: list[HandObs] = field(default_factory=list)


def _openness(lm: np.ndarray) -> float:
    """손가락 끝이 손목에서 얼마나 멀리 펼쳐졌는지를 손 크기로 정규화."""
    wrist = lm[WRIST]
    tips = lm[list(FINGERTIPS)]
    palm = np.linalg.norm(lm[9] - wrist) + 1e-6  # 손목~중지 뿌리 = 손 크기 기준
    spread = np.mean([np.linalg.norm(t - wrist) for t in tips])
    return float(spread / palm)


class HandTracker:
    def __init__(self, max_hands: int = 4, detection_conf: float = 0.5,
                 tracking_conf: float = 0.5) -> None:
        # max_hands=4 — 마술사 2손 + 관객 손까지 잡기 위함. MediaPipe는 검출된
        # 손 수에 비례해 비용이 들기에 2손만 보이면 추가 비용 없음.
        model_path = ensure_hand_model()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, index: int, time_sec: float, image_bgr) -> FrameObs:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEO 모드는 단조 증가하는 ms 타임스탬프를 요구
        ts = int(time_sec * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts

        result = self._detector.detect_for_video(mp_image, ts)
        obs = FrameObs(index=index, time_sec=time_sec)
        if not result.hand_landmarks:
            return obs

        handedness = result.handedness or []
        for i, hand_lms in enumerate(result.hand_landmarks):
            lm = np.array([[p.x, p.y] for p in hand_lms], dtype=np.float32)
            xmin, ymin = lm.min(axis=0)
            xmax, ymax = lm.max(axis=0)
            bbox = (float(xmin), float(ymin), float(xmax), float(ymax))
            label = "Unknown"
            if i < len(handedness) and handedness[i]:
                label = handedness[i][0].category_name
            obs.hands.append(HandObs(
                label=label,
                center=lm.mean(axis=0),
                bbox=bbox,
                openness=_openness(lm),
                landmarks=lm,
            ))
        return obs

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
