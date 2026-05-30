"""명명된 카드 트릭(루틴) 카탈로그 — 슬레이트 단위가 아니라 완성된 트릭 단위.

`techniques.py`는 패스/팜/더블리프트 같은 개별 슬레이트 사전이고, 이 파일은
'레드 핫 마마', 'Ambitious Card' 같은 **루틴(완성 트릭)** 사전이다. 각 항목은
관객 시각에서 본 effect, 시간순 expected beats, 사용 기법 시퀀스, 시각 단서를
정의해서 agent가 다음과 같이 분기 추론할 수 있게 한다:

  '이 영상이 만약 Red Hot Mama라면 [setup → selection → cut → spread reveal →
   double lift → force] beat이 보여야 한다. 실제로 관측된 beat과 비교해보면...'

이 카탈로그가 비면 시스템은 항상 'transformation' 같은 일반 카테고리로만
일반화돼 명명된 트릭을 못 잡는다(Red Hot Mama → 'transformation' 오인).
"""

from __future__ import annotations

import re


# Effect 카테고리 (큰 분류):
#   coincidence   — 두 카드(또는 카드와 다른 것)가 우연히 일치
#   transposition — 두 카드의 위치 교환
#   transformation— 한 카드가 다른 카드로 변함
#   prediction    — 미리 정한 결과가 적중
#   ambitious     — 한 카드가 반복적으로 맨 위로 올라옴
#   production    — 없던 것이 나타남
#   vanish        — 있던 것이 사라짐
#   restoration   — 찢긴/잘린 것이 회복
#   sympathy      — 두 묶음이 같은 변화를 보임
#   revelation    — 숨겨진 정보 드러남 (관객 카드 찾기)
#   mental        — 마음 읽기 / 예측


TRICKS = {
    # ===== Coincidence / Color match =====
    "red_hot_mama": {
        "ko": "레드 핫 마마",
        "en": "Red Hot Mama",
        "aliases": ["red hot mama", "color match", "back color match",
                    "color coincidence"],
        "effect": "coincidence",
        "desc": (
            "맨 밑에 색 다른 백(back) 카드를 미리 깔고, 관객이 카드를 선택해 "
            "맨 위에 놓는다. 덱을 반으로 cut(색 다른 카드와 선택 카드가 인접). "
            "덱을 펼치면 색 다른 카드 한 장이 보이고, 마술사는 '이게 너의 카드'라며 "
            "더블리프트로 선택 카드를 reveal. 이어서 force로 같은 카드를 또 뽑게 해 "
            "2단 reveal."
        ),
        "beats": [
            ("setup", "맨 밑에 색 다른 백 카드 배치(관객엔 안 보임)"),
            ("selection", "관객이 카드 선택"),
            ("control", "선택 카드를 맨 위로, 덱 반으로 cut"),
            ("spread_reveal", "덱 펼침 → 색 다른 카드 1장 발견"),
            ("double_lift_reveal", "더블리프트로 그게 선택 카드임을 보여줌"),
            ("force", "색 다른 카드와 같은 카드를 또 뽑게 하는 force"),
        ],
        "techniques": ["setup_gimmick", "double_lift", "card_force"],
        "visual_cues": [
            "맨 밑 카드를 보여주지 않음 (의도적 회피)",
            "덱을 펼쳤을 때 한 장만 백 색이 다름",
            "한 장처럼 두 장 뒤집기(더블리프트) 동작",
            "덱 한가운데에서 카드 멈춤(force 동작)",
        ],
        "query": "red hot mama card trick tutorial",
    },

    # ===== Ambitious Card =====
    "ambitious_card": {
        "ko": "앰비셔스 카드",
        "en": "Ambitious Card",
        "aliases": ["ambitious card", "card to top", "rising card to top"],
        "effect": "ambitious",
        "desc": (
            "관객이 선택한 카드를 덱 중간에 넣어도 계속 맨 위로 올라오는 트릭. "
            "보통 3~5회 반복되며, 각 회마다 다른 카드 컨트롤 기법(패스, 더블리프트, "
            "팜 등)을 사용해 같은 효과를 다른 방식으로 보여준다."
        ),
        "beats": [
            ("selection", "관객 카드 선택"),
            ("insert", "선택 카드를 덱 중간에 삽입"),
            ("control", "비밀리에 맨 위로 컨트롤 (패스/팜/더블리프트)"),
            ("reveal", "맨 위에서 선택 카드 reveal"),
            ("repeat", "위 control→reveal을 3~5회 반복"),
        ],
        "techniques": ["classic_pass", "double_lift", "tilt", "side_steal",
                       "card_palm"],
        "visual_cues": [
            "같은 카드를 여러 번 reveal",
            "덱 중간에 카드 삽입 동작 반복",
            "맨 위 카드 뒤집기 반복",
        ],
        "query": "ambitious card routine tutorial",
    },

    # ===== Triumph =====
    "triumph": {
        "ko": "트라이엄프",
        "en": "Triumph",
        "aliases": ["triumph", "vernon triumph"],
        "effect": "restoration",
        "desc": (
            "관객 카드를 덱 중간에 넣은 뒤, 덱의 절반을 face-up으로 뒤집어 "
            "반대편 face-down 절반과 섞는다. 펼쳤을 때 face up/down이 무작위로 "
            "섞여 있지만, 마술사가 손짓 한 번에 모든 카드가 한 방향으로 정렬되고 "
            "한 장만 반대 방향 = 관객 카드."
        ),
        "beats": [
            ("selection", "관객 카드 선택, 덱 중간에 삽입"),
            ("triumph_shuffle", "절반을 face-up으로 뒤집어 face-down과 섞음"),
            ("show_chaos", "펼쳐 face up/down 혼란 보여줌"),
            ("magic_moment", "손짓 + 덱 정렬"),
            ("reveal", "한 장만 반대 = 관객 카드"),
        ],
        "techniques": ["false_shuffle", "triumph_shuffle", "double_undercut"],
        "visual_cues": [
            "덱의 절반을 뒤집는 동작",
            "face-up과 face-down 카드가 섞인 상태로 펼침",
            "최종 펼침에 한 장만 반대 방향",
        ],
        "query": "triumph card trick tutorial vernon",
    },

    # ===== Out of This World =====
    "out_of_this_world": {
        "ko": "아웃 오브 디스 월드 (OOTW)",
        "en": "Out of This World",
        "aliases": ["out of this world", "ootw", "red black separation"],
        "effect": "prediction",
        "desc": (
            "관객이 덱을 한 장씩 face-down으로 두 묶음으로 나누면(빨강이라고 "
            "생각하는 것 vs 검정이라고 생각하는 것), 펼쳤을 때 정말로 빨강과 "
            "검정이 완벽 분리. 핵심: 미리 분류된 덱 + 가짜 mixing + indicator 카드."
        ),
        "beats": [
            ("setup", "덱을 빨강·검정으로 미리 분류"),
            ("audience_split", "관객이 직관으로 두 묶음에 한 장씩 분배"),
            ("color_switch", "중간에 indicator를 교체해 색 분류 보존"),
            ("reveal", "두 묶음을 뒤집어 완벽한 색 분리 보여줌"),
        ],
        "techniques": ["setup_stack", "color_indicator", "false_shuffle"],
        "visual_cues": [
            "덱을 두 묶음으로 분류하는 긴 과정",
            "두 indicator 카드(빨강·검정 대표) 사용",
            "최종 뒤집기 후 빨강·검정 완전 분리",
        ],
        "query": "out of this world card trick tutorial",
    },

    # ===== Card to Pocket / Wallet =====
    "card_to_pocket": {
        "ko": "카드 투 포켓",
        "en": "Card to Pocket",
        "aliases": ["card to pocket", "card in pocket"],
        "effect": "transposition",
        "desc": (
            "관객 카드가 덱에서 사라져 마술사 주머니/지갑에서 나타남. 핵심: "
            "팜으로 카드를 손에 숨긴 뒤 자연스러운 동작 중 주머니로 load."
        ),
        "beats": [
            ("selection", "관객 카드 선택"),
            ("palm", "카드를 손바닥에 숨김"),
            ("load", "주머니/지갑으로 load"),
            ("vanish_show", "덱에서 카드 사라짐 보여줌"),
            ("produce", "주머니에서 카드 produce"),
        ],
        "techniques": ["card_palm", "topit", "side_steal"],
        "visual_cues": [
            "손이 주머니로 향함",
            "덱에서 선택 카드 못 찾음",
            "주머니에서 카드 빼냄",
        ],
        "query": "card to pocket tutorial",
    },

    # ===== Sandwich =====
    "sandwich": {
        "ko": "샌드위치",
        "en": "Sandwich Effect",
        "aliases": ["sandwich", "sandwich effect", "card sandwich"],
        "effect": "production",
        "desc": (
            "두 장의 같은 카드(예: 검정 잭 둘) 사이에 관객 카드가 나타남. "
            "샌드위치 카드들을 따로 빼두고, 덱을 섞은 뒤 샌드위치를 펼치면 "
            "관객 카드가 그 사이에 끼어 있음."
        ),
        "beats": [
            ("selection", "관객 카드 선택"),
            ("sandwich_setup", "두 장의 같은 카드(jacks/kings)를 따로 빼서 보여줌"),
            ("magic_moment", "샌드위치를 펼치거나 손짓"),
            ("reveal", "두 카드 사이에 관객 카드 끼어 있음"),
        ],
        "techniques": ["double_lift", "card_palm", "side_steal", "spread_cull"],
        "visual_cues": [
            "두 장의 같은 카드(jacks 등) 분리",
            "마지막에 그 두 카드 사이에 다른 카드 끼어 있음",
        ],
        "query": "sandwich card trick tutorial jacks",
    },

    # ===== Color Changing Deck =====
    "color_changing_deck": {
        "ko": "컬러 체인징 덱",
        "en": "Color Changing Deck",
        "aliases": ["color changing deck", "deck switch",
                    "deck color change"],
        "effect": "transformation",
        "desc": (
            "전체 덱의 백 색이 일순간에 다른 색으로 변함. 보통 손짓 한 번에 "
            "전체 덱이 빨강 → 파랑 등으로 변화. 미리 준비된 다른 색 덱과의 switch."
        ),
        "beats": [
            ("show_deck", "한 가지 색 덱 보여줌(예: 빨강 백)"),
            ("magic_moment", "손짓·동작"),
            ("reveal", "전체 덱 색이 변했음 reveal"),
        ],
        "techniques": ["deck_switch"],
        "visual_cues": [
            "덱 전체 백 색이 영상 중 한 시점에 바뀜",
            "마술사 핸드/주머니로 가는 동작 (switch)",
        ],
        "query": "color changing deck tutorial",
    },

    # ===== Anniversary Waltz / 2 cards become 1 =====
    "anniversary_waltz": {
        "ko": "애니버서리 왈츠 (두 카드가 한 카드로)",
        "en": "Anniversary Waltz",
        "aliases": ["anniversary waltz", "two cards become one",
                    "fused cards"],
        "effect": "transformation",
        "desc": (
            "관객 두 명이 각자 카드 한 장씩 선택. 마술사가 손짓 한 번에 두 카드가 "
            "물리적으로 결합돼 한 카드(두 카드의 face가 한 카드의 양면에)가 됨. "
            "보통 gimmick 카드 + glue · double-face 사용."
        ),
        "beats": [
            ("two_selections", "두 관객이 각각 카드 선택"),
            ("magic_moment", "손짓·결합 시퀀스"),
            ("reveal", "두 카드가 한 카드로 합쳐졌음 reveal (양면)"),
        ],
        "techniques": ["gimmick_card", "card_switch"],
        "visual_cues": [
            "두 관객이 각각 카드 선택",
            "최종 reveal에 두 카드의 face가 한 카드의 양면",
        ],
        "query": "anniversary waltz card trick tutorial",
    },

    # ===== Do as I Do =====
    "do_as_i_do": {
        "ko": "두 애즈 아이 두",
        "en": "Do as I Do",
        "aliases": ["do as i do", "twin selection"],
        "effect": "coincidence",
        "desc": (
            "관객과 마술사가 각자 덱으로 같은 동작을 따라하다 카드 한 장씩 선택. "
            "두 카드가 같은 카드로 일치. 핵심: 마술사가 force로 자기 카드를 "
            "관객 카드와 같게 만듦."
        ),
        "beats": [
            ("mirror_actions", "관객과 마술사 같은 동작 따라하기"),
            ("two_selections", "각자 카드 선택"),
            ("reveal", "두 카드가 같은 카드로 일치"),
        ],
        "techniques": ["card_force", "false_shuffle"],
        "visual_cues": [
            "두 명이 각자 덱 다룸",
            "최종 reveal에 두 카드 동일",
        ],
        "query": "do as i do card trick tutorial",
    },

    # ===== Twisting the Aces =====
    "twisting_aces": {
        "ko": "트위스팅 더 에이스",
        "en": "Twisting the Aces",
        "aliases": ["twisting the aces", "twisting"],
        "effect": "transformation",
        "desc": (
            "에이스 4장을 face-down으로 보여주고 손짓할 때마다 한 장씩 face-up이 됨. "
            "최종적으로 모두 face-up. Elmsley count, Jordan count 같은 packet count "
            "기법으로 카드 상태를 속임."
        ),
        "beats": [
            ("show_four_aces", "에이스 4장 face-down 보여줌"),
            ("twist_1", "손짓 → 첫 번째 에이스 face-up"),
            ("twist_2", "두 번째 에이스 face-up"),
            ("twist_3", "세 번째 에이스 face-up"),
            ("final_reveal", "네 번째도 face-up — 모두 face-up"),
        ],
        "techniques": ["elmsley_count", "jordan_count", "packet_count"],
        "visual_cues": [
            "4장 묶음만 사용 (전체 덱 X)",
            "카드 세는 동작 반복",
            "매번 한 장씩 face-up 증가",
        ],
        "query": "twisting the aces dai vernon tutorial",
    },

    # ===== Card Stab =====
    "card_stab": {
        "ko": "카드 스탭",
        "en": "Card Stab",
        "aliases": ["card stab", "knife stab", "stab routine"],
        "effect": "revelation",
        "desc": (
            "관객 카드를 덱에 섞은 뒤 마술사가 칼/연필로 덱을 stab. 뽑힌 카드가 "
            "관객 카드. 핵심: 카드를 미리 컨트롤된 위치에 두고 그 위치를 칼로 stab."
        ),
        "beats": [
            ("selection", "관객 카드 선택, 덱에 섞음"),
            ("control", "비밀리에 카드를 알려진 위치로 컨트롤"),
            ("stab", "칼/연필로 덱을 stab"),
            ("reveal", "stab 지점의 카드가 관객 카드"),
        ],
        "techniques": ["card_control", "key_card", "marked_card"],
        "visual_cues": [
            "칼/연필 등 도구 사용",
            "덱에 도구를 꽂는 극적 동작",
        ],
        "query": "card stab trick tutorial",
    },

    # ===== Invisible Deck =====
    "invisible_deck": {
        "ko": "인비저블 덱",
        "en": "Invisible Deck",
        "aliases": ["invisible deck", "ultra mental deck"],
        "effect": "prediction",
        "desc": (
            "관객이 '상상의 덱'에서 카드를 고른 뒤, 마술사가 실제 덱을 펼치면 "
            "관객이 말한 카드만 face-down(나머지는 face-up). gimmick deck "
            "(rough-smooth principle)."
        ),
        "beats": [
            ("imaginary_selection", "관객이 머릿속으로 카드 선택"),
            ("reveal", "실제 덱을 펼치면 그 카드만 반대 방향"),
        ],
        "techniques": ["gimmick_deck"],
        "visual_cues": [
            "마술사가 덱을 잠깐 카메라 밖으로 가져감(미세 조작)",
            "최종 펼침에 한 장만 face-down",
        ],
        "query": "invisible deck trick tutorial",
    },
}


# ---------- 조회 헬퍼 ----------
def _norm(s: str) -> str:
    return s.strip().lower().replace("-", " ")


def lookup(name: str) -> dict | None:
    """트릭명(한/영, 별칭, 느슨)으로 트릭 항목 조회. 못 찾으면 None."""
    q = _norm(name)
    best, best_len = None, 0
    for entry in TRICKS.values():
        cands = [_norm(a) for a in entry["aliases"]]
        cands += [_norm(entry["en"]), _norm(entry["ko"])]
        for a in cands:
            if not a:
                continue
            if q == a:
                return entry
            if a in q and len(a) > best_len:
                best, best_len = entry, len(a)
    return best


def by_effect(effect: str | None = None) -> list[dict]:
    """effect 카테고리(coincidence/transformation 등)로 트릭 필터링.
    effect=None이면 전체."""
    if not effect:
        return list(TRICKS.values())
    e = effect.strip().lower()
    return [t for t in TRICKS.values() if t["effect"] == e]


def all_effects() -> list[str]:
    return sorted({t["effect"] for t in TRICKS.values()})


def search_url(name: str) -> str:
    """트릭에 대한 YouTube 검색 URL."""
    import urllib.parse
    entry = lookup(name)
    q = entry["query"] if entry else f"{name} card magic tutorial"
    return f"https://www.youtube.com/results?search_query={urllib.parse.quote(q)}"
