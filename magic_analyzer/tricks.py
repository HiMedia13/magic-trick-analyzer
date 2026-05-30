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

    # ===== Coincidence (추가) =====
    "twin_revelation": {
        "ko": "트윈 리벨레이션", "en": "Twin Revelation",
        "aliases": ["twin revelation", "mate finder", "twin card"],
        "effect": "coincidence",
        "desc": "관객과 마술사가 각각 카드를 골랐는데 두 카드가 같은 카드(또는 mate — 같은 값/색)로 일치.",
        "beats": [
            ("two_decks", "두 덱 또는 두 묶음 준비"),
            ("two_selections", "관객·마술사 각각 카드 선택"),
            ("reveal", "두 카드 동시 비교 — 같은 카드/mate"),
        ],
        "techniques": ["card_force", "key_card", "false_shuffle"],
        "visual_cues": ["두 덱·묶음 사용", "두 카드 동시 reveal"],
        "query": "twin card revelation mate finder tutorial",
    },
    "card_at_any_number": {
        "ko": "카드 앳 애니 넘버 (ACAAN)",
        "en": "Any Card at Any Number (ACAAN)",
        "aliases": ["acaan", "any card at any number", "berglas effect"],
        "effect": "prediction",
        "desc": "관객이 카드 이름과 숫자를 자유롭게 부르면 마술사가 그 숫자번째 위치에서 그 카드를 꺼냄. 마술의 holy grail.",
        "beats": [
            ("call_card", "관객이 카드명 부름"),
            ("call_number", "관객이 숫자(1~52) 부름"),
            ("count", "마술사가 그 숫자만큼 카드 셈"),
            ("reveal", "정확히 그 카드"),
        ],
        "techniques": ["memorized_deck", "estimation", "key_card", "false_count"],
        "visual_cues": ["두 자유 선택", "한 장씩 세기", "정확 매칭 reveal"],
        "query": "any card at any number berglas tutorial",
    },
    "math_force_match": {
        "ko": "수학적 매칭",
        "en": "Mathematical Force Match",
        "aliases": ["mathematical match", "math force", "9 card"],
        "effect": "coincidence",
        "desc": "관객의 자유로운 셔플·계산 후 도달한 카드가 마술사 예측과 일치. 수학적 force가 핵심.",
        "beats": [
            ("setup", "덱 일부에 force 카드 배치"),
            ("audience_count", "관객이 자유로워 보이는 계산"),
            ("reveal", "결과 카드 = 예측 카드"),
        ],
        "techniques": ["math_force", "9_card_principle", "spelling_force"],
        "visual_cues": ["관객이 자유롭게 카드 셈", "예측 봉투 reveal"],
        "query": "mathematical card force trick tutorial",
    },
    "gemini_twins": {
        "ko": "제미니 트윈스",
        "en": "Gemini Twins",
        "aliases": ["gemini twins", "karl fulves twins"],
        "effect": "coincidence",
        "desc": "Karl Fulves의 셀프워킹 트릭. 관객이 두 번 cut + 카드 끼워 넣으면 그 두 카드가 mate 페어로 일치.",
        "beats": [
            ("two_cuts", "관객이 덱을 두 번 cut"),
            ("insert_marks", "각 cut 위에 표식 카드(보통 두 jacks) 삽입"),
            ("reveal", "표식 위 두 카드가 mate"),
        ],
        "techniques": ["setup_stack"],
        "visual_cues": ["두 번 cut", "두 표식 카드"],
        "query": "gemini twins karl fulves card trick",
    },
    "lazy_man_card_trick": {
        "ko": "레이지 맨 카드 트릭",
        "en": "Lazy Man's Card Trick",
        "aliases": ["lazy man card trick", "lazy mans"],
        "effect": "coincidence",
        "desc": "관객이 카드 선택·섞기·cut 모두 직접 하고 마술사는 손도 안 댐. 그래도 결과적으로 관객 카드가 위치한 곳을 마술사가 정확히 짚음.",
        "beats": [
            ("selection", "관객이 직접 카드 선택"),
            ("audience_shuffle", "관객이 직접 섞고 cut"),
            ("reveal", "마술사가 카드 위치 정확히 지목"),
        ],
        "techniques": ["key_card", "crimp"],
        "visual_cues": ["마술사 손이 덱에 닿지 않음", "관객 주도 셔플"],
        "query": "lazy mans card trick tutorial",
    },
    "twenty_one_card_trick": {
        "ko": "21장 카드 트릭",
        "en": "Twenty-One Card Trick",
        "aliases": ["21 card trick", "twenty one card trick", "11th card"],
        "effect": "revelation",
        "desc": "21장을 세 줄로 나눠 관객이 자기 카드 있는 줄을 3번 가리키면, 마술사가 정확히 11번째에서 찾아냄. 클래식 자기수렴(self-working) 트릭.",
        "beats": [
            ("deal_21", "21장을 7장씩 3줄로 분배"),
            ("point_3x", "관객이 자기 카드 있는 줄을 3번 가리킴"),
            ("regather", "매번 그 줄을 가운데로 모아 다시 분배"),
            ("reveal", "11번째 카드가 관객 카드"),
        ],
        "techniques": ["mathematical_principle"],
        "visual_cues": ["7×3 분배 3회 반복", "11번째 카드 reveal"],
        "query": "21 card trick tutorial",
    },
    "sympathetic_aces": {
        "ko": "심퍼세틱 에이스",
        "en": "Sympathetic Aces",
        "aliases": ["sympathetic aces", "ace sympathy"],
        "effect": "sympathy",
        "desc": "두 묶음에 각각 4장 에이스 배치. 한쪽 에이스가 변하면 다른 쪽도 같이 변함(공감).",
        "beats": [
            ("setup_two_packets", "두 묶음에 4 에이스씩 배치"),
            ("change_one", "한 묶음의 에이스를 변화시킴"),
            ("reveal_sympathy", "다른 묶음도 같이 변함 확인"),
        ],
        "techniques": ["packet_switch", "double_lift"],
        "visual_cues": ["두 묶음 사용", "동시 변화"],
        "query": "sympathetic aces routine tutorial",
    },
    "do_as_i_do_variation": {
        "ko": "두 애즈 아이 두 변형",
        "en": "Do as I Do (Variation)",
        "aliases": ["mirror card", "mirror selection"],
        "effect": "coincidence",
        "desc": "관객과 마술사 두 덱으로 동일 동작 mirroring. 결과 카드 정확 일치.",
        "beats": [
            ("two_decks", "두 덱 분배"),
            ("mirror", "동작 mirroring"),
            ("reveal", "두 선택 카드 일치"),
        ],
        "techniques": ["card_force", "false_shuffle"],
        "visual_cues": ["두 명 동시 동작", "거울 reveal"],
        "query": "do as i do variation tutorial",
    },

    # ===== Transformation (추가) =====
    "classic_color_change": {
        "ko": "클래식 컬러 체인지",
        "en": "Classic Color Change",
        "aliases": ["color change", "erdnase color change", "visual change"],
        "effect": "transformation",
        "desc": "마술사가 손짓 한 번에 보여주던 카드가 다른 카드로 시각적으로 변함. Erdnase, Hofzinser 등의 다양한 변형.",
        "beats": [
            ("show_card", "한 카드 보여줌"),
            ("wave_or_pass", "손짓·터치"),
            ("reveal_change", "다른 카드로 변함"),
        ],
        "techniques": ["color_change", "card_palm", "top_change"],
        "visual_cues": ["한 카드만 보여주다가 다른 카드로 즉시 변함", "손이 카드 위를 지나감"],
        "query": "classic color change card tutorial erdnase",
    },
    "wild_card": {
        "ko": "와일드 카드",
        "en": "Wild Card",
        "aliases": ["wild card", "peter kane wild card"],
        "effect": "transformation",
        "desc": "Peter Kane 작. 한 종류 카드들이 한 장씩 다른 종류로 변함. 결국 모두 같은 카드로 통일.",
        "beats": [
            ("show_set", "8~10장의 한 종류 카드 보여줌"),
            ("change_one_by_one", "터치할 때마다 한 장씩 다른 카드로 변함"),
            ("final_state", "모두 같은 새 카드로 통일"),
        ],
        "techniques": ["packet_count", "double_face_card", "card_switch"],
        "visual_cues": ["packet 8~10장", "반복적 변화"],
        "query": "wild card peter kane tutorial",
    },
    "mcdonalds_aces": {
        "ko": "맥도날드 에이스",
        "en": "McDonald's Aces",
        "aliases": ["mcdonald aces", "mcdonald's $100 routine", "ace assembly"],
        "effect": "transformation",
        "desc": "4개 묶음에 한 장씩 에이스를 두고 나머지는 보통 카드. 마지막에 모든 에이스가 한 묶음에 모임. 'McDonald's $100 Routine'으로도 알려짐.",
        "beats": [
            ("show_4_aces", "4 에이스를 4 위치에 분배"),
            ("each_assembly", "각 위치의 에이스가 leader 묶음으로 이동"),
            ("final_reveal", "leader 묶음에 4 에이스 모두 모임"),
        ],
        "techniques": ["double_lift", "elmsley_count", "card_switch", "gimmick_card"],
        "visual_cues": ["4 묶음 분배", "차례로 한 묶음으로 모임"],
        "query": "mcdonald aces routine ace assembly tutorial",
    },
    "three_card_monte": {
        "ko": "쓰리 카드 몬테",
        "en": "Three Card Monte",
        "aliases": ["3 card monte", "three card monte", "find the lady",
                    "monte"],
        "effect": "transformation",
        "desc": "세 카드(보통 두 잭+퀸) 중 퀸을 찾는 게임. 관객이 따라가지만 항상 틀림. 일부 변형은 퀸이 색까지 변함.",
        "beats": [
            ("show_3", "세 카드 보여줌, 한 장이 target"),
            ("shuffle", "빠르게 위치 섞음"),
            ("audience_guess", "관객이 target 위치 추측"),
            ("reveal", "관객 추측 틀림(target은 다른 위치/변형)"),
        ],
        "techniques": ["mexican_turnover", "monte_throw", "card_switch"],
        "visual_cues": ["3장만 사용", "위치 빠르게 섞음"],
        "query": "three card monte tutorial find the lady",
    },
    "universal_card": {
        "ko": "유니버설 카드",
        "en": "Universal Card",
        "aliases": ["universal card"],
        "effect": "transformation",
        "desc": "한 카드가 어떤 카드든 될 수 있음을 보여줌. 관객이 부른 카드로 변하는 시연 반복.",
        "beats": [
            ("show_blank_or_card", "기본 카드 보여줌"),
            ("audience_call", "관객이 임의 카드명 외침"),
            ("change", "그 카드로 변함"),
            ("repeat", "여러 번 반복"),
        ],
        "techniques": ["double_lift", "card_switch", "top_change"],
        "visual_cues": ["한 카드 반복 변화", "관객 호명 즉시 변화"],
        "query": "universal card trick tutorial",
    },
    "blank_to_face": {
        "ko": "블랭크 투 페이스 (프린트 효과)",
        "en": "Blank to Face (Print Effect)",
        "aliases": ["blank to face", "print effect", "blank card print"],
        "effect": "transformation",
        "desc": "백지 카드(또는 빈 카드 묶음)에 손짓하면 그림이 'print'되어 나타남.",
        "beats": [
            ("show_blanks", "빈 카드 묶음 보여줌"),
            ("magic_wave", "손짓·문지름"),
            ("reveal_print", "카드 face가 나타남"),
        ],
        "techniques": ["double_face_card", "packet_count", "card_switch"],
        "visual_cues": ["빈 카드가 그림 카드로 변함"],
        "query": "blank to face card trick print effect",
    },
    "all_backs": {
        "ko": "올 백스",
        "en": "All Backs",
        "aliases": ["all backs", "dai vernon all backs"],
        "effect": "transformation",
        "desc": "Dai Vernon. 덱의 모든 카드가 양면이 백이라고 보여준 뒤, 다시 face가 나타남.",
        "beats": [
            ("show_all_backs", "양면 모두 백 보여줌"),
            ("magic_moment", "손짓"),
            ("reveal_faces", "다시 face 나타남"),
        ],
        "techniques": ["false_count", "double_lift"],
        "visual_cues": ["양면 백 → face 변화"],
        "query": "all backs dai vernon tutorial",
    },
    "chicago_opener": {
        "ko": "시카고 오프너 (레드 핫 마마 변형)",
        "en": "Chicago Opener",
        "aliases": ["chicago opener", "red hot mama variation"],
        "effect": "coincidence",
        "desc": "Red Hot Mama의 변형. 한 카드 백 색이 다른 것을 reveal, 그 카드가 chosen이라며 turn하면 다시 face가 변함. 다중 reveal.",
        "beats": [
            ("setup", "다른 색 백 카드 배치"),
            ("selection", "관객 카드 선택"),
            ("first_reveal", "한 카드 백 색 다름 보여줌"),
            ("second_reveal", "그 카드 face가 chosen으로 변함"),
        ],
        "techniques": ["double_lift", "card_switch", "setup_gimmick"],
        "visual_cues": ["색 다른 카드 한 장", "여러 단계 reveal"],
        "query": "chicago opener card trick tutorial",
    },
    "monte_color": {
        "ko": "컬러 몬테",
        "en": "Color Monte",
        "aliases": ["color monte", "red and black monte"],
        "effect": "transformation",
        "desc": "Three card monte 변형. 관객이 매번 'red 카드'를 잘못 짚는데, 결국 마지막에 모두 black이 됨.",
        "beats": [
            ("show_red_black", "red·black 카드들 보여줌"),
            ("repeated_misses", "관객 매번 위치 잘못 짚음"),
            ("final_change", "결국 모두 black"),
        ],
        "techniques": ["packet_count", "color_change"],
        "visual_cues": ["3 packet 사용", "색 식별 게임"],
        "query": "color monte card trick tutorial",
    },
    "backstrokes": {
        "ko": "백스트로크",
        "en": "Backstrokes",
        "aliases": ["backstrokes", "back changes"],
        "effect": "transformation",
        "desc": "덱의 등(back)이 한 카드씩 다른 색·문양으로 변함.",
        "beats": [
            ("show_backs", "덱 등 한 가지 색"),
            ("backstroke", "쓸어내림"),
            ("reveal_change", "등이 다른 색/문양으로 변함"),
        ],
        "techniques": ["deck_switch", "double_face_card"],
        "visual_cues": ["등 색 변화"],
        "query": "card back change tutorial",
    },
    "mexican_turnover_change": {
        "ko": "멕시칸 턴오버 체인지",
        "en": "Mexican Turnover Change",
        "aliases": ["mexican turnover", "mexican change"],
        "effect": "transformation",
        "desc": "한 카드를 다른 카드로 뒤집어 보여주는 척 실제로는 swap. 한 카드가 다른 카드로 변하는 모먼트.",
        "beats": [
            ("show_card_a", "카드 A 보여줌"),
            ("turnover_swap", "다른 카드로 face down 카드 뒤집기"),
            ("reveal_b", "카드 A 자리에 카드 B"),
        ],
        "techniques": ["mexican_turnover"],
        "visual_cues": ["테이블에서 카드 뒤집는 동작 직후 변함"],
        "query": "mexican turnover tutorial",
    },

    # ===== Prediction (추가) =====
    "open_prediction": {
        "ko": "오픈 프리딕션",
        "en": "Open Prediction",
        "aliases": ["open prediction", "annemann prediction"],
        "effect": "prediction",
        "desc": "Annemann. 마술사가 예측 카드(예: 'Two of Spades')를 미리 공개한 뒤 관객이 자유롭게 카드를 face-down으로 한 장 두면 그게 정확히 예측 카드.",
        "beats": [
            ("open_predict", "예측 카드명 공개"),
            ("free_deal", "관객이 face down 한 장 자유 선택"),
            ("reveal", "그 카드가 예측 카드"),
        ],
        "techniques": ["card_force", "memorized_deck"],
        "visual_cues": ["예측 미리 공개", "관객 한 장 face down 분리"],
        "query": "open prediction annemann tutorial",
    },
    "b_wave": {
        "ko": "비웨이브 (B'wave)",
        "en": "B'wave",
        "aliases": ["bwave", "b wave", "max maven b'wave"],
        "effect": "prediction",
        "desc": "Max Maven. 관객이 4 퀸 중 하나 선택. 그 퀸만 face-down에서 face-up, 나머지는 백 색까지 다른 카드 reveal.",
        "beats": [
            ("show_4_queens", "4 퀸 face down 묶음"),
            ("audience_pick", "관객이 한 퀸 선택"),
            ("reveal", "선택 퀸만 face up, 나머지 3장은 다른 색 백"),
        ],
        "techniques": ["gimmick_card", "packet_routine"],
        "visual_cues": ["4 퀸 packet", "선택 카드만 face up, 나머지 색 다름"],
        "query": "b wave max maven tutorial",
    },
    "newspaper_prediction": {
        "ko": "뉴스페이퍼 프리딕션",
        "en": "Newspaper Prediction",
        "aliases": ["newspaper prediction", "headline prediction"],
        "effect": "prediction",
        "desc": "신문 헤드라인이나 광고에 미리 적힌 단어/카드가 관객의 자유 선택과 일치.",
        "beats": [
            ("show_paper", "신문 보여줌"),
            ("audience_choice", "관객 자유 선택"),
            ("reveal_match", "신문 내용과 선택 일치"),
        ],
        "techniques": ["gimmick_prop", "card_force"],
        "visual_cues": ["신문 prop", "헤드라인 reveal"],
        "query": "newspaper prediction magic trick",
    },
    "envelope_prediction": {
        "ko": "엔벨로프 프리딕션",
        "en": "Sealed Envelope Prediction",
        "aliases": ["envelope prediction", "sealed prediction"],
        "effect": "prediction",
        "desc": "봉인된 봉투에 적힌 예측이 관객의 자유 선택과 일치. 가장 보편적인 mental 트릭 포맷.",
        "beats": [
            ("show_envelope", "봉인 봉투 보여줌"),
            ("audience_pick", "관객 카드 선택"),
            ("open_envelope", "봉투 열어 예측 reveal"),
            ("match", "예측 = 선택 카드"),
        ],
        "techniques": ["card_force", "switch_envelope", "billet_switch"],
        "visual_cues": ["봉투 prop", "봉투 개봉 reveal"],
        "query": "sealed envelope prediction card magic",
    },
    "lottery_prediction": {
        "ko": "로또 프리딕션",
        "en": "Lottery Prediction",
        "aliases": ["lottery prediction", "lotto numbers"],
        "effect": "prediction",
        "desc": "관객이 자유로 선택한 숫자들이 미리 적힌 예측과 일치. 종종 봉투/봉인된 종이 사용.",
        "beats": [
            ("show_prediction", "예측 봉투/종이 미리 보여줌"),
            ("audience_picks_numbers", "관객 자유 선택 숫자(또는 카드)"),
            ("reveal", "예측과 일치"),
        ],
        "techniques": ["card_force", "billet_switch"],
        "visual_cues": ["숫자 선택", "예측 종이 reveal"],
        "query": "lottery prediction magic trick tutorial",
    },
    "seven_keys": {
        "ko": "7개 열쇠 (Seven Keys to Baldpate)",
        "en": "Seven Keys to Baldpate",
        "aliases": ["seven keys", "baldpate keys"],
        "effect": "prediction",
        "desc": "여러 자물쇠 중 관객이 고르는 하나만 마술사 열쇠가 열림. 카드 버전은 자유 선택 카드만 lock 풀림.",
        "beats": [
            ("show_locks", "여러 lock 보여줌"),
            ("audience_pick", "관객 선택"),
            ("reveal", "선택 lock만 열림"),
        ],
        "techniques": ["force", "gimmick_prop"],
        "visual_cues": ["여러 lock 사용", "한 lock만 열림"],
        "query": "seven keys to baldpate magic",
    },
    "q_and_a": {
        "ko": "Q&A 액트",
        "en": "Q&A Act",
        "aliases": ["q and a", "question and answer act"],
        "effect": "mental",
        "desc": "관객들의 비밀 질문을 마술사가 보지 않고 정확히 읽고 답함. 멘탈리즘 핵심 루틴.",
        "beats": [
            ("audience_writes", "관객들이 비밀 질문 작성"),
            ("collect", "쪽지 모음"),
            ("read_and_answer", "마술사가 보지 않고 읽고 답"),
        ],
        "techniques": ["billet_switch", "one_ahead", "center_tear"],
        "visual_cues": ["쪽지 사용", "마술사가 안 보고 답"],
        "query": "q and a act mentalism tutorial",
    },
    "center_tear": {
        "ko": "센터 티어",
        "en": "Center Tear",
        "aliases": ["center tear", "billet tear"],
        "effect": "mental",
        "desc": "관객이 종이에 단어/카드명 작성 → 마술사가 종이를 찢어 태우는 동안 가운데 부분을 몰래 빼냄 → 관객이 적은 것 reveal.",
        "beats": [
            ("audience_writes", "관객 단어 작성"),
            ("fold_tear", "종이 접고 찢기"),
            ("secret_peek", "찢기 중 가운데 부분 빼내 peek"),
            ("burn", "쪽지 태우기"),
            ("reveal", "관객 단어 reveal"),
        ],
        "techniques": ["center_tear", "billet_switch"],
        "visual_cues": ["종이 찢기/태우기", "마술사가 안 보고 reveal"],
        "query": "center tear mentalism tutorial",
    },

    # ===== Transposition (추가) =====
    "two_card_transposition": {
        "ko": "투 카드 트랜스포",
        "en": "Two Card Transposition",
        "aliases": ["two card transpo", "tc transpo"],
        "effect": "transposition",
        "desc": "두 카드(예: 빨간 잭, 검은 잭)가 위치를 교환. 한 장은 마술사 손에, 다른 장은 관객 손에.",
        "beats": [
            ("show_two", "두 카드 위치 명확히 보여줌"),
            ("magic_moment", "손짓"),
            ("reveal_swap", "두 카드 위치 바뀜"),
        ],
        "techniques": ["double_lift", "card_switch", "top_change"],
        "visual_cues": ["두 카드 비교", "각 카드가 다른 손/위치"],
        "query": "two card transposition tutorial",
    },
    "card_to_wallet": {
        "ko": "카드 투 월렛",
        "en": "Card to Wallet",
        "aliases": ["card to wallet", "himber wallet card"],
        "effect": "transposition",
        "desc": "선택 카드가 덱에서 사라져 마술사 지갑(또는 봉인된 곳) 안에서 발견됨. 'Himber wallet', 'Mullica wallet' 등 전용 도구 사용.",
        "beats": [
            ("selection", "관객 카드 선택"),
            ("vanish", "카드가 덱에서 사라짐"),
            ("wallet_reveal", "지갑에서 카드 produce"),
        ],
        "techniques": ["card_palm", "load_wallet", "card_switch"],
        "visual_cues": ["마술사 지갑 사용", "지갑 안에서 카드 빼냄"],
        "query": "card to wallet tutorial himber",
    },
    "card_to_mouth": {
        "ko": "카드 투 마우스",
        "en": "Card to Mouth",
        "aliases": ["card to mouth", "card from mouth"],
        "effect": "transposition",
        "desc": "선택 카드가 사라졌다가 마술사 입에서 나옴.",
        "beats": [
            ("selection", "관객 카드 선택"),
            ("vanish", "덱에서 사라짐"),
            ("mouth_reveal", "마술사 입에서 카드 빼냄"),
        ],
        "techniques": ["card_palm", "mouth_load"],
        "visual_cues": ["손이 입 부근으로", "카드 입에서 빼냄"],
        "query": "card to mouth trick tutorial",
    },
    "card_to_lemon": {
        "ko": "카드 투 레몬",
        "en": "Card to Lemon",
        "aliases": ["card to lemon", "card in lemon", "card to orange"],
        "effect": "transposition",
        "desc": "관객 카드가 사라져 마술사가 자른 레몬/오렌지 속에서 발견됨. Setup이 까다로운 고난도 클로즈업.",
        "beats": [
            ("show_lemon", "레몬을 미리 보여줌"),
            ("selection", "관객 카드 선택"),
            ("vanish", "카드 사라짐"),
            ("cut_lemon", "레몬 자름"),
            ("reveal", "레몬 안에 접힌 카드"),
        ],
        "techniques": ["lemon_load", "card_force", "card_palm"],
        "visual_cues": ["레몬/오렌지 prop", "과일 자르기"],
        "query": "card in lemon trick tutorial",
    },
    "open_travelers": {
        "ko": "오픈 트래블러스",
        "en": "Open Travelers",
        "aliases": ["open travelers", "vernon open travelers"],
        "effect": "transposition",
        "desc": "Dai Vernon. 4 에이스가 4 묶음에 있다가 한 묶음으로 'open'하게(보이는 방식으로) 모임. Closed assembly와 대비.",
        "beats": [
            ("4_packets", "4 묶음에 에이스 한 장씩"),
            ("visible_travel", "에이스가 천천히 한 묶음으로 이동(visible)"),
            ("reveal", "한 묶음에 4 에이스 모임"),
        ],
        "techniques": ["card_palm", "card_switch", "false_count"],
        "visual_cues": ["4 묶음", "visible 이동"],
        "query": "open travelers dai vernon tutorial",
    },
    "cards_across": {
        "ko": "카드 어크로스",
        "en": "Cards Across",
        "aliases": ["cards across", "3 cards across"],
        "effect": "transposition",
        "desc": "두 관객에게 같은 수의 카드 분배. 마술사 손짓에 한쪽 카드 3장이 다른 쪽으로 'across'. 카운트 결과 일치.",
        "beats": [
            ("two_packets", "두 관객에게 같은 수 분배"),
            ("magic_moment", "손짓 — 3장이 across"),
            ("count_reveal", "두 묶음 다시 세서 변화 확인"),
        ],
        "techniques": ["card_palm", "load", "false_count"],
        "visual_cues": ["두 관객", "카운트 변화"],
        "query": "cards across trick tutorial",
    },
    "card_to_pocket_signed": {
        "ko": "사인드 카드 투 포켓",
        "en": "Signed Card to Pocket",
        "aliases": ["signed card to pocket", "scott alexander card"],
        "effect": "transposition",
        "desc": "관객이 카드에 직접 서명 후 덱으로. 카드가 사라져 마술사 주머니에서 발견 — 그 서명까지 일치.",
        "beats": [
            ("sign", "관객 카드에 서명"),
            ("vanish", "덱에서 사라짐"),
            ("pocket_reveal", "주머니에서 같은 서명 카드 빼냄"),
        ],
        "techniques": ["card_palm", "topit"],
        "visual_cues": ["관객 서명 동작", "주머니 reveal"],
        "query": "signed card to pocket tutorial",
    },

    # ===== Production (추가) =====
    "four_ace_production": {
        "ko": "포 에이스 프로덕션",
        "en": "Four Ace Production",
        "aliases": ["four ace production", "ace production"],
        "effect": "production",
        "desc": "마술사가 셔플된 덱에서 4 에이스를 차례로 produce. Cardistry 요소 포함된 표현 다양.",
        "beats": [
            ("shuffle", "덱 섞음"),
            ("produce_each", "에이스 한 장씩 produce"),
            ("final", "4 에이스 모두 보여줌"),
        ],
        "techniques": ["spread_cull", "side_steal", "card_palm",
                       "ace_production_flourish"],
        "visual_cues": ["덱에서 카드 빼내는 동작 4번", "에이스 4장 fan"],
        "query": "four ace production card flourish tutorial",
    },
    "royal_flush_production": {
        "ko": "로열 플러시 프로덕션",
        "en": "Royal Flush Production",
        "aliases": ["royal flush production"],
        "effect": "production",
        "desc": "카지노 dealing 시연에서 항상 royal flush를 dealing. 일반 셔플 후에 보여주는 것이 인상적.",
        "beats": [
            ("deal_hands", "포커 핸드 dealing"),
            ("reveal", "마술사 핸드가 royal flush"),
        ],
        "techniques": ["stack_deal", "bottom_deal", "second_deal"],
        "visual_cues": ["포커 dealing 동작", "royal flush 5장"],
        "query": "royal flush production card cheating demonstration",
    },
    "card_from_thin_air": {
        "ko": "카드 프롬 에어",
        "en": "Card from Thin Air",
        "aliases": ["card from air", "appearing card", "card from nowhere"],
        "effect": "production",
        "desc": "빈 손에서 카드가 시각적으로 나타남(produce).",
        "beats": [
            ("show_empty_hand", "빈 손 보여줌"),
            ("produce", "카드가 손에서 나타남"),
        ],
        "techniques": ["back_palm", "front_palm", "manipulation_card"],
        "visual_cues": ["빈 손 → 카드 나타남", "manipulation flourish"],
        "query": "card production from thin air manipulation",
    },
    "card_fountain": {
        "ko": "카드 파운틴",
        "en": "Card Fountain (Springing)",
        "aliases": ["card fountain", "card spring", "springing cards"],
        "effect": "production",
        "desc": "한 손에서 다른 손으로 덱이 분수처럼 솟구침. Cardistry/dexterity flourish.",
        "beats": [
            ("squeeze_deck", "덱 한 손으로 압축"),
            ("spring", "덱을 다른 손으로 spring"),
        ],
        "techniques": ["card_spring_flourish"],
        "visual_cues": ["카드가 폭포처럼 손에서 손으로"],
        "query": "card spring fountain cardistry tutorial",
    },
    "card_from_box": {
        "ko": "카드 프롬 박스",
        "en": "Card from Card Case",
        "aliases": ["card from box", "card from case"],
        "effect": "production",
        "desc": "사라진 카드가 카드 케이스 내부, 또는 봉인된 새 케이스에서 발견.",
        "beats": [
            ("selection", "관객 카드 선택"),
            ("vanish", "사라짐"),
            ("box_reveal", "케이스에서 카드 produce"),
        ],
        "techniques": ["card_switch", "card_palm", "secret_load"],
        "visual_cues": ["덱 케이스 prop", "케이스 안에서 카드"],
        "query": "card from sealed case tutorial",
    },
    "blizzard": {
        "ko": "블리자드",
        "en": "Blizzard",
        "aliases": ["blizzard", "frank everhart blizzard"],
        "effect": "production",
        "desc": "여러 카드의 한 카드가 관객 카드로 변하면서 빈 카드들로 변환 + reveal. 다중 packet 효과.",
        "beats": [
            ("show_packets", "여러 packet 보여줌"),
            ("magic_moment", "손짓"),
            ("reveal", "관객 카드 produce + 나머지 변환"),
        ],
        "techniques": ["packet_count", "card_switch", "double_face_card"],
        "visual_cues": ["packet 사용", "다중 카드 변화"],
        "query": "blizzard card trick tutorial",
    },

    # ===== Vanish (추가) =====
    "card_vanish_visual": {
        "ko": "비주얼 카드 배니쉬",
        "en": "Visual Card Vanish",
        "aliases": ["card vanish", "visual card vanish"],
        "effect": "vanish",
        "desc": "보여주던 카드가 시각적으로 사라짐. Palm, paddle move 등으로 한 순간에 disappear.",
        "beats": [
            ("show_card", "카드 보여줌"),
            ("vanish", "카드 시각적으로 사라짐"),
        ],
        "techniques": ["card_palm", "paddle_move", "snap_change"],
        "visual_cues": ["한 카드가 손에서 즉시 사라짐"],
        "query": "visual card vanish tutorial",
    },
    "vanishing_deck": {
        "ko": "배니싱 덱",
        "en": "Vanishing Deck",
        "aliases": ["vanishing deck", "deck vanish"],
        "effect": "vanish",
        "desc": "전체 덱이 한 순간에 사라지거나 한 카드만 남고 vanish. 대형 효과.",
        "beats": [
            ("show_deck", "덱 보여줌"),
            ("magic_moment", "손짓·천 덮기"),
            ("vanish", "덱 사라짐"),
        ],
        "techniques": ["deck_palm", "deck_switch"],
        "visual_cues": ["덱 전체 사라짐"],
        "query": "vanishing deck of cards tutorial",
    },
    "card_to_smoke": {
        "ko": "카드 투 스모크",
        "en": "Card to Smoke",
        "aliases": ["card to smoke", "smoke vanish"],
        "effect": "vanish",
        "desc": "카드가 연기로 변하면서 사라짐. Flash paper나 gimmick.",
        "beats": [
            ("show_card", "카드 보여줌"),
            ("burn_or_squeeze", "불·짓이김 동작"),
            ("vanish_smoke", "연기 + 카드 사라짐"),
        ],
        "techniques": ["flash_paper", "card_switch"],
        "visual_cues": ["연기 효과", "카드 disappear"],
        "query": "card to smoke trick tutorial",
    },

    # ===== Restoration (추가) =====
    "torn_and_restored_card": {
        "ko": "톤 앤 리스토어드 카드",
        "en": "Torn and Restored Card",
        "aliases": ["torn and restored", "torn card restoration"],
        "effect": "restoration",
        "desc": "관객 카드를 마술사가 네 조각으로 찢은 뒤 한 조각씩 붙여 원래로 복원. 한 조각만 남기고 마지막에 매칭.",
        "beats": [
            ("show_card", "관객 카드 보여줌"),
            ("tear_to_pieces", "4 조각으로 찢음"),
            ("restore", "조각들 점진적 복원"),
            ("final_match", "복원 카드의 모자란 조각이 관객 주머니에서"),
        ],
        "techniques": ["card_switch", "duplicate_card", "card_fold"],
        "visual_cues": ["카드 찢기", "점진적 복원"],
        "query": "torn and restored card tutorial",
    },
    "card_through_handkerchief": {
        "ko": "카드 쓰루 핸커치프",
        "en": "Card Through Handkerchief",
        "aliases": ["card through handkerchief", "card through silk"],
        "effect": "penetration",
        "desc": "관객 카드가 손수건/실크를 관통.",
        "beats": [
            ("selection", "관객 카드"),
            ("wrap", "손수건으로 카드 감쌈"),
            ("penetrate", "카드가 손수건 뚫고 나옴"),
        ],
        "techniques": ["card_palm", "silk_handling"],
        "visual_cues": ["손수건 사용", "관통 효과"],
        "query": "card through handkerchief tutorial",
    },
    "card_through_newspaper": {
        "ko": "카드 쓰루 뉴스페이퍼",
        "en": "Card Through Newspaper",
        "aliases": ["card through newspaper"],
        "effect": "penetration",
        "desc": "관객 카드가 신문을 관통. 신문을 여러 번 접은 뒤 카드가 그 안에서 나타남.",
        "beats": [
            ("selection", "관객 카드 선택"),
            ("fold_newspaper", "신문 여러 번 접음"),
            ("penetrate", "카드가 신문 안에서 나타남"),
        ],
        "techniques": ["card_palm", "load", "fold"],
        "visual_cues": ["신문 prop", "신문 펼치며 카드 reveal"],
        "query": "card through newspaper tutorial",
    },
    "card_through_glass": {
        "ko": "카드 쓰루 글래스",
        "en": "Card Through Glass",
        "aliases": ["card through glass", "card through window"],
        "effect": "penetration",
        "desc": "관객 카드가 유리/창문을 관통해 반대편에 붙어 있음. Street magic 클래식.",
        "beats": [
            ("selection", "관객 카드 선택"),
            ("vanish", "카드 사라짐"),
            ("through_glass", "유리 반대편에서 카드 발견"),
        ],
        "techniques": ["card_force", "duplicate_card", "secret_helper"],
        "visual_cues": ["유리/창문 prop", "반대편 카드 reveal"],
        "query": "card through window glass tutorial",
    },

    # ===== Attraction / 부착 =====
    "card_on_ceiling": {
        "ko": "카드 온 실링",
        "en": "Card on Ceiling",
        "aliases": ["card on ceiling", "card to ceiling"],
        "effect": "attraction",
        "desc": "관객 카드가 사라져 천장에 붙어 있음. Spit + force 카드를 천장에 던져 부착.",
        "beats": [
            ("selection", "관객 카드"),
            ("throw_deck", "덱을 천장으로 던짐"),
            ("reveal", "카드 한 장이 천장에 붙어 있음 = 관객 카드"),
        ],
        "techniques": ["card_force", "spit_method"],
        "visual_cues": ["덱을 천장으로 던짐", "천장에 카드 부착"],
        "query": "card on ceiling tutorial spit method",
    },
    "card_on_forehead": {
        "ko": "카드 온 포어헤드",
        "en": "Card on Forehead",
        "aliases": ["card on forehead", "stick card forehead"],
        "effect": "attraction",
        "desc": "카드가 관객/마술사 이마에 자연스럽게 부착.",
        "beats": [
            ("show_card", "카드 face 보여줌"),
            ("press_forehead", "이마에 누름"),
            ("release", "손 떼도 카드가 이마에 붙어 있음"),
        ],
        "techniques": ["spit_method", "static"],
        "visual_cues": ["이마 부근 카드"],
        "query": "card stuck to forehead trick",
    },

    # ===== Ambitious 변형 =====
    "pop_up_card": {
        "ko": "팝업 카드",
        "en": "Pop-up Card",
        "aliases": ["pop up card", "rising card"],
        "effect": "ambitious",
        "desc": "덱에 묻힌 관객 카드가 손가락 까딱에 시각적으로 위로 튀어나옴(올라옴).",
        "beats": [
            ("selection", "관객 카드 선택"),
            ("bury", "덱에 묻음"),
            ("pop_up", "손짓에 카드가 위로 튀어나옴"),
        ],
        "techniques": ["thumb_pop", "deck_riffle", "concealed_string"],
        "visual_cues": ["덱에서 한 카드가 위로 솟구침"],
        "query": "rising card pop up trick tutorial",
    },
    "spelling_routine": {
        "ko": "스펠링 루틴",
        "en": "Spelling Routine",
        "aliases": ["spelling routine", "spell-a-card"],
        "effect": "revelation",
        "desc": "관객 카드명을 한 글자씩 spell하면서 카드 한 장씩 deal. 마지막 글자에서 카드가 관객 카드.",
        "beats": [
            ("selection", "관객 카드 선택"),
            ("spell_deal", "카드명 spell하며 deal"),
            ("reveal", "마지막 글자 = 관객 카드"),
        ],
        "techniques": ["false_count", "spelling_force", "stacked_deck"],
        "visual_cues": ["글자별 카드 deal", "spelling 동작"],
        "query": "spelling card trick spell a card tutorial",
    },
    "down_under_deal": {
        "ko": "다운언더 딜",
        "en": "Down/Under Deal",
        "aliases": ["down under deal", "australian shuffle"],
        "effect": "revelation",
        "desc": "carlo, Down(table), Under(go to back) 패턴으로 카드 dealing. 마지막에 남는 카드가 관객 카드.",
        "beats": [
            ("packet", "packet 준비"),
            ("alternating_deal", "Down/Under 교대 deal"),
            ("last_card", "마지막 남은 카드 = 관객 카드"),
        ],
        "techniques": ["mathematical_principle"],
        "visual_cues": ["carlo Down/Under deal 패턴", "마지막 1장"],
        "query": "down under deal card trick tutorial",
    },
    "stop_trick": {
        "ko": "스톱 트릭",
        "en": "Stop Trick",
        "aliases": ["stop trick", "stop me deal"],
        "effect": "revelation",
        "desc": "마술사가 카드 한 장씩 deal, 관객이 'stop' 외친 시점의 카드가 관객 카드.",
        "beats": [
            ("selection", "관객 카드"),
            ("deal", "마술사 dealing"),
            ("audience_stops", "관객 'stop'"),
            ("reveal", "그 카드 = 관객 카드"),
        ],
        "techniques": ["estimation", "force", "second_deal"],
        "visual_cues": ["dealing 중 stop"],
        "query": "stop trick card magic tutorial",
    },

    # ===== Sympathy (Sympathetic 추가) =====
    "sympathetic_packets": {
        "ko": "심퍼세틱 팩킷",
        "en": "Sympathetic Packets",
        "aliases": ["sympathetic packets"],
        "effect": "sympathy",
        "desc": "두 packet이 어떤 변화든 함께 보임(섞기, 뒤집기, 순서 정렬 등).",
        "beats": [
            ("show_packets", "두 packet 준비"),
            ("change_one", "한 packet 변화"),
            ("reveal_sympathy", "두 packet 동일 변화"),
        ],
        "techniques": ["packet_switch", "false_shuffle"],
        "visual_cues": ["두 packet 사용", "동기적 변화"],
        "query": "sympathetic packets card trick",
    },

    # ===== Mental =====
    "think_a_card": {
        "ko": "씽크 어 카드",
        "en": "Think of a Card",
        "aliases": ["think of a card", "mental selection"],
        "effect": "mental",
        "desc": "관객이 머릿속에 카드를 정하고 마술사가 그 카드를 알아냄. 도구 없이 mental selection.",
        "beats": [
            ("mental_selection", "관객 머릿속 카드"),
            ("clue_phase", "마술사가 단서·반응 관찰"),
            ("reveal", "그 카드를 정확히 지목"),
        ],
        "techniques": ["psychological_force", "fishing", "cold_reading"],
        "visual_cues": ["관객이 말 없이 생각", "마술사가 추론"],
        "query": "think of a card mentalism tutorial",
    },
    "book_test": {
        "ko": "북 테스트",
        "en": "Book Test",
        "aliases": ["book test"],
        "effect": "mental",
        "desc": "관객이 임의 페이지 임의 단어를 선택, 마술사가 보지 않고 그 단어를 맞춤.",
        "beats": [
            ("show_book", "책 보여줌"),
            ("audience_choice", "페이지·단어 선택"),
            ("reveal", "마술사가 단어 reveal"),
        ],
        "techniques": ["book_test_force", "memorized_book", "marked_book"],
        "visual_cues": ["책 prop", "페이지·단어 reveal"],
        "query": "book test mentalism tutorial",
    },
    "pendulum_card": {
        "ko": "펜듈럼 카드",
        "en": "Pendulum Card",
        "aliases": ["pendulum card", "pendulum reveal"],
        "effect": "mental",
        "desc": "팬듈럼이 카드들 위를 지나다 관객 카드 위에서 멈춤.",
        "beats": [
            ("show_pendulum", "팬듈럼 prop"),
            ("hover_cards", "카드들 위 팬듈럼 흔들기"),
            ("stop_reveal", "관객 카드 위에서 멈춤"),
        ],
        "techniques": ["pendulum_handling", "force"],
        "visual_cues": ["팬듈럼 흔들림", "특정 카드 위에서 정지"],
        "query": "pendulum card reveal trick",
    },
    "color_sense": {
        "ko": "컬러 센스",
        "en": "Color Sense",
        "aliases": ["color sense", "red black sense"],
        "effect": "mental",
        "desc": "마술사가 등 뒤로 손 두고 빨강·검정을 정확히 분리. blindfold 가능.",
        "beats": [
            ("audience_shuffle", "관객이 덱 섞음"),
            ("behind_back", "마술사 손 뒤로"),
            ("separate", "빨강·검정 분리"),
            ("reveal", "완벽 분리"),
        ],
        "techniques": ["bridge", "marked_card", "tactile"],
        "visual_cues": ["손 뒤로/blindfold", "최종 빨강·검정 두 묶음"],
        "query": "color sense red black separation mentalism",
    },
    "name_a_card": {
        "ko": "네임 어 카드",
        "en": "Name a Card",
        "aliases": ["name a card", "card to position"],
        "effect": "prediction",
        "desc": "관객이 카드명 자유 호명, 마술사가 그 카드를 즉시 produce하거나 미리 정한 위치에서 reveal.",
        "beats": [
            ("audience_name", "관객 카드명 호명"),
            ("produce_reveal", "그 카드 즉시 produce/reveal"),
        ],
        "techniques": ["memorized_deck", "card_palm"],
        "visual_cues": ["호명 → 즉시 produce"],
        "query": "name any card production tutorial",
    },

    # ===== Assembly / Packet =====
    "oil_and_water": {
        "ko": "오일 앤 워터",
        "en": "Oil and Water",
        "aliases": ["oil and water", "red and black separation"],
        "effect": "transposition",
        "desc": "빨간 카드 3장 + 검정 카드 3장을 교차로 배열, 손짓 한 번에 색이 자연 분리. 'Oil and water do not mix'.",
        "beats": [
            ("show_mixed", "빨·검 교차 packet"),
            ("magic_moment", "손짓·tap"),
            ("reveal_separate", "빨강·검정 분리 확인"),
            ("repeat", "여러 phase 반복"),
        ],
        "techniques": ["packet_count", "double_lift", "elmsley_count"],
        "visual_cues": ["빨·검 packet", "phase 반복"],
        "query": "oil and water card trick tutorial",
    },
    "reset": {
        "ko": "리셋",
        "en": "Reset",
        "aliases": ["reset", "paul harris reset"],
        "effect": "transposition",
        "desc": "Paul Harris. 4 에이스와 4 킹이 서로 위치를 반복적으로 바꿈. 마지막에 'reset'으로 원위치.",
        "beats": [
            ("show_aces_kings", "에이스 4장 + 킹 4장"),
            ("repeat_swap", "여러 phase 동안 위치 swap 반복"),
            ("final_reset", "원위치 복귀"),
        ],
        "techniques": ["elmsley_count", "card_switch", "double_lift"],
        "visual_cues": ["8장 packet", "phase 여러 번"],
        "query": "reset paul harris tutorial",
    },
    "slow_motion_aces": {
        "ko": "슬로우 모션 에이스",
        "en": "Slow Motion Aces",
        "aliases": ["slow motion aces", "vernon slow motion"],
        "effect": "production",
        "desc": "Vernon. 4 에이스가 4 묶음에 흩어졌다가 천천히(슬로우모션) 한 묶음으로 이동.",
        "beats": [
            ("setup", "4 묶음에 에이스 한 장씩"),
            ("slow_transit", "에이스가 한 묶음으로 천천히 이동"),
            ("reveal", "한 묶음에 4 에이스"),
        ],
        "techniques": ["card_palm", "double_lift", "false_count"],
        "visual_cues": ["4 묶음", "느린 시각 효과"],
        "query": "slow motion aces vernon tutorial",
    },
    "collectors": {
        "ko": "콜렉터스",
        "en": "Collectors",
        "aliases": ["collectors", "roy walton collectors"],
        "effect": "production",
        "desc": "Roy Walton. 4 잭(collectors)이 덱 어딘가에 사라진 3 관객 카드를 모아 가짐. 마지막에 4 잭 사이에 3 관객 카드.",
        "beats": [
            ("3_selections", "관객 카드 3장"),
            ("vanish", "관객 카드 사라짐"),
            ("show_4_jacks", "4 잭 보여줌"),
            ("reveal_sandwich", "4 잭 사이에 3 관객 카드"),
        ],
        "techniques": ["card_palm", "side_steal", "double_lift"],
        "visual_cues": ["3 관객 카드", "4 잭 사이 sandwich"],
        "query": "collectors roy walton tutorial",
    },
    "asher_twist": {
        "ko": "애셔 트위스트",
        "en": "Asher Twist",
        "aliases": ["asher twist", "lee asher twist"],
        "effect": "transformation",
        "desc": "Lee Asher. 손바닥 위 4장이 trick handling으로 한 장씩 face-up.",
        "beats": [
            ("show_4_face_down", "4장 face down"),
            ("twist_each", "한 장씩 twist"),
            ("all_face_up", "모두 face up"),
        ],
        "techniques": ["asher_twist", "double_lift"],
        "visual_cues": ["4장 packet", "twist 동작 반복"],
        "query": "asher twist lee asher tutorial",
    },
    "twisting_revelations": {
        "ko": "트위스팅 리벨레이션",
        "en": "Twisting Revelations",
        "aliases": ["twisting revelations"],
        "effect": "transformation",
        "desc": "Twisting Aces의 확장. 4 카드가 매 phase 다른 카드로 변함.",
        "beats": [
            ("show_4", "4장 packet"),
            ("twist_phase", "phase별 변환"),
            ("final_set", "최종적으로 모두 다른 카드"),
        ],
        "techniques": ["elmsley_count", "jordan_count"],
        "visual_cues": ["4장 packet", "phase별 변환"],
        "query": "twisting revelations card tutorial",
    },

    # ===== Misc 명품 루틴 =====
    "tagged": {
        "ko": "태그드",
        "en": "TAGGED",
        "aliases": ["tagged", "rick lax tagged"],
        "effect": "prediction",
        "desc": "Rick Lax. 관객이 자유 선택한 카드의 정보가 미리 받은 사진/이미지에 적혀 있음. 디지털 영상 기반 mental.",
        "beats": [
            ("show_prediction_image", "예측 이미지 미리 보여줌"),
            ("audience_choice", "관객 자유 선택"),
            ("reveal", "선택 = 예측 이미지 내용"),
        ],
        "techniques": ["multiple_outs", "instant_stooge"],
        "visual_cues": ["이미지/사진 prop"],
        "query": "tagged rick lax card trick",
    },
    "hofzinser_problem": {
        "ko": "호프진저의 에이스 문제",
        "en": "Hofzinser Ace Problem",
        "aliases": ["hofzinser ace problem", "hofzinser problem"],
        "effect": "transposition",
        "desc": "Hofzinser. 관객 카드(예: 5 of hearts)가 4 에이스 중 같은 슈트 에이스(Ace of Hearts)와 위치 교환.",
        "beats": [
            ("selection", "관객 카드"),
            ("show_4_aces", "4 에이스"),
            ("magic_moment", "손짓"),
            ("transposition_reveal", "관객 카드와 같은 슈트 에이스 위치 교환"),
        ],
        "techniques": ["card_switch", "double_lift", "cull"],
        "visual_cues": ["4 에이스 + 관객 카드", "슈트 매칭"],
        "query": "hofzinser ace problem tutorial",
    },
    "bizarre_twist": {
        "ko": "비자 트위스트",
        "en": "Bizarre Twist",
        "aliases": ["bizarre twist", "modern twisting"],
        "effect": "transformation",
        "desc": "Twisting Aces 현대 변형. 마지막에 4장이 관객이 부른 카드로 변함.",
        "beats": [
            ("show_4_aces", "4 에이스 packet"),
            ("twist_phases", "twist 여러 phase"),
            ("final_change", "최종에 관객 호명 카드로 변환"),
        ],
        "techniques": ["elmsley_count", "card_switch"],
        "visual_cues": ["4장 packet", "최종 변환"],
        "query": "bizarre twist card trick tutorial",
    },
    "spectator_cuts_aces": {
        "ko": "관객이 에이스를 cut",
        "en": "Spectator Cuts to the Aces",
        "aliases": ["spectator cuts aces", "spectator cuts to the aces"],
        "effect": "revelation",
        "desc": "관객이 자유롭게 4번 cut한 결과 4 묶음의 top이 모두 에이스. 마술사 손은 거의 안 닿음.",
        "beats": [
            ("audience_cuts_4x", "관객이 4번 cut"),
            ("flip_tops", "4 묶음 top 뒤집음"),
            ("reveal_4_aces", "모두 에이스"),
        ],
        "techniques": ["setup_stack", "crimp", "side_steal"],
        "visual_cues": ["관객 cut 4번", "4 묶음 top 에이스"],
        "query": "spectator cuts to the aces tutorial",
    },
    "color_changing_aces": {
        "ko": "컬러 체인징 에이스",
        "en": "Color Changing Aces",
        "aliases": ["color changing aces"],
        "effect": "transformation",
        "desc": "4 에이스가 face는 그대로지만 백 색이 한 장씩 다른 색으로 변함.",
        "beats": [
            ("show_4_aces", "4 에이스 face"),
            ("turn_over", "백 보여줌"),
            ("reveal_color", "백 색이 다른 색"),
        ],
        "techniques": ["double_lift", "double_face_card", "gimmick_card"],
        "visual_cues": ["에이스 4장", "백 색 변화"],
        "query": "color changing aces trick tutorial",
    },
    "triumph_modern": {
        "ko": "모던 트라이엄프",
        "en": "Modern Triumph",
        "aliases": ["modern triumph", "open triumph"],
        "effect": "restoration",
        "desc": "Triumph의 visible 변형. face up/down 섞기가 더 명확하게 보임, 손짓에 정렬.",
        "beats": [
            ("selection", "관객 카드"),
            ("visible_chaos_shuffle", "visible 혼란 셔플"),
            ("show_chaos", "혼란 상태 펼침"),
            ("magic", "정렬 — 한 장만 반대"),
        ],
        "techniques": ["triumph_shuffle", "false_shuffle"],
        "visual_cues": ["face up/down 명확 혼합", "최종 정렬"],
        "query": "modern triumph card trick tutorial",
    },
    "biddle_trick": {
        "ko": "비들 트릭",
        "en": "Biddle Trick",
        "aliases": ["biddle trick", "biddle move"],
        "effect": "vanish",
        "desc": "관객 카드를 5장 사이에 끼웠다가 다시 펼치니 그 카드만 사라짐. 카드는 다른 곳에서 발견.",
        "beats": [
            ("selection", "관객 카드"),
            ("place_in_5", "5장 사이에 끼움"),
            ("biddle_count", "5장 다시 세서 보여줌 — 4장만"),
            ("reveal", "카드는 마술사 손/주머니"),
        ],
        "techniques": ["biddle_move", "card_palm"],
        "visual_cues": ["5장 packet", "카드 카운트"],
        "query": "biddle trick card tutorial",
    },
    "any_card_at_any_number_classic": {
        "ko": "ACAAN 클래식",
        "en": "ACAAN Classic Version",
        "aliases": ["acaan classic", "classic acaan"],
        "effect": "prediction",
        "desc": "마술사가 미리 카드명과 숫자를 적은 봉투를 보여준 뒤, 관객이 그대로 발견. ACAAN의 단순 버전.",
        "beats": [
            ("show_envelope", "예측 봉투"),
            ("audience_count", "관객이 그 숫자만큼 카운트"),
            ("reveal", "그 위치 카드 = 예측 카드"),
        ],
        "techniques": ["card_force", "stacked_deck"],
        "visual_cues": ["봉투 예측", "카드 카운트"],
        "query": "acaan classic card prediction tutorial",
    },
    "magicians_choice": {
        "ko": "매지션스 초이스",
        "en": "Magician's Choice (Equivoque)",
        "aliases": ["magicians choice", "equivoque"],
        "effect": "revelation",
        "desc": "관객이 카드/물건을 자유 선택하는 듯하지만 마술사가 결과를 통제. 'eliminate' or 'use' 모호 표현.",
        "beats": [
            ("offer_choices", "여러 선택지 제시"),
            ("audience_pick", "관객 선택"),
            ("interpret_to_force", "선택을 force 카드로 해석"),
            ("reveal", "force 카드 reveal"),
        ],
        "techniques": ["equivoque", "verbal_force"],
        "visual_cues": ["여러 선택지", "결과적으로 force 카드"],
        "query": "magicians choice equivoque tutorial",
    },
    "card_warp": {
        "ko": "카드 워프",
        "en": "Card Warp",
        "aliases": ["card warp", "jerry andrus card warp"],
        "effect": "transformation",
        "desc": "Jerry Andrus. 두 카드 중 하나를 접어서 다른 카드를 통과시킬 때 마술적으로 face up/down 변화 + 위치 변환.",
        "beats": [
            ("show_two_cards", "두 카드 (한 장은 다른 색)"),
            ("fold_handling", "한 카드 접어서 다른 카드 통과"),
            ("reveal_warp", "위치·방향 변화"),
        ],
        "techniques": ["folded_card", "card_warp"],
        "visual_cues": ["접힌 카드", "두 카드 처리"],
        "query": "card warp jerry andrus tutorial",
    },
    "kiss_my_aces": {
        "ko": "키스 마이 에이스",
        "en": "Kiss My Aces",
        "aliases": ["kiss my aces"],
        "effect": "production",
        "desc": "4 에이스가 차례로 visible하게 마술사 손으로 produce. 다양한 flourish 결합.",
        "beats": [
            ("shuffle", "덱 셔플"),
            ("produce_4_aces", "4 에이스 차례 produce"),
        ],
        "techniques": ["cull", "side_steal", "card_palm"],
        "visual_cues": ["4 에이스 차례 produce"],
        "query": "kiss my aces card production tutorial",
    },
    "indecent": {
        "ko": "인디센트",
        "en": "Indecent",
        "aliases": ["indecent", "ellusionist indecent"],
        "effect": "transposition",
        "desc": "관객 두 명이 각자 카드 선택, 두 카드가 두 관객 손 사이에서 위치 교환.",
        "beats": [
            ("two_selections", "두 관객 카드"),
            ("hand_each", "각자 손에 카드"),
            ("magic_moment", "손짓"),
            ("reveal_swap", "위치 교환"),
        ],
        "techniques": ["card_switch", "card_palm"],
        "visual_cues": ["두 관객", "손-손 교환"],
        "query": "indecent card trick ellusionist",
    },
    "anniversary_waltz_modern": {
        "ko": "모던 애니버서리 왈츠",
        "en": "Modern Anniversary Waltz",
        "aliases": ["modern anniversary waltz", "fused"],
        "effect": "transformation",
        "desc": "관객 2명 카드가 한 카드로 fuse — 양면이 각각 한 카드. signed 변형 가능.",
        "beats": [
            ("two_signed", "두 관객 카드 서명"),
            ("magic_moment", "손짓·터치"),
            ("reveal_fused", "한 카드 양면이 각자의 서명"),
        ],
        "techniques": ["double_face_card", "card_switch", "duplicate_card"],
        "visual_cues": ["두 서명", "양면 카드 reveal"],
        "query": "anniversary waltz fused cards tutorial",
    },
    "vernon_substitute_trunk": {
        "ko": "Vernon's Substitute Trunk Card",
        "en": "Vernon's Substitute Trunk",
        "aliases": ["vernon substitute trunk", "substitute card"],
        "effect": "transposition",
        "desc": "Vernon. 두 카드 중 한 카드가 다른 위치(주머니/지갑)로 substitute.",
        "beats": [
            ("show_two", "두 카드"),
            ("magic", "손짓"),
            ("reveal_substitute", "한 카드가 다른 곳에서 발견"),
        ],
        "techniques": ["card_palm", "card_switch"],
        "visual_cues": ["두 카드", "교환·이동"],
        "query": "vernon substitute trunk card trick",
    },
    "double_card_change": {
        "ko": "더블 카드 체인지",
        "en": "Double Card Change",
        "aliases": ["double card change", "two card change"],
        "effect": "transformation",
        "desc": "두 카드가 동시에 다른 두 카드로 변함. 비주얼 효과.",
        "beats": [
            ("show_two", "두 카드"),
            ("simultaneous_change", "둘 다 동시 변환"),
        ],
        "techniques": ["double_lift", "card_switch", "color_change"],
        "visual_cues": ["두 카드", "동시 변화"],
        "query": "double card change tutorial",
    },
    "card_to_impossible_place": {
        "ko": "임파서블 플레이스",
        "en": "Card to Impossible Place",
        "aliases": ["card to impossible place", "card in impossible location"],
        "effect": "transposition",
        "desc": "사라진 관객 카드가 봉인된 박스/유리병/지퍼 백 등 도저히 들어갈 수 없는 곳에서 발견.",
        "beats": [
            ("selection", "관객 카드"),
            ("vanish", "사라짐"),
            ("reveal_impossible", "봉인 장소에서 발견"),
        ],
        "techniques": ["card_switch", "duplicate_card", "secret_load"],
        "visual_cues": ["봉인 prop", "불가능 위치 reveal"],
        "query": "card to impossible location tutorial",
    },
    "tiny_plunger": {
        "ko": "타이니 플런저",
        "en": "Tiny Plunger / Plunger Principle",
        "aliases": ["plunger card", "plunger principle"],
        "effect": "revelation",
        "desc": "덱에 두 카드 사이에 관객 카드 끼우면 그 카드가 덱 밖으로 push out.",
        "beats": [
            ("selection", "관객 카드"),
            ("sandwich_insert", "두 카드 사이 끼움"),
            ("push_out", "관객 카드만 push out"),
        ],
        "techniques": ["plunger_move"],
        "visual_cues": ["덱에서 카드 push out"],
        "query": "plunger card trick tutorial",
    },
    "any_card_at_any_position": {
        "ko": "원하는 위치의 원하는 카드",
        "en": "Card at Stated Position",
        "aliases": ["card at position"],
        "effect": "prediction",
        "desc": "관객이 위치 숫자만 부르면 그 위치 카드가 미리 정한 카드와 일치.",
        "beats": [
            ("show_prediction", "예측 카드 보여줌"),
            ("audience_position", "관객 위치 숫자"),
            ("count_reveal", "그 위치 = 예측"),
        ],
        "techniques": ["estimation", "stacked_deck"],
        "visual_cues": ["예측 + 위치 카운트"],
        "query": "card at any position prediction",
    },
    "magicians_choice_outs": {
        "ko": "멀티플 아웃츠 (Multiple Outs)",
        "en": "Multiple Outs Routine",
        "aliases": ["multiple outs", "outs routine"],
        "effect": "prediction",
        "desc": "마술사가 여러 다른 예측을 준비, 관객 선택에 따라 맞는 것을 'reveal'. 모든 선택에 outs.",
        "beats": [
            ("multiple_predictions", "여러 예측 prep"),
            ("audience_choice", "관객 선택"),
            ("reveal_matching", "선택에 맞는 예측 reveal"),
        ],
        "techniques": ["multiple_outs", "switch"],
        "visual_cues": ["여러 prop", "선택 후 매칭 reveal"],
        "query": "multiple outs prediction routine",
    },
    "card_at_dealt_number": {
        "ko": "딜한 숫자의 카드",
        "en": "Card at Dealt Number",
        "aliases": ["card at dealt number", "count to card"],
        "effect": "revelation",
        "desc": "관객이 자유롭게 한 번에 카운트한 횟수가 관객 카드 위치와 일치.",
        "beats": [
            ("selection", "관객 카드"),
            ("audience_count", "관객 자유 카운트"),
            ("reveal", "그 위치 카드 = 관객 카드"),
        ],
        "techniques": ["spelling_force", "math_force"],
        "visual_cues": ["관객 카운트", "위치 매칭"],
        "query": "card at counted number tutorial",
    },
    "psychic_stop": {
        "ko": "사이킥 스톱",
        "en": "Psychic Stop",
        "aliases": ["psychic stop", "stop deal"],
        "effect": "mental",
        "desc": "마술사가 dealing 중 관객이 'stop'하면 그 카드가 미리 적은 예측과 일치.",
        "beats": [
            ("prediction", "예측 미리 적음"),
            ("deal", "마술사 dealing"),
            ("audience_stops", "관객 'stop'"),
            ("reveal_match", "예측 = stop 카드"),
        ],
        "techniques": ["force", "second_deal", "estimation"],
        "visual_cues": ["dealing", "stop 후 매칭"],
        "query": "psychic stop card trick tutorial",
    },
    "card_through_table": {
        "ko": "카드 쓰루 테이블",
        "en": "Card Through Table",
        "aliases": ["card through table"],
        "effect": "penetration",
        "desc": "관객 카드가 테이블을 관통해 아래로 떨어지거나 마술사 손에 잡힘.",
        "beats": [
            ("selection", "관객 카드"),
            ("place_on_table", "카드 테이블에 놓음"),
            ("magic", "손짓·tap"),
            ("under_table", "테이블 아래에서 카드 produce"),
        ],
        "techniques": ["card_palm", "lapping"],
        "visual_cues": ["테이블 위·아래"],
        "query": "card through table tutorial",
    },
    "lift_the_curtain": {
        "ko": "리프트 더 커튼",
        "en": "Lift the Curtain",
        "aliases": ["lift the curtain", "behind curtain reveal"],
        "effect": "revelation",
        "desc": "마술사가 손수건/천을 들면 관객 카드가 그 아래에서 발견.",
        "beats": [
            ("selection", "관객 카드"),
            ("cover_with_silk", "천 덮음"),
            ("lift", "천 들어 올림"),
            ("reveal", "관객 카드"),
        ],
        "techniques": ["card_load", "silk_handling"],
        "visual_cues": ["천 prop"],
        "query": "card under handkerchief reveal",
    },
    "torn_corner_match": {
        "ko": "톤 코너 매치",
        "en": "Torn Corner Match",
        "aliases": ["torn corner", "corner match"],
        "effect": "coincidence",
        "desc": "관객 카드의 한 모서리를 찢어 주머니에. 카드 사라진 뒤 다른 곳에서 발견 — 찢긴 모서리가 정확히 매치.",
        "beats": [
            ("selection", "관객 카드"),
            ("tear_corner", "모서리 찢어 관객에게"),
            ("vanish", "카드 사라짐"),
            ("reveal_match", "다른 곳에서 발견 + 모서리 매치"),
        ],
        "techniques": ["duplicate_card", "card_switch"],
        "visual_cues": ["찢긴 모서리", "매치 검증"],
        "query": "torn corner card match tutorial",
    },
    "haunted_deck": {
        "ko": "헌티드 덱",
        "en": "Haunted Deck",
        "aliases": ["haunted deck", "haunted pack"],
        "effect": "ambitious",
        "desc": "덱이 스스로 움직여 관객 카드가 위로 cut됨. 보이지 않는 invisible thread.",
        "beats": [
            ("selection", "관객 카드"),
            ("deck_on_hand", "덱 손바닥 위"),
            ("self_cut", "덱이 스스로 cut"),
            ("reveal", "cut 위치 = 관객 카드"),
        ],
        "techniques": ["invisible_thread", "gimmick"],
        "visual_cues": ["덱이 자동 움직임", "scarf 없음"],
        "query": "haunted deck card trick tutorial",
    },
    "ace_assembly_visual": {
        "ko": "비주얼 에이스 어셈블리",
        "en": "Visual Ace Assembly",
        "aliases": ["visual ace assembly"],
        "effect": "transposition",
        "desc": "4 묶음의 에이스가 시각적으로 빠르게 한 묶음으로 모임. 빠른 visual.",
        "beats": [
            ("setup_4_packets", "4 묶음"),
            ("rapid_assembly", "빠른 visual transit"),
            ("reveal", "한 묶음에 4 에이스"),
        ],
        "techniques": ["packet_count", "double_lift"],
        "visual_cues": ["4 묶음", "빠른 시각 효과"],
        "query": "visual ace assembly tutorial",
    },
    "card_through_finger": {
        "ko": "카드 쓰루 핑거",
        "en": "Card Through Finger",
        "aliases": ["card through finger"],
        "effect": "penetration",
        "desc": "카드가 마술사 손가락을 관통하는 듯한 비주얼.",
        "beats": [
            ("show_card_finger", "카드와 손가락"),
            ("penetration_visual", "관통 visual"),
        ],
        "techniques": ["paddle_move", "secret_split"],
        "visual_cues": ["카드 + 손가락 관통 듯"],
        "query": "card through finger trick",
    },
    "find_the_lady": {
        "ko": "파인드 더 레이디",
        "en": "Find the Lady",
        "aliases": ["find the lady", "monte short"],
        "effect": "revelation",
        "desc": "3 카드 중 퀸 찾기. 관객이 매번 틀림. Three card monte의 한 form.",
        "beats": [
            ("show_3", "3 카드(퀸 + 2장 다른)"),
            ("mix", "위치 섞음"),
            ("guess", "관객 위치 짐작"),
            ("reveal_miss", "관객 짐작 틀림"),
        ],
        "techniques": ["monte_throw"],
        "visual_cues": ["3장만 사용"],
        "query": "find the lady three card monte",
    },
    "card_at_glance": {
        "ko": "카드 앳 글랜스",
        "en": "Card at a Glance",
        "aliases": ["card at a glance", "glance reveal"],
        "effect": "mental",
        "desc": "마술사가 덱을 한 번 fan으로 펼친 뒤 즉시 관객 카드를 알아냄. Glimpse 기법.",
        "beats": [
            ("selection", "관객 카드"),
            ("brief_glance", "마술사가 덱 fan glance"),
            ("reveal", "관객 카드 명시"),
        ],
        "techniques": ["glimpse", "key_card"],
        "visual_cues": ["덱 fan 잠깐"],
        "query": "card glance peek tutorial",
    },
    "out_to_lunch": {
        "ko": "아웃 투 런치",
        "en": "Out to Lunch",
        "aliases": ["out to lunch"],
        "effect": "transformation",
        "desc": "Gimmick card case 사용. 빈 카드들이 face 카드들로 변함.",
        "beats": [
            ("show_blanks", "빈 카드들"),
            ("place_in_case", "case에 넣음"),
            ("reveal", "case 열면 face 카드"),
        ],
        "techniques": ["out_to_lunch_gimmick"],
        "visual_cues": ["card case 사용", "빈 → face 변화"],
        "query": "out to lunch card trick tutorial",
    },
    "professor_nightmare": {
        "ko": "프로페서 나이트메어",
        "en": "Professor's Nightmare",
        "aliases": ["professors nightmare", "equal unequal"],
        "effect": "transformation",
        "desc": "본래는 로프 트릭이지만 카드 변형도 존재. 3장(서로 다른 길이 비유)이 같아지고 다시 다름.",
        "beats": [
            ("show_3_different", "3장 (서로 구별)"),
            ("equalize", "동일하게"),
            ("restore_different", "다시 구별 상태"),
        ],
        "techniques": ["sleight_of_hand"],
        "visual_cues": ["3장 packet", "동일·구별 반복"],
        "query": "professor's nightmare card",
    },
    "predicted_position": {
        "ko": "예측된 위치",
        "en": "Predicted Position",
        "aliases": ["predicted position"],
        "effect": "prediction",
        "desc": "마술사가 미리 카드 위치(예: 17번째)를 예측. 관객 자유 동작 후 그 위치 카드가 관객 카드와 일치.",
        "beats": [
            ("write_position", "위치 숫자 예측 적음"),
            ("audience_action", "관객 자유 동작"),
            ("count_reveal", "예측 위치 = 관객 카드"),
        ],
        "techniques": ["math_force", "stacked_deck"],
        "visual_cues": ["위치 카운트", "예측 매칭"],
        "query": "predicted position card trick",
    },
    "card_to_glass": {
        "ko": "카드 투 글래스",
        "en": "Card to Sealed Glass",
        "aliases": ["card to glass", "card in sealed glass"],
        "effect": "transposition",
        "desc": "관객 카드가 사라져 봉인된 유리병/잔 안에서 발견.",
        "beats": [
            ("selection", "관객 카드"),
            ("vanish", "카드 사라짐"),
            ("glass_reveal", "유리병 안에서 카드"),
        ],
        "techniques": ["secret_load", "card_palm"],
        "visual_cues": ["유리 prop", "병 안 카드"],
        "query": "card to sealed glass tutorial",
    },
    "matrix": {
        "ko": "매트릭스",
        "en": "Matrix",
        "aliases": ["matrix", "coin matrix"],
        "effect": "production",
        "desc": "Al Schneider. 본래는 동전 트릭이지만 카드 응용 존재 — 4개 코너에 한 장씩 두었다가 한 코너로 모음.",
        "beats": [
            ("4_corners", "4 코너에 카드(또는 동전)"),
            ("magic_moment", "tap each corner"),
            ("assemble", "한 코너로 모임"),
        ],
        "techniques": ["card_switch", "load"],
        "visual_cues": ["4 코너 배치", "차례 이동"],
        "query": "matrix card coin assembly tutorial",
    },
    "shuffled_to_top": {
        "ko": "셔플드 투 탑",
        "en": "Shuffled to Top",
        "aliases": ["shuffled to top"],
        "effect": "ambitious",
        "desc": "관객 카드를 덱 가운데 묻은 뒤 관객이 직접 셔플해도 카드가 맨 위에 있음.",
        "beats": [
            ("selection", "관객 카드"),
            ("audience_shuffle", "관객 셔플"),
            ("reveal_top", "맨 위에 그 카드"),
        ],
        "techniques": ["false_shuffle_audience", "key_card"],
        "visual_cues": ["관객이 직접 섞음", "맨 위 카드"],
        "query": "shuffled to top card trick",
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
