# magic-trick-analyzer 🃏🪙

카드/동전 마술 **영상**을 넣으면, **손 + 카드/동전 객체**를 함께 추적해 **"수상한 순간"**을 타임라인으로 짚어주고, OpenAI 비전 에이전트가 기법을 추론·설명하는 분석 도구.

> ⚠️ **솔직한 한계**
> 마술은 사람의 지각을 속이도록 설계돼 있어, 영상 한 개로 비밀을 **확정**하는 것은 불가능합니다.
> 이 도구는 비밀을 단정하지 않고, **통계적으로 의심스러운 구간**(손이 사라짐·두 손이 닿음·급동작·주먹쥠)을
> 찾아 사람이 다시 돌려보도록 돕는 **보조 도구**입니다.

## 주요 기능
- 🖐️ **손 신호 탐지** — MediaPipe 손 추적으로 팜·전달·급동작·주먹쥠을 점수화 → 비최대억제(NMS)로 또렷한 순간만
- 🎯 **객체 신호 탐지** — YOLO 카드 검출기 + HoughCircles 동전 검출로 객체 등장/소실 전이를 잡음 (`OBJECT_CHANGE` 신호)
- 🃏🪙 **카드/동전 자동 판별** — `--mode auto`: 영상 프레임을 보고 종류를 분류해 해당 모드로 분석
- 🤖 **도구 호출 에이전트** — LangGraph ReAct + OpenAI `gpt-4o`. 의심 순간을 직접 들여다보고(비전) 기법을 추론
- 🎓 **기법 용어집(33종) + 참고 영상** — 작동 원리·관찰 단서·튜토리얼 링크 제공
- 🔍 **예시 라이브러리 매칭(few-shot)** — 손 궤적 시그니처로 기법 유사도 비교
- 🏷️ **Active learning 라벨링** — 웹 UI에서 의심 시점·임의 시점에 사용자가 직접 기법을 붙여 라이브러리 누적
- 🌐 **웹 UI(Flask)** — URL/업로드 → 마킹 영상·타임라인·기법 설명·매칭·라벨링을 한 화면에
- 📊 **LangSmith 추적**(선택) · 🎬 **입력**: 로컬 파일 또는 YouTube URL(yt-dlp 자동 다운로드)

## 무엇을 탐지하나

손 키네마틱 신호 5종 + 객체 변화 신호 1종, **총 6종**을 가중합해 점수화합니다.

| 신호 | 출처 | 의미(가능성) |
|------|------|------|
| `VANISH` | 손 | 보이던 손이 사라짐 — 팜으로 숨김 / 주머니로 디치 |
| `BORDER` | 손 | 손이 화면 가장자리로 빠짐 — 랩/주머니/오프스크린 이동 |
| `CONTACT` | 손 | 두 손이 맞닿음 — 몰래 전달 / 로드 |
| `FAST` | 손 | 손이 순간적으로 빠르게 움직임 — 패스/슬레이트 |
| `GRAB` | 손 | 펼친 손이 갑자기 주먹 — 코인/카드 팜 |
| `OBJECT_CHANGE` | 객체 | 카드/동전 자체의 등장·소실·면적 급감 — 매끄러운 슬레이트 보완 |

손 키네마틱은 매끄러운 슬레이트(잘 된 팜·부드러운 컨트롤)를 놓치기 때문에 객체 검출을
독립 신호로 추가했습니다. 카드는 YOLOv8 52-class detector
(`mustafakemal0146/playing-cards-yolov8`, HF에서 lazy 다운로드), 동전은 `cv2.HoughCircles`
휴리스틱을 사용합니다.

`--mode card` 와 `--mode coin` 은 위 신호의 가중치가 다릅니다. **`--mode auto`** 를 주면
영상 프레임을 OpenAI 비전으로 분석해 카드/동전을 **자동 판별**한 뒤 그 모드로 분석합니다
(웹 UI의 "🔮 자동 감지" 옵션, `OPENAI_API_KEY` 필요).

## 설치

Python 3.12 권장 (MediaPipe 호환).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 사용법

입력은 **로컬 파일 경로** 또는 **YouTube URL** 둘 다 됩니다. URL이면 yt-dlp로
자동 다운로드(360p, ffmpeg 불필요)한 뒤 분석하며, 같은 영상은 `downloads/`에 캐시됩니다.

```powershell
# 기본 분석 (리포트만)
python main.py trick.mp4 --mode card

# YouTube URL 직접 분석
python main.py "https://youtu.be/VIDEO_ID" --mode coin --llm

# 손 관절+의심구간이 표시된 영상과 의심 프레임 이미지까지 저장
python main.py trick.mp4 --mode coin --annotate --save-frames 5

# 더 민감하게 (더 많은 구간 잡기)
python main.py trick.mp4 --mode card --score-thresh 0.4
```

### 옵션
| 옵션 | 설명 |
|------|------|
| `--mode {card,coin,auto}` | 마술 종류 (기본 card). `auto`면 영상 보고 자동 판별(`OPENAI_API_KEY` 필요) |
| `--out DIR` | 결과 폴더 (기본 `out`) |
| `--annotate` | 손 관절+배너 표시된 `annotated.webm` 저장 (브라우저 재생 가능) |
| `--save-frames K` | 상위 K개 의심 순간 정점 프레임 이미지 저장 |
| `--stride N` | N프레임마다 1장만 분석(속도용) |
| `--score-thresh F` | 봉우리 채택 임계값 (기본 0.9, 낮출수록 많이 잡음) |
| `--min-gap S` | 의심 순간 사이 최소 간격 초 (기본 1.2) |
| `--window S` | 봉우리 앞뒤로 구간에 포함할 초 (기본 0.8) |
| `--max-results N` | 최대 의심 순간 개수 (기본: 제한 없음) |
| `--llm` | 상위 의심 구간을 **OpenAI 비전 모델로 추론** (`OPENAI_API_KEY` 필요) |
| `--llm-model M` | OpenAI 비전 모델명 (기본 `gpt-4o`) |
| `--llm-top K` | LLM으로 추론할 상위 구간 개수 (기본 5) |

## 좋은 결과를 위한 팁 (실측 기반)
- **클로즈업 + 단독 출연** 영상에서 가장 잘 작동한다. 손이 화면을 충분히 채워야
  MediaPipe가 두 손을 안정적으로 추적한다. (카드 튜토리얼/테이블 클로즈업이 이상적)
- **무대 와이드샷 + 관객 동석**은 부정확하다 — 엉뚱한 사람 손을 잡거나 작은 손을 놓친다.
- 너무 적게/많이 잡히면 `--score-thresh`로 조절 (기본 0.9 ≈ 상위 5% 순간).
- **해상도는 거의 무관**하다. MediaPipe 손 검출기가 프레임을 ~192px로 다운스케일하므로,
  같은 영상을 360p↔1080p로 비교해도 손 검출률은 사실상 동일(80.5%↔79.9%)하고 1080p는
  24% 느리기만 했다. 화질보다 **프레이밍(클로즈업)**이 결과를 좌우한다.

## 출력물 (`out/`)
- `report.txt` — 사람이 읽는 타임라인 리포트
- `report.json` — 구간/점수/신호 상세 (다른 도구·LLM에 넘기기 좋음)
- `annotated.webm` — (옵션) 관절·의심구간 표시 영상 (VP8, 브라우저 재생 가능)
- `suspect_NN_*.jpg` — (옵션) 의심 구간 정점 프레임
- `llm.txt` / `llm.json` — (옵션 `--llm`) 구간별 OpenAI 비전 추론 결과

## 동작 원리
1. **손 추적 + 객체 검출(병행)** — MediaPipe Tasks(HandLandmarker, VIDEO 모드)로 프레임마다
   손 21개 관절을 정규화 좌표로 추출. 동시에 stride=3 간격으로 YOLO 카드 detector +
   HoughCircles 동전 검출을 돌려 객체 bbox/원 리스트 수집
2. **신호 계산** — 손 개수 변화(VANISH) / 가장자리 *진입 이벤트*(BORDER) / 두 손 거리(CONTACT)
   / 이동속도(FAST) / 펼침정도 급감(GRAB) / 객체 count·면적 전이(OBJECT_CHANGE)
3. **점수화·봉우리 검출** — 모드별 가중합 → 스무딩 → **점수 봉우리를 비최대 억제(NMS)**로
   집어 `봉우리 ±window` 창으로 보고. 클로즈업은 신호가 상시 켜지므로 '임계 초과 구간'을
   잡으면 수십 초 덩어리가 된다 — 봉우리만 집어 또렷한 순간을 준다.

## 웹 UI
브라우저에서 URL 입력/파일 업로드 → 옵션 선택 → 결과(마킹 영상·타임라인·의심
프레임·AI 설명)를 한 화면에서 봅니다. 검증된 CLI를 백엔드로 그대로 재사용합니다.

```powershell
python webapp/app.py
# 브라우저에서 http://127.0.0.1:5000 접속
```

분석은 백그라운드로 돌아가고 화면이 진행상태를 폴링합니다. `AI 설명`을 켜려면
`OPENAI_API_KEY`(.env)가 필요합니다. 작업 상태는 `webapp/jobs/<id>/job.json`에
영속화되어 서버 재시작 후에도 결과 페이지가 복원됩니다.

### Active learning 라벨링 (웹 UI)
결과 페이지에서 사용자가 직접 기법 라벨을 붙여 라이브러리를 누적할 수 있습니다.
도구의 데이터 vs 사람의 판단 간 ground truth gap을 메우는 핵심 통로입니다.

- **의심 시점 라벨링** — 각 의심 구간 카드에 접기/펼치기 라벨러. 펼쳐서 기법
  드롭다운(33종) 선택 또는 자유 입력 → "라벨 등록". 등록 성공 시 summary가
  "✓ 라벨 등록됨: <기법명>"로 갱신되고 라이브러리에 즉시 추가
- **임의 시점 라벨링** — 탐지가 놓친 시점을 위한 카드. 시각(`01:23.45` 또는
  `83.45`) + 기법 직접 지정. "▶ 현재 영상 시점" 버튼으로 마킹 영상의 currentTime
  자동 입력

등록된 라벨은 `library/signatures.json`에 `source=webapp:<job_id>@<sec>`로
출처와 함께 저장됩니다. 다음 분석부터 `match_technique` 도구가 새 시그니처와
코사인 유사도를 비교해 매칭에 반영합니다.

## 에이전트 분석 (LangGraph ReAct + 도구 호출 + LangSmith, 옵션)
`--llm`을 켜면 고정 파이프라인이 아니라 **도구를 스스로 호출하는 ReAct 에이전트**
(`langgraph.prebuilt.create_react_agent`, `gpt-4o`)가 분석합니다. 에이전트가 쓰는 도구:

- **`list_suspect_moments()`** — 자동 탐지된 의심 순간(시각/신호/점수) 목록
- **`inspect_moment(time_sec)`** — 그 시각의 프레임(직전/정점/직후)을 **비전으로 들여다보는**
  도구. 에이전트가 *어디를 볼지 직접 결정*해 호출합니다.
- **`explain_technique(기법명)`** — 의심 기법의 **자세한 작동 원리 + 참고 튜토리얼 영상 링크**를
  반환(용어집 `techniques.py`). 결과는 "🎓 기법 설명" 카드로 표시됩니다.

- **`match_technique(time_sec)`** — 그 순간의 손 궤적을 **기법 예시 라이브러리와 비교**해
  가장 닮은 기법과 유사도를 반환(데이터 기반 단서, 아래 참고).

에이전트는 목록 확인 → 의심 순간 inspect/match → 기법 explain → 종합 결론(전체 트릭 추정)을 냅니다.
기법 설명에는 작동 원리·관찰 단서·참고 영상이 포함되고, 분석한 의심 프레임(이미지)도 함께 봅니다.

### 예시 라이브러리 (few-shot 매칭)
대규모 학습 대신, **기법 시연 클립의 손 궤적 '시그니처'를 모아두고** 의심 순간을
코사인 유사도로 비교합니다(`library.py`). 라이브러리는 점진적으로 키울 수 있습니다:

```powershell
# 자동: 기법 튜토리얼 검색·다운로드 → 최상위 의심 순간을 그 기법 예시로 등록
python scripts/build_library.py --auto "double lift" "french drop" "classic palm"
# 수동(가장 정확): 검증된 영상/시각을 직접 등록
python scripts/build_library.py --technique "프렌치 드롭" --video coin.mp4 --time 12.3 --mode coin
```

시그니처는 손목 기준 정규화 + 고정 길이 리샘플(위치·크기·속도차 흡수)로 `library/signatures.json`에
저장됩니다. 라이브러리가 비어 있으면 `match_technique`는 비활성(나머지 분석은 정상).
매칭은 모드(카드/동전)로 후보를 걸러, 카드 영상이 동전 기법에 매칭되는 교차 오류를 막습니다.
한계: 튜토리얼 시연 ≠ 실제 공연, 작은 라이브러리는 커버리지 제한 — 검증된 예시를 늘릴수록 좋아집니다.

#### 자동 라벨링 (해설 음성 STT 기반)
'마술 비밀 공개' 류 채널은 해설자가 음성으로 기법 이름을 말합니다. 그 음성 자체가
라벨이므로 사람 라벨링 없이 자동 누적이 가능합니다.

```powershell
# YouTube URL(들) → 자동 다운로드 → Whisper 전사 → 기법 언급 검출 → 시그니처 등록
python scripts/auto_label.py "https://youtu.be/URL1" "https://youtu.be/URL2"

# 후보만 보고 등록은 건너뛰기(검토용)
python scripts/auto_label.py "https://youtu.be/URL" --dry-run

# 해설자가 시연 직후에 말하는 경향이면 음수 offset으로 시점 보정
python scripts/auto_label.py "https://youtu.be/URL" --offset -2.0
```

faster-whisper로 영어 자막을 추출하고 `techniques.lookup()`의 별칭 매칭으로 기법명을
찾아 부정문/도입부 문장을 자동 제외합니다. 등록된 라벨은 `source=stt:<id>@<sec>|<문장>`로
출처가 기록됩니다.

### LangSmith 추적 (선택)
에이전트 실행과 **도구 호출이 LangSmith 트레이스에 그대로 기록**됩니다(키가 있을 때).
이미지는 inspect 도구의 비전 서브호출에만 들어가 메인 트레이스가 가볍습니다. 키는 `.env`로 줍니다
(`.env.example` 참고):

```dotenv
OPENAI_API_KEY=sk-...           # 필수
LANGSMITH_API_KEY=lsv2_...      # 선택(추적). 있으면 자동으로 추적 on + 프로젝트 magic-analyzer
```

```powershell
python main.py samples/coin_cu.mp4 --mode coin --llm --llm-top 5
```

결과는 콘솔 + `out/llm.txt` + `out/llm.json`(구간별 추론 + **전체 트릭 추정**)에 저장됩니다.
OpenAI 키가 없으면 분석은 정상 진행되고 에이전트 단계만 건너뜁니다.

> 참고: LLM은 Anthropic이 아니라 **OpenAI**(`langchain-openai`)를 사용합니다.

## 개발용 튜닝
`scripts/tune.py` 는 손 추적 결과를 pickle로 캐시한 뒤 detect 파라미터만 바꿔가며
빠르게 비교한다(영상당 추적 1회). 점수 분포(`--hist`)도 출력한다.
```powershell
python scripts/tune.py samples/coin.mp4 --mode coin --score-thresh 1.0 --hist
```

## 로드맵
- [x] 의심 프레임을 멀티모달 LLM(OpenAI)에 넘겨 "무슨 일이 일어났는지" 추론
- [x] 도구 호출 에이전트(LangGraph ReAct) + LangSmith 추적
- [x] 기법 용어집 + 예시 라이브러리 few-shot 매칭
- [x] 카드/동전 자동 판별(`--mode auto`)
- [x] 간단한 웹 UI (Flask, `webapp/app.py`)
- [x] **동전/카드 객체 자체 추적** — YOLO 카드 detector + HoughCircles로 `OBJECT_CHANGE` 신호 추가
- [x] **웹 UI active learning 라벨링** — 의심·임의 시점에 사용자가 기법 라벨링 → 라이브러리 누적
- [ ] 매칭 정밀도 개선(DTW 등) · 라벨 데이터가 쌓이면 임계값 자동 튜닝/분류기 학습
- [ ] 카드 detector의 face-down 약점 보완(rectangular-back 검출) · 동전 검출 정확도 향상

## 프로젝트 구조
```
magic_analyzer/   video·fetch(입력) · hands·assets(추적) · detect(탐지) · report(출력)
                  agent·classify(에이전트/자동판별) · objects(YOLO 카드+Hough 동전)
                  techniques·library(지식/매칭) · cli(파이프라인)
scripts/          build_library.py(라이브러리 구축) · tune.py(파라미터 튜닝)
webapp/           app.py(Flask · /label · /techniques · 작업 영속화) · templates/index.html
library/          signatures.json(기법 시그니처 — auto 생성 + 사용자 라벨링 누적)
```

## 면책
교육·연습 복기(자기 분석) 목적의 보조 도구입니다. 타인의 공연 비밀을 단정·폭로하는 용도가 아닙니다.
