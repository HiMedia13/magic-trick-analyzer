"""magic-analyzer 웹 UI (Flask).

브라우저에서 YouTube URL/파일 업로드 → 모드·옵션 선택 → 분석 결과(타임라인,
의심 프레임, 마킹 영상, AI 설명)를 한 화면에 보여준다.

분석은 오래 걸리므로 검증된 CLI(main.py)를 백그라운드 서브프로세스로 돌리고,
브라우저가 /status/<job>을 폴링한다. 결과물은 job 디렉토리의 파일을 그대로 읽는다.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from flask import (Flask, abort, jsonify, render_template, request,
                   send_from_directory)
from werkzeug.utils import secure_filename

# 라벨링은 이 앱 안에서 직접 시그니처 추출 후 등록(서브프로세스 우회).
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from magic_analyzer.fetch import resolve_video_input  # noqa: E402
from magic_analyzer.library import save_entry, signature_from_video  # noqa: E402
from magic_analyzer.techniques import TECHNIQUES, lookup  # noqa: E402

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent
JOBS_DIR = BASE / "jobs"
MAIN = PROJECT / "main.py"
PYTHON = sys.executable  # 이 앱을 돌리는 venv 파이썬 그대로 사용

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 업로드 최대 500MB
_jobs: dict[str, dict] = {}  # job_id -> {proc, dir, modes, idx, video_arg, ...}
_status_lock = threading.Lock()  # /status 시퀀서 전이를 원자화(중복 spawn 방지)

MODES = ("card", "coin")          # 선택 가능한 실제 모드
SEG_OK = ("card", "coin", "auto")  # 결과 폴더(세그먼트)로 허용되는 이름


def _save_job_state(job_id: str, job: dict) -> None:
    """프로세스 객체를 제외한 job 메타데이터를 디스크에 저장. 서버 재시작 후에도
    /status가 결과를 찾아볼 수 있게 한다(인메모리 _jobs가 휘발해도 영상/리포트는
    jobs/<id>/<mode>/ 아래에 남는다)."""
    meta = {"modes": job["modes"], "idx": job["idx"],
            "video_arg": job["video_arg"],
            "annotate": job["annotate"], "llm": job["llm"]}
    (job["dir"] / "job.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def _load_job_state(job_id: str) -> dict | None:
    """디스크에서 job 메타데이터를 복원. proc은 None — 재시작 후엔 살아있는
    프로세스가 없으므로 결과 파일 존재 여부로만 done/failed를 판정한다."""
    job_dir = JOBS_DIR / job_id
    meta_path = job_dir / "job.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return {"dir": job_dir, "modes": meta["modes"], "idx": meta["idx"],
            "video_arg": meta["video_arg"], "annotate": meta["annotate"],
            "llm": meta["llm"], "proc": None}


@app.route("/")
def index():
    return render_template("index.html")


def _spawn(job: dict, mode: str) -> subprocess.Popen:
    """선택된 모드 하나에 대해 CLI(main.py)를 jobs/<id>/<mode>/ 로 출력하며 실행."""
    out_dir = job["dir"] / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PYTHON, str(MAIN), job["video_arg"], "--mode", mode,
           "--out", str(out_dir), "--save-frames", "6", "--max-results", "12"]
    if job["annotate"]:
        cmd.append("--annotate")
    if job["llm"]:
        cmd += ["--llm", "--llm-top", "6"]
    # 자식이 핸들을 상속하므로, Popen 직후 부모 쪽 핸들은 닫아 누수를 막는다.
    log = open(out_dir / "log.txt", "w", encoding="utf-8")
    try:
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                cwd=str(PROJECT))
    finally:
        log.close()


@app.route("/analyze", methods=["POST"])
def analyze():
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # 모드 선택: auto가 있으면 단일 auto 실행(영상 보고 판별), 아니면 card/coin 다중.
    sel = [m for m in request.form.getlist("mode") if m in SEG_OK]
    sel = list(dict.fromkeys(sel))
    modes = ["auto"] if "auto" in sel else ([m for m in sel if m in MODES] or ["card"])

    url = (request.form.get("url") or "").strip()
    if url:
        video_arg = url
    else:
        f = request.files.get("file")
        if not f or not f.filename:
            return jsonify({"error": "YouTube URL을 입력하거나 영상 파일을 올려주세요."}), 400
        dest = job_dir / ("input_" + secure_filename(f.filename))
        f.save(dest)
        video_arg = str(dest)

    job = {
        "dir": job_dir, "modes": modes, "idx": 0, "video_arg": video_arg,
        "annotate": request.form.get("annotate") == "on",
        "llm": request.form.get("llm") == "on",
    }
    # 모드는 순차 실행(같은 URL 동시 다운로드 충돌·CV 리소스 경쟁 방지).
    # /status가 한 모드 완료를 감지하면 다음 모드를 띄우는 시퀀서 역할을 한다.
    job["proc"] = _spawn(job, modes[0])
    _jobs[job_id] = job
    _save_job_state(job_id, job)
    return jsonify({"job_id": job_id})


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _collect(mode_dir: Path, seg: str) -> dict:
    """한 폴더(seg)의 결과를 묶는다. seg는 폴더명(auto 가능), mode는 판별된 실제 종류."""
    report = _read_json(mode_dir / "report.json") or {}
    resolved = report.get("mode", seg if seg in MODES else "card")
    llm = _read_json(mode_dir / "llm.json")
    # LLM 추론을 세그먼트에 붙임. agent.py가 inspect_moment의 time_sec을
    # canonical peak_sec(소수 둘째 자리)으로 스냅하므로 정확 일치를 우선 시도하고,
    # 만약을 위한 폴백으로 ±0.1초 nearest peak를 둔다.
    results = (llm or {}).get("results", [])
    for s in report.get("segments", []):  # 'seg' 파라미터와 충돌 금지(별도 이름)
        ps = s.get("peak_sec")
        if ps is None or not results:
            continue
        best = min(results, key=lambda r: abs(r.get("peak_sec", 1e9) - ps))
        if abs(best.get("peak_sec", 1e9) - ps) <= 0.1:
            s["inference"] = best.get("inference", "")
    frames = sorted(p.name for p in mode_dir.glob("suspect_*.jpg"))
    return {
        "mode": resolved,   # 판별된 실제 종류(라벨용)
        "seg": seg,         # 결과 폴더명(파일 URL용, auto 가능)
        "report": report,
        "summary": (llm or {}).get("summary"),     # 에이전트 전체 트릭 추정
        "techniques": (llm or {}).get("techniques", []),  # 기법 설명 + 참고 영상
        "matches": (llm or {}).get("matches", []),  # 라이브러리 데이터 기반 매칭
        "frames": frames,
        "video": (mode_dir / "annotated.webm").exists(),
    }


@app.route("/status/<job_id>")
def status(job_id):
    # 인메모리 _jobs에 없으면 디스크에서 복원(서버 재시작 후 복귀 케이스).
    job = _jobs.get(job_id) or _load_job_state(job_id)
    if not job:
        abort(404)

    # 전이(다음 모드 spawn)는 원자적이어야 한다 — 동시 폴링이 같은 전이를 두 번
    # 수행해 같은 URL을 동시 다운로드하면 yt-dlp 'unable to rename file'이 난다.
    with _status_lock:
        modes, idx = job["modes"], job["idx"]
        cur_mode = modes[idx]

        log_tail = ""
        log_path = job["dir"] / cur_mode / "log.txt"
        if log_path.exists():
            log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]

        progress = f"{idx + 1}/{len(modes)}"

        # 디스크 복원 케이스(proc=None): 살아있는 프로세스가 없으므로 모든 모드의
        # report.json이 있으면 done, 아니면 failed로만 응답(재실행은 사용자가 결정).
        if job.get("proc") is None:
            all_done = all((job["dir"] / m / "report.json").exists() for m in modes)
            if all_done:
                results = [_collect(job["dir"] / m, m) for m in modes]
                return jsonify({"status": "done", "job_id": job_id,
                                "results": results, "log": log_tail})
            return jsonify({"status": "failed", "mode": cur_mode,
                            "progress": progress, "log": log_tail})

        rc = job["proc"].poll()
        if rc is None:  # 현재 모드 진행 중
            return jsonify({"status": "running", "mode": cur_mode,
                            "progress": progress, "log": log_tail})

        if rc != 0:  # 현재 모드 실패
            return jsonify({"status": "failed", "mode": cur_mode,
                            "progress": progress, "log": log_tail})

        # 현재 모드 완료 → 다음 모드가 남았으면 띄우고 계속 running
        if idx + 1 < len(modes):
            job["idx"] = idx + 1
            job["proc"] = _spawn(job, modes[idx + 1])
            _save_job_state(job_id, job)
            return jsonify({"status": "running", "mode": modes[idx + 1],
                            "progress": f"{idx + 2}/{len(modes)}", "log": log_tail})

        # 모든 모드 완료 → 모드별 결과 묶음 반환
        results = [_collect(job["dir"] / m, m) for m in modes]
        return jsonify({"status": "done", "job_id": job_id, "results": results,
                        "log": log_tail})


@app.route("/techniques")
def techniques_list():
    """라벨 UI 드롭다운용 — 용어집의 카드/동전 겸용 포함 기법 목록."""
    out = []
    for key, e in TECHNIQUES.items():
        out.append({"key": key, "ko": e["ko"], "en": e["en"], "type": e["type"]})
    # 한글 가나다 정렬
    out.sort(key=lambda x: x["ko"])
    return jsonify(out)


@app.route("/label", methods=["POST"])
def label():
    """사용자가 의심 시점에 기법 라벨을 붙여 라이브러리에 등록.

    body: {job_id, peak_sec, technique}
    """
    data = request.get_json(silent=True) or {}
    job_id = data.get("job_id")
    peak_sec = data.get("peak_sec")
    technique = (data.get("technique") or "").strip()

    if not job_id or peak_sec is None or not technique:
        return jsonify({"ok": False, "error": "job_id, peak_sec, technique 모두 필요"}), 400

    # job 메타에서 입력 영상 경로 복원(인메모리 우선, 없으면 디스크)
    job = _jobs.get(job_id) or _load_job_state(job_id)
    if not job:
        return jsonify({"ok": False, "error": "해당 job_id를 찾을 수 없음"}), 404
    video_arg = job["video_arg"]

    # 캐논 기법 키/한글명 결정(용어집에 있으면 그쪽 따름)
    e = lookup(technique)
    if e:
        key_en, name_ko = e["en"], e["ko"]
    else:
        key_en, name_ko = technique, technique  # 자유 입력은 그대로 등록

    try:
        local = resolve_video_input(video_arg)
    except Exception as ex:
        return jsonify({"ok": False, "error": f"영상 경로 해석 실패: {ex}"}), 500

    try:
        sig = signature_from_video(str(local), float(peak_sec))
    except Exception as ex:
        return jsonify({"ok": False, "error": f"시그니처 추출 실패: {ex}"}), 500
    if sig is None:
        return jsonify({"ok": False,
                        "error": "손 미검출 — 다른 시점을 선택하거나 영상 품질을 확인하세요"}), 422

    save_entry(key_en, name_ko, sig,
               source=f"webapp:{job_id}@{float(peak_sec):.2f}")
    return jsonify({"ok": True, "technique": key_en, "name_ko": name_ko,
                    "peak_sec": float(peak_sec)})


@app.route("/jobs/<job_id>/<mode>/<path:filename>")
def job_file(job_id, mode, filename):
    # 세그먼트 화이트리스트 + secure_filename으로 경로 탈출 방지
    if mode not in SEG_OK:
        abort(404)
    safe = secure_filename(filename)
    if not safe:
        abort(404)
    return send_from_directory(JOBS_DIR / job_id / mode, safe)


if __name__ == "__main__":
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
