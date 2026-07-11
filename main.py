"""
main.py — Backend API Layer
FastAPI + DSPy + YouTube Data API
"""

import os
import re
import json
import time
import threading
import random
import httpx
import asyncio
import uuid
from fastapi import BackgroundTasks
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from dspy_module import (
    configure_lm,
    load_module,
    load_quiz_module,
    load_relevance_module,
    VARKModule,
    QuizModule,
    RemediationModule,
    VideoQueryModule,
    VideoRelevanceModule,
    search_youtube,
    _fetch_video_stats,
    _fetch_transcript,
    _fmt_count,
)


# ──────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────
app = FastAPI(title="VARK Study API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend files from ./public
app.mount("/static", StaticFiles(directory="public"), name="static")


# ──────────────────────────────────────────────
# Startup: configure DSPy
# ──────────────────────────────────────────────

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
vark_module: Optional[VARKModule] = None
quiz_module: Optional[QuizModule] = None
query_module: Optional[VideoQueryModule] = None
relevance_module: Optional[VideoRelevanceModule] = None
remediation_module: Optional[RemediationModule] = None

@app.on_event("startup")
async def startup_event():
    global vark_module, quiz_module, query_module, relevance_module, remediation_module
    try:
        configure_lm()  # reads TYPHOON_API_KEY from env
        vark_module       = load_module("vark_model.json")
        quiz_module       = load_quiz_module("quiz_model.json")
        query_module      = VideoQueryModule()
        relevance_module  = load_relevance_module()
        remediation_module = RemediationModule()  # zero-shot, no demos
        print("✅ DSPy modules ready")
    except Exception as e:
        print(f"⚠️  DSPy startup error: {e}")


# ──────────────────────────────────────────────
# Helper: Extract text from uploaded PDF
# ──────────────────────────────────────────────
async def extract_pdf_text(pdf_bytes: bytes) -> str:
    """
    สกัดข้อความจาก PDF bytes
    ลองใช้ pymupdf (fitz) ก่อน — เร็วกว่าและไม่มี FontBBox warning
    ถ้าไม่มีให้ fallback ไป pdfminer พร้อม suppress warning
    """
    # ── Strategy 1: pymupdf (fitz) ──────────────────────────────
    try:
        import fitz  # pymupdf
        doc   = fitz.open(stream=pdf_bytes, filetype="pdf")
        parts = [page.get_text("text") for page in doc]
        doc.close()
        text  = "\n".join(parts).strip()
        if text:
            return text[:15000] if len(text) > 15000 else text
    except ImportError:
        pass  # fitz ไม่ได้ติดตั้ง → ใช้ pdfminer แทน
    except Exception as e:
        print(f"pymupdf extraction warning: {e}")

    # ── Strategy 2: pdfminer (suppress FontBBox + logging noise) ─
    try:
        import io
        import logging
        # ปิด warning ทุก logger ของ pdfminer ที่ noise
        for noisy in ("pdfminer.pdfdocument", "pdfminer.pdfpage",
                      "pdfminer.pdfinterp", "pdfminer.pdfdevice",
                      "pdfminer.pdffont", "pdfminer.cmapdb",
                      "pdfminer.converter", "pdfminer"):
            logging.getLogger(noisy).setLevel(logging.ERROR)

        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.layout import LAParams

        output = io.StringIO()
        extract_text_to_fp(
            io.BytesIO(pdf_bytes),
            output,
            laparams=LAParams(),
            output_type="text",
            codec="utf-8",
        )
        text = output.getvalue().strip()
        return text[:15000] if len(text) > 15000 else text

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"PDF extraction failed: {e}")


# ──────────────────────────────────────────────
# Helper: Search YouTube
# ──────────────────────────────────────────────
def _fix_json_escapes(s: str) -> str:
    """
    แก้ backslash ที่ไม่ valid ใน JSON string ให้กลายเป็น \\\\ ที่ถูกต้อง
    โมเดลมักเขียน KaTeX เป็น single backslash (\\[ , \\frac, \\approx) ทั้งที่ JSON
    ต้องใช้ double backslash → ทำให้ json.loads ล้มด้วย 'Invalid \\escape'
    ฟังก์ชันนี้ปล่อย escape ที่ถูกต้องไว้ (\\" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX)
    และ "ทำให้ถูก" เฉพาะ backslash เดี่ยว ๆ ที่ตามด้วยตัวอักษรอื่น
    """
    def repl(m: "re.Match") -> str:
        bs  = m.group(1)            # run ของ backslash ที่ติดกัน
        nxt = m.group(2)            # ตัวอักษรถัดไป (อาจเป็น '')
        n   = len(bs)
        pairs = "\\\\" * (n // 2)   # คู่ที่สมบูรณ์ = escaped backslash อยู่แล้ว
        if n % 2 == 0:
            return pairs + nxt
        # เหลือ backslash เดี่ยว 1 ตัว — เก็บเฉพาะ escape ที่ "ชัดเจน" ไว้
        # (\" \\ \/ \uXXXX) ส่วน \f \b \n \r \t ถือเป็นคำสั่ง LaTeX (\frac \theta …)
        # ไม่ใช่ control char → double ให้กลายเป็น backslash literal เพื่อกัน KaTeX พัง
        if nxt in '"/u':
            return pairs + "\\" + nxt    # เป็น escape ที่ถูกต้อง → คงไว้
        return pairs + "\\\\" + nxt      # ที่เหลือ → escape ให้เป็น backslash literal

    return re.sub(r"(\\+)(.?)", repl, s, flags=re.DOTALL)


def _parse_json_loose(raw: str) -> list:
    """
    Parse JSON array อย่าง robust — strip markdown fences, whitespace, trailing commas
    รองรับ DSPy ChainOfThought ที่อาจมี reasoning text นำหน้า JSON array
    และ KaTeX backslash ที่ escape ไม่ครบ (\\[ , \\frac, ...)
    """
    text = raw.strip()

    # strip ```json ... ``` or ``` ... ```
    text = re.sub(r"```[a-z]*\s*", "", text)
    text = re.sub(r"\s*```", "", text)
    text = text.strip()

    # ── ตัดเอาเฉพาะ array ก้อนนอกสุด: จาก '[' ตัวแรกถึง ']' ตัวสุดท้าย ──
    #    (greedy — กัน ']' ที่อยู่กลางสูตร KaTeX อย่าง \\] ตัดก้อนสั้นเกินไป)
    start = text.find("[")
    end   = text.rfind("]")
    if start != -1 and end > start:
        candidate = text[start:end + 1]
        # strip trailing commas before ] or }
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        for attempt in (candidate, _fix_json_escapes(candidate)):
            try:
                result = json.loads(attempt)
                if isinstance(result, list) and result:
                    return result
            except Exception:
                pass

    # ── ลอง parse ทั้ง string ──
    clean = re.sub(r",\s*([}\]])", r"\1", text)
    for attempt in (clean, _fix_json_escapes(clean)):
        try:
            result = json.loads(attempt)
            return result if isinstance(result, list) else [result]
        except Exception:
            pass

    # ── last-resort: extract quoted strings ──
    found = re.findall(r'"([^"\n]{5,})"', text)
    return found if found else []


def _run_relevance_module(topic: str, videos_list_text: str) -> tuple[str, str]:
    """Sync wrapper — calls the DSPy VideoRelevanceModule (Typhoon).
    Returns (relevant_indices_json, vark_per_video_json)."""
    if relevance_module is None:
        return "[]", "{}"
    try:
        pred = relevance_module(topic=topic, videos_with_transcripts=videos_list_text)
        return pred.relevant_indices or "[]", pred.vark_per_video or "{}"
    except Exception as e:
        print(f"[relevance] Typhoon module error: {e}")
        return "[]", "{}"


async def filter_videos_by_relevance(videos: list[dict], topic: str) -> list[dict]:
    """
    1. Fetch transcripts for all real videos in parallel (YouTube scrape, no LLM cost).
    2. One Typhoon call (DSPy VideoRelevanceModule) to judge all videos at once.
       Gemini is NOT used here — it stays as eval judge only.
    Falls back to original list if Typhoon fails or filters everything out.
    """
    import asyncio

    real  = [v for v in videos if not v.get("is_search_link") and v.get("video_id")]
    links = [v for v in videos if v.get("is_search_link")]

    if not real:
        return videos

    # Step 1: parallel transcript fetch (with an overall time budget) + batch stats.
    #
    # On Render, youtube-transcript-api is IP-blocked, so every uncached transcript
    # falls through to Gemini, which can stall under rate limits. The frontend aborts
    # /generate at 180s, so we cap the whole transcript phase: whatever finishes in
    # time is used, the rest are judged on title/views only. Threads that didn't make
    # the budget keep running in the background and cache their result for next time.
    budget = float(os.environ.get("RELEVANCE_TRANSCRIPT_BUDGET", "60"))

    async def _transcripts_within_budget() -> list[str]:
        tasks = [asyncio.ensure_future(asyncio.to_thread(_fetch_transcript, v["video_id"])) for v in real]
        t_start = time.perf_counter()
        done, pending = await asyncio.wait(tasks, timeout=budget)
        out = []
        for t in tasks:
            if t in done and not t.cancelled() and t.exception() is None:
                out.append(t.result() or "")
            else:
                out.append("")  # didn't finish within budget (or errored) → no transcript
        got = sum(1 for x in out if x)
        print(
            f"[timing] transcripts: {got}/{len(real)} fetched in {time.perf_counter()-t_start:.1f}s "
            f"(budget {budget:.0f}s, {len(pending)} still running in background)",
            flush=True,
        )
        return out

    transcripts, stats = await asyncio.gather(
        _transcripts_within_budget(),
        _fetch_video_stats([v["video_id"] for v in real]),
    )

    # Step 2: build numbered list with views/likes for Typhoon
    lines = "\n".join(
        f"{i+1}. Title: {v['title']}\n"
        f"   Channel: {v.get('channel','')}\n"
        f"   Views: {_fmt_count(stats.get(v['video_id'],{}).get('views',0))}  "
        f"Likes: {_fmt_count(stats.get(v['video_id'],{}).get('likes',0))}\n"
        f"   Transcript: {(t or '(no transcript)')[:500]}"
        for i, (v, t) in enumerate(zip(real, transcripts))
    )

    # Step 3: single Typhoon call
    _t_rel = time.perf_counter()
    raw_indices, raw_vark = await asyncio.to_thread(_run_relevance_module, topic[:1000], lines)
    print(f"[timing] relevance_llm: {time.perf_counter()-_t_rel:.1f}s", flush=True)

    # Step 4: parse indices (max 10) + vark map
    try:
        m = re.search(r"\[.*?\]", raw_indices, re.DOTALL)
        indices_list = json.loads(m.group(0)) if m else list(range(1, len(real) + 1))
        indices = set(indices_list[:10])  # hard cap at 10
    except Exception:
        indices = set(range(1, min(len(real) + 1, 11)))

    try:
        m2 = re.search(r"\{.*?\}", raw_vark, re.DOTALL)
        vark_map: dict = json.loads(m2.group(0)) if m2 else {}
    except Exception:
        vark_map = {}

    passed = []
    for i, v in enumerate(real, 1):
        if i in indices:
            v_copy = dict(v)
            v_copy["vark"] = vark_map.get(str(i), "")
            passed.append(v_copy)

    print(f"[relevance] {len(passed)}/{len(real)} videos passed Typhoon relevance check")
    return (passed + links) if passed else real[:10] + links


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.get("/")
@app.get("/index.html")
async def root():
    return FileResponse("public/index.html")


@app.get("/study")
@app.get("/study.html")
async def study():
    return FileResponse("public/study.html")


@app.get("/config")
async def get_config():
    """
    ส่ง public config ให้ frontend
    Typhoon API key ไม่ expose แล้ว — เรียกผ่าน /quiz, /generate ของ backend
    """
    return {
        "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
    }


@app.post("/tts")
async def text_to_speech(body: dict):
    """
    gTTS — รับ { "text": "...", "lang": "th" } แล้วคืน audio/mpeg (mp3)
    ใช้ Google Translate TTS endpoint ผ่าน gtts library (ไม่ต้องใช้ API key)
    """
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    lang = (body.get("lang") or "th").strip() or "th"

    import asyncio
    import io
    from fastapi.responses import Response
    from gtts import gTTS

    def _synthesize() -> bytes:
        buf = io.BytesIO()
        gTTS(text=text, lang=lang, slow=False).write_to_fp(buf)
        return buf.getvalue()

    try:
        audio_bytes = await asyncio.to_thread(_synthesize)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"gTTS error: {e}")


@app.get("/video-info/{video_id}")
async def get_video_info(video_id: str):
    """
    ดึงข้อมูลเพิ่มเติมของ YouTube video (description, tags, categoryId)
    เพื่อใช้วิเคราะห์ VARK style ที่ละเอียดขึ้นจากเนื้อหาจริงของคลิป

    - ถ้ามี YOUTUBE_API_KEY → ดึง snippet (description + tags) ผ่าน YouTube Data API v3
    - ถ้าไม่มี → ดึงผ่าน YouTube oEmbed (ได้แค่ title, author)
    """
    api_key = os.environ.get("YOUTUBE_API_KEY")

    # ── Strategy 1: YouTube Data API v3 (full description + tags) ──
    if api_key:
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                resp = await client.get(
                    "https://www.googleapis.com/youtube/v3/videos",
                    params={
                        "part":  "snippet,topicDetails",
                        "id":    video_id,
                        "key":   api_key,
                        "hl":    "th",
                    },
                )
                data = resp.json()
                items = data.get("items", [])
                if items:
                    snippet = items[0].get("snippet", {})
                    topic   = items[0].get("topicDetails", {})
                    return {
                        "video_id":    video_id,
                        "title":       snippet.get("title", ""),
                        "channel":     snippet.get("channelTitle", ""),
                        "description": snippet.get("description", "")[:1500],
                        "tags":        snippet.get("tags", [])[:20],
                        "category_id": snippet.get("categoryId", ""),
                        "topic_categories": topic.get("topicCategories", []),
                        "source":      "youtube_api",
                    }
        except Exception as e:
            print(f"YouTube video-info API error [{video_id}]: {e}")

    # ── Strategy 2: oEmbed fallback (title + author only) ──────────
    try:
        async with httpx.AsyncClient(timeout=6) as client:
            resp = await client.get(
                "https://www.youtube.com/oembed",
                params={
                    "url":    f"https://www.youtube.com/watch?v={video_id}",
                    "format": "json",
                },
            )
            if resp.status_code == 200:
                oe = resp.json()
                return {
                    "video_id":    video_id,
                    "title":       oe.get("title", ""),
                    "channel":     oe.get("author_name", ""),
                    "description": "",
                    "tags":        [],
                    "category_id": "",
                    "topic_categories": [],
                    "source":      "oembed",
                }
    except Exception as e:
        print(f"oEmbed error [{video_id}]: {e}")

    # ── Not found ──────────────────────────────────────────────────
    return {
        "video_id":    video_id,
        "title":       "",
        "channel":     "",
        "description": "",
        "tags":        [],
        "category_id": "",
        "topic_categories": [],
        "source":      "none",
    }


@app.post("/generate")
async def generate(
    pdf: UploadFile = File(...),
    vark_style: str = Form(...),
    topic: str      = Form(""),
):
    """
    รับ PDF + VARK profile → stream progress event ระหว่างทาง แล้วปิดท้ายด้วย
    event "result" ที่มี payload เดิม (learning_material + videos + ...)
    """
    if vark_module is None:
        raise HTTPException(status_code=503, detail="AI module not initialized")

    # อ่านไฟล์ก่อนเข้า generator — UploadFile อาจถูกปิดหลัง request คืนค่าแล้ว
    pdf_bytes = await pdf.read()

    async def event_stream():
        _t_req = time.perf_counter()
        def _lap(label: str, since: float) -> float:
            now = time.perf_counter()
            print(f"[timing] {label}: {now - since:.1f}s (total {now - _t_req:.1f}s)", flush=True)
            return now

        try:
            # ── Step 1: PDF extract ──────────────────────────
            yield _sse("progress", {"pct": 5, "label": "📄 กำลังอ่านไฟล์ PDF..."})
            
            _t = time.perf_counter()
            context = await extract_pdf_text(pdf_bytes)
            _t = _lap("pdf_extract", _t)
            if topic:
                context = f"{context}\n\n[คำสั่งเพิ่มเติม: {topic}]"

            # ── Step 2: VARK study-guide generation ──────────
            yield _sse("progress", {"pct": 20, "label": "👨‍🏫 กำลังสร้างสื่อการเรียนรู้..."})
            print(f"[debug] active threads at vark_generate start: {threading.active_count()}")
            try:
                prediction = await asyncio.to_thread(vark_module, context=context, vark_style=vark_style)
            except Exception as e:
                yield _sse("error", {"detail": f"AI generation error: {e}"})
                return
            _t = _lap("vark_generate", _t)
            learning_material = prediction.learning_material or ""

            # ── Step 3: YouTube query generation ─────────────
            yield _sse("progress", {"pct": 50, "label": "📽️ กำลังวางแผนค้นหาวิดีโอ..."})
            youtube_queries_raw = "[]"
            if query_module is not None:
                try:
                    q_pred = await asyncio.to_thread(query_module, pdf_content=context)
                    youtube_queries_raw = q_pred.youtube_queries or "[]"
                except Exception as e:
                    print(f"[VideoQueryModule] error: {e}")
            _t = _lap("query_generate", _t)

            try:
                yt_queries = _parse_json_loose(youtube_queries_raw)
                yt_queries = [q for q in yt_queries if isinstance(q, str) and q.strip()]
            except Exception:
                yt_queries = []
            if not yt_queries:
                fallback_q = topic.strip() if topic.strip() else context[:80].strip()
                yt_queries = [fallback_q] if fallback_q else ["การเรียนรู้"]

            # ── Step 4: YouTube search ───────────────────────
            yield _sse("progress", {"pct": 65, "label": "🔍 กำลังค้นหาวิดีโอ YouTube ที่เกี่ยวข้อง..."})
            videos = await search_youtube(yt_queries, max_per_query=3)
            _t = _lap("youtube_search", _t)

            # ── Step 5: Relevance filter ──────────────────────
            yield _sse("progress", {"pct": 85, "label": "🎯 กำลังคัดกรองวิดีโอที่ตรงเนื้อหา..."})
            topic_hint = topic.strip() or context[:150].strip()
            videos = await filter_videos_by_relevance(videos, topic_hint)
            _t = _lap("relevance_filter", _t)

            vark_data = {}
            try:
                vark_data = json.loads(vark_style)
            except Exception:
                pass
            session_id = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{id(prediction)}"

            # ── Done: ส่ง payload จริง ─────────────────────────
            yield _sse("progress", {"pct": 100, "label": "✅ เสร็จแล้ว!"})
            yield _sse("result", {
                "session_id":        session_id,
                "learning_material": learning_material,
                "youtube_queries":   yt_queries,
                "videos":            videos,
                "context_snippet":   context[:500],
                "vark_style":        vark_data,
            })
        except Exception as e:
            yield _sse("error", {"detail": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


class QuizRequest(BaseModel):
    learning_material: str
    vark_style: dict


@app.post("/quiz")
async def generate_quiz(req: QuizRequest):
    """
    สร้าง MCQ จาก section content + VARK profile
    คืน JSON array ของคำถาม (validated)
    """
    if quiz_module is None:
        raise HTTPException(status_code=503, detail="Quiz module not initialized")

    try:
        pred = await asyncio.to_thread(
            quiz_module,
            learning_material=req.learning_material[:4000],
            vark_style=json.dumps(req.vark_style, ensure_ascii=False),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quiz generation error: {e}")

    raw = pred.questions or "[]"
    try:
        parsed = _parse_json_loose(raw)
    except Exception:
        parsed = []

    keys = ["A", "B", "C", "D"]
    questions = []
    for q in parsed if isinstance(parsed, list) else []:
        if not isinstance(q, dict):
            continue
        opts = q.get("options") or {}
        if not all(k in opts for k in keys):
            continue
        if q.get("answer") not in keys:
            continue
        # Shuffle options so the correct answer isn't always A
        correct_text = str(opts[q["answer"]])
        values = [str(opts[k]) for k in keys]
        random.shuffle(values)
        new_opts = dict(zip(keys, values))
        new_answer = next(k for k, v in new_opts.items() if v == correct_text)
        questions.append({
            "q":           str(q.get("q", "")).strip(),
            "options":     new_opts,
            "answer":      new_answer,
            "vark":        q.get("vark") if q.get("vark") in ("V","A","R","K") else "R",
            "concept":     str(q.get("concept", "")).strip(),
            "explanation": str(q.get("explanation", "")).strip(),
        })

    return {"questions": questions}


class AdaptRequest(BaseModel):
    learning_material: str
    vark_style: dict
    results: list[dict]   # [{concept, chosenText, isRight, ...}] from the graded quiz


@app.post("/adapt")
async def adapt(req: AdaptRequest):
    """
    Adaptive loop: from the graded quiz `results`, re-teach ONLY the concepts the learner
    missed. Returns targeted remediation Markdown (empty + mastered=True if nothing wrong).
    Frontend can then call /quiz on the remediation to generate fresh 'check' questions.
    """
    if remediation_module is None:
        raise HTTPException(status_code=503, detail="Remediation module not initialized")

    # Keep only wrong answers that carry a concept; dedupe by (concept, chosenText).
    seen, weak = set(), []
    for r in req.results or []:
        if not isinstance(r, dict) or r.get("isRight"):
            continue
        concept = str(r.get("concept", "")).strip()
        if not concept:
            continue
        chosen_text = str(r.get("chosenText", "")).strip()
        key = (concept, chosen_text)
        if key in seen:
            continue
        seen.add(key)
        weak.append({"concept": concept, "chosenText": chosen_text})

    if not weak:
        return {"remediation": "", "mastered": True, "weak_concepts": []}

    try:
        pred = await asyncio.to_thread(
            remediation_module,
            learning_material=req.learning_material[:4000],
            vark_style=json.dumps(req.vark_style, ensure_ascii=False),
            weak_spots=json.dumps(weak, ensure_ascii=False),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Remediation error: {e}")

    return {
        "remediation": pred.remediation or "",
        "mastered": False,
        "weak_concepts": sorted({w["concept"] for w in weak}),
    }

# ──────────────────────────────────────────────
# for mobile background to work
# ──────────────────────────────────────────────
# simple in-memory job store — {job_id: {"status": ..., "progress": ..., "result": ..., "error": ...}}
_jobs: dict[str, dict] = {}

async def _run_generation_job(job_id: str, pdf_bytes: bytes, vark_style: str, topic: str):
    def update(pct, label):
        _jobs[job_id]["progress"] = {"pct": pct, "label": label}

    try:
        update(5, "📄 กำลังอ่านไฟล์ PDF...")
        context = await extract_pdf_text(pdf_bytes)
        if topic:
            context = f"{context}\n\n[คำสั่งเพิ่มเติม: {topic}]"

        update(20, "👨‍🏫 กำลังสร้างสื่อการเรียนรู้...")
        prediction = await asyncio.to_thread(vark_module, context=context, vark_style=vark_style)
        learning_material = prediction.learning_material or ""

        update(50, "📽️ กำลังวางแผนค้นหาวิดีโอ...")
        youtube_queries_raw = "[]"
        if query_module is not None:
            q_pred = await asyncio.to_thread(query_module, pdf_content=context)
            youtube_queries_raw = q_pred.youtube_queries or "[]"

        yt_queries = _parse_json_loose(youtube_queries_raw)
        yt_queries = [q for q in yt_queries if isinstance(q, str) and q.strip()] or [topic.strip() or context[:80].strip() or "การเรียนรู้"]

        update(65, "🔍 กำลังค้นหาวิดีโอ YouTube ที่เกี่ยวข้อง...")
        videos = await search_youtube(yt_queries, max_per_query=3)

        update(85, "🎯 กำลังคัดกรองวิดีโอที่ตรงเนื้อหา...")
        topic_hint = topic.strip() or context[:150].strip()
        videos = await filter_videos_by_relevance(videos, topic_hint)

        vark_data = {}
        try:
            vark_data = json.loads(vark_style)
        except Exception:
            pass

        update(100, "✅ เสร็จแล้ว!")
        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["result"] = {
            "learning_material": learning_material,
            "youtube_queries": yt_queries,
            "videos": videos,
            "context_snippet": context[:500],
            "vark_style": vark_data,
        }
    except Exception as e:
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = str(e)


@app.post("/generate/start")
async def generate_start(
    pdf: UploadFile = File(...),
    vark_style: str = Form(...),
    topic: str = Form(""),
):
    if vark_module is None:
        raise HTTPException(status_code=503, detail="AI module not initialized")
    pdf_bytes = await pdf.read()
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running", "progress": {"pct": 0, "label": "starting"}, "result": None, "error": None}
    asyncio.create_task(_run_generation_job(job_id, pdf_bytes, vark_style, topic))
    return {"job_id": job_id}


@app.get("/generate/status/{job_id}")
async def generate_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job

@app.post("/evaluate")
async def evaluate():
    """
    Evaluate VARK module on train dataset — Gemini judge per A/B rubric (0-10).
    Returns score report with mean_total and per-criterion averages.
    """
    import asyncio
    from dspy_module import evaluate_testset, get_trainset, _next_eval_path

    try:
        trainset = get_trainset()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load train dataset: {e}")

    if not trainset:
        raise HTTPException(status_code=400, detail="Train dataset is empty")

    module = vark_module
    if module is None:
        raise HTTPException(status_code=503, detail="AI module not initialized")

    report_path = _next_eval_path("vark")
    try:
        report = await asyncio.to_thread(
            evaluate_testset,
            module,
            trainset,
            blind=False,
            report_path=report_path,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {e}")

    return {
        "status": "ok",
        "n": report.get("n"),
        "mean_total": report.get("mean_total"),
        "criterion_avgs": report.get("criterion_avgs"),
        "judge": report.get("judge", {}).get("model"),
        "report_path": report_path,
        "report_md": report_path[:-5] + ".md",
    }

    
# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # Render (และ host อื่น) ส่ง port มาทาง $PORT — local ใช้ 8000 + reload
    port = int(os.environ.get("PORT", "8000"))
    reload = os.environ.get("PORT") is None
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload)