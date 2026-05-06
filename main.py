"""
main.py — Backend API Layer
FastAPI + DSPy + YouTube Data API
"""

import os
import re
import json
import httpx
import jsonlines
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import dspy
from dspy_module import (
    configure_lm,
    load_module,
    load_quiz_module,
    load_classifier_module,
    VARKModule,
    QuizModule,
    VideoClassifierModule,
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
vark_module: Optional[VARKModule] = None
quiz_module: Optional[QuizModule] = None
classifier_module: Optional[VideoClassifierModule] = None

@app.on_event("startup")
async def startup_event():
    global vark_module, quiz_module, classifier_module
    try:
        configure_lm()  # reads GOOGLE_API_KEY from env
        vark_module       = load_module("vark_model.json")
        quiz_module       = load_quiz_module("quiz_model.json")
        classifier_module = load_classifier_module("video_classifier_model.json")
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
def _parse_json_loose(raw: str) -> list:
    """
    Parse JSON array อย่าง robust — strip markdown fences, whitespace, trailing commas
    รองรับ DSPy ChainOfThought ที่อาจมี reasoning text นำหน้า JSON array
    """
    text = raw.strip()

    # strip ```json ... ``` or ``` ... ```
    text = re.sub(r"```[a-z]*\s*", "", text)
    text = re.sub(r"\s*```", "", text)
    text = text.strip()

    # ── ลอง find JSON array [...] ที่อยู่ในข้อความ (รองรับ CoT prefix) ──
    array_match = re.search(r'\[.*?\]', text, re.DOTALL)
    if array_match:
        candidate = array_match.group(0)
        # strip trailing commas before ] or }
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            result = json.loads(candidate)
            if isinstance(result, list) and result:
                return result
        except Exception:
            pass

    # ── ลอง parse ทั้ง string ──
    clean = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        result = json.loads(clean)
        return result if isinstance(result, list) else [result]
    except Exception:
        pass

    # ── last-resort: extract quoted strings ──
    found = re.findall(r'"([^"\n]{5,})"', text)
    return found if found else []


async def search_youtube(queries: list[str], max_per_query: int = 2) -> list[dict]:
    """
    ค้นหา YouTube videos
    - ถ้ามี YOUTUBE_API_KEY → ใช้ YouTube Data API v3
    - ถ้าไม่มี / API error / quota หมด → สร้าง search-URL cards แทน
    """
    api_key = os.environ.get("YOUTUBE_API_KEY")

    def make_search_links(qs):
        """Fallback: search-link cards ที่คลิกแล้วเปิด YouTube ได้เลย"""
        cards = []
        for q in qs[:5]:
            q_enc = q.replace(" ", "+")
            cards.append({
                "video_id":       None,
                "title":          q,
                "channel":        "ค้นหาใน YouTube",
                "thumbnail":      "",
                "url":            f"https://www.youtube.com/results?search_query={q_enc}",
                "embed_url":      "",
                "is_search_link": True,
            })
        return cards

    # ── ไม่มี key → fallback ทันที ─────────────────────────────
    if not api_key:
        return make_search_links(queries)

    # ── มี key → เรียก YouTube Data API v3 ─────────────────────
    results   = []
    seen_ids  = set()

    async with httpx.AsyncClient(timeout=10) as client:
        for query in queries[:5]:
            try:
                resp = await client.get(
                    "https://www.googleapis.com/youtube/v3/search",
                    params={
                        "part":             "snippet",
                        "q":                query,
                        "type":             "video",
                        "maxResults":       max_per_query,
                        "key":              api_key,
                        "relevanceLanguage":"th",
                    },
                )
                data = resp.json()

                # ── ตรวจจับ quota/error response จาก Google ────
                if "error" in data:
                    err_msg = data["error"].get("message", "unknown")
                    print(f"YouTube API error: {err_msg} — falling back to search links")
                    return make_search_links(queries)

                for item in data.get("items", []):
                    vid_id = item["id"].get("videoId")
                    if not vid_id or vid_id in seen_ids:
                        continue
                    seen_ids.add(vid_id)
                    snippet = item.get("snippet", {})
                    results.append({
                        "video_id":       vid_id,
                        "title":          snippet.get("title", ""),
                        "channel":        snippet.get("channelTitle", ""),
                        "thumbnail":      snippet.get("thumbnails", {})
                                                  .get("medium", {}).get("url", ""),
                        "url":            f"https://www.youtube.com/watch?v={vid_id}",
                        "embed_url":      f"https://www.youtube.com/embed/{vid_id}",
                        "is_search_link": False,
                    })
            except Exception as e:
                print(f"YouTube search error for '{query}': {e}")

    # ── API ไม่ให้ผลลัพธ์เลย → fallback ─────────────────────────
    if not results:
        print("YouTube API returned no results — falling back to search links")
        return make_search_links(queries)

    return results


# ──────────────────────────────────────────────
# Helper: Search Images (Google Custom Search)
# ──────────────────────────────────────────────
async def _search_images_wikimedia(image_queries: list[dict]) -> list[dict]:
    """
    Fallback: ค้นหาภาพจาก Wikimedia Commons (ฟรี ไม่ต้องใช้ API key)
    ใช้ MediaWiki Action API — opensearch + imageinfo
    """
    results   = []
    seen_urls = set()

    async with httpx.AsyncClient(timeout=8) as client:
        for q_obj in image_queries[:3]:
            q = q_obj.get("q", "")
            if not q:
                continue
            try:
                # 1) ค้นหา page titles ที่ตรงกัน
                search_resp = await client.get(
                    "https://commons.wikimedia.org/w/api.php",
                    params={
                        "action":   "query",
                        "list":     "search",
                        "srsearch": f"{q} filetype:bitmap",
                        "srnamespace": 6,   # namespace 6 = File:
                        "srlimit":  4,
                        "format":   "json",
                    },
                )
                search_data = search_resp.json()
                titles = [item["title"] for item in
                          search_data.get("query", {}).get("search", [])]
                if not titles:
                    continue

                # 2) ดึง URL จริงของแต่ละไฟล์
                info_resp = await client.get(
                    "https://commons.wikimedia.org/w/api.php",
                    params={
                        "action":  "query",
                        "titles":  "|".join(titles[:4]),
                        "prop":    "imageinfo",
                        "iiprop":  "url|thumburl|extmetadata",
                        "iiurlwidth": 600,
                        "format":  "json",
                    },
                )
                info_data = info_resp.json()
                pages = info_data.get("query", {}).get("pages", {})

                for page in pages.values():
                    ii_list = page.get("imageinfo", [])
                    if not ii_list:
                        continue
                    ii  = ii_list[0]
                    url = ii.get("thumburl") or ii.get("url", "")
                    if not url or url in seen_urls:
                        continue
                    # กรอง SVG / WebP ที่บาง browser render ไม่ได้
                    if url.lower().endswith((".svg", ".webp", ".ogg", ".ogv")):
                        continue
                    seen_urls.add(url)
                    meta  = ii.get("extmetadata", {})
                    title = (meta.get("ObjectName", {}).get("value", "")
                             or page.get("title", "").replace("File:", ""))
                    results.append({
                        "url":             url,
                        "thumbnail":       url,
                        "title":           title,
                        "source":          "Wikimedia Commons",
                        "alt_description": q_obj.get("altDescription", title),
                        "query":           q,
                    })
                    if len(results) >= 6:
                        return results
            except Exception as e:
                print(f"Wikimedia search error for '{q}': {e}")

    return results


async def search_images(image_queries: list[dict]) -> list[dict]:
    """
    รับ list ของ image query objects จาก DSPy (มี q, imgType, imgSize, rights, altDescription)
    - ถ้ามี GOOGLE_SEARCH_API + GOOGLE_SEARCH_CX → ใช้ Google Custom Search
    - ถ้าไม่มี → fallback ไป Wikimedia Commons (ฟรี ไม่ต้อง key)
    """
    if not image_queries:
        return []

    api_key = os.environ.get("GOOGLE_SEARCH_API")
    cx      = os.environ.get("GOOGLE_SEARCH_CX")

    # ── ไม่มี Google key → ใช้ Wikimedia fallback ──────────────
    if not api_key or not cx:
        print("[search_images] No Google Search key — falling back to Wikimedia Commons")
        return await _search_images_wikimedia(image_queries)

    # ── Official: Google Custom Search API ─────────────────────
    results   = []
    seen_urls = set()

    async with httpx.AsyncClient(timeout=5) as client:
        for q_obj in image_queries[:3]:
            try:
                params = {
                    "key":         api_key,
                    "cx":          cx,
                    "q":           q_obj.get("q", ""),
                    "searchType":  "image",
                    "imgSize":     q_obj.get("imgSize", "large"),
                    "imgType":     q_obj.get("imgType", "photo"),
                    "rights":      q_obj.get("rights", "cc_publicdomain"),
                    "num":         2,
                    "safe":        "active",
                }
                resp = await client.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params=params,
                )
                data = resp.json()
                for item in data.get("items", []):
                    url = item.get("link", "")
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    results.append({
                        "url":             url,
                        "thumbnail":       item.get("image", {}).get("thumbnailLink", url),
                        "title":           item.get("title", ""),
                        "source":          item.get("displayLink", ""),
                        "alt_description": q_obj.get("altDescription", ""),
                        "query":           q_obj.get("q", ""),
                    })
            except Exception as e:
                print(f"Image search error for '{q_obj.get('q')}': {e}")

    return results
class FeedbackItem(BaseModel):
    session_id: str
    vark_style: dict
    context_snippet: str        # first 500 chars of context
    learning_material: str
    youtube_queries: str
    liked: bool                 # True = user liked this output


class FeedbackRequest(BaseModel):
    items: list[FeedbackItem]   # Top-5 liked items from frontend


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("public/study.html")


@app.get("/config")
async def get_config():
    """
    ส่ง public config ให้ frontend
    Typhoon API key ไม่ expose แล้ว — เรียกผ่าน /quiz, /classify-video, /generate ของ backend
    """
    return {
        "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
        # image search พร้อมเสมอ — มี Wikimedia Commons เป็น fallback เมื่อไม่มี Google key
        "image_search_ready": True,
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
    vark_style: str = Form(...),   # JSON string
    topic: str      = Form(""),    # optional instruction
):
    """
    รับ PDF + VARK profile → คืน learning_material (Markdown) + YouTube videos
    """
    if vark_module is None:
        raise HTTPException(status_code=503, detail="AI module not initialized")

    # 1. Extract text from PDF
    pdf_bytes = await pdf.read()
    context   = await extract_pdf_text(pdf_bytes)

    if topic:
        context = f"{context}\n\n[คำสั่งเพิ่มเติม: {topic}]"

    # 2. Run DSPy module
    try:
        prediction = vark_module(context=context, vark_style=vark_style)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generation error: {e}")

    learning_material = prediction.learning_material or ""
    youtube_queries_raw = prediction.youtube_queries or "[]"
    image_queries_raw   = prediction.image_queries   or "[]"

    # 3. Parse YouTube queries (robust — handles markdown fences, trailing commas, CoT text)
    try:
        yt_queries = _parse_json_loose(youtube_queries_raw)
        yt_queries = [q for q in yt_queries if isinstance(q, str) and q.strip()]
    except Exception:
        yt_queries = []

    # ถ้ายังไม่ได้ query เลย ให้ใช้หัวข้อจาก topic หรือ context แทน
    if not yt_queries:
        fallback_q = topic.strip() if topic.strip() else context[:80].strip()
        yt_queries = [fallback_q] if fallback_q else ["การเรียนรู้"]

    print(f"[DEBUG] youtube_queries_raw: {repr(youtube_queries_raw[:200])}")
    print(f"[DEBUG] yt_queries parsed: {yt_queries}")

    # 3b. Parse image queries
    try:
        img_queries = _parse_json_loose(image_queries_raw)
        # กรองเฉพาะ object ที่มี key "q"
        if isinstance(img_queries, list):
            img_queries = [q for q in img_queries if isinstance(q, dict) and q.get("q")]
        else:
            img_queries = []
    except Exception:
        img_queries = []

    print(f"[DEBUG] image_queries_raw: {repr(image_queries_raw[:200])}")
    print(f"[DEBUG] img_queries parsed: {img_queries}")

    # 4. Fetch YouTube videos
    videos = await search_youtube(yt_queries)

    # 4b. Fetch images (only when Visual queries exist)
    images = await search_images(img_queries) if img_queries else []

    # 5. Build session record (for feedback system)
    vark_data = {}
    try:
        vark_data = json.loads(vark_style)
    except Exception:
        pass

    session_id = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{id(prediction)}"

    return {
        "session_id": session_id,
        "learning_material": learning_material,
        "youtube_queries": yt_queries,
        "videos": videos,
        "images": images,
        "context_snippet": context[:500],
        "vark_style": vark_data,
    }


class QuizRequest(BaseModel):
    learning_material: str
    vark_style: dict
    difficulty: str = "medium"   # easy | medium | hard
    count: int = 5


@app.post("/quiz")
async def generate_quiz(req: QuizRequest):
    """
    สร้าง MCQ จาก learning material + VARK profile + difficulty
    คืน JSON array ของคำถาม (validated)
    """
    if quiz_module is None:
        raise HTTPException(status_code=503, detail="Quiz module not initialized")

    diff = req.difficulty if req.difficulty in ("easy", "medium", "hard") else "medium"
    count = max(1, min(int(req.count or 5), 20))

    try:
        pred = quiz_module(
            learning_material=req.learning_material[:3500],
            vark_style=json.dumps(req.vark_style, ensure_ascii=False),
            difficulty=diff,
            count=count,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quiz generation error: {e}")

    raw = pred.questions or "[]"
    try:
        parsed = _parse_json_loose(raw)
    except Exception:
        parsed = []

    questions = []
    for q in parsed if isinstance(parsed, list) else []:
        if not isinstance(q, dict):
            continue
        opts = q.get("options") or {}
        if not all(k in opts for k in ("A", "B", "C", "D")):
            continue
        if q.get("answer") not in ("A", "B", "C", "D"):
            continue
        questions.append({
            "q":           str(q.get("q", "")).strip(),
            "options":     {k: str(opts[k]) for k in ("A", "B", "C", "D")},
            "answer":      q["answer"],
            "vark":        q.get("vark") if q.get("vark") in ("V","A","R","K") else "R",
            "diff":        q.get("diff") if q.get("diff") in ("easy","medium","hard") else diff,
            "explanation": str(q.get("explanation", "")).strip(),
        })

    return {"questions": questions}


class VideoClassifyRequest(BaseModel):
    title: str
    channel: str = ""
    description: str = ""
    tags: list[str] = []
    topic_categories: list[str] = []
    page_snippets: list[str] = []   # ["p1: ...", "p2: ..."]
    vark_style: dict


@app.post("/classify-video")
async def classify_video(req: VideoClassifyRequest):
    """
    จัดประเภท YouTube video ตาม VARK + จับคู่กับหน้า PDF
    คืน {vark, subtopics, pages_covered}
    """
    if classifier_module is None:
        raise HTTPException(status_code=503, detail="Classifier module not initialized")

    metadata = "\n".join(filter(None, [
        f"Description: {req.description[:400]}" if req.description else "",
        f"Tags: {', '.join(req.tags[:8])}" if req.tags else "",
        (f"Categories: {', '.join(c.split('/')[-1] for c in req.topic_categories)}"
         if req.topic_categories else ""),
    ]))
    weight_desc = ", ".join(
        f"{k} {req.vark_style.get(k, 0)}%"
        for k in ("V", "A", "R", "K")
        if req.vark_style.get(k, 0)
    ) or f"dominant {req.vark_style.get('dominant', 'R')}"

    try:
        pred = classifier_module(
            video_title=req.title,
            video_channel=req.channel,
            video_metadata=metadata,
            page_snippets="\n".join(req.page_snippets[:30]),
            vark_weight_desc=weight_desc,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classifier error: {e}")

    raw = (pred.classification or "").strip()
    parsed = {}
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            parsed = json.loads(m.group(0))
        except Exception:
            parsed = {}

    vark = [s for s in (parsed.get("vark") or [])
            if isinstance(s, str) and s.upper() in ("V", "A", "R", "K")]
    vark = list(dict.fromkeys(s.upper() for s in vark))[:4]

    subtopics = [str(s).strip()[:16]
                 for s in (parsed.get("subtopics") or [])
                 if isinstance(s, str) and s.strip()][:4]

    pages_covered = sorted({
        int(n) for n in (parsed.get("pages_covered") or [])
        if isinstance(n, int) and n >= 1
    })[:8]

    return {
        "vark": vark,
        "subtopics": subtopics,
        "pages_covered": pages_covered,
    }


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    """
    รับ Top-5 liked items จาก Frontend
    บันทึกเป็น JSONL เพื่อใช้เป็น Dataset สำหรับ re-compile ในอนาคต
    """
    dataset_path = "feedback_dataset.jsonl"
    liked_items  = [item for item in req.items if item.liked]

    if not liked_items:
        return {"saved": 0, "message": "No liked items to save"}

    with jsonlines.open(dataset_path, mode="a") as writer:
        for item in liked_items:
            writer.write({
                "timestamp":         datetime.utcnow().isoformat(),
                "session_id":        item.session_id,
                "vark_style":        item.vark_style,
                "context_snippet":   item.context_snippet,
                "learning_material": item.learning_material,
                "youtube_queries":   item.youtube_queries,
                "liked":             item.liked,
            })

    return {
        "saved":   len(liked_items),
        "message": f"Saved {len(liked_items)} items to {dataset_path}",
    }


@app.get("/feedback/dataset")
async def get_feedback_dataset():
    """
    ดู feedback dataset (สำหรับ admin/developer)
    """
    dataset_path = "feedback_dataset.jsonl"
    if not os.path.exists(dataset_path):
        return {"records": [], "count": 0}

    records = []
    with jsonlines.open(dataset_path) as reader:
        for obj in reader:
            records.append(obj)

    return {"records": records, "count": len(records)}


@app.post("/recompile")
async def recompile():
    """
    Re-compile DSPy module โดยใช้ feedback dataset
    เรียกใช้เมื่อมีข้อมูล feedback เพียงพอ (เช่น > 20 records)
    """
    global vark_module

    dataset_path = "feedback_dataset.jsonl"
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=400, detail="No feedback dataset found")

    # Load dataset
    examples = []
    with jsonlines.open(dataset_path) as reader:
        for obj in reader:
            if obj.get("liked"):
                ex = dspy.Example(
                    context=obj["context_snippet"],
                    vark_style=json.dumps(obj["vark_style"]),
                    learning_material=obj["learning_material"],
                    youtube_queries=obj["youtube_queries"],
                ).with_inputs("context", "vark_style")
                examples.append(ex)

    if len(examples) < 3:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 3 liked examples, got {len(examples)}"
        )

    from dspy.teleprompt import BootstrapFewShot
    from dspy_module import vark_metric

    teleprompter = BootstrapFewShot(metric=vark_metric, max_bootstrapped_demos=3)
    compiled     = teleprompter.compile(VARKModule(), trainset=examples)
    compiled.save("vark_model.json")

    vark_module = compiled
    return {"message": f"Recompiled with {len(examples)} examples ✅"}


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)