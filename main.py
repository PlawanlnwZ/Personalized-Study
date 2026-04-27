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
from dspy_module import configure_lm, load_module, VARKModule


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

@app.on_event("startup")
async def startup_event():
    global vark_module
    try:
        configure_lm()  # reads GOOGLE_API_KEY from env
        vark_module = load_module("vark_model.json")
        print("✅ DSPy module ready")
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
    - ถ้าไม่มี → สร้าง search-URL cards ให้ผู้ใช้คลิกค้นหาเองได้ (no blank state)
    """
    api_key = os.environ.get("YOUTUBE_API_KEY")

    # ── Fallback: สร้าง search-link cards โดยไม่ต้องใช้ API key ──
    if not api_key:
        results = []
        for q in queries[:5]:
            q_enc = q.replace(" ", "+")
            results.append({
                "video_id":   None,
                "title":      q,
                "channel":    "ค้นหาใน YouTube",
                "thumbnail":  "",
                "url":        f"https://www.youtube.com/results?search_query={q_enc}",
                "embed_url":  "",
                "is_search_link": True,
            })
        return results

    # ── Official API path ──────────────────────────────────────
    results = []
    seen_ids = set()

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
                for item in data.get("items", []):
                    vid_id = item["id"].get("videoId")
                    if not vid_id or vid_id in seen_ids:
                        continue
                    seen_ids.add(vid_id)
                    snippet = item.get("snippet", {})
                    results.append({
                        "video_id":  vid_id,
                        "title":     snippet.get("title", ""),
                        "channel":   snippet.get("channelTitle", ""),
                        "thumbnail": snippet.get("thumbnails", {})
                                            .get("medium", {}).get("url", ""),
                        "url":       f"https://www.youtube.com/watch?v={vid_id}",
                        "embed_url": f"https://www.youtube.com/embed/{vid_id}",
                        "is_search_link": False,
                    })
            except Exception as e:
                print(f"YouTube search error for '{query}': {e}")

    return results


# ──────────────────────────────────────────────
# Helper: Search Images (Google Custom Search)
# ──────────────────────────────────────────────
async def search_images(image_queries: list[dict]) -> list[dict]:
    """
    รับ list ของ image query objects จาก DSPy (มี q, imgType, imgSize, rights, altDescription)
    - ถ้ามี GOOGLE_SEARCH_API + GOOGLE_SEARCH_CX → ใช้ Google Custom Search
    - ถ้าไม่มี → ใช้ Unsplash Source (ไม่ต้อง API key, ฟรี)
    """
    if not image_queries:
        return []

    api_key = os.environ.get("GOOGLE_SEARCH_API")
    cx      = os.environ.get("GOOGLE_SEARCH_CX")

    # ── Fallback: Unsplash Source (no API key needed) ──────────
    if not api_key or not cx:
        results = []
        for q_obj in image_queries[:5]:
            q = q_obj.get("q", "").strip()
            if not q:
                continue
            q_slug = q.replace(" ", ",")
            url    = f"https://source.unsplash.com/featured/800x600/?{q_slug}"
            results.append({
                "url":             url,
                "thumbnail":       url,
                "title":           q,
                "source":          "unsplash.com",
                "alt_description": q_obj.get("altDescription", q),
                "query":           q,
            })
        return results

    # ── Official: Google Custom Search API ─────────────────────
    results   = []
    seen_urls = set()

    async with httpx.AsyncClient(timeout=12) as client:
        for q_obj in image_queries[:5]:
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
    ส่ง public config ให้ frontend ใช้งาน
    รวมถึง TYPHOON_API_KEY สำหรับให้ frontend เรียก Typhoon API โดยตรง
    (ใช้สำหรับ VARK tag classification บน video เท่านั้น)
    """
    return {
        "typhoon_api_key":   os.environ.get("TYPHOON_API_KEY", ""),
        "api_base":          "https://api.opentyphoon.ai/v1",
        "image_search_ready": bool(
            os.environ.get("GOOGLE_SEARCH_API") and
            os.environ.get("GOOGLE_SEARCH_CX")
        ),
    }


@app.post("/tts")
async def text_to_speech(body: dict):
    """
    Proxy Gemini TTS — รับ { "text": "..." } แล้วคืน audio/wav
    อ่าน GEMINI_API_KEY จาก environment (ไม่ expose key ให้ browser)
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash-preview-tts:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": "Aoede"}
                }
            },
        },
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=payload)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        data      = resp.json()
        part      = data["candidates"][0]["content"]["parts"][0]["inlineData"]
        b64_audio = part["data"]
        mime_type = part.get("mimeType", "audio/L16;rate=24000")

        import base64, struct
        from fastapi.responses import Response

        raw_bytes = base64.b64decode(b64_audio)

        # ถ้า Gemini คืน raw PCM (audio/L16) → แปลงเป็น WAV ก่อนส่ง
        if "L16" in mime_type or "pcm" in mime_type.lower():
            import re as _re
            rate_m    = _re.search(r"rate=(\d+)", mime_type)
            sample_rate = int(rate_m.group(1)) if rate_m else 24000
            num_samples = len(raw_bytes) // 2
            # WAV header (PCM 16-bit mono)
            header = struct.pack(
                "<4sI4s4sIHHIIHH4sI",
                b"RIFF", 36 + len(raw_bytes), b"WAVE",
                b"fmt ", 16, 1, 1,
                sample_rate, sample_rate * 2, 2, 16,
                b"data", len(raw_bytes),
            )
            audio_bytes = header + raw_bytes
            return Response(content=audio_bytes, media_type="audio/wav")

        # ประเภทอื่น (audio/wav, audio/mp3 ฯลฯ) → ส่งตรง
        return Response(content=raw_bytes, media_type=mime_type)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini TTS error: {e}")


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