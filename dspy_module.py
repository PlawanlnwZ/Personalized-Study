import dspy
import json
import os
import random
import sys
import time
from typing import Optional

import httpx
import requests

try:
    from dotenv import load_dotenv
    # override=True so .env wins over stale shell env vars (e.g. an old exhausted
    # YOUTUBE_API_KEY lingering in the PowerShell session). On Render there is no .env
    # file, so this is a no-op and dashboard env vars are used as normal.
    load_dotenv(override=True)
except ImportError:
    pass  # ไม่มี dotenv ก็ยังใช้ env var ปกติได้


# ──────────────────────────────────────────────
# 0. YouTube helpers (ใช้ร่วมกันระหว่าง main.py runtime API และ video eval)
# ──────────────────────────────────────────────
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
        for query in queries[:7]:
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


def _fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n) if n else "?"


async def _fetch_video_stats(video_ids: list[str]) -> dict:
    """Batch-fetch viewCount and likeCount for up to 50 video IDs in one API call."""
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key or not video_ids:
        return {}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://www.googleapis.com/youtube/v3/videos",
                params={"part": "statistics", "id": ",".join(video_ids[:50]), "key": api_key},
            )
            data = resp.json()
            result = {}
            for item in data.get("items", []):
                stats = item.get("statistics", {})
                result[item["id"]] = {
                    "views": int(stats.get("viewCount", 0)),
                    "likes": int(stats.get("likeCount", 0)),
                }
            return result
    except Exception as e:
        print(f"[video_stats] fetch error: {e}")
        return {}


_TRANSCRIPT_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".transcript_cache.json")
_transcript_cache: dict = {}
_transcript_cache_lock = __import__("threading").Lock()  # parallel fetches write the cache

def _load_transcript_cache() -> None:
    global _transcript_cache
    with _transcript_cache_lock:
        if _transcript_cache or not os.path.exists(_TRANSCRIPT_CACHE_FILE):
            return
        try:
            with open(_TRANSCRIPT_CACHE_FILE, encoding="utf-8") as f:
                _transcript_cache = json.load(f)
            print(f"[transcript_cache] loaded {len(_transcript_cache)} entries")
        except Exception:
            pass

def _cache_transcript(video_id: str, text: str) -> None:
    """Store one transcript and persist the cache (thread-safe snapshot write)."""
    with _transcript_cache_lock:
        _transcript_cache[video_id] = text
        try:
            with open(_TRANSCRIPT_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(dict(_transcript_cache), f, ensure_ascii=False)
        except Exception as e:
            print(f"[transcript_cache] save error: {e}")


# Cap how much of each video Gemini ingests. Passing a YouTube URL tokenizes the WHOLE
# video by default; videoMetadata start/end offset limits ingestion → controls cost/quota.
# First N seconds is plenty to judge relevance + VARK. Tune via GEMINI_TRANSCRIBE_SECONDS.
_GEMINI_CLIP_SECONDS = int(os.environ.get("GEMINI_TRANSCRIBE_SECONDS", "120"))
# gemini-2.5-flash handles Thai audio well and accepts video input. Do NOT default to the
# judge's gemini-3.1-flash-lite — lite models often reject video/YouTube input.
_GEMINI_TRANSCRIBE_MODEL = os.environ.get("GEMINI_TRANSCRIBE_MODEL", "gemini-2.5-flash")
# Retries when the free-tier per-minute request cap returns 429 (window resets each minute).
_GEMINI_MAX_RETRIES = int(os.environ.get("GEMINI_TRANSCRIBE_RETRIES", "2"))
# Per-attempt backoff (seconds) base when rate-limited. The RPM window is ~60s, so long
# sleeps stack up and blow the /generate request past the frontend's 180s abort. Keep this
# small at runtime so a rate-limited video is dropped quickly (the transcript budget in
# filter_videos_by_relevance then judges it on title/views). Eval can raise it via env.
_GEMINI_BACKOFF_BASE = float(os.environ.get("GEMINI_TRANSCRIBE_BACKOFF", "6"))


def _fetch_gemini_transcript(video_id: str, max_chars: int = 2000) -> str:
    """Transcribe a YouTube video by passing its URL straight to Gemini. Returns '' on failure.

    No audio download (so no datacenter-IP 403) and no torch — Gemini fetches the video
    server-side, so this works on Render too. Thread-safe (plain HTTP), so callers may run
    it in parallel.

    Free-tier Gemini has a per-minute request cap; parallel callers burst past it and get
    429s. We retry with jittered backoff (the window resets each minute) so a burst doesn't
    lose videos. If still 429 after retries, return '' uncached → next run retries that video.
    """
    key = _gemini_api_key()
    if not key:
        return ""
    # Outer guard: ANY failure (SDK import, client/types construction, SDK drift, request)
    # must return '' so a single bad video can't crash the parallel pool.map in the eval.
    try:
        import time
        import random
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        url = f"https://www.youtube.com/watch?v={video_id}"
        parts = [
            types.Part(
                file_data=types.FileData(file_uri=url),
                video_metadata=types.VideoMetadata(
                    start_offset="0s", end_offset=f"{_GEMINI_CLIP_SECONDS}s"
                ),
            ),
            types.Part(text=(
                "Transcribe the spoken audio of this educational video (Thai or English). "
                "Output ONLY the transcript text — no commentary, no timestamps."
            )),
        ]
        for attempt in range(_GEMINI_MAX_RETRIES + 1):
            try:
                resp = client.models.generate_content(
                    model=_GEMINI_TRANSCRIBE_MODEL,
                    contents=types.Content(parts=parts),
                )
                text = (resp.text or "").strip()
                print(f"[gemini-transcript] {video_id} ({len(text)} chars)")
                return text[:max_chars]
            except Exception as e:
                msg = str(e)
                is_rate_limited = "429" in msg or "RESOURCE_EXHAUSTED" in msg
                if is_rate_limited and attempt < _GEMINI_MAX_RETRIES and _GEMINI_BACKOFF_BASE > 0:
                    # RPM window resets each minute; short jittered backoff keeps latency
                    # bounded so a rate-limited video is dropped fast instead of stalling
                    # /generate. Raise GEMINI_TRANSCRIBE_BACKOFF for eval if you want
                    # full retries; set it to 0 to fail immediately on 429.
                    time.sleep(_GEMINI_BACKOFF_BASE * (attempt + 1) + random.uniform(0, 3))
                    continue
                tag = "rate-limited (429)" if is_rate_limited else msg[:140]
                print(f"[gemini-transcript] {video_id}: {tag}", flush=True)
                return ""
    except Exception as e:
        print(f"[gemini-transcript] {video_id}: setup error: {str(e)[:120]}")
        return ""


def _fetch_transcript(video_id: str, max_chars: int = 2000) -> str:
    """Fetch transcript: cache → youtube-transcript-api → Gemini. Returns '' on all failures.

    youtube-transcript-api is free/fast but blocked from shared datacenter IPs (Render, AWS,
    GCP); set YOUTUBE_PROXY_URL=http://user:pass@host:port (Webshare, Oxylabs, Bright Data) or
    HTTPS_PROXY to route through a residential proxy. When it fails or returns nothing, Gemini
    transcribes the video server-side from its URL (no download, no 403, works on Render).
    """
    _load_transcript_cache()
    if video_id in _transcript_cache:
        return _transcript_cache[video_id][:max_chars]

    langs = ["th", "en", "en-US", "en-GB"]
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        proxy_url = (
            os.environ.get("YOUTUBE_PROXY_URL")
            or os.environ.get("HTTPS_PROXY")
            or os.environ.get("https_proxy")
        )
        proxy_cfg = None
        if proxy_url:
            try:
                from youtube_transcript_api.proxies import GenericProxyConfig
                proxy_cfg = GenericProxyConfig(http_url=proxy_url, https_url=proxy_url)
            except Exception as e:
                print(f"[transcript] proxy config error (continuing without proxy): {e}")

        api = YouTubeTranscriptApi(proxy_config=proxy_cfg)
        fetched = api.fetch(video_id, languages=langs)
        text = " ".join(s.text for s in fetched)
        if text:
            _cache_transcript(video_id, text)
            return text[:max_chars]
    except Exception:
        pass

    text = _fetch_gemini_transcript(video_id, max_chars)
    if text:
        _cache_transcript(video_id, text)
    return text


# ──────────────────────────────────────────────
# 1. Generator model slot — เสียบ AI ได้หลายตัวเพื่อเทียบผลลัพธ์ใน eval
# ──────────────────────────────────────────────
# "Slot" สำหรับเสียบ generator model ที่อยากเอามาเทียบกัน:
#   - เพิ่ม entry เพื่อเทียบหลายตัว / ลบออกเหลือ 1 ตัวก็ได้ (add up or down to 1)
#   - แต่ละ key = label ที่จะโชว์ใน report (เช่น "Typhoon --> 9.7")
#   - แต่ละ value:
#       model        : litellm model id (ใช้ prefix "openai/" สำหรับ endpoint แบบ OpenAI-compatible)
#       api_base     : base URL ของ provider
#       api_key_env  : ชื่อ env var ที่เก็บ API key ของ provider นั้น
#       max_tokens / temperature : ตามต้องการ
# ตัวที่ไม่มี API key ใน env จะถูก "ข้าม" อัตโนมัติตอน eval (ไม่ error ทั้ง run)
GENERATOR_MODELS: dict[str, dict] = {
    "Typhoon": {
        "model": "openai/typhoon-v2.5-30b-a3b-instruct",
        "api_base": "https://api.opentyphoon.ai/v1",
        "api_key_env": "TYPHOON_API_KEY",
        "max_tokens": 16384,   # Typhoon API hard cap
        "temperature": 0.7,
    },
    "GPTOSS": {
        "model": "openai/openai/gpt-oss-120b",
        "api_base": "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY",
        "max_tokens": 16384,
        "temperature": 0.7,
    },
    # "Nemotron3-Ultra": {
    #     "model": "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
    #     "api_base": "https://openrouter.ai/api/v1",
    #     "api_key_env": "OPENROUTER_API_KEY",
    #     "max_tokens": 16384,
    #     "temperature": 0.7,
    # },
    # "Qwen3-next-instruct": {
    #     "model": "openrouter/qwen/qwen3-next-80b-a3b-instruct:free",
    #     "api_base": "https://openrouter.ai/api/v1",
    #     "api_key_env": "OPENROUTER_API_KEY",
    #     "max_tokens": 16384,
    #     "temperature": 0.7,
    # },
}

# ★★ MODEL ที่เว็บใช้ตอน Generate — "คนเขียนโค้ด" แก้ตรงนี้ที่เดียว ★★
# ต้องเป็น key ใน GENERATOR_MODELS ข้างบน (เช่น "Typhoon" หรือ "Qwen2.5")
# (ค่านี้ยังใช้เป็น fallback label ใน eval report ด้วย)
DEFAULT_GENERATOR = "Typhoon"


def build_generator_lm(label: str) -> dspy.LM:
    """สร้าง dspy.LM จาก slot GENERATOR_MODELS ตาม label
    raise ValueError ถ้า label ไม่มีใน slot หรือไม่มี API key ใน env
    """
    cfg = GENERATOR_MODELS.get(label)
    if not cfg:
        raise ValueError(
            f"unknown generator '{label}' — มีใน slot: {list(GENERATOR_MODELS)}"
        )
    env = cfg["api_key_env"]
    key = os.environ.get(env) or os.environ.get(env + " ")
    if not key:
        raise ValueError(f"{env} required for generator '{label}'")
    return dspy.LM(
        model=cfg["model"],
        api_key=key.strip(),
        api_base=cfg.get("api_base"),
        max_tokens=cfg.get("max_tokens", 16384),
        temperature=cfg.get("temperature", 0.7),
        cache=False,
        timeout=120,
    )


def configure_lm(api_key: Optional[str] = None):
    """ตั้ง global generator LM ที่เว็บใช้ตอน Generate (study guide + quiz)

      → เปลี่ยน model ที่เว็บใช้ = แก้ค่า DEFAULT_GENERATOR ด้านบน
        ให้ชี้ไป label ใดก็ได้ใน GENERATOR_MODELS (เช่น "Typhoon" / "Qwen2.5")
      (ถ้าอยากใช้รุ่นใหม่ ให้เพิ่ม entry ใน GENERATOR_MODELS ก่อน แล้วตั้ง DEFAULT_GENERATOR)
    api_key override ใช้กับ Typhoon (backward-compat กับ main.py startup)
    """
    cfg = GENERATOR_MODELS[DEFAULT_GENERATOR]
    key = api_key or os.environ.get(cfg["api_key_env"])
    if not key:
        raise ValueError("API Key is required")

    lm = dspy.LM(
        model=cfg["model"],
        api_key=key,
        api_base=cfg["api_base"],
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        cache=False,
        timeout=120,
    )
    dspy.settings.configure(lm=lm)
    print(f"[runtime] generator model = {DEFAULT_GENERATOR} ({cfg['model']})")
    return lm


def _gen_context(gen_lm):
    """context manager สำหรับรัน generator ด้วย LM ที่ระบุ
    (ถ้า gen_lm=None → ใช้ global LM เดิม)
    """
    import contextlib
    return dspy.context(lm=gen_lm) if gen_lm is not None else contextlib.nullcontext()


def _gen_error_note(e: Exception) -> tuple[bool, str]:
    """ตรวจว่า exception จาก generator (ตอน generate เนื้อหา/quiz/query/relevance) เป็น
    rate-limit จาก provider ไหม 

    ใช้ตรงจุดที่เรียก generator module ในลูป eval — เพื่อ "ข้าม" ตัวอย่างที่พังแทนที่จะ
    ปล่อยให้ exception หลุดขึ้นไปทำให้ทั้งสคริปต์ crash (เสีย progress ของ model/ตัวอย่างอื่น)
    """
    msg = str(e)
    is_rl = (
        "429" in msg
        or "RateLimitError" in type(e).__name__
        or "rate-limited" in msg.lower()
        or "RESOURCE_EXHAUSTED" in msg
    )
    if is_rl:
        return True, (
            f"⚠️ ติด Rate Limit (429) จาก provider — ข้ามตัวอย่างนี้ไป (ไม่กระทบตัวอย่าง/โมเดลอื่น)\n\n"
            f"รายละเอียด: {msg[:300]}"
        )
    return False, f"❌ Generator error: {type(e).__name__}: {msg[:300]}"

import threading

_retry_after_store: dict[int, int] = {}  # thread_id → retry_after seconds
_retry_after_lock = threading.Lock()

_original_send = httpx.Client.send

def _patched_send(self, request, **kwargs):
    response = _original_send(self, request, **kwargs)
    if response.status_code in (429, 503):
        val = response.headers.get("Retry-After")
        if val and val.isdigit():
            with _retry_after_lock:
                _retry_after_store[threading.get_ident()] = int(val)
    return response

httpx.Client.send = _patched_send


def _pop_retry_after() -> int | None:
    """Pop the Retry-After value captured for this thread, or None if not set."""
    with _retry_after_lock:
        return _retry_after_store.pop(threading.get_ident(), None)


def _call_with_retry(fn, *args, max_retries=3, base_backoff=10, **kwargs):
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs), None
        except Exception as e:
            msg = str(e)
            is_retryable = (
                "429" in msg
                or "503" in msg
                or "RateLimitError" in type(e).__name__
                or "ServiceUnavailableError" in type(e).__name__
                or "AdapterParseError" in type(e).__name__
                or "rate-limited" in msg.lower()
                or "service unavailable" in msg.lower()
                or "overloaded" in msg.lower()
                or "RESOURCE_EXHAUSTED" in msg
            )
            if not is_retryable:
                _, note = _gen_error_note(e)
                return None, note

            if attempt >= max_retries:
                _, note = _gen_error_note(e)
                return None, note

            # Check if the patch captured a Retry-After header
            retry_after = _pop_retry_after()
            if retry_after is not None:
                wait = retry_after + random.uniform(0, 3)
                tag = "429" if "429" in msg else "503"
                print(f"[retry] {tag} hit — Retry-After header: {retry_after}s, "
                      f"waiting {wait:.1f}s (attempt {attempt+1}/{max_retries})")
            else:
                wait = base_backoff * (2 ** attempt) + random.uniform(0, 5)
                tag = "429" if "429" in msg else "503"
                print(f"[retry] {tag} hit — no Retry-After header, "
                      f"backing off {wait:.1f}s (attempt {attempt+1}/{max_retries})")

            time.sleep(wait)
#VARK PROMPT
class VARKProjector(dspy.Signature):
    """Accepts raw text content and a VARK learning style profile, then generates a customized, 
    style-optimized learning module using an internal, single-stage Chain-of-Thought analysis. 
    Additionally, generates optimized search queries for retrieving relevant YouTube video resources."""

    context: str = dspy.InputField(
        desc="Raw text content extracted from PDFs, documents, or source materials (plain text)"
    )

    vark_style: str = dspy.InputField(
        desc='JSON string representing the learner\'s VARK profile, e.g., {"V":40,"A":20,"R":30,"K":10,"dominant":"V"}'
    )
    
    learning_material: str = dspy.OutputField(
        desc="""
    Generate a comprehensive learning module based on the provided context/PDF by strictly adhering to the following rules and absolute constraints:
    1. Language Consistency:
    - Use the same language as the context/PDF. If the context is in English, the entire output must be written in English. Maintain technical terms or source examples in their original language as required by the context.

    2. Completeness and Length (Highest Priority):
    - Cover every key point and concept present in the context. Do not omit, truncate, or over-summarize any information. Prefer rich, detailed explanations with examples over terse bullet points.
    - The length of the generated output must scale according to the size of the provided context:
    * Short context (<500 words) -> Output must be >= 1,200 words.
    * Medium context (500–2,000 words) -> Output must be >= 2,000 words.
    * Long context (>2,000 words) -> Output must be >= 70% of the context length or >= 3,000 words (whichever is greater).
    - **Per-section balance:** Each of the five sections (R1, R2, R3, A1, K1) must be substantial in its own right — none may be a short stub. In particular R2, R3, and K1 must be developed in full (multiple paragraphs / multiple worked examples / multiple steps), so that no single tab renders nearly empty. R1 and A1 remain the longest, but every section must stand on its own as a complete, well-developed resource.

    3. Mathematical Formulations:
    - Use \\( ... \\) for inline equations and \\[ ... \\] for display equations.
    - **Absolute Rule:** The opening delimiter, the mathematical formula, and the closing delimiter must all reside on the exact same line. For example: \\[ z = \\frac{401.8 - 393.3}{393.3} \\approx 0.0216 \\]
    - Do not insert any line breaks between the delimiters and the formula, as this breaks KaTeX rendering. For multiple equations, separate them by assigning exactly one fully self-contained equation per line.

    4. MANDATORY DOCUMENT STRUCTURE — exactly five top-level sections, each opened by an EXACT machine marker heading on its own line, in this exact order:
        ## R1: Full Version
        ## R2: Conclusion Version
        ## R3: Tutorial Version
        ## A1: บทบรรยายเนื้อหาหลักสูตร
        ## K1: แบบจำลอง / ทดลอง
    - **Absolute Rule:** Emit these five marker headings VERBATIM (same prefix `## R1:` / `## R2:` / `## R3:` / `## A1:` / `## K1:`). A downstream parser splits the document on them, so do not rename, reorder, translate, merge, drop, or add marker headings, and do not wrap them in bold or quotes. The text after the colon is a human label; keep it but you may localize it to the context language. Output nothing before `## R1:` and nothing after the K1 section ends.
    - Do NOT emit a V (video) section: video content is supplied separately by the system (YouTube), not by you.
    - Within each section you MAY use lower-level headings (### …) named naturally after the actual subject matter. Every distinct topic from the context must be covered.
    - All five sections cover the SAME underlying topics — they are different presentations of one lesson, not different lessons. Requirements per section:
    * **R1: Full Version** — the complete, authoritative read/write reference and the longest section. Clear textual explanations in structured Markdown (headings, bullet/numbered lists, definition lists). **Bold key terms**, give precise definitions, and present comparison data as real Markdown pipe tables (| col | col |) and hierarchical bullet lists where they clarify relationships. Must cover every key point. Do not instruct learners to 'summarize in their own words' or 'paraphrase'.
    * **R2: Conclusion Version** — a condensed pre-exam review / cheat sheet that distills the key takeaways and core concepts of EVERY topic into highly scannable bullet points or compact summary tables. Must be complete to the end (never truncated). Introduce no material beyond what R1 covers.
    * **R3: Tutorial Version** — a thorough, step-by-step, do-it-yourself walkthrough that teaches the same concepts as a guided tutorial. For EACH major concept or procedure in the context, include: (a) a short explanation of the method/approach and WHY each step is done, (b) at least one fully worked example that shows every step and the final result inline, and (c) where it helps understanding, a second worked scenario solved completely. Use numbered steps and show the arithmetic/derivation explicitly. **This section must be substantial — several hundred words — not a bare list of formulas.** **Absolute Rule for R3: because every solution is already shown inline as a worked example, DO NOT include ANY answer-key / hidden-answer block here — no 'ans' delimiters, no "ดูเฉลย", no spoiler block anywhere in R3.** (Answer keys belong only in K1, per rule 6, where the learner must deduce the answer themselves.) End with a brief "Lesson Learnt" line stating the practical takeaway.
    * **A1: บทบรรยายเนื้อหาหลักสูตร** — a spoken-style lecture/narration that delivers the FULL original source content, written to be read aloud. Constraints:
        (0) **Full coverage, no abridgement:** Narrate the ENTIRE source content from beginning to end in its original order — every topic, sub-point, definition, example, and number present in the context must be spoken. This is NOT a summary or highlight reel: do not condense, skip, generalize, or shorten. A1 must be at least as complete as R1 and is typically the longest spoken section; when in doubt, narrate more, not less. (Equations may be voiced in words.)
        (1) At least 70% must be standard narrative paragraphs (NO `>` blockquote symbol) that establish context, explain theory, and elaborate concepts in a flowing, conversational lecture voice.
        (2) You may intersperse targeted character dialogue (e.g., instructor and student). Dialogue MUST be a blockquote beginning with a short situational description, then character lines on subsequent rows, e.g.:
            > Situation: The instructor asks the class about Subject-Verb Agreement.
            > Instructor: "Why do we say 'He is happy' but 'They are happy'?"
            > Student: "I'm not entirely sure. What is the core rule behind that?"
        (3) Keep each dialogue turn to 1–3 lines. Scale the number of dialogue blocks with the 'A' score in vark_style: A >= 70 → 3–5 blocks; A 40–69 → 2–3 blocks; A 20–39 → 1–2 blocks; A < 20 → exactly 1 block. Never place more than 2 blockquotes consecutively.
        (4) **Never use the `>` symbol** for headings, general text, tips, notes, summaries, example lists, or bullets — it is reserved exclusively for the Situation+Dialogue format above.
    * **K1: แบบจำลอง / ทดลอง** — a hands-on simulation/experiment: a concrete, do-able activity (an experiment, a simulation scenario to solve, a bug-hunt, or real-data calculations) with explicit step-by-step instructions and an interactive practice section. **Every activity MUST conclude with an explicitly labeled "Lesson Learnt" section** stating the real practical insight gained (not a generic platitude). Include answer keys per rule 6 for any task the learner must solve.

    5. Visual & formatting absolute rules (apply across ALL sections):
    - Use the dominant trait in vark_style to decide which presentation to make richest — but all five sections must still be fully realized; do not hollow out any section.
    - **Never reference or simulate images in any form.** No placeholder text or descriptions of non-existent imagery/diagrams/graphics (avoid prefixes like 'Image:', 'Figure:', 'Diagram of...', 'Photo of...'). Convey visual relationships exclusively via real Markdown pipe tables (| col | col |) or hierarchical bullet lists.
    - **Never render tabular or comparative data as ASCII art (e.g. +---+---+ box-drawing) or inside code fences (```)** — ASCII grids rely on fixed-width alignment that breaks with proportional Thai glyphs. Every comparison or data table MUST be a real Markdown pipe table.

    6. Exercise Formats and Answer Key Constraints:
    - **Absolute Rule: Matching exercises of any kind are strictly prohibited.** Do not create matching tables, column-matching tasks, or term-to-definition matching. Instead, utilize alternative formats such as fill-in-the-blanks, short-answers, sequence ordering/sorting, text completion, or situational scenarios.
    - **Strict Answer Key Formatting:** 
    * **Answer blocks ('ans') are allowed ONLY in the K1 section** — never in R1, R2, R3, or A1. In K1, for any interactive task where the student must deduce the answer themselves, you must provide an answer key immediately following the task. Do NOT add an answer block when the solution is already explicitly visible in the preceding text (e.g., a worked example or an explanatory table that already shows the full result). R3 always shows its solutions inline and therefore never contains an answer block.
    * The answer block must be strictly delimited by the word 'ans' enclosed in single quotes. The delimiter row must stand completely alone on its own line, contain no other text, and start at column 0 (left-aligned without indentation or markdown code fences).
    * The precise format must look exactly like this:
        'ans'
        <Insert correct answers and explanations here>
        'ans'
    ***The question itself must hide the solution.** For fill-in-the-blank questions, use underscores (e.g., 'My glasses ____ on the table.'). For error-correction, present the problem clearly without revealing the answer in the prompt text (e.g., use 'The manager ___ busy. (Original: are)' instead of showing the corrected sentence directly in the question). Keep the correct, final versions or solutions strictly inside the 'ans' block. Never expose answers in standard body paragraphs using phrases like '**Answer:** ...'.
    """
    )


# ──────────────────────────────────────────────
# 3. DSPy Modules
# ──────────────────────────────────────────────

class VARKModule(dspy.Module):
    """
    Runtime module — single stage (ChainOfThought)
      สร้างสื่อการเรียนรู้จาก context + vark_style โดยตรง
      (CoT ทำให้ model วิเคราะห์/วางแผนเองในตัว ไม่ต้องมี analyze stage แยก)
    """
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(VARKProjector)

    def forward(self, context: str, vark_style: str) -> dspy.Prediction:
        ctx_chars = len(context or "")
        ctx_words = len((context or "").split())

        draft = self.generate(context=context, vark_style=vark_style)

        mat = draft.learning_material or ""
        mat_chars = len(mat)
        # char-based ratio — robust ทั้งไทย (ไม่มี space) และอังกฤษ
        char_ratio = (mat_chars / ctx_chars) if ctx_chars else 0.0

        # truncation signals ที่เชื่อได้จริง (ไม่ขึ้นกับภาษา):
        #   1. ``` ไม่จับคู่ (จำนวน fence เป็นเลขคี่) → code block ค้างกลาง
        #   2. ลงท้ายด้วย ``- `` หรือ ``1. `` หรือ ``> `` แล้วไม่มีเนื้อหา → list item ว่าง
        #   3. context ใหญ่แต่ material char สั้นกว่ามาก
        fence_count = mat.count("```")
        unclosed_fence = (fence_count % 2) != 0
        tail = mat.rstrip()[-3:]
        dangling_marker = tail.endswith(("- ", "* ", "> ")) or tail.endswith("..")

        warns = []
        if unclosed_fence:
            warns.append("unclosed ``` code fence")
        if dangling_marker:
            warns.append("ends with empty list/blockquote marker")
        if ctx_chars > 2000 and char_ratio < 0.5:
            warns.append(f"material chars only {char_ratio:.0%} of context")
        warn = ("  ⚠️ " + "; ".join(warns)) if warns else ""

        print(
            f"[VARKModule] done | ctx={ctx_chars} chars / {ctx_words} words "
            f"| material={mat_chars} chars | char_ratio={char_ratio:.2f}x context "
            f"| fences={fence_count}{warn}"
        )
        # ── print ผลลัพธ์ที่ generate ออกมา ──
        print("─" * 70)
        print("[VARKModule] learning_material:\n")
        print(mat)
        print("─" * 70)

        return dspy.Prediction(
            learning_material=draft.learning_material,
        )


#QUIZ PROMPT
class QuizGenerator(dspy.Signature):
    """Generate an MCQ quiz from the learning material, following the VARK style."""
    learning_material: str = dspy.InputField(desc="Markdown learning material")
    vark_style: str        = dspy.InputField(
        desc='JSON e.g. {"V":40,"A":20,"R":30,"K":10,"dominant":"V"}'
    )
    count: int             = dspy.InputField(desc="Number of questions to generate (max 10)")
    questions: str         = dspy.OutputField(
        desc=r"""An MCQ quiz as a JSON array — generate exactly `count` questions (max 10).
    **Output format (strict):** Respond with a JSON array ONLY. No markdown fences (```), no leading/trailing text or explanation, no trailing comma. It must be directly parseable by JSON.parse.
    **Format it to be readable (pretty-print):** indent 2 spaces, each object / each key on its own line — do not cram everything onto one line (it must still be 100% valid JSON).
    Each item is an object containing every key: {"q":"<question>","options":{"A":"..","B":"..","C":"..","D":".."},"answer":"A|B|C|D","vark":"V|A|R|K","concept":"<short topic label>","explanation":"1–2 sentences"}
    Example of the desired format (blank line + indentation like this):
    [
      {
        "q": "...",
        "options": {
          "A": "...",
          "B": "...",
          "C": "...",
          "D": "..."
        },
        "answer": "B",
        "vark": "R",
        "concept": "...",
        "explanation": "..."
      }
    ]

    **Completeness & relevance:**
    1) Every question must be answerable from `learning_material` ALONE — do not rely on knowledge outside the lesson or ask about things not in the content.
    2) Questions must capture the **key points / main concepts** of the lesson — do not ask about trivial details, minor numbers, or unimportant wording.
    3) Each question must be clear and unambiguous, and **do not ask about the same point more than once** — spread them to cover multiple topics in the content.

    **Adjust the question style to the dominant VARK trait in `vark_style`**:
    - Visual (V): Focuses on charts, diagrams, hierarchies, shapes, and clear spatial layouts. Avoids heavy text blocks.
    - Auditory (A): Focuses on discussions, lectures, verbal analogies, podcasts, and rhythm/sound metaphors.
    - Read/Write (R): Focuses on text-based explanations, lists, definitions, essays, and manuals.
    - Kinesthetic (K): Focuses on concrete examples, real-world applications, hands-on scenarios, case studies, and physical trials.

    **Choice quality (4 options A–D):**
    4) The distractors (the 3 wrong options) must be reasonable, believable, and close to the real answer — they must not be so obviously wrong that the answer can be guessed instantly.
    5) All 4 options must be genuinely distinct, in both wording and meaning — no duplicate / synonymous options.
    6) The options must be similar in length (do not let the correct one be noticeably longer / more detailed) and **do not use shortcut options like "All of the above / None of the above"**.

    **Answer-key correctness:**
    7) `answer` (the letter A/B/C/D) must match the option that is correct per the actual content, for every question.
    8) `explanation` must explain reasoning consistent with the answer, concise, 1–2 sentences (why that option is correct and/or why the others are wrong).

    **Format & language:**
    9) The JSON must be valid and every item must include all keys (q, options{A,B,C,D}, answer, vark, concept, explanation).
    10) Each item's `vark` tag must match the style the question actually uses, and **the language must match `learning_material`** (Thai content → questions/options/explanations entirely in Thai; English content → English; except technical terms / code, which keep their original language).
    12) `concept` is a SHORT label (2–6 words, in the content's language) naming the single specific topic/idea that question tests — e.g. "กฎข้อ 2 ของนิวตัน (F=ma)", "Present Simple: เติม -s/-es", "การหาจุดดุลยภาพ". Questions on the same idea must share the SAME concept string verbatim so weak areas can be grouped. It names the idea, not the modality.
    11) **Mathematical formulas (KaTeX):** if a question/option/explanation contains an equation, write inline with `\\( ... \\)` and display with `\\[ ... \\]` — because the formula lives inside a JSON string, you **must always escape the backslash as two backslashes** (write `\\(`, `\\)`, `\\[`, `\\]`, `\\frac`, `\\times`, `\\approx`, etc.) so the JSON stays valid, and after JSON.parse a single backslash remains for KaTeX to render. **The entire equation must be in one string / on one line** — do not break to a new line in the middle of a formula, e.g. `"explanation": "Derived from \\( z = \\frac{v}{c} \\approx 0.0216 \\), therefore we conclude..."`."""
    )


class QuizModule(dspy.Module):
    """Runtime module สำหรับสร้าง quiz — ใช้ ChainOfThought เพื่อให้ output เป็น JSON ที่ valid"""
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(QuizGenerator)

    def forward(self, learning_material: str, vark_style: str,
                count: int = 10) -> dspy.Prediction:
        # cap จำนวนข้อไว้ที่สูงสุด 10 ข้อ
        count = max(1, min(int(count), 10))
        return self.generate(
            learning_material=learning_material,
            vark_style=vark_style,
            count=count,
        )


# REMEDIATION PROMPT — adaptive loop: re-teach only what the learner missed on the quiz
class RemediationProjector(dspy.Signature):
    """Re-teach ONLY the concepts a learner answered incorrectly on the quiz, approaching
    each from a DIFFERENT angle than the original lesson and directly correcting the specific
    misconception their wrong choice reveals. This is short, targeted review — not a re-run
    of the whole lesson."""

    learning_material: str = dspy.InputField(
        desc="The original lesson the learner already studied (Markdown)."
    )
    vark_style: str = dspy.InputField(
        desc='JSON VARK profile, e.g. {"V":40,"A":20,"R":30,"K":10,"dominant":"V"}'
    )
    weak_spots: str = dspy.InputField(
        desc='JSON array of the questions the learner got wrong: '
             '[{"concept":"<topic label>","chosenText":"<the wrong option they picked>"}]. '
             'Multiple entries may share the same concept.'
    )
    remediation: str = dspy.OutputField(
        desc=r"""A focused review module (Markdown) that re-teaches ONLY the weak concepts.

    Rules:
    1) Language: match `learning_material` exactly (Thai content → ALL prose, headings, table cells, and labels in Thai; English → English). Keep only technical terms/code/equations/variable names in their original form. Do NOT translate or re-title concepts into another language.
    2) Group by concept: emit ONE `### <concept>` subsection per DISTINCT concept in weak_spots (dedupe — if several wrong answers share a concept, cover it once, addressing each distinct misconception inside it). **The heading after `### ` MUST be the `concept` string copied VERBATIM from weak_spots** — do not rename, translate, expand, or invent a new title for it. Output nothing before the first `### ` heading.
    3) Inside each subsection, in this order:
       - A short re-explanation that takes a DIFFERENT approach than a plain restatement — use a fresh analogy, a contrasting example, or a step-by-step derivation the original may not have emphasized. Do NOT just repeat the original wording.
       - A line explicitly named **ทำไมตัวเลือกที่เลือกถึงผิด** (or "Why that answer is wrong" in English) that targets the misconception implied by `chosenText` — name the wrong idea and correct it directly.
       - One concrete worked example OR a memory hook the learner can hold onto.
    4) Keep it concise — this is remediation, not a new lesson. Aim for ~80–200 words per concept. Cover every distinct concept; do not introduce unrelated topics.
    5) Use the dominant trait in `vark_style` to choose HOW to re-explain (V → a Markdown pipe table or hierarchy; A → a short spoken-style walkthrough; R → precise definitions/lists; K → a hands-on mini scenario), but never reference or describe images, and render any comparative data as real Markdown pipe tables (never ASCII art or code fences).
    6) Mathematical formulas (KaTeX): inline `\\( ... \\)`, display `\\[ ... \\]`, each equation fully on one line.
    """
    )


class RemediationModule(dspy.Module):
    """Adaptive-loop module — re-teaches only the concepts missed on the quiz.

    Reuses the existing QuizModule downstream for fresh 'check' questions, so this module
    has a single job: produce targeted re-teach material from the learner's wrong answers.
    """
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(RemediationProjector)

    def forward(self, learning_material: str, vark_style: str,
                weak_spots: str) -> dspy.Prediction:
        return self.generate(
            learning_material=(learning_material or "")[:4000],
            vark_style=vark_style,
            weak_spots=weak_spots or "[]",
        )


#VIDEO PROMPT
class VideoVARKClassifier(dspy.Signature):
    """Classify a YouTube video by VARK style + subtopics.

    Page mapping (matching to PDF pages) is NOT done at this stage —
    it is a separate runtime step that takes the actual PDF and compares it against this result.
    """
    video_title: str       = dspy.InputField(desc="Video title")
    video_channel: str     = dspy.InputField(desc="Channel name")
    video_metadata: str    = dspy.InputField(
        desc="The video's description / tags / categories (combined into text)"
    )
    vark_weight_desc: str  = dspy.InputField(
        desc='The learner\'s VARK profile, e.g. "V 60%, K 30%, R 10%"'
    )
    classification: str    = dspy.OutputField(
        desc="""The video classification as a single JSON object.
    **Output format (strict):** Respond with a JSON object ONLY. No markdown fences (```), no leading/trailing text or explanation, no trailing comma. It must be directly parseable by JSON.parse.
    schema: {"vark":["V","K"],"subtopics":["..",".."]}
    example: {"vark":["V","R"],"subtopics":["if-else","loop","function"]}

    **Assigning `vark` (an array of the 1–4 styles that are "actually present" in the video):**
    Judge mainly from video_title / video_channel / video_metadata, then select only the modalities the video's content actually has —
    V (Visual) = has diagrams/animation/slides/graphics/illustrations on screen;
    A (Auditory) = has narration/speaking/lecture/talk;
    R (Read/Write) = has on-screen text/code/documents to read;
    K (Kinesthetic) = has hands-on doing/demo/walkthrough/following along step by step.
    1) You must include **every modality that actually appears** — do not miss any (e.g. a tutorial video with a demo must include all the relevant styles, not just one).
    2) You must **not include a style that is not in the video** — do not guess / no false positives (e.g. a video with no actual demo must not include K).
    3) Decide each of V/A/R/K strictly per the definitions above.
    Note: `vark_weight_desc` (the learner profile) is for supplementary reference only — assign vark based on the "actual video content", not on the learner's weights.

    **Setting `subtopics` (an array of keywords):**
    4) Must reflect the actual content/topics of the video (extracted from title/metadata), not vague generic words.
    5) 2–4 keywords.
    6) Each keyword short, ≤16 characters, and **the language matches the video** (Thai video → Thai keywords, English video → English; technical terms / command names may keep their original language)."""
    )


class VideoClassifierModule(dspy.Module):
    """Runtime module สำหรับ classify video — output สั้น ใช้ Predict ก็พอ"""
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(VideoVARKClassifier)

    def forward(self, **kwargs) -> dspy.Prediction:
        return self.classify(**kwargs)


# VIDEO QUERY PROMPT — สร้าง YouTube search queries จากเนื้อหา PDF
class VideoQueryGenerator(dspy.Signature):
    """Read the content from a PDF and generate search queries for finding YouTube
    study videos that match the content."""
    pdf_content: str     = dspy.InputField(desc="Content extracted from the PDF (plain text)")
    youtube_queries: str = dspy.OutputField(
        desc='Respond with a JSON array ONLY, no other text, no markdown fences. '
             '3-5 queries, e.g. ["if-else C tutorial", "if-else statement in C"] '
             'The language should match the content (Thai if the content is Thai).'
    )


class VideoQueryModule(dspy.Module):
    """Runtime module — สร้าง YouTube search queries จากเนื้อหา PDF (ChainOfThought)"""
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(VideoQueryGenerator)

    def forward(self, pdf_content: str) -> dspy.Prediction:
        return self.generate(pdf_content=(pdf_content or "")[:6000])


# VIDEO RELEVANCE PROMPT — กรองวิดีโอตามความเกี่ยวข้อง + จัด VARK รายคลิป
class VideoRelevanceJudge(dspy.Signature):
    """Look at a list of YouTube videos (with transcripts) and select only the clips
    that match the topic the learner is currently studying, then classify the VARK of each clip that passes."""
    topic: str                   = dspy.InputField(
        desc="The topic/content the learner is currently studying"
    )
    videos_with_transcripts: str = dspy.InputField(
        desc="A numbered list of videos (1., 2., ...) with title/channel/views/transcript"
    )
    relevant_indices: str        = dspy.OutputField(
        desc='A JSON array of the sequence numbers of clips that match the topic (select up to 10), '
             'e.g. [1,3,4]. Respond with JSON ONLY, no other text.'
    )
    vark_per_video: str          = dspy.OutputField(
        desc='A JSON object mapping sequence number→VARK styles, e.g. {"1":"V","3":"VK"} '
             '(V=diagram/animation, A=lecture/talk, R=text/code, K=hands-on/demo) '
             'Respond with JSON ONLY, no other text.'
    )


class VideoRelevanceModule(dspy.Module):
    """Runtime module — กรองวิดีโอตามความเกี่ยวข้อง + จัด VARK (ChainOfThought)"""
    def __init__(self):
        super().__init__()
        self.judge = dspy.ChainOfThought(VideoRelevanceJudge)

    def forward(self, topic: str, videos_with_transcripts: str) -> dspy.Prediction:
        return self.judge(
            topic=topic,
            videos_with_transcripts=videos_with_transcripts,
        )


import re as _re

# Judge: Gemini flash lite — Google AI Studio (key = GEMINI_API_KEY)
JUDGE_MODEL_A = os.environ.get("JUDGE_MODEL_A", "gemini/gemini-3.1-flash-lite")

# เกณฑ์รายข้อ (ตัด sub-sub items ออกแล้ว — เหลือ A.1–A.4, B.1–B.5)
VARK_CRITERIA = ["A.1", "A.2", "A.3", "A.4", "A.5","B.1", "B.2", "B.3", "B.4", "B.5"]
QUIZ_CRITERIA = [f"C.{i}" for i in range(1, 11)]
VIDEO_CRITERIA = [f"D.{i}" for i in range(1, 11)]  # D.1–D.10 (pipeline: query + relevance)


def _judge_label(model: str) -> str:
    """ดึง short friendly label จากชื่อ model สำหรับ log
    gemini/gemini-3.1-flash-lite → 'Gemini'
    """
    m = model.lower()
    if "gemini" in m:
        return "Gemini"
    if "gpt" in m:
        return "GPT"
    return model.split("/")[-1]


_JUDGE: Optional[dspy.LM] = None


def configure_judge(
    gemini_key: Optional[str] = None,
    model: Optional[str] = None,
) -> dspy.LM:
    g_key = (
        gemini_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GEMINI_API_KEY ")
    )
    if not g_key:
        raise ValueError("GEMINI_API_KEY required for judge (Google AI Studio)")
    return dspy.LM(
        model=model or JUDGE_MODEL_A,
        api_key=g_key.strip(),
        # scores + feedback (รีวิวรวมเป็น string เดียว) — เผื่อ output ยาวพอ
        max_tokens=16000,
        temperature=0.0,
        cache=False,
    )


def _get_judge() -> dspy.LM:
    global _JUDGE
    if _JUDGE is None:
        _JUDGE = configure_judge()
    return _JUDGE


def _parse_criteria_scores(text, criteria: list[str]):
    """parse judge JSON output → per-criterion score dict (แต่ละข้อ 0–10)
    รองรับ key เกิน/ขาด — ขาด = 0; ค่าที่ไม่ใช่เลข = 0; clamp ให้อยู่ใน 0–10
    คืน **None** ถ้า output parse เป็น dict ไม่ได้ (judge พัง — แยกจากกรณีได้ 0 จริง
    เพื่อไม่ให้ adapter/parse failure ปลอมเป็นคะแนน 0 ใน training metric)
    """
    data = _safe_json(str(text)) if text is not None else None
    if not isinstance(data, dict):
        return None
    per = {}
    for c in criteria:
        v = data.get(c, 0)
        try:
            per[c] = max(0, min(10, int(round(float(v)))))
        except (TypeError, ValueError):
            per[c] = 0
    return per


def _criteria_total_0_10(per: dict, criteria: list[str]) -> float:
    """รวม per-criterion (แต่ละข้อ 0–10) → เฉลี่ยเป็นคะแนนรวม 0–10"""
    if not criteria:
        return 0.0
    return round(sum(per.get(c, 0) for c in criteria) / len(criteria), 2)

#VARK JUDGE
class VARKJudgeScore(dspy.Signature):
    """You are an expert educational auditor specializing in VARK learning
    methodologies. Evaluate a generated VARK study guide (learning material)
    against the criteria below. Score **each criterion 0–10**, judging the
    QUALITY of each dimension — NOT merely whether a component is present.

    ── Anchored scoring bands (STRICT EVALUATION) ──
    9–10 = Exceptional Masterpiece: Genuinely publication-ready. Explanations show profound 
        pedagogical insight, analogies are brilliant, layouts are flawless.
    7–8  = Excellent/Good: Solid, highly accurate, and complete. This is the EXPECTED score 
        for high-quality generation with no obvious errors.
    4–6  = Average/Mediocre: Present but generic, uses shallow/robotic explanations, 
        or contains minor formatting inconsistencies.
        
    Treat the following as serious quality defects that pull a criterion DOWN
    into the mediocre band or lower (the more severe or numerous, the lower) —
    a criterion exhibiting any of them cannot be rated Excellent:
      - Truncation: content cut off mid-sentence/table/section (e.g. a cheat sheet
        or activity that stops partway).
      - Absolute-Rule violations: leaking explicit learning-style labels
        ('VARK','Visual','Auditory','Read/Write','Kinesthetic','(V)/(A)/(R)/(K)');
        matching exercises (term-to-definition); tables drawn as ASCII art /
        box-art or inside code fences instead of real Markdown pipe tables;
        simulated images ('Image:','Figure:','Diagram of...'); answers of questions exposed in
        body text instead of inside an 'ans' block.

    - A. Coverage & instructional quality (vs. the lesson's topics)
    A.1 Topic coverage — judge breadth: every main topic in `context` is present;
        dropping whole topics scores low even if what remains is good.
    A.2 Depth & correctness per topic — each topic fully and accurately explained
        with clear instructional structure; thin, padded, or truncated topics score low.
    A.3 All four learning dimensions are woven in AND each genuinely serves
        learning, not a hollow checkbox.
    A.4 End-of-lesson quiz/exercise quality — answerable from the material,
        targets key points, with a correct answer key.
    A.5 Can be used effectively for the student to learn (educational quality, clarity, and usefulness)
    
    - B. Per-component QUALITY (how WELL each is done, not mere presence)
    B.1 Read/Write: key terms emphasized and an accurate, well-structured summary
        that is genuinely useful for review (not a bare restatement).
    B.2 Visual: Markdown pipe tables / hierarchical structures that are correct,
        information-dense, and clarify relationships (a trivial 2-row filler
        table is mediocre, not excellent).
    B.3 Auditory: narration/dialogue that genuinely explains concepts and reads
        naturally; filler or off-topic dialogue scores low.
    B.4 Kinesthetic: a concrete, do-able hands-on activity (real steps/data/
        scenario) ending in a 'Lesson Learnt' that states a real insight, not a
        generic platitude.
    B.5 Pre-exam review / cheat sheet that is complete to the end (NOT truncated)
        and condenses the key points of EVERY topic into scannable form.

    Note: B judges the execution quality of each component, not its suitability
    for an individual learner (do not look at the vark_style dominant).
    """
    context = dspy.InputField(desc="Original source content (text from PDF/document)")
    vark_style = dspy.InputField(
        desc='VARK profile JSON (for reference only — B does not adjust to the learner)'
    )
    reference_rationale = dspy.InputField(
        desc=(
            "Optional gold-standard rationale (criteria the dataset author wrote as a reference). "
            "If non-empty → use it as ground truth to help judge. If empty → blind judge mode"
        )
    )
    learning_material = dspy.InputField(desc="Generated study guide to evaluate")
    scores = dspy.OutputField(
        desc='A JSON object of all 10 criteria (A.1–A.5, B.1–B.5), integer values 0–10'
    )
    feedback = dspy.OutputField(
        desc="""A review written entirely in Thai. Structure: FIRST, output a score list 
         for ALL criteria with newline per criteria — list every single criterion 
         regardless of score, e.g.:
         "Feedback: Vark"
         "A.1: 8/10"
         "A.2: 3/10"
         "A.3: 9/10"
         "A.4: 7/10"
         "A.5: 5/10"
         "B.1: 8/10"
         "B.2: 8/10"
         "B.3: 7/10"
         "B.4: 9/10"
         "B.5: 8/10"
         FINALLY, provide a single, continuous paragraph 
         Every criterion scoring below 8 must be explicitly explained by name 
         (e.g., 'A.5 got 4 because...') detailing its weaknesses, alongside 
         general strengths and improvements. Do not use bullet points or extra 
         JSON headers."""
    )

#QUIZ JUDGE
class QuizJudgeScore(dspy.Signature):
    """You are an evaluation judge for a VARK-style multiple-choice quiz.
    Evaluate against criteria C.1–C.10 below, **0–10 points each** (10 = fully passes, 0 = completely fails).

    - Question Quality
    C.1 Every question is answerable from `learning_material` (does not rely on knowledge outside the lesson)
    C.2 Questions target the lesson's key points (do not ask about trivial details / minor numbers)
    C.3 Questions are clear, unambiguous, and not repeated

    - Distractor / Choice Quality 
    C.4 Distractors (the 3 wrong options) are reasonable, close to the real answer, not obviously wrong
    C.5 All 4 choices are genuinely distinct (no duplicate wording/meaning)
    C.6 Choices are similar in length, with no "all of the above / none of the above" shortcut

    - Answer Correctness 
    C.7 `answer` letter matches the option that is correct per the content, for every question
    C.8 `explanation` gives correct reasoning consistent with the answer

    - Format & Alignment
    C.9 JSON valid + every item has all keys (q, options, answer, vark, explanation)
    C.10 `vark` tag is appropriate for the question, and the language matches `learning_material` (Thai→Thai)
    
    """
    learning_material = dspy.InputField(desc="Source material the quiz is based on")
    vark_style = dspy.InputField(desc="Target VARK profile JSON")
    reference_rationale = dspy.InputField(
        desc=(
            "Optional gold reference (e.g. required topics that should be covered). "
            "If non-empty → use as ground truth. If empty → blind judge mode. "
            "Note: no expected dominant VARK is specified — review the question set purely on quality."
        )
    )
    questions = dspy.InputField(desc="Generated quiz JSON array to evaluate")
    scores = dspy.OutputField(
        desc="A JSON object of C.1–C.10, integer values 0–10"
    )
    feedback = dspy.OutputField(
        desc="""A review written entirely in Thai. Structure: FIRST, output a score list with newline per criteria
             example
             "Feedback: Quiz"
             "C.1: 6/10"
             "C.2: 3/10"
             "C.3: 9/10"
             FINALLY, provide a single, continuous paragraph 
             Every criterion scoring below 8 must be explicitly explained by name (e.g., 'C.5 got 4 because...') 
             detailing its weaknesses, alongside general strengths and improvements. Do not use bullet points or extra JSON headers."""
    )


def _vark_judge(example, prediction, judge_lm, with_reference: bool = True): 
    """รัน VARK judge (Gemini) 1 ครั้ง บน 1 example
    คืน (per_criterion dict, total_0_10, feedback) หรือ (None, -1.0, err) ถ้า error
    feedback = รีวิวรวมเป็น string เดียว (judge เขียนเอง ไม่แยกราย criterion)
    with_reference=False → blind mode (judge ไม่เห็น expected/rationale)
    """
    judge = dspy.ChainOfThought(VARKJudgeScore)
    material = getattr(prediction, "learning_material", "") or ""
    ctx = (example.context or "")[:3000]
    reference = _format_reference(example) if with_reference else ""
    try:
        with dspy.context(lm=judge_lm):
            result = judge(
                context=ctx,
                vark_style=example.vark_style,
                reference_rationale=reference,
                learning_material=material,
                cache = False
            )
    except Exception as e:
        print(f"[vark_judge:{getattr(judge_lm, 'model', '?')}] error: {type(e).__name__}: {e}")
        return None, -1.0, f"[judge error] {type(e).__name__}: {e}"
    per = _parse_criteria_scores(getattr(result, "scores", ""), VARK_CRITERIA)
    fb = str(getattr(result, "feedback", "") or "").strip()
    if per is None:
        print(f"[vark_judge:{getattr(judge_lm, 'model', '?')}] unparseable scores: "
              f"{str(getattr(result, 'scores', ''))[:120]!r}")
        return None, -1.0, fb or "unparseable judge scores"
    return per, _criteria_total_0_10(per, VARK_CRITERIA), fb


def _quiz_judge(example, prediction, judge_lm, with_reference: bool = True):
    """รัน Quiz judge (Gemini) 1 ครั้ง บน 1 example
    คืน (per_criterion dict, total_0_10, feedback) หรือ (None, -1.0, err)
    feedback = รีวิวรวมเป็น string เดียว (judge เขียนเอง ไม่แยกราย criterion)
    """
    judge = dspy.ChainOfThought(QuizJudgeScore)
    questions = (getattr(prediction, "questions", "") or "")[:6000]
    material = (example.learning_material or "")[:4000]
    reference = _format_quiz_reference(example) if with_reference else ""
    try:
        with dspy.context(lm=judge_lm):
            result = judge(
                learning_material=material,
                vark_style=example.vark_style,
                reference_rationale=reference,
                questions=questions,
            )
    except Exception as e:
        print(f"[quiz_judge:{getattr(judge_lm, 'model', '?')}] error: {type(e).__name__}: {e}")
        return None, -1.0, f"[judge error] {type(e).__name__}: {e}"
    per = _parse_criteria_scores(getattr(result, "scores", ""), QUIZ_CRITERIA)
    fb = str(getattr(result, "feedback", "") or "").strip()
    if per is None:
        print(f"[quiz_judge:{getattr(judge_lm, 'model', '?')}] unparseable scores: "
              f"{str(getattr(result, 'scores', ''))[:120]!r}")
        return None, -1.0, fb or "unparseable judge scores"
    return per, _criteria_total_0_10(per, QUIZ_CRITERIA), fb


# VIDEO JUDGE

class VideoPipelineJudge(dspy.Signature):
    """
    You are an evaluation judge for a YouTube-video recommendation pipeline.
    The pipeline has 2 stages: (1) generate search queries from the PDF content, 
    (2) search YouTube, select relevant clips, and classify the VARK learning style of each clip.
    
    Evaluate the overall result against criteria D.1–D.10, 0–10 points each (10 = fully passes, 0 = completely fails).

    - Query Quality
    D.1 `youtube_queries` match the topic/content in `pdf_content`
    D.2 The queries are varied, covering multiple aspects/subtopics (3–5 queries)
    D.3 The language of the queries matches the content (Thai→Thai, English→English)

    - Selection Quality
    D.4 The clips selected in `relevant_indices` are actually relevant to the topic
    D.5 Transcripts are actually relevant to the topic and have transcript

    - VARK Classification
    D.6 `vark_per_video` assigns V/A/R/K of each selected clip correctly per the clip's actual nature

    - Format & Validity
    D.7 `relevant_indices` is a valid JSON array of sequence numbers that actually exist
    D.8 `vark_per_video` is a JSON object whose keys match the selected indices

    - Overall Integration & Accuracy
    D.9 Strategic alignment between the generated queries and the ultimate quality of the fetched results
    D.10 Overall calibration of the system's evaluation and format constraints

    Note: If the video list is empty, evaluate D.4–D.10 as far as possible, and focus heavily on D.1–D.3.
    """
    
    pdf_content = dspy.InputField(desc="Source content from the PDF the learner is studying")
    youtube_queries = dspy.InputField(desc="The search queries the system generated (JSON array)")
    videos_with_transcripts = dspy.InputField(desc="The list of videos found, numbered, with metadata and transcripts")
    relevant_indices = dspy.InputField(desc="Sequence numbers of clips the system selected as relevant (JSON array)")
    vark_per_video = dspy.InputField(desc="Map of sequence number to VARK assigned by the system (JSON object)")
    
    scores = dspy.OutputField(
        desc="A raw JSON object containing the integer scores for D.1 through D.10 Example: {\"D.1\":10,\"D.2\":8,...}"
    )
    feedback = dspy.OutputField(
        desc="""A review written entirely in Thai. Structure: FIRST, output a score list with newline per criteria
             example 
             "Feedback: Video"
             "D.1: 6/10"
             "D.2: 3/10"
             "D.3: 9/10"
             FINALLY, provide a single, continuous paragraph 
             Every criterion scoring below 8 must be explicitly explained by name (e.g., 'D.5 got 4 because...') 
             detailing its weaknesses, alongside general strengths and improvements. Do not use bullet points or extra JSON headers."""
    )

def _video_pipeline_judge(pdf_content, youtube_queries, videos_with_transcripts,
                          relevant_indices, vark_per_video, judge_lm):
    """รัน Video pipeline judge (Gemini, D rubric) 1 ครั้ง บน pipeline output 1 ชุด
    คืน (per_criterion dict, total_0_10, feedback) หรือ (None, -1.0, err)
    feedback = รีวิวรวมเป็น string เดียว (judge เขียนเอง ไม่แยกราย criterion)
    """
    judge = dspy.ChainOfThought(VideoPipelineJudge)
    try:
        with dspy.context(lm=judge_lm):
            result = judge(
                pdf_content=(pdf_content or "")[:4000],
                youtube_queries=(youtube_queries or "")[:1500],
                videos_with_transcripts=(videos_with_transcripts or "")[:8000],
                relevant_indices=(relevant_indices or "")[:500],
                vark_per_video=(vark_per_video or "")[:1000],
                cache = False
            )
    except Exception as e:
        print(f"[video_judge:{getattr(judge_lm, 'model', '?')}] error: {type(e).__name__}: {e}")
        return None, -1.0, f"[judge error] {type(e).__name__}: {e}"
    per = _parse_criteria_scores(getattr(result, "scores", ""), VIDEO_CRITERIA)
    fb = str(getattr(result, "feedback", "") or "").strip()
    if per is None:
        print(f"[video_judge:{getattr(judge_lm, 'model', '?')}] unparseable scores: "
              f"{str(getattr(result, 'scores', ''))[:120]!r}")
        return None, -1.0, fb or "unparseable judge scores"
    return per, _criteria_total_0_10(per, VIDEO_CRITERIA), fb


def _format_reference(example: dspy.Example) -> str:
    """รวม example.expected เป็น reference text สำหรับ vark/video judge

    รองรับ schema:
      video: vark / subtopics / rationale
      vark : rationale อย่างเดียว (ถ้ามี — โดยปกติไม่มี)
    คืน '' ถ้าไม่มี expected → judge จะทำงานในโหมด blind
    (quiz ใช้ _format_quiz_reference แยก — ไม่ตัดสินจาก dominant)
    """
    exp = getattr(example, "expected", None)
    if not isinstance(exp, dict) or not exp:
        return ""
    parts = []
    # video-style
    if exp.get("vark"):
        parts.append(f"Gold vark: {exp['vark']}")
    if exp.get("subtopics"):
        parts.append(f"Gold subtopics: {exp['subtopics']}")
    # shared
    if exp.get("rationale"):
        parts.append(f"Rationale: {exp['rationale']}")
    return "\n".join(parts)


def _format_quiz_reference(example: dspy.Example) -> str:
    """Quiz reference สำหรับ judge — **ตัด dominant ออก** (eval ไม่ตัดสินจาก dominant แล้ว)
    ใส่เฉพาะ required topics ที่ควรครอบคลุม ถ้ามี; ไม่งั้นคืน '' → judge รีวิวชุดคำถามล้วนๆ
    """
    exp = getattr(example, "expected", None)
    if not isinstance(exp, dict) or not exp:
        return ""
    parts = []
    if exp.get("topics"):
        parts.append(f"Required topics: {exp['topics']}")
    return "\n".join(parts)


# alias เพื่อ backward-compat
_format_video_reference = _format_reference

def _gemini_api_key() -> str:
    return (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GEMINI_API_KEY ")
        or ""
    ).strip()


def _next_eval_path(kind: str) -> str:
    """คืน path ใหม่ (เลขรันต่อเนื่อง) สำหรับเก็บ eval report ใต้ reports/
    เช่น reports/vark_eval1.json, reports/quiz_eval2.json, reports/video_eval1.json
    """
    os.makedirs("reports", exist_ok=True)
    n = 1
    while os.path.exists(os.path.join("reports", f"{kind}_eval{n}.json")):
        n += 1
    return os.path.join("reports", f"{kind}_eval{n}.json")


def _render_report_body(report: dict, heading_level: int = 1) -> str:
    """ส่วนเนื้อ report ของ AI ตัวเดียว (title + คะแนน + per-example output/feedback)
    heading_level: 1 = ใช้ '#'/'##'/'###' (report เดี่ยว);
                   2 = ใช้ '##'/'###'/'####' (section ในไฟล์รวมหลาย AI)
    """
    gen = report.get("generator", DEFAULT_GENERATOR)
    h1 = "#" * heading_level
    h2 = "#" * (heading_level + 1)
    h3 = "#" * (heading_level + 2)
    lines: list[str] = []
    lines.append(f"{h1} {report.get('module', 'Eval report')} — {gen}")
    lines.append("")
    lines.append(f"- Generator (AI): **{gen}**")
    lines.append(f"- Mode: `{report.get('mode', '')}`")
    n_total = report.get("n")
    n_skipped = report.get("n_skipped", 0)
    if n_skipped:
        lines.append(f"- Samples: {n_total}  (scored: {report.get('n_scored', n_total)}, "
                     f"skipped/error: {n_skipped})")
    else:
        lines.append(f"- Samples: {n_total}")
    lines.append(f"- Mean total: **{gen} --> {report.get('mean_total')}/10** "
                 f"(คำนวณจากตัวอย่างที่ generate สำเร็จเท่านั้น)")
    lines.append(f"- Judge: `{report.get('judge', {}).get('model', '')}`")
    if report.get("expected_match_avg") is not None:
        lines.append(f"- Expected-match: {report['expected_match_avg']} "
                     f"({report.get('n_labeled')}/{report.get('n')} labeled)")
    lines.append("")
    crit = report.get("criterion_avgs", {})
    if crit:
        lines.append(f"{h2} Per-criterion averages (0–10)")
        lines.append("")
        lines.append("| Criterion | Avg |")
        lines.append("| --- | --- |")
        for c, v in crit.items():
            lines.append(f"| {c} | {v} |")
        lines.append("")

    lines.append(f"{h2} Per-example output & feedback")
    lines.append("")

    for fb in report.get("feedbacks") or []:
        i = fb.get("i")
        total = fb.get("total")
        note = (fb.get("feedback") or "")
        if total is not None:
            shown = f"{total}/10"
        elif "rate limit" in note.lower():
            shown = "RATE-LIMITED (skipped)"
        else:
            shown = "ERR"
        lines.append(f"{h3} Example {i} — {gen} total {shown}")
        lines.append("")

        scores = fb.get("scores")
        if isinstance(scores, dict):
            lines.append("**Scores:** " + ", ".join(
                f"{c}: {scores.get(c, 0)}" for c in scores
            ))
            lines.append("")

        review = note.strip()
        if review:
            lines.append("**Feedback:**")
            lines.append("")
            lines.append(review)
            lines.append("")

        output = (fb.get("output") or "").strip()
        if output:
            # เรนเดอร์ output เป็น markdown ตรงๆ (ตาราง/หัวข้อ/dialogue render ออกมาจริง)
            lines.append("**Output:**")
            lines.append("")
            lines.append(output)
            lines.append("")

    return "\n".join(lines)


def _render_report_md(report: dict, comparison: str = "") -> str:
    """report เดี่ยว (1 AI) — comparison (ถ้ามี) แปะบนสุด แล้วตามด้วยเนื้อ report"""
    parts: list[str] = []
    if comparison:
        parts.append(comparison.rstrip())
        parts.append("")
        parts.append("---")
        parts.append("")
    parts.append(_render_report_body(report, heading_level=1))
    return "\n".join(parts)


def _render_combined_report_md(model_reports: list, comparison: str = "") -> str:
    """ไฟล์รวมต่อ 1 target — comparison บนสุด แล้วตามด้วย section ของแต่ละ AI
    model_reports: list ของ (label, report_dict) เรียงตาม parser ที่ใส่
    """
    parts: list[str] = []
    if comparison:
        parts.append(comparison.rstrip())
        parts.append("")
        parts.append("---")
        parts.append("")
    for _label, rep in model_reports:
        parts.append(_render_report_body(rep, heading_level=1).rstrip())
        parts.append("")
        parts.append("---")
        parts.append("")
    # ตัด separator ตัวท้ายสุดออก
    while parts and parts[-1] in ("", "---"):
        parts.pop()
    return "\n".join(parts) + "\n"


def _write_report(report: dict, report_path: Optional[str],
                  comparison: str = "") -> None:
    """เขียน report เป็น JSON + .md ลง report_path (ถ้าระบุ)
    comparison: ตารางเทียบคะแนนทุก AI — แปะไว้บนสุดของ .md (ว่าง = ไม่มี)
    """
    if not report_path:
        return
    try:
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        md_path = (report_path[:-5] if report_path.endswith(".json")
                   else report_path) + ".md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(_render_report_md(report, comparison=comparison))
        print(f"[report] saved {report_path} + {md_path}")
    except Exception as e:
        print(f"[report] write failed: {e}")


def _write_combined_report(target: str, model_reports: list, results: dict,
                           judge_model: str, mode: str, comparison: str,
                           report_path: Optional[str]) -> None:
    """เขียน report รวมหลาย AI ของ 1 target ลงไฟล์เดียว (JSON + .md)
    model_reports: list ของ (label, report_dict) เรียงตาม parser
    results[target]: {label: mean_total} ใช้ทำ summary ใน JSON
    """
    if not report_path:
        return
    try:
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        combined = {
            "target": target,
            "title": TARGET_TITLES.get(target, target),
            "mode": mode,
            "judge": {"model": judge_model},
            "comparison": results.get(target, {}),       # {label: mean_total}
            "models": {label: rep for label, rep in model_reports},
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        md_path = (report_path[:-5] if report_path.endswith(".json")
                   else report_path) + ".md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(_render_combined_report_md(model_reports, comparison=comparison))
        labels = ", ".join(label for label, _ in model_reports)
        print(f"[report] saved {report_path} + {md_path}  ({labels})")
    except Exception as e:
        print(f"[report] combined write failed: {e}")


def _criterion_avgs(per_list: list, criteria: list[str]) -> dict:
    """เฉลี่ยคะแนนรายเกณฑ์ (0–10) ข้ามทุก example (ข้าม example ที่ judge error / skip)"""
    valid = [p for p in per_list if isinstance(p, dict)]
    if not valid:
        return {c: 0.0 for c in criteria}
    return {
        c: round(sum(p.get(c, 0) for p in valid) / len(valid), 2)
        for c in criteria
    }


def _build_criteria_report(label, n, totals, per_list, criteria, model_name,
                           mode, extra=None) -> dict:
    """สร้าง + print report สำหรับ judge แบบ per-criterion (0–10)
    คืน dict ที่มี mean_total (0–10), criterion_avgs, judge.model
    totals ที่เป็น None (ตัวอย่างที่ generator พังระหว่างทาง เช่น โดน rate-limit)
    จะไม่ถูกนับรวมใน mean_total / criterion_avgs — กันไม่ให้คะแนนเพี้ยนจาก error
    """
    valid_totals = [t for t in totals if t is not None and t >= 0]
    skipped = n - len(valid_totals)
    mean_total = round(sum(valid_totals) / len(valid_totals), 2) if valid_totals else 0.0
    crit_avgs = _criterion_avgs(per_list, criteria)
    report = {
        "module": label,
        "mode": mode,
        "n": n,
        "n_scored": len(valid_totals),
        "n_skipped": skipped,
        "mean_total": mean_total,           # คะแนนรวมเฉลี่ย 0–10 (เฉพาะตัวอย่างที่สำเร็จ)
        "criterion_avgs": crit_avgs,        # ราย criterion 0–10
        "judge": {"model": model_name},
    }
    if extra:
        report.update(extra)
    print(f"\n📊 {label} — LLM-as-Judge ({_judge_label(JUDGE_MODEL_A)}, per-criterion 0–10)")
    print(f"   Samples      : {n}  (scored: {len(valid_totals)}, skipped/error: {skipped})")
    print(f"   Mean total   : {mean_total}/10")
    print(f"   Per-criterion averages (0–10):")
    for c in criteria:
        print(f"     {c:<5}: {crit_avgs[c]}")
    return report


def evaluate_testset(module, testset: list, blind: bool = False,
                     report_path: Optional[str] = None,
                     gen_lm=None, gen_label: Optional[str] = None) -> dict:
    """Evaluate VARK study guide testset — Gemini judge (A/B rubric, per-criterion 0–10)
    รายงาน mean_total (0–10) + criterion_avgs รายเกณฑ์
    blind=True → judge ไม่เห็น expected.rationale (ลด reference leak)
    gen_lm    → generator LM ที่อยากเทียบ (None = ใช้ global LM เดิม)
    gen_label → ชื่อ AI ที่จะโชว์ใน report (เช่น "Typhoon")

    ถ้า generator ล้มระหว่าง generate (เช่น โดน rate-limit 429 จาก provider) ตัวอย่างนั้น
    จะถูก "ข้าม" (ไม่นับคะแนน + บันทึกเหตุผลไว้ใน feedback) แทนที่จะปล่อยให้ exception
    หลุดขึ้นไป crash ทั้งสคริปต์ — โมเดล/ตัวอย่างอื่นที่เหลือยังรันต่อได้ตามปกติ
    """
    judge = _get_judge()
    mode = "blind" if blind else "ref-augmented"
    gen_label = gen_label or DEFAULT_GENERATOR
    print(f"\n🔎 VARK eval mode: {mode}  | AI: {gen_label}  ({_judge_label(JUDGE_MODEL_A)})")

    totals, per_list, feedbacks = [], [], []
    n = len(testset)
    for i, ex in enumerate(testset, 1):
        with _gen_context(gen_lm):
            pred, err_note = _call_with_retry(
                module, context=ex.context, vark_style=ex.vark_style,
                max_retries=3, base_backoff=10
            )
        if pred is None:
            print(f"[{i}/{n}] {gen_label} generation FAILED — skipping. {err_note[:120]}")
            per_list.append(None)
            totals.append(None)
            feedbacks.append({"i": i, "total": None, "scores": None,
                            "output": "", "feedback": err_note})
            continue

        per, total, fb = _vark_judge(ex, pred, judge, with_reference=not blind)
        per_list.append(per)
        totals.append(total if per is not None else None)
        # output = สิ่งที่ AI พิมพ์ออกมา (learning_material)
        out_text = pred.learning_material or ""
        feedbacks.append({"i": i, "total": total if per is not None else None,
                          "scores": per,
                          "output": out_text,
                          "feedback": fb})
        shown = f"{total}/10" if per is not None else "ERR"
        print(f"[{i}/{n}] {gen_label} total={shown}")
        print(f"{fb}")

    report = _build_criteria_report(
        f"VARK [{mode}]", n, totals, per_list, VARK_CRITERIA,
        JUDGE_MODEL_A.split("/")[-1], mode,
        extra={"feedbacks": feedbacks, "generator": gen_label},
    )
    _write_report(report, report_path)
    return report


def _safe_json(s: str):
    """พยายาม parse JSON — คืน None ถ้า fail (รองรับ ```json fence)"""
    if not s:
        return None
    txt = s.strip()
    # ลอก markdown fences ถ้ามี
    if txt.startswith("```"):
        txt = _re.sub(r"^```[a-zA-Z]*\s*", "", txt)
        txt = _re.sub(r"\s*```$", "", txt)
    try:
        return json.loads(txt)
    except Exception:
        return None


def evaluate_quiz_testset(module, testset: list, blind: bool = False,
                          report_path: Optional[str] = None,
                          gen_lm=None, gen_label: Optional[str] = None) -> dict:
    """Evaluate QuizModule — Gemini judge (C rubric, per-criterion 0–10)
    ให้ judge รีวิวชุดคำถามที่ generate ออกมาล้วนๆ — ไม่เทียบกับ expected
    blind=True → judge ไม่เห็น reference (expected.topics)
    gen_lm/gen_label → generator AI ที่อยากเทียบ (None = global LM เดิม)

    ถ้า generator ล้มระหว่าง generate (เช่น โดน rate-limit 429 จาก provider) ตัวอย่างนั้น
    จะถูกข้าม — ไม่ทำให้ทั้ง batch/สคริปต์ crash
    """
    judge = _get_judge()
    mode = "blind" if blind else "ref-augmented"
    gen_label = gen_label or DEFAULT_GENERATOR
    print(f"\n🔎 Quiz eval mode: {mode}  | AI: {gen_label}  ({_judge_label(JUDGE_MODEL_A)})")
    totals, per_list, feedbacks = [], [], []
    n = len(testset)
    for i, ex in enumerate(testset, 1):
        with _gen_context(gen_lm):
            pred, err_note = _call_with_retry(
                module,
                learning_material=ex.learning_material,
                vark_style=ex.vark_style,
                count=ex.count,
                max_retries=3, base_backoff=10
            )
        if pred is None:
            print(f"[Quiz eval] {gen_label} example {i}/{n} generation FAILED — skipping. {err_note[:120]}")
            per_list.append(None)
            totals.append(None)
            feedbacks.append({"i": i, "total": None, "scores": None,
                            "output": "", "feedback": err_note})
            continue
        per, total, fb = _quiz_judge(ex, pred, judge, with_reference=not blind)
        per_list.append(per)
        totals.append(total if per is not None else None)
        feedbacks.append({"i": i, "total": total if per is not None else None,
                          "scores": per,
                          "output": pred.questions or "",
                          "feedback": fb})

        # ── print ชุดคำถามที่ generate ออกมา (ให้เห็นผลลัพธ์ระหว่าง eval) ──
        shown = f"{total}/10" if per is not None else "ERR"
        print("─" * 70)
        print(f"[Quiz eval] {gen_label} example {i}/{n} — total={shown}")
        print("[Quiz eval] questions:\n")
        print(pred.questions or "(empty)")
        print("─" * 70)
        print(fb)
    report = _build_criteria_report(
        f"Quiz [{mode}]", n, totals, per_list, QUIZ_CRITERIA,
        JUDGE_MODEL_A.split("/")[-1], mode,
        extra={"feedbacks": feedbacks, "generator": gen_label},
    )
    _write_report(report, report_path)
    return report


def _build_video_lines(videos: list[dict]) -> str:
    """สร้างรายการวิดีโอแบบมีเลขลำดับ (1., 2., ...) พร้อม title/channel/views/transcript
    สำหรับป้อนให้ VideoRelevanceModule — mirror ของ filter_videos_by_relevance ใน main.py
    (sync version สำหรับ eval)

    เฉพาะคลิปที่ "มี transcript จริง" เท่านั้นที่จะถูกใส่เข้า list — คลิปที่ fetch
    transcript ไม่สำเร็จ (ทั้ง youtube-transcript-api และ Gemini fallback ล้มเหลว/โดน
    rate-limit) จะถูกกรองทิ้งตั้งแต่ตรงนี้ ไม่ส่งเข้า LLM เลย เพื่อไม่ให้ LLM ต้องเดา
    ความเกี่ยวข้องจากคลิปที่ไม่มีข้อมูลอะไรให้ตัดสินใจ (title อย่างเดียวไม่พอ)
    """
    import asyncio
    real = [v for v in videos if not v.get("is_search_link") and v.get("video_id")]
    if not real:
        return ""
    vids = [v["video_id"] for v in real]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        transcripts = list(pool.map(_fetch_transcript, vids))

    # ── กรองคลิปที่ไม่มี transcript ออกก่อนส่งเข้า prompt ──
    paired = [(v, t) for v, t in zip(real, transcripts) if t and t.strip()]
    dropped = len(real) - len(paired)
    if dropped:
        print(f"[video_lines] dropped {dropped}/{len(real)} clip(s) with no transcript "
              f"(transcript fetch failed / no captions / rate-limited)")
    if not paired:
        return ""

    vids_kept = [v["video_id"] for v, _ in paired]
    try:
        stats = asyncio.run(_fetch_video_stats(vids_kept))
    except Exception as e:
        print(f"[video_eval] stats fetch error: {e}")
        stats = {}

    return "\n".join(
        f"{i+1}. Title: {v['title']}\n"
        f"   Channel: {v.get('channel','')}\n"
        f"   Views: {_fmt_count(stats.get(v['video_id'],{}).get('views',0))}  "
        f"Likes: {_fmt_count(stats.get(v['video_id'],{}).get('likes',0))}\n"
        f"   Transcript: {t[:500]}"
        for i, (v, t) in enumerate(paired)
    )


def evaluate_video_testset(query_module, relevance_module, testset: list,
                           blind: bool = False,
                           report_path: Optional[str] = None,
                           gen_lm=None, gen_label: Optional[str] = None) -> dict:
    """Evaluate video pipeline (option B): pdf_content → queries → live YouTube search
    → relevance + VARK → Gemini text judge (D rubric, per-criterion 0–10).

    query_module    = VideoQueryModule (สร้าง search queries จาก pdf_content)
    relevance_module = VideoRelevanceModule (เลือกคลิปที่เกี่ยว + จัด VARK)
    blind ไม่มีผลกับ pipeline นี้ (ไม่มี reference) — เก็บไว้เพื่อ signature เดียวกับตัวอื่น
    gen_lm/gen_label → generator AI ที่อยากเทียบ (None = global LM เดิม)

    ถ้า generator ล้มระหว่าง generate query หรือ relevance (เช่น โดน rate-limit 429)
    ตัวอย่างนั้นจะถูกข้าม — ไม่ทำให้ทั้ง batch/สคริปต์ crash
    """
    import asyncio
    judge = _get_judge()
    gen_label = gen_label or DEFAULT_GENERATOR
    if not os.environ.get("YOUTUBE_API_KEY"):
        print("⚠️  ไม่มี YOUTUBE_API_KEY — search_youtube จะคืน search-link cards "
              "(ไม่มี transcript/วิดีโอจริง) judge จะเน้นประเมิน D.1–D.3 (query) เป็นหลัก")
    mode = "pipeline"
    print(f"\n🔎 Video eval mode: {mode}  | AI: {gen_label}  ({_judge_label(JUDGE_MODEL_A)})")
    totals, per_list, feedbacks = [], [], []
    n = len(testset)
    for i, ex in enumerate(testset, 1):
        pdf = ex.pdf_content

        with _gen_context(gen_lm):
            qpred, err_note = _call_with_retry(
                query_module, pdf_content=pdf,
                max_retries=3, base_backoff=10
            )
        if qpred is None:
            print(f"[Video eval] {gen_label} example {i}/{n} query-gen FAILED — skipping. {err_note[:120]}")
            per_list.append(None)
            totals.append(None)
            feedbacks.append({"i": i, "total": None, "scores": None,
                              "output": "", "feedback": err_note})
            continue

        queries_raw = qpred.youtube_queries or "[]"
        queries = _safe_json(queries_raw)
        if not isinstance(queries, list):
            queries = _re.findall(r'"([^"\n]{3,})"', queries_raw) or [pdf[:60]]
        queries = [str(q) for q in queries][:7]

        try:
            videos = asyncio.run(search_youtube(queries, max_per_query=3))
        except Exception as e:
            print(f"[video_eval] search error: {e}")
            videos = []
        video_lines = _build_video_lines(videos)

        if video_lines:
            with _gen_context(gen_lm):
                rpred, err_note = _call_with_retry(
                    relevance_module,
                    topic=pdf[:1000],
                    videos_with_transcripts=video_lines,
                    max_retries=3, base_backoff=10
                )
            if rpred is None:
                print(f"[Video eval] {gen_label} example {i}/{n} relevance-gen FAILED — skipping. "
                      f"{err_note[:120]}")
                per_list.append(None)
                totals.append(None)
                feedbacks.append({"i": i, "total": None, "scores": None,
                                  "output": f"queries: {queries_raw}\n"
                                            f"--- videos searched ---\n{video_lines or '(none)'}",
                                  "feedback": err_note})
                continue
            rel_idx = rpred.relevant_indices or "[]"
            vark_pv = rpred.vark_per_video or "{}"
        else:
            rel_idx, vark_pv = "[]", "{}"

        per, total, fb = _video_pipeline_judge(
            pdf, queries_raw, video_lines, rel_idx, vark_pv, judge,
        )
        per_list.append(per)
        totals.append(total if per is not None else None)
        output_blob = (
            f"queries: {queries_raw}\n"
            f"relevant_indices: {rel_idx}\n"
            f"vark_per_video: {vark_pv}\n"
            f"--- videos searched ---\n{video_lines or '(none)'}"
        )
        feedbacks.append({"i": i, "total": total if per is not None else None,
                          "scores": per,
                          "output": output_blob,
                          "feedback": fb})

        shown = f"{total}/10" if per is not None else "ERR"
        print("─" * 70)
        print(f"[Video eval] {gen_label} example {i}/{n} — total={shown}  "
              f"({len(queries)} queries, {'videos' if video_lines else 'no-videos'})")
        print("[Video eval] output:\n")
        print(output_blob)
        print("─" * 70)
        print(fb)
    extra = {"feedbacks": feedbacks, "generator": gen_label}
    report = _build_criteria_report(
        f"Video [{mode}]", n, totals, per_list, VIDEO_CRITERIA,
        JUDGE_MODEL_A.split("/")[-1], mode, extra=extra,
    )
    _write_report(report, report_path)
    return report


# ──────────────────────────────────────────────
# 4b. Multi-model comparison — รัน eval ของแต่ละ AI ใน slot แล้วเทียบคะแนน
# ──────────────────────────────────────────────
# ชื่อ section ที่จะโชว์ในรายงานเทียบ (target → หัวข้อ)
TARGET_TITLES = {"vark": "Content", "quiz": "Quiz", "video": "Video"}

# เกณฑ์ (criteria) ของแต่ละ target — ใช้สร้างตาราง rubric เทียบ Model × criteria
TARGET_CRITERIA = {"vark": VARK_CRITERIA, "quiz": QUIZ_CRITERIA, "video": VIDEO_CRITERIA}


def _alloc_eval_path(kind: str, used: set) -> str:
    """จองเลขรัน report ใหม่ (reports/{kind}_eval{n}.json) โดยกันเลขซ้ำใน batch เดียวกัน
    ใช้ตอน defer การเขียน (ไฟล์ยังไม่ถูกสร้าง) — เช็คทั้งไฟล์จริงและที่จองไว้แล้วใน `used`
    """
    os.makedirs("reports", exist_ok=True)
    n = 1
    while True:
        p = os.path.join("reports", f"{kind}_eval{n}.json")
        if not os.path.exists(p) and p not in used:
            used.add(p)
            return p
        n += 1


def _render_comparison_md(results: dict, judge_model: str, mode: str,
                          model_labels: list[str]) -> str:
    """สร้างบล็อกเทียบคะแนนของแต่ละ AI (ไว้แปะบนสุดของ report) ตามรูปแบบ:
        ## Content
        Typhoon --> 9.7
        Qwen2.5 --> 9.3
    results[target][label] = mean_total (0–10) หรือ None ถ้า error
    """
    lines: list[str] = []
    lines.append("# 📊 Model comparison")
    lines.append("")
    lines.append(f"- Judge: `{judge_model}`")
    lines.append(f"- Mode: `{mode}`")
    lines.append(f"- AI ที่เทียบ (slot): {', '.join(model_labels) or '(none)'}")
    lines.append("")
    for target in ("vark", "quiz", "video"):
        scores = results.get(target)
        if not scores:
            continue
        lines.append(f"## {TARGET_TITLES[target]}")
        lines.append("")
        # เรียงจากคะแนนสูง → ต่ำ (ตัว error/None ไปท้ายสุด)
        ordered = sorted(
            scores.items(),
            key=lambda kv: (kv[1] is None, -(kv[1] if kv[1] is not None else 0)),
        )
        for label, val in ordered:
            shown = f"{val}" if val is not None else "ERR"
            lines.append(f"{label} --> {shown}")
        lines.append("")
    return "\n".join(lines)


def _alloc_rubric_path(used: Optional[set] = None) -> str:
    """จองเลขรันใหม่สำหรับไฟล์ rubric รวม (reports/reportrubrics_{n}.md)
    กันเลขซ้ำทั้งกับไฟล์จริงและที่จองไว้ใน `used`
    """
    os.makedirs("reports", exist_ok=True)
    used = used or set()
    n = 1
    while True:
        p = os.path.join("reports", f"reportrubrics_{n}.md")
        if not os.path.exists(p) and p not in used:
            used.add(p)
            return p
        n += 1


def _render_rubric_md(target_reports: dict, results: dict, judge_model: str,
                      mode: str, model_labels: list[str]) -> str:
    """สร้างไฟล์ rubric รวม — ตารางเทียบ Model × criteria ของทุก target ในไฟล์เดียว

    target_reports[target] = list ของ (label, report_dict) เรียงตาม parser
    แต่ละ report_dict มี criterion_avgs (ราย criterion 0–10) + mean_total
    ในแต่ละแถว (criterion) จะ **ตัวหนา** คะแนนของ AI ที่ทำได้สูงสุด เพื่อให้เห็นว่าตัวไหนเด่น
    """
    lines: list[str] = []
    lines.append("# 📊 Rubric comparison — Model × criteria")
    lines.append("")
    lines.append(f"- Judge: `{judge_model}`")
    lines.append(f"- Mode: `{mode}`")
    lines.append(f"- AI ที่เทียบ (slot): {', '.join(model_labels) or '(none)'}")
    lines.append("")
    lines.append("คะแนนแต่ละช่อง = ค่าเฉลี่ยรายเกณฑ์ (0–10) ข้ามทุกตัวอย่างที่ generate สำเร็จ — "
                 "ตัวหนา = AI ที่ทำคะแนนสูงสุดในเกณฑ์นั้น")
    lines.append("")

    for target in ("vark", "quiz", "video"):
        mreports = target_reports.get(target) or []
        if not mreports:
            continue
        labels = [label for label, _ in mreports]
        criteria = TARGET_CRITERIA.get(target, [])

        lines.append(f"## {TARGET_TITLES.get(target, target)}")
        lines.append("")
        lines.append("| Criterion | " + " | ".join(labels) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in labels) + " |")

        for c in criteria:
            row_vals = []
            for _label, rep in mreports:
                v = (rep.get("criterion_avgs") or {}).get(c)
                row_vals.append(v if isinstance(v, (int, float)) else None)
            best = max((v for v in row_vals if v is not None), default=None)
            cells = []
            for v in row_vals:
                if v is None:
                    cells.append("—")
                elif best is not None and v == best:
                    cells.append(f"**{v}**")
                else:
                    cells.append(f"{v}")
            lines.append(f"| {c} | " + " | ".join(cells) + " |")

        # แถวสรุป mean total ต่อ AI
        totals = []
        for _label, rep in mreports:
            t = rep.get("mean_total")
            totals.append(t if isinstance(t, (int, float)) else None)
        best_t = max((t for t in totals if t is not None), default=None)
        tcells = []
        for t in totals:
            if t is None:
                tcells.append("—")
            elif best_t is not None and t == best_t:
                tcells.append(f"**{t}**")
            else:
                tcells.append(f"{t}")
        lines.append("| **Mean total** | " + " | ".join(tcells) + " |")

        # แถวสรุปจำนวนที่ถูกข้าม (rate-limit / error) ต่อ AI
        skip_cells = []
        any_skip = False
        for _label, rep in mreports:
            sk = rep.get("n_skipped", 0)
            if sk:
                any_skip = True
            skip_cells.append(str(sk))
        if any_skip:
            lines.append("| Skipped (rate-limit/error) | " + " | ".join(skip_cells) + " |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _write_rubric_report(target_reports: dict, results: dict, judge_model: str,
                         mode: str, model_labels: list[str],
                         path: Optional[str] = None) -> Optional[str]:
    """เขียนไฟล์ rubric รวม (reports/reportrubrics_{n}.md) — คืน path ที่เขียน หรือ None"""
    if not any(target_reports.get(t) for t in target_reports):
        return None
    path = path or _alloc_rubric_path()
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        md = _render_rubric_md(target_reports, results, judge_model, mode, model_labels)
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"[report] saved rubric table {path}")
        return path
    except Exception as e:
        print(f"[report] rubric write failed: {e}")
        return None


def evaluate_models_comparison(targets: list[str],
                               model_labels: Optional[list[str]] = None,
                               blind: bool = False) -> dict:
    """รัน eval ของแต่ละ generator AI ใน slot — รวมทุก AI ไว้ใน report เดียวต่อ target
    เช่น reports/vark_eval{n}.json + .md = รวม Typhoon + Qwen ในไฟล์เดียว (เรียงตาม parser)
    พร้อมตารางเทียบคะแนน "บนสุด" ของไฟล์

    ถ้า AI ตัวใดตัวหนึ่งพังระหว่าง generate (เช่น โดน rate-limit 429 จาก free-tier
    provider) ตัวอย่างนั้นจะถูกข้ามเฉพาะจุด (ดู evaluate_testset/evaluate_quiz_testset/
    evaluate_video_testset) — ไม่ทำให้ AI ตัวอื่นหรือ target อื่นในลูปนี้หยุดทำงานไปด้วย
    targets       : subset ของ ["vark", "quiz", "video"]
    model_labels  : list ของ label จาก GENERATOR_MODELS (None = ทั้ง slot)
    คืน results[target][label] = mean_total
    """
    if model_labels is None:
        model_labels = list(GENERATOR_MODELS)
    _get_judge()  # ensure Gemini judge พร้อม (raise ถ้าไม่มี GEMINI_API_KEY)
    mode = "blind" if blind else "ref-augmented"

    results: dict[str, dict] = {}
    used_labels: list[str] = []
    # เก็บ report ราย target → list ของ (label, report_dict) เรียงตาม parser
    target_reports: dict[str, list] = {t: [] for t in targets}
    for label in model_labels:
        try:
            gen_lm = build_generator_lm(label)
        except Exception as e:
            print(f"⚠️  ข้าม AI '{label}': {e}")
            continue
        used_labels.append(label)
        # ตั้งเป็น global default ด้วย เผื่อ call ที่ไม่ได้ wrap
        dspy.settings.configure(lm=gen_lm)
        print(f"\n{'═'*66}\n▶ Generator AI: {label}  ({gen_lm.model})\n{'═'*66}")

        # report_path=None → ยังไม่เขียนไฟล์ตอนนี้ (defer ไปรวมเป็นไฟล์เดียวต่อ target)
        if "vark" in targets:
            rep = evaluate_testset(load_module(), get_testset(), blind=blind,
                                   report_path=None, gen_lm=gen_lm, gen_label=label)
            results.setdefault("vark", {})[label] = rep.get("mean_total")
            target_reports["vark"].append((label, rep))
        if "quiz" in targets:
            rep = evaluate_quiz_testset(load_quiz_module(), get_quiz_testset(),
                                        blind=blind, report_path=None,
                                        gen_lm=gen_lm, gen_label=label)
            results.setdefault("quiz", {})[label] = rep.get("mean_total")
            target_reports["quiz"].append((label, rep))
        if "video" in targets:
            rep = evaluate_video_testset(VideoQueryModule(), load_relevance_module(),
                                         get_video_testset(), report_path=None,
                                         gen_lm=gen_lm, gen_label=label)
            results.setdefault("video", {})[label] = rep.get("mean_total")
            target_reports["video"].append((label, rep))

    # ── รู้คะแนนครบแล้ว → เขียน 1 ไฟล์ต่อ target (รวมทุก AI + comparison บนสุด) ──
    comparison = _render_comparison_md(results, JUDGE_MODEL_A, mode, used_labels)
    used_paths: set = set()
    for target in targets:
        mreports = target_reports.get(target) or []
        if not mreports:
            continue
        path = _alloc_eval_path(target, used_paths)
        _write_combined_report(target, mreports, results, JUDGE_MODEL_A,
                               mode, comparison, path)

    # ── ไฟล์ rubric รวม — ตารางเทียบ Model × criteria ของทุก target ในไฟล์เดียว ──
    _write_rubric_report(target_reports, results, JUDGE_MODEL_A, mode, used_labels)

    print("\n" + comparison)
    return results


# ──────────────────────────────────────────────
# 5. Training Dataset (ขยายเพิ่ม)
# ──────────────────────────────────────────────


def get_trainset(path: str = "dataset/train/vark_train.json") -> list[dspy.Example]:
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    examples = []
    for r in rows:
        kw = {
            "context": r["context"],
            "vark_style": json.dumps(r["vark_style"], ensure_ascii=False),
        }
        if r.get("expected"):
            kw["expected"] = r["expected"]
        examples.append(
            dspy.Example(**kw).with_inputs("context", "vark_style")
        )
    return examples


def get_testset() -> list[dspy.Example]:
    # eval ใช้ train อย่างเดียว (val/test ถูก merge เข้า train แล้ว)
    return get_trainset()


def get_quiz_trainset(path: str = "dataset/train/quiz_train.json") -> list[dspy.Example]:
    """
    Quiz trainset format (JSON array):
      [
        {
          "id": "quiz_001",
          "learning_material": "<markdown ของ study guide>",
          "vark_style": {"V":50,"A":10,"R":10,"K":30,"dominant":"V"},
          "count": 5,
          "notes": "..."
        }, ...
      ]
    มีแค่ input — output (questions) จะถูก bootstrap โดย DSPy เอง
    """
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    examples = []
    for r in rows:
        kw = {
            "learning_material": r["learning_material"],
            "vark_style": json.dumps(r["vark_style"], ensure_ascii=False),
            "count": int(r.get("count", 5)),
        }
        if r.get("expected"):
            kw["expected"] = r["expected"]
        examples.append(
            dspy.Example(**kw).with_inputs(
                "learning_material", "vark_style", "count"
            )
        )
    return examples


def get_quiz_testset() -> list[dspy.Example]:
    # eval ใช้ train อย่างเดียว (val/test ถูก merge เข้า train แล้ว)
    return get_quiz_trainset()


def get_video_testset() -> list[dspy.Example]:
    # eval ใช้ train อย่างเดียว
    return get_video_trainset()


def _vark_to_desc(vark: dict) -> str:
    """แปลง VARK JSON {V:60,A:10,R:10,K:20} → 'V 60%, A 10%, R 10%, K 20%'"""
    parts = [f"{s} {int(vark.get(s, 0))}%" for s in "VARK" if int(vark.get(s, 0)) > 0]
    return ", ".join(parts)


def get_video_trainset(path: str = "dataset/train/video_train.json") -> list[dspy.Example]:
    """
    Video pipeline trainset — schema ใหม่: pdf_content อย่างเดียว
      [
        { "id": "vid_001", "pdf_content": "<เนื้อหาที่สกัดจาก PDF>" },
        ...
      ]
    pipeline (option B) จะ: pdf_content → สร้าง YouTube queries → ค้นหาจริง
    → เลือกคลิปที่เกี่ยว + จัด VARK → ประเมินด้วย Gemini judge (D rubric)
    ไม่มี gold label — judge ทำงานแบบ reference-free (`expected` ใส่ได้แต่ optional)
    """
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)

    examples = []
    for r in rows:
        pdf = r.get("pdf_content", "")
        if not pdf or not str(pdf).strip():
            raise ValueError(
                f"Entry {r.get('id','?')}: ต้องมี 'pdf_content' (เนื้อหาจาก PDF)"
            )
        kw = {"pdf_content": str(pdf)}
        if r.get("expected"):
            kw["expected"] = r["expected"]
        examples.append(
            dspy.Example(**kw).with_inputs("pdf_content")
        )
    return examples


# ──────────────────────────────────────────────
# 7. Load for runtime
# ──────────────────────────────────────────────
def load_module(model_path: str = "vark_model.json") -> VARKModule:
    """
    Load compiled demos จาก vark_model.json เข้า VARKModule (single stage)

    Strategy:
      1) ใช้ DSPy native module.load() ก่อน — handle JSON layout ทุก version
         และ inject demos เข้า generate ให้อัตโนมัติ
      2) Fallback: parse JSON เองแล้ว inject เข้า generate
    """
    module = VARKModule()

    if not os.path.exists(model_path):
        # zero-shot คือโหมดปกติแล้ว (prompt อยู่ใน signature ครบ ไม่ต้อง compile)
        print("ℹ️  ใช้ zero-shot VARKModule (prompt จาก signature — ไม่ต้อง compile)")
        return module

    # ── Path 1: DSPy native loader ─────────────────────────────
    try:
        module.load(model_path)
        g = len(getattr(module.generate.predict, "demos", []) or [])
        if g:
            print(f"✅ Loaded VARKModule (generate: {g} demos)")
            return module
        print("[load_module] native load: no demos populated, falling back to manual parse")
    except Exception as e:
        print(f"[load_module] native load failed ({e}), falling back to manual parse")

    # ── Path 2: manual JSON parse ──────────────────────────────
    try:
        with open(model_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        print(f"[load_module] JSON keys: {list(state.keys())[:10]}")

        def _extract(stage: str) -> list:
            for key in (stage, f"{stage}.predict"):
                v = state.get(key)
                if isinstance(v, dict) and v.get("demos"):
                    return v["demos"]
            flat = state.get(f"{stage}.predict.demos") or state.get(f"{stage}.demos")
            return flat or []

        generate_raw = _extract("generate")
        loaded_g = _inject_demos(module.generate, generate_raw) if generate_raw else False

        if loaded_g:
            print(f"✅ Manual injection — generate: {len(generate_raw)} demos")
        else:
            print("⚠️  No demos found — using zero-shot VARKModule")
            print(f"    Full JSON structure: {json.dumps(state, ensure_ascii=False)[:400]}")

    except Exception as e:
        print(f"⚠️  DSPy startup warning (non-fatal): {e}")
        print("     Continuing with zero-shot VARKModule")

    return module


# ──────────────────────────────────────────────
# 7b. Load helpers for Quiz / Video Classifier
# ──────────────────────────────────────────────
def _inject_demos(predict_obj, demos_raw: list) -> bool:
    """ลอง inject demos เข้า .predict.demos ก่อน, ถ้าไม่ได้ลอง .demos ตรงๆ"""
    demos = [dspy.Example(**d) if not isinstance(d, dspy.Example) else d
             for d in demos_raw]
    try:
        predict_obj.predict.demos = demos
        return True
    except AttributeError:
        try:
            predict_obj.demos = demos
            return True
        except AttributeError:
            return False


def _load_demos_from_json(model_path: str) -> list:
    """อ่าน demos จากไฟล์ DSPy JSON หลายเวอร์ชัน — คืน [] ถ้าไม่เจอ"""
    if not os.path.exists(model_path):
        return []
    try:
        with open(model_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception as e:
        print(f"[load_demos] read error {model_path}: {e}")
        return []

    for key, val in state.items():
        if isinstance(val, dict) and val.get("demos"):
            return val["demos"]
    for key in ("generate.predict.demos", "classify.demos", "classify.predict.demos"):
        if state.get(key):
            return state[key]
    return []


def load_quiz_module(model_path: str = "quiz_model.json") -> "QuizModule":
    """Load compiled QuizModule — zero-shot fallback ถ้าไม่มีไฟล์"""
    module = QuizModule()
    demos = _load_demos_from_json(model_path)
    if demos and _inject_demos(module.generate, demos):
        print(f"✅ QuizModule: loaded {len(demos)} demos from {model_path}")
    else:
        print(f"ℹ️  QuizModule: zero-shot mode (no demos at {model_path})")
    return module


def load_classifier_module(
    model_path: str = "video_classifier_model.json",
) -> "VideoClassifierModule":
    """Load compiled VideoClassifierModule — zero-shot fallback ถ้าไม่มีไฟล์"""
    module = VideoClassifierModule()
    demos = _load_demos_from_json(model_path)
    if demos and _inject_demos(module.classify, demos):
        print(f"✅ VideoClassifierModule: loaded {len(demos)} demos from {model_path}")
    else:
        print(f"ℹ️  VideoClassifierModule: zero-shot mode (no demos at {model_path})")
    return module


def load_relevance_module(
    model_path: str = "video_relevance_model.json",
) -> "VideoRelevanceModule":
    """Load VideoRelevanceModule — zero-shot fallback ถ้าไม่มีไฟล์"""
    module = VideoRelevanceModule()
    demos = _load_demos_from_json(model_path)
    if demos and _inject_demos(module.judge, demos):
        print(f"✅ VideoRelevanceModule: loaded {len(demos)} demos from {model_path}")
    else:
        print(f"ℹ️  VideoRelevanceModule: zero-shot mode (no demos at {model_path})")
    return module


# ──────────────────────────────────────────────
# 8. Entrypoint
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    # Whisper transcripts can contain arbitrary Unicode (e.g. hallucinated CJK/Tamil on
    # non-speech audio). The default Windows console is cp874 (Thai) and would crash on a
    # print() of such chars — harden stdout/stderr so a stray char can't kill a long eval run.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="DSPy pipeline: enrich → evaluate (vark / quiz / video). "
                    "ทุก module เป็น zero-shot + Gemini judge eval (ไม่มี compile step)"
    )
    parser.add_argument(
        "--target", type=str, default="vark",
        choices=["vark", "quiz", "video", "all"],
        help="which module to operate on"
    )
    parser.add_argument("--api-key", type=str, default=None,
                        help="Typhoon API Key (else read TYPHOON_API_KEY from env)")
    parser.add_argument("--models", type=str, default=None,
                        help="comma-list ของ AI ใน slot GENERATOR_MODELS ที่จะเทียบกัน "
                             "เช่น --models Typhoon,Qwen2.5 (ละไว้ = ทุกตัวใน slot). "
                             "เพิ่ม/ลด model ได้ในตัวแปร GENERATOR_MODELS")
    parser.add_argument("--output", type=str, default=None,
                        help="output model path (single target only)")
    parser.add_argument("--enrich", action="store_true",
                        help="enrich video dataset(s) from YouTube API before compile "
                             "(applies to video/all only; needs YOUTUBE_API_KEY)")
    parser.add_argument("--eval", action="store_true",
                        help="evaluate test set in reference-augmented mode "
                             "(judges see expected.rationale)")
    parser.add_argument("--eval-blind", action="store_true",
                        help="evaluate test set in blind mode "
                             "(judges do NOT see expected) — combine with --eval to run both")
    args = parser.parse_args()

    do_vark  = args.target in ("vark", "all")
    do_quiz  = args.target in ("quiz", "all")
    do_video = args.target in ("video", "all")

    # ─ model paths — --output only applies เมื่อ target เป็น single ─
    vark_path  = (args.output if args.target == "vark"  else None) or "vark_model.json"
    quiz_path  = (args.output if args.target == "quiz"  else None) or "quiz_model.json"
    video_path = (args.output if args.target == "video" else None) or "video_classifier_model.json"
    video_relevance_path = "video_relevance_model.json"

    # ── 1. Enrich (video only) ─────────────────────────────────
    if args.enrich:
        if not do_video:
            print("⚠️  --enrich ใช้ได้กับ target video/all เท่านั้น — ข้าม")
        elif not os.environ.get("YOUTUBE_API_KEY"):
            print("❌ YOUTUBE_API_KEY required for --enrich")
            sys.exit(1)


    # ── 2. ทุก module เป็น zero-shot (ไม่มี compile step แล้ว) ──
    # การทำงานหลักคือ eval ด้วย Gemini judge บน zero-shot module / โมเดลที่โหลด
    vark_module = quiz_module_obj = video_module_obj = None
    if not (args.eval or args.eval_blind):
        print("ℹ️  ไม่มี compile step — ใช้ --eval / --eval-blind เพื่อประเมินด้วย Gemini judge")

    # ── 3. Evaluate testset ─────────────────────────────────────
    eval_modes: list[bool] = []  # blind flag per pass
    if args.eval:
        eval_modes.append(False)         # ref-augmented
    if args.eval_blind:
        eval_modes.append(True)          # blind

    if eval_modes:
        # เลือก AI จาก slot — ละ --models ไว้ = เทียบทุกตัวใน GENERATOR_MODELS
        if args.models:
            model_labels = [m.strip() for m in args.models.split(",") if m.strip()]
        else:
            model_labels = list(GENERATOR_MODELS)
        # --api-key override → เซ็ตเข้า env ของ Typhoon ให้ build_generator_lm หยิบไปใช้
        if args.api_key:
            os.environ["TYPHOON_API_KEY"] = args.api_key

        targets = [t for t, on in
                   (("vark", do_vark), ("quiz", do_quiz), ("video", do_video)) if on]
        # video pipeline ไม่มี reference → blind ไม่มีผล แต่ orchestrator จัดการเอง
        for blind in eval_modes:
            evaluate_models_comparison(targets, model_labels, blind=blind)