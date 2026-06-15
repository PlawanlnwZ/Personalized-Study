"""
streamlit_app.py — Personalized Study (Streamlit UI)

เวอร์ชัน Streamlit ของแอป VARK Study — เรียก DSPy modules ตรง ๆ ใน process เดียว
(ไม่ต้องรัน FastAPI แยก) ครอบคลุม: สร้างสื่อการเรียนรู้ + วิดีโอ YouTube + Quiz + อ่านออกเสียง

รัน:  streamlit run streamlit_app.py
ต้องมี env: OPENROUTER_API_KEY / TYPHOON_API_KEY (generator), YOUTUBE_API_KEY (วิดีโอจริง, optional)
"""

import asyncio
import io
import json
import re

import dspy
import streamlit as st

from dspy_module import (
    GENERATOR_MODELS,
    DEFAULT_GENERATOR,
    build_generator_lm,
    load_module,
    load_quiz_module,
    load_relevance_module,
    VideoQueryModule,
    search_youtube,
    _fetch_transcript,
    _fetch_video_stats,
    _fmt_count,
    _safe_json,
)

st.set_page_config(page_title="Personalized Study", page_icon="📚", layout="wide")


# ──────────────────────────────────────────────
# Helpers — PDF, JSON, math, videos
# ──────────────────────────────────────────────
def extract_pdf_text(pdf_bytes: bytes, max_chars: int = 15000) -> str:
    """สกัดข้อความจาก PDF — ลอง pymupdf ก่อน, fallback ไป pdfminer"""
    try:
        import fitz  # pymupdf
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join(p.get_text("text") for p in doc).strip()
        doc.close()
        if text:
            return text[:max_chars]
    except Exception:
        pass

    import logging
    for noisy in ("pdfminer.pdfdocument", "pdfminer.pdfpage", "pdfminer.pdfinterp",
                  "pdfminer.pdfdevice", "pdfminer.pdffont", "pdfminer.cmapdb",
                  "pdfminer.converter", "pdfminer"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    from pdfminer.high_level import extract_text_to_fp
    from pdfminer.layout import LAParams
    out = io.StringIO()
    extract_text_to_fp(io.BytesIO(pdf_bytes), out, laparams=LAParams(),
                       output_type="text", codec="utf-8")
    return out.getvalue().strip()[:max_chars]


def parse_json_loose(raw: str) -> list:
    """parse JSON array แบบ robust — strip fences/trailing commas, รองรับ CoT นำหน้า"""
    if not raw:
        return []
    text = re.sub(r"```[a-zA-Z]*\s*", "", raw)
    text = re.sub(r"\s*```", "", text).strip()
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        candidate = re.sub(r",\s*([}\]])", r"\1", text[start:end + 1])
        try:
            res = json.loads(candidate)
            if isinstance(res, list):
                return res
        except Exception:
            pass
    parsed = _safe_json(text)
    if isinstance(parsed, list):
        return parsed
    return re.findall(r'"([^"\n]{3,})"', text)


def to_streamlit_math(md: str) -> str:
    r"""แปลง delimiter ของโมเดล (\( \) , \[ \]) → รูปแบบ KaTeX ที่ st.markdown รองรับ ($ , $$)"""
    md = re.sub(r"\\\[(.+?)\\\]", lambda m: f"\n$$ {m.group(1).strip()} $$\n", md, flags=re.DOTALL)
    md = re.sub(r"\\\((.+?)\\\)", lambda m: f"${m.group(1).strip()}$", md, flags=re.DOTALL)
    return md


def render_material(md: str) -> None:
    """เรนเดอร์สื่อการเรียนรู้ — แปลงสูตรคณิต + ซ่อนบล็อกเฉลย ('ans' ... 'ans') ไว้ใน expander"""
    md = to_streamlit_math(md)
    # แยกตามบรรทัดที่เป็น 'ans' ล้วน — index คี่ = เนื้อหาเฉลย
    parts = re.split(r"(?m)^[ \t]*'ans'[ \t]*$", md)
    for i, seg in enumerate(parts):
        seg = seg.strip()
        if not seg:
            continue
        if i % 2 == 1:
            with st.expander("📝 เฉลย / Answer key"):
                st.markdown(seg)
        else:
            st.markdown(seg)


def detect_lang(text: str) -> str:
    """เดาภาษาจากตัวอักษรไทย — คืน 'th' หรือ 'en' สำหรับ gTTS"""
    return "th" if re.search(r"[฀-๿]", text or "") else "en"


def tts_bytes(text: str, lang: str) -> bytes:
    from gtts import gTTS
    buf = io.BytesIO()
    gTTS(text=text[:3000], lang=lang, slow=False).write_to_fp(buf)
    return buf.getvalue()


async def _filter_relevance(videos: list, topic: str, relevance_module) -> list:
    """กรองวิดีโอตามความเกี่ยวข้อง + จัด VARK (mirror ของ main.py filter_videos_by_relevance)"""
    real = [v for v in videos if not v.get("is_search_link") and v.get("video_id")]
    links = [v for v in videos if v.get("is_search_link")]
    if not real or relevance_module is None:
        return videos

    transcripts, stats = await asyncio.gather(
        asyncio.gather(*[asyncio.to_thread(_fetch_transcript, v["video_id"]) for v in real]),
        _fetch_video_stats([v["video_id"] for v in real]),
    )
    lines = "\n".join(
        f"{i+1}. Title: {v['title']}\n"
        f"   Channel: {v.get('channel','')}\n"
        f"   Views: {_fmt_count(stats.get(v['video_id'],{}).get('views',0))}  "
        f"Likes: {_fmt_count(stats.get(v['video_id'],{}).get('likes',0))}\n"
        f"   Transcript: {(t or '(no transcript)')[:500]}"
        for i, (v, t) in enumerate(zip(real, transcripts))
    )

    def _run():
        try:
            pred = relevance_module(topic=topic[:1000], videos_with_transcripts=lines)
            return pred.relevant_indices or "[]", pred.vark_per_video or "{}"
        except Exception as e:
            print(f"[relevance] error: {e}")
            return "[]", "{}"

    raw_idx, raw_vark = await asyncio.to_thread(_run)
    try:
        m = re.search(r"\[.*?\]", raw_idx, re.DOTALL)
        indices = set((json.loads(m.group(0)) if m else list(range(1, len(real) + 1)))[:10])
    except Exception:
        indices = set(range(1, min(len(real) + 1, 11)))
    try:
        m2 = re.search(r"\{.*?\}", raw_vark, re.DOTALL)
        vark_map = json.loads(m2.group(0)) if m2 else {}
    except Exception:
        vark_map = {}

    passed = []
    for i, v in enumerate(real, 1):
        if i in indices:
            vc = dict(v)
            vc["vark"] = vark_map.get(str(i), "")
            passed.append(vc)
    return (passed + links) if passed else real[:10] + links


# ──────────────────────────────────────────────
# Model loading (cached per generator label)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_bundle(label: str) -> dict:
    """สร้าง LM + โหลดทุก module สำหรับ generator ที่เลือก (cache ต่อ label)"""
    lm = build_generator_lm(label)
    return {
        "lm": lm,
        "vark": load_module(),
        "quiz": load_quiz_module(),
        "query": VideoQueryModule(),
        "relevance": load_relevance_module(),
    }


# ──────────────────────────────────────────────
# Sidebar — model + VARK profile
# ──────────────────────────────────────────────
st.sidebar.title("⚙️ ตั้งค่า")

labels = list(GENERATOR_MODELS)
default_idx = labels.index(DEFAULT_GENERATOR) if DEFAULT_GENERATOR in labels else 0
model_label = st.sidebar.selectbox("🤖 Generator model", labels, index=default_idx)

st.sidebar.markdown("### 🎯 VARK profile")
v = st.sidebar.slider("Visual (V)", 0, 100, 40)
a = st.sidebar.slider("Auditory (A)", 0, 100, 20)
r = st.sidebar.slider("Read/Write (R)", 0, 100, 30)
k = st.sidebar.slider("Kinesthetic (K)", 0, 100, 10)
dominant = max({"V": v, "A": a, "R": r, "K": k}.items(), key=lambda kv: kv[1])[0]
vark_profile = {"V": v, "A": a, "R": r, "K": k, "dominant": dominant}
vark_style = json.dumps(vark_profile, ensure_ascii=False)
st.sidebar.caption(f"Dominant: **{dominant}**")


# ──────────────────────────────────────────────
# Main — input
# ──────────────────────────────────────────────
st.title("📚 Personalized Study")
st.caption("อัปโหลด PDF → สร้างสื่อการเรียนรู้ปรับตามสไตล์ VARK + วิดีโอ + แบบทดสอบ")

col_in1, col_in2 = st.columns([2, 1])
with col_in1:
    pdf_file = st.file_uploader("📄 อัปโหลด PDF", type=["pdf"])
with col_in2:
    topic = st.text_input("คำสั่งเพิ่มเติม (optional)", placeholder="เช่น เน้นบทที่ 3")

generate = st.button("✨ สร้างสื่อการเรียนรู้", type="primary", disabled=pdf_file is None)

if generate and pdf_file is not None:
    try:
        bundle = get_bundle(model_label)
    except Exception as e:
        st.error(f"โหลดโมเดล '{model_label}' ไม่ได้: {e}")
        st.stop()

    with st.spinner("กำลังอ่าน PDF…"):
        context = extract_pdf_text(pdf_file.read())
    if topic.strip():
        context = f"{context}\n\n[คำสั่งเพิ่มเติม: {topic.strip()}]"

    if not context.strip():
        st.error("สกัดข้อความจาก PDF ไม่ได้")
        st.stop()

    lm = bundle["lm"]
    with st.spinner(f"กำลังสร้างสื่อการเรียนรู้ด้วย {model_label}…"):
        with dspy.context(lm=lm):
            pred = bundle["vark"](context=context, vark_style=vark_style)
        material = pred.learning_material or ""

    with st.spinner("กำลังค้นหาวิดีโอที่เกี่ยวข้อง…"):
        try:
            with dspy.context(lm=lm):
                qpred = bundle["query"](pdf_content=context)
            queries = [q for q in parse_json_loose(qpred.youtube_queries or "[]")
                       if isinstance(q, str) and q.strip()]
            if not queries:
                queries = [topic.strip() or context[:80].strip() or "การเรียนรู้"]
            videos = asyncio.run(search_youtube(queries, max_per_query=3))
            with dspy.context(lm=lm):
                videos = asyncio.run(
                    _filter_relevance(videos, topic.strip() or context[:150], bundle["relevance"])
                )
        except Exception as e:
            print(f"[videos] error: {e}")
            queries, videos = [], []

    # เก็บผลลัพธ์ใน session_state (กัน rerun ล้าง) + ล้าง quiz เก่า
    st.session_state.update({
        "material": material,
        "context": context,
        "vark_style": vark_style,
        "vark_profile": vark_profile,
        "model_label": model_label,
        "videos": videos,
        "queries": queries,
    })
    st.session_state.pop("quiz", None)
    st.session_state.pop("quiz_submitted", None)


# ──────────────────────────────────────────────
# Output tabs
# ──────────────────────────────────────────────
if "material" in st.session_state:
    tab_mat, tab_vid, tab_quiz = st.tabs(["📖 เนื้อหา", "🎬 วิดีโอ", "📝 แบบทดสอบ"])

    # ── เนื้อหา + TTS ──
    with tab_mat:
        material = st.session_state["material"]
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("🔊 อ่านออกเสียง"):
                with st.spinner("กำลังสังเคราะห์เสียง…"):
                    try:
                        audio = tts_bytes(material, detect_lang(material))
                        st.session_state["tts_audio"] = audio
                    except Exception as e:
                        st.error(f"TTS error: {e}")
        if st.session_state.get("tts_audio"):
            st.audio(st.session_state["tts_audio"], format="audio/mpeg")
            st.caption("อ่านออกเสียงเฉพาะ ~3000 ตัวอักษรแรก")
        st.divider()
        render_material(material)
        st.download_button("⬇️ ดาวน์โหลด (.md)", material,
                           file_name="learning_material.md", mime="text/markdown")

    # ── วิดีโอ ──
    with tab_vid:
        videos = st.session_state.get("videos") or []
        if st.session_state.get("queries"):
            st.caption("คำค้น: " + " · ".join(st.session_state["queries"]))
        if not videos:
            st.info("ไม่พบวิดีโอ — ตั้งค่า YOUTUBE_API_KEY เพื่อค้นหาวิดีโอจริง")
        else:
            cols = st.columns(2)
            for i, vid in enumerate(videos):
                with cols[i % 2]:
                    badge = f" `{vid['vark']}`" if vid.get("vark") else ""
                    st.markdown(f"**{vid.get('title','(no title)')}**{badge}")
                    st.caption(vid.get("channel", ""))
                    if vid.get("is_search_link"):
                        st.link_button("🔍 เปิดผลค้นหาใน YouTube", vid.get("url", "#"))
                    elif vid.get("url"):
                        st.video(vid["url"])
                    st.write("")

    # ── Quiz ──
    with tab_quiz:
        n_q = st.slider("จำนวนข้อ", 1, 10, 5)
        if st.button("🧠 สร้างแบบทดสอบ"):
            try:
                bundle = get_bundle(st.session_state["model_label"])
                with st.spinner("กำลังสร้างคำถาม…"):
                    with dspy.context(lm=bundle["lm"]):
                        qpred = bundle["quiz"](
                            learning_material=st.session_state["material"][:4000],
                            vark_style=st.session_state["vark_style"],
                            count=n_q,
                        )
                parsed = parse_json_loose(qpred.questions or "[]")
                questions = []
                for q in parsed:
                    if not isinstance(q, dict):
                        continue
                    opts = q.get("options") or {}
                    if not all(x in opts for x in "ABCD"):
                        continue
                    if q.get("answer") not in list("ABCD"):
                        continue
                    questions.append({
                        "q": str(q.get("q", "")).strip(),
                        "options": {x: str(opts[x]) for x in "ABCD"},
                        "answer": q["answer"],
                        "vark": q.get("vark") if q.get("vark") in list("VARK") else "R",
                        "explanation": str(q.get("explanation", "")).strip(),
                    })
                st.session_state["quiz"] = questions
                st.session_state.pop("quiz_submitted", None)
            except Exception as e:
                st.error(f"สร้างแบบทดสอบไม่ได้: {e}")

        quiz = st.session_state.get("quiz")
        if quiz:
            with st.form("quiz_form"):
                answers = {}
                for idx, q in enumerate(quiz):
                    st.markdown(f"**ข้อ {idx+1}.** {to_streamlit_math(q['q'])}")
                    answers[idx] = st.radio(
                        f"q{idx}",
                        list("ABCD"),
                        format_func=lambda x, q=q: f"{x}. {q['options'][x]}",
                        index=None,
                        label_visibility="collapsed",
                        key=f"ans_{idx}",
                    )
                    st.write("")
                submitted = st.form_submit_button("✅ ส่งคำตอบ")
            if submitted:
                st.session_state["quiz_submitted"] = answers

            graded = st.session_state.get("quiz_submitted")
            if graded is not None:
                correct = sum(1 for idx, q in enumerate(quiz) if graded.get(idx) == q["answer"])
                st.success(f"คะแนน: {correct}/{len(quiz)}")
                for idx, q in enumerate(quiz):
                    user = graded.get(idx)
                    ok = user == q["answer"]
                    icon = "✅" if ok else "❌"
                    with st.expander(f"{icon} ข้อ {idx+1} — เฉลย: {q['answer']} `{q['vark']}`"):
                        st.markdown(f"คุณตอบ: **{user or '(ไม่ได้ตอบ)'}** · เฉลย: **{q['answer']}**")
                        st.markdown(to_streamlit_math(q["explanation"]))
else:
    st.info("⬆️ อัปโหลด PDF แล้วกด **สร้างสื่อการเรียนรู้** เพื่อเริ่ม")
