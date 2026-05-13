import dspy
import json
import os
import random
import sys
import time
from typing import Optional

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # ไม่มี dotenv ก็ยังใช้ env var ปกติได้


# ──────────────────────────────────────────────
# 1. Configure DSPy with TYPHOON
# ──────────────────────────────────────────────
def configure_lm(api_key: Optional[str] = None):
    key = api_key or os.environ.get("TYPHOON_API_KEY")
    if not key:
        raise ValueError("API Key is required")

    lm = dspy.LM(
        model="openai/typhoon-v2.5-30b-a3b-instruct",
        api_key=key,
        api_base="https://api.opentyphoon.ai/v1",
        # Typhoon API hard cap = 16384
        max_tokens=16384,
        temperature=0.7,
        cache=False,
    )
    dspy.settings.configure(lm=lm)
    return lm


# ──────────────────────────────────────────────
# 2a. Signature: วิเคราะห์เนื้อหา (Stage 1)
#     ให้ AI คิดก่อนว่าเนื้อหานี้คืออะไร จะสอนยังไงดี
# ──────────────────────────────────────────────
class ContentAnalyzer(dspy.Signature):
    """
    วิเคราะห์เนื้อหาที่รับมา และวางแผนการสอนที่เหมาะสมกับสไตล์ VARK
    คิดอย่างละเอียดว่าเนื้อหามีแนวคิดอะไร ตัวอย่างอะไร และจะปรับให้เหมาะ VARK ได้อย่างไร
    """
    context: str = dspy.InputField(desc="เนื้อหาที่สกัดจาก PDF หรือเอกสาร")
    vark_style: str = dspy.InputField(
        desc='JSON ของโปรไฟล์ VARK เช่น {"V":40,"A":20,"R":30,"K":10,"dominant":"V"}'
    )
    key_concepts: str = dspy.OutputField(
        desc=(
            "แนวคิดหลัก 5-10 ข้อที่ควรสอน พร้อมระดับความสำคัญ (JSON array) "
            "ต้องครอบคลุม **ทุกประเด็นสำคัญใน context** — ห้ามตัดทิ้งแม้เป็นประเด็นย่อย "
            "ถ้า context ยาวมีหลายหัวข้อ ให้แตกออกเป็นข้อย่อยให้ครบ "
            "อย่าสรุปรวบให้เหลือน้อยเกินจริง"
        )
    )
    teaching_strategy: str = dspy.OutputField(
        desc=(
            "กลยุทธ์การสอนที่เหมาะกับ VARK dominant: "
            "V=เน้นภาพและโครงสร้าง, A=เน้นการเล่าเรื่อง, การฟัง"
            "R=เน้นนิยามและ outline, K=เน้นขั้นตอนและแบบฝึกหัด "
            "อธิบายว่าจะจัดโครงสร้างเนื้อหาอย่างไร"
        )
    )
    difficulty_assessment: str = dspy.OutputField(
        desc="ประเมินความยากของเนื้อหา (easy/medium/hard) และเหตุผล"
    )


# ──────────────────────────────────────────────
# 2b. Signature หลัก: สร้างสื่อ (Stage 2)
#     รับผลจาก Stage 1 มาด้วย ทำให้คิดต่อยอดได้ดีขึ้น
# ──────────────────────────────────────────────
class VARKProjector(dspy.Signature):
    """
    รับเนื้อหา, โปรไฟล์ VARK, และแผนการสอนจาก Stage 1
    สร้างสื่อการเรียนรู้ที่ปรับให้เหมาะกับสไตล์นั้น พร้อม Query สำหรับค้นหา YouTube
    """
    context: str = dspy.InputField(desc="เนื้อหาที่สกัดจาก PDF หรือเอกสาร (plain text)")
    vark_style: str = dspy.InputField(
        desc='JSON string ของโปรไฟล์ VARK เช่น {"V":40,"A":20,"R":30,"K":10,"dominant":"V"}'
    )
    key_concepts: str = dspy.InputField(
        desc="แนวคิดหลักที่วิเคราะห์แล้วจาก Stage 1 (JSON array)"
    )
    teaching_strategy: str = dspy.InputField(
        desc="กลยุทธ์การสอนที่วางแผนแล้วจาก Stage 1"
    )
    learning_material: str = dspy.OutputField(
        desc=(
            "สื่อการเรียนรู้ในรูปแบบ Markdown ที่ปรับให้เหมาะกับสไตล์ VARK "
            "**ภาษา:** ใช้ภาษาเดียวกับ context/PDF — ถ้า context เป็นภาษาไทย ต้องเขียนเนื้อหาทั้งหมดเป็นภาษาไทย "
            "(ห้ามเปลี่ยนไปเป็นภาษาอังกฤษ) ยกเว้นศัพท์เฉพาะ/ตัวอย่างประโยคที่ต้องคงภาษาเดิมตามเนื้อหา. "
            "**ความครบถ้วน (สำคัญสุด):** ต้องครอบคลุม **ทุกประเด็น** ใน context และทุกข้อใน key_concepts "
            "ห้ามตัดประเด็นทิ้ง ห้ามรวบยอด — แต่ละหัวข้อใน context ต้องมีย่อหน้า/หัวข้อย่อยของตัวเอง "
            "ความยาวต้องสเกลตาม context: ถ้า context สั้น (<500 คำ) → เนื้อหา ≥ 800 คำ; "
            "context กลาง (500-2000 คำ) → เนื้อหา ≥ 1500 คำ; "
            "context ยาว (>2000 คำ) → เนื้อหา ≥ 60% ของ context หรือ ≥ 2500 คำ (แล้วแต่อันไหนมากกว่า). "
            "โครงสร้าง: 1) อธิบายแนวคิดหลัก (ทุกข้อใน key_concepts) 2) ตัวอย่างและการประยุกต์ใช้ 3) แบบฝึกหัดหรือสรุป "
            "ปรับตาม VARK dominant อย่างเคร่งครัด — "
            "V=ต้องมีตาราง Markdown (|col|col|) จริงๆ + ใช้โครงสร้างเปรียบเทียบ/จัดกลุ่ม/bullet hierarchy. "
            "**ห้ามอ้างอิงรูปภาพในทุกรูปแบบเด็ดขาด** (ระบบไม่แสดงรูปแล้ว): "
            "ห้ามมีข้อความที่บรรยาย/อ้างถึงรูปภาพในทุกตำแหน่ง (ย่อหน้า, หัวข้อ, cell ในตาราง, bullet, blockquote) — "
            "รวมถึงคำขึ้นต้น 'ภาพ:', 'รูป:', 'image:', 'pic:', 'photo:', "
            "และคำที่ขึ้นต้นด้วย 'ภาพ<...>' / 'รูป<...>' เช่น 'รูปหญิงคนเดียว', 'รูปแว่นตา', 'ภาพแผนผัง' (placeholder ที่ไม่มีรูปจริง). "
            "ถ้าจะทำ matching exercise ห้ามจับคู่ 'รูป↔ประโยค' — ให้ใช้ 'คำ/วลี↔ประโยค' หรือ 'คุณลักษณะ↔ตัวอย่าง' แทน. "
            "ถ้าต้องสื่อ visual ให้ใช้ Markdown table จริง, ASCII diagram (ใน code fence), "
            "หรือ bullet เปรียบเทียบเชิงโครงสร้าง — ไม่ใช่บรรยายภาพที่ไม่มี;"
            "A=ต้องเขียนในรูป story (เล่าเรื่อง) ที่มี narrator บรรยายเหตุการณ์/อธิบายแนวคิด "
            "แล้วแทรก dialogue ของตัวละคร Aural เป็นจุดๆ ใช้คำ story/imagine/dialogue/เล่า/สนทนา. "
            "**โครงสร้างบังคับสำหรับโหมด A:** "
            "(1) ส่วนใหญ่ของเนื้อหา (>=70%) ต้องเป็น *ย่อหน้าบรรยาย/อธิบายแบบปกติ* (ไม่มี `>`) "
            "เล่าฉาก, ปูบริบท, ขยายความแนวคิด, สรุปบทเรียน. "
            "(2) Dialogue ใช้ blockquote โดยขึ้นต้นบล็อกด้วย `> Situation: <บรรยายฉากสั้นๆ>` "
            "แล้วบรรทัดถัดมาต่อเป็นบทพูดในรูป `> ชื่อคนพูด: \"...\"` เช่น:\n"
            "> Situation: ครูวันถามนักเรียนเรื่อง S-V Agreement\n"
            "> ครูวัน: \"ทำไมเราพูดว่า He is happy แต่ They are happy ล่ะ?\"\n"
            "> นักเรียน: \"ไม่แน่ใจครับ มันต่างกันยังไงครับ?\"\n"
            "ตั้งชื่อตัวละคร (ครู, นักเรียน, ชื่อจริง ฯลฯ) ให้เหมาะกับบริบทของเนื้อหา "
            "ใช้แทรกเป็นจุดๆ เฉพาะตอนที่ตัวละครพูด/ถามสำคัญ — เป็นไฮไลต์ ไม่ใช่เนื้อหาหลัก. "
            "(3) **จำนวนบล็อก dialogue ต้องสเกลตามค่า A ใน vark_style:** "
            "A >= 70 → ใส่ได้ 3–5 บล็อกในเนื้อหาทั้งหมด; "
            "A 40–69 → ใส่ได้ 2–3 บล็อก; "
            "A 20–39 → ใส่ได้แค่ 1–2 บล็อก เท่านั้น (ไม่งั้นจะกลบเนื้อหาส่วนอื่น); "
            "A < 20 → ใส่แค่ 0–1 บล็อก. "
            "ห้ามมี blockquote ติดต่อกันเกิน 2 บล็อก. "
            "(4) **ห้ามใช้ `>` เด็ดขาด** กับ: heading (เช่น `> **Practice Sentences:**` ผิด), "
            "คำอธิบายทั่วไป, tip, note, สรุป, รายการตัวอย่างประโยค, bullet list — "
            "สิ่งเหล่านี้ต้องเขียนเป็นย่อหน้า/heading/list ธรรมดา (ไม่มี `>`). "
            "`>` มีไว้สำหรับ Situation+dialogue เท่านั้น. "
            "บทพูดในแต่ละ blockquote กระชับ 1–3 บรรทัด; "
            "R=อธิบายเนื้อหาเป็นข้อความที่จัด format ดี (heading, bullet, numbered list, ตาราง, definition list, นิยามคำสำคัญ) "
            "ให้ครบและละเอียด — ห้ามสั่งให้ผู้เรียน 'สรุปด้วยภาษาตัวเอง' หรือ 'paraphrase' หรือ 'เขียนสรุปของตัวเอง' "
            "ส่วนแบบฝึกหัด ให้ใช้รูปแบบ recall/short-answer/fill-in-blank พร้อมเฉลย และเสริม 'แนวทางการอ่าน/จดโน้ต' "
            "ใช้คำ definition/outline/note/นิยาม/หลักการ/ข้อสังเกต/แนวทาง/ดังนี้; "
            "K=ต้องมี step-by-step และ exercise ให้ผู้เรียนลงมือทำ "
            "และใช้คำ try/fill in/match/complete/apply/exercise/practice/ขั้นตอน/ลองทำ/จับคู่/เติม/ฝึก. "
            "**สำคัญ — รูปแบบเฉลยของแบบฝึกหัด (เคร่งครัด):**\n"
            "ใส่บล็อกเฉลย **เฉพาะเมื่อโจทย์ยังไม่ได้เฉลยให้เห็น**:\n"
            "ตารางอธิบายแนวคิด/นิยาม/ตัวอย่าง ที่มีตัวอย่างประโยคครบในตารางอยู่แล้ว → **ห้ามใส่** ```เฉลย``` ต่อท้าย (ซ้ำซ้อน)\n"
            "**โจทย์/แบบฝึกหัด/กิจกรรมทุกประเภทที่นักเรียนต้องคิดเอง → ต้องใส่ ```เฉลย``` เสมอ** ครอบคลุม: "
            "fill-in-blank, find-the-mistake, match/จับคู่, ordering/จัดเรียงลำดับ/sort, short-answer, "
            "true-false, multiple-choice, scenario/case-study, transform/แปลงประโยค, แต่งประโยค. "
            "ถ้ามีคำว่า 'กิจกรรม', 'แบบฝึกหัด', 'ลองทำ', 'จัดเรียง', 'จับคู่', 'เติม', 'แก้ไข', 'เลือก' ในหัวข้อ → ต้องมีบล็อกเฉลยตามหลัง (ไม่มีข้อยกเว้น)\n"
            "รูปแบบ fence ที่ถูกต้อง** (ไม่งั้น renderer จะไม่จับ):\n"
            "ต้องเขียน ```เฉลย ติดกันบรรทัดเดียว — ห้ามขึ้นบรรทัดใหม่ระหว่าง ``` กับ เฉลย\n"
            "บล็อกต้องเริ่มต้นที่ column 0 (ขอบซ้ายสุด) — ห้าม indent เข้าไปในรายการย่อย\n"
            "ตัวอย่างที่ถูก:\n"
            "```เฉลย\n"
            "<คำตอบและคำอธิบาย>\n"
            "```\n"
            "**โจทย์ต้องมีช่องว่างให้เติม/แก้จริง** — ห้ามแสดงคำตอบในตัวโจทย์:\n"
            "Fill-in-blank: ใช้ ___ หรือ ____ แทนคำตอบในประโยค เช่น 'My glasses ____ on the table.'\n"
            "Find-the-mistake: เขียนเป็น 'The manager ___ busy. (เดิม: are)' แทนการแสดงประโยคที่ถูกแล้ว — ให้นักเรียนเติมรูปที่ถูก แล้วใส่คำตอบใน ```เฉลย``` แยก\n"
            "ห้ามแสดงประโยคถูก/ผิดทั้งคู่ในส่วนโจทย์ — เก็บประโยคที่ถูกไว้ในบล็อกเฉลยเท่านั้น\n"
            "ห้ามเขียนเฉลยแบบเปิดเผยในย่อหน้าธรรมดา (เช่น '**เฉลย:** ...') — ต้องอยู่ใน ```เฉลย ... ``` เสมอ"
        )
    )
    youtube_queries: str = dspy.OutputField(
        desc=(
            "ตอบด้วย JSON array เท่านั้น ห้ามมี text อื่น ห้ามมี markdown fences "
            '3-5 queries เช่น ["if-else C tutorial", "คำสั่ง if-else ภาษา C"]'
        )
    )
    image_queries: str = dspy.OutputField(
        desc=(
            "JSON array เท่านั้น สร้างเฉพาะ dominant=V หรือ มี V เป็น secondary "
            'เช่น [{"q":"if-else flowchart","imgType":"photo","imgSize":"large","rights":"cc_publicdomain","altDescription":"แผนผัง"}] '
            "ถ้าไม่ใช่ V ตอบ: []"
        )
    )


# ──────────────────────────────────────────────
# 3. DSPy Modules
# ──────────────────────────────────────────────

class VARKModule(dspy.Module):
    """
    Runtime module — Chain of Thought 2 stages

    Pipeline:
      Stage 1: ChainOfThought วิเคราะห์เนื้อหาและวางแผนการสอน
      Stage 2: ChainOfThought สร้างสื่อโดยใช้แผนจาก Stage 1
    """
    def __init__(self):
        super().__init__()
        # Stage 1: วิเคราะห์เนื้อหา + วางแผน (ใช้ CoT เพื่อให้ model คิดอย่างเป็นระบบ)
        self.analyze = dspy.ChainOfThought(ContentAnalyzer)
        # Stage 2: สร้างสื่อจากแผนที่วางไว้ (CoT ช่วยให้ output สอดคล้องกับ strategy)
        self.generate = dspy.ChainOfThought(VARKProjector)

    def forward(self, context: str, vark_style: str) -> dspy.Prediction:
        ctx_chars = len(context or "")
        ctx_words = len((context or "").split())

        # ── Stage 1: วิเคราะห์และวางแผน ──
        analysis = self.analyze(context=context, vark_style=vark_style)

        try:
            kc_count = len(json.loads(analysis.key_concepts))
        except Exception:
            kc_count = -1
        print(
            f"[VARKModule] Stage1 done | ctx={ctx_chars} chars / {ctx_words} words "
            f"| key_concepts={kc_count} items"
        )

        # ── Stage 2: สร้างสื่อโดยอิงจากแผน ──
        draft = self.generate(
            context=context,
            vark_style=vark_style,
            key_concepts=analysis.key_concepts,
            teaching_strategy=analysis.teaching_strategy,
        )

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
            f"[VARKModule] Stage2 done | material={mat_chars} chars "
            f"| char_ratio={char_ratio:.2f}x context | fences={fence_count}{warn}"
        )

        return dspy.Prediction(
            learning_material=draft.learning_material,
            youtube_queries=draft.youtube_queries,
            image_queries=draft.image_queries,
            # metadata จาก reasoning chain (ใช้ debug/logging)
            key_concepts=analysis.key_concepts,
            teaching_strategy=analysis.teaching_strategy,
        )


# ──────────────────────────────────────────────
# 2e. Signature: Quiz Generator
# ──────────────────────────────────────────────
class QuizGenerator(dspy.Signature):
    """สร้างแบบทดสอบ MCQ จากเนื้อหาเรียน ตามสไตล์ VARK และระดับความยาก"""
    learning_material: str = dspy.InputField(desc="Markdown learning material")
    vark_style: str        = dspy.InputField(
        desc='JSON เช่น {"V":40,"A":20,"R":30,"K":10,"dominant":"V"}'
    )
    difficulty: str        = dspy.InputField(desc="easy / medium / hard")
    count: int             = dspy.InputField(desc="จำนวนข้อที่ต้องสร้าง")
    questions: str         = dspy.OutputField(
        desc=(
            "ตอบ JSON array เท่านั้น ห้ามมี markdown fences หรือข้อความอื่น "
            "แต่ละข้อ: "
            '{"q":"...","options":{"A":"..","B":"..","C":"..","D":".."},'
            '"answer":"A|B|C|D","vark":"V|A|R|K","diff":"easy|medium|hard",'
            '"explanation":"1–2 ประโยค"}. '
            "ปรับคำถามตาม dominant VARK: "
            "V=ตาราง/แผนภาพ/เปรียบเทียบ, A=story/dialogue, "
            "R=นิยาม/ขั้นตอน/คำศัพท์, K=apply/debug/scenario. "
            "ภาษาตามเนื้อหา (ไทยถ้าเนื้อหาเป็นไทย)"
        )
    )


class QuizModule(dspy.Module):
    """Runtime module สำหรับสร้าง quiz — ใช้ ChainOfThought เพื่อให้ output เป็น JSON ที่ valid"""
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(QuizGenerator)

    def forward(self, learning_material: str, vark_style: str,
                difficulty: str, count: int) -> dspy.Prediction:
        return self.generate(
            learning_material=learning_material,
            vark_style=vark_style,
            difficulty=difficulty,
            count=count,
        )


# ──────────────────────────────────────────────
# 2f. Signature: Video VARK Classifier
# ──────────────────────────────────────────────
class VideoVARKClassifier(dspy.Signature):
    """จัดประเภท YouTube video ตามสไตล์ VARK + subtopics

    Page mapping (จับคู่กับหน้า PDF) ไม่ทำในขั้นนี้ —
    แยกเป็น runtime step ที่รับ PDF จริงมาเทียบกับผลลัพธ์นี้
    """
    video_title: str       = dspy.InputField(desc="ชื่อคลิป")
    video_channel: str     = dspy.InputField(desc="ชื่อช่อง")
    video_metadata: str    = dspy.InputField(
        desc="description / tags / categories ของคลิป (รวมเป็น text)"
    )
    vark_weight_desc: str  = dspy.InputField(
        desc='โปรไฟล์ VARK ของผู้เรียน เช่น "V 60%, K 30%, R 10%"'
    )
    classification: str    = dspy.OutputField(
        desc=(
            "ตอบ JSON เท่านั้น: "
            '{"vark":["V","K"],"subtopics":["..","..."]}. '
            "vark = 1–4 styles ที่ตรงกับเนื้อหาคลิปจริง "
            "(V=diagram/animation, A=lecture/talk, R=text/code, K=hands-on/demo). "
            "subtopics = 2–4 keywords ≤16 chars (ไทยถ้าคลิปเป็นไทย)."
        )
    )


class VideoClassifierModule(dspy.Module):
    """Runtime module สำหรับ classify video — output สั้น ใช้ Predict ก็พอ"""
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(VideoVARKClassifier)

    def forward(self, **kwargs) -> dspy.Prediction:
        return self.classify(**kwargs)


import re as _re

# ──────────────────────────────────────────────
# 4b. LLM-as-Judge (Gemini + GPT-OSS) — Primary metric
# ──────────────────────────────────────────────
#
# แนวทาง: ใช้ LLM ตัวอื่นเป็น judge ประเมิน output ของ VARKModule
# - ส่ง (context, vark_style, learning_material) ให้ judge
# - judge คืนคะแนน 0/1/2 ตาม rubric
# - ใช้ 2 judges จาก 2 provider เพื่อลด self-preference & individual bias:
#     Judge A: Gemini 3.1 flash lite   ผ่าน Google AI Studio (GEMINI_API_KEY)
#     Judge B: GPT-OSS                 ผ่าน OpenRouter        (GPTOSS_API_KEY)
# - คะแนนสุดท้าย = ค่าเฉลี่ยของ judges ที่ parse สำเร็จ / 2.0  (0.0–1.0)
# - ใช้เป็น metric ทั้งใน BootstrapFewShot (compile) และ evaluate_testset
#
# Bias ที่ควรระวัง
#   - Verbosity bias: judges ชอบคำตอบยาว → เราจำกัด material ที่ส่งไปเป็น 5000 chars
#   - Position bias: ไม่เกี่ยว เพราะไม่มี pairwise
#   - Self-preference: ลดด้วยการใช้ 2 judges จาก provider ต่างกัน (Google + OpenAI)
# ──────────────────────────────────────────────

# Judge A: Gemini 3.1 flash lite — Google AI Studio (key = GEMINI_API_KEY)
# Judge B: GPT-OSS                — OpenRouter        (key = GPTOSS_API_KEY)
JUDGE_MODEL_A = os.environ.get("JUDGE_MODEL_A", "gemini/gemini-3.1-flash-lite")
JUDGE_MODEL_B = os.environ.get("JUDGE_MODEL_B", "openrouter/openai/gpt-oss-20b")


def _judge_label(model: str) -> str:
    """ดึง short friendly label จากชื่อ model สำหรับ log

    gemini/gemini-3.1-flash-lite      → 'Gemini'
    openrouter/openai/gpt-oss-20b     → 'GPT-OSS'
    """
    m = model.lower()
    if "gemini" in m:
        return "Gemini"
    if "gpt-oss" in m or "gpt_oss" in m:
        return "GPT-OSS"
    if "gpt" in m:
        return "GPT"
    return model.split("/")[-1]

_JUDGES: Optional[tuple[dspy.LM, dspy.LM]] = None


def configure_judges(
    gemini_key: Optional[str] = None,
    gptoss_key: Optional[str] = None,
    model_a: Optional[str] = None,
    model_b: Optional[str] = None,
) -> tuple[dspy.LM, dspy.LM]:
    """สร้าง 2 judge LMs จาก 2 provider ต่างกัน — cache ใน _JUDGES สำหรับ reuse
      Judge A (Gemini)  ← Google AI Studio,  key=GEMINI_API_KEY  (.env)
      Judge B (GPT-OSS) ← OpenRouter,        key=GPTOSS_API_KEY  (.env)
    """
    g_key = gemini_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY ")
    o_key = gptoss_key or os.environ.get("GPTOSS_API_KEY")

    if not g_key:
        raise ValueError("GEMINI_API_KEY required for Judge A (Google AI Studio)")
    if not o_key:
        raise ValueError("GPTOSS_API_KEY required for Judge B (OpenRouter)")

    ma = model_a or JUDGE_MODEL_A
    mb = model_b or JUDGE_MODEL_B

    # Judge A: Google AI Studio — LiteLLM ใช้ prefix gemini/...  ไม่ต้องตั้ง api_base
    judge_a = dspy.LM(
        model=ma,
        api_key=g_key.strip(),
        max_tokens=400,
        temperature=0.0,
        cache=False,
    )
    # Judge B: OpenRouter — ต้องตั้ง api_base ให้ชี้ openrouter
    # max_tokens 8000: GPT-OSS chain-of-thought reasoning ยาว โดยเฉพาะ Quiz judge
    # ที่ rubric มี 3 primary + secondary criteria (เคย truncate ที่ 3000)
    judge_b = dspy.LM(
        model=mb,
        api_key=o_key.strip(),
        api_base="https://openrouter.ai/api/v1",
        max_tokens=8000,
        temperature=0.0,
        cache=False,
    )
    return judge_a, judge_b


def _get_judges() -> tuple[dspy.LM, dspy.LM]:
    global _JUDGES
    if _JUDGES is None:
        _JUDGES = configure_judges()
    return _JUDGES

# report คู่
# Gemini คนละ version for vid
# transcript
class VARKJudgeScore(dspy.Signature):
    """You are an evaluation judge for a VARK-tailored study guide (learning material).

    Score on a 0-2 scale using 4 criteria. ALL must be met for a 2.

    ── 1. Content Coverage (เนื้อหาครบ) ──
    - ครอบคลุมแนวคิดหลักทั้งหมดจาก `context` หรือไม่?
    - มีตัวอย่าง / sub-topic สำคัญจาก context หรือไม่?
    - ห้ามขาดประเด็นสำคัญที่ผู้เรียนต้องรู้

    ── 2. VARK Style Alignment (ตรงสไตล์) ──
    ตาม dominant style ใน `vark_style`:
      V (Visual)      — ตาราง, แผนภาพ, การเปรียบเทียบเชิงโครงสร้าง
      A (Aural)       — เล่าเรื่อง + dialogue block (Situation + บทพูดตัวละคร)
      R (Read/Write)  — นิยาม, outline, จดโน้ตจัด format, อธิบายเป็นข้อความ
      K (Kinesthetic) — step-by-step, แบบฝึกหัด, scenario apply
    Material ต้องใช้ cue ของสไตล์ dominant อย่างชัดเจน

    ── 3. Pedagogical Quality (สอนเป็น) ──
    - อธิบาย (ไม่ใช่แค่สรุป list)
    - โครงสร้างชัด (heading, section, list)
    - มีส่วนตรวจสอบความเข้าใจ/แบบฝึก ตามสไตล์ VARK

    ── 4. Language ──
    - ใช้ภาษาเดียวกับ `context` (Thai context → Thai material)

    ── Rubric ──
      0 = Poor:      ≥1 criterion ขาดอย่างชัดเจน (เนื้อหาขาดสำคัญ / VARK ไม่ตรง / โครงสร้างพัง / ผิดภาษา)
      1 = Partial:   ผ่านเกณฑ์หลัก แต่มี gap ใน 1+ criterion
      2 = Excellent: ผ่านครบทั้ง 4

    Output `score` เป็นเลขจำนวนเต็ม 0, 1, หรือ 2 เท่านั้น
    """
    context = dspy.InputField(desc="Original source content (text from PDF/document)")
    vark_style = dspy.InputField(
        desc='Target VARK profile JSON, e.g. {"V":40,"A":20,"R":30,"K":10,"dominant":"V"}'
    )
    reference_rationale = dspy.InputField(
        desc=(
            "Optional gold-standard rationale (เกณฑ์ที่ dataset author เขียนเป็น reference). "
            "ถ้าไม่ว่าง → judge ใช้เป็น ground truth ตัดสินว่า material ตรงเกณฑ์ไหม. "
            "ถ้าว่าง → blind judge mode"
        )
    )
    learning_material = dspy.InputField(desc="Generated study guide to evaluate")
    reasoning = dspy.OutputField(
        desc="1-2 ประโยค — ถ้าคะแนน < 2 บอกว่า criterion ไหนอ่อนที่สุด"
    )
    score = dspy.OutputField(desc="Single integer: 0, 1, or 2")


class QuizJudgeScore(dspy.Signature):
    """You are an evaluation judge for a VARK-style multiple-choice quiz.

    Score on a 0-2 scale. Focus on 3 PRIMARY criteria:

    ── 1. คุณภาพโจทย์ (Question Quality) — ดีเทียบกับบทเรียนไหม ──
    - แต่ละข้อตอบได้จาก `learning_material` หรือไม่? (ห้ามอิงความรู้นอกบทเรียน)
    - โจทย์ตรงประเด็นสำคัญของบทเรียน — ไม่ใช่ถามรายละเอียดเล็กน้อย/ตัวเลขจิ๊บจ๊อย
    - คำถามชัดเจน ไม่กำกวม ไม่มีคำถามซ้ำกัน

    ── 2. คุณภาพ choice (Distractor Quality) — choice ที่สร้างดีไหม ──
    - distractor (3 ตัวที่ผิด) ดูสมเหตุสมผล — ใกล้คำตอบจริง ไม่ใช่ผิดอย่างเห็นได้ชัด
    - choice ทั้ง 4 แตกต่างกันจริง (ไม่มี wording ซ้ำ/ความหมายซ้ำ)
    - ความยาวใกล้เคียง (ไม่ใช่ "ตัวถูกยาวกว่าเพื่อนเห็นๆ")
    - ไม่มี "ถูกทุกข้อ" / "ผิดทุกข้อ" แบบ shortcut เว้นแต่มีเหตุผลจริง

    ── 3. คำตอบที่เฉลย (Answer Correctness) — เฉลยถูกไหม ──
    - `answer` letter ต้องตรงกับ option ที่ถูกตามเนื้อหา `learning_material`
    - `explanation` อธิบายเหตุผลที่ถูกต้อง สอดคล้องกับเฉลย
    - **ถ้าข้อใดเฉลยผิด → คะแนนต้อง ≤ 1 เสมอ**

    ── Secondary checks (ลดคะแนนเล็กน้อย ไม่ใช่ critical) ──
    - JSON valid, มี key ครบ (`q`, `options`, `answer`, `vark`, `diff`, `explanation`)
    - `vark` tag สไตล์ตรงกับลักษณะคำถาม (V=ภาพ/ตาราง, A=story, R=นิยาม, K=apply)
    - `diff` ตรงระดับที่ขอใน `difficulty`
    - ภาษาตรงกับ `learning_material` (ไทย → ไทย)

    ── Rubric ──
      0 = Poor:      ≥1 ข้อ answer ผิด, หรือ JSON invalid, หรือ off-topic, หรือ distractor ห่วยทั้งชุด
      1 = Partial:   ทุกข้อ on-topic + answer ถูก แต่ distractor บางข้ออ่อน
                     หรือ VARK/difficulty เพี้ยนใน ≥1 ข้อ
      2 = Excellent: ทุกข้อผ่านทั้ง 3 primary criteria + secondary ส่วนใหญ่

    Output `score` เป็นเลขจำนวนเต็ม 0, 1, หรือ 2 เท่านั้น
    """
    learning_material = dspy.InputField(desc="Source material the quiz is based on")
    vark_style = dspy.InputField(desc="Target VARK profile JSON")
    difficulty = dspy.InputField(desc="Requested difficulty: easy / medium / hard")
    reference_rationale = dspy.InputField(
        desc=(
            "Optional gold reference (required topics, expected dominant_vark, rationale). "
            "ถ้าไม่ว่าง → judge ใช้เป็น ground truth: คำถามต้องครอบคลุม topics, "
            "vark tag ส่วนใหญ่ตรง dominant, ตรรกะของ quiz สอดคล้องกับ rationale. "
            "ถ้าว่าง → blind judge mode"
        )
    )
    questions = dspy.InputField(desc="Generated quiz JSON array to evaluate")
    reasoning = dspy.OutputField(
        desc=(
            "1-2 ประโยค — ถ้าคะแนน < 2 ระบุข้อที่อ่อนที่สุด + "
            "criterion ไหนใน 3 ข้อหลักไม่ผ่าน (question/choice/answer)"
        )
    )
    score = dspy.OutputField(desc="Single integer: 0, 1, or 2")


def _judge_score(example, prediction, judge_lm, with_reference: bool = True) -> int:
    """รัน 1 judge บน 1 example. คืน 0/1/2 หรือ -1 ถ้า parse ไม่ได้
    with_reference=False → blind mode (judge ไม่เห็น expected/rationale)
    """
    judge = dspy.Predict(VARKJudgeScore)
    material = (prediction.learning_material or "")[:5000]
    ctx = (example.context or "")[:3000]
    reference = _format_reference(example) if with_reference else ""
    try:
        with dspy.context(lm=judge_lm):
            result = judge(
                context=ctx,
                vark_style=example.vark_style,
                reference_rationale=reference,
                learning_material=material,
            )
        m = _re.search(r"[012]", str(getattr(result, "score", "")))
        return int(m.group()) if m else -1
    except Exception as e:
        print(f"[judge:{getattr(judge_lm, 'model', '?')}] error: {type(e).__name__}: {e}")
        return -1


def vark_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """
    LLM-as-Judge metric — เฉลี่ยคะแนนจาก 2 judges
      Gemini 3.1 flash lite + GPT-OSS (ผ่าน OpenRouter)
    คืนคะแนน 0.0–1.0 (judge raw 0/1/2 → /2); คืน 0.0 ถ้า judges ไม่พร้อมใช้
    threshold 0.5 ≈ "ทั้ง 2 judge ให้อย่างน้อย 1"
    """
    try:
        judge_a, judge_b = _get_judges()
    except Exception as e:
        if trace:
            print(f"[vark_metric] judges unavailable ({e}) — return 0.0")
        return 0.0

    s_a = _judge_score(example, prediction, judge_a)
    s_b = _judge_score(example, prediction, judge_b)
    valid = [s for s in (s_a, s_b) if s >= 0]
    if not valid:
        return 0.0
    avg_norm = sum(valid) / len(valid) / 2.0

    if trace:
        la = _judge_label(JUDGE_MODEL_A)
        lb = _judge_label(JUDGE_MODEL_B)
        print(
            f"[vark_metric] {la}={s_a} {lb}={s_b} "
            f"avg_raw={sum(valid)/len(valid):.2f}/2 norm={avg_norm:.3f}"
        )

    return round(avg_norm, 4)


def _judge_score_with_feedback(
    example, prediction, judge_lm, with_reference: bool = True
) -> tuple[int, str]:
    """รัน 1 judge บน 1 example — คืน (score 0/1/2 หรือ -1, reasoning text)
    ใช้สำหรับ GEPA ที่ต้องการ textual feedback เพื่อ reflect ปรับ prompt
    """
    judge = dspy.Predict(VARKJudgeScore)
    material = (prediction.learning_material or "")[:5000]
    ctx = (example.context or "")[:3000]
    reference = _format_reference(example) if with_reference else ""
    try:
        with dspy.context(lm=judge_lm):
            result = judge(
                context=ctx,
                vark_style=example.vark_style,
                reference_rationale=reference,
                learning_material=material,
            )
        reasoning = str(getattr(result, "reasoning", "") or "").strip()
        m = _re.search(r"[012]", str(getattr(result, "score", "")))
        return (int(m.group()) if m else -1), reasoning
    except Exception as e:
        return -1, f"[judge error] {type(e).__name__}: {e}"


def vark_metric_gepa(
    gold, pred, trace=None, pred_name=None, pred_trace=None
):
    """GEPA-compatible metric — เฉลี่ย 2 judges (Gemini + GPT-OSS)
    คืน dspy.Prediction(score, feedback) ให้ reflection_lm ใช้ปรับ prompt
    """
    try:
        judge_a, judge_b = _get_judges()
    except Exception as e:
        return dspy.Prediction(score=0.0, feedback=f"judges unavailable: {e}")

    s_a, fb_a = _judge_score_with_feedback(gold, pred, judge_a)
    s_b, fb_b = _judge_score_with_feedback(gold, pred, judge_b)
    valid = [s for s in (s_a, s_b) if s >= 0]
    score = (sum(valid) / len(valid) / 2.0) if valid else 0.0

    la = _judge_label(JUDGE_MODEL_A)
    lb = _judge_label(JUDGE_MODEL_B)
    feedback = (
        f"Avg judge score = {score:.3f} (norm 0-1)\n"
        f"{la} gave {s_a}/2: {fb_a}\n"
        f"{lb} gave {s_b}/2: {fb_b}"
    )
    return dspy.Prediction(score=round(score, 4), feedback=feedback)


def _quiz_judge_score(example, prediction, judge_lm, with_reference: bool = True) -> int:
    """รัน 1 judge สำหรับ Quiz output — คืน 0/1/2 หรือ -1 ถ้า parse ไม่ได้
    with_reference=False → blind mode
    """
    judge = dspy.Predict(QuizJudgeScore)
    questions = (prediction.questions or "")[:6000]
    material = (example.learning_material or "")[:4000]
    reference = _format_reference(example) if with_reference else ""
    try:
        with dspy.context(lm=judge_lm):
            result = judge(
                learning_material=material,
                vark_style=example.vark_style,
                difficulty=example.difficulty,
                reference_rationale=reference,
                questions=questions,
            )
        m = _re.search(r"[012]", str(getattr(result, "score", "")))
        return int(m.group()) if m else -1
    except Exception as e:
        print(f"[quiz_judge:{getattr(judge_lm, 'model', '?')}] error: {type(e).__name__}: {e}")
        return -1


def quiz_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """LLM-as-Judge metric สำหรับ QuizModule — เฉลี่ย 2 judges → 0.0–1.0"""
    try:
        judge_a, judge_b = _get_judges()
    except Exception as e:
        if trace:
            print(f"[quiz_metric] judges unavailable ({e}) — return 0.0")
        return 0.0

    s_a = _quiz_judge_score(example, prediction, judge_a)
    s_b = _quiz_judge_score(example, prediction, judge_b)
    valid = [s for s in (s_a, s_b) if s >= 0]
    if not valid:
        return 0.0
    avg_norm = sum(valid) / len(valid) / 2.0

    if trace:
        la = _judge_label(JUDGE_MODEL_A)
        lb = _judge_label(JUDGE_MODEL_B)
        print(f"[quiz_metric] {la}={s_a} {lb}={s_b} norm={avg_norm:.3f}")
    return round(avg_norm, 4)


# ─ Gemini native video endpoint (Judge A สำหรับ video — ดู YouTube จริง) ─
_GEMINI_NATIVE_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"


def _format_reference(example: dspy.Example) -> str:
    """รวม example.expected เป็น reference text สำหรับ judge ทุกตัว

    รองรับทุก schema:
      video: vark / subtopics / rationale
      quiz : topics / dominant_vark / rationale
      vark : rationale อย่างเดียว (ถ้ามี — โดยปกติไม่มี)
    คืน '' ถ้าไม่มี expected → judge จะทำงานในโหมด blind
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
    # quiz-style
    if exp.get("topics"):
        parts.append(f"Required topics: {exp['topics']}")
    if exp.get("dominant_vark"):
        parts.append(f"Expected dominant VARK: {exp['dominant_vark']}")
    # shared
    if exp.get("rationale"):
        parts.append(f"Rationale: {exp['rationale']}")
    return "\n".join(parts)


# alias เพื่อ backward-compat
_format_video_reference = _format_reference


def _video_judge_a_native(
    example: dspy.Example,
    prediction: dspy.Prediction,
    model_name: str,
    api_key: str,
    with_reference: bool = True,
) -> int:
    """Judge A — Gemini ดู YouTube video เองผ่าน file_data.file_uri

    bypass DSPy/LiteLLM เพราะ signature ของ Gemini สำหรับ video URL
    ต้องส่งเป็น content part `file_data` ซึ่ง LiteLLM ยังไม่ exposed ตรงๆ
    with_reference=False → blind mode (Gemini ไม่เห็น expected)
    """
    video_url = getattr(example, "video_url", "") or ""
    if not video_url or "REPLACE" in video_url:
        print("[video_judge_a] no video_url — skip")
        return -1

    classification = (prediction.classification or "")[:2000]
    # ลอก prefix "gemini/" ที่ DSPy/LiteLLM ใช้
    model_id = model_name.split("/", 1)[-1] if "/" in model_name else model_name
    # Gemini ไม่ชอบ query string ยาว — strip ทุกอย่างหลัง watch?v=...
    clean_url = _re.split(r"[&#]", video_url, maxsplit=1)[0]

    reference = _format_reference(example) if with_reference else ""
    reference_block = (
        f"\nReference (gold standard — use as ground truth):\n{reference}\n"
        if reference else ""
    )

    ref_criterion = (
        "4. Reference consistency: candidate must be consistent with the Reference; "
        "near-match + rationale-supported = 2; contradicts rationale = 0.\n"
    ) if reference else ""

    prompt = (
        "Watch the attached YouTube video and judge a VARK classification.\n\n"
        f"Learner VARK profile: {example.vark_weight_desc}\n"
        f"{reference_block}\n"
        f"Classification (JSON) to evaluate:\n{classification}\n\n"
        "Evaluation criteria:\n"
        "1. VARK fit: vark[] matches what you actually see/hear "
        "(V=visual diagram/animation, A=lecture/talk, R=text/code, K=hands-on/demo).\n"
        "2. Subtopic relevance: 2–4 short keywords (≤16 chars) that reflect "
        "actual content; Thai video → Thai subtopics.\n"
        "3. JSON validity: valid JSON; required keys = vark, subtopics.\n"
        + ref_criterion
        + "\nRubric: 0 = poor, 1 = partial, 2 = excellent.\n"
        "Respond with ONLY a single digit: 0, 1, or 2."
    )

    payload = {
        "contents": [{
            "parts": [
                {"file_data": {"file_uri": clean_url}},
                {"text": prompt},
            ],
        }],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 32},
    }

    url = f"{_GEMINI_NATIVE_ENDPOINT}/{model_id}:generateContent"
    # key ส่งทาง header — ไม่ leak ใน URL/error log
    headers = {"x-goog-api-key": api_key}
    transient = {429, 500, 502, 503, 504}

    last_err: Optional[Exception] = None
    for attempt in range(4):  # 1 ครั้งแรก + 3 retries
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=180)
            if r.status_code in transient:
                wait = (2 ** attempt) + random.random()
                print(f"[video_judge_a] HTTP {r.status_code} — retry in {wait:.1f}s "
                      f"(attempt {attempt + 1}/4)")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            m = _re.search(r"[012]", text)
            return int(m.group()) if m else -1
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            last_err = e
            wait = (2 ** attempt) + random.random()
            print(f"[video_judge_a] {type(e).__name__} — retry in {wait:.1f}s")
            time.sleep(wait)
        except Exception as e:
            # non-retryable (4xx อื่นๆ, parse error) — ออกเลย
            print(f"[video_judge_a (Gemini native)] {type(e).__name__}: {e}")
            return -1

    print(f"[video_judge_a (Gemini native)] giving up after 4 attempts ({last_err})")
    return -1


def _gemini_api_key() -> str:
    return (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GEMINI_API_KEY ")
        or ""
    ).strip()


def video_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """LLM-as-Judge metric สำหรับ video classifier — Gemini ดู YouTube video ตัวเดียว
    คะแนน 0/1/2 → normalize เป็น 0.0–1.0; คืน 0.0 ถ้า judge ไม่ valid
    """
    g_key = _gemini_api_key()
    if not g_key:
        if trace:
            print("[video_metric] no GEMINI_API_KEY — return 0.0")
        return 0.0

    s = _video_judge_a_native(example, prediction, JUDGE_MODEL_A, g_key)
    if s < 0:
        return 0.0
    norm = s / 2.0
    if trace:
        la = _judge_label(JUDGE_MODEL_A)
        print(f"[video_metric] {la}[watches video]={s} norm={norm:.3f}")
    return round(norm, 4)


def evaluate_testset(module, testset: list, blind: bool = False) -> dict:
    """
    Evaluate VARK study guide testset ด้วย LLM-as-Judge 2 ตัว
      - Gemini 3.1 flash lite (judge A)
      - GPT-OSS via OpenRouter (judge B)
    รายงาน mean, distribution, agreement rate, mean abs diff
    blind=True → judges ไม่เห็น expected.rationale (ลด reference leak)
    """
    judge_a, judge_b = _get_judges()
    la = _judge_label(JUDGE_MODEL_A)
    lb = _judge_label(JUDGE_MODEL_B)
    mode = "blind" if blind else "ref-augmented"
    print(f"\n🔎 VARK eval mode: {mode}  ({la} + {lb})")

    scores_a, scores_b = [], []
    n = len(testset)
    for i, ex in enumerate(testset, 1):
        pred = module(context=ex.context, vark_style=ex.vark_style)
        sa = _judge_score(ex, pred, judge_a, with_reference=not blind)
        sb = _judge_score(ex, pred, judge_b, with_reference=not blind)
        scores_a.append(sa)
        scores_b.append(sb)
        print(f"  [{i:>3}/{n}] {la}={sa} {lb}={sb}")

    return _print_judge_report(f"VARK [{mode}]", n, scores_a, scores_b)


def _summarize_judge_scores(scores: list[int]) -> dict:
    valid = [s for s in scores if s >= 0]
    if not valid:
        return {"scored": 0, "mean_raw": 0.0, "mean_norm": 0.0, "dist": {0: 0, 1: 0, 2: 0}}
    return {
        "scored": len(valid),
        "mean_raw": round(sum(valid) / len(valid), 3),
        "mean_norm": round(sum(valid) / len(valid) / 2, 3),
        "dist": {k: valid.count(k) for k in (0, 1, 2)},
    }


def _print_judge_report(label: str, n: int, scores_a: list[int], scores_b: list[int]) -> dict:
    name_a = JUDGE_MODEL_A.split("/")[-1]
    name_b = JUDGE_MODEL_B.split("/")[-1]
    la = _judge_label(JUDGE_MODEL_A)
    lb = _judge_label(JUDGE_MODEL_B)
    sum_a = _summarize_judge_scores(scores_a)
    sum_b = _summarize_judge_scores(scores_b)
    pairs = [(a, b) for a, b in zip(scores_a, scores_b) if a >= 0 and b >= 0]
    exact = (sum(1 for a, b in pairs if a == b) / len(pairs)) if pairs else 0.0
    mad = (sum(abs(a - b) for a, b in pairs) / len(pairs)) if pairs else 0.0
    report = {
        "module": label,
        "n": n,
        "judge_a": {"model": name_a, **sum_a},
        "judge_b": {"model": name_b, **sum_b},
        "agreement_rate": round(exact, 3),
        "mean_abs_diff": round(mad, 3),
    }
    print(f"\n📊 {label} — LLM-as-Judge Evaluation Report")
    print(f"   Samples         : {n}")
    for tag, summ in ((la, report["judge_a"]), (lb, report["judge_b"])):
        print(f"\n   {tag} — {summ['model']}")
        print(f"     scored       : {summ['scored']}/{n}")
        print(f"     mean (0-2)   : {summ['mean_raw']}")
        print(f"     normalized   : {summ['mean_norm']}")
        print(f"     distribution : 0={summ['dist'][0]}  1={summ['dist'][1]}  2={summ['dist'][2]}")
    print(f"\n   Agreement (exact match) : {report['agreement_rate']}")
    print(f"   Mean abs diff (0–2)     : {report['mean_abs_diff']}")
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


def _compare_video_to_expected(prediction_json: str, expected: dict) -> dict:
    """
    เทียบ video classification output กับ expected labels
    คืน dict ของผลเปรียบเทียบแต่ละ field + match_score รวม
    """
    actual = _safe_json(prediction_json)
    if not isinstance(actual, dict):
        return {"parsed": False, "match_score": 0.0}

    out: dict = {"parsed": True}
    scores: list[float] = []

    if "vark" in expected:
        e, a = set(expected["vark"]), set(actual.get("vark", []) or [])
        jac = len(e & a) / len(e | a) if (e | a) else 1.0
        out["vark"] = {"expected": sorted(e), "actual": sorted(a),
                       "exact": e == a, "jaccard": round(jac, 3)}
        scores.append(jac)

    if "subtopics" in expected:
        kws = expected["subtopics"] or []
        actual_subs = actual.get("subtopics", []) or []
        joined = " ".join(actual_subs).lower()
        hits = [k for k in kws if k.lower() in joined]
        recall = len(hits) / len(kws) if kws else 1.0
        out["subtopics"] = {"expected": kws, "actual": actual_subs,
                            "hits": hits, "misses": [k for k in kws if k not in hits],
                            "recall": round(recall, 3)}
        scores.append(recall)

    out["match_score"] = round(sum(scores) / len(scores), 3) if scores else 0.0
    return out


def _compare_quiz_to_expected(prediction_json: str, expected: dict) -> dict:
    """
    เทียบ quiz output (JSON array) กับ expected labels
    เช็ค: topics ที่ต้องคลุม, dominant_vark share
    """
    actual = _safe_json(prediction_json)
    if not isinstance(actual, list):
        return {"parsed": False, "match_score": 0.0}

    out: dict = {"parsed": True, "n_questions": len(actual)}
    scores: list[float] = []

    if "topics" in expected and expected["topics"]:
        topics = expected["topics"]
        big = []
        for q in actual:
            if not isinstance(q, dict):
                continue
            big.append(str(q.get("q", "")))
            opts = q.get("options") or {}
            if isinstance(opts, dict):
                big.extend(str(v) for v in opts.values())
            big.append(str(q.get("explanation", "")))
        joined = " ".join(big).lower()
        hits = [t for t in topics if t.lower() in joined]
        recall = len(hits) / len(topics)
        out["topics"] = {"expected": topics, "hits": hits,
                         "misses": [t for t in topics if t not in hits],
                         "recall": round(recall, 3)}
        scores.append(recall)

    if "dominant_vark" in expected:
        dom = expected["dominant_vark"]
        tags = [q.get("vark", "") for q in actual if isinstance(q, dict)]
        share = (sum(1 for t in tags if t == dom) / len(tags)) if tags else 0.0
        out["dominant_vark"] = {"expected": dom, "share": round(share, 3),
                                "ok": share >= 0.5}
        scores.append(share)

    out["match_score"] = round(sum(scores) / len(scores), 3) if scores else 0.0
    return out


def evaluate_quiz_testset(module, testset: list, blind: bool = False) -> dict:
    """Evaluate QuizModule + เทียบกับ expected ถ้ามี label
    blind=True → judges ไม่เห็น expected.topics/dominant_vark/rationale
    """
    judge_a, judge_b = _get_judges()
    la = _judge_label(JUDGE_MODEL_A)
    lb = _judge_label(JUDGE_MODEL_B)
    mode = "blind" if blind else "ref-augmented"
    print(f"\n🔎 Quiz eval mode: {mode}  ({la} + {lb})")
    scores_a, scores_b = [], []
    match_scores: list[float] = []
    comparisons: list[dict] = []
    n = len(testset)
    for i, ex in enumerate(testset, 1):
        pred = module(
            learning_material=ex.learning_material,
            vark_style=ex.vark_style,
            difficulty=ex.difficulty,
            count=ex.count,
        )
        sa = _quiz_judge_score(ex, pred, judge_a, with_reference=not blind)
        sb = _quiz_judge_score(ex, pred, judge_b, with_reference=not blind)
        scores_a.append(sa)
        scores_b.append(sb)

        line = f"  [{i:>3}/{n}] {la}={sa} {lb}={sb}"
        expected = getattr(ex, "expected", None)
        if expected:
            cmp = _compare_quiz_to_expected(pred.questions, expected)
            comparisons.append(cmp)
            if cmp.get("parsed"):
                match_scores.append(cmp["match_score"])
                bits = []
                if "topics" in cmp:
                    bits.append(f"topics {cmp['topics']['recall']:.0%}")
                if "dominant_vark" in cmp:
                    ok = "✓" if cmp["dominant_vark"]["ok"] else "✗"
                    bits.append(f"dom-vark {cmp['dominant_vark']['share']:.0%} {ok}")
                if bits:
                    line += " | " + " ".join(bits)
            else:
                line += " | ⚠ invalid JSON"
        print(line)

    report = _print_judge_report(f"Quiz [{mode}]", n, scores_a, scores_b)
    if match_scores:
        avg = sum(match_scores) / len(match_scores)
        report["expected_match_avg"] = round(avg, 3)
        report["n_labeled"] = len(match_scores)
        print(f"   Expected-match  : {avg:.3f}  ({len(match_scores)}/{n} labeled)")
    return report


def evaluate_video_testset(module, testset: list, blind: bool = False) -> dict:
    """Evaluate VideoClassifierModule — Gemini ดู YouTube video ตัวเดียว
    + เทียบกับ expected ถ้ามี label
    blind=True → Gemini ไม่เห็น expected (ดูจากคลิปล้วนๆ)
    """
    g_key = _gemini_api_key()
    if not g_key:
        raise ValueError("GEMINI_API_KEY required for video judge")
    mode = "blind" if blind else "ref-augmented"
    print(f"\n🔎 Video eval mode: {mode}")
    la = _judge_label(JUDGE_MODEL_A)
    name = f"{la} ({JUDGE_MODEL_A.split('/')[-1]}) [video]"
    scores: list[int] = []
    match_scores: list[float] = []
    n = len(testset)
    for i, ex in enumerate(testset, 1):
        pred = module(
            video_title=ex.video_title,
            video_channel=ex.video_channel,
            video_metadata=ex.video_metadata,
            vark_weight_desc=ex.vark_weight_desc,
        )
        s = _video_judge_a_native(ex, pred, JUDGE_MODEL_A, g_key,
                                  with_reference=not blind)
        scores.append(s)

        line = f"  [{i:>3}/{n}] {name}={s}"
        expected = getattr(ex, "expected", None)
        if expected:
            cmp = _compare_video_to_expected(pred.classification, expected)
            if cmp.get("parsed"):
                match_scores.append(cmp["match_score"])
                bits = []
                if "vark" in cmp:
                    bits.append("vark " + ("✓" if cmp["vark"]["exact"]
                                           else f"~{cmp['vark']['jaccard']:.0%}"))
                if "subtopics" in cmp:
                    bits.append(f"subt {cmp['subtopics']['recall']:.0%}")
                if bits:
                    line += " | " + " ".join(bits)
            else:
                line += " | ⚠ invalid JSON"
        print(line)

    summ = _summarize_judge_scores(scores)
    print(f"\n📊 Video [{mode}] — LLM-as-Judge (Gemini watches video)")
    print(f"   Samples         : {n}")
    print(f"   Judge — {name}")
    print(f"     scored       : {summ['scored']}/{n}")
    print(f"     mean (0-2)   : {summ['mean_raw']}")
    print(f"     normalized   : {summ['mean_norm']}")
    print(f"     distribution : 0={summ['dist'][0]}  1={summ['dist'][1]}  2={summ['dist'][2]}")

    report = {"module": "Video", "n": n, "judge": {"model": name, **summ}}
    if match_scores:
        avg = sum(match_scores) / len(match_scores)
        report["expected_match_avg"] = round(avg, 3)
        report["n_labeled"] = len(match_scores)
        print(f"   Expected-match  : {avg:.3f}  ({len(match_scores)}/{n} labeled)")
    return report


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


def get_valset() -> list[dspy.Example]:
    return get_trainset("dataset/val/vark_val.json")


def get_testset() -> list[dspy.Example]:
    return get_trainset("dataset/test/vark_test.json")


def get_quiz_trainset(path: str = "dataset/train/quiz_train.json") -> list[dspy.Example]:
    """
    Quiz trainset format (JSON array):
      [
        {
          "id": "quiz_001",
          "learning_material": "<markdown ของ study guide>",
          "vark_style": {"V":50,"A":10,"R":10,"K":30,"dominant":"V"},
          "difficulty": "medium",          # easy / medium / hard
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
            "difficulty": r["difficulty"],
            "count": int(r.get("count", 5)),
        }
        if r.get("expected"):
            kw["expected"] = r["expected"]
        examples.append(
            dspy.Example(**kw).with_inputs(
                "learning_material", "vark_style", "difficulty", "count"
            )
        )
    return examples


def get_quiz_valset() -> list[dspy.Example]:
    return get_quiz_trainset("dataset/val/quiz_val.json")


def get_quiz_testset() -> list[dspy.Example]:
    return get_quiz_trainset("dataset/test/quiz_test.json")


def get_video_valset() -> list[dspy.Example]:
    return get_video_trainset("dataset/val/video_val.json")


def get_video_testset() -> list[dspy.Example]:
    return get_video_trainset("dataset/test/video_test.json")


def _vark_to_desc(vark: dict) -> str:
    """แปลง VARK JSON {V:60,A:10,R:10,K:20} → 'V 60%, A 10%, R 10%, K 20%'"""
    parts = [f"{s} {int(vark.get(s, 0))}%" for s in "VARK" if int(vark.get(s, 0)) > 0]
    return ", ".join(parts)


def get_video_trainset(path: str = "dataset/train/video_train.json") -> list[dspy.Example]:
    """
    Video classifier trainset — รองรับ 2 รูปแบบ:

    A) Compact (ผู้ใช้กรอกเอง):
       {
         "id": "vid_001",
         "video_url": "https://www.youtube.com/watch?v=...",
         "vark_style": {"V":60,"A":10,"R":10,"K":20,"dominant":"V"}
       }
       → ต้องรัน `python enrich_video_dataset.py <path>` ก่อน
         เพื่อให้ YouTube API เติม video_title/channel/metadata

    B) Full (หลัง enrich แล้ว):
       เพิ่ม video_title, video_channel, video_metadata, video_id, collected_at

    vark_style (JSON) จะถูกแปลงเป็น vark_weight_desc โดยอัตโนมัติ
    หรือใส่ vark_weight_desc string ตรงๆ ก็ได้
    """
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)

    examples = []
    for r in rows:
        # ─ vark: รับทั้ง JSON หรือ pre-formatted string ─
        if r.get("vark_weight_desc"):
            desc = r["vark_weight_desc"]
        elif isinstance(r.get("vark_style"), dict):
            desc = _vark_to_desc(r["vark_style"])
        else:
            raise ValueError(
                f"Entry {r.get('id','?')}: ต้องมี 'vark_style' (JSON) "
                f"หรือ 'vark_weight_desc' (string)"
            )

        # ─ check enrichment status ─
        title = r.get("video_title", "")
        channel = r.get("video_channel", "")
        if (not title) or (not channel) or "REPLACE" in title or "REPLACE" in channel:
            raise ValueError(
                f"Entry {r.get('id','?')}: ยังไม่ enrich — ขาด video_title/channel.\n"
                f"  รัน: python enrich_video_dataset.py {path}"
            )

        kw = {
            "video_title": title,
            "video_channel": channel,
            "video_metadata": r.get("video_metadata", ""),
            "vark_weight_desc": desc,
            # video_url ใช้สำหรับ Gemini judge ดู YouTube — ไม่ใช่ input ของ classifier
            "video_url": r.get("video_url", ""),
        }
        if r.get("expected"):
            kw["expected"] = r["expected"]
        examples.append(
            dspy.Example(**kw).with_inputs(
                "video_title", "video_channel", "video_metadata",
                "vark_weight_desc"
            )
        )
    return examples

# ──────────────────────────────────────────────
# 6. Compile (ใช้ VARKModule ตัวเต็ม + BootstrapFewShot)
#     compile target ต้องเป็น VARKModule (2-stage) เพื่อให้ demos ที่บันทึก
#     มี field ครบ (key_concepts, teaching_strategy, image_queries)
#     ตรงกับ signature ที่ runtime ใช้จริง
# ──────────────────────────────────────────────
def compile_with_gepa(
    model_path: str = "vark_model.json",
    api_key: Optional[str] = None,
    auto: Optional[str] = "light",
    max_metric_calls: Optional[int] = None,
):
    """GEPA optimizer — reflective prompt evolution
    ใช้ judge reasoning เป็น feedback ให้ reflection_lm propose prompt ใหม่

    Budget (เลือก 1):
      auto: 'light' / 'medium' / 'heavy'   (auto-budget; default 'light')
      max_metric_calls: int                 (override → ตั้ง budget เอง เช่น 30)
    """
    configure_lm(api_key)
    from dspy.teleprompt import GEPA

    trainset = get_trainset()
    valset = get_valset()

    # Reflection LM — ใช้ Gemini (เร็ว+ฟรี) แทน Typhoon (ช้า)
    # GEPA จะเรียก reflection LM หลายครั้งเพื่อ propose prompt ใหม่
    # Gemini Flash Lite สรุปเหตุผลและเขียน prompt ได้ดี + รวดเร็ว
    g_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY ")
    if g_key:
        reflection_lm = dspy.LM(
            model=JUDGE_MODEL_A,
            api_key=g_key.strip(),
            max_tokens=8000,
            temperature=1.0,
            cache=False,
        )
        print(f"[GEPA] reflection_lm = {JUDGE_MODEL_A} (Gemini — fast)")
    else:
        reflection_lm = dspy.LM(
            model="openai/typhoon-v2.5-30b-a3b-instruct",
            api_key=os.environ.get("TYPHOON_API_KEY") or api_key,
            api_base="https://api.opentyphoon.ai/v1",
            max_tokens=8000,
            temperature=1.0,
            cache=False,
        )
        print("[GEPA] reflection_lm = Typhoon (fallback — no GEMINI_API_KEY)")

    gepa_kwargs = dict(
        metric=vark_metric_gepa,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=2,
        track_stats=True,
        warn_on_score_mismatch=False,
        seed=0,
    )
    if max_metric_calls is not None:
        gepa_kwargs["max_metric_calls"] = max_metric_calls
        print(f"[GEPA] budget: max_metric_calls={max_metric_calls}")
    else:
        gepa_kwargs["auto"] = auto
        print(f"[GEPA] budget: auto={auto}")

    teleprompter = GEPA(**gepa_kwargs)
    compiled = teleprompter.compile(VARKModule(), trainset=trainset, valset=valset)
    compiled.save(model_path)
    print(f"✅ GEPA-optimized model saved to {model_path}")
    return compiled


def compile_and_save(model_path: str = "vark_model.json", api_key: Optional[str] = None):
    configure_lm(api_key)

    from dspy.teleprompt import BootstrapFewShot

    trainset = get_trainset()

    # trainset order: V V K R K R A A (8 examples)
    # ก่อนหน้านี้ใช้ max_bootstrapped_demos=4 → หยุดที่ index 0–3 (V,V,K,R)
    # ทำให้ A ไม่เคยถูก bootstrap เลย — runtime จึงไม่มี demo สไตล์ A
    # ปรับเป็น 8 เพื่อครอบคลุมทุก style
    #
    # max_labeled_demos=0 — get_trainset() คืน example ที่มีแค่ input (ไม่มี
    # learning_material/youtube_queries) จึงใช้เป็น labeled demo ไม่ได้
    #
    # metric=vark_metric (LLM-as-Judge)
    #   Judge A: Gemini 3.1 flash lite  via Google AI Studio (GEMINI_API_KEY)
    #   Judge B: GPT-OSS                via OpenRouter        (GPTOSS_API_KEY)
    # ทั้ง 2 keys โหลดจาก .env อัตโนมัติ — ถ้าหาย key ใด key หนึ่งจะ fallback เป็น F1
    # metric_threshold=0.5 → เฉลี่ย 2 judges ต้อง ≥ 0.5 (≈ ทั้งคู่ให้อย่างน้อย 1)
    # หมายเหตุ: compile รอบนี้จะเรียก judge API หลายครั้ง (~ 2 × ตัวอย่าง × รอบ)
    teleprompter = BootstrapFewShot(
        metric=vark_metric,
        metric_threshold=0.5,
        max_bootstrapped_demos=8,
        max_labeled_demos=0,
        max_rounds=2,
    )
    compiled = teleprompter.compile(VARKModule(), trainset=trainset)

    compiled.save(model_path)
    print(f"✅ Compiled model saved to {model_path}")
    return compiled


def compile_quiz_and_save(
    model_path: str = "quiz_model.json",
    api_key: Optional[str] = None,
    trainset_path: str = "dataset/train/quiz_train.json",
):
    """Compile QuizModule ด้วย BootstrapFewShot + quiz_metric (LLM-as-Judge)"""
    configure_lm(api_key)
    from dspy.teleprompt import BootstrapFewShot

    trainset = get_quiz_trainset(trainset_path)
    print(f"📚 Quiz trainset: {len(trainset)} examples")

    teleprompter = BootstrapFewShot(
        metric=quiz_metric,
        metric_threshold=0.5,
        max_bootstrapped_demos=min(8, len(trainset)),
        max_labeled_demos=0,
        max_rounds=2,
    )
    compiled = teleprompter.compile(QuizModule(), trainset=trainset)
    compiled.save(model_path)
    print(f"✅ Quiz model saved to {model_path}")
    return compiled


def compile_video_and_save(
    model_path: str = "video_classifier_model.json",
    api_key: Optional[str] = None,
    trainset_path: str = "dataset/train/video_train.json",
):
    """Compile VideoClassifierModule ด้วย BootstrapFewShot + video_metric (LLM-as-Judge)"""
    configure_lm(api_key)
    from dspy.teleprompt import BootstrapFewShot

    trainset = get_video_trainset(trainset_path)
    print(f"📚 Video trainset: {len(trainset)} examples")

    teleprompter = BootstrapFewShot(
        metric=video_metric,
        metric_threshold=0.5,
        max_bootstrapped_demos=min(8, len(trainset)),
        max_labeled_demos=0,
        max_rounds=2,
    )
    compiled = teleprompter.compile(VideoClassifierModule(), trainset=trainset)
    compiled.save(model_path)
    print(f"✅ Video classifier model saved to {model_path}")
    return compiled


# ──────────────────────────────────────────────
# 7. Load for runtime
# ──────────────────────────────────────────────
def load_module(model_path: str = "vark_model.json") -> VARKModule:
    """
    Load compiled demos จาก vark_model.json เข้า VARKModule (2-stage)

    Strategy:
      1) ใช้ DSPy native module.load() ก่อน — handle JSON layout ทุก version
         และ inject demos เข้าทั้ง analyze + generate ให้อัตโนมัติ
      2) Fallback: parse JSON เองและ inject แยกแต่ละ stage
         (รองรับไฟล์เก่าที่ compile ด้วย VARKModuleLite ซึ่งเก็บ demos
          ใต้ key 'generate' เท่านั้น)
    """
    module = VARKModule()

    if not os.path.exists(model_path):
        print("⚠️  No compiled model found — using zero-shot module")
        return module

    # ── Path 1: DSPy native loader ─────────────────────────────
    try:
        module.load(model_path)
        a = len(getattr(module.analyze.predict, "demos", []) or [])
        g = len(getattr(module.generate.predict, "demos", []) or [])
        if a or g:
            print(f"✅ Loaded VARKModule (analyze: {a} demos, generate: {g} demos)")
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

        analyze_raw  = _extract("analyze")
        generate_raw = _extract("generate")

        loaded_a = _inject_demos(module.analyze,  analyze_raw)  if analyze_raw  else False
        loaded_g = _inject_demos(module.generate, generate_raw) if generate_raw else False

        if loaded_a or loaded_g:
            print(f"✅ Manual injection — analyze: {len(analyze_raw) if loaded_a else 0} demos, "
                  f"generate: {len(generate_raw) if loaded_g else 0} demos")
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


# ──────────────────────────────────────────────
# 8. Entrypoint
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DSPy pipeline: enrich → compile → evaluate (vark / quiz / video)"
    )
    parser.add_argument(
        "--target", type=str, default="vark",
        choices=["vark", "quiz", "video", "all"],
        help="which module to operate on"
    )
    parser.add_argument("--api-key", type=str, default=None,
                        help="Typhoon API Key (else read TYPHOON_API_KEY from env)")
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
    parser.add_argument("--skip-compile", action="store_true",
                        help="skip compile; load existing model.json — pair with --eval / --eval-blind")
    parser.add_argument("--optimizer", type=str, default="bootstrap",
                        choices=["bootstrap", "gepa"],
                        help="vark compile optimizer (default: bootstrap)")
    parser.add_argument("--gepa-budget", type=str, default="light",
                        choices=["light", "medium", "heavy"],
                        help="GEPA budget level (only when --optimizer gepa)")
    parser.add_argument("--gepa-max-calls", type=int, default=None,
                        help="GEPA explicit budget (overrides --gepa-budget). "
                             "เช่น 30 = เร็วมาก, 60 = ปานกลาง")
    args = parser.parse_args()

    do_vark  = args.target in ("vark", "all")
    do_quiz  = args.target in ("quiz", "all")
    do_video = args.target in ("video", "all")

    # ─ model paths — --output only applies เมื่อ target เป็น single ─
    vark_path  = (args.output if args.target == "vark"  else None) or "vark_model.json"
    quiz_path  = (args.output if args.target == "quiz"  else None) or "quiz_model.json"
    video_path = (args.output if args.target == "video" else None) or "video_classifier_model.json"

    # ── 1. Enrich (video only) ─────────────────────────────────
    if args.enrich:
        if not do_video:
            print("⚠️  --enrich ใช้ได้กับ target video/all เท่านั้น — ข้าม")
        elif not os.environ.get("YOUTUBE_API_KEY"):
            print("❌ YOUTUBE_API_KEY required for --enrich")
            sys.exit(1)
        else:
            import enrich_video_dataset as evd
            for p in (
                "dataset/train/video_train.json",
                "dataset/val/video_val.json",
                "dataset/test/video_test.json",
            ):
                evd.enrich_file(p)

    # ── 2. Compile ──────────────────────────────────────────────
    vark_module = quiz_module_obj = video_module_obj = None
    if not args.skip_compile:
        if do_vark:
            if args.optimizer == "gepa":
                vark_module = compile_with_gepa(
                    model_path=vark_path,
                    api_key=args.api_key,
                    auto=args.gepa_budget if args.gepa_max_calls is None else None,
                    max_metric_calls=args.gepa_max_calls,
                )
            else:
                vark_module = compile_and_save(model_path=vark_path, api_key=args.api_key)
        if do_quiz:
            quiz_module_obj = compile_quiz_and_save(model_path=quiz_path, api_key=args.api_key)
        if do_video:
            video_module_obj = compile_video_and_save(model_path=video_path, api_key=args.api_key)

    # ── 3. Evaluate testset ─────────────────────────────────────
    eval_modes: list[bool] = []  # blind flag per pass
    if args.eval:
        eval_modes.append(False)         # ref-augmented
    if args.eval_blind:
        eval_modes.append(True)          # blind

    if eval_modes:
        # ตอน skip-compile + eval ยังไม่ได้ configure LM — เรียกเอง
        if args.skip_compile:
            configure_lm(args.api_key)
        if do_vark:
            mod = vark_module or load_module(vark_path)
            ts = get_testset()
            for blind in eval_modes:
                evaluate_testset(mod, ts, blind=blind)
        if do_quiz:
            mod = quiz_module_obj or load_quiz_module(quiz_path)
            ts = get_quiz_testset()
            for blind in eval_modes:
                evaluate_quiz_testset(mod, ts, blind=blind)
        if do_video:
            mod = video_module_obj or load_classifier_module(video_path)
            ts = get_video_testset()
            for blind in eval_modes:
                evaluate_video_testset(mod, ts, blind=blind)