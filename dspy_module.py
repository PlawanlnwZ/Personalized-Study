"""
dspy_module.py — AI Logic Layer (Enhanced)
DSPy + Typhoon สำหรับสร้างสื่อการเรียนรู้แบบ VARK
ปรับปรุง: Chain of Thought + Multi-stage Reasoning + Self-Refinement
"""

import dspy
import json
import os
from typing import Optional


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
        max_tokens=16000,
        temperature=0.7,
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
        desc="แนวคิดหลัก 3-5 ข้อที่ควรสอน พร้อมระดับความสำคัญ (JSON array)"
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
            "ต้องเขียนให้ครบถ้วนและยาวเพียงพอ อย่างน้อย 800-1200 คำ ครอบคลุมเนื้อหาทั้งหมด "
            "โครงสร้าง: 1) อธิบายแนวคิดหลัก 2) ตัวอย่างและการประยุกต์ใช้ 3) แบบฝึกหัดหรือสรุป "
            "ปรับตาม VARK dominant อย่างเคร่งครัด — "
            "V=ต้องมีตาราง Markdown (|col|col|) และใช้คำ table/chart/diagram/visual/แผนภาพ/แสดง; "
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
            "**สำคัญ — รูปแบบเฉลยของแบบฝึกหัด:** ทุกครั้งที่มีโจทย์/คำถาม/แบบฝึกหัด ต้องใส่เฉลยใน fenced block "
            "ที่ใช้ภาษา 'เฉลย' เพื่อให้ frontend แสดงเป็นปุ่มกดดูเฉลย เช่น:\n"
            "```เฉลย\n"
            "<คำตอบและคำอธิบาย>\n"
            "```\n"
            "ห้ามเขียนเฉลยแบบเปิดเผย (เช่น '**เฉลย:** ...' ในย่อหน้าธรรมดา) — ต้องอยู่ใน ```เฉลย ... ``` เสมอ"
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
            "JSON array เท่านั้น สร้างเฉพาะ dominant=V "
            'เช่น [{"q":"if-else flowchart","imgType":"photo","imgSize":"large","rights":"cc_publicdomain","altDescription":"แผนผัง"}] '
            "ถ้าไม่ใช่ V ตอบ: []"
        )
    )


# ──────────────────────────────────────────────
# 2c. Signature Lite — ใช้ตอน compile/train เท่านั้น
# ──────────────────────────────────────────────
class VARKProjectorLite(dspy.Signature):
    """สร้างสื่อการเรียนรู้แบบ VARK จากเนื้อหาที่กำหนด"""
    context: str = dspy.InputField(desc="เนื้อหาจากเอกสาร")
    vark_style: str = dspy.InputField(desc='JSON เช่น {"V":60,"dominant":"V"}')
    learning_material: str = dspy.OutputField(
        desc=(
            "สื่อการเรียนรู้ Markdown อย่างน้อย 500-700 คำ ปรับตาม VARK dominant อย่างเคร่งครัด. "
            "ภาษา: ใช้ภาษาเดียวกับ context — ถ้า context เป็นไทย ต้องตอบภาษาไทย ห้ามสลับเป็นอังกฤษ "
            "(ยกเว้นศัพท์เฉพาะ/ตัวอย่างประโยคที่อยู่ในเนื้อหาเดิม): "
            "V=ต้องมีตาราง Markdown (|col|col|) และใช้คำ table/chart/diagram/visual/แผนภาพ/แสดง; "
            "A=ต้องเขียนในรูป story (เล่าเรื่อง) มี narrator บรรยาย แล้วแทรก dialogue ของ Aural เป็นจุดๆ "
            "ใช้คำ story/imagine/dialogue/เล่า/สนทนา. "
            "**โครงสร้าง A:** ส่วนใหญ่ (>=70%) เป็นย่อหน้าบรรยาย/อธิบายธรรมดา (ไม่มี `>`); "
            "ใช้ blockquote เฉพาะ Situation+dialogue: ขึ้นต้น `> Situation: <ฉาก>` "
            "แล้วบรรทัดถัดไป `> ชื่อคน: \"...\"` เช่น `> ครู: \"...\"`, `> นักเรียน: \"...\"`. "
            "**จำนวนบล็อกตามค่า A:** A>=70 ใส่ 3–5 บล็อก, A 40–69 ใส่ 2–3, "
            "A 20–39 ใส่แค่ 1–2 บล็อก, A<20 ใส่ 0–1 บล็อก. ห้ามต่อกันเกิน 2. "
            "ห้ามใช้ `>` กับ heading, คำอธิบาย, note, สรุป, ตัวอย่างประโยค, bullet — "
            "ใช้ได้เฉพาะ Situation+dialogue เท่านั้น; "
            "R=อธิบายเป็นข้อความ format ดี (heading, bullet, นิยาม, ตาราง) ครบถ้วน "
            "ห้ามสั่งให้ผู้เรียน 'สรุปด้วยภาษาตัวเอง' — ให้แนวทางการอ่าน/จดโน้ตแทน "
            "ใช้คำ definition/outline/note/นิยาม/หลักการ/แนวทาง/ดังนี้; "
            "K=ต้องมี step-by-step และ exercise ให้ผู้เรียนลงมือทำ "
            "และใช้คำ try/fill in/match/complete/apply/exercise/practice/ขั้นตอน/ลองทำ/จับคู่/เติม/ฝึก. "
            "ทุกแบบฝึกหัด/โจทย์ ต้องใส่เฉลยใน fenced block ภาษา 'เฉลย' เช่น ```เฉลย\\n<คำตอบ>\\n``` "
            "เพื่อให้ frontend แสดงเป็นปุ่มกดดูเฉลย"
        )
    )
    youtube_queries: str = dspy.OutputField(
        desc='JSON array เท่านั้น 3 queries เช่น ["topic tutorial", "หัวข้อ ภาษาไทย"]'
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
        # ── Stage 1: วิเคราะห์และวางแผน ──
        analysis = self.analyze(context=context, vark_style=vark_style)

        # ── Stage 2: สร้างสื่อโดยอิงจากแผน ──
        draft = self.generate(
            context=context,
            vark_style=vark_style,
            key_concepts=analysis.key_concepts,
            teaching_strategy=analysis.teaching_strategy,
        )

        return dspy.Prediction(
            learning_material=draft.learning_material,
            youtube_queries=draft.youtube_queries,
            image_queries=draft.image_queries,
            # metadata จาก reasoning chain (ใช้ debug/logging)
            key_concepts=analysis.key_concepts,
            teaching_strategy=analysis.teaching_strategy,
        )


class VARKModuleLite(dspy.Module):
    """
    Compile/train module — ChainOfThought เดี่ยว ประหยัด token
    ใช้ CoT แทน Predict เพื่อให้ตัวอย่างที่ bootstrap มีคุณภาพสูงขึ้น
    """
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(VARKProjectorLite)

    def forward(self, context: str, vark_style: str) -> dspy.Prediction:
        return self.generate(context=context, vark_style=vark_style)


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
    """จัดประเภท YouTube video ตามสไตล์ VARK + จับคู่กับหน้า PDF"""
    video_title: str       = dspy.InputField(desc="ชื่อคลิป")
    video_channel: str     = dspy.InputField(desc="ชื่อช่อง")
    video_metadata: str    = dspy.InputField(
        desc="description / tags / categories ของคลิป (รวมเป็น text)"
    )
    page_snippets: str     = dspy.InputField(
        desc='ข้อความตัวอย่างจาก PDF แต่ละหน้า เช่น "p1: ..." "p2: ..."'
    )
    vark_weight_desc: str  = dspy.InputField(
        desc='โปรไฟล์ VARK ของผู้เรียน เช่น "V 60%, K 30%, R 10%"'
    )
    classification: str    = dspy.OutputField(
        desc=(
            "ตอบ JSON เท่านั้น: "
            '{"vark":["V","K"],"subtopics":["..","..."],"pages_covered":[1,2]}. '
            "vark = 1–4 styles ที่ตรงกับเนื้อหาคลิปจริง "
            "(V=diagram/animation, A=lecture/talk, R=text/code, K=hands-on/demo). "
            "subtopics = 2–4 keywords ≤16 chars (ไทยถ้าคลิปเป็นไทย). "
            "pages_covered = หมายเลขหน้า PDF ที่ตรงกับคลิป (max 8) — "
            "ถ้าไม่ตรงให้ตอบ []"
        )
    )


class VideoClassifierModule(dspy.Module):
    """Runtime module สำหรับ classify video — output สั้น ใช้ Predict ก็พอ"""
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(VideoVARKClassifier)

    def forward(self, **kwargs) -> dspy.Prediction:
        return self.classify(**kwargs)


# ──────────────────────────────────────────────
# 4. VARK Metric (Classification Metrics — Precision / Recall / F1)
# ──────────────────────────────────────────────
#
# แนวคิด (AI Builders rubric):
#   สร้าง binary classification ต่อ token ว่า "ใช่ dominant style" หรือเปล่า
#
#   TP = keyword ของ dominant style ที่ปรากฏใน output   (ถูกสไตล์)
#   FP = keyword ที่เป็น exclusive ของ style อื่น ที่ปรากฏ (เบี่ยงสไตล์)
#   FN = keyword dominant ที่ขาดหายไปจาก output         (พลาด)
#
#   หลักการสำคัญ:
#   1. FP นับเฉพาะ "exclusive keywords" — คือ keyword ที่ไม่ได้ใช้ร่วมกันข้ามสไตล์
#   2. Recall rate ถ่วงน้ำหนักตาม keyword ที่ unique ต่อ style นั้นจริงๆ
#
#   Precision = TP / (TP + FP)  → output "ตรงสไตล์" แค่ไหน
#   Recall    = TP / (TP + FN)  → ครอบคลุม keyword dominant ได้แค่ไหน
#   F1        = 2 * P * R / (P + R)  → สมดุลระหว่างสองตัวข้างต้น
#
# คะแนนสุดท้าย (0.0–1.0):
#   F1-VARK   × 0.6   (ตรงสไตล์ VARK — น้ำหนักสูงสุด)
#   Quality   × 0.2   (โครงสร้าง Markdown ครบ)
#   Length    × 0.1   (ความยาวเพียงพอ)
#   YouTube   × 0.1   (มี queries ครบ)
# ──────────────────────────────────────────────

VARK_KEYWORDS = {
    "V": ["ตาราง", "แผนภาพ", "diagram", "table", "chart", "graph", "visual",
          "รูปภาพ", "แสดง", "|"],
    "A": ["ฟัง", "พูด", "imagine", "story", "เล่า",
          "dialogue", "listen", "discuss", "สนทนา"],
    "R": ["ดังนี้", "นิยาม", "definition", "outline", "สรุป", "note",
          "เขียน", "อ่าน"],
    "K": ["ลองทำ", "exercise", "quiz", "ทดสอบ", "step",
          "ขั้นตอน", "practice", "real-world", "โจทย์",
          "try", "fill in", "match", "complete", "apply",
          "จับคู่", "เติม", "ฝึก", "กิจกรรม", "scenario"],
}

QUALITY_INDICATORS = [
    "##",       # มี heading (โครงสร้างชัด)
    "```",      # มี code block
    "**",       # มี bold (เน้นคำสำคัญ)
    "\n\n",     # มี paragraph break (ไม่แน่นเกิน)
]

_EXCLUSIVE_KEYWORDS: dict[str, set[str]] = {}

def _build_exclusive_keywords() -> dict[str, set[str]]:
    """คำนวณ exclusive keywords ครั้งเดียวตอน import"""
    all_kws: dict[str, set[str]] = {s: set(kws) for s, kws in VARK_KEYWORDS.items()}
    exclusive: dict[str, set[str]] = {}
    for style, kws in all_kws.items():
        other = set().union(*(v for s, v in all_kws.items() if s != style))
        exclusive[style] = kws - other
    return exclusive

_EXCLUSIVE_KEYWORDS = _build_exclusive_keywords()


def _precision_recall_f1(material: str, vark_dist: dict) -> tuple[float, float, float]:
    """
    คำนวณ Precision, Recall, F1 ที่ถ่วงน้ำหนักตาม VARK distribution เต็มรูปแบบ
    ไม่ใช่แค่ dominant — secondary styles ก็ทำคะแนนได้ตามสัดส่วนใน vark_style

    หาก target = {V:50, A:10, R:20, K:20}:
      - V keywords → TP น้ำหนัก 0.5
      - R keywords → TP น้ำหนัก 0.2
      - K keywords → TP น้ำหนัก 0.2  (ไม่โดนลงโทษเหมือน metric เดิมที่นับเป็น FP)
      - A keywords → TP น้ำหนัก 0.1
      - keyword จาก style ที่ target = 0 เท่านั้น ที่จะนับเป็น FP
    """
    total = sum(vark_dist.get(s, 0) for s in "VARK")
    if total == 0:
        return 0.0, 0.0, 0.0
    weights = {s: vark_dist.get(s, 0) / total for s in "VARK"}

    tp = 0.0
    fn = 0.0
    for style, kws in VARK_KEYWORDS.items():
        w = weights[style]
        if w == 0:
            continue  # style นี้ไม่อยู่ในโปรไฟล์ — handle ใน FP loop ด้านล่าง
        hits = sum(1 for kw in kws if kw.lower() in material)
        tp += w * hits
        fn += w * (len(kws) - hits)

    # FP — เฉพาะ exclusive keyword ของ style ที่ target = 0
    fp = 0.0
    for style, kws in _EXCLUSIVE_KEYWORDS.items():
        if weights[style] == 0:
            fp += sum(1 for kw in kws if kw.lower() in material)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return round(precision, 4), round(recall, 4), round(f1, 4)


def vark_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """
    คะแนน 0.0–1.0 ประเมินคุณภาพ output ของ VARKModule

    breakdown:
      F1-VARK  × 0.6  — วัดด้วย Precision/Recall/F1 จาก keyword classification
      Quality  × 0.2  — โครงสร้าง Markdown (heading, bold, paragraph)
      Length   × 0.1  — ความยาวเนื้อหาเพียงพอ (>= 300 words)
      YouTube  × 0.1  — มี YouTube queries >= 3 รายการ
    """
    try:
        vark = json.loads(example.vark_style)
        dominant = vark.get("dominant", "R")
    except Exception:
        vark = {"V": 0, "A": 0, "R": 100, "K": 0, "dominant": "R"}
        dominant = "R"

    material = (prediction.learning_material or "").lower()

    # ── F1-VARK score (0.0–0.6) ──
    precision, recall, f1 = _precision_recall_f1(material, vark)
    vark_score = f1 * 0.6

    # ── คะแนนคุณภาพโครงสร้าง (0.0–0.2) ──
    quality_hits = sum(1 for ind in QUALITY_INDICATORS if ind in material)
    quality_score = min(quality_hits / len(QUALITY_INDICATORS), 1.0) * 0.2

    # ── คะแนนความยาว (0.0–0.1) ──
    word_count = len(material.split())
    length_score = 0.1 if word_count >= 400 else (word_count / 400) * 0.1

    # ── คะแนน YouTube queries (0.0–0.1) ──
    yt_score = 0.0
    try:
        yt = json.loads(prediction.youtube_queries)
        if isinstance(yt, list) and len(yt) >= 3:
            yt_score = 0.1
    except Exception:
        pass

    total = vark_score + quality_score + length_score + yt_score

    if trace:
        print(
            f"[vark_metric] dominant={dominant} | "
            f"P={precision:.3f} R={recall:.3f} F1={f1:.3f} | "
            f"quality={quality_score:.2f} length={length_score:.2f} yt={yt_score:.2f} | "
            f"total={round(min(total,1.0),4)}"
        )

    return round(min(total, 1.0), 4)


def evaluate_testset(module, testset: list) -> dict:
    """
    รัน vark_metric บน testset ทั้งหมด แล้วรายงาน
    Precision, Recall, F1, Semantic Similarity เฉลี่ย และ Accuracy (dominant ถูกต้อง)

    Usage:
        module = load_module("vark_model.json")
        testset = load_split("test")
        report = evaluate_testset(module, testset)
        print(report)
    """
    precisions, recalls, f1s = [], [], []
    correct_dominant = 0

    for ex in testset:
        pred = module(context=ex.context, vark_style=ex.vark_style)
        material = (pred.learning_material or "").lower()

        try:
            vark = json.loads(ex.vark_style)
            dominant = vark.get("dominant", "R")
        except Exception:
            vark = {"V": 0, "A": 0, "R": 100, "K": 0, "dominant": "R"}
            dominant = "R"

        p, r, f = _precision_recall_f1(material, vark)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

        # Accuracy — นับว่า output มี keyword dominant มากกว่า style อื่นไหม
        best_style = max(
            VARK_KEYWORDS,
            key=lambda s: sum(1 for kw in VARK_KEYWORDS[s] if kw.lower() in material)
        )
        if best_style == dominant:
            correct_dominant += 1

    n = len(testset)
    report = {
        "n": n,
        "precision_avg": round(sum(precisions) / n, 4),
        "recall_avg":    round(sum(recalls) / n, 4),
        "f1_avg":        round(sum(f1s) / n, 4),
        "accuracy":      round(correct_dominant / n, 4),
    }
    print("\n📊 Evaluation Report")
    print(f"   Samples   : {report['n']}")
    print(f"   Precision : {report['precision_avg']}")
    print(f"   Recall    : {report['recall_avg']}")
    print(f"   F1        : {report['f1_avg']}")
    print(f"   Accuracy  : {report['accuracy']}")
    return report


# ──────────────────────────────────────────────
# 5. Training Dataset (ขยายเพิ่ม)
# ──────────────────────────────────────────────


def get_trainset() -> list[dspy.Example]:
    with open("dataset/train/vark_train.json", encoding="utf-8") as f:
        rows = json.load(f)
    return [
        dspy.Example(
            context=r["context"],
            vark_style=json.dumps(r["vark_style"], ensure_ascii=False),
        ).with_inputs("context", "vark_style")
        for r in rows
    ]

# ──────────────────────────────────────────────
# 6. Compile (ใช้ VARKModuleLite + BootstrapFewShot)
# ──────────────────────────────────────────────
def compile_and_save(model_path: str = "vark_model.json", api_key: Optional[str] = None):
    configure_lm(api_key)

    from dspy.teleprompt import BootstrapFewShot

    trainset = get_trainset()

    # max_bootstrapped_demos + max_labeled_demos ควรรวมกันไม่เกิน len(trainset)
    # trainset ปัจจุบันมี 8 examples (V×2, A×2, R×2, K×2)
    # BootstrapFewShot จะวน trainset จนครบ max_bootstrapped_demos
    teleprompter = BootstrapFewShot(
        metric=vark_metric,
        max_bootstrapped_demos=4,   # 4 augmented + 4 labeled = 8 demo slots ตรงกับ trainset
        max_labeled_demos=4,        # 1 labeled ต่อ dominant style
        max_rounds=2,               # วน 2 รอบ เผื่อ example บางตัวไม่ผ่าน metric รอบแรก
    )
    compiled = teleprompter.compile(VARKModuleLite(), trainset=trainset)

    compiled.save(model_path)
    print(f"✅ Compiled model saved to {model_path}")
    return compiled


# ──────────────────────────────────────────────
# 7. Load for runtime
# ──────────────────────────────────────────────
def load_module(model_path: str = "vark_model.json") -> VARKModule:
    """
    Load compiled demos จาก vark_model.json เข้าสู่ VARKModule

    DSPy บันทึก JSON ต่างกันตามเวอร์ชัน:
      - เก่า: { "generate": { "demos": [...] } }
      - ใหม่: { "generate.predict": { "demos": [...] } }  หรือ
              { "generate.predict.demos": [...] }
    ฟังก์ชันนี้ลองทุก pattern จนเจอ demos
    """
    module = VARKModule()

    if not os.path.exists(model_path):
        print("⚠️  No compiled model found — using zero-shot module")
        return module

    try:
        with open(model_path, "r", encoding="utf-8") as f:
            state = json.load(f)

        # DEBUG: แสดง top-level keys ช่วย diagnose ครั้งแรก
        print(f"[load_module] JSON keys: {list(state.keys())[:10]}")

        # ลองหา demos จากทุก pattern ที่ DSPy เคย/อาจใช้
        demos_raw = []

        # Pattern A — DSPy เก่า: { "generate": { "demos": [...] } }
        if not demos_raw:
            demos_raw = (state.get("generate") or {}).get("demos", [])

        # Pattern B — DSPy ใหม่: { "generate.predict": { "demos": [...] } }
        if not demos_raw:
            demos_raw = (state.get("generate.predict") or {}).get("demos", [])

        # Pattern C — flat key: { "generate.predict.demos": [...] }
        if not demos_raw:
            demos_raw = state.get("generate.predict.demos", [])

        # Pattern D — VARKModuleLite ใหม่ใช้ ChainOfThought key ต่างออกไป
        if not demos_raw:
            for key in state:
                val = state[key]
                if isinstance(val, dict) and "demos" in val and val["demos"]:
                    demos_raw = val["demos"]
                    print(f"[load_module] Found demos under key: '{key}'")
                    break
                elif isinstance(val, list) and val and isinstance(val[0], dict):
                    # demos อาจเป็น list โดยตรงใต้บาง key
                    if "augmented" in str(val[0]) or "context" in str(val[0]):
                        demos_raw = val
                        print(f"[load_module] Found demos list under key: '{key}'")
                        break

        if demos_raw:
            demos = [dspy.Example(**d) if not isinstance(d, dspy.Example) else d
                     for d in demos_raw]
            # inject เข้า generate stage (ลองทั้ง .predict.demos และ .demos)
            try:
                module.generate.predict.demos = demos
                print(f"✅ Loaded {len(demos)} demos → generate.predict.demos")
            except AttributeError:
                try:
                    module.generate.demos = demos
                    print(f"✅ Loaded {len(demos)} demos → generate.demos")
                except AttributeError:
                    print("⚠️  Cannot inject demos — attribute path not found")
        else:
            print("⚠️  No demos found in any known pattern — using zero-shot")
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

    parser = argparse.ArgumentParser(description="Compile VARKModule with DSPy")
    parser.add_argument("--api-key", type=str, default=None, help="Typhoon API Key")
    parser.add_argument("--output",  type=str, default="vark_model.json")
    args = parser.parse_args()

    compile_and_save(model_path=args.output, api_key=args.api_key)