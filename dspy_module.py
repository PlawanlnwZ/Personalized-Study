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
            "V=เน้นภาพและโครงสร้าง, A=เน้นการเล่าเรื่อง, "
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
            "ต้องเขียนให้ครบถ้วนและยาวเพียงพอ อย่างน้อย 800-1200 คำ ครอบคลุมเนื้อหาทั้งหมด "
            "โครงสร้าง: 1) อธิบายแนวคิดหลัก 2) ตัวอย่างและการประยุกต์ใช้ 3) แบบฝึกหัดหรือสรุป "
            "Visual=ตาราง Markdown/diagram; Aural=storytelling/dialogue; "
            "Read/Write=outline/definition; Kinesthetic=step-by-step/mini quiz"
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
# 2c. Signature: Self-Refinement (Stage 3)
#     AI ตรวจสอบงานตัวเองและปรับปรุงจุดอ่อน
# ──────────────────────────────────────────────
class VARKRefiner(dspy.Signature):
    """
    ตรวจสอบคุณภาพของสื่อการเรียนรู้ที่สร้างแล้ว
    ระบุจุดอ่อนและปรับปรุงให้ตรงกับสไตล์ VARK มากขึ้น
    """
    vark_style: str = dspy.InputField(desc="JSON โปรไฟล์ VARK ของผู้เรียน")
    draft_material: str = dspy.InputField(desc="สื่อการเรียนรู้ draft ที่สร้างจาก Stage 2")
    quality_issues: str = dspy.OutputField(
        desc="จุดอ่อน 2-3 ข้อที่พบใน draft (JSON array of strings)"
    )
    refined_material: str = dspy.OutputField(
        desc=(
            "สื่อการเรียนรู้ที่ปรับปรุงแล้ว Markdown ครบถ้วน 800-1200 คำ "
            "แก้ไขจุดอ่อนที่ระบุ และเพิ่มความสอดคล้องกับสไตล์ VARK"
        )
    )


# ──────────────────────────────────────────────
# 2d. Signature Lite — ใช้ตอน compile/train เท่านั้น
# ──────────────────────────────────────────────
class VARKProjectorLite(dspy.Signature):
    """สร้างสื่อการเรียนรู้แบบ VARK จากเนื้อหาที่กำหนด"""
    context: str = dspy.InputField(desc="เนื้อหาจากเอกสาร")
    vark_style: str = dspy.InputField(desc='JSON เช่น {"V":60,"dominant":"V"}')
    learning_material: str = dspy.OutputField(
        desc=(
            "สื่อการเรียนรู้ Markdown 400-600 คำ ปรับตาม VARK dominant: "
            "V=ตาราง/แผนภาพ, A=เล่าเรื่อง/dialogue, R=outline/นิยาม, K=ขั้นตอน/quiz"
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
    Runtime module — ใช้ Chain of Thought + Multi-stage reasoning
    
    Pipeline:
      Stage 1: ChainOfThought วิเคราะห์เนื้อหาและวางแผนการสอน
      Stage 2: ChainOfThought สร้างสื่อโดยใช้แผนจาก Stage 1
      Stage 3: Predict ตรวจสอบและ refine output (self-refinement)
    
    ทำไม ChainOfThought ถึงดีกว่า Predict:
    - บังคับให้ model "คิดก่อนตอบ" ผ่าน rationale field
    - ลด hallucination เพราะ reasoning มีความสอดคล้องภายใน
    - ผลลัพธ์แม่นยำขึ้นโดยเฉพาะกับ structured output (JSON)
    """
    def __init__(self):
        super().__init__()
        # Stage 1: วิเคราะห์เนื้อหา + วางแผน (ใช้ CoT เพื่อให้ model คิดอย่างเป็นระบบ)
        self.analyze = dspy.ChainOfThought(ContentAnalyzer)
        # Stage 2: สร้างสื่อจากแผนที่วางไว้ (CoT ช่วยให้ output สอดคล้องกับ strategy)
        self.generate = dspy.ChainOfThought(VARKProjector)
        # Stage 3: Self-refinement (Predict พอ เพราะรับ draft มาแล้ว ไม่ต้อง reason ใหม่)
        self.refine = dspy.Predict(VARKRefiner)

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

        # ── Stage 3: Self-refinement ──
        refined = self.refine(
            vark_style=vark_style,
            draft_material=draft.learning_material,
        )

        # คืนค่า Prediction ที่รวม output ทุก stage
        return dspy.Prediction(
            # ใช้ refined_material เป็น output หลัก
            learning_material=refined.refined_material,
            youtube_queries=draft.youtube_queries,
            image_queries=draft.image_queries,
            # metadata จาก reasoning chain (ใช้ debug/logging)
            key_concepts=analysis.key_concepts,
            teaching_strategy=analysis.teaching_strategy,
            quality_issues=refined.quality_issues,
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
# 4. VARK Metric (ปรับปรุง — วัดละเอียดขึ้น)
# ──────────────────────────────────────────────
VARK_KEYWORDS = {
    "V": ["ตาราง", "แผนภาพ", "diagram", "table", "chart", "graph", "visual",
          "รูปภาพ", "แสดง", "##", "---", "|"],
    "A": ["ฟัง", "พูด", "อธิบาย", "เหมือน", "imagine", "story", "เล่า",
          "dialogue", "listen", "discuss", "สนทนา"],
    "R": ["ดังนี้", "นิยาม", "definition", "outline", "สรุป", "note",
          "เขียน", "อ่าน", "list", "bullet", "1.", "2.", "-"],
    "K": ["ลองทำ", "ตัวอย่าง", "exercise", "quiz", "ทดสอบ", "step",
          "ขั้นตอน", "practice", "real-world", "โจทย์"],
}

# คำที่บ่งบอกว่าเนื้อหาครอบคลุมและมีคุณภาพ
QUALITY_INDICATORS = [
    "##",       # มี heading (โครงสร้างชัด)
    "```",      # มี code block
    "**",       # มี bold (เน้นคำสำคัญ)
    "\n\n",     # มี paragraph break (ไม่แน่นเกิน)
]


def vark_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    try:
        vark = json.loads(example.vark_style)
        dominant = vark.get("dominant", "R")
    except Exception:
        dominant = "R"

    material = (prediction.learning_material or "").lower()

    # ── คะแนน VARK alignment (0.0-0.6) ──
    keywords = VARK_KEYWORDS.get(dominant, [])
    hits = sum(1 for kw in keywords if kw.lower() in material)
    vark_score = min(hits / max(len(keywords) * 0.4, 1), 1.0) * 0.6

    # ── คะแนนคุณภาพโครงสร้าง (0.0-0.2) ──
    quality_hits = sum(1 for ind in QUALITY_INDICATORS if ind in material)
    quality_score = min(quality_hits / len(QUALITY_INDICATORS), 1.0) * 0.2

    # ── คะแนนความยาว (0.0-0.1) — ไม่สั้นเกินไป ──
    word_count = len(material.split())
    length_score = 0.1 if word_count >= 300 else (word_count / 300) * 0.1

    # ── คะแนน YouTube queries (0.0-0.1) ──
    yt_score = 0.0
    try:
        yt = json.loads(prediction.youtube_queries)
        if isinstance(yt, list) and len(yt) >= 3:
            yt_score = 0.1
    except Exception:
        pass

    total = vark_score + quality_score + length_score + yt_score
    return round(min(total, 1.0), 4)


# ──────────────────────────────────────────────
# 5. Training Dataset (ขยายเพิ่ม)
# ──────────────────────────────────────────────
def get_trainset() -> list[dspy.Example]:
    return [
        dspy.Example(
            context=(
                "if-else เป็นโครงสร้างควบคุมในภาษา C "
                "ใช้สำหรับตรวจสอบเงื่อนไข ถ้าเงื่อนไขเป็นจริงจะทำคำสั่งในบล็อก if "
                "ถ้าเท็จจะทำคำสั่งในบล็อก else"
            ),
            vark_style='{"V":60,"A":10,"R":20,"K":10,"dominant":"V"}',
        ).with_inputs("context", "vark_style"),

        dspy.Example(
            context=(
                "Loop for ในภาษา C ใช้สำหรับทำซ้ำ "
                "มีส่วนประกอบ 3 ส่วน คือ init, condition, update "
                "เช่น for(int i=0; i<10; i++)"
            ),
            vark_style='{"V":10,"A":20,"R":60,"K":10,"dominant":"R"}',
        ).with_inputs("context", "vark_style"),

        dspy.Example(
            context=(
                "ฟังก์ชันในภาษา C คือชุดคำสั่งที่ทำงานซ้ำได้ "
                "ประกาศด้วย return_type function_name(params) "
                "เรียกใช้งานโดยระบุชื่อและ argument"
            ),
            vark_style='{"V":10,"A":10,"R":20,"K":60,"dominant":"K"}',
        ).with_inputs("context", "vark_style"),

        # ── ตัวอย่างเพิ่มเติม สไตล์ A ──
        dspy.Example(
            context=(
                "Array ในภาษา C คือการเก็บข้อมูลชนิดเดียวกันหลายค่าในตัวแปรเดียว "
                "ประกาศด้วย int arr[5]; และเข้าถึงด้วย index เริ่มต้นที่ 0"
            ),
            vark_style='{"V":10,"A":60,"R":20,"K":10,"dominant":"A"}',
        ).with_inputs("context", "vark_style"),

        # ── ตัวอย่างเพิ่มเติม mixed style ──
        dspy.Example(
            context=(
                "Pointer ในภาษา C คือตัวแปรที่เก็บ address ของตัวแปรอื่น "
                "ประกาศด้วย int *ptr; และใช้ & เพื่อดึง address, * เพื่อ dereference"
            ),
            vark_style='{"V":30,"A":10,"R":30,"K":30,"dominant":"K"}',
        ).with_inputs("context", "vark_style"),
    ]


# ──────────────────────────────────────────────
# 6. Compile (ใช้ VARKModuleLite + BootstrapFewShot)
# ──────────────────────────────────────────────
def compile_and_save(model_path: str = "vark_model.json", api_key: Optional[str] = None):
    configure_lm(api_key)

    from dspy.teleprompt import BootstrapFewShot

    # max_bootstrapped_demos=3 เพิ่มจาก 2 — ให้ตัวอย่างมากขึ้น
    teleprompter = BootstrapFewShot(
        metric=vark_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=2,        # ใช้ labeled demo เสริม bootstrapped
    )
    compiled = teleprompter.compile(VARKModuleLite(), trainset=get_trainset())

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
# 8. Entrypoint
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compile VARKModule with DSPy")
    parser.add_argument("--api-key", type=str, default=None, help="Typhoon API Key")
    parser.add_argument("--output",  type=str, default="vark_model.json")
    args = parser.parse_args()

    compile_and_save(model_path=args.output, api_key=args.api_key)