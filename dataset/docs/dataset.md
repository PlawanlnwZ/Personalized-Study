# VARK DATASET — วิชาภาษาอังกฤษ (Manual Only)

======================
# Quick start -> python dspy_module.py --api-key TYPHOON_API_KEY --output vark_model.json

A **manual-only** labeled dataset for training and evaluating `VARKModule` (DSPy + Typhoon) — 
a system that generates VARK-adapted learning materials for English grammar concepts.

> **Policy:** ห้ามใช้ synthetic / SMOTE / LLM-generated examples ทุก entry ต้องเขียนมือ
> เพื่อ honest evaluation และคุณภาพ label ที่ตรวจสอบได้จริง

---

## Directory Structure

```
dataset/
├── TEMPLATE.md              # แม่แบบสำหรับเพิ่ม entry ใหม่ (เขียนมือ)
├── train/
│   └── vark_train.json      # 5 manual examples (V/A/R/K mixed)
├── val/
│   └── vark_val.json        # ว่าง — เพิ่ม manual entries เอง
├── test/
│   └── vark_test.json       # 3 manual examples
├── raw/
│   └── sources.md           # Provenance tracking
└── docs/
    ├── dataset.md           ← you are here
    └── quality_checklist.md # AI Builders rubric checks
```

---

## Schema

Each JSON file is an array of objects with the following fields:

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique identifier e.g. `train_001` |
| `context` | string | Thai/English grammar teaching content (input to model) |
| `vark_style` | object | VARK scores V/A/R/K (0–100, sum=100) + `dominant` key |
| `source` | string | `"manual"` เสมอ — ไม่มีค่าอื่น |
| `split` | string | `train` / `val` / `test` |
| `notes` | string | คำอธิบายความคาดหวัง output (1 ประโยค) |

### VARK Style Example
```json
{
  "V": 60, "A": 10, "R": 20, "K": 10, "dominant": "V"
}
```
- **V** (Visual): expects Markdown tables, diagrams, structured layout  
- **A** (Aural): expects storytelling, dialogue, narrative explanation  
- **R** (Read/Write): expects outline, formal definitions, bullet lists  
- **K** (Kinesthetic): expects step-by-step walkthroughs, mini quizzes  

---

## วิธีเพิ่ม Entry ใหม่

อ่าน `dataset/TEMPLATE.md` — มี block template + checklist ครบ

หลักการสั้น ๆ:
1. เขียนมือเท่านั้น (ห้าม LLM generate)
2. V+A+R+K รวม = 100
3. `dominant` = style ที่ score สูงสุด
4. ห้ามหัวข้อซ้ำข้าม train/val/test (zero leakage)

---

## Topics per Split

| Split | Topics |
|---|---|
| train | Present Simple, Past Continuous, Conditional Type 1, Passive Voice, Reported Speech |
| val   | (ว่าง — รอเพิ่ม) |
| test  | Question Tags, Gerunds & Infinitives, Prepositions of Time |

---

## Split Summary

| Split | Count | Source |
|---|---|---|
| train | 5 | manual |
| val   | 0 | manual (empty — to fill) |
| test  | 3 | manual |

> **AI Builders Rule (strict):** ทุก split ต้อง manual เท่านั้น ไม่มี synthetic ใด ๆ

---

## How to Load

```python
import json, dspy

def load_split(split: str) -> list[dspy.Example]:
    with open(f"dataset/{split}/vark_{split}.json", encoding="utf-8") as f:
        rows = json.load(f)
    return [
        dspy.Example(
            context=r["context"],
            vark_style=json.dumps(r["vark_style"], ensure_ascii=False),
        ).with_inputs("context", "vark_style")
        for r in rows
    ]

trainset = load_split("train")   # 5 examples
valset   = load_split("val")     # 0 examples (empty)
testset  = load_split("test")    # 3 examples
```

---

## License

Content: **CC-BY-SA 4.0** — you may use, remix, and share with attribution.
