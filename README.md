# VARK Personalized Study System

ระบบสร้างสื่อการเรียนรู้แบบ AI ที่ปรับให้ตรงกับสไตล์ VARK ของผู้เรียน  
ใช้ **DSPy + Gemini 1.5 Flash** เป็น AI Layer, **FastAPI** เป็น Backend, และ HTML/JS เป็น Frontend

---

## โครงสร้างไฟล์

```
project/
├── dspy_module.py      # AI Logic: DSPy Signature, Module, Metric, Optimizer
├── main.py             # FastAPI Backend: /generate, /feedback, /recompile
├── requirements.txt    # Python dependencies
├── vark_model.json     # Compiled DSPy model (สร้างหลัง optimize)
├── feedback_dataset.jsonl  # User feedback (สร้างอัตโนมัติ)
└── public/             # Static Frontend Files
    ├── index.html      # VARK Quiz
    ├── study.html      # Personalized Study (NEW - split-pane UI)
    ├── style.css       # Quiz styles
    └── main.js         # Quiz logic
```

---

## การติดตั้ง

### 1. ติดตั้ง Python dependencies

```bash
uv add -r requirements.txt
```

> **หมายเหตุ Whisper**: ต้องติดตั้ง `ffmpeg` ก่อน
> ```bash
> # macOS
> brew install ffmpeg
> # Ubuntu/Debian
> sudo apt install ffmpeg
> ```

### 2. ตั้งค่า Environment Variables

สร้างไฟล์ `.env` หรือ export ใน terminal:

```bash
# จำเป็น: Gemini API Key (https://aistudio.google.com)
export GOOGLE_API_KEY="AIza..."

# จำเป็น: YouTube Data API v3 Key (https://console.cloud.google.com)
# เปิด YouTube Data API v3 แล้วสร้าง API Key
export YOUTUBE_API_KEY="AIza..."

```

### 3. (ตัวเลือก) Compile DSPy Model ก่อนรัน

```bash

#window
python dspy_module.py --api-key TYPHOON_API_KEY --output vark_model.json
```

ถ้าไม่รัน ระบบจะใช้ zero-shot โดยอัตโนมัติ

---
feedback real
curl -X POST http://localhost:8000/recompile

## การรัน

### Start Backend (FastAPI)

```bash
python main.py
# หรือ
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API จะอยู่ที่ `http://localhost:8000`  
Swagger UI: `http://localhost:8000/docs`

### Start Frontend (Static files)

วางไฟล์ใน `public/` แล้ว serve ด้วย node server:

```bash
# ติดตั้ง dependencies (ครั้งแรก)
npm install express

# รัน
node --watch server.js
```

Frontend จะอยู่ที่ `http://localhost:3000`

---

## API Endpoints

| Method | Path | คำอธิบาย |
|--------|------|----------|
| `POST` | `/generate` | รับ PDF + VARK style → คืน learning material + YouTube |
| `POST` | `/feedback` | บันทึก liked sessions เป็น training data |
| `GET`  | `/feedback/dataset` | ดู feedback dataset ทั้งหมด |
| `POST` | `/recompile` | Re-optimize DSPy model ด้วย feedback data |

### POST /generate

**Form Data:**
- `pdf`: ไฟล์ PDF
- `vark_style`: JSON string เช่น `{"V":40,"A":20,"R":30,"K":10,"dominant":"V"}`
- `topic`: คำสั่งเพิ่มเติม (optional)

**Response:**
```json
{
  "session_id": "20240101120000_...",
  "learning_material": "## สรุปเนื้อหา...",
  "youtube_queries": ["if-else C tutorial", "คำสั่ง if-else ภาษา C"],
  "videos": [{"video_id": "...", "title": "...", "embed_url": "..."}],
  "context_snippet": "...",
  "vark_style": {"V":40, ...}
}
```

---

## Re-training Flow

เมื่อมี feedback เพียงพอ (≥ 3 liked sessions):

```bash
curl -X POST http://localhost:8000/recompile
```

หรือเรียกใช้ผ่าน Python:

```python
import dspy_module
dspy_module.compile_and_save(model_path="vark_model.json")
```

---

## Frontend Study Page Features

| Feature | ที่ไหน |
|---------|--------|
| PDF Sidebar + Add/Remove | คลิก **+ Add** ใน sidebar ซ้าย |
| PDF Viewer (scroll/page) | Panel กลาง |
| Generate Material | แถบ Generate บน Study Guide |
| Study Guide (Markdown) | Tab "Study Guide" |
| Summary | Tab "Summary" |
| YouTube Embed | Tab "Videos" |
| Like / Feedback | ปุ่ม 👍 ใต้ Study Guide |
| Speech-to-Text | ปุ่ม 🎙️ (Web Speech API, Thai) |

---

## การเชื่อมต่อ Frontend ↔ Backend

ใน `study.html` แก้ไข:
```javascript
const API_BASE = 'http://localhost:8000';
```
เป็น URL ของ server จริง เช่น `https://your-domain.com`

#### ทดลอง

   Commands ที่ใช้บ่อย:

  $env:PYTHONIOENCODING = "utf-8"
  #  Target เดียว, compile แล้ว eval blind ทันที
  python dspy_module.py --target vark --eval-blind

  # Blind eval อย่างเดียว (judges ไม่เห็น expected)
  python dspy_module.py --target all --skip-compile --eval-blind

  # Reference-augmented eval อย่างเดียว (เดิม)
  python dspy_module.py --target all --skip-compile --eval

  # ★ ที่อยากใช้ — รันคู่กันเทียบ gap
  python dspy_module.py --target all --skip-compile --eval --eval-blind

  # Full pipeline: enrich → compile → eval ทั้ง 2 modes
  python dspy_module.py --target all --enrich --eval --eval-blind

  # Target เดียว, compile แล้ว blind eval ทันที
  python dspy_module.py --target video --eval-blind

  # Try GEPA
  python dspy_module.py --target vark --optimizer gepa --gepa-max-calls 30
  # Bootstrap
  python dspy_module.py --target vark --optimizer bootstrap