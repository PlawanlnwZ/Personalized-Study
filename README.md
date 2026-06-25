# https://personalized-study-usam.onrender.com

# VARK Personalized Study System

เว็บแอปที่ช่วย **สร้างสื่อการเรียนรู้จากไฟล์ PDF** ให้ตรงกับสไตล์การเรียนของแต่ละคน (VARK)

> **VARK คืออะไร?** เป็นทฤษฎีที่แบ่งคนออกเป็น 4 แบบตามวิธีที่เรียนรู้ได้ดีที่สุด
> - **V** (Visual) — เรียนรู้ผ่านภาพ แผนภาพ คลิปวิดีโอ
> - **A** (Aural) — เรียนรู้ผ่านการฟัง
> - **R** (Read/Write) — เรียนรู้ผ่านการอ่าน/เขียน
> - **K** (Kinesthetic) — เรียนรู้ผ่านการลงมือทำ ตัวอย่างจริง

---

## แอปนี้ทำอะไรได้บ้าง

1. **ทำแบบทดสอบ VARK** เพื่อหาว่าคุณเป็นผู้เรียนแบบไหน
2. **อัปโหลด PDF** (เช่น สไลด์เรียน หรือ ชีท) แล้วให้ AI สรุปเนื้อหาให้
   ในรูปแบบที่เหมาะกับสไตล์การเรียนของคุณ
3. **แนะนำวิดีโอ YouTube** ที่เกี่ยวข้องกับเนื้อหา
4. **สร้างแบบทดสอบ (Quiz)** จากเนื้อหานั้นโดยอัตโนมัติ
5. **อ่านเนื้อหาออกเสียง (Text-to-Speech)**

เบื้องหลังใช้ AI หลายตัวทำงานร่วมกัน — ดูหัวข้อ ["ใช้ AI อะไรบ้าง"](#ใช้-ai-อะไรบ้าง) ด้านล่าง

---

## เริ่มต้นใช้งานแบบเร็ว (Quick Start)

ทำตาม 3 ขั้นตอนนี้ก็รันได้เลย

### ขั้นที่ 1 — ติดตั้งโปรแกรมที่ต้องใช้

ต้องมี **Python 3.10 ขึ้นไป** ในเครื่อง จากนั้นติดตั้งไลบรารีที่จำเป็น:

```bash
pip install -r requirements.txt
```

### ขั้นที่ 2 — ใส่ API Key

แอปนี้เรียกใช้บริการ AI ภายนอก จึงต้องมี "กุญแจ" (API Key) ก่อน
สร้างไฟล์ชื่อ `.env` ไว้ในโฟลเดอร์โปรเจกต์ แล้วใส่ค่าตามนี้:

```bash
# จำเป็น — โมเดลหลักที่ใช้สร้างเนื้อหา (Typhoon)
TYPHOON_API_KEY="..."

# จำเป็น — โมเดลสำรอง GPT-OSS (ผ่าน OpenRouter)
OPENROUTER_API_KEY="..."

# จำเป็น — ใช้ถอดเสียงวิดีโอ และใช้เป็น "กรรมการ" ตอนประเมินผล
# ขอได้ที่ https://aistudio.google.com
GEMINI_API_KEY="..."

# ไม่บังคับ — ค้นวิดีโอ YouTube จริง
# ถ้าไม่ใส่ แอปจะแสดงเป็นลิงก์ค้นหาแทนวิดีโอ
# ขอได้ที่ https://console.cloud.google.com (เปิด "YouTube Data API v3")
YOUTUBE_API_KEY="..."
```

### ขั้นที่ 3 — รันเซิร์ฟเวอร์

**ผู้ใช้ Windows:** ต้องตั้งค่า encoding ก่อน ไม่งั้นโปรแกรมจะ error ตอนเปิด

```powershell
$env:PYTHONIOENCODING = "utf-8"
uvicorn main:app --reload
```

**ผู้ใช้ macOS / Linux:**

```bash
uvicorn main:app --reload
```

เปิดเบราว์เซอร์ไปที่:

| หน้า | ลิงก์ |
|------|------|
| แบบทดสอบ VARK | http://localhost:8000/ |
| หน้าเรียน (อัปโหลด PDF) | http://localhost:8000/study |

> เซิร์ฟเวอร์ตัวเดียว (FastAPI) ทำหน้าที่ทั้งหลังบ้านและหน้าเว็บ
> **ไม่ต้องเปิด Node แยก** — ไฟล์ `server.js` เป็นแค่ตัวช่วยตอนพัฒนาเท่านั้น

---

## โครงสร้างไฟล์

```
project/
├── main.py             # หลังบ้าน (FastAPI) — รับไฟล์ จัดการ endpoint ต่าง ๆ
├── dspy_module.py      # หัวใจของ AI — โค้ดที่คุยกับโมเดล + ตัวช่วยเรื่อง YouTube
├── requirements.txt    # รายการไลบรารี Python
├── render.yaml         # ตั้งค่าสำหรับ deploy ขึ้น Render
├── dataset/train/      # ข้อมูลตัวอย่างไว้ทดสอบ/ประเมินผล
└── public/             # หน้าเว็บ (Frontend)
    ├── index.html      #   หน้าแบบทดสอบ VARK
    ├── main.js         #   ตรรกะของแบบทดสอบ
    ├── study.html      #   หน้าเรียน (แบ่งจอ: PDF / เนื้อหา / วิดีโอ)
    └── style.css       #   สไตล์หน้าเว็บ
```

---

## ใช้ AI อะไรบ้าง

แอปเรียกใช้ AI ผ่านไลบรารี **DSPy** โดยแต่ละตัวมีหน้าที่ต่างกัน:

| โมเดล | หน้าที่ | กุญแจที่ใช้ |
|-------|--------|-----------|
| **Typhoon** | สร้างเนื้อหาหลัก (สรุป + แบบทดสอบ) | `TYPHOON_API_KEY` |
| **GPT-OSS** | โมเดลสำรองสำหรับเปรียบเทียบผล | `OPENROUTER_API_KEY` |
| **Gemini** | ถอดเสียงวิดีโอ + เป็น "กรรมการ" ให้คะแนนตอนประเมิน | `GEMINI_API_KEY` |

> **หมายเหตุ:** ทุกโมเดลทำงานแบบ *zero-shot* คือใส่คำสั่งครบในตัวมันเลย
> **ไม่มีขั้นตอน train / compile** การปรับปรุงคุณภาพทำโดยแก้คำสั่ง (prompt)
> ในไฟล์ `dspy_module.py` โดยตรง แล้วประเมินผลใหม่

---

## API Keys ที่ต้องใช้

| กุญแจ | บังคับไหม | ใช้ทำอะไร |
|-------|----------|-----------|
| `TYPHOON_API_KEY` | ✅ จำเป็น | โมเดลหลักที่สร้างเนื้อหา |
| `OPENROUTER_API_KEY` | ✅ จำเป็น | โมเดลสำรอง GPT-OSS |
| `GEMINI_API_KEY` | ✅ จำเป็น | ถอดเสียงวิดีโอ + ประเมินผล |
| `YOUTUBE_API_KEY` | ✅ จำเป็น | ค้นวิดีโอจริง (ถ้าไม่มีจะใช้ลิงก์ค้นหาแทน) |

---

## API Endpoints หลัก

| Method | Path | คำอธิบาย |
|--------|------|----------|
| `POST` | `/generate` | รับ PDF + สไตล์ VARK → คืนสื่อการเรียน + วิดีโอ |
| `POST` | `/quiz` | สร้างแบบทดสอบจากเนื้อหา |
| `POST` | `/adapt` | ปรับเนื้อหาตามที่ผู้ใช้ขอ |
| `POST` | `/tts` | อ่านข้อความออกเสียง (Text-to-Speech) |
| `POST` | `/evaluate` | ให้ Gemini ประเมินคุณภาพเนื้อหา |

### ตัวอย่าง: POST `/generate`

**ส่งข้อมูลแบบ Form:**
- `pdf` — ไฟล์ PDF
- `vark_style` — ข้อความ JSON เช่น `{"V":40,"A":20,"R":30,"K":10,"dominant":"V"}`
- `topic` — คำสั่งเพิ่มเติม (ใส่หรือไม่ใส่ก็ได้)

**ผลลัพธ์ที่ได้กลับมา:**
```json
{
  "session_id": "20240101120000_...",
  "learning_material": "## สรุปเนื้อหา...",
  "youtube_queries": ["if-else C tutorial", "คำสั่ง if-else ภาษา C"],
  "videos": [{"video_id": "...", "title": "...", "embed_url": "..."}],
  "context_snippet": "...",
  "vark_style": {"V": 40, "...": "..."}
}
```

---

## ฟีเจอร์ในหน้าเรียน (Study Page)

| ฟีเจอร์ | อยู่ตรงไหน |
|---------|-----------|
| เพิ่ม/ลบ ไฟล์ PDF | ปุ่ม **+ Add** ในแถบซ้าย |
| อ่าน PDF (เลื่อน/เปลี่ยนหน้า) | แผงตรงกลาง |
| สั่งสร้างเนื้อหา | แถบ Generate เหนือ Study Guide |
| สื่อการเรียน (Markdown) | แท็บ "Study Guide" |
| สรุปย่อ | แท็บ "Summary" |
| วิดีโอ YouTube | แท็บ "Videos" |

---

## การประเมินผลโมเดล (สำหรับนักพัฒนา)

ทุกโมเดลเป็น zero-shot การปรับปรุงคือ "แก้ prompt → ประเมินผลใหม่"
โดยใช้ Gemini เป็นกรรมการให้คะแนน

> **Windows:** ตั้ง encoding ก่อนทุกครั้ง → `$env:PYTHONIOENCODING = "utf-8"`

| ต้องการประเมิน | คำสั่ง |
|----------------|--------|
| Study Guide (อ้างอิงเฉลย) | `python dspy_module.py --target vark --eval` |
| Study Guide (ไม่ดูเฉลย) | `python dspy_module.py --target vark --eval-blind` |
| Study Guide (ทั้งสองแบบ) | `python dspy_module.py --target vark --eval --eval-blind` |
| Quiz | `python dspy_module.py --target quiz --eval-blind` |
| ทุกอย่าง | `python dspy_module.py --target all --eval --eval-blind` |

- เลือก `--target` ได้ระหว่าง `vark` / `quiz` / `video` / `all`
- รายงานผลถูกบันทึกไว้ที่โฟลเดอร์ `reports/` (เช่น `reports/vark_eval1.json` และ `.md`)

---

## การนำขึ้นออนไลน์ (Deploy)

โปรเจกต์ตั้งค่าไว้สำหรับ deploy บน **Render** ผ่านไฟล์ `render.yaml`

- คำสั่งติดตั้ง: `pip install -r requirements.txt`
- การถอดเสียงวิดีโอบน Render ใช้ Gemini ถอดให้จาก URL โดยตรง
  (เพราะ IP ของ Render มักโดน YouTube บล็อก) — จึงต้องมี `GEMINI_API_KEY`
- ปรับแต่งเพิ่มได้ผ่าน env: `GEMINI_TRANSCRIBE_SECONDS` (ค่าเริ่มต้น 120 วินาที)
  และ `GEMINI_TRANSCRIBE_MODEL` (ค่าเริ่มต้น `gemini-2.5-flash`)
