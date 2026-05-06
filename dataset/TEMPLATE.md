# VARK Dataset — Manual Entry Template

> **Policy:** Manual-only dataset. ห้ามใช้ synthetic / SMOTE / LLM-generated entries.
> ทุก entry ต้องเขียนมือและตรวจ label เอง

---

## วิธีเพิ่ม entry ใหม่

1. เปิดไฟล์ของ split ที่ต้องการ:
   - `dataset/train/vark_train.json` — สำหรับ training (BootstrapFewShot demos)
   - `dataset/val/vark_val.json` — สำหรับ validation (เลือก hyperparameter)
   - `dataset/test/vark_test.json` — สำหรับ final evaluation (อย่าใช้ตอน train!)

2. Copy block ด้านล่างไปวางต่อท้าย array (อย่าลืม comma คั่น)

3. กรอก `id`, `context`, `vark_style`, `notes` ให้ครบ

4. ตรวจตาม checklist ด้านล่างก่อนบันทึก

---

## Block Template (copy แล้ววาง)

```json
{
  "id": "train_XXX",
  "context": "<<เนื้อหาที่จะให้ AI สอน — ภาษาไทย/อังกฤษ ผสมได้ ความยาว ~50-150 คำ>>",
  "vark_style": {
    "V": 0,
    "A": 0,
    "R": 0,
    "K": 0,
    "dominant": "?"
  },
  "source": "manual",
  "split": "train",
  "notes": "<<ใส่ความคาดหวังของ output เช่น 'V dominant — expects table comparing X vs Y'>>"
}
```

---

## Worked Examples (ดูแล้วเขียนตาม)

ตัวอย่าง entry กระจายครบทั้ง 3 split (train / val / test) และครบ 4 dominant style
**อย่า copy หัวข้อเหล่านี้ไปใช้** เพราะอาจซ้ำกับ entry ที่มีอยู่แล้ว ให้คิดหัวข้อใหม่

> สังเกต: `id` prefix และ `split` ต้องตรงกันเสมอ
> เช่น `train_006` → `"split": "train"`, `val_001` → `"split": "val"`, `test_004` → `"split": "test"`

---

### Example A — TRAIN split, V (Visual) dominant

เลือก V เมื่อหัวข้อเหมาะกับการ **เปรียบเทียบเป็นตาราง / แผนผัง / timeline**
มักเป็นหัวข้อที่มี "หลายรูปแบบให้จำ" หรือ "โครงสร้างที่เห็นภาพได้"

```json
{
  "id": "train_006",
  "context": "Comparison of Adjectives เปรียบเทียบคุณสมบัติ 3 ระดับ: Positive (tall), Comparative (taller / more beautiful), Superlative (the tallest / the most beautiful) คำสั้น 1-2 พยางค์ใช้ -er/-est คำยาวใช้ more/most นำหน้า เช่น 'This building is taller than that one. It is the tallest in the city.'",
  "vark_style": {
    "V": 60,
    "A": 10,
    "R": 20,
    "K": 10,
    "dominant": "V"
  },
  "source": "manual",
  "split": "train",
  "notes": "V dominant — expects 3-column table comparing Positive/Comparative/Superlative with example adjectives"
}
```

---

### Example B — VAL split, A (Aural) dominant

เลือก A เมื่อหัวข้อเหมาะกับการ **เล่าเรื่อง / บทสนทนา / สถานการณ์จำลอง**
มักเป็นหัวข้อที่ต้องเข้าใจผ่านบริบทการพูดมากกว่ากฎ

```json
{
  "id": "val_001",
  "context": "Polite Requests ในภาษาอังกฤษใช้รูปต่าง ๆ เช่น 'Could you...?', 'Would you mind...?', 'Do you think you could...?' แต่ละแบบสุภาพต่างระดับกัน เช่น 'Could you pass the salt?' สุภาพปานกลาง 'Would you mind opening the window?' สุภาพมากกว่า ใช้ในร้านอาหาร โรงแรม หรือสถานการณ์ทางการ",
  "vark_style": {
    "V": 10,
    "A": 60,
    "R": 20,
    "K": 10,
    "dominant": "A"
  },
  "source": "manual",
  "split": "val",
  "notes": "A dominant — expects dialogue between waiter/customer or guest/receptionist using each request form naturally"
}
```

---

### Example C — TEST split, R (Read/Write) dominant

เลือก R เมื่อหัวข้อเหมาะกับการ **อ่านนิยาม / outline / bullet list**
มักเป็นหัวข้อที่มีกฎเขียนเป็นข้อ ๆ ได้ชัดเจน

```json
{
  "id": "test_004",
  "context": "Determiners คือคำที่วางหน้าคำนามเพื่อระบุ ได้แก่ articles (a, an, the), demonstratives (this, that, these, those), possessives (my, your, his), quantifiers (some, any, much, many) แต่ละประเภทใช้ต่างกันตามคำนามและความหมาย เช่น 'this book' (ใกล้), 'that book' (ไกล), 'my books' (ของฉัน), 'many books' (จำนวนมาก)",
  "vark_style": {
    "V": 15,
    "A": 10,
    "R": 60,
    "K": 15,
    "dominant": "R"
  },
  "source": "manual",
  "split": "test",
  "notes": "R dominant — expects definition list categorizing 4 determiner types with rules and examples per category"
}
```

---

### Example D — TRAIN split, K (Kinesthetic) dominant

เลือก K เมื่อหัวข้อเหมาะกับการ **ทำแบบฝึกหัด / step-by-step / ลงมือทำ**
มักเป็นหัวข้อที่ "ต้องลองสร้างประโยค" หรือ "แปลงรูปประโยค" ถึงเข้าใจ

```json
{
  "id": "train_007",
  "context": "Sentence Transformation เปลี่ยนรูปประโยคโดยรักษาความหมายเดิม เช่น Active → Passive ('Tom ate the cake.' → 'The cake was eaten by Tom.'), Direct → Reported ('She said, I am happy.' → 'She said that she was happy.'), Affirmative → Negative ('He likes coffee.' → 'He doesn't like coffee.') ฝึกแปลงจะช่วยเข้าใจโครงสร้างประโยคแต่ละแบบ",
  "vark_style": {
    "V": 10,
    "A": 10,
    "R": 20,
    "K": 60,
    "dominant": "K"
  },
  "source": "manual",
  "split": "train",
  "notes": "K dominant — expects step-by-step transformation exercise (give source sentence, learner writes target form, then check answer)"
}
```

---

## กฎการกรอก field

### `id`
- รูปแบบ: `<split>_<3-digit>` เช่น `train_006`, `val_001`, `test_004`
- ต้อง unique ทั้ง dataset (ห้ามซ้ำข้าม split)

### `context`
- เนื้อหาเรียน 1 หัวข้อ (ไม่ปนหลายหัวข้อใน entry เดียว)
- ต้องเป็นข้อความ plain text ไม่มี markdown
- มีตัวอย่างการใช้งานจริงอย่างน้อย 1 ตัวอย่าง
- หลีกเลี่ยงหัวข้อซ้ำกับ entry อื่น (โดยเฉพาะข้าม train/val/test — ห้าม leakage!)

### `vark_style`
- V + A + R + K ต้องรวมกัน = 100
- `dominant` ต้องเป็น style ที่ score สูงสุด (ถ้าเสมอ เลือกที่ตรงเจตนาผู้สอนมากที่สุด)
- score ของ dominant ควร ≥ 45 เพื่อให้สัญญาณชัดเจน

### `source`
- ต้องเป็น `"manual"` เสมอ (ไม่มีค่าอื่น)

### `split`
- ต้องตรงกับชื่อไฟล์ (`train` / `val` / `test`)

### `notes`
- 1 ประโยคบอกว่าคาดหวัง output แบบไหน
- รูปแบบที่แนะนำ: `<dominant> dominant — expects <รูปแบบที่คาดหวัง>`

---

## Pre-commit Checklist

- [ ] `id` ไม่ซ้ำ entry อื่น
- [ ] `context` ไม่ซ้ำหัวข้อกับ split อื่น (zero leakage)
- [ ] V+A+R+K รวมเป็น 100
- [ ] `dominant` ตรงกับ style ที่ score สูงสุด
- [ ] `split` ตรงกับไฟล์
- [ ] `notes` อธิบายความคาดหวังของ output

---

## Distribution Targets (ขั้นต่ำที่ควรมี)

| Split | Min count | V | A | R | K |
|---|---|---|---|---|---|
| train | 12 | ≥3 | ≥3 | ≥3 | ≥3 |
| val   | 4  | ≥1 | ≥1 | ≥1 | ≥1 |
| test  | 4  | ≥1 | ≥1 | ≥1 | ≥1 |

> เป้าหมาย: balanced VARK distribution ทุก split เพื่อให้ metric น่าเชื่อถือ
