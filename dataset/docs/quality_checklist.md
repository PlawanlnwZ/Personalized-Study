# Data Quality Checklist
## AI Builders Rubric — VARK Dataset (Manual Only)

> **Policy:** ทุก entry ต้อง manual เท่านั้น ห้าม synthetic / SMOTE / LLM-generated

---

### ✅ The Non-Trivial Effort Test

- [x] **Origin explained**: ทุก entry เขียนมือโดย project owner ตาม English grammar curriculum
  ดู `raw/sources.md` สำหรับรายละเอียด
- [x] **Non-trivial prep step performed**:
  - Hand-crafted grammar teaching contexts (Thai + English mixed)
  - Manual VARK labeling per entry — score V/A/R/K แล้ว derive `dominant` จาก argmax
  - ตรวจ leakage ระหว่าง train/val/test (ไม่มีหัวข้อซ้ำข้าม split)
  - ทำ TEMPLATE.md เพื่อ standardize การเพิ่ม entry ใหม่ในอนาคต
- [x] **Before/after shown**: `raw/sources.md` บันทึกว่าเคยมี synthetic แล้วถูกลบออก

---

### ✅ Label Quality

- [x] **Manual sample inspected**: ทุก entry ถูกตรวจด้วยมือ
  field `notes` บันทึกความคาดหวังของ output style
- [x] **Labels consistent**: `dominant` คือ argmax ของ V/A/R/K
  ผู้ตรวจคนอื่นสามารถ re-derive ได้เองโดยไม่กำกวม
- [x] **Synthetic label error rate**: N/A — ไม่มี synthetic
  ทุก label ตรวจด้วยมือ → label error เป็นไปได้แค่ human error เท่านั้น

---

### ✅ Input Quality

- [x] **Duplicates checked**: ทุก `id` unique ตรวจ context strings ด้วยมือ ไม่มี duplicate/near-duplicate
- [x] **Corrupted/empty inputs checked**: ทุก `context` เป็น non-empty string
  ทุก `vark_style` มีครบ 4 keys + `dominant` และ V+A+R+K = 100
- [x] **Encoding checked**: UTF-8 ทั้งหมด ภาษาไทย render ถูกต้อง
- [x] **Input distribution checked**:

  ⚠️ **Note**: dataset เล็กมาก (8 entries) — train ปัจจุบัน balance V/A/R/K ได้พอใช้
  val ว่าง รอเพิ่ม
  เป้าหมาย: train ≥12, val ≥4, test ≥4 (ดู TEMPLATE.md)

---

### ✅ Train / Val / Test Splits

- [x] **Split strategy**: Manually assigned (ไม่ random) เพื่อให้ทุก dominant style ปรากฏใน train
- [x] **Zero leakage**: หัวข้อใน test (Question Tags, Gerunds, Prepositions) ไม่ทับกับ train เลย
- [x] **No time dimension**: ไม่มี temporal ordering — time-based split ไม่จำเป็น
- [x] **Test set representative**: test ครอบคลุม V/A/R (ขาด K) — flag ไว้รอเพิ่ม
- [x] **Manual-only across all splits**: ตรวจแล้วว่าทุก entry มี `source: "manual"`

---

### ⚠️ Known Limitations

1. **Small size (8 entries)**: เพียงพอสำหรับ few-shot demo (BootstrapFewShot) แต่ไม่พอสำหรับ statistical evaluation
   เป้าหมาย: ขยายเป็น 20+ entries (manual ทั้งหมด)
2. **Domain narrow**: ทุก context เกี่ยวกับ English grammar generalisation ข้ามวิชาอื่นไม่ได้ทดสอบ
3. **Val ว่าง**: ตอนนี้ไม่มี validation set — hyperparameter tuning ทำไม่ได้จนกว่าจะเพิ่ม entries
4. **K dominant ใน test ขาด**: ควรเพิ่ม test entry แบบ K ก่อน publish ผล evaluation

---

### Next Steps (for project expansion)

- [ ] เพิ่ม val entries อย่างน้อย 4 (manual) ครอบคลุม V/A/R/K
- [ ] เพิ่ม test K dominant อย่างน้อย 1 entry
- [ ] ขยาย train เป็น 12+ entries
- [ ] Inter-annotator agreement check สำหรับ VARK dominant labels (ถ้ามีคนตรวจคนที่สอง)
- [ ] รัน `vark_metric` บน test set แล้ว report baseline score
