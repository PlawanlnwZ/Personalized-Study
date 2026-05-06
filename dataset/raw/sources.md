# Raw Data Sources — VARK Dataset (วิชาภาษาอังกฤษ)

This file tracks the origin of every example in this dataset,
as required by the AI Builders data quality checklist.

> **Policy:** Manual-only dataset — ทุก entry เขียนมือและตรวจ label เอง
> ห้าม synthetic / SMOTE / LLM-generated examples

## Sources

### manual (เขียนมือทั้งหมด)
- **Origin**: Hand-authored by project owner, modeled after standard English grammar curriculum
- **IDs**: `train_001`–`train_005`, `test_001`–`test_003`
- **Topics (train)**: Present Simple, Past Continuous, Conditional Type 1, Passive Voice, Reported Speech
- **Topics (test)**: Question Tags, Gerunds & Infinitives, Prepositions of Time
- **Format**: Plain Thai/English explanation with at least one example sentence
- **Label process**: VARK scores assigned by inspecting which teaching modality fits the topic best;
  `dominant` is computed deterministically as the argmax of V/A/R/K scores
- **Inspection status**: All examples manually reviewed before commit
- **License**: Internal / project-owned


## Data Effort Level
Per the AI Builders rubric:
- **Level 3**: Hand-crafted from grammar curriculum + manual labeling
- ทุก entry ผ่านการตรวจสอบโดยมนุษย์ ไม่มี automated generation
