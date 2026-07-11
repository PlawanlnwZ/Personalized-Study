// ============================================================
// VARK Quiz — Main Script
// ข้อมูล VARK ถูกบันทึกเบื้องหลังโดยไม่แสดงใน UI
// ============================================================

// ── Backend base URL (FastAPI) — same-origin บน production (Render), localhost ตอน dev ─
const API_BASE = window.location.origin;

// ══════════════════════════════════════════════════════════════
//  TTS Engine — gTTS via /tts backend endpoint
//  (เสียงโจทย์ 16 ข้อยังใช้ไฟล์ VARKsound/sound{n}.wav เหมือนเดิม)
// ══════════════════════════════════════════════════════════════

const TTS = {
  enabled:    false,
  speaking:   false,
  _audio:     null,   // Audio object สำหรับเสียงโจทย์
  _narratorAudio: null, // Audio object สำหรับเสียง VARKy (gTTS)
  _readTimer: null,   // pending readCurrentQuestion setTimeout — cleared on stop()

  // ── เริ่มทุกครั้งที่ refresh = ปิดเสียงไว้ก่อน (ไม่ restore localStorage) ─
  init() {
    this.enabled = false;
    try { localStorage.removeItem('__vark_tts__'); } catch (e) {}
    this._syncButton();
  },

  // ── sync toggle button visuals to current `enabled` ──────
  _syncButton() {
    const btn  = document.getElementById('tts-toggle-btn');
    const icon = document.getElementById('tts-icon');
    const lbl  = document.getElementById('tts-label');
    if (!btn || !icon || !lbl) return;
    if (this.enabled) {
      btn.classList.add('tts-on');
      icon.textContent = '🔊';
      lbl.textContent  = 'ปิดเสียงอ่านโจทย์';
    } else {
      btn.classList.remove('tts-on');
      icon.textContent = '🔇';
      lbl.textContent  = 'เปิดเสียงอ่านโจทย์';
    }
  },

  // ── toggle เปิด/ปิด ──────────────────────────────────────
  toggle() {
    this.enabled = !this.enabled;
    try { localStorage.setItem('__vark_tts__', this.enabled ? '1' : '0'); } catch (e) {}
    this._syncButton();

    if (this.enabled) {
      showToast('✨ เปิดเสียงอ่านโจทย์');
      const quiz = document.getElementById('quiz-screen');
      if (quiz && quiz.classList.contains('active')) {
        this.readCurrentQuestion();
      } else {
        // intro / result screen → พากย์บท VARKy ปัจจุบัน
        const nt = document.getElementById('narrator-text');
        if (nt) this.speakNarrator(nt.textContent);
      }
    } else {
      this.stop();
      showToast('🔇 ปิดเสียงอ่านโจทย์แล้ว');
    }
  },

  // ── หยุดเสียงทันที ───────────────────────────────────────
  stop() {
    // Cancel any pending question-read timer (fixes Q16 audio playing after "ดูผลลัพธ์")
    if (this._readTimer) {
      clearTimeout(this._readTimer);
      this._readTimer = null;
    }
    if (this._audio) {
      this._audio.pause();
      this._audio.src = '';
      this._audio = null;
    }
    if (this._narratorAudio) {
      this._narratorAudio.pause();
      const src = this._narratorAudio.src;
      this._narratorAudio.src = '';
      this._narratorAudio = null;
      if (src && src.startsWith('blob:')) URL.revokeObjectURL(src);
    }
    this.speaking = false;
    this._setWave(false);
  },

  // ── พากย์บท Narrator ด้วย gTTS (Phase 3, gTTS edition) ──
  async speakNarrator(text) {
    if (!this.enabled) return;
    const clean = String(text || '').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
    if (!clean) return;

    // Stop previous narrator audio only — question .wav stays
    if (this._narratorAudio) {
      this._narratorAudio.pause();
      const prev = this._narratorAudio.src;
      this._narratorAudio.src = '';
      this._narratorAudio = null;
      if (prev && prev.startsWith('blob:')) URL.revokeObjectURL(prev);
    }

    const reqId = ++this._narratorReqId;
    try {
      const resp = await fetch(`${API_BASE}/tts`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ text: clean, lang: 'th' }),
      });
      if (!resp.ok) throw new Error(`tts ${resp.status}`);
      const blob = await resp.blob();
      // ผู้ใช้กดปิด/พูดเรื่องใหม่ระหว่างรอ → ทิ้งผลลัพธ์
      if (!this.enabled || reqId !== this._narratorReqId) return;

      const url   = URL.createObjectURL(blob);
      const audio = new Audio(url);
      this._narratorAudio = audio;

      audio.onplay  = () => this._setWave(true);
      audio.onended = () => {
        URL.revokeObjectURL(url);
        if (this._narratorAudio === audio) this._narratorAudio = null;
        if (!this.speaking) this._setWave(false);
      };
      audio.onerror = () => {
        URL.revokeObjectURL(url);
        if (this._narratorAudio === audio) this._narratorAudio = null;
        if (!this.speaking) this._setWave(false);
      };

      await audio.play();
    } catch (err) {
      console.warn('[TTS] gTTS narrator error:', err);
      if (this._narratorAudio === null && !this.speaking) this._setWave(false);
    }
  },

  _narratorReqId: 0,

  // ── เล่นไฟล์เสียงของข้อปัจจุบัน ─────────────────────────
  readCurrentQuestion() {
    if (!this.enabled) return;
    this.stop();

    // Bank 1 → sound1-16, Bank 2 → sound17-32
    const n     = currentQ + 1 + (currentBankIdx * 16);
    const url   = `/static/VARKsound/sound${n}.wav`;
    const audio = new Audio(url);
    this._audio = audio;

    this.speaking = true;
    this._setWave(true);

    audio.onended = () => { this.speaking = false; this._setWave(false); };
    audio.onerror = () => {
      this.speaking = false; this._setWave(false);
      console.warn(`[TTS] ไม่พบไฟล์เสียง: ${url}`);
    };

    audio.play().catch(() => {
      this.speaking = false; this._setWave(false);
    });
  },

  // ── wave indicator ────────────────────────────────────────
  _setWave(on) {
    const bar = document.getElementById('tts-speaking-bar');
    if (bar) bar.classList.toggle('visible', on);
  }
};

function toggleTTS() { TTS.toggle(); }

// ── VARK Question Bank #1 ───────────────────────────────────
const BANK_1 = [
  {
    text: "คุณกำลังให้ความช่วยเหลือคนที่ต้องการจะไปสนามบิน ตัวเมือง หรือสถานีรถไฟในเมืองที่คุณอยู่ คุณจะ",
    options: [
      { label: "ก", text: "พาไป" },
      { label: "ข", text: "บอกทางให้" },
      { label: "ค", text: "เขียนทางที่จะไป (โดยไม่มีแผนที่)" },
      { label: "ง", text: "วาดแผนที่ให้ หรือ ให้แผนที่" }
    ],
    scoring: { ก: "K", ข: "A", ค: "R", ง: "V" }
  },
  {
    text: "ถ้าคุณไม่แน่ใจว่าคำว่า 'dependent' หรือ 'dependant' สะกดอย่างไร คุณจะ",
    options: [
      { label: "ก", text: "นึกภาพคำนั้นในใจ และเลือกตามที่คุณคิดว่าน่าจะใช้" },
      { label: "ข", text: "ลองออกเสียงแต่ละคำในใจและเลือกเอาหนึ่งคำ" },
      { label: "ค", text: "เปิดหาในพจนานุกรม" },
      { label: "ง", text: "เขียนคำทั้งสองลงบนกระดาษ แล้วเลือกเอาคำหนึ่ง" }
    ],
    scoring: { ก: "V", ข: "A", ค: "R", ง: "K" }
  },
  {
    text: "คุณกำลังวางแผนที่จะไปพักผ่อนกับกลุ่มเพื่อนๆ คุณต้องการฟังข้อคิดเห็นจากพวกเขาเกี่ยวกับแผนงานนั้น คุณจะ",
    options: [
      { label: "ก", text: "อธิบายประเด็นที่สำคัญๆ" },
      { label: "ข", text: "ใช้แผนที่และเว็บไซต์เพื่อแสดงสถานที่ประกอบ" },
      { label: "ค", text: "ถ่ายเอกสารแผนงานของคุณให้เพื่อน" },
      { label: "ง", text: "โทรศัพท์ ส่งข้อความเต็ม หรือ ส่ง e-mail ให้เพื่อน" }
    ],
    scoring: { ก: "K", ข: "V", ค: "R", ง: "A" }
  },
  {
    text: "คุณจะปรุงอาหารซึ่งเป็นมื้อพิเศษสำหรับครอบครัวของคุณ คุณจะ",
    options: [
      { label: "ก", text: "ลงมือปรุงอาหารที่รู้จัก โดยไม่ต้องใช้คู่มือ" },
      { label: "ข", text: "ถามเพื่อนเพื่อขอคำแนะนำ" },
      { label: "ค", text: "ดูคู่มือประกอบการปรุงอาหารเพื่อให้เกิดแนวความคิดจากภาพในหนังสือ" },
      { label: "ง", text: "ใช้คู่มือที่มีรายละเอียด และขั้นตอนการปรุงอาหาร" }
    ],
    scoring: { ก: "K", ข: "A", ค: "V", ง: "R" }
  },
  {
    text: "กลุ่มนักท่องเที่ยวต้องการรู้เกี่ยวกับสวนสาธารณะ หรือเขตสงวนพันธุ์สัตว์ป่าในบริเวณใกล้ๆ ที่คุณพักอยู่ คุณจะ",
    options: [
      { label: "ก", text: "พูดคุยในรายละเอียด หรือ เตรียมเรื่องที่จะพูดคุยเกี่ยวกับสวนสาธารณะหรือเขตสงวนพันธุ์สัตว์ป่าแก่นักท่องเที่ยว" },
      { label: "ข", text: "แสดงภาพในอินเตอร์เน็ต แสดงรูปภาพ หรือภาพในหนังสือแก่นักท่องเที่ยว" },
      { label: "ค", text: "พานักท่องเที่ยวไปที่สวนสาธารณะหรือเขตสงวนพันธุ์สัตว์ป่าและเดินเที่ยวไปด้วยกัน" },
      { label: "ง", text: "ให้หนังสือหรือคู่มือการท่องเที่ยวเกี่ยวกับสวนสาธารณะหรือเขตสงวนพันธุ์สัตว์ป่าแก่นักท่องเที่ยว" }
    ],
    scoring: { ก: "A", ข: "V", ค: "K", ง: "R" }
  },
  {
    text: "คุณจะซื้อกล้องดิจิตอล หรือ โทรศัพท์มือถือ นอกเหนือจากเหตุผลเรื่องราคาแล้ว มีอะไรที่มีอิทธิพลต่อการตัดสินใจของคุณในการที่จะ (ซื้อ หรือ ไม่ซื้อ)",
    options: [
      { label: "ก", text: "ลองใช้หรือทดสอบสินค้า" },
      { label: "ข", text: "อ่านในรายละเอียดตัวสินค้า" },
      { label: "ค", text: "รูปแบบที่ทันสมัย และดูดี" },
      { label: "ง", text: "การแนะนำสินค้าของพนักงานขาย" }
    ],
    scoring: { ก: "K", ข: "R", ค: "V", ง: "A" }
  },
  {
    text: "ลองนึกย้อนกลับไปเมื่อตอนที่คุณหัดทำอะไรใหม่ๆ เช่นการถีบจักรยาน คุณเรียนรู้สิ่งใหม่ได้จาก",
    options: [
      { label: "ก", text: "ดูคนอื่นสาธิตวิธีการให้ดู" },
      { label: "ข", text: "ฟังคนอื่นอธิบายและถามคำถาม" },
      { label: "ค", text: "ดูจากแผนภูมิ แผนภาพ หรือสื่อที่เข้าใจได้จากการมอง" },
      { label: "ง", text: "อ่านคำแนะนำ เช่นคู่มือหรือตำรา" }
    ],
    scoring: { ก: "K", ข: "A", ค: "V", ง: "R" }
  },
  {
    text: "คุณมีปัญหาที่เข่า คุณอยากให้หมอ",
    options: [
      { label: "ก", text: "บอกเกี่ยวกับเว็บไซต์หรืออะไรก็ได้ที่มีรายละเอียดให้อ่าน" },
      { label: "ข", text: "ใช้เข่าพลาสติกจำลอง แสดงปัญหาที่เกิดขึ้น" },
      { label: "ค", text: "อธิบายว่าเข่ามีปัญหาอะไร" },
      { label: "ง", text: "ใช้แผนภาพแสดงความผิดปกติของเข่า" }
    ],
    scoring: { ก: "R", ข: "K", ค: "A", ง: "V" }
  },
  {
    text: "คุณต้องเรียนรู้โปรแกรมทักษะหรือเกมใหม่บนคอมพิวเตอร์ คุณจะ",
    options: [
      { label: "ก", text: "อ่านคู่มือที่มากับโปรแกรม" },
      { label: "ข", text: "พูดคุยกับผู้มีความรู้เกี่ยวกับโปรแกรมนั้น" },
      { label: "ค", text: "ใช้แป้นควบคุมหรือแผงแป้นอักขระ (keyboard) ช่วย" },
      { label: "ง", text: "ทำตามแผนภาพที่มาพร้อมกับโปรแกรม" }
    ],
    scoring: { ก: "R", ข: "A", ค: "K", ง: "V" }
  },
  {
    text: "ฉันชอบเว็บไซต์ที่มี",
    options: [
      { label: "ก", text: "มีสิ่งที่ฉันสามารถกด เปลี่ยน หรือ ทดลอง" },
      { label: "ข", text: "การออกแบบที่น่าสนใจและรูปลักษณ์ชวนมอง" },
      { label: "ค", text: "รายละเอียดเนื้อหาน่าสนใจ รายการและการอธิบายเนื้อหาในเว็บไซต์" },
      { label: "ง", text: "มีฟังก์ชัน เพลงให้ฟัง รายการวิทยุหรือการสัมภาษณ์" }
    ],
    scoring: { ก: "K", ข: "V", ค: "R", ง: "A" }
  },
  {
    text: "นอกเหนือจากเรื่องราคา สิ่งใดที่มีอิทธิพลต่อการตัดสินใจเลือกซื้อหนังสือใหม่ประเภทมิใช่บันเทิงคดี",
    options: [
      { label: "ก", text: "รูปแบบสะดุดตา" },
      { label: "ข", text: "อ่านคร่าวๆ บางตอนของหนังสือแล้วเข้าใจง่าย" },
      { label: "ค", text: "เพื่อนพูดถึง และแนะนำให้ซื้อ" },
      { label: "ง", text: "มีเรื่องที่เกี่ยวกับชีวิตจริง ประสบการณ์และตัวอย่าง" }
    ],
    scoring: { ก: "V", ข: "R", ค: "A", ง: "K" }
  },
  {
    text: "คุณใช้หนังสือ ซีดี หรือ เว็บไซต์ เพื่อเรียนรู้เกี่ยวกับการถ่ายรูปโดยใช้กล้องดิจิตอล คุณอยากจะ",
    options: [
      { label: "ก", text: "มีโอกาสถามคำถามและพูดคุยเกี่ยวกับตัวกล้องและรายละเอียดต่างๆ" },
      { label: "ข", text: "มีคู่มือรายละเอียดรายการและจดนำวิธีการใช้" },
      { label: "ค", text: "มีแผนภาพแสดงการทำงานของกล้องแยกทีละส่วน" },
      { label: "ง", text: "มีตัวอย่างเปรียบเทียบให้เห็นข้อดีข้อเสียและการปรับปรุงแก้ไข" }
    ],
    scoring: { ก: "A", ข: "R", ค: "V", ง: "K" }
  },
  {
    text: "คุณชอบผู้สอนหรือผู้นำเสนอที่ใช้วิธีการ",
    options: [
      { label: "ก", text: "สาธิต หุ่นจำลอง หรือมีช่วงเวลาให้ฝึกปฏิบัติ" },
      { label: "ข", text: "ถามและตอบข้อซักถาม พูดคุย อภิปรายกลุ่ม หรือ เชิญวิทยากรภายนอกมาร่วม" },
      { label: "ค", text: "แจกเอกสาร หนังสือ หรือ บทความต่างๆ ให้อ่าน" },
      { label: "ง", text: "ใช้แผนภาพ แผนภูมิ กราฟ ประกอบ" }
    ],
    scoring: { ก: "K", ข: "A", ค: "R", ง: "V" }
  },
  {
    text: "เมื่อเสร็จสิ้นจากการแข่งขันหรือการทดสอบและคุณต้องการอยากจะทราบผลย้อนกลับ คุณอยากได้ผลย้อนกลับในลักษณะ",
    options: [
      { label: "ก", text: "ใช้ตัวอย่างจากสิ่งที่คุณได้ทำไปแล้ว" },
      { label: "ข", text: "บรรยายผลของการทดสอบหรือการแข่งขันของคุณ" },
      { label: "ค", text: "ผู้ให้ผลย้อนกลับพูดคุยกับคุณเป็นส่วนตัว" },
      { label: "ง", text: "ใช้รูปแบบของกราฟแสดงผลสัมฤทธิ์ที่คุณทำได้" }
    ],
    scoring: { ก: "K", ข: "R", ค: "A", ง: "V" }
  },
  {
    text: "คุณจะเลือกสั่งอาหารในภัตตาคารหรือร้านกาแฟ คุณจะ",
    options: [
      { label: "ก", text: "เลือกสั่งที่คุณเคยสั่งมาก่อน" },
      { label: "ข", text: "ฟังคำแนะนำจากบริกรหรือขอให้เพื่อนแนะนำการเลือก" },
      { label: "ค", text: "เลือกสั่งจากคำอธิบายในรายการอาหาร" },
      { label: "ง", text: "ดูว่าคนอื่นๆ กำลังรับประทานอะไร หรือดูจากภาพตัวอย่างของรายการอาหารแต่ละจาน" }
    ],
    scoring: { ก: "K", ข: "A", ค: "R", ง: "V" }
  },
  {
    text: "คุณต้องกล่าวสุนทรพจน์พิเศษในงานประชุมสำคัญ คุณจะ",
    options: [
      { label: "ก", text: "จัดทำแผนภูมิหรือกราฟ เพื่อช่วยอธิบายสิ่งต่างๆ" },
      { label: "ข", text: "เขียนเฉพาะคำสำคัญๆ และฝึกกล่าวสุนทรพจน์จนคล่อง" },
      { label: "ค", text: "เขียนสุนทรพจน์และจดจำจากการอ่านซ้ำไปซ้ำมาหลายๆ หน" },
      { label: "ง", text: "พยายามหาตัวอย่าง หรือเรื่องราว ประกอบเพื่อให้การพูดดูเป็นเรื่องจริง และนำไปใช้ประโยชน์ได้" }
    ],
    scoring: { ก: "V", ข: "A", ค: "R", ง: "K" }
  }
];

// ── VARK Question Bank #2 (Optional, for averaging accuracy) ─
const BANK_2 = [
  {
    text: "เมื่อต้องเลือกสายอาชีพหรือสาขาวิชาที่เรียน สิ่งเหล่านี้สำคัญสำหรับคุณ",
    options: [
      { label: "ก", text: "การสื่อสารกับผู้อื่นผ่านการพูดคุยถกเถียง" },
      { label: "ข", text: "การทำงานกับงานออกแบบ แผนที่ หรือแผนภูมิ" },
      { label: "ค", text: "การนำความรู้ไปประยุกต์ใช้ในสถานการณ์จริง" },
      { label: "ง", text: "การใช้คำพูดได้ดีในการสื่อสารผ่านการเขียน" }
    ],
    scoring: { ก: "A", ข: "V", ค: "K", ง: "R" }
  },
  {
    text: "คุณชอบผู้สอนหรือวิทยากรที่ใช้",
    options: [
      { label: "ก", text: "เอกสารประกอบการสอน หนังสือ หรือบทอ่าน" },
      { label: "ข", text: "การสาธิต แบบจำลอง หรือการฝึกปฏิบัติจริง" },
      { label: "ค", text: "ไดอะแกรม แผนภูมิ แผนที่ หรือกราฟ" },
      { label: "ง", text: "การถามตอบ การพูดคุย การอภิปรายกลุ่ม หรือวิทยากรรับเชิญ" }
    ],
    scoring: { ก: "R", ข: "K", ค: "V", ง: "A" }
  },
  {
    text: "เมื่อคุณกำลังเรียนรู้ คุณ",
    options: [
      { label: "ก", text: "ชอบพูดคุยรายละเอียดให้เข้าใจทะลุปรุโปร่ง" },
      { label: "ข", text: "อ่านหนังสือ บทความ และเอกสารประกอบการสอน" },
      { label: "ค", text: "ใช้ตัวอย่างและการนำไปประยุกต์ใช้จริง" },
      { label: "ง", text: "มองเห็นรูปแบบ (Patterns) ในสิ่งต่างๆ" }
    ],
    scoring: { ก: "A", ข: "R", ค: "K", ง: "V" }
  },
  {
    text: "เมื่อเรียนรู้จากอินเทอร์เน็ต คุณชอบ",
    options: [
      { label: "ก", text: "บทความที่มีรายละเอียดครบถ้วน" },
      { label: "ข", text: "พอดแคสต์และวิดีโอที่คุณสามารถฟังผู้เชี่ยวชาญพูดได้" },
      { label: "ค", text: "วิดีโอที่แสดงวิธีการลงมือทำสิ่งต่างๆ" },
      { label: "ง", text: "งานออกแบบและฟีเจอร์ที่น่าสนใจทางสายตา" }
    ],
    scoring: { ก: "R", ข: "A", ค: "K", ง: "V" }
  },
  {
    text: "หากคุณมีปัญหาในการประกอบเฟอร์นิเจอร์ที่มาเป็นชิ้นส่วน คุณจะ",
    options: [
      { label: "ก", text: "กลับไปดูไดอะแกรมขั้นตอนการประกอบอีกครั้งเพื่อดูว่าพลาดอะไรไป" },
      { label: "ข", text: "ขอคำแนะนำหรือความช่วยเหลือจากคนอื่น" },
      { label: "ค", text: "กลับไปอ่านคำแนะนำที่เป็นตัวอักษรทีละขั้นตอนอีกครั้งเพื่อดูว่าพลาดอะไรไป" },
      { label: "ง", text: "ลองจัดวางชิ้นส่วนต่างๆ เพื่อดูว่ามันประกอบเข้ากันได้อย่างไร" }
    ],
    scoring: { ก: "V", ข: "A", ค: "R", ง: "K" }
  },
  {
    text: "คุณกำลังจัดทำประวัติศาสตร์ของพื้นที่ที่คุณอาศัยอยู่ คุณจะ",
    options: [
      { label: "ก", text: "รวบรวมแผนที่และแผนภูมิเก่าๆ" },
      { label: "ข", text: "อ่านบทความและข้อมูลอื่นๆ ในหนังสือพิมพ์เก่าและเอกสารต่างๆ" },
      { label: "ค", text: "บันทึกเรื่องราวจากผู้คนที่เล่าเรื่องราวในอดีต" },
      { label: "ง", text: "เปรียบเทียบภาพถ่ายทางประวัติศาสตร์ของพื้นที่กับสภาพปัจจุบัน" }
    ],
    scoring: { ก: "V", ข: "R", ค: "A", ง: "K" }
  },
  {
    text: "คุณต้องการแน่ใจว่าทำท่ากายบริหารได้ถูกต้อง คุณจะ",
    options: [
      { label: "ก", text: "เปรียบเทียบสิ่งที่คุณทำกับการสาธิตในวิดีโอ" },
      { label: "ข", text: "ศึกษาไดอะแกรมที่แสดงวิธีการทำท่ากายบริหารที่ถูกต้อง" },
      { label: "ค", text: "ตรวจสอบรายการหัวข้อสำคัญที่ต้องทำให้ถูกต้อง" },
      { label: "ง", text: "ฟังคำอธิบายเกี่ยวกับวิธีการทำท่ากายบริหารนั้นๆ" }
    ],
    scoring: { ก: "K", ข: "V", ค: "R", ง: "A" }
  },
  {
    text: "เมื่อคุณเสร็จสิ้นการแข่งขันหรือการทดสอบ และคุณต้องการคำแนะนำติชม (Feedback)",
    options: [
      { label: "ก", text: "โดยใช้คำบรรยายผลลัพธ์ของคุณเป็นลายลักษณ์อักษร" },
      { label: "ข", text: "โดยใช้ตัวอย่างจากสิ่งที่คุณได้ทำไป" },
      { label: "ค", text: "โดยใช้กราฟแสดงให้เห็นว่าผลงานของคุณพัฒนาขึ้นอย่างไร" },
      { label: "ง", text: "จากใครสักคนที่มาพูดคุยรายละเอียดกับคุณ" }
    ],
    scoring: { ก: "R", ข: "K", ค: "V", ง: "A" }
  },
  {
    text: "คุณต้องการหาข้อมูลเกี่ยวกับที่พัก ก่อนจะไปเยี่ยมชม คุณต้องการ",
    options: [
      { label: "ก", text: "ดูวิดีโอของสถานที่นั้น" },
      { label: "ข", text: "พูดคุยกับเจ้าของหรือผู้จัดการ" },
      { label: "ค", text: "ดูผังห้องและแผนที่ของพื้นที่รอบๆ" },
      { label: "ง", text: "อ่านรายละเอียดที่เป็นตัวพิมพ์ของห้องและฟีเจอร์ต่างๆ" }
    ],
    scoring: { ก: "K", ข: "A", ค: "V", ง: "R" }
  },
  {
    text: "คุณต้องการออมเงินเพิ่มขึ้นและต้องตัดสินใจเลือกระหว่างตัวเลือกต่างๆ คุณจะ",
    options: [
      { label: "ก", text: "ใช้กราฟแสดงตัวเลือกที่แตกต่างกันในช่วงเวลาต่างๆ" },
      { label: "ข", text: "อ่านโบรชัวร์ที่อธิบายตัวเลือกอย่างละเอียด" },
      { label: "ค", text: "พิจารณาตัวอย่างของแต่ละตัวเลือกโดยใช้ข้อมูลทางการเงินของคุณเอง" },
      { label: "ง", text: "พูดคุยกับผู้เชี่ยวชาญเกี่ยวกับตัวเลือกเหล่านั้น" }
    ],
    scoring: { ก: "V", ข: "R", ค: "K", ง: "A" }
  },
  {
    text: "คุณต้องการเรียนรู้วิธีการถ่ายภาพให้ดีขึ้น คุณจะ",
    options: [
      { label: "ก", text: "ใช้ไดอะแกรมแสดงส่วนประกอบของกล้องและหน้าที่ของแต่ละส่วน" },
      { label: "ข", text: "ถามคำถามและพูดคุยเกี่ยวกับกล้องและฟีเจอร์ต่างๆ" },
      { label: "ค", text: "ใช้คู่มือการใช้งานที่เป็นตัวอักษร" },
      { label: "ง", text: "ใช้ตัวอย่างภาพที่ดีและไม่ดีเพื่อดูวิธีการปรับปรุง" }
    ],
    scoring: { ก: "V", ข: "A", ค: "R", ง: "K" }
  },
  {
    text: "คุณต้องการเรียนรู้เกี่ยวกับโครงการใหม่ คุณจะขอ",
    options: [
      { label: "ก", text: "ไดอะแกรมแสดงขั้นตอนของโครงการพร้อมแผนภูมิแสดงประโยชน์และต้นทุน" },
      { label: "ข", text: "โอกาสในการพูดคุยถกเถียงเกี่ยวกับโครงการ" },
      { label: "ค", text: "ตัวอย่างกรณีที่โครงการนี้ถูกนำไปใช้จริงจนประสบความสำเร็จ" },
      { label: "ง", text: "รายงานที่เป็นลายลักษณ์อักษรอธิบายคุณลักษณะหลักของโครงการ" }
    ],
    scoring: { ก: "V", ข: "A", ค: "K", ง: "R" }
  },
  {
    text: "เว็บไซต์หนึ่งมีวิดีโอสอนการทำกราฟหรือแผนภูมิพิเศษ มีคนกำลังพูด มีรายการข้อความ และมีไดอะแกรม คุณจะเรียนรู้ได้มากที่สุดจาก",
    options: [
      { label: "ก", text: "การดูไดอะแกรม" },
      { label: "ข", text: "การฟัง" },
      { label: "ค", text: "การอ่านตัวหนังสือ" },
      { label: "ง", text: "การดูการลงมือทำ" }
    ],
    scoring: { ก: "V", ข: "A", ค: "R", ง: "K" }
  },
  {
    text: "คุณต้องการหาข้อมูลเพิ่มเติมเกี่ยวกับทัวร์ที่คุณกำลังจะไป คุณจะ",
    options: [
      { label: "ก", text: "ใช้แผนที่เพื่อดูว่าแต่ละสถานที่ตั้งอยู่ตรงไหน" },
      { label: "ข", text: "ดูรายละเอียดเกี่ยวกับจุดเด่นและกิจกรรมในทัวร์" },
      { label: "ค", text: "พูดคุยกับคนวางแผนทัวร์หรือคนอื่นๆ ที่จะไปทัวร์นี้ด้วยกัน" },
      { label: "ง", text: "อ่านรายละเอียดของทัวร์ในกำหนดการเดินทาง" }
    ],
    scoring: { ก: "V", ข: "K", ค: "A", ง: "R" }
  },
  {
    text: "คุณต้องการเรียนรู้วิธีการเล่นบอร์ดเกมหรือเกมไพ่ใหม่ๆ คุณจะ",
    options: [
      { label: "ก", text: "อ่านกติกาการเล่น" },
      { label: "ข", text: "ฟังใครสักคนอธิบายและถามคำถาม" },
      { label: "ค", text: "ใช้ไดอะแกรมที่อธิบายขั้นตอนการเล่น การเคลื่อนที่ และกลยุทธ์ต่างๆ" },
      { label: "ง", text: "ดูคนอื่นเล่นก่อนที่จะเข้าร่วมเล่นด้วย" }
    ],
    scoring: { ก: "R", ข: "A", ค: "V", ง: "K" }
  },
  {
    text: "คุณต้องการเรียนรู้วิธีการทำสิ่งใหม่ๆ บนคอมพิวเตอร์ คุณจะ",
    options: [
      { label: "ก", text: "ทำตามไดอะแกรมในหนังสือ" },
      { label: "ข", text: "พูดคุยกับคนที่เชี่ยวชาญโปรแกรมนั้น" },
      { label: "ค", text: "เริ่มใช้งานเลยและเรียนรู้ผ่านการลองผิดลองถูก" },
      { label: "ง", text: "อ่านคู่มือการใช้งานที่มาพร้อมกับโปรแกรม" }
    ],
    scoring: { ก: "V", ข: "A", ค: "K", ง: "R" }
  }
];

// Active question set — switches between BANK_1 and BANK_2 (Phase 4)
let questions = BANK_1;

// ── VARK Metadata ────────────────────────────────────────────
const VARK_LABELS = {
  V: { name: "Visual",      thai: "การมองเห็น", color: "#6c8fff", desc: "เรียนรู้ผ่านภาพ ไดอะแกรม และกราฟ" },
  A: { name: "Aural",       thai: "การได้ยิน",  color: "#ff7c5c", desc: "เรียนรู้ผ่านการฟังและการพูดคุย" },
  R: { name: "Read/Write",  thai: "การอ่าน/เขียน", color: "#5cd6b0", desc: "เรียนรู้ผ่านการอ่านและเขียน" },
  K: { name: "Kinesthetic", thai: "การลงมือทำ", color: "#f5c842", desc: "เรียนรู้ผ่านประสบการณ์และการปฏิบัติ" }
};

// ── State ────────────────────────────────────────────────────
let currentQ = 0;
let answers  = new Array(16).fill(null).map(() => new Set());

// Phase 4: bank tracking — bank 1 first, bank 2 optional, average pct across completed banks
let currentBankIdx = 0;       // 0 = BANK_1, 1 = BANK_2
let bankResults    = [];      // per-bank: { raw, pct }

// ── Persist bank state — รอด reload/bfcache เพื่อให้ Bank 2 เฉลี่ยกับ Bank 1 ได้แม้หน้าจะถูก reload
function _saveBankState() {
  try {
    sessionStorage.setItem('__vark_bank_state__', JSON.stringify({
      idx:     currentBankIdx,
      results: bankResults,
    }));
  } catch (e) {}
}
function _loadBankState() {
  try {
    const raw = sessionStorage.getItem('__vark_bank_state__');
    if (!raw) return;
    const s = JSON.parse(raw);
    if (typeof s.idx === 'number')   currentBankIdx = s.idx;
    if (Array.isArray(s.results))    bankResults    = s.results;
  } catch (e) {}
}
function _clearBankState() {
  try { sessionStorage.removeItem('__vark_bank_state__'); } catch (e) {}
}
_loadBankState();

// ── Screen Management ────────────────────────────────────────
function startQuiz() {
  // เริ่มจาก intro = ทำใหม่ตั้งแต่ Bank 1 (รีเซ็ต state ที่อาจค้างจาก reload)
  currentBankIdx = 0;
  bankResults    = [];
  questions      = BANK_1;
  currentQ       = 0;
  answers        = new Array(16).fill(null).map(() => new Set());
  _clearBankState();

  showScreen('quiz-screen');
  StepBar.setStep(1);
  StepBar.initQuestionCells(16);
  Narrator.quizStart();
  renderQuestion();
}

function showScreen(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
}

// ── Render Question ──────────────────────────────────────────
function renderQuestion() {
  const q   = questions[currentQ];
  const pct = (currentQ / 16) * 100;

  document.getElementById('progress-fill').style.width = pct + '%';
  document.getElementById('cur-q').textContent         = currentQ + 1;
  document.getElementById('q-number').textContent      = `ข้อที่ ${String(currentQ + 1).padStart(2, '0')} / 16`;
  document.getElementById('q-text').textContent        = q.text;

  // Macro chevron + narrator
  StepBar.setQuestionProgress(currentQ, answers);
  Narrator.quizMid(currentQ);

  const optList = document.getElementById('options-list');
  optList.innerHTML = '';

  q.options.forEach(opt => {
    const div       = document.createElement('div');
    div.className   = 'option' + (answers[currentQ].has(opt.label) ? ' selected' : '');
    div.innerHTML   = `
      <div class="opt-label">${opt.label}</div>
      <div class="opt-text">${opt.text}</div>
      <svg class="opt-check" viewBox="0 0 18 18" fill="none">
        <circle cx="9" cy="9" r="8.5" stroke="#6c8fff" stroke-width="1.5"/>
        <path d="M5 9l3 3 5-6" stroke="#6c8fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>`;
    div.addEventListener('click', () => toggleOption(opt.label, div));
    optList.appendChild(div);
  });

  // Re-trigger animation
  const card = document.getElementById('question-card');
  card.style.animation = 'none';
  card.offsetHeight;
  card.style.animation = 'slideIn 0.35s cubic-bezier(0.4,0,0.2,1)';

  // Back button: always enabled — Q1 returns to intro (Bank 1) or result (Bank 2)
  const backBtn = document.getElementById('btn-back');
  backBtn.disabled    = false;
  backBtn.textContent = currentQ === 0
    ? (currentBankIdx === 0 ? '← กลับหน้าแรก' : '← กลับผลลัพธ์')
    : '← ย้อนกลับ';
  document.getElementById('btn-next').textContent = currentQ === 15 ? 'ดูผลลัพธ์ 🎉' : 'ถัดไป →';

  // ── TTS: อ่านโจทย์และตัวเลือกหลัง animation เล็กน้อย ──
  // Track timer so showResults() / TTS.stop() can cancel it before it fires
  if (TTS._readTimer) clearTimeout(TTS._readTimer);
  TTS._readTimer = setTimeout(() => {
    TTS._readTimer = null;
    TTS.readCurrentQuestion();
  }, 380);
}

// ── Option Toggle ────────────────────────────────────────────
function toggleOption(label, el) {
  const set = answers[currentQ];
  if (set.has(label)) {
    set.delete(label);
    el.classList.remove('selected');
  } else {
    set.add(label);
    el.classList.add('selected');
  }
}

// ── Navigation ───────────────────────────────────────────────
function nextQuestion() {
  if (currentQ < 15) { currentQ++; renderQuestion(); }
  else               { showResults(); }
}

function prevQuestion() {
  if (currentQ > 0) {
    currentQ--;
    renderQuestion();
    return;
  }
  // From Q1: Bank 1 → intro, Bank 2 → result (Bank 1's result still preserved)
  TTS.stop();
  if (currentBankIdx === 0) {
    showScreen('intro-screen');
    Narrator.intro();
  } else {
    showScreen('result-screen');
  }
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Scoring ──────────────────────────────────────────────────
function calcScores() {
  const counts = { V: 0, A: 0, R: 0, K: 0 };
  questions.forEach((q, i) => {
    answers[i].forEach(label => {
      const cat = q.scoring[label];
      if (cat) counts[cat]++;
    });
  });
  return counts;
}

// ── Show Results ─────────────────────────────────────────────
function showResults() {
  TTS.stop(); // หยุดอ่านเมื่อจบแบบสอบถาม
  const raw   = calcScores();
  const total = Object.values(raw).reduce((a, b) => a + b, 0) || 1;
  const pct   = {};
  for (const k in raw) pct[k] = parseFloat(((raw[k] / total) * 100).toFixed(1));

  // Phase 4: store this bank's standalone result, then merge across all completed banks
  bankResults[currentBankIdx] = { raw, pct };
  _saveBankState();
  const completed = bankResults.filter(Boolean);
  const banksDone = completed.length;

  // Display values: average pct across completed banks; raw is summed for "X ข้อ" display
  let displayRaw = { ...raw };
  let displayPct = { ...pct };
  if (banksDone >= 2) {
    displayRaw  = { V: 0, A: 0, R: 0, K: 0 };
    const sumP  = { V: 0, A: 0, R: 0, K: 0 };
    completed.forEach(b => {
      ['V','A','R','K'].forEach(k => {
        displayRaw[k] += b.raw[k];
        sumP[k]       += b.pct[k];
      });
    });
    displayPct = {};
    ['V','A','R','K'].forEach(k => {
      displayPct[k] = parseFloat((sumP[k] / banksDone).toFixed(1));
    });
  }

  // Multi-modality detection on the *displayed* (possibly averaged) percentages
  const maxPct    = Math.max(...Object.values(displayPct));
  const dominants = ['V','A','R','K'].filter(k => displayPct[k] === maxPct && maxPct > 0);
  const isMulti   = dominants.length > 1;

  // Bank-progress badge + Bank 2 CTA
  const badge = document.getElementById('bank-progress-badge');
  if (badge) {
    badge.textContent = banksDone >= 2
      ? '📊 ผลจาก 2/2 ชุดคำถาม (ค่าเฉลี่ย — แม่นยำสูง)'
      : `📊 ผลจาก ${banksDone}/2 ชุดคำถาม`;
  }
  const bank2cta = document.getElementById('bank2-cta');
  if (bank2cta) bank2cta.style.display = banksDone >= 2 ? 'none' : '';

  const display = document.getElementById('dominant-display');
  if (dominants.length === 0) {
    display.innerHTML = `<p style="color:var(--text-muted); margin:12px 0;">ยังไม่ได้ตอบคำถามเลย</p>`;
  } else {
    const badges = dominants.map(k => {
      const inf = VARK_LABELS[k];
      return `<div class="dominant-badge" style="background:${inf.color}22; border:1.5px solid ${inf.color}55; color:${inf.color}; margin:0;">
        ${inf.name} · ${inf.thai} <span style="opacity:0.6; font-weight:500; font-size:0.85rem;">${displayPct[k]}%</span>
      </div>`;
    }).join('');
    const subline = isMulti
      ? `<p style="font-size:0.88rem; color:var(--text-muted); margin-top:10px;"><strong style="color:var(--accent-v);">สไตล์ผสม (Multimodal)</strong> — คุณเรียนรู้ได้ดีหลายช่องทางพร้อมกัน</p>`
      : `<p style="font-size:0.88rem; color:var(--text-muted); margin-top:4px;">${VARK_LABELS[dominants[0]].desc}</p>`;
    display.innerHTML = `
      <div style="display:flex; flex-wrap:wrap; gap:10px; justify-content:center; margin:12px auto;">${badges}</div>
      ${subline}`;
  }

  // Score bars (use displayRaw/displayPct so averaging is reflected)
  const grid = document.getElementById('scores-grid');
  grid.innerHTML = '';
  ['V','A','R','K'].forEach(cat => {
    const inf = VARK_LABELS[cat];
    const row = document.createElement('div');
    row.className = 'score-row';
    row.innerHTML = `
      <div class="score-row-top">
        <div class="score-name">
          <div class="score-dot" style="background:${inf.color}"></div>
          <span class="score-label">${inf.name}</span>
          <span class="score-sub">${inf.thai}</span>
        </div>
        <div class="score-nums"><strong>${displayPct[cat]}%</strong> (${displayRaw[cat]} ข้อ)</div>
      </div>
      <div class="score-bar-bg">
        <div class="score-bar-fill" style="background:${inf.color};" data-pct="${displayPct[cat]}"></div>
      </div>`;
    grid.appendChild(row);
  });

  // Animate bars
  setTimeout(() => {
    document.querySelectorAll('.score-bar-fill').forEach(el => {
      el.style.width = el.dataset.pct + '%';
    });
  }, 100);

  // Update next-step CTA label
  const nextLbl = document.getElementById('next-style-label');
  if (nextLbl) {
    nextLbl.textContent = dominants.length
      ? dominants.map(k => VARK_LABELS[k].name).join(' + ')
      : 'VARK';
  }

  // ── บันทึกข้อมูลเบื้องหลัง (ใช้ค่าเฉลี่ย ถ้าทำครบ 2 ชุด) ──
  _saveTrainingData({ raw: displayRaw, pct: displayPct, dominants, banksDone });

  // Macro chevron + narrator
  StepBar.setStep(2);
  StepBar.setVarkResult(displayRaw);
  const narratorLabel = dominants.length
    ? dominants.map(k => VARK_LABELS[k].name).join(' + ')
    : 'ยังไม่มีผล';
  Narrator.result(narratorLabel, isMulti, banksDone);

  showScreen('result-screen');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Internal: Store Training Data (hidden) ───────────────────
function _saveTrainingData({ raw, pct, dominants, banksDone }) {
  const record = {
    session_id:    Date.now(),
    timestamp:     new Date().toISOString(),
    vark_scores:   raw,
    vark_percent:  pct,
    dominant_style: dominants[0] || null,  // backward-compat: study.html legacy field
    dominants:     dominants,               // Phase 2: full list of co-dominant styles
    is_multimodal: dominants.length > 1,
    banks_done:    banksDone || 1,          // Phase 4: 1 = bank-1 only, 2 = averaged across both
    answers:       answers.map(s => [...s])
  };
  

  // Store in sessionStorage (ไม่แสดงใน UI)
  const existing = JSON.parse(sessionStorage.getItem('__vark_train__') || '[]');
  existing.push(record);
  sessionStorage.setItem('__vark_train__', JSON.stringify(existing));

  // Persist latest result so study.html survives refresh / bfcache / late navigation
  try { localStorage.setItem('__vark_latest__', JSON.stringify(record)); } catch (e) {}

  // Also expose on window for backend integration if needed
  window.__varkSession = record;
}

// ── Public helper for backend retrieval ─────────────────────
function getTrainingData() {
  return JSON.parse(sessionStorage.getItem('__vark_train__') || '[]');
}

// ── Navigate to Personalized Study page ─────────────────────
function goToStudy() {
  // ส่ง VARK result ผ่าน sessionStorage; fallback localStorage ถ้า window state ถูกล้าง
  const data = window.__varkSession
    || JSON.parse(localStorage.getItem('__vark_latest__') || 'null');
  if (data) sessionStorage.setItem('__vark_current__', JSON.stringify(data));
  window.location.href = 'study.html';
}

// ── Restart ──────────────────────────────────────────────────
function restartQuiz() {
  TTS.stop();
  currentBankIdx = 0;
  questions      = BANK_1;
  bankResults    = [];
  _clearBankState();
  currentQ       = 0;
  answers        = new Array(16).fill(null).map(() => new Set());
  // Reset chevron VARK indicators
  document.querySelectorAll('.step-cell.is-on').forEach(c => c.classList.remove('is-on'));
  StepBar.setStep(1);
  Narrator.quizStart();
  showScreen('quiz-screen');
  renderQuestion();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Start Bank 2 (Optional, after Bank 1 result) ─────────────
function startBank2() {
  if (currentBankIdx >= 1) return;     // already on / past bank 2
  TTS.stop();
  currentBankIdx = 1;
  _saveBankState();
  questions      = BANK_2;
  currentQ       = 0;
  answers        = new Array(16).fill(null).map(() => new Set());
  // Reset chevron VARK indicators (Step 2 dots) — back to Step 1
  document.querySelectorAll('.step-cell.is-on').forEach(c => c.classList.remove('is-on'));
  StepBar.setStep(1);
  StepBar.setQuestionProgress(0, answers);
  Narrator.bank2Start();
  showScreen('quiz-screen');
  renderQuestion();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Toast ────────────────────────────────────────────────────
function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2500);
}

// ══════════════════════════════════════════════════════════════
//  Chevron Step Bar — macro 3-step progress
// ══════════════════════════════════════════════════════════════
const StepBar = {
  setStep(n) {
    document.querySelectorAll('.step-chev').forEach(el => {
      const s = Number(el.dataset.step);
      el.classList.toggle('is-active', s === n);
      el.classList.toggle('is-done',   s <  n);
      el.classList.toggle('is-future', s >  n);
      const numEl = el.querySelector('.step-num');
      if (numEl) numEl.textContent = (s < n) ? '✓' : numEl.dataset.num || s;
    });
  },
  initQuestionCells(total = 16) {
    const cells = document.getElementById('step1-cells');
    if (!cells || cells.dataset.inited) return;
    cells.innerHTML = '';
    for (let i = 0; i < total; i++) {
      const c = document.createElement('span');
      c.className = 'step-cell';
      cells.appendChild(c);
    }
    cells.dataset.inited = '1';
  },
  setQuestionProgress(currentIdx, answersArr) {
    const cells = document.querySelectorAll('#step1-cells .step-cell');
    cells.forEach((c, i) => {
      const answered = answersArr && answersArr[i] && answersArr[i].size > 0;
      c.classList.toggle('is-filled', answered && i !== currentIdx);
      c.classList.toggle('is-current', i === currentIdx);
    });
  },
  setVarkResult(scores) {
    const total = Object.values(scores).reduce((a, b) => a + b, 0) || 1;
    ['V','A','R','K'].forEach(k => {
      const cell = document.querySelector(`.step-cell[data-vark="${k}"]`);
      if (!cell) return;
      cell.classList.toggle('is-on', (scores[k] / total) >= 0.20);
    });
  }
};

// ══════════════════════════════════════════════════════════════
//  Narrator — ครูแนะแนวส่วนตัว (guidance counselor copy)
// ══════════════════════════════════════════════════════════════
const Narrator = {
  set(html, opts) {
    const el = document.getElementById('narrator-text');
    if (el) el.innerHTML = html;
    if (opts && opts.speak) TTS.speakNarrator(el ? el.textContent : html);
  },
  intro() {
    this.set('เฮ้! ฉัน <strong>VARKy</strong> 🦉 จะพาคุณหาสไตล์การเรียนที่ใช่ — มี <strong>3 ขั้นตอน</strong> ทำแบบทดสอบ → ดูผลลัพธ์ VARK → เรียนกับ PDF ของคุณเอง', { speak: true });
  },
  quizStart() {
    // ไม่พากย์เสียง — VARKy พูดเฉพาะ intro กับ result
    this.set('เลือกคำตอบที่ตรงกับตัวคุณที่สุด <strong>เลือกได้หลายข้อในคำถามเดียว</strong> ถ้าไม่มีข้อไหนตรงเลย เว้นไว้ก็ได้!');
  },
  bank2Start() {
    this.set('ดีมาก! 🎯 <strong>ชุดที่ 2</strong> — คำถามต่างไปแต่หลักการเดิม อีก <strong>16 ข้อ</strong> ระบบจะเฉลี่ยผลทั้งสองชุดให้แม่นยำขึ้น');
  },
  quizMid(idx) {
    // กลางการทำแบบสอบถาม — ไม่พากย์เสียง เพราะจะชนกับเสียงโจทย์ (sound{n}.wav)
    const left = 16 - idx - 1;
    if (left <= 0)      this.set('ข้อสุดท้ายแล้ว! 💪 ตอบให้ตรงกับความรู้สึกจริง ๆ');
    else if (left <= 3) this.set(`เกือบจบแล้ว เหลืออีกแค่ <strong>${left + 1} ข้อ</strong> ลุย!`);
    else if (idx === 7) this.set('ทำมาครึ่งทางแล้ว 🎯 ต่อเลย!');
    else                this.set(`ตอนนี้อยู่ข้อ <strong>${idx + 1} / 16</strong> — ไม่มีถูกผิด ตอบตามความรู้สึกจริงได้เลย`);
  },
  result(label, isMulti, banksDone) {
    const accSuffix = banksDone >= 2 ? ' (ค่าเฉลี่ยจาก 2 ชุด)' : '';
    const tip       = banksDone < 2 ? ' อยากให้แม่นยำกว่านี้ ลองทำชุดที่ 2 ได้ 👇' : '';
    if (isMulti) {
      this.set(`ผลออกมาแล้ว! คุณมี <strong>สไตล์ผสม</strong>${accSuffix} — <strong>${label}</strong> ✨ คุณเรียนรู้ได้ดีหลายช่องทาง${tip}`, { speak: true });
    } else {
      this.set(`ผลออกมาแล้ว! สไตล์เด่นของคุณคือ <strong>${label}</strong>${accSuffix} ✨${tip}`, { speak: true });
    }
  }
};