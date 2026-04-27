// ============================================================
// VARK Quiz — Main Script
// ข้อมูล VARK ถูกบันทึกเบื้องหลังโดยไม่แสดงใน UI
// ============================================================

// ══════════════════════════════════════════════════════════════
//  TTS Engine — Gemini TTS (primary) + Web Speech fallback
//
//  Priority:
//    1. Gemini TTS  →  generativelanguage.googleapis.com  (คุณภาพสูงสุด)
//    2. Web Speech  →  speechSynthesis                    (fallback)
//
//  การตั้งค่า:
//    - ใส่ GEMINI_API_KEY ใน config.js  หรือ
//    - กำหนด window.GEMINI_API_KEY ก่อนโหลด script นี้
// ══════════════════════════════════════════════════════════════

const TTS = {
  enabled:  false,
  speaking: false,
  _audio:   null,   // Audio object ปัจจุบัน

  // ── toggle เปิด/ปิด ──────────────────────────────────────
  toggle() {
    this.enabled = !this.enabled;
    const btn  = document.getElementById('tts-toggle-btn');
    const icon = document.getElementById('tts-icon');
    const lbl  = document.getElementById('tts-label');

    if (this.enabled) {
      btn.classList.add('tts-on');
      icon.textContent = '🔊';
      lbl.textContent  = 'ปิดเสียงอ่านโจทย์';
      showToast('✨ เปิดเสียงอ่านโจทย์');
      const quiz = document.getElementById('quiz-screen');
      if (quiz && quiz.classList.contains('active')) this.readCurrentQuestion();
    } else {
      btn.classList.remove('tts-on');
      icon.textContent = '🔇';
      lbl.textContent  = 'เปิดเสียงอ่านโจทย์';
      this.stop();
      showToast('🔇 ปิดเสียงอ่านโจทย์แล้ว');
    }
  },

  // ── หยุดเสียงทันที ───────────────────────────────────────
  stop() {
    if (this._audio) {
      this._audio.pause();
      this._audio.src = '';
      this._audio = null;
    }
    this.speaking = false;
    this._setWave(false);
  },

  // ── เล่นไฟล์เสียงของข้อปัจจุบัน ─────────────────────────
  readCurrentQuestion() {
    if (!this.enabled) return;
    this.stop();

    const n     = currentQ + 1;                        // 1–16
    const url   = `VARKsound/sound${n}.wav`;
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

// ── VARK Question Bank ──────────────────────────────────────
const questions = [
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

// ── Screen Management ────────────────────────────────────────
function startQuiz() {
  showScreen('quiz-screen');
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

  document.getElementById('btn-back').disabled      = currentQ === 0;
  document.getElementById('btn-next').textContent   = currentQ === 15 ? 'ดูผลลัพธ์ 🎉' : 'ถัดไป →';

  // ── TTS: อ่านโจทย์และตัวเลือกหลัง animation เล็กน้อย ──
  setTimeout(() => TTS.readCurrentQuestion(), 380);
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
  if (currentQ > 0) { currentQ--; renderQuestion(); }
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

  // Dominant style
  const dominant = Object.entries(raw).sort((a, b) => b[1] - a[1])[0][0];
  const info      = VARK_LABELS[dominant];

  document.getElementById('dominant-display').innerHTML = `
    <div class="dominant-badge" style="background:${info.color}22; border:1.5px solid ${info.color}55; color:${info.color}; margin:12px auto; display:inline-flex;">
      ${info.name} · ${info.thai}
    </div>
    <p style="font-size:0.88rem; color:var(--text-muted); margin-top:4px;">${info.desc}</p>`;

  // Score bars
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
        <div class="score-nums"><strong>${pct[cat]}%</strong> (${raw[cat]} ข้อ)</div>
      </div>
      <div class="score-bar-bg">
        <div class="score-bar-fill" style="background:${inf.color};" data-pct="${pct[cat]}"></div>
      </div>`;
    grid.appendChild(row);
  });

  // Animate bars
  setTimeout(() => {
    document.querySelectorAll('.score-bar-fill').forEach(el => {
      el.style.width = el.dataset.pct + '%';
    });
  }, 100);

  // ── บันทึกข้อมูลเบื้องหลัง (ไม่แสดงใน UI) ──────────────
  _saveTrainingData({ raw, pct, dominant });

  showScreen('result-screen');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Internal: Store Training Data (hidden) ───────────────────
function _saveTrainingData({ raw, pct, dominant }) {
  const record = {
    session_id:    Date.now(),
    timestamp:     new Date().toISOString(),
    vark_scores:   raw,
    vark_percent:  pct,
    dominant_style: dominant,
    answers:       answers.map(s => [...s])
  };
  

  // Store in sessionStorage (ไม่แสดงใน UI)
  const existing = JSON.parse(sessionStorage.getItem('__vark_train__') || '[]');
  existing.push(record);
  sessionStorage.setItem('__vark_train__', JSON.stringify(existing));

  // Also expose on window for backend integration if needed
  window.__varkSession = record;
}

// ── Public helper for backend retrieval ─────────────────────
function getTrainingData() {
  return JSON.parse(sessionStorage.getItem('__vark_train__') || '[]');
}

// ── Navigate to Personalized Study page ─────────────────────
function goToStudy() {
  // ส่ง VARK result ไปหน้าถัดไปผ่าน sessionStorage
  const data = window.__varkSession;
  if (data) sessionStorage.setItem('__vark_current__', JSON.stringify(data));
  window.location.href = 'study.html';
}

// ── Restart ──────────────────────────────────────────────────
function restartQuiz() {
  TTS.stop();
  currentQ = 0;
  answers  = new Array(16).fill(null).map(() => new Set());
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