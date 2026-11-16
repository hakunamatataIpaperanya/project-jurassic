// Project Jurassic — ภาษา (ไทย / อังกฤษ) และธีม (มืด / สว่าง) ของหน้าเว็บ
//   L(th, en)   ข้อความในโค้ด: คืนภาษาที่เลือก
//   tr(th)      ค่าข้อมูลจากทะเบียน / โมเดล (ประเภทซาก กลุ่มซาก จังหวัด ส่วนของซาก) -> ภาษาที่เลือก (ไม่มีในตาราง = คงเดิม)
//   ข้อความคงที่ใน index.html ใช้แอตทริบิวต์ data-en (innerHTML), data-en-ph (placeholder), data-en-aria (aria-label)
//   ค่าที่เลือกจำไว้ในเบราว์เซอร์ (localStorage; ใช้ไม่ได้ = ใช้ค่าเริ่มต้น ไทย / มืด)
//   โหลดใน <head> เพื่อใส่ธีมก่อนหน้าเว็บแสดง (ไม่กะพริบจากมืดเป็นสว่าง); app.js เรียก I18N.apply() หลังสร้างหน้า
"use strict";

const PREFS = (() => {
  const get = (k, d) => { try { return localStorage.getItem(k) || d; } catch (e) { return d; } };
  const set = (k, v) => { try { localStorage.setItem(k, v); } catch (e) { /* ไม่มี storage */ } };
  return { get, set };
})();

const I18N = (() => {
  let lang = PREFS.get("pj-lang", "th") === "en" ? "en" : "th";
  const subs = [];

  // ---------------------------------------------------------------- ค่าข้อมูล: ไทย -> อังกฤษ
  const DATA = {
    // ประเภทซากในทะเบียน (f_type) และคลาสประเภทซากของแผนที่ (classes_B) / กลุ่มซาก (fl.other_group)
    "ไดโนเสาร์": "Dinosaur", "แบรคิโอพอด": "Brachiopod", "สัตว์กีบคู่": "Even-toed ungulate", "ปลากระดูกแข็ง": "Bony fish", "จระเข้": "Crocodile",
    "หอยกาบคู่/หอยสองฝา": "Bivalve", "ปลากระดูกอ่อน": "Cartilaginous fish", "แอนโธซัว": "Anthozoan (coral)", "เต่า": "Turtle", "เซฟาโลพอด": "Cephalopod",
    "วาฬ": "Whale", "หอยกาบเดี่ยว/หอยฝาเดียว": "Gastropod", "ไพรเมต": "Primate", "ไบรโอซัว": "Bryozoan", "สัตว์กินเนื้อ": "Carnivoran",
    "ฟอแรมมินิเฟอร์": "Foraminifera", "ซอโรพอด": "Sauropod", "ไทรโลไบต์": "Trilobite", "เอไคโนเดิร์ม": "Echinoderm", "เศษพืช": "Plant fragments",
    "ครัสตาเชียน": "Crustacean", "ฟอแรมินิเฟอร์": "Foraminifera", "รอยตีน": "Footprint", "ฉลาม": "Shark", "สัตว์เลี้ยงลูกด้วยนม": "Mammal",
    "แกรปโตไลต์": "Graptolite", "สัตว์กีบคี่": "Odd-toed ungulate", "สัตว์เลื้อยคลาน": "Reptile", "สัตว์มีกระดูกสันหลัง": "Vertebrate",
    "ใบไม้": "Leaf", "ไม้กลายเป็นหิน": "Petrified wood", "รอยชอนไช": "Burrow", "พอริเฟอรา": "Porifera (sponge)", "ปลา": "Fish",
    "พืชกลุ่มเฟิร์น": "Fern-like plant", "สาหร่าย": "Algae", "สัตว์งวง": "Proboscidean", "สัตว์ไม่มีกระดูกสันหลัง": "Invertebrate",
    "สัตว์ฟันแทะ": "Rodent", "เทนทาคิวไลต์": "Tentaculite", "กิ้งก่าและงู": "Lizard / snake", "ฟอแรมินิเฟอรา": "Foraminifera", "อื่น ๆ": "Other",
    "ไครนอยด์": "Crinoid", "รอยเจาะและชอนไช": "Boring / burrow", "ไฟโตซอร์": "Phytosaur", "มูลสัตว์": "Coprolite", "เฮกซาพอด": "Hexapod (insect)",
    "ไนดาเรีย": "Cnidarian", "ฟองน้ำ": "Sponge", "ฟอแรมมินิเฟอ": "Foraminifera", "เทบูเลต้า": "Tabulate coral", "หอยกาบคู่": "Bivalve",
    "เฟิร์น": "Fern", "บ่าง": "Colugo", "ค้างคาว": "Bat", "มวลสารพอก": "Concretion", "โปรบอสซิเดีย": "Proboscidean",
    "เทสทูดีเนต": "Testudinate (turtle)", "สกุล เฟรเนลอปซิส เอ": "Frenelopsis (genus) A", "สกุล เฟรเนลอปซิส บี": "Frenelopsis (genus) B",
    "เรดิโอลาเรีย": "Radiolarian", "สัตว์เลื้อยคลานอื่น": "Other reptile", "หอยสองฝา": "Bivalve", "หอยฝาเดียว": "Gastropod", "ปะการัง": "Coral",
    "ฟอแรม/ฟิวซูลินิด": "Foram / fusulinid", "ฟอแรม / ฟิวซูลินิด": "Foram / fusulinid", "พืช": "Plant", "ร่องรอย": "Trace fossil",
    "ประเภทอื่น ๆ": "Other type",
    // ส่วนของซาก (ปุ่มเลือกเร็ว)
    "ฟัน": "Tooth", "เศษกระดูก": "Bone fragment", "กระดูกสันหลัง": "Vertebra", "กระดูกซี่โครง": "Rib", "กระดูกหาง": "Tail bone",
    "เปลือกหอยสองฝา": "Bivalve shell", "เปลือกแบรคิโอพอด": "Brachiopod shell", "เกล็ดปลา": "Fish scale", "กระดองปู": "Crab carapace",
    // จังหวัด (ขอบเขตจังหวัดของ ทธ.)
    "กรุงเทพมหานคร": "Bangkok", "กระบี่": "Krabi", "กาญจนบุรี": "Kanchanaburi", "กาฬสินธุ์": "Kalasin", "กำแพงเพชร": "Kamphaeng Phet",
    "ขอนแก่น": "Khon Kaen", "จันทบุรี": "Chanthaburi", "ฉะเชิงเทรา": "Chachoengsao", "ชลบุรี": "Chon Buri", "ชัยนาท": "Chai Nat",
    "ชัยภูมิ": "Chaiyaphum", "ชุมพร": "Chumphon", "เชียงราย": "Chiang Rai", "เชียงใหม่": "Chiang Mai", "ตรัง": "Trang", "ตราด": "Trat",
    "ตาก": "Tak", "นครนายก": "Nakhon Nayok", "นครปฐม": "Nakhon Pathom", "นครพนม": "Nakhon Phanom", "นครราชสีมา": "Nakhon Ratchasima",
    "นครศรีธรรมราช": "Nakhon Si Thammarat", "นครสวรรค์": "Nakhon Sawan", "นนทบุรี": "Nonthaburi", "นราธิวาส": "Narathiwat", "น่าน": "Nan",
    "บึงกาฬ": "Bueng Kan", "บุรีรัมย์": "Buri Ram", "ปทุมธานี": "Pathum Thani", "ประจวบคีรีขันธ์": "Prachuap Khiri Khan", "ปราจีนบุรี": "Prachin Buri",
    "ปัตตานี": "Pattani", "พระนครศรีอยุธยา": "Phra Nakhon Si Ayutthaya", "พะเยา": "Phayao", "พังงา": "Phang Nga", "พัทลุง": "Phatthalung",
    "พิจิตร": "Phichit", "พิษณุโลก": "Phitsanulok", "เพชรบุรี": "Phetchaburi", "เพชรบูรณ์": "Phetchabun", "แพร่": "Phrae", "ภูเก็ต": "Phuket",
    "มหาสารคาม": "Maha Sarakham", "มุกดาหาร": "Mukdahan", "แม่ฮ่องสอน": "Mae Hong Son", "ยโสธร": "Yasothon", "ยะลา": "Yala",
    "ร้อยเอ็ด": "Roi Et", "ระนอง": "Ranong", "ระยอง": "Rayong", "ราชบุรี": "Ratchaburi", "ลพบุรี": "Lop Buri", "ลำปาง": "Lampang",
    "ลำพูน": "Lamphun", "เลย": "Loei", "ศรีสะเกษ": "Si Sa Ket", "สกลนคร": "Sakon Nakhon", "สงขลา": "Songkhla", "สตูล": "Satun",
    "สมุทรปราการ": "Samut Prakan", "สมุทรสงคราม": "Samut Songkhram", "สมุทรสาคร": "Samut Sakhon", "สระแก้ว": "Sa Kaeo", "สระบุรี": "Saraburi",
    "สิงห์บุรี": "Sing Buri", "สุโขทัย": "Sukhothai", "สุพรรณบุรี": "Suphan Buri", "สุราษฎร์ธานี": "Surat Thani", "สุรินทร์": "Surin",
    "หนองคาย": "Nong Khai", "หนองบัวลำภู": "Nong Bua Lam Phu", "อ่างทอง": "Ang Thong", "อำนาจเจริญ": "Amnat Charoen", "อุดรธานี": "Udon Thani",
    "อุตรดิตถ์": "Uttaradit", "อุทัยธานี": "Uthai Thani", "อุบลราชธานี": "Ubon Ratchathani",
  };
  const BACK = {};                                     // อังกฤษ (ตัวพิมพ์เล็ก) -> ไทย: จังหวัด / ส่วนของซากที่ผู้ใช้พิมพ์หรือเลือก สำหรับส่งให้โมเดล
  for (const [th, en] of Object.entries(DATA)) if (!(en.toLowerCase() in BACK)) BACK[en.toLowerCase()] = th;

  // ยุคธรณี (คีย์ภาษาอังกฤษ)
  const PERIOD = {
    Precambrian: ["พรีแคมเบรียน", "Precambrian"], Cambrian: ["แคมเบรียน", "Cambrian"], Ordovician: ["ออร์โดวิเชียน", "Ordovician"],
    Silurian: ["ไซลูเรียน", "Silurian"], Devonian: ["ดีโวเนียน", "Devonian"], Carboniferous: ["คาร์บอนิเฟอรัส", "Carboniferous"],
    Permian: ["เพอร์เมียน", "Permian"], Triassic: ["ไทรแอสซิก", "Triassic"], Jurassic: ["จูแรสซิก", "Jurassic"],
    Cretaceous: ["ครีเทเชียส", "Cretaceous"], Tertiary: ["เทอร์เชียรี", "Tertiary"], Quaternary: ["ควอเทอร์นารี", "Quaternary"],
  };

  // คำถาม "หน้าตา" (feature_lib.APPEARANCE) ตาม id ของการ์ด / ตัวเลือก
  const LOOK_GROUP = { "ส่วนของสัตว์มีกระดูกสันหลัง": "Vertebrate parts", "เปลือกและโครงแข็ง": "Shells and hard skeletons", "พืช": "Plants",
    "รอยที่สัตว์ทิ้งไว้": "Traces left by animals", "ไม่แน่ใจ": "Not sure" };
  const LOOK = {
    tooth: ["Tooth", "Shiny enamel", "What shape is the tooth?"], bone: ["Bone", "Porous inside when broken", "Which part of the skeleton?"],
    plate: ["Shell plate", "Hard plate with a pattern or pits"], fish: ["Fish", "Scales, a fish imprint or fish bones", "Which part of the fish can you see?"],
    twoshell: ["Two-valved shell", "Or a single curved valve like a clam", "Look at one valve from above"], coil: ["Coiled shell", "", "How is it coiled?"],
    cone: ["Rod or long cone", "", "What does it look like?"], trilobite: ["Insect-like, three lobes", "Three lengthwise parts, segmented"],
    crab: ["Crab shell / jointed legs", ""], grain: ["Spindle-shaped grains", "Like rice grains, about 0.5–1 cm long"],
    coral: ["Block with rows of openings", "Honeycomb, porous or mesh", "What are the openings like?"], stem: ["Round segmented stem", "Like a stack of coins"],
    graptolite: ["Saw-blade-like lines", "Fine lines on shale"], leaf: ["Leaf", "Or a leaf imprint"], wood: ["Wood", "Visible grain or growth rings"],
    track: ["Footprint", "A toe-shaped hollow"], burrow: ["Burrow / dung-like lump", "Tubes, holes or winding trails in rock"],
    unknown: ["None of these", "The system guesses from overall statistics (less accurate)"],
  };
  const LOOK_OPT = {
    "tooth/blade": ["Flat and sharp, triangular or blade-like", "Edges may be serrated"], "tooth/cone": ["Pointed cone, round in cross-section", "Like a crocodile tooth or fang"],
    "tooth/molar": ["Molar with grinding cusps or ridges", "Like cow, pig or monkey teeth"], "tooth/crusher": ["Low and flat, smooth or wrinkled", "Usually small, for crushing shells"],
    "bone/vertebra": ["Vertebra or tail bone", "Segmented, with knobs or wings"], "bone/limb": ["Leg, arm, finger or joint bone", "Long with flared ends"],
    "bone/rib": ["Rib", "Long, curved, flat"], "bone/skull": ["Jaw or skull", "May have tooth sockets"], "bone/frag": ["Small bone fragment", "Can't tell which part"],
    "fish/scales": ["Shiny scales", "Diamond-shaped or round"], "fish/body": ["Fish imprint or fish skeleton", "The fish shape is visible"],
    "fish/head": ["Fish skull or jaw", ""], "twoshell/sym": ["Left and right halves match, beak in the middle", "Often with ribs radiating from the beak"],
    "twoshell/asym": ["Beak tilted to one side", "Left and right halves differ"], "coil/flat": ["Coiled flat in one plane", "Like a coin or a ram's horn"],
    "coil/spire": ["Coiled into a pointed spiral", "Like a pond snail"], "cone/chambers": ["Divided into chambers along its length", "Visible when broken or polished"],
    "cone/rings": ["Small pointed cone with rings", "A few centimetres long"], "cone/spiral": ["Long spiral like a pagoda", ""],
    "coral/cells": ["Openings visible by eye, honeycomb or rows of tubes", ""], "coral/mesh": ["Fine mesh sheet like lace", ""],
    "coral/sponge": ["Irregular porous lump like a sponge", ""],
  };

  // ---------------------------------------------------------------- API
  const L = (th, en) => (lang === "en" ? en : th);
  const tr = (th) => (lang === "en" && th != null && DATA[th] ? DATA[th] : th);
  const toThai = (s) => (s != null && BACK[String(s).trim().toLowerCase()]) || s;   // ค่าที่ผู้ใช้ป้อนเป็นภาษาอังกฤษ -> ค่าที่โมเดลรู้จัก
  const en = (th) => (th != null && DATA[th]) || th;                         // ชื่ออังกฤษเสมอ (ไม่ขึ้นกับภาษาที่เลือก) เช่น ป้ายจังหวัดบนแผนที่
  const period = (key) => (PERIOD[key] ? PERIOD[key][lang === "en" ? 1 : 0] : key);
  const look = (card, field) => {                                             // field: 0 ชื่อ, 1 คำอธิบาย, 2 คำถามต่อ
    const v = LOOK[card];
    return lang === "en" && v ? v[field] ?? null : undefined;
  };
  const lookOpt = (card, opt, field) => {
    if (lang !== "en") return undefined;
    if (opt === "unsure") return field === 0 ? "Not sure" : null;
    const v = LOOK_OPT[`${card}/${opt}`];
    return v ? v[field] ?? null : undefined;
  };
  const lookGroup = (g) => (lang === "en" ? LOOK_GROUP[g] || g : g);

  function apply(root = document) {
    // ข้อความคงที่ใน HTML: เก็บต้นฉบับภาษาไทยไว้ครั้งแรก แล้วสลับตามภาษา
    root.querySelectorAll("[data-en]").forEach((el) => {
      if (el.dataset.th === undefined) el.dataset.th = el.innerHTML;
      el.innerHTML = lang === "en" ? el.dataset.en : el.dataset.th;
    });
    root.querySelectorAll("[data-en-ph]").forEach((el) => {
      if (el.dataset.thPh === undefined) el.dataset.thPh = el.getAttribute("placeholder") || "";
      el.setAttribute("placeholder", lang === "en" ? el.dataset.enPh : el.dataset.thPh);
    });
    root.querySelectorAll("[data-en-aria]").forEach((el) => {
      if (el.dataset.thAria === undefined) el.dataset.thAria = el.getAttribute("aria-label") || "";
      el.setAttribute("aria-label", lang === "en" ? el.dataset.enAria : el.dataset.thAria);
    });
    document.documentElement.lang = lang;
    document.title = L("Project Jurassic · แผนที่ซากดึกดำบรรพ์ประเทศไทย", "Project Jurassic · Thailand Fossil Atlas");
  }

  function set(l) {
    lang = l === "en" ? "en" : "th";
    PREFS.set("pj-lang", lang);
    apply();
    subs.forEach((fn) => fn(lang));
  }

  return { get lang() { return lang; }, set, L, tr, toThai, en, period, look, lookOpt, lookGroup, apply, onChange: (fn) => subs.push(fn) };
})();

const THEME = (() => {
  let theme = PREFS.get("pj-theme", "dark") === "light" ? "light" : "dark";
  const subs = [];
  const apply = () => { document.documentElement.dataset.theme = theme; };
  apply();
  return {
    get theme() { return theme; },
    set(t) { theme = t === "light" ? "light" : "dark"; PREFS.set("pj-theme", theme); apply(); subs.forEach((fn) => fn(theme)); },
    onChange: (fn) => subs.push(fn),
  };
})();

const L = I18N.L;
const tr = I18N.tr;
