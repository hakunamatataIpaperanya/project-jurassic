// Project Jurassic web app (frontend). "Operations terminal" layout with 4 screens (INDEX / MAP / IDENTIFY / MORE),
// written for general users: plain Thai (or English), levels instead of raw scores, technical details folded away.
// Language / theme switches: i18n.js (L(th, en), tr(th), THEME); every renderer reads the current language, relabel() / retheme() re-render.
// Data and live predictions come from /api/* (webapp/server.py → notebooks/predict_lib.py), all from the Project_Jurassic folder.
// Static site (Cloudflare Pages, scripts/build_static_site.py): the page carries <meta name="jurassic-static"> and api() hands every call
// to local.js instead (exported JSON + the same models as ONNX running in the browser).
"use strict";

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// เว็บแบบไม่มีเซิร์ฟเวอร์: ที่อยู่ของข้อมูล / โมเดลจาก <meta name="jurassic-static"> (ไม่มี = ใช้ server ตามเดิม)
const STATIC = (() => { const m = document.querySelector('meta[name="jurassic-static"]'); return m ? JSON.parse(m.content) : null; })();
if (STATIC) Object.assign(JurassicLocal.cfg, STATIC);

// ระดับโอกาส: เทียบกับกริด 5 กม. ทั้งประเทศ (เปอร์เซ็นไทล์)
//   สีตามธีม: มืด = ฟ้ากรมท่า -> ม่วง -> ส้ม -> เหลือง (ยิ่งสว่างยิ่งสูง), สว่าง = หมอก -> เขียว -> ส้ม -> ส้มแดง (ยิ่งเข้มยิ่งสูง)
//   label / text / color / ink เป็น getter (อ่านภาษาและธีมปัจจุบัน)
const PALETTE = {
  dark: { vhigh: ["#ffd86b", "#2a1a00"], high: ["#f29a3f", "#1d0e02"], mid: ["#7b5aa6", "#ffffff"], low: ["#23255a", "#dcd6ea"], none: ["#6e6b92", "#ffffff"] },
  light: { vhigh: ["#b4401c", "#ffffff"], high: ["#e58a3c", "#2b1405"], mid: ["#a7c48a", "#24301a"], low: ["#e2e6da", "#3a342b"], none: ["#8b8273", "#ffffff"] },
};
const level = (key, min, mapOpacity, th, en) => ({
  key, min, mapOpacity,
  get label() { return L(th[0], en[0]); }, get text() { return L(th[1], en[1]); },
  get color() { return PALETTE[THEME.theme][key][0]; }, get ink() { return PALETTE[THEME.theme][key][1]; },
});
const LEVELS = [
  level("vhigh", 90, 0.9, ["สูงมาก", "10% แรกของประเทศ"], ["Very high", "Top 10% of the country"]),
  level("high", 75, 0.75, ["สูง", "10–25% แรกของประเทศ"], ["High", "Top 10–25% of the country"]),
  level("mid", 50, 0.45, ["ปานกลาง", "สูงกว่าครึ่งหนึ่งของประเทศ"], ["Medium", "Above half of the country"]),
  level("low", 0, 0, ["ต่ำ", "ต่ำกว่าครึ่งหนึ่งของประเทศ"], ["Low", "Below half of the country"]),
];
const NONE = level("none", -1, 0, ["ข้อมูลไม่พอ", "ระบบไม่เคยเห็นพื้นที่ลักษณะนี้"], ["Not enough data", "The system has never seen an area like this"]);
const levelOf = (pct) => LEVELS.find((l) => pct >= l.min);
// วันที่: ไทย = พ.ศ. (เช่น 31 ก.ค. 2567), อังกฤษ = ค.ศ.
const fmtDate = (iso, month = false) => new Intl.DateTimeFormat(I18N.lang === "en" ? "en-GB" : "th-TH",
  month ? { month: "short", year: "numeric" } : { day: "numeric", month: "short", year: "numeric" }).format(new Date(`${iso}T00:00:00`));
const pill = (l) => `<span class="pill" style="background:${l.color};color:${l.ink}">${l.label}</span>`;

const TYPE_INFO = {
  "ไดโนเสาร์": ["🦕", "สัตว์เลื้อยคลานยุคมีโซโซอิก พบมากในภาคอีสาน", "Mesozoic reptiles, common in the Northeast (Isan)"],
  "สัตว์เลื้อยคลานอื่น": ["🐊", "เช่น จระเข้ เต่า", "e.g. crocodiles, turtles"],
  "ปลา": ["🐟", "ปลาโบราณ รวมทั้งฉลาม", "Ancient fish, including sharks"],
  "สัตว์เลี้ยงลูกด้วยนม": ["🦣", "เช่น ช้างโบราณ หมู กวาง ลิง มักพบในถ้ำ", "e.g. ancient elephants, pigs, deer, monkeys; often found in caves"],
  "แบรคิโอพอด": ["🐚", "หอยตะเกียง สัตว์ทะเลมีเปลือกสองฝา (ไม่ใช่หอยแท้)", "Lamp shells: two-valved marine animals (not true molluscs)"],
  "หอยสองฝา": ["🦪", "หอยกาบ หอยนางรม", "Clams and oysters"],
  "หอยฝาเดียว": ["🐌", "หอยโข่ง หอยขม", "Snails"],
  "เซฟาโลพอด": ["🦑", "หมึกโบราณมีเปลือก เช่น แอมโมไนต์", "Ancient shelled relatives of squid, e.g. ammonites"],
  "ปะการัง": ["🪸", "ปะการังโบราณในทะเลตื้น", "Ancient corals of shallow seas"],
  "ไครนอยด์": ["🌼", "พลับพลึงทะเล มักพบเป็นข้อกลม ๆ ในหินปูน", "Sea lilies, often found as round stem discs in limestone"],
  "ไบรโอซัว": ["🕸️", "สัตว์ทะเลขนาดเล็กอยู่เป็นกลุ่ม คล้ายตาข่าย", "Tiny colonial marine animals that look like netting"],
  "ไทรโลไบต์": ["🪲", "สัตว์ทะเลมีเปลือกแข็งเป็นปล้อง อายุเก่าแก่มาก", "Very ancient segmented marine animals with hard shells"],
  "แกรปโตไลต์": ["〰️", "สัตว์ทะเลอยู่เป็นกลุ่ม ทิ้งรอยคล้ายใบเลื่อยเล็ก ๆ บนหิน", "Colonial marine animals that leave tiny saw-blade marks on rock"],
  "ฟอแรม/ฟิวซูลินิด": ["🌾", "สิ่งมีชีวิตเซลล์เดียวมีเปลือก ขนาดราวเมล็ดข้าว", "Shelled single-celled organisms about the size of a rice grain"],
  "พืช": ["🌿", "ใบไม้ ลำต้น ไม้กลายเป็นหิน", "Leaves, stems, petrified wood"],
  "ร่องรอย": ["👣", "รอยเท้า รอยคืบคลาน รอยขุด (ไม่ใช่ตัวซาก)", "Footprints, trails, burrows (not the body itself)"],
};
const typeIcon = (t) => (TYPE_INFO[t] || ["🦴"])[0];
const typeDesc = (t) => (TYPE_INFO[t] ? L(TYPE_INFO[t][1], TYPE_INFO[t][2]) : "");

// ยุคธรณี: ชื่อจาก i18n.js (I18N.period), ช่วงอายุจาก PERIOD_MA ด้านล่าง
const eraName = (k) => I18N.period(k);
const eraRange = (k) => (k === "Precambrian" ? L("เก่ากว่า 541 ล้านปี", "older than 541 million years") : maText(k, k));

const PLACES = [
  ["ภูเวียง ขอนแก่น", 16.6803, 102.2357, "Phu Wiang, Khon Kaen"], ["ภูกุ้มข้าว กาฬสินธุ์", 16.6947, 103.5247, "Phu Kum Khao, Kalasin"],
  ["เขาขาว สระบุรี", 14.6993, 100.8573, "Khao Khao, Saraburi"], ["เกาะตะรุเตา สตูล", 6.62, 99.66, "Ko Tarutao, Satun"], ["กรุงเทพฯ", 13.75, 100.5, "Bangkok"],
];
const PART_PICKS = ["ฟัน", "เศษกระดูก", "กระดูกสันหลัง", "กระดูกซี่โครง", "กระดูกหาง", "เปลือกหอยสองฝา", "เปลือกแบรคิโอพอด", "เกล็ดปลา", "ใบไม้", "กระดองปู"];   // ค่าที่ส่งให้โมเดล = ภาษาไทยเสมอ
const THAI_BOUNDS = [[97.3, 5.6], [105.7, 20.5]];
// ชั้น "ยุคของหิน": สีมาตรฐานของแต่ละยุคตามตารางธรณีกาลสากล (International Commission on Stratigraphy) + อายุ (ล้านปีก่อน)
const PERIOD_COLOR = { Precambrian: "#f73563", Cambrian: "#7fa056", Ordovician: "#009270", Silurian: "#b3e1b6", Devonian: "#cb8c37",
  Carboniferous: "#67a599", Permian: "#f04028", Triassic: "#812b92", Jurassic: "#34b2c9", Cretaceous: "#7fc64e", Tertiary: "#fd9a52", Quaternary: "#f9f97f" };
const PERIOD_MA = { Precambrian: [4600, 541], Cambrian: [541, 485], Ordovician: [485, 444], Silurian: [444, 419], Devonian: [419, 359],
  Carboniferous: [359, 299], Permian: [299, 252], Triassic: [252, 201], Jurassic: [201, 145], Cretaceous: [145, 66], Tertiary: [66, 2.6], Quaternary: [2.6, 0] };
const SCREENS = { index: ["00", "HOMEPAGE"], map: ["01", "FOSSIL MAP"], identify: ["02", "IDENTIFY"], more: ["03", "MORE"] };

const state = {
  screen: "index", mode: "potential", bg: "bg_bias", type: null, aoa: true,
  grid: null, surf: null, meta: null, options: null, crit: {}, siteByLabel: new Map(),
  point: null, marker: null, siteMarker: null, abort: null, ready: false, fitted: false,
  styleTheme: THEME.theme, styled: THEME.theme,                    // ธีมของแผนที่ฐานที่โหลด / ที่ย้อมสีและวาดพื้นผิวแล้ว
  smooth: true, nb: null, disp: null, prov: null, shares: null, provRank: null, hoverPending: false, spec: null,
  look: { mode: "look", card: null, option: null, fitted: false }, lastPoint: null, modalK: null,
  geo: { data: null, loading: null, on: false, sel: null, layers: false },
};

// เซิร์ฟเวอร์ยุ่ง (คนเข้าเยอะ): 503 จากตัวจำกัดคิวของ server.py, 429 / 502 / 504 / 520–524 จาก Cloudflare หรือเชื่อมต่อไม่ได้
class BusyError extends Error {
  constructor(retryAfter) { super("busy"); this.name = "BusyError"; this.retryAfter = retryAfter || 20; }
}
const BUSY_STATUS = new Set([429, 502, 503, 504, 520, 521, 522, 523, 524]);

async function api(path, body, signal) {
  if (STATIC) return JurassicLocal.api(path, body);                 // อ่านไฟล์ / ทำนายในเบราว์เซอร์ (local.js)
  const opt = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal } : { signal };
  let r;
  try {
    r = await fetch(path, opt);
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new BusyError(20);                                         // เชื่อมต่อไม่ได้ (เซิร์ฟเวอร์ไม่ตอบ / tunnel หลุด)
  }
  if (BUSY_STATUS.has(r.status)) throw new BusyError(Number(r.headers.get("Retry-After")) || 20);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch (e) { /* ไม่ใช่ JSON */ }
    throw new Error(msg);
  }
  return r.json();
}

// การ์ดขอโทษเมื่อเซิร์ฟเวอร์ยุ่ง: ทีเร็กซ์พิกเซลวิ่ง (แบบเดียวกับ busy.html) + ข้อความสุภาพ + นับถอยหลังแล้วลองใหม่เอง (กดลองใหม่ได้ทันที)
const BUSY_DINO = `<svg class="busy-dino rex" viewBox="0 0 22 21" shape-rendering="crispEdges" aria-hidden="true"><path d="M12 0h8v1h-8zM11 1h2v1h-2zM14 1h7v1h-7zM11 2h10v1h-10zM11 3h10v1h-10zM11 4h10v1h-10zM11 5h5v1h-5zM11 6h8v1h-8zM0 7h1v1h-1zM10 7h5v1h-5zM0 8h1v1h-1zM9 8h6v1h-6zM0 9h2v1h-2zM8 9h7v1h-7zM0 10h3v1h-3zM7 10h10v1h-10zM0 11h4v1h-4zM6 11h9v1h-9zM16 11h1v1h-1zM0 12h14v1h-14zM1 13h13v1h-13zM2 14h11v1h-11zM3 15h9v1h-9zM4 16h7v1h-7z"/><rect class="blink" x="13" y="1" width="1" height="1"/><path class="leg leg-a" d="M4 17h3v1h-3zM9 17h4v1h-4zM4 18h2v1h-2zM4 19h1v1h-1zM4 20h2v1h-2z"/><path class="leg leg-b" d="M4 17h4v1h-4zM9 17h2v1h-2zM10 18h1v1h-1zM10 19h1v1h-1zM10 20h2v1h-2z"/><path class="leg leg-stand" d="M4 17h3v1h-3zM9 17h2v1h-2zM4 18h2v1h-2zM10 18h1v1h-1zM4 19h1v1h-1zM10 19h1v1h-1zM4 20h2v1h-2zM10 20h2v1h-2z"/></svg><div class="busy-ground"></div>`;

function showBusy(slot, seconds, retry) {
  if (!slot) return;
  const wait = Math.min(Math.max(seconds || 20, 5), 60);
  slot.innerHTML = `<div class="busy-card" role="status" aria-live="polite"><div class="busy-scene">${BUSY_DINO}</div>
    <div class="busy-text"><b>${L("ขอโทษนะ ตอนนี้คนแวะมาเยอะมากเลย", "Sorry! It's a little crowded right now")}</b>
      <span>${L("รอสักครู่แล้วระบบจะลองใหม่ให้เองนะคะ", "Please hang on, we'll try again for you.")}</span>
      <span class="busy-row"><button type="button" class="busy-retry">${L("ลองใหม่ตอนนี้", "Try again now")}</button>
        <span class="busy-count">${L("ลองใหม่อัตโนมัติใน", "Retrying in")} <b>${wait}</b> ${L("วินาที", "s")}</span></span></div></div>`;
  let left = wait, timer = null;
  const run = () => { clearTimeout(timer); if (slot.isConnected) retry(); };
  const tick = () => {
    if (!slot.isConnected) return;                                      // ผู้ใช้ทำอย่างอื่นไปแล้ว: เลิกนับ
    const b = slot.querySelector(".busy-count b");
    if (b) b.textContent = left;
    if (left <= 0) return run();
    left -= 1;
    timer = setTimeout(tick, 1000);
  };
  slot.querySelector(".busy-retry").addEventListener("click", run);
  tick();
}

// ---------------------------------------------------------------- การสลับหน้า
function go(screen, push = true, anim = true) {
  if (!SCREENS[screen]) screen = "index";
  state.screen = screen;
  document.body.dataset.screen = screen;
  $$(".nav a").forEach((a) => a.classList.toggle("on", a.dataset.go === screen));
  const [n, name] = SCREENS[screen];
  $("#ind-num").textContent = n;
  $("#ind-frac").textContent = `// ${n} / 03`;
  $("#ind-name").textContent = name;
  const el = $(`#s-${screen}`);
  if (anim) { el.classList.remove("enter"); void el.offsetWidth; el.classList.add("enter"); }   // ภาพเคลื่อนไหวสำรอง (เมื่อไม่ได้ใช้ View Transitions)
  if (push && location.hash !== `#${screen}`) history.pushState(null, "", `#${screen}`);
  if (screen === "map" || screen === "identify") requestAnimationFrame(() => { map.resize(); fitThailand(); });
  if (screen === "identify") requestAnimationFrame(fitLookIcons);
}

// ---------------------------------------------------------------- ภาพเคลื่อนไหวตอนเปลี่ยนหน้า / เปลี่ยนโหมด
// ใช้ View Transitions API: เปลี่ยนหน้า = เนื้อหาเลื่อนจาง (ไปหน้าถัดไปเข้าจากขวา ย้อนกลับเข้าจากซ้าย) แถบบนอยู่กับที่,
// เปลี่ยนโหมด = พระอาทิตย์ขึ้น / ตกจากขอบล่าง; เบราว์เซอร์ที่ไม่รองรับใช้ภาพเคลื่อนไหวแบบง่าย, ผู้ใช้ที่ตั้งลดภาพเคลื่อนไหวเปลี่ยนทันที
const reduceMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

function navigate(screen, push = true) {
  if (!SCREENS[screen]) screen = "index";
  if (screen === state.screen || !document.startViewTransition || reduceMotion()) return go(screen, push, !reduceMotion());
  const order = Object.keys(SCREENS), root = document.documentElement;
  root.dataset.vtDir = order.indexOf(screen) > order.indexOf(state.screen) ? "fwd" : "back";
  root.classList.add("vt-page");
  // ระหว่างอัปเดตหน้า เบราว์เซอร์หยุดวาด (requestAnimationFrame ไม่ทำงาน): ปรับขนาดแผนที่ทันที ไม่รอเฟรมถัดไป
  const vt = document.startViewTransition(() => { go(screen, push, false); if (screen === "map" || screen === "identify") { map.resize(); fitThailand(); } });
  vt.finished.finally(() => root.classList.remove("vt-page"));
}

function switchTheme() {
  const next = THEME.theme === "dark" ? "light" : "dark", root = document.documentElement;
  if (reduceMotion()) { THEME.set(next); return; }
  if (!document.startViewTransition) {                     // ไม่รองรับ: จางสีทั้งหน้า
    root.classList.add("theme-fade");
    THEME.set(next);
    setTimeout(() => root.classList.remove("theme-fade"), 600);
    return;
  }
  // มืด -> สว่าง = พระอาทิตย์ขึ้น (แสงโผล่จากขอบล่างแล้วแผ่ขึ้นไป โทนส้มอุ่นค่อย ๆ จางเป็นสีจริง),
  // สว่าง -> มืด = พระอาทิตย์ตก (หน้าสว่างหดลงสู่ขอบล่างพร้อมโทนส้ม เผยฟ้ากลางคืนจากด้านบน); ภาพเคลื่อนไหวอยู่ใน style.css
  const cls = next === "light" ? "vt-sunrise" : "vt-sunset";
  root.classList.add(cls);
  const vt = document.startViewTransition(() => THEME.set(next));
  vt.finished.finally(() => root.classList.remove(cls));
}

function fitThailand() {
  // ตั้งมุมมองครั้งแรกตอนแผนที่แสดงจริง (ตอนซ่อนอยู่ขนาดเป็น 0)
  if (state.fitted || !state.ready || !$("#mapcanvas").clientWidth) return;
  map.fitBounds(THAI_BOUNDS, { padding: mapPadding(), duration: 0 });
  state.fitted = true;
}

// ---------------------------------------------------------------- แผนที่
// แผนที่ฐานจาก OpenFreeMap (OpenStreetMap, ไม่ต้องใช้ key): โทนมืด / สว่างตามธีม (ชื่อสถานที่บนแผนที่ฐานแสดงทั้งอักษรละตินและอักษรท้องถิ่นอยู่แล้ว)
const BASEMAP = { dark: "https://tiles.openfreemap.org/styles/dark", light: "https://tiles.openfreemap.org/styles/positron" };
// ย้อมพื้นและน้ำของแผนที่ฐานให้เข้ากับธีม (มืด = ฟ้ากรมท่า, สว่าง = หมอก) ทุกครั้งที่โหลดแผนที่ฐาน
const BASE_TINT = {
  dark: { background: "#0d0e2c", water: "#161946", landcover_wood: "#141536", landuse_park: "#141536" },
  light: { background: "#eef0ea", water: "#c8d5d2", park: "#e2e8da", landcover_wood: "#dde4d3" },
};
function tintBasemap() {
  for (const [id, c] of Object.entries(BASE_TINT[THEME.theme])) {
    const l = map.getLayer(id);
    if (l) map.setPaintProperty(id, l.type === "background" ? "background-color" : "fill-color", c);
  }
}
const map = new maplibregl.Map({
  container: "mapcanvas",                                     // (id ไม่ชน #map ใน URL)
  style: BASEMAP[THEME.theme],
  center: [100.9, 13.3], zoom: 5.1, minZoom: 4, maxZoom: 13, maxBounds: [[90, 2], [112, 24]],
  attributionControl: { compact: true },
});
map.on("style.load", tintBasemap);
map.on("styledata", () => syncTheme());
map.on("idle", () => syncTheme());
map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");

const mapPadding = () => ({ top: 30, bottom: 30, left: window.innerWidth > 900 ? 460 : 20, right: 30 });
// สีขั้นของพื้นผิวโอกาสพบซาก (ขอบล่างของแต่ละขั้น = เปอร์เซ็นไทล์ทั้งประเทศ): ต่ำ 5 ขั้น · ปานกลาง 2 · สูง 2 · สูงมาก 2 — กลุ่มตรงกับ LEVELS
const SURF_STEPS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.625, 0.75, 0.825, 0.9, 0.95];
const SURF_GROUPS = [[0, 5, 3], [5, 7, 2], [7, 9, 1], [9, 11, 0]];        // [ขั้นแรก, ขั้นสุดท้าย + 1, ลำดับใน LEVELS]
const SURF_COLORS = {
  dark: ["#16183d", "#1c1f4b", "#232659", "#2a2d68", "#323678", "#5f4691", "#7f5daa", "#d97a30", "#f29a3f", "#ffcf5a", "#fff1a3"],
  light: ["#f5f3ea", "#eeeee0", "#e7e9d6", "#e0e4cc", "#d7dec1", "#bcd29c", "#96ba75", "#f0a860", "#e07f35", "#c4502a", "#93290f"],
};
const what = () => (state.mode === "potential" ? L("โอกาสพบซาก", "Fossil potential") : L(`โอกาสพบ${state.type}`, `${tr(state.type)} potential`));
const provName = (th) => tr(th);
// ชื่อแหล่ง ทธ. "ชื่อแหล่ง (จังหวัด)": ชื่อแหล่งเป็นชื่อเฉพาะภาษาไทย (ไม่มีคำแปลในข้อมูล) — โหมดอังกฤษแปลเฉพาะจังหวัดในวงเล็บ
const placeName = (s) => (I18N.lang === "en" ? String(s).replace(/\(([^()]+)\)\s*$/, (m, pv) => (I18N.en(pv) !== pv ? `(${I18N.en(pv)})` : m)) : s);

// ---------------------------------------------------------------- พื้นผิวโอกาสพบซาก (ภาพเดียวทั้งประเทศ ตัดตามขอบเขตประเทศ)
//   ค่าของช่องกริด 5 กม. (จุดบนตาราง EPSG:32647 ทุก 5 กม.) -> ประมาณค่าแบบ bilinear ทุกพิกเซล -> สีขั้นตามระดับ -> ตัดตามขอบเขตประเทศ
//   วาดเป็นภาพในพิกัด Web Mercator (ภาพของ MapLibre วางแบบเส้นตรงในพิกัดนี้) แล้วซ้อนเส้น / ชื่อจังหวัด และเส้นชายแดนไว้ด้านบน
const SURF_BOUNDS = [97.3, 5.5, 105.7, 20.5];                  // lon / lat ต่ำสุด, lon / lat สูงสุด
const SURF_WIDTH = 1400;                                      // พิกเซลตามแนวนอน (~650 ม. ต่อพิกเซล)
const mercY = (lat) => Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360));
const invMercY = (y) => (Math.atan(Math.sinh(y)) * 180) / Math.PI;

function utm47(lon, lat) {
  // WGS84 -> UTM โซน 47 เหนือ (EPSG:32647) ตามสูตรอนุกรมของ Snyder (1987) — คลาดเคลื่อนไม่ถึงเมตรในไทย
  const a = 6378137, f = 1 / 298.257223563, k0 = 0.9996, e2 = f * (2 - f), e4 = e2 * e2, e6 = e4 * e2, ep2 = e2 / (1 - e2);
  const phi = (lat * Math.PI) / 180, sn = Math.sin(phi), cs = Math.cos(phi), tn = Math.tan(phi);
  const N = a / Math.sqrt(1 - e2 * sn * sn), T = tn * tn, C = ep2 * cs * cs, A = cs * (((lon - 99) * Math.PI) / 180);
  const M = a * ((1 - e2 / 4 - (3 * e4) / 64 - (5 * e6) / 256) * phi - ((3 * e2) / 8 + (3 * e4) / 32 + (45 * e6) / 1024) * Math.sin(2 * phi)
    + ((15 * e4) / 256 + (45 * e6) / 1024) * Math.sin(4 * phi) - ((35 * e6) / 3072) * Math.sin(6 * phi));
  const x = 500000 + k0 * N * (A + ((1 - T + C) * A ** 3) / 6 + ((5 - 18 * T + T * T + 72 * C - 58 * ep2) * A ** 5) / 120);
  const y = k0 * (M + N * tn * ((A * A) / 2 + ((5 - T + 9 * C + 4 * C * C) * A ** 4) / 24 + ((61 - 58 * T + T * T + 600 * C - 330 * ep2) * A ** 6) / 720));
  return [x, y];
}

function surfaceGeom() {
  // คำนวณครั้งเดียว: พิกเซลที่อยู่ในประเทศ (หน้ากากจากขอบเขตประเทศ ขอบเรียบ) + ตำแหน่งของพิกเซลบนตารางกริด
  if (state.surf) return state.surf;
  const g = state.grid, [lo0, la0, lo1, la1] = SURF_BOUNDS, W = SURF_WIDTH;
  const y0 = mercY(la0), y1 = mercY(la1), H = Math.round((W * (y1 - y0)) / (((lo1 - lo0) * Math.PI) / 180));
  let ix0 = Infinity, iy0 = Infinity, ix1 = -Infinity, iy1 = -Infinity;
  for (let i = 0; i < g.n; i++) {
    ix0 = Math.min(ix0, g.ix[i]); ix1 = Math.max(ix1, g.ix[i]); iy0 = Math.min(iy0, g.iy[i]); iy1 = Math.max(iy1, g.iy[i]);
  }
  ix0 -= 3; iy0 -= 3;
  const NX = ix1 - ix0 + 4, NY = iy1 - iy0 + 4;
  const canvas = document.createElement("canvas");
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext("2d");
  const px = (lon) => ((lon - lo0) / (lo1 - lo0)) * W, py = (lat) => ((y1 - mercY(lat)) / (y1 - y0)) * H;
  const geom = state.prov.country.geometry, polys = geom.type === "Polygon" ? [geom.coordinates] : geom.coordinates;
  ctx.beginPath();
  for (const rings of polys) for (const ring of rings) ring.forEach(([lon, lat], k) => (k ? ctx.lineTo(px(lon), py(lat)) : ctx.moveTo(px(lon), py(lat))));
  ctx.fillStyle = "#000";
  ctx.fill("evenodd");
  const mask = ctx.getImageData(0, 0, W, H).data;
  ctx.clearRect(0, 0, W, H);
  let n = 0;
  for (let k = 3; k < mask.length; k += 4) if (mask[k]) n++;
  const pix = new Int32Array(n), fx = new Float32Array(n), fy = new Float32Array(n), cover = new Uint8Array(n);
  let q = 0;
  for (let r = 0; r < H; r++) {
    const lat = invMercY(y1 - ((r + 0.5) / H) * (y1 - y0));
    for (let c = 0; c < W; c++) {
      const a = mask[(r * W + c) * 4 + 3];
      if (!a) continue;
      const [x, y] = utm47(lo0 + ((c + 0.5) / W) * (lo1 - lo0), lat);
      pix[q] = r * W + c; fx[q] = x / 5000 - ix0; fy[q] = y / 5000 - iy0; cover[q] = a;
      q++;
    }
  }
  state.surf = { W, H, NX, NY, ix0, iy0, pix, fx, fy, cover, canvas, ctx, img: ctx.createImageData(W, H), url: null,
    coords: [[lo0, la1], [lo1, la1], [lo1, la0], [lo0, la0]] };
  return state.surf;
}

const hexRgb = (h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
const surfClass = (v) => { let k = 0; while (k + 1 < SURF_STEPS.length && v >= SURF_STEPS[k + 1]) k++; return k; };

function renderSurface() {
  // วาดพื้นผิวจากค่าที่แสดงอยู่ (state.disp: เปอร์เซ็นไทล์ 0–1 + ธง "ข้อมูลไม่พอ") แล้วส่งให้ชั้น "pot" ของแผนที่
  if (!state.disp || !map.getSource("pot")) return;
  const S = surfaceGeom(), g = state.grid, { v, o } = state.disp, NX = S.NX, N = NX * S.NY;
  let V = new Float32Array(N).fill(NaN), O = new Float32Array(N).fill(NaN);
  for (let i = 0; i < g.n; i++) { const k = g.ix[i] - S.ix0 + (g.iy[i] - S.iy0) * NX; V[k] = v[i]; O[k] = o[i]; }
  // ชายฝั่ง / ชายแดน: ศูนย์กลางช่องอาจตกนอกประเทศ — เติมค่าออกไป 2 ช่องจากช่องข้างเคียง ให้ภาพเต็มถึงเส้นขอบเขต
  for (let pass = 0; pass < 2; pass++) {
    const V2 = V.slice(), O2 = O.slice();
    for (let k = NX + 1; k < N - NX - 1; k++) {
      if (V[k] === V[k]) continue;
      let sv = 0, so = 0, c = 0;
      for (const d of [-NX - 1, -NX, -NX + 1, -1, 1, NX - 1, NX, NX + 1]) if (V[k + d] === V[k + d]) { sv += V[k + d]; so += O[k + d]; c++; }
      if (c) { V2[k] = sv / c; O2[k] = so / c; }
    }
    V = V2; O = O2;
  }
  const lut = SURF_STEPS.map((_, k) => hexRgb(SURF_COLORS[THEME.theme][k])), none = hexRgb(NONE.color), d = S.img.data;
  d.fill(0);
  for (let q = 0; q < S.pix.length; q++) {
    const i0 = Math.floor(S.fx[q]), j0 = Math.floor(S.fy[q]);
    if (i0 < 0 || j0 < 0 || i0 >= NX - 1 || j0 >= S.NY - 1) continue;
    const tx = S.fx[q] - i0, ty = S.fy[q] - j0, k = i0 + j0 * NX;
    const ks = [k, k + 1, k + NX, k + NX + 1], ws = [(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty];
    let sv = 0, so = 0, sw = 0;
    for (let m = 0; m < 4; m++) if (V[ks[m]] === V[ks[m]]) { sv += ws[m] * V[ks[m]]; so += ws[m] * O[ks[m]]; sw += ws[m]; }
    if (sw <= 0) continue;
    const rgb = state.aoa && so / sw > 0.5 ? none : lut[surfClass(sv / sw)], p = S.pix[q] * 4;
    d[p] = rgb[0]; d[p + 1] = rgb[1]; d[p + 2] = rgb[2]; d[p + 3] = S.cover[q];
  }
  S.ctx.putImageData(S.img, 0, 0);
  const job = (S.job = (S.job || 0) + 1);
  S.canvas.toBlob((blob) => {
    if (job !== S.job || !blob) return;                         // มีการวาดใหม่กว่าแล้ว
    const url = URL.createObjectURL(blob), src = map.getSource("pot"), old = S.url;
    S.url = url;
    if (src) src.updateImage({ url, coordinates: S.coords });
    if (old) setTimeout(() => URL.revokeObjectURL(old), 10000);
  });
}

function percentileRank(values) {
  const n = values.length, idx = Array.from(values.keys()).sort((a, b) => values[a] - values[b]);
  const out = new Float32Array(n);
  idx.forEach((i, r) => { out[i] = r / (n - 1); });
  return out;
}

function buildNeighbours(g) {
  // ช่องข้างเคียง 3 × 3 (ช่องละ 5 กม.) น้ำหนักแบบเกาส์: กลาง 4, ด้านข้าง 2, มุม 1
  const key = new Map();
  for (let i = 0; i < g.n; i++) key.set(g.ix[i] * 10000 + g.iy[i], i);
  const offs = [], w = new Float32Array(9);
  for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
    w[offs.length] = dx === 0 && dy === 0 ? 4 : dx === 0 || dy === 0 ? 2 : 1;
    offs.push([dx, dy]);
  }
  const idx = new Int32Array(g.n * 9).fill(-1);
  for (let i = 0; i < g.n; i++) offs.forEach(([dx, dy], j) => {
    const t = key.get((g.ix[i] + dx) * 10000 + g.iy[i] + dy);
    if (t !== undefined) idx[i * 9 + j] = t;
  });
  return { idx, w };
}

function smoothed(v) {
  // เฉลี่ยกับช่องรอบ ๆ (~15 กม.) แล้วจัดอันดับใหม่ทั้งประเทศ — ใช้แสดงผลเท่านั้น
  const { idx, w } = state.nb, n = v.length, out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0, sw = 0;
    for (let j = 0; j < 9; j++) { const t = idx[i * 9 + j]; if (t >= 0) { s += w[j] * v[t]; sw += w[j]; } }
    out[i] = s / sw;
  }
  return percentileRank(out);
}

function rawValues() {
  // ค่าจริงต่อช่อง (เปอร์เซ็นไทล์ 0–1 เทียบทั้งประเทศ) + ธง "ข้อมูลไม่พอ"
  const g = state.grid, n = g.n;
  if (state.mode === "potential") {
    return { v: Float32Array.from(g.rank[state.bg], (x) => x / 1000), o: Uint8Array.from(g.inside[state.bg], (x) => (x ? 0 : 1)) };
  }
  const t = g.types[state.type], pot = g.potential.bg_bias, raw = new Float32Array(n);
  for (let i = 0; i < n; i++) raw[i] = (t[i] / 1000) * (pot[i] / 1000);       // โอกาสพบซาก × ความน่าจะเป็นของชนิด
  return { v: percentileRank(raw), o: Uint8Array.from(g.inside.B, (x) => (x ? 0 : 1)) };
}

function provinceShares(v, o) {
  // สัดส่วนพื้นที่ (ช่อง 5 กม.) ในจังหวัดที่อยู่ระดับ "สูง" ขึ้นไป (ค่าจริง ไม่ใช่ค่าเฉลี่ยแสดงผล)
  const np = state.prov.polygons.features.length, hi = new Float32Array(np), all = new Float32Array(np);
  for (let i = 0; i < v.length; i++) {
    const p = state.grid.prov[i];
    if (p < 0) continue;
    all[p]++;
    if (v[i] >= 0.75 && !o[i]) hi[p]++;
  }
  return Float32Array.from(hi, (h, p) => (all[p] ? h / all[p] : 0));
}

function themePaint(id) {
  // สีของชั้นข้อมูลของเราที่ขึ้นกับธีม (ใช้ตอนสร้างชั้น และตอนเปลี่ยนแผนที่ฐาน)
  const light = THEME.theme === "light";
  switch (id) {
    case "prov-line": return { "line-color": light ? "#3a342b" : "#e8e4ff" };
    case "country-line": return { "line-color": light ? "#2a251f" : "#ffffff" };
    case "prov-label": return { "text-color": light ? "#2a251f" : "#ffffff", "text-halo-color": light ? "#eef0ea" : "#0b0c29" };
    case "sites": return { "circle-stroke-color": light ? "#3a342b" : "#000000" };
    case "pbdb": return { "circle-color": light ? "#3d7a3a" : "#ffd86b" };   // ตรงกับจุดในคำอธิบาย (.dot.pbdb = --yellow)
    case "geo-line": return { "line-color": light ? "#2b2b2b" : "#05080a" };
    default: return {};
  }
}

const OWN_LAYERS = new Set(["pot", "prov-hit", "geo-fill", "geo-line", "prov-line", "country-line", "sites", "pbdb", "prov-label"]);

function setBasemap() {
  // เปลี่ยนแผนที่ฐานตามธีม แล้วย้ายชั้นข้อมูลของเรา (ข้อมูล การกรอง การซ่อน / แสดง) ไปไว้บนแผนที่ฐานใหม่ พร้อมสีตามธีม
  map.setStyle(BASEMAP[THEME.theme], {
    transformStyle: (prev, next) => {
      state.styleTheme = THEME.theme;
      if (!prev) return next;
      // สำเนาใหม่ของชั้น (ห้ามแก้วัตถุใน prev: maplibre เทียบ prev กับ next เพื่อหาสิ่งที่เปลี่ยน)
      const own = prev.layers.filter((l) => OWN_LAYERS.has(l.id)).map((l) => ({ ...l, paint: { ...(l.paint || {}), ...themePaint(l.id) } }));
      const sources = {};
      own.forEach((l) => { sources[l.source] = prev.sources[l.source]; });
      return { ...next, sources: { ...next.sources, ...sources }, layers: [...next.layers, ...own] };
    },
  });
}

function syncTheme() {
  // เปลี่ยนแผนที่ฐานแบบ diff ไม่มี "style.load": เมื่อ style ของธีมใหม่ (transformStyle จดไว้) โหลดเสร็จ ย้อมสีพื้น
  // และวาดพื้นผิวโอกาสพบซากด้วยสีของธีมนั้น — ครั้งเดียวต่อการเปลี่ยนธีม
  if (state.styleTheme !== THEME.theme || state.styled === THEME.theme || !map.isStyleLoaded()) return;
  state.styled = THEME.theme;
  tintBasemap();
  if (state.ready) renderSurface();
}

const provLabelField = () => (I18N.lang === "en" ? ["coalesce", ["get", "name_en"], ["get", "name"]] : ["get", "name"]);

function updateLayer() {
  if (!state.grid) return;
  const { v, o } = rawValues();
  const dv = state.smooth ? smoothed(v) : v;
  state.disp = { v: dv, o };
  renderSurface();
  state.shares = provinceShares(v, o);
  renderRanking();
  renderLegend();
}

function renderLegend() {
  const el = $("#maplegend");
  if (state.geo.on && state.geo.data) {
    const sel = state.geo.data.periods.filter((p) => state.geo.sel.has(p.period)).reverse();
    el.innerHTML = `<div class="title">${L("ยุคของหิน · แผนที่ธรณีวิทยา 1:250,000", "Rock age · 1:250,000 geological map")}</div>
      ${sel.length ? `<div class="geo-leg">${sel.map((p) => `<div class="row"><span class="sw" style="background:${PERIOD_COLOR[p.period]}"></span><b>${esc(eraName(p.period))}</b></div>`).join("")}</div>`
        : `<div class="muted">${L("ยังไม่ได้เลือกยุค", "No period selected")}</div>`}
      <div class="geo-mob">${L(`${sel.length} ยุค · ดูสีของแต่ละยุคในรายการใต้แผนที่`, `${sel.length} period${sel.length === 1 ? "" : "s"} · see each period's colour in the list below the map`)}</div>
      <div class="foot">${L("ใหม่ → เก่า · สีตามยุคที่เก่าที่สุดของหน่วยหิน · ชี้เมาส์เพื่อดูชื่อหมวดหิน", "Young → old · coloured by the oldest period of each rock unit · hover to see the formation name")}</div>`;
    el.hidden = false;
    $("#type-select").hidden = state.mode !== "type";
    $("#type-note").hidden = state.mode !== "type";
    return;
  }
  // แถบสีแบบขั้น (เหมือนแผนที่ฝนของกรมอุตุฯ): ขั้นละช่องเท่ากัน, ใต้แถบ = ระดับที่ผลวิเคราะห์จุดใช้ (ต่ำ / ปานกลาง / สูง / สูงมาก)
  const cols = SURF_COLORS[THEME.theme], n = cols.length, at = (k) => `${(k / n) * 100}%`;
  const tick = (k, t) => `<span style="left:${at(k)}">${t}</span>`;
  el.innerHTML = `<div class="title">${esc(what())} ${L("(เทียบทั้งประเทศ)", "(relative to the whole country)")}</div>
    <div class="arrows"><span>‹ ${L("โอกาสต่ำ", "Lower")}</span><span>${L("โอกาสสูง", "Higher")} ›</span></div>
    <div class="steps">${cols.map((c) => `<span style="background:${c}"></span>`).join("")}</div>
    <div class="steptk">${tick(0, "0")}${tick(5, "50")}${tick(7, "75")}${tick(9, "90")}${tick(n, "100")}</div>
    <div class="steplv">${SURF_GROUPS.map(([a, b, li]) => `<span style="left:${at(a)};width:${at(b - a)}">${LEVELS[li].label}</span>`).join("")}</div>
    ${state.aoa ? `<div class="row"><span class="sw" style="background:${NONE.color}"></span><b>${NONE.label}</b><span class="muted">${NONE.text}</span></div>` : ""}
    <div class="foot">${L("ตัวเลข = เปอร์เซ็นไทล์ของพื้นที่ทั้งประเทศ · ", "Numbers = percentile of the whole country's area · ")}${state.smooth ? L("เฉลี่ยพื้นที่รอบ ๆ ~15 กม. · ", "averaged over ~15 km · ") : ""}${L("คลิกเพื่อวิเคราะห์จุดนั้นละเอียด", "click to analyse a point in detail")}</div>`;
  el.hidden = false;
  $("#type-select").hidden = state.mode !== "type";
  $("#type-note").hidden = state.mode !== "type";
  if (state.mode === "type") $("#type-note").textContent = `${typeIcon(state.type)} ${typeDesc(state.type)}`;
}

function renderRanking() {
  const pf = state.prov.polygons.features, sh = state.shares;
  const order = Array.from(sh.keys()).sort((a, b) => sh[b] - sh[a]);
  state.provRank = new Int16Array(order.length);
  order.forEach((k, r) => { state.provRank[k] = r; });
  const top = order.slice(0, 8), max = sh[top[0]] || 1;
  $("#rank-title").innerHTML = L(`จังหวัดที่น่าสำรวจที่สุด${state.mode === "type" ? ` · ${esc(state.type)}` : ""} <em>TOP PROVINCES</em>`,
    `Top provinces to explore${state.mode === "type" ? ` · ${esc(tr(state.type))}` : ""}`);
  $("#ranklist").innerHTML = top.map((k, r) => `<li data-p="${k}"><span class="n">${String(r + 1).padStart(2, "0")}</span><span>${esc(provName(pf[k].properties.name))}</span>
    <span class="bar"><span style="width:${(sh[k] / max) * 100}%"></span></span><span class="v">${Math.round(sh[k] * 100)}%</span></li>`).join("");
  $$("#ranklist li").forEach((li) => li.addEventListener("click", () => zoomProvince(+li.dataset.p)));
}

function zoomProvince(k) {
  const b = state.prov.polygons.features[k].properties.bbox;
  map.fitBounds([[b[0], b[1]], [b[2], b[3]]], { padding: mapPadding(), duration: 700, maxZoom: 9 });
}

function nearestCell(lat, lon) {
  const g = state.grid, c = Math.cos((lat * Math.PI) / 180);
  let best = 0, bd = Infinity;
  for (let i = 0; i < g.n; i++) {
    const d = (g.lat[i] - lat) ** 2 + ((g.lon[i] - lon) * c) ** 2;
    if (d < bd) { bd = d; best = i; }
  }
  return { i: best, km: Math.sqrt(bd) * 111 };
}

function topTypeOf(i) {
  const g = state.grid, cls = state.meta.classes_B;
  let top = 0;
  cls.forEach((c, j) => { if (g.types[c][i] > g.types[cls[top]][i]) top = j; });
  return cls[top];
}

function showTip(e) {
  const tip = $("#hovertip");
  if (!state.ready) return;
  const hit = map.queryRenderedFeatures(e.point, { layers: ["prov-hit"] });
  if (!hit.length) { tip.hidden = true; return; }
  const k = hit[0].properties.id, name = provName(hit[0].properties.name);
  const act = state.screen === "identify" ? L("คลิกเพื่อเลือกเป็นจุดที่พบ ›", "Click to set as the find location ›") : L("คลิกเพื่อวิเคราะห์จุดนี้ละเอียด ›", "Click to analyse this point in detail ›");
  let html;
  if (state.geo.on && state.geo.layers) {
    const g = map.queryRenderedFeatures(e.point, { layers: ["geo-fill"] })[0];
    if (!g) html = `<b class="name">${esc(name)}</b><span class="muted">${L("ไม่มีหน่วยหินของยุคที่เลือกตรงนี้", "No rock unit of the selected periods here")}</span>`;
    else {
      const q = g.properties, per = state.geo.data.periods.find((p) => p.period === q.old);
      const same = q.old === q.young, nm = same ? eraName(q.old) : `${eraName(q.old)} – ${eraName(q.young)}`;
      html = `<b class="name">${esc(name)}</b><span class="lv" style="background:${PERIOD_COLOR[q.old]};color:#05080a">${L(`ยุค${esc(nm)}`, esc(nm))}</span>
        ${maText(q.old, q.young)}<br><b>${esc(L(q.label, q.label_en))}</b> <span class="muted">(${L("สัญลักษณ์", "symbol")} ${esc(q.sym)} · ${esc(L(q.kind, q.kind_en))})</span>
        ${per && per.sites ? L(`<br>แหล่งซากที่ขึ้นทะเบียนในยุค${esc(eraName(q.old))}: ${per.sites} แหล่งทั่วประเทศ`, `<br>Registered fossil sites from the ${esc(eraName(q.old))}: ${per.sites} nationwide`) : ""}
        ${map.getZoom() < 8.5 ? `<br><span class="muted">${L("ซูมเข้าเพื่อดูหน่วยหินตรงตำแหน่ง (ซูมออก 1 จุดบนจอกว้างหลายกิโลเมตร)", "Zoom in to see the rock unit at this exact spot (zoomed out, one screen pixel spans several kilometres)")}</span>` : ""}
        <div class="act">${act}</div>`;
    }
  } else {
    const { i, km } = nearestCell(e.lngLat.lat, e.lngLat.lng);
    const pct = Math.round(state.shares[k] * 100), rk = state.provRank[k] + 1, np = state.shares.length;
    const share = `<br><span class="muted">${L(`ทั้งจังหวัด: พื้นที่โอกาสสูง ${pct}% · อันดับ ${rk} จาก ${np}`, `Whole province: ${pct}% high-potential area · rank ${rk} of ${np}`)}</span>`;
    if (km > 4) { html = `<b class="name">${esc(name)}</b><span class="muted">${L("ไม่มีข้อมูลตรงนี้", "No data here")}</span>${share}`; }
    else {
      const lv = state.aoa && state.disp.o[i] ? NONE : levelOf(state.disp.v[i] * 100);
      const tt = topTypeOf(i);
      html = `<b class="name">${esc(name)}</b><span class="lv" style="background:${lv.color};color:${lv.ink}">${lv.label}</span>${esc(what())}
        <br>${L("ถ้ามีซาก น่าจะเป็น", "If there are fossils, most likely")}: ${typeIcon(tt)} ${esc(tr(tt))}${share}
        <div class="act">${act}</div>`;
    }
  }
  tip.innerHTML = html;
  const W = $("#mapwrap").clientWidth, H = $("#mapwrap").clientHeight;
  tip.style.left = `${e.point.x}px`;
  tip.style.top = `${e.point.y}px`;
  tip.style.transform = `translate(${e.point.x > W - 300 ? "calc(-100% - 14px)" : "14px"}, ${e.point.y > H - 140 ? "calc(-100% - 14px)" : "14px"})`;
  tip.hidden = false;
}

// ---------------------------------------------------------------- ชั้น "ยุคของหิน" (แผนที่ธรณี 1:250,000 ของ ทธ.)
function maText(old, young) {
  const a = PERIOD_MA[old][0], b = PERIOD_MA[young][1];
  return b === 0 ? L(`${a} ล้านปีก่อน – ปัจจุบัน`, `${a} million years ago – present`) : L(`${a}–${b} ล้านปีก่อน`, `${a}–${b} million years ago`);
}

function loadGeology() {
  // ดึงครั้งเดียว (ล่วงหน้าหลังเปิดหน้า เพื่อให้กดเปิดแล้วขึ้นทันที)
  if (!state.geo.loading) {
    state.geo.loading = api("/api/geology").then((d) => {
      state.geo.data = d;
      state.geo.sel = state.geo.sel || new Set(d.periods.map((p) => p.period));
      renderGeoList();
      return d;
    }).catch((err) => { state.geo.loading = null; $("#geo-list").innerHTML = `<li class="muted">${L("โหลดแผนที่ธรณีไม่ได้", "Could not load the geological map")} (${esc(err.message)})</li>`; throw err; });
  }
  return state.geo.loading;
}

function renderGeoList() {
  // เรียงแบบลำดับชั้นหิน: ยุคใหม่อยู่บน ยุคเก่าอยู่ล่าง
  const d = state.geo.data;
  $("#geo-list").innerHTML = d.periods.slice().reverse().map((p) => `<li data-p="${p.period}" class="${state.geo.sel.has(p.period) ? "" : "off"}"
      role="checkbox" aria-checked="${state.geo.sel.has(p.period)}" tabindex="0">
      <input type="checkbox" tabindex="-1" ${state.geo.sel.has(p.period) ? "checked" : ""} aria-hidden="true">
      <span class="sw" style="background:${PERIOD_COLOR[p.period]}"></span>
      <span class="nm"><b>${esc(eraName(p.period))}</b><small>${maText(p.period, p.period)}</small></span>
      <span class="st">${L("พื้นที่", "Area")} ${p.pct}%<br>${p.sites ? L(`<b>${p.sites}</b> แหล่งซาก`, `<b>${p.sites}</b> fossil site${p.sites === 1 ? "" : "s"}`) : L("ไม่มีแหล่งซาก", "No fossil sites")}</span></li>`).join("");
  $$("#geo-list li[data-p]").forEach((li) => {
    const flip = () => { const k = li.dataset.p; state.geo.sel.has(k) ? state.geo.sel.delete(k) : state.geo.sel.add(k); applyGeo(); };
    li.addEventListener("click", flip);
    li.addEventListener("keydown", (e) => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); flip(); } });
  });
}

function applyGeo() {
  // เปิด / ปิดชั้นยุคของหิน: เปิดแล้วซ่อนสีโอกาสพบซาก (ไม่ให้สีปนกัน) และกรองตามยุคที่เลือก
  const g = state.geo;
  if (g.on && g.data && !g.layers) {
    map.addSource("geo", { type: "geojson", data: g.data, tolerance: 0.15 });
    map.addLayer({ id: "geo-fill", type: "fill", source: "geo",
      paint: { "fill-color": ["match", ["get", "old"], ...Object.entries(PERIOD_COLOR).flat(), "#888888"],
        "fill-opacity": ["interpolate", ["linear"], ["zoom"], 5, 0.72, 10, 0.55] } }, "prov-line");
    map.addLayer({ id: "geo-line", type: "line", source: "geo", minzoom: 6.5,
      paint: { ...themePaint("geo-line"), "line-opacity": 0.45, "line-width": ["interpolate", ["linear"], ["zoom"], 6.5, 0.3, 10, 0.9] } }, "prov-line");
    g.layers = true;
  }
  if (g.layers) {
    const vis = g.on ? "visible" : "none", filter = ["in", ["get", "old"], ["literal", [...g.sel]]];
    ["geo-fill", "geo-line"].forEach((id) => { map.setLayoutProperty(id, "visibility", vis); map.setFilter(id, filter); });
  }
  if (map.getLayer("pot")) map.setLayoutProperty("pot", "visibility", g.on ? "none" : "visible");
  $("#geo-panel").hidden = !g.on;
  if (g.data) renderGeoList();
  renderLegend();
}

function onHover(e) {
  if (state.hoverPending) return;
  state.hoverPending = true;
  requestAnimationFrame(() => { state.hoverPending = false; showTip(e); });
}

// ---------------------------------------------------------------- ผลวิเคราะห์จุด
const likely = (p) => (p >= 0.6 ? [L("เป็นไปได้มาก", "Very likely"), "l3"] : p >= 0.3 ? [L("เป็นไปได้", "Likely"), "l2"] : [L("เป็นไปได้น้อย", "Less likely"), ""]);
const coord = (lat, lon) => `${lat.toFixed(4)} // ${lon.toFixed(4)}`;

// สีหมุดตามธีม (สีของ Marker กำหนดตอนสร้าง: เปลี่ยนธีมแล้วสร้างใหม่ที่ตำแหน่งเดิม ดู recolorMarkers)
const PIN = { dark: { point: "#f29a3f", site: "#ffd86b" }, light: { point: "#c4581c", site: "#3d7a3a" } };
function setMarker(lat, lon) {
  if (!state.marker) state.marker = new maplibregl.Marker({ color: PIN[THEME.theme].point });
  state.marker.setLngLat([lon, lat]).addTo(map);
  state.point = { lat, lon };
  pointLabel();
}

function pointLabel() {
  if (!state.point) return;
  const { lat, lon } = state.point, xy = `${lat.toFixed(4)}, ${lon.toFixed(4)}`;
  $("#point-label").textContent = L(`เลือกจุด ${xy} แล้ว (คลิกใหม่เพื่อเปลี่ยน)`, `Point ${xy} selected (click again to change)`);
}

function renderPoint(r) {
  const bg = state.meta.default_background;
  const pct = r[`percentile_${bg}`];
  const inside = r[`inside_${bg}`];
  const lv = levelOf(pct);
  const era = PERIOD_MA[r.rock_era] ? r.rock_era : null;
  const near = r.nearest_site_km < 0.5 ? L("จุดนี้คือแหล่งซากที่ขึ้นทะเบียนแล้ว", "This point is a registered fossil site")
    : L(`แหล่งซากที่ขึ้นทะเบียนใกล้ที่สุดอยู่ห่าง ${r.nearest_site_km} กม.`, `The nearest registered fossil site is ${r.nearest_site_km} km away`);
  const why = !inside ? L("ระบบไม่เคยเห็นพื้นที่ที่มีลักษณะแบบนี้", "The system has never seen an area like this")
    : pct >= 99 ? L("อยู่ใน 1% แรกของประเทศ", "In the top 1% of the country") : L(`สูงกว่าพื้นที่ ${Math.floor(pct)}% ของประเทศ`, `Higher than ${Math.floor(pct)}% of the country`);
  const tech = state.meta.backgrounds.map((b) => `<dt>${L("คะแนนโมเดล", "Model score")} (${b})</dt><dd>${r[`potential_${b}`].toFixed(3)} · ${L("อันดับ", "percentile")} ${r[`percentile_${b}`]}%</dd>`).join("");
  const io = (x) => (x ? L("ใน", "in") : L("นอก", "out"));
  return `<h2>${r.province ? L(`จังหวัด${esc(r.province)}`, `${esc(provName(r.province))} Province`) : L("จุดที่เลือก", "Selected point")}</h2>
    <div class="where">${coord(r.lat, r.lon)}</div>
    ${r.grid_cell && r.grid_cell.km != null ? `<div class="note">${L(`ผลวิเคราะห์ของช่องกริด 5 กม. ที่ใกล้จุดนี้ที่สุด (ศูนย์กลางช่องห่าง ${r.grid_cell.km} กม.) — คำนวณไว้ล่วงหน้าทั้งประเทศ`,
      `Analysis of the nearest 5 km grid cell (its centre is ${r.grid_cell.km} km away), precomputed for the whole country`)}</div>` : ""}
    <h3>${L("โอกาสพบซากดึกดำบรรพ์", "Fossil potential")}</h3>
    <div class="level">${pill(inside ? lv : NONE)}<span class="why">${why}</span></div>
    ${inside ? "" : `<div class="warn">${L("ข้อมูลไม่พอ — พื้นที่นี้ต่างจากแหล่งซากที่ระบบเคยเรียนรู้ (เช่น ภูเขาหินแกรนิต หรือที่ราบในเมือง) ผลอาจไม่แม่น",
      "Not enough data: this area differs from the fossil sites the system learned from (e.g. granite mountains or urban plains), so the result may be inaccurate")}</div>`}
    ${r.s2_scene ? "" : `<div class="warn">${L("ไม่ได้ใช้ภาพถ่ายดาวเทียม ผลอาจคลาดเคลื่อนมาก", "Satellite imagery was not used; the result may be far off")}</div>`}
    <h3>${L("ถ้ามีซาก น่าจะเป็น", "If there are fossils, they are probably")}</h3>
    ${r.inside_types ? "" : `<div class="note">${L("ข้อมูลไม่พอสำหรับบอกชนิดซากในพื้นที่นี้ ใช้เป็นแนวทางคร่าว ๆ เท่านั้น", "Not enough data to tell fossil types here; use this only as a rough guide")}</div>`}
    ${r.top_types.map(([t, p]) => { const [w, c] = likely(p); return `<div class="trow"><span class="ico">${typeIcon(t)}</span>
      <span class="nm">${esc(tr(t))}</span><span class="tag ${c}">${w}</span><span class="desc">${esc(typeDesc(t))}</span></div>`; }).join("")}
    <h3>${L("เกี่ยวกับพื้นที่นี้", "About this area")}</h3>
    <ul class="facts">
      <li><span>01</span><span>${era ? L(`หินบริเวณนี้เป็นหินยุค<b>${eraName(era)}</b> (${eraRange(era)})`, `The rock here is from the <b>${eraName(era)}</b> (${eraRange(era)})`)
        : L("ไม่มีข้อมูลอายุหินบริเวณนี้", "No rock-age data for this area")}</span></li>
      <li><span>02</span><span>${near}</span></li>
      <li><span>03</span><span>${L("มีลักษณะคล้ายแหล่ง", "Similar to the sites")}: ${r.similar_sites.map((n) => esc(placeName(n))).join(", ")}</span></li>
    </ul>
    <p style="margin:12px 0 0"><button class="link" id="use-point">${L("พบซากที่นี่? ลองทายชื่อซาก ›", "Found a fossil here? Try identifying it ›")}</button></p>
    <details class="tech"><summary>TECHNICAL DETAILS${L(" · รายละเอียดทางเทคนิค", "")}</summary><dl>
      ${tech}
      <dt>${L("ขอบเขตข้อมูล (AOA)", "Area of applicability (AOA)")}</dt><dd>${state.meta.backgrounds.map((b) => `${b}: ${io(r[`inside_${b}`])}`).join(" · ")} · ${L("ชนิดซาก", "fossil types")}: ${io(r.inside_types)}</dd>
      <dt>${L("หน่วยหิน / แผ่นเปลือกโลก", "Rock unit / terrane")}</dt><dd>${esc(r.rock_unit)} / ${esc(r.terrane)}</dd>
      <dt>${L("ภาพดาวเทียม", "Satellite image")}</dt><dd>${esc(r.s2_scene || L("ไม่ได้ใช้", "not used"))}</dd>
      ${r.grid_cell && r.grid_cell.km != null ? `<dt>${L("ศูนย์กลางช่องกริดที่ใช้", "Grid cell centre used")}</dt><dd>${coord(r.grid_cell.lat, r.grid_cell.lon)}</dd>`
        : `<dt>${L("ใช้เวลา", "Time taken")}</dt><dd>${r.seconds} ${L("วินาที", "s")}</dd>`}
    </dl></details>`;
}

function showPoint(r) {
  state.lastPoint = r;
  $("#result").innerHTML = renderPoint(r);
  $("#use-point").addEventListener("click", () => { navigate("identify"); $('input[name="loc"][value="point"]').checked = true; });
}

async function analysePoint(lat, lon) {
  setMarker(lat, lon);
  const card = $("#result");
  card.hidden = false;
  const { i } = nearestCell(lat, lon);
  const pre = state.aoa && state.disp.o[i] ? NONE.label : levelOf(state.disp.v[i] * 100).label;
  state.lastPoint = null;
  card.innerHTML = `<h2>${L("จุดที่เลือก", "Selected point")}</h2><div class="where">${coord(lat, lon)}</div>
    <div class="status"><span class="spin"></span>${STATIC ? L("กำลังเปิดผลวิเคราะห์ของพื้นที่นี้…", "Opening the analysis for this area…")
      : L("กำลังวิเคราะห์จุดนี้ด้วยภาพถ่ายดาวเทียมและข้อมูลธรณีวิทยา… (ราว 10–20 วินาที)", "Analysing this point with satellite imagery and geological data… (about 10–20 seconds)")}</div>
    <div class="note">${L("ระหว่างรอ: ค่าประมาณจากแผนที่", "While you wait: estimate from the map")} = <b>${pre}</b></div>`;
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  if (state.abort) state.abort.abort();
  state.abort = new AbortController();
  try {
    const r = await api("/api/predict_location", { lat, lon, with_s2: $("#with-s2").checked }, state.abort.signal);
    showPoint(r);
  } catch (err) {
    if (err.name === "AbortError") return;
    if (err instanceof BusyError) {
      card.innerHTML = `<h2>${L("จุดที่เลือก", "Selected point")}</h2><div class="where">${coord(lat, lon)}</div><div class="busy-slot"></div>`;
      showBusy(card.querySelector(".busy-slot"), err.retryAfter, () => analysePoint(lat, lon));
      return;
    }
    card.innerHTML = `<h2>${L("จุดที่เลือก", "Selected point")}</h2><div class="where">${coord(lat, lon)}</div><div class="err">${L("วิเคราะห์ไม่สำเร็จ", "Analysis failed")}: ${esc(err.message)}</div>`;
  }
}

function goTo(lat, lon) {
  map.flyTo({ center: [lon, lat], zoom: Math.max(map.getZoom(), 8.5) });
  if (state.screen === "identify") setMarker(lat, lon);
  else analysePoint(lat, lon);
}

// ---------------------------------------------------------------- ทายชื่อซาก
function siteFromInput() {
  return state.siteByLabel.get($("#site-input").value.trim()) || null;
}

function onSiteChange() {
  const s = siteFromInput(), hint = $("#site-hint");
  if (!s) { hint.textContent = $("#site-input").value ? L("ไม่พบแหล่งนี้ — เลือกจากรายการที่ขึ้นมา", "Site not found; pick one from the list") : ""; return; }
  $('input[name="loc"][value="site"]').checked = true;
  hint.textContent = s.known ? L("แหล่งนี้เคยมีการระบุชื่อซากแล้ว ระบบจึงทายได้แม่นกว่า", "Fossils from this site have been identified before, so the guess is more accurate")
    : L("แหล่งนี้ยังไม่เคยมีการระบุชื่อซาก", "No fossils from this site have been identified yet");
  if (!state.siteMarker) state.siteMarker = new maplibregl.Marker({ color: PIN[THEME.theme].site, scale: 0.8 });
  state.siteMarker.setLngLat([s.lon, s.lat]).addTo(map);
  map.flyTo({ center: [s.lon, s.lat], zoom: Math.max(map.getZoom(), 8.5) });
}

const confWord = (p) => (p >= 0.6 ? L("สูง", "high") : p >= 0.3 ? L("ปานกลาง", "medium") : L("ต่ำ", "low"));

function ageLine(g, grp) {
  // ซากนี้ "ส่วนใหญ่พบในชั้นหินยุคไหน" — จำนวนแหล่งต่อยุค (ทะเบียน ทธ. + PBDB; /api/period_profile)
  const pp = state.pp;
  if (!pp) return "";
  let prof = g === "อื่น ๆ" ? null : pp.genus[g], who = "";
  if (!prof || !prof.n) {                                            // สกุลที่ไม่มีข้อมูลแหล่ง / "ยังระบุชื่อไม่ได้" = ใช้ข้อมูลของกลุ่มซาก
    prof = grp ? pp.group[grp] : null;
    who = g === "อื่น ๆ" ? L(`ซากกลุ่ม${grp}`, `${tr(grp)} fossils:`) : L(`ยังไม่มีข้อมูลแหล่งของสกุลนี้ · ซากกลุ่ม${grp}`, `No site data for this genus yet · ${tr(grp)} fossils:`);
  }
  if (!prof || !prof.n) return "";
  const [top, k] = prof.p[0], share = k / prof.n, n = prof.n, tn = esc(eraName(top));
  const lead = share > 0.5 ? L(`ส่วนใหญ่พบในชั้นหินยุค<b>${tn}</b>`, `mostly found in <b>${tn}</b> rocks`) : L(`พบหลายยุค มากสุดในชั้นหินยุค<b>${tn}</b>`, `found in several periods, most in <b>${tn}</b> rocks`);
  const rest = prof.p.slice(1, 4).map(([q, c]) => `${esc(eraName(q))} ${c}`).join(" · ") + (prof.p.length > 4 ? " · …" : "");
  const bar = prof.p.map(([q, c]) => `<span style="width:${(c / n) * 100}%;background:${PERIOD_COLOR[q]}" title="${esc(eraName(q))} ${c} ${L("แหล่ง", c === 1 ? "site" : "sites")}"></span>`).join("");
  const lead1 = who ? lead : L(lead, lead.charAt(0).toUpperCase() + lead.slice(1));            // อังกฤษ: ขึ้นต้นประโยคด้วยตัวใหญ่
  return `<div class="age" title="${L(`นับจำนวนแหล่ง: ทะเบียนกรมทรัพยากรธรณี ${prof.dmr} แหล่ง + ฐานข้อมูลนักวิจัย PBDB ${prof.pbdb} แหล่ง`, `Site counts: ${prof.dmr} from the DMR register + ${prof.pbdb} from the PBDB research database`)}">
    <span class="ab">${bar}</span><span>${who ? `${esc(who)} ` : ""}${lead1} <span class="muted">(${maText(top, top)}) · ${L(`${k} จาก ${n} แหล่ง`, `${k} of ${n} sites`)}${n < 3 && !prof.weak ? L(" · ข้อมูลน้อย", " · little data") : ""}${rest ? ` · ${L("ยุคอื่น", "other periods")}: ${rest}` : ""}</span>
    ${prof.weak ? `<span class="weak">${L("ข้อมูลไม่แน่นอน: ตัวอย่างสกุลนี้เชื่อมกับแหล่ง หรือยุคของแหล่ง ด้วยความเชื่อมั่นต่ำ", "Uncertain: specimens of this genus are linked to sites, or sites to periods, with low confidence")}</span>` : ""}</span></div>`;
}

function renderSpecimen(r, ftype) {
  const items = r.top.map(([g, p, grp], k) => {
    const { name, thai, key } = genusView(g, grp || ftype);
    return `<div class="genus pick ${k === 0 ? "top on" : ""}" data-k="${k}">
      <button type="button" class="holo-mini" data-k="${k}" aria-label="${L("ดูภาพ", "Show")} ${esc(name)}">${HOLO.svg(key, HOLO.label(key))}</button>
      <div><div class="rank">RANK // 0${k + 1}</div>
      <div class="gname">${g === "อื่น ๆ" ? `<span style="font-style:normal">${name}</span>` : esc(name)} ${thai ? `<span class="thai">(${esc(thai)})</span>` : ""}</div>
      <div class="conf">${L("ความมั่นใจ", "Confidence")} <b>${confWord(p)}</b><span class="bar"><span style="width:${Math.round(p * 100)}%"></span></span>${p < 0.005 ? "&lt;1" : Math.round(p * 100)}%</div>
      ${r.species[k] ? `<div class="sp">${L("ชนิดที่พบบ่อย", "Most common species")}: <i>${esc(r.species[k])}</i></div>` : ""}
      ${ageLine(g, grp || (state.pp && state.pp.genus_group[g]) || null)}</div></div>`;
  }).join("");
  const known = r.route === "แหล่งเดิม";
  const where = (state.spec && state.spec.where) || "none";        // site / point / province / none
  const noloc = where === "none";
  const route = known ? L("ลักษณะซาก + ข้อมูลของแหล่งนี้ ซึ่งเคยระบุชื่อซากแล้ว", "fossil features + records of this site, whose fossils have been identified before")
    : where === "site" ? L("ลักษณะซาก + ชั้นหินของแหล่งนี้", "fossil features + the rock unit of this site")
    : where === "point" ? L("ลักษณะซาก + ชั้นหินตรงจุดที่พบ", "fossil features + the rock unit at the find spot")
    : where === "province" ? L("ลักษณะซาก + จังหวัดที่พบ", "fossil features + the province")
    : L("ลักษณะซากอย่างเดียว (ไม่ทราบที่พบ)", "fossil features only (location unknown)");
  const look = r.mode === "appearance";
  // ความแม่นที่บอกผู้ใช้ = ผลวัดของเส้นทางที่ผู้ใช้ตอบจริง (notebook 15 หัวข้อ 7): แบบคำตอบ × บอก / ไม่บอกที่พบ
  //   ไม่บอกที่พบใช้ spatial_block (ระมัดระวังกว่า); พิมพ์แค่จังหวัดใช้ตัวเลขของกรณีบอกที่พบ (อาจสูงกว่าจริงเล็กน้อย)
  const LE = state.meta.look_eval || {};
  const ev = LE[known ? "random" : "spatial_block"] || {};
  const sfx = noloc ? "_noloc" : "";
  const full = look && !r.leaf.endsWith("/unsure") && r.leaf !== "unknown";
  const key = look ? (r.leaf === "unknown" ? "look_unknown" : full ? "look_full" : "look_card")
    : (state.spec && state.spec.hasCom ? "base" : "type_nocom");
  const acc = ev[key + sfx], accType = ev["type_nocom" + sfx];
  const pct = (x) => Math.round(x * 100);
  // ความแม่น = ผลทดสอบกับซากที่รู้ชื่อแล้ว: ชื่อที่ถูกอยู่ใน 3 อันดับแรกกี่ชิ้นจาก 100 + วิธีที่จะแม่นขึ้น (เลือกประเภทเอง / บอกที่พบ)
  const crit = state.crit[known ? "G1" : "G2"];
  const top3 = acc ? pct(acc.top3) : crit ? pct(crit["ผล"]) : null;
  const more = [];
  if (acc && look && accType) more.push(L(`ถ้าเลือกประเภทซากเองถูก: ${pct(accType.top3)} ชิ้น`, `choosing the right type yourself: ${pct(accType.top3)}`));
  if (acc && noloc) {                                                // บอกที่พบแล้วจะเป็นเท่าไร: พื้นที่ใหม่ – แหล่งที่เคยระบุชื่อซาก
    const lo = (LE.spatial_block || {})[key], hi = (LE.random || {})[key];
    if (lo && hi) more.push(L(`ถ้าบอกที่พบ: ${pct(lo.top3)}–${pct(hi.top3)} ชิ้น`, `giving the location: ${pct(lo.top3)}–${pct(hi.top3)}`));
  }
  const accText = top3 === null ? "" : `<div><b>${L("ความแม่น", "Accuracy")}</b> ${L(`ทดสอบ 100 ชิ้น ชื่อที่ถูกอยู่ใน 3 อันดับแรก ${top3} ชิ้น`,
    `tested on 100 fossils, the right name was in the top 3 for ${top3}`)}${more.length ? ` (${more.join(" · ")})` : ""}</div>`;
  const types = look ? `<div class="guess"><div class="guess-t">${L("จากหน้าตาที่เลือก ระบบคิดว่าเป็นซากของ", "From the appearance you chose, the fossil is probably:")}</div>
      ${r.types.map(([t, w]) => `<div class="guess-row"><span>${esc(tr(t))}</span><span class="bar"><span style="width:${pct(w)}%"></span></span><b>${pct(w) < 1 ? "&lt;1" : pct(w)}%</b></div>`).join("")}
    </div>` : "";
  return `<h2>${L("ซากนี้น่าจะเป็น", "This fossil is probably")}</h2>
    ${types}
    <div class="holo" id="holo-big"></div>
    <div class="holo-tools"><button type="button" class="holo-expand" id="holo-expand"><span aria-hidden="true">⤢</span> ${L("ขยายดูภาพใหญ่", "View larger")}</button></div>
    ${items}
    <div class="note"><div><b>${L("ทายจาก", "Based on")}</b> ${route}</div>${accText}</div>
    <p class="hint" style="margin-top:8px">${L("ผลนี้เป็นข้อเสนอเบื้องต้น ควรให้ผู้เชี่ยวชาญยืนยัน", "This is a preliminary suggestion; an expert should confirm it.")}</p>`;
}

function genusView(g, group) {
  // ชื่อที่แสดง + รูปร่าง hologram; "อื่น ๆ" แยกตามกลุ่ม (server ส่งกลุ่มมาเป็นค่าที่ 3 ของแต่ละอันดับ)
  const other = g === "อื่น ๆ";
  const own = state.pp && state.pp.genus_group[g];                  // อังกฤษ: ชื่อเรียกภาษาไทยไม่มีคำแปล ใช้กลุ่มซากของสกุลแทน
  return {
    name: other ? L("ยังระบุชื่อไม่ได้", "Not identifiable yet") : g,   // คลาส "อื่น ๆ" = สกุลที่มีตัวอย่างระบุชื่อแล้วไม่ถึง 5 ชิ้น (notebook 11)
    thai: other ? L(`${group ? `${group} · ` : ""}น่าจะเป็นสกุลที่ในไทยพบตัวอย่างไม่ถึง 5 ชิ้น`, `${group ? `${tr(group)} · ` : ""}probably a genus with fewer than 5 specimens found in Thailand`)
      : L((state.options.genus_names || {})[g], own ? tr(own) : ""),
    key: other ? "question" : HOLO.shapeFor(g, group),            // ยังระบุชื่อไม่ได้ = เครื่องหมาย ?
  };
}

function holoHud(k) {
  // รูปร่าง + ป้ายบนภาพของอันดับ k (ใช้ทั้งกรอบในผล และกรอบขยายดู)
  const s = state.spec, [g, p, grp] = s.r.top[k], group = grp || s.ftype, v = genusView(g, group);
  const other = g === "อื่น ๆ";
  return { key: v.key, hud: {
    // "ยังระบุชื่อไม่ได้": บนภาพใช้คำอธิบายสั้น (กลุ่มอยู่ที่ป้าย FORM, คำอธิบายเต็มอยู่ในรายการด้านล่าง) ไม่ให้ทับเครื่องหมาย ? บนจอมือถือ
    kicker: `RANK // 0${k + 1}`, name: other ? L(esc(v.name), "Unidentified") : esc(v.name), thai: other ? L("สกุลที่พบในไทย<br>ไม่ถึง 5 ชิ้น", "genus with fewer than<br>5 finds in Thailand") : esc(v.thai || ""),
    form: other && group ? esc(tr(group)) : null,
    match: `${p < 0.005 ? "<1" : Math.round(p * 100)}%`, word: L(`ความมั่นใจ${confWord(p)}`, `${confWord(p)} confidence`), aria: `${L("ภาพประกอบ", "Illustration:")} ${HOLO.label(v.key)}`,
  } };
}

function showHolo(k) {
  // แสดงอันดับ k เป็น hologram ใหญ่ (3D จุดแสง; เล่นภาพปรากฏใหม่ทุกครั้งที่สลับ) — ปุ่มใต้กรอบเปิดกรอบขยายดู (openHoloView)
  const box = $("#holo-big");
  if (!state.spec || !box) return;
  const { key, hud } = holoHud(k);
  HOLO.show(box, key, hud);
  state.spec.k = k;
  document.querySelectorAll("#spec-result .genus").forEach((el) => el.classList.toggle("on", +el.dataset.k === k));
}

function openHoloView(k) {
  // ขยายดู: กรอบใหญ่กลางจอ ตัวสัตว์ใหญ่และจุดหนาแน่นขึ้น — ปิดด้วยปุ่ม × / Esc / คลิกพื้นหลัง
  closeHoloView();
  const { key, hud } = holoHud(k);
  const wrap = document.createElement("div");
  wrap.className = "holo-modal";
  wrap.setAttribute("role", "dialog");
  wrap.setAttribute("aria-modal", "true");
  wrap.setAttribute("aria-label", `${L("ขยายดู", "Enlarged view")}: ${HOLO.label(key)}`);
  wrap.innerHTML = `<div class="holo holo-viewbox"></div>
    <button type="button" class="holo-close" aria-label="${L("ปิด", "Close")}">×</button>
    <p class="holo-modal-tip">${matchMedia("(hover: hover)").matches ? L("เลื่อนเมาส์บนภาพเพื่อหมุนดูรอบตัว", "Move the mouse over the image to turn it")
      : L("แตะลากบนภาพเพื่อหมุนดูรอบตัว", "Drag on the image to turn it")}${matchMedia("(orientation: portrait) and (max-width: 700px)").matches
      ? L(" · หมุนเครื่องเป็นแนวนอนเพื่อดูภาพใหญ่ขึ้น", " · turn your phone sideways for a bigger view") : ""}</p>`;
  document.body.appendChild(wrap);
  document.body.classList.add("holo-open");
  const box = wrap.querySelector(".holo-viewbox");
  HOLO.show(box, key, hud, { view: true });
  wrap.addEventListener("click", (e) => { if (e.target === wrap || e.target.closest(".holo-close")) closeHoloView(); });
  wrap.querySelector(".holo-close").focus();
  state.holoView = { wrap, box };
}

function closeHoloView() {
  const v = state.holoView;
  if (!v) return;
  if (v.box._stop) v.box._stop();
  v.wrap.remove();
  document.body.classList.remove("holo-open");
  state.holoView = null;
  const btn = document.querySelector("#holo-expand");
  if (btn) btn.focus();
}

function mountSpecimen(out, k = 0) {
  out.querySelectorAll(".holo-mini .holo-svg").forEach(HOLO.fit);
  out.querySelectorAll(".genus.pick").forEach((el) => el.addEventListener("click", () => showHolo(+el.dataset.k)));
  const ex = out.querySelector("#holo-expand");
  if (ex) ex.addEventListener("click", () => openHoloView(state.spec.k || 0));
  showHolo(k);
}

async function submitSpecimen(ev) {
  ev.preventDefault();
  const f = new FormData(ev.target), out = $("#spec-result");
  const val = (k) => (f.get(k) || "").trim() || null;
  // ค่าที่ส่งให้โมเดลเป็นภาษาไทยเสมอ: จังหวัด / ส่วนของซากที่พิมพ์หรือเลือกเป็นภาษาอังกฤษ แปลงกลับด้วย I18N.toThai
  const body = { com_name: val("com_name"), province: I18N.toThai(val("province")), district: val("district") };
  out.hidden = false;
  if (state.look.mode === "look") {                                    // ไม่รู้ประเภท: ส่งคำตอบเรื่องหน้าตา ระบบเดาประเภทเอง
    if (!state.look.card) { out.innerHTML = `<div class="err">${L("กรุณาเลือกภาพที่คล้ายซากที่พบที่สุด (ข้อ 01)", "Please pick the picture that looks most like your find (step 01)")}</div>`; return; }
    body.look_card = state.look.card;
    body.look_option = state.look.option;
  } else {
    Object.assign(body, { f_type: val("f_type"), f_part: I18N.toThai(val("f_part")) });
    body.f_group = (state.options.type_group || {})[body.f_type] || null;            // กลุ่มซาก: ระบบเติมให้จากประเภท
    if (!body.f_type) { out.innerHTML = `<div class="err">${L("กรุณาเลือกประเภทซาก (ข้อ 01) หรือกลับไปเลือกจากหน้าตา", "Please choose a fossil type (step 01) or go back to choosing by appearance")}</div>`; return; }
  }
  const loc = f.get("loc");
  if (loc === "site") {
    const s = siteFromInput();
    if (!s) { out.innerHTML = `<div class="err">${L("เลือกแหล่งซากจากรายการ หรือเลือกตัวเลือกไม่ทราบ", "Pick a fossil site from the list, or choose Unknown")}</div>`; return; }
    body.site_OBJECTID = s.id;
    body.province = body.province || s.province;
  } else if (loc === "point") {
    if (!state.point) { out.innerHTML = `<div class="err">${L("คลิกเลือกจุดบนแผนที่ก่อน หรือเลือกตัวเลือกไม่ทราบ", "Click a point on the map first, or choose Unknown")}</div>`; return; }
    body.lat = state.point.lat; body.lon = state.point.lon;
  }
  const btn = ev.target.querySelector("button[type=submit]");
  btn.disabled = true;
  out.innerHTML = `<div class="status"><span class="spin"></span>${L("กำลังทายชื่อซาก…", "Identifying the fossil…")}</div>`;
  try {
    if (STATIC && !(await JurassicLocal.textCached(body).catch(() => true))) {
      // ข้อความที่ไม่มีในตาราง: โหลดโมเดลภาษา WangchanBERTa ครั้งแรก (~210 MB) — ครั้งต่อไปเบราว์เซอร์ใช้ของเดิม
      const st = out.querySelector(".status");
      const msg = (pct) => `<span class="spin"></span>${L(`กำลังโหลดโมเดลภาษาไทย WangchanBERTa สำหรับคำที่พิมพ์เอง (ราว 210 MB เฉพาะครั้งแรก)… ${pct}%`,
        `Loading the Thai language model WangchanBERTa for the words you typed (about 210 MB, first time only)… ${pct}%`)}`;
      st.innerHTML = msg(0);
      JurassicLocal.cfg.onBertProgress = (done, total) => { st.innerHTML = msg(Math.round((100 * done) / total)); };
    }
    const r = await api("/api/predict_specimen", body);
    await state.ppLoading;                                           // ข้อมูล "พบในชั้นหินยุคไหน" (โหลดตั้งแต่เปิดหน้า)
    const ftype = body.f_type || (r.types && r.types[0][0]);          // รูปร่าง hologram ของ "อื่น ๆ"
    const where = body.site_OBJECTID ? "site" : body.lat != null ? "point" : body.province ? "province" : "none";
    state.spec = { r, ftype, hasCom: !!body.com_name, where };
    out.innerHTML = renderSpecimen(r, ftype);
    mountSpecimen(out);
    out.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    if (err instanceof BusyError) {
      out.innerHTML = `<div class="busy-slot"></div>`;
      showBusy(out.querySelector(".busy-slot"), err.retryAfter, () => ev.target.requestSubmit());
      return;
    }
    out.innerHTML = `<div class="err">${L("ทายไม่สำเร็จ", "Identification failed")}: ${esc(err.message)}</div>`;
  } finally {
    btn.disabled = false;
  }
}

// ---------------------------------------------------------------- ข้อ 01: หน้าตาซาก (การ์ดรูป + คำถามต่อ)
// ข้อความของการ์ด / ตัวเลือก: ภาษาไทยจาก server (feature_lib.APPEARANCE), ภาษาอังกฤษจาก i18n.js ตาม id (ไม่มีคำแปล = ใช้ภาษาไทย)
const pick = (en, th) => (en === undefined ? th : en);
const cardText = (c, f, th) => pick(I18N.look(c.id, f), th);
const optText = (c, o, f, th) => pick(I18N.lookOpt(c.id, o.id, f), th);

function renderLook(cards) {
  $("#look-cards").after($("#look-ask"));                         // คำถามต่ออาจถูกย้ายไปอยู่ในรายการการ์ด: ย้ายออกก่อนวาดใหม่
  const groups = [...new Set(cards.map((c) => c.group))];
  $("#look-cards").innerHTML = groups.map((g) => `<div class="look-g">${esc(I18N.lookGroup(g))}</div><div class="look-grid">
    ${cards.filter((c) => c.group === g).map((c) => { const lab = cardText(c, 0, c.label), hint = cardText(c, 1, c.hint);
      return `<button type="button" class="look-card" data-card="${c.id}" aria-pressed="false">
      <span class="look-ic">${HOLO.svg(c.icon, esc(lab))}</span><b>${esc(lab)}</b>${hint ? `<small>${esc(hint)}</small>` : ""}</button>`; }).join("")}
  </div>`).join("");
  $$("#look-cards .look-card").forEach((b) => b.addEventListener("click", () => pickCard(b.dataset.card)));
}

function pickCard(id, keep = false) {
  // keep = วาดใหม่หลังเปลี่ยนภาษา: คงตัวเลือกเดิม ไม่เลื่อนจอ
  const c = state.options.appearance.find((x) => x.id === id);
  Object.assign(state.look, { card: id, option: keep ? state.look.option : null });
  $$("#look-cards .look-card").forEach((b) => { const on = b.dataset.card === id; b.classList.toggle("on", on); b.setAttribute("aria-pressed", on); });
  const ask = $("#look-ask");
  $(`.look-card[data-card="${id}"]`).closest(".look-grid").after(ask);   // คำถามต่ออยู่ใต้กลุ่มของการ์ดที่เลือก
  ask.hidden = !c.options.length;
  if (!c.options.length) return;
  $("#look-q").innerHTML = `${esc(cardText(c, 2, c.ask))} <em>${L("เลือกได้ถ้าเห็นชัด", "optional, if clearly visible")}</em>`;
  $("#look-opts").innerHTML = c.options.map((o) => { const lab = optText(c, o, 0, o.label), hint = optText(c, o, 1, o.hint);
    return `<button type="button" class="look-opt ${o.id === state.look.option ? "on" : ""}" data-opt="${o.id}" aria-pressed="${o.id === state.look.option}">
    ${o.icon ? `<span class="look-ic sm">${HOLO.svg(o.icon, esc(lab))}</span>` : ""}<span><b>${esc(lab)}</b>${hint ? `<small>${esc(hint)}</small>` : ""}</span></button>`; }).join("");
  $$("#look-opts .look-opt").forEach((b) => b.addEventListener("click", () => {
    state.look.option = b.dataset.opt;
    $$("#look-opts .look-opt").forEach((x) => { const on = x === b; x.classList.toggle("on", on); x.setAttribute("aria-pressed", on); });
  }));
  $$("#look-opts .holo-svg").forEach(HOLO.fit);
  if (!keep) ask.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function fitLookIcons() {
  // ภาพบนการ์ดต้องปรับกรอบตอนมองเห็นแล้ว (ตอนซ่อนอยู่วัดขนาดไม่ได้)
  if (state.look.fitted || !$("#look-cards").offsetParent) return;
  $$("#look-cards .holo-svg").forEach(HOLO.fit);
  state.look.fitted = true;
}

function setLookMode(mode) {
  state.look.mode = mode;
  $("#look-mode").hidden = mode !== "look";
  $("#type-mode").hidden = mode !== "type";
  if (mode === "look") requestAnimationFrame(fitLookIcons);
}

// ---------------------------------------------------------------- INDEX + MORE
const ICONS = {
  acc: `<svg viewBox="0 0 44 44" fill="none" stroke="currentColor" stroke-width="2.4"><rect x="8" y="8" width="28" height="28" transform="rotate(45 22 22)"/><path d="M14 26l6-6 5 4 6-8"/></svg>`,
  read: `<svg viewBox="0 0 44 44" fill="none" stroke="currentColor" stroke-width="2.4"><path d="M8 34h28M12 30V20M20 30V12M28 30V22M36 30V16"/></svg>`,
  data: `<svg viewBox="0 0 44 44" fill="none" stroke="currentColor" stroke-width="2.4"><ellipse cx="22" cy="11" rx="13" ry="5"/><path d="M9 11v22c0 3 6 5 13 5s13-2 13-5V11M9 22c0 3 6 5 13 5s13-2 13-5"/></svg>`,
  find: `<svg viewBox="0 0 44 44" fill="none" stroke="currentColor" stroke-width="2.4"><path d="M22 6l14 8v16l-14 8-14-8V14z"/><path d="M22 16v10M22 30v2"/></svg>`,
};

function moreCards() {
  const c = state.crit, o = state.options, k = state.meta && state.meta.click;
  const s = (cid) => (c[cid] ? Math.round(c[cid]["ผล"] * 100) : "–");
  const nS = o.sites.length.toLocaleString(), nP = o.n_specimens.toLocaleString();
  return [
    { key: "acc", img: "p_roc.png", th: L("ความแม่นของระบบ", "How accurate it is"), en: "ACCURACY", kicker: "ACCURACY // 01",
      body: `<p>${L("ทดสอบกับข้อมูลที่ระบบไม่เคยเห็นตอนเรียนรู้", "Tested on data the system never saw while learning")}</p>
        <div class="mstat"><span class="big">${k ? Math.round(k.share_at_or_above.high * 100) : "–"}<small>/100</small></span><span>${L(
          "ทดสอบกดจุดที่มีซากจริง: ระบบบอกว่าโอกาสสูงถูก",
          "Clicking spots with real fossils: the map rates them high")}</span></div>
        ${k && k.new_area ? `<div class="mstat"><span class="big">${Math.round(k.new_area.share_at_or_above.high * 100)}<small>/100</small></span><span>${L(
          `เฉพาะพื้นที่ใหม่ที่ ทธ. ยังไม่มีข้อมูล (ห่างแหล่งที่ขึ้นทะเบียนเกิน ${k.new_area.min_km} กม.): ระบบบอกว่าโอกาสสูงถูก — ถ้าสุ่มเดาจะถูก ${Math.round(k.random.high * 100)}`,
          `New areas the DMR has no record of (over ${k.new_area.min_km} km from any registered site): rated high — random guessing gets ${Math.round(k.random.high * 100)}`)}</span></div>` : ""}
        <div class="mstat"><span class="big">${s("B2")}<small>/100</small></span><span>${L("บอกชนิดซากถูก (ชนิดจริงอยู่ใน 3 ชนิดที่ระบบเสนอ)",
          "Fossil type right (the real type is among the 3 suggested)")}</span></div>
        <div class="mstat"><span class="big">${s("G1")}<small>/100</small></span><span>${L("ทายชื่อซากในแหล่งที่เคยระบุชื่อแล้ว: ชื่อที่ถูกอยู่ใน 3 อันดับแรก", "Naming fossils at sites with identified fossils: correct name in the top 3")}</span></div>
        <div class="mstat"><span class="big">${s("G2")}<small>/100</small></span><span>${L("ทายชื่อซากในพื้นที่ใหม่: ชื่อที่ถูกอยู่ใน 3 อันดับแรก", "Naming fossils in new areas: correct name in the top 3")}</span></div>
        <div class="mstat"><span class="big">${s("P2")}<small>/100</small></span><span>${L("ทายยุคของแหล่งในพื้นที่ใหม่ได้คลาดไม่เกิน 1 ยุค", "Site period in new areas predicted within one period")}</span></div>` },
    { key: "read", img: "p_map.png", th: L("วิธีอ่านผล", "Reading the results"), en: "HOW TO READ", kicker: "GUIDE // 02",
      body: `<p>${L("สีบนแผนที่เทียบแต่ละพื้นที่ 5 × 5 กม. กับพื้นที่อื่นทั้งประเทศ", "Map colours compare each 5 × 5 km area with every other area in the country")}</p>
        <div class="mlevels">${LEVELS.map((l) => `<div class="row"><span class="sw" style="background:${l.color}"></span><b>${l.label}</b> — ${l.text}</div>`).join("")}
        <div class="row"><span class="sw" style="background:${NONE.color}"></span><b>${NONE.label}</b> — ${NONE.text}</div></div>
        <p>${L(`• พื้นที่ระดับ<b>ต่ำ</b> ไม่ได้แปลว่าไม่มีซาก แค่ยังไม่มีหลักฐานว่าคล้ายแหล่งที่รู้จัก<br>
           • <b>ข้อมูลไม่พอ</b> คือพื้นที่ที่ต่างจากทุกแหล่งที่ระบบเคยเรียนรู้ ผลจึงเชื่อถือได้น้อย<br>
           • ผลทั้งหมดเป็นการคาดการณ์จากคอมพิวเตอร์ ไม่ใช่การยืนยันว่ามีซาก`,
          `• <b>Low</b> does not mean there are no fossils, only that there is no evidence yet that the area resembles known sites<br>
           • <b>Not enough data</b> marks areas unlike any site the system learned from, so results there are less reliable<br>
           • All results are computer predictions, not confirmation that fossils are present`)}</p>` },
    { key: "data", img: "p_data.png", th: L("แหล่งข้อมูล", "Where the data comes from"), en: "DATA SOURCES", kicker: "DATA // 03",
      body: `<p>${L(`ระบบเรียนรู้จาก<b>แหล่งซาก ${nS} แห่ง</b>ที่ขึ้นทะเบียนกับกรมทรัพยากรธรณี (จุด<i class="sw-dot sw-site"></i>ในภาพ)
          และ<b>ตัวอย่างซาก ${nP} ชิ้น</b> ตรวจซ้ำกับจุดที่นักวิจัยรายงานในฐานข้อมูล Paleobiology Database (จุด<i class="sw-dot sw-pbdb"></i>)`,
          `The system learned from <b>${nS} fossil sites</b> registered with the Department of Mineral Resources (<i class="sw-dot sw-site"></i>dots in the image)
          and <b>${nP} fossil specimens</b>, cross-checked against finds reported by researchers in the Paleobiology Database (<i class="sw-dot sw-pbdb"></i>dots)`)}</p>
        <p>${L(`ข้อมูลประกอบ: แผนที่ธรณีวิทยา 1:250,000 ของกรมทรัพยากรธรณี, ภาพถ่ายดาวเทียม Sentinel-2, แบบจำลองความสูง Copernicus,
          แบบจำลองการเคลื่อนของแผ่นเปลือกโลก (EarthByte) และพจนานุกรมหน่วยหินไทย ThaiLex`,
          `Supporting data: the DMR 1:250,000 geological map, Sentinel-2 satellite imagery, the Copernicus elevation model,
          a plate-tectonic reconstruction (EarthByte) and the ThaiLex lexicon of Thai rock units`)}</p>
        ${dataDates()}` },
    { key: "find", img: "p_rock.png", th: L("พบซากต้องทำอย่างไร", "If you find a fossil"), en: "IF YOU FIND ONE", kicker: "NOTICE // 04",
      body: L(`<p>ซากดึกดำบรรพ์ได้รับความคุ้มครองตาม<b>พระราชบัญญัติคุ้มครองซากดึกดำบรรพ์ พ.ศ. 2551</b></p>
        <p>• ไม่ควรขุด เคลื่อนย้าย หรือเก็บซากเอง<br>• ถ่ายภาพและจดตำแหน่งที่พบ<br>• แจ้งกรมทรัพยากรธรณี เพื่อให้ผู้เชี่ยวชาญตรวจสอบ</p>
        <p class="hint">ภาพพื้นหลัง: อายุหินทั่วประเทศจากแผนที่ธรณีวิทยา (<i class="sw-dot sw-old"></i>= หินเก่า, <i class="sw-dot sw-young"></i>= หินอายุน้อย)</p>`,
        `<p>Fossils are protected under Thailand's <b>Fossil Protection Act B.E. 2551 (2008)</b></p>
        <p>• Do not dig up, move or keep the fossil yourself<br>• Photograph it and note where it was found<br>• Report it to the Department of Mineral Resources so an expert can examine it</p>
        <p class="hint">Background image: rock ages across the country from the geological map (<i class="sw-dot sw-old"></i>= old rock, <i class="sw-dot sw-young"></i>= young rock)</p>`) },
  ];
}

function dataDates() {
  // รายการ "ข้อมูล ณ วันที่" ของแต่ละแหล่งข้อมูล (หน้า MORE: แหล่งข้อมูล)
  const dd = (state.meta && state.meta.data_dates) || {};
  const rows = Object.values(dd).map((d) => `<li>${esc(L(d.th, d.en))}: <b>${d.date ? fmtDate(d.date) : `${fmtDate(d.from, true)} – ${fmtDate(d.to, true)}`}</b></li>`);
  return rows.length ? `<p><b>${L("ข้อมูล ณ วันที่", "Data as of")}</b></p><ul class="asof-list">${rows.join("")}</ul>
    <p class="hint">${L("ทะเบียนตัวอย่าง = วันที่ของไฟล์จาก ทธ.; รายการอื่น = วันที่ดาวน์โหลด", "Specimen register = date of the DMR file; others = download date")}</p>` : "";
}

function renderMore() {
  const cards = moreCards();
  $("#more-cols").innerHTML = cards.map((c, k) => `<article class="col" data-k="${k}" data-key="${c.key}" tabindex="0">
      <div class="bg" style="background-image:url('/static/img/${c.img}')"></div>
      <div class="idx">0${k + 1} // 04</div>
      <div class="ct"><div class="ico">${ICONS[c.key]}</div><h3>${c.th}</h3><div class="en">${c.en}</div>
        <div class="vm">VIEW MORE ›</div><div class="ul"></div></div></article>`).join("");
  $$("#more-cols .col").forEach((el) => {
    const open = () => openModal(+el.dataset.k);
    el.addEventListener("click", open);
    el.addEventListener("keydown", (e) => { if (e.key === "Enter") open(); });
  });
}

function openModal(k) {
  const c = moreCards()[k];
  state.modalK = k;
  $("#modal-kicker").textContent = c.kicker;
  $("#modal-title").textContent = c.th;
  $("#modal-body").innerHTML = c.body;
  $("#modal").hidden = false;
}

function renderStats() {
  const o = state.options, k = state.meta && state.meta.click;
  const items = [
    [o.sites.length.toLocaleString(), "", L("แหล่งซากที่ขึ้นทะเบียน", "Registered fossil sites"), "REGISTERED SITES"],
    [o.n_specimens.toLocaleString(), "", L("ตัวอย่างซากในทะเบียน", "Registered specimens"), "SPECIMENS"],
    // ความแม่นเมื่อกดแผนที่ในพื้นที่ใหม่: จุดที่พบซากจริง (PBDB อิสระ) ที่ห่างแหล่ง ทธ. ทุกแหล่งเกิน 5 กม. ระบบบอกโอกาส "สูง" ขึ้นไปกี่จุดจาก 100
    //   (notebook 16 -> web_click_stats.json; สุ่มเดาได้ 25)
    [k ? Math.round((k.new_area || k).share_at_or_above.high * 100) : "–", "/100",
      L("ชี้ถูกในพื้นที่ใหม่ ที่ ทธ. ยังไม่มีข้อมูล", "Right in areas new to the DMR"), "NEW-AREA HIT RATE"],
  ];
  $("#stats").innerHTML = items.map(([n, u, t, e]) => `<div class="stat"><div class="n">${n}<small>${u}</small></div><div class="t">${t}</div><div class="e">${e}</div></div>`).join("");
  // ข้อมูล ณ วันที่ (data/source_dates.json + ช่วงวันที่ของภาพดาวเทียมผ่าน /api/meta; เว็บแบบ static บอกวันที่ build ด้วย)
  const dd = (state.meta && state.meta.data_dates) || {}, asof = [];
  if (dd.registry) asof.push(L(`ทะเบียนตัวอย่าง ทธ. ${fmtDate(dd.registry.date)}`, `DMR specimen register ${fmtDate(dd.registry.date)}`));
  if (dd.sites) asof.push(L(`แหล่งซาก ทธ. / PBDB / แผนที่ธรณี ${fmtDate(dd.sites.date)}`, `DMR sites / PBDB / geological map ${fmtDate(dd.sites.date)}`));
  if (dd.sentinel2) asof.push(L(`ภาพดาวเทียม ${fmtDate(dd.sentinel2.from, true)} – ${fmtDate(dd.sentinel2.to, true)}`,
    `satellite imagery ${fmtDate(dd.sentinel2.from, true)} – ${fmtDate(dd.sentinel2.to, true)}`));
  if (STATIC && STATIC.built) asof.push(L(`ปรับปรุงเว็บ ${fmtDate(STATIC.built)}`, `site updated ${fmtDate(STATIC.built)}`));
  $("#asof").innerHTML = asof.length ? `<b>${L("ข้อมูล ณ", "Data as of")}</b> ${asof.join(" · ")}` : "";
}

// ---------------------------------------------------------------- UI
function wireControls() {
  $$("[data-go]").forEach((el) => el.addEventListener("click", (e) => { e.preventDefault(); navigate(el.dataset.go); }));
  window.addEventListener("popstate", () => navigate(location.hash.slice(1), false));
  try { if (localStorage.getItem("pj-intro-hidden") === "1") $("#intro").hidden = true; } catch (e) { /* ไม่มี storage */ }
  $("#intro-close").addEventListener("click", () => {
    $("#intro").hidden = true;
    try { localStorage.setItem("pj-intro-hidden", "1"); } catch (e) { /* ไม่มี storage */ }
  });
  renderPlaces();
  $("#theme-btn").addEventListener("click", switchTheme);
  $("#lang-btn").addEventListener("click", () => I18N.set(I18N.lang === "en" ? "th" : "en"));
  prefButtons();
  $("#locate").addEventListener("click", () => {
    if (!navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition((pos) => {
      const { latitude: la, longitude: lo } = pos.coords;
      const inThai = lo >= THAI_BOUNDS[0][0] && lo <= THAI_BOUNDS[1][0] && la >= THAI_BOUNDS[0][1] && la <= THAI_BOUNDS[1][1];
      if (inThai) goTo(la, lo);
      else alert(L("ตำแหน่งของคุณอยู่นอกประเทศไทย ระบบวิเคราะห์ได้เฉพาะในประเทศไทย", "Your location is outside Thailand; the system only works within Thailand."));
    }, () => alert(L("ไม่สามารถหาตำแหน่งได้ (อาจไม่ได้อนุญาตให้เข้าถึงตำแหน่ง)", "Could not get your location (location access may not be allowed).")));
  });
  $$("#mode button").forEach((b) => b.addEventListener("click", () => {
    state.mode = b.dataset.mode;
    $$("#mode button").forEach((x) => x.classList.toggle("on", x === b));
    updateLayer();
  }));
  $("#type-select").addEventListener("change", (e) => { state.type = e.target.value; updateLayer(); });
  $$("#bg button").forEach((b) => b.addEventListener("click", () => {
    state.bg = b.dataset.bg;
    $$("#bg button").forEach((x) => x.classList.toggle("on", x === b));
    updateLayer();
  }));
  $("#aoa").addEventListener("change", (e) => { state.aoa = e.target.checked; updateLayer(); });
  $("#smooth").addEventListener("change", (e) => { state.smooth = e.target.checked; updateLayer(); });
  $("#show-sites").addEventListener("change", (e) => map.setLayoutProperty("sites", "visibility", e.target.checked ? "visible" : "none"));
  $("#show-geo").addEventListener("change", async (e) => {
    state.geo.on = e.target.checked;
    $("#geo-panel").hidden = !state.geo.on;
    if (state.geo.on && !state.geo.data) { try { await loadGeology(); } catch (err) { return; } }
    if (state.ready) applyGeo();
    if (state.geo.on) $("#geo-panel").scrollIntoView({ behavior: "smooth", block: "nearest" });
  });
  const setSel = (f) => { if (!state.geo.data) return; state.geo.sel = new Set(state.geo.data.periods.filter(f).map((p) => p.period)); applyGeo(); };
  $("#geo-all").addEventListener("click", () => setSel(() => true));
  $("#geo-none").addEventListener("click", () => setSel(() => false));
  $("#geo-fossil").addEventListener("click", () => setSel((p) => p.sites > 0));
  $("#show-pbdb").addEventListener("change", (e) => map.setLayoutProperty("pbdb", "visibility", e.target.checked ? "visible" : "none"));
  $("#site-input").addEventListener("change", onSiteChange);
  $("#spec-form").addEventListener("submit", submitSpecimen);
  const closeModal = () => { $("#modal").hidden = true; state.modalK = null; };
  $("#modal-close").addEventListener("click", closeModal);
  $("#modal").addEventListener("click", (e) => { if (e.target.id === "modal") closeModal(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closeHoloView(); closeModal(); } });
}

function renderPlaces() {
  $("#places").innerHTML = PLACES.map(([n, , , en], k) => `<button class="chip" data-k="${k}">${esc(L(n, en))}</button>`).join("");
  $$("#places .chip").forEach((b) => b.addEventListener("click", () => { const [, la, lo] = PLACES[+b.dataset.k]; goTo(la, lo); }));
}

function prefButtons() {
  // ปุ่มบนแถบบน: แสดงสิ่งที่จะเปลี่ยนไป (โหมดมืดแสดงดวงอาทิตย์ = กดเพื่อเป็นโหมดสว่าง, ภาษาไทยแสดง EN = กดเพื่อเป็นภาษาอังกฤษ)
  const dark = THEME.theme === "dark", th = I18N.lang === "th";
  const tb = $("#theme-btn"), lb = $("#lang-btn");
  // ไอคอน SVG (ไม่ใช้ตัวอักษร ☀ / ☾ ซึ่งแต่ละฟอนต์วางไม่ตรงกลางวงกลม)
  tb.innerHTML = dark
    ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4.2"/><path d="M12 2.5v2.6M12 18.9v2.6M2.5 12h2.6M18.9 12h2.6M5.3 5.3l1.8 1.8M16.9 16.9l1.8 1.8M5.3 18.7l1.8-1.8M16.9 7.1l1.8-1.8"/></svg>`
    : `<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M20.2 14.6A8.4 8.4 0 0 1 9.4 3.8a8.4 8.4 0 1 0 10.8 10.8z"/></svg>`;
  tb.title = dark ? L("โหมดสว่าง", "Light mode") : L("โหมดมืด", "Dark mode");
  tb.setAttribute("aria-label", dark ? L("เปลี่ยนเป็นโหมดสว่าง", "Switch to light mode") : L("เปลี่ยนเป็นโหมดมืด", "Switch to dark mode"));
  lb.textContent = th ? "EN" : "ไทย";
  lb.title = th ? "English" : "ภาษาไทย";
  lb.setAttribute("aria-label", th ? "Switch to English" : "เปลี่ยนเป็นภาษาไทย");
  lb.lang = th ? "en" : "th";
}

// ---------------------------------------------------------------- ข้อมูลในฟอร์มที่ขึ้นกับภาษา
function fillTypeSelect() {
  const cur = $("#type-select").value || state.type;
  $("#type-select").innerHTML = state.meta.classes_B.map((c) => `<option value="${esc(c)}">${typeIcon(c)} ${esc(tr(c))}</option>`).join("");
  $("#type-select").value = cur;
}

function fillFType() {
  // ค่า (value) เป็นภาษาไทยตามทะเบียนเสมอ; ชื่อภาษาอังกฤษที่ซ้ำกัน (เช่น ฟอแรมมินิเฟอร์ / ฟอแรมินิเฟอร์) ต่อท้ายด้วยชื่อไทยให้แยกได้
  const sel = $("#f_type"), cur = sel.value, vals = state.options.f_type;
  const count = {};
  vals.forEach((v) => { const e = tr(v); count[e] = (count[e] || 0) + 1; });
  sel.innerHTML = `<option value="">${L("— เลือก —", "— Choose —")}</option>`
    + vals.map((v) => { const e = tr(v); return `<option value="${esc(v)}">${esc(count[e] > 1 && e !== v ? `${e} (${v})` : e)}</option>`; }).join("");
  sel.value = cur;
}

function renderParts() {
  const cur = $("#f_part").value;
  const parts = PART_PICKS.filter((p) => state.options.parts.includes(p));
  $("#part-chips").innerHTML = parts.map((p) => `<button type="button" class="chip" data-v="${esc(p)}">${esc(tr(p))}</button>`).join("");
  const sync = () => $$("#part-chips .chip").forEach((x) => x.classList.toggle("on", x.dataset.v === I18N.toThai($("#f_part").value.trim())));
  $$("#part-chips .chip").forEach((b) => b.addEventListener("click", () => {
    const on = !b.classList.contains("on");
    $("#f_part").value = on ? tr(b.dataset.v) : "";                // แสดงตามภาษา ส่งให้โมเดลเป็นภาษาไทย (I18N.toThai ตอนส่ง)
    sync();
  }));
  if (cur) $("#f_part").value = tr(I18N.toThai(cur));
  sync();
}

function fillProvinces() {
  // จังหวัด: ภาษาอังกฤษแสดงชื่ออังกฤษ (แปลงกลับเป็นไทยตอนส่ง)
  const inp = $('input[name="province"]');
  if (inp.value) inp.value = tr(I18N.toThai(inp.value));
  $("#dl-province").innerHTML = state.options.province.map((v) => `<option value="${esc(tr(v))}"></option>`).join("");
}

function hoverPopup(layer, html) {
  const pop = new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 8 });
  map.on("mouseenter", layer, (e) => { map.getCanvas().style.cursor = "pointer"; pop.setLngLat(e.lngLat).setHTML(html(e.features[0].properties)).addTo(map); });
  map.on("mouseleave", layer, () => { map.getCanvas().style.cursor = ""; pop.remove(); });
}

async function init() {
  wireControls();
  if (STATIC) $("#with-s2").closest("label").style.display = "none";  // เว็บแบบ static: ผลกริดคำนวณด้วยภาพดาวเทียมไว้แล้ว ไม่มีให้เลือก
  go(location.hash.slice(1) || "index", false);
  // หน้าแรก: ภาพ 3 มิติ (hero.js) เริ่มทันทีที่ได้ข้อมูลตัวอย่าง ไม่ต้องรอกริด
  state.ppLoading = api("/api/period_profile").then((d) => { state.pp = d; }).catch(() => {});
  // ภาพสำรอง (img.scan-fallback) ซ่อนไว้ระหว่างโหลด และแสดงเฉพาะเมื่อภาพ 3 มิติเริ่มไม่ได้ (เดิมกะพริบขึ้นทุกครั้งที่รีเฟรช)
  const noHero = () => { if ($("#hero-scan .scan-fallback")) $("#hero-scan").classList.add("no3d"); };
  const showcase = api("/api/showcase").then((sc) => HERO.start($("#hero-scan"), sc, { range: eraRange })).then(noHero, noHero);
  const data = Promise.all([api("/api/meta"), api("/api/grid"), api("/api/pbdb"), api("/api/options"), api("/api/provinces")]);
  try {
    const [meta, grid, pbdb, options, prov] = await data;
    Object.assign(state, { meta, grid, options, prov, nb: buildNeighbours(grid), type: meta.classes_B[0],
      crit: Object.fromEntries((meta.criteria || []).map((c) => [c["รหัส"], c])) });
    renderStats();
    renderMore();
    try { sessionStorage.removeItem("pj-busy-tries"); } catch (e) { /* หน้าขอโทษ (busy.html) นับรอบรอใหม่ */ }
    showcase.then(() => HERO.setMap(grid, options.sites));
    await new Promise((res) => (map.loaded() ? res() : map.on("load", res)));
    fillTypeSelect();
    prov.labels.features.forEach((f) => { f.properties.name_en = I18N.en(f.properties.name); });
    // พื้นผิวโอกาสพบซาก (ภาพ, renderSurface) -> เส้นจังหวัด -> เส้นชายแดน -> ชื่อจังหวัด ซ้อนกันตามลำดับ
    const [lo0, la0, lo1, la1] = SURF_BOUNDS;
    map.addSource("pot", { type: "image", url: "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7",
      coordinates: [[lo0, la1], [lo1, la1], [lo1, la0], [lo0, la0]] });
    map.addLayer({ id: "pot", type: "raster", source: "pot",
      paint: { "raster-opacity": ["interpolate", ["linear"], ["zoom"], 7, 0.92, 10, 0.6], "raster-fade-duration": 0, "raster-resampling": "linear" } });
    map.addSource("prov", { type: "geojson", data: prov.polygons });
    map.addSource("prov-labels", { type: "geojson", data: prov.labels });
    map.addSource("country", { type: "geojson", data: prov.country });
    map.addLayer({ id: "prov-hit", type: "fill", source: "prov", paint: { "fill-color": "#000", "fill-opacity": 0 } });   // ใช้หาจังหวัดตอนชี้เมาส์
    map.addLayer({ id: "prov-line", type: "line", source: "prov",
      paint: { ...themePaint("prov-line"), "line-opacity": ["interpolate", ["linear"], ["zoom"], 5, 0.55, 9, 0.75], "line-width": ["interpolate", ["linear"], ["zoom"], 5, 0.6, 9, 1.5] } });
    map.addLayer({ id: "country-line", type: "line", source: "country",
      paint: { ...themePaint("country-line"), "line-opacity": 0.9, "line-width": ["interpolate", ["linear"], ["zoom"], 5, 1.3, 9, 2.4] } });

    map.addSource("sites", { type: "geojson", data: { type: "FeatureCollection",
      features: options.sites.map((s) => ({ type: "Feature", properties: { name: s.name }, geometry: { type: "Point", coordinates: [s.lon, s.lat] } })) } });
    map.addLayer({ id: "sites", type: "circle", source: "sites", layout: { visibility: "none" },
      paint: { "circle-radius": ["interpolate", ["linear"], ["zoom"], 5, 2.5, 10, 6], "circle-color": "#ffffff", ...themePaint("sites"), "circle-stroke-width": 1 } });
    hoverPopup("sites", (p) => `<b>${L("แหล่งซากที่ขึ้นทะเบียน", "Registered fossil site")}</b><br>${esc(placeName(p.name))}`);

    map.addSource("pbdb", { type: "geojson", data: { type: "FeatureCollection",
      features: pbdb.map((p) => ({ type: "Feature", properties: { name: p.name, age: p.age, types: p.types.join(", ") }, geometry: { type: "Point", coordinates: [p.lon, p.lat] } })) } });
    map.addLayer({ id: "pbdb", type: "circle", source: "pbdb", layout: { visibility: "none" },
      paint: { "circle-radius": ["interpolate", ["linear"], ["zoom"], 5, 3, 10, 6], ...themePaint("pbdb"), "circle-stroke-color": "#000", "circle-stroke-width": 1 } });
    hoverPopup("pbdb", (p) => `<b>${L("นักวิจัยเคยรายงานซาก", "Fossils reported by researchers")}</b> (${L("ฐานข้อมูล PBDB", "PBDB database")})<br>${esc(p.name)}`
      + `${p.age ? `<br>${L("ยุค", "Age")}: ${esc(p.age)}` : ""}${p.types ? `<br>${L("ชนิด", "Types")}: ${esc(p.types.split(", ").map(tr).join(", "))}` : ""}`);
    map.addLayer({ id: "prov-label", type: "symbol", source: "prov-labels", minzoom: 5,
      layout: { "text-field": provLabelField(), "text-font": ["Noto Sans Regular"], "text-size": ["interpolate", ["linear"], ["zoom"], 5, 10, 8, 14], "text-padding": 3 },
      paint: { ...themePaint("prov-label"), "text-halo-width": 1.6 } });
    map.on("mousemove", onHover);
    map.on("mouseout", () => { $("#hovertip").hidden = true; });

    map.on("click", (e) => {
      const { lat, lng } = e.lngLat;
      if (state.screen === "identify") {                         // หน้าทายชื่อ: คลิกเพื่อเลือกจุดที่พบ
        setMarker(lat, lng);
        $('input[name="loc"][value="point"]').checked = true;
        return;
      }
      analysePoint(lat, lng);
    });

    renderLook(options.appearance);
    $("#to-type").addEventListener("click", () => setLookMode("type"));
    $("#to-look").addEventListener("click", () => setLookMode("look"));
    if (state.screen === "identify") requestAnimationFrame(fitLookIcons);
    fillFType();
    renderParts();
    $("#f_part").addEventListener("input", () => $$("#part-chips .chip").forEach((x) => x.classList.toggle("on", x.dataset.v === I18N.toThai($("#f_part").value.trim()))));
    const fill = (id, values) => { $(id).innerHTML = values.map((v) => `<option value="${esc(v)}"></option>`).join(""); };
    fill("#dl-com_name", options.com_name);
    fillProvinces();
    fill("#dl-district", options.district);
    options.sites.forEach((s) => state.siteByLabel.set(`${s.name} · #${s.id}`, s));
    fill("#dl-sites", [...state.siteByLabel.keys()]);
    updateLayer();
    state.ready = true;
    fitThailand();
    $("#loading").classList.add("hide");
    if (state.geo.on) loadGeology().then(applyGeo).catch(() => {});      // กดเปิดไว้ก่อนแผนที่พร้อม
    else setTimeout(() => loadGeology().catch(() => {}), 2500);          // ดึงล่วงหน้า: กดเปิดแล้วขึ้นทันที
  } catch (err) {
    $("#loading-text").removeAttribute("data-en");                  // ไม่ให้การเปลี่ยนภาษาเขียนทับข้อความผิดพลาด
    if (err instanceof BusyError) {
      $("#loading-text").innerHTML = `<div class="busy-slot"></div>`;
      showBusy($("#loading-text .busy-slot"), err.retryAfter, () => location.reload());
    } else {
      $("#loading-text").innerHTML = `<span class="err">${L("โหลดข้อมูลไม่สำเร็จ", "Could not load data")}: ${esc(err.message)}</span>`;
    }
    const sp = $(".spinner");
    if (sp) sp.remove();
  }
}

// ---------------------------------------------------------------- เปลี่ยนภาษา / ธีม
function relabel() {
  // วาดทุกส่วนที่สร้างจากโค้ดใหม่ตามภาษา (ข้อความคงที่ใน HTML: I18N.apply ทำไปก่อนแล้ว)
  prefButtons();
  renderPlaces();
  pointLabel();
  HERO.relabel();
  $("#hovertip").hidden = true;
  if (state.options) { renderStats(); renderMore(); }
  if (state.modalK !== null) openModal(state.modalK);
  if (!state.ready) return;
  fillTypeSelect(); fillFType(); renderParts(); fillProvinces();
  renderLook(state.options.appearance);
  state.look.fitted = false;
  if (state.look.card) pickCard(state.look.card, true);
  fitLookIcons();
  renderRanking(); renderLegend();
  if (state.geo.data) renderGeoList();
  if (map.getLayer("prov-label")) map.setLayoutProperty("prov-label", "text-field", provLabelField());
  if (state.lastPoint && !$("#result").hidden) showPoint(state.lastPoint);
  if (state.spec && !$("#spec-result").hidden && $("#holo-big")) {
    const out = $("#spec-result");
    out.innerHTML = renderSpecimen(state.spec.r, state.spec.ftype);
    mountSpecimen(out, state.spec.k || 0);
  }
}

function recolorMarkers() {
  for (const [k, opt] of [["marker", { color: PIN[THEME.theme].point }], ["siteMarker", { color: PIN[THEME.theme].site, scale: 0.8 }]]) {
    const m = state[k];
    if (!m) continue;
    const shown = !!m.getElement().parentNode, at = m.getLngLat();
    m.remove();
    state[k] = new maplibregl.Marker(opt);
    if (at) state[k].setLngLat(at);
    if (shown) state[k].addTo(map);
  }
}

function retheme() {
  // สีตามธีม: แผนที่ฐาน + ชั้นข้อมูล (setBasemap) หมุด และส่วนที่ใช้สีระดับโอกาส
  prefButtons();
  setBasemap();
  recolorMarkers();
  if (!state.ready) return;
  renderSurface();                                              // สีของพื้นผิวอยู่ในภาพ: วาดใหม่ทันที (syncTheme วาดซ้ำเมื่อแผนที่ฐานใหม่โหลดเสร็จ)
  renderLegend();
  if (state.modalK !== null) openModal(state.modalK);
  if (state.lastPoint && !$("#result").hidden) showPoint(state.lastPoint);
}

I18N.onChange(relabel);
THEME.onChange(retheme);
I18N.apply();
init();
