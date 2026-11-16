// Project Jurassic — โหมดเว็บแบบไม่มีเซิร์ฟเวอร์ (static site บน Cloudflare Pages)
//   แทน POST /api/predict_location และ /api/predict_specimen ของ webapp/server.py ด้วยการคำนวณในเบราว์เซอร์ของผู้ใช้:
//   - วิเคราะห์จุด: เปิดผลของช่องกริด 5 กม. ที่ใกล้จุดที่คลิกที่สุด (data/cells.json = predict_location ที่ศูนย์กลางทุกช่อง, scripts/web_static.py)
//     จังหวัด และระยะถึงแหล่ง ทธ. ที่ใกล้ที่สุด คำนวณจากจุดที่คลิกจริง
//   - ทายสกุล: โมเดลเดียวกับ server แปลงเป็น ONNX (scripts/web_models.py) รันด้วย onnxruntime-web + ขั้นเตรียมข้อมูลที่เขียนใหม่ให้ได้ค่าเดียวกับ
//     predict_lib.Predictor.specimen_features ทุกขั้น (tests/test_static_site.py เทียบกับ Python ครบทั้งทะเบียน)
//     ข้อความของแหล่งใหม่: embedding ที่คำนวณไว้แล้ว (ข้อความในทะเบียน + โหมดหน้าตา) ตรงกับ Python; ข้อความอื่นใช้ WangchanBERTa
//     น้ำหนัก float16 (โหลดครั้งแรกที่ต้องใช้ ~210 MB) ซึ่งให้สกุลอันดับ 1 ตรงกับ Python ราว 97%
// ใช้ได้ทั้งในเบราว์เซอร์ (window.JurassicLocal) และใน Node (module.exports — tools/web/parity.mjs)
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.JurassicLocal = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const ORT_VERSION = "1.31.1", TOKENIZERS_VERSION = "0.2.0";       // ต้องตรงกับ tools/web/package.json
  const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  const TOKENIZERS_CDN = `https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@${TOKENIZERS_VERSION}/dist/tokenizers.mjs`;
  const THAI_BOX = [97.3, 105.7, 5.6, 20.5];                          // เดียวกับ server.THAI_BOX
  const TINY = 1.0000000180025095e-35;                                // LightGBM นับ |x| ≤ ค่านี้เป็น 0
  const WORD = "[\\p{L}\\p{N}_]";                                     // \w ของ Python (Unicode)

  const cfg = {
    data: "data/", models: "models/", bert: "models/bert/",
    fetchJSON: async (url) => { const r = await fetch(url); if (!r.ok) throw new Error(`${r.status} ${url}`); return r.json(); },
    fetchBuffer: async (url) => { const r = await fetch(url); if (!r.ok) throw new Error(`${r.status} ${url}`); return r.arrayBuffer(); },
    importOrt: async () => { const ort = await import(ORT_CDN + "ort.wasm.min.mjs"); ort.env.wasm.wasmPaths = ORT_CDN; return ort; },
    importTokenizers: () => import(TOKENIZERS_CDN),
    onBertProgress: null,                                             // (โหลดแล้ว, ทั้งหมด) เป็นไบต์
  };
  const memo = {};
  function once(key, make) {
    if (!memo[key]) memo[key] = make().catch((e) => { delete memo[key]; throw e; });
    return memo[key];
  }

  // ---------------------------------------------------------------- ตัวช่วย
  function pyRound(x, d) {
    // round() ของ Python: ปัดครึ่งไปเลขคู่ เมื่อค่าอยู่กึ่งกลางพอดี
    const m = 10 ** d, y = x * m, f = Math.floor(y), diff = y - f;
    const n = diff > 0.5 ? f + 1 : diff < 0.5 ? f : (f % 2 === 0 ? f : f + 1);
    return n / m;
  }
  const isBlank = (v) => v === null || v === undefined;
  const str = (v) => (isBlank(v) ? "nan" : String(v));

  async function ort() {
    return once("ort", async () => {
      const o = await cfg.importOrt();
      o.env.wasm.numThreads = 1;                                     // หลายเธรดต้องใช้ cross-origin isolation — โมเดลเล็ก เธรดเดียวพอ
      return o;
    });
  }

  // ---------------------------------------------------------------- จุดในรูปหลายเหลี่ยม (GeoJSON)
  function inRing(x, y, ring) {
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [xi, yi] = ring[i], [xj, yj] = ring[j];
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  }
  function inGeometry(x, y, g) {
    const polys = g.type === "Polygon" ? [g.coordinates] : g.type === "MultiPolygon" ? g.coordinates : [];
    return polys.some((rings) => inRing(x, y, rings[0]) && !rings.slice(1).some((h) => inRing(x, y, h)));
  }
  function bboxOf(g) {
    let a = Infinity, b = Infinity, c = -Infinity, d = -Infinity;
    const polys = g.type === "Polygon" ? [g.coordinates] : g.coordinates;
    for (const rings of polys) for (const [x, y] of rings[0]) { a = Math.min(a, x); b = Math.min(b, y); c = Math.max(c, x); d = Math.max(d, y); }
    return [a, b, c, d];
  }
  function indexFeatures(features) {
    return features.filter((f) => f.geometry).map((f) => ({ f, bb: bboxOf(f.geometry) }));
  }
  function hit(index, lon, lat) {
    for (const { f, bb } of index) {
      if (lon < bb[0] || lon > bb[2] || lat < bb[1] || lat > bb[3]) continue;
      if (inGeometry(lon, lat, f.geometry)) return f;
    }
    return null;
  }
  const provinces = () => once("prov", async () => indexFeatures((await cfg.fetchJSON(cfg.data + "provinces.json")).polygons.features));
  const geology = () => once("geo", async () => indexFeatures((await cfg.fetchJSON(cfg.data + "geology.json")).features));
  async function provinceAt(lon, lat) {
    const f = hit(await provinces(), lon, lat);                       // ขอบเขตแบบย่อ (~1 กม.) — จุดชิดเส้นแบ่งจังหวัดอาจต่างจาก server
    return f ? f.properties.name : null;
  }
  async function symbolAt(lon, lat) {
    const f = hit(await geology(), lon, lat);                         // แผนที่ธรณีแบบย่อ (~200 ม.) ชุดเดียวกับชั้น "ยุคของหิน"
    return f ? f.properties.sym : "none";
  }

  // ---------------------------------------------------------------- วิเคราะห์จุด
  const cells = () => once("cells", () => cfg.fetchJSON(cfg.data + "cells.json"));

  function checkThai(lon, lat, msg) {
    const [lo0, lo1, la0, la1] = THAI_BOX;
    if (!(lo0 <= lon && lon <= lo1 && la0 <= lat && lat <= la1)) throw new Error(msg);
  }

  async function predictLocation(body) {
    const lat = Number(body.lat), lon = Number(body.lon);
    checkThai(lon, lat, "พิกัดอยู่นอกประเทศไทย (โมเดล train เฉพาะในไทย)");
    const C = await cells(), cos = Math.cos((lat * Math.PI) / 180);
    let i = 0, best = Infinity;
    for (let k = 0; k < C.n; k++) {
      const d = (C.lat[k] - lat) ** 2 + ((C.lon[k] - lon) * cos) ** 2;
      if (d < best) { best = d; i = k; }
    }
    const out = { lat, lon };
    for (const bg of C.backgrounds) {
      out[`potential_${bg}`] = C.potential[bg][i];
      out[`percentile_${bg}`] = C.percentile[bg][i];
    }
    for (const bg of C.backgrounds) out[`inside_${bg}`] = !!C.inside[bg][i];
    out.inside_types = !!C.inside.types[i];
    let km = Infinity;                                                // แบบเดียวกับ predict_location (กม. โดยประมาณ)
    for (const [slon, slat] of C.sites_lonlat) km = Math.min(km, Math.hypot((slat - lat) * 111.0, (slon - lon) * 111.0 * cos));
    out.nearest_site_km = pyRound(km, 1);
    out.similar_sites = C.similar[i].map((k) => C.site_names[k]);
    out.top_types = C.top_types[i].map(([k, p]) => [C.classes_B[k], p]);
    out.rock_unit = C.rock_unit_names[C.rock_unit[i]];
    out.rock_era = C.rock_era_names[C.rock_era[i]];
    out.terrane = C.terrane_names[C.terrane[i]];
    out.s2_scene = C.s2_scene[i] < 0 ? null : C.s2_scene_names[C.s2_scene[i]];
    out.warnings = out.s2_scene ? [] : [C.no_s2_warning];
    out.seconds = 0;
    out.province = await provinceAt(lon, lat);
    out.grid_cell = { lon: C.lon[i], lat: C.lat[i], km: pyRound(Math.sqrt(best) * 111, 1) };
    return out;
  }

  // ---------------------------------------------------------------- ตารางของขั้นทายสกุล (models/specimen.json + .bin)
  const spec = () => once("spec", async () => {
    const [S, buf] = await Promise.all([cfg.fetchJSON(cfg.models + "specimen.json"), cfg.fetchBuffer(cfg.models + "specimen.bin")]);
    const arr = (ref) => {
      const n = ref.shape.reduce((a, b) => a * b, 1);
      return ref.dtype === "f8" ? new Float64Array(buf, ref.offset, n) : new Float32Array(buf, ref.offset, n);
    };
    const ws = S.whitespace.replace(/[\\\]\-^]/g, "\\$&");
    const W = `[${ws}]`;
    const esc = (s) => s.replace(/[.*+?^${}()|[\]\\/-]/g, "\\$&");
    const keep = "(?!(?:teeth|tooth|bone|bones|shell|skull|jaw|vertebra|vertebrae|neural|spine|fragment|fragments|and|or|with)(?!" + WORD + "))";
    S.re = {
      ws2: new RegExp(`${W}${W}+`, "gu"), wsRun: new RegExp(`${W}+`, "gu"), wsSplit: new RegExp(`${W}+`, "u"),
      wsEdge: new RegExp(`^${W}+|${W}+$`, "gu"), notFossil: new RegExp(S.not_fossil, "gu"),
      // feature_lib.genus_name_pattern: ชื่อสกุลในทะเบียน (+ sp. / cf. / ชื่อชนิด) — \b ของ Python แทนด้วยการมอง \w ฝั่งซ้าย / ขวา
      genus: new RegExp(`(?<!${WORD})(?:${S.genus_names.map(esc).join("|")})(?!${WORD})` +
        `(?:${W}+(?:cf\\.|aff\\.)?${W}*(?:sp\\.|spp\\.|${keep}[a-z][a-z\\-]{2,}))?`, "gu"),
    };
    S.tf = { vocab: new Map(S.tfidf.vocab.map((g, i) => [g, i])), idf: arr(S.tfidf.idf), svd: arr(S.tfidf.svd), k: S.tfidf.svd.shape[0] };
    S.bt = { mean: arr(S.bert.pca_mean), comp: arr(S.bert.pca_components), table: arr(S.bert.table), k: S.bert.pca_components.shape[0],
             dim: S.bert.pca_components.shape[1], index: new Map(S.bert.texts.map((t, i) => [t, i])) };
    S.vocabSet = new Set(S.com_vocab);
    S.groupMap = Object.fromEntries(Object.entries(S.group_map).map(([g, c]) => [g, new Set(c)]));
    return S;
  });

  function lowerPy(s) { return s.toLowerCase(); }

  function cleanPart(S, part) {
    // specimen_text: ลบชื่อสกุลที่เขียนไว้ใน f_part แล้วรวมช่องว่าง
    return part.replace(S.re.genus, " ").replace(S.re.wsRun, " ").replace(S.re.wsEdge, "");
  }
  function specimenText(S, row) {
    return `${row.com_name_generic} | ${row.f_type} | ${cleanPart(S, row.f_part)}`;
  }

  function tfidf(S, text) {
    // TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf, l2) + TruncatedSVD — ลำดับการบวกเดียวกับ scipy / sklearn
    const [lo, hi] = S.tfidf.ngram, counts = new Map();
    const doc = lowerPy(text).replace(S.re.ws2, " ");
    for (const w of doc.split(S.re.wsSplit)) {
      if (!w) continue;
      const ch = Array.from(` ${w} `), L = ch.length;
      for (let n = lo; n <= hi; n++) {
        let off = 0;
        const add = () => { const j = S.tf.vocab.get(ch.slice(off, off + n).join("")); if (j !== undefined) counts.set(j, (counts.get(j) || 0) + 1); };
        add();
        while (off + n < L) { off += 1; add(); }
        if (off === 0) break;
      }
    }
    const idx = [...counts.keys()].sort((a, b) => a - b);
    const val = idx.map((j) => (Math.log(counts.get(j)) + 1) * S.tf.idf[j]);
    let ss = 0;
    for (const v of val) ss += v * v;
    const norm = ss > 0 ? Math.sqrt(ss) : 1;
    const V = S.tfidf.vocab.length, z = new Array(S.tf.k).fill(0);
    for (let q = 0; q < idx.length; q++) {
      const v = val[q] / norm;
      for (let k = 0; k < S.tf.k; k++) z[k] += v * S.tf.svd[k * V + idx[q]];
    }
    return z;
  }

  // ---------------------------------------------------------------- WangchanBERTa (เฉพาะข้อความที่ไม่มีในตาราง)
  const bert = () => once("bert", async () => {
    const [o, tk] = await Promise.all([ort(), cfg.importTokenizers()]);
    const [man, tj, tc] = await Promise.all([cfg.fetchJSON(cfg.bert + "manifest.json"), cfg.fetchJSON(cfg.bert + "tokenizer.json"),
      cfg.fetchJSON(cfg.bert + "tokenizer_config.json")]);
    const model = new Uint8Array(man.bytes);
    let done = 0;
    for (const p of man.parts) {
      const b = new Uint8Array(await cfg.fetchBuffer(cfg.bert + p));
      model.set(b, done);
      done += b.length;
      if (cfg.onBertProgress) cfg.onBertProgress(done, man.bytes);
    }
    const session = await o.InferenceSession.create(model, { executionProviders: ["wasm"] });
    return { o, session, tok: new tk.Tokenizer(tj, tc), maxLen: man.max_length };
  });

  async function bertIds(text) {
    // tokenizer(text, truncation=True, max_length=64): <s> + โทเค็นไม่เกิน 62 + </s>
    const B = await bert();
    let ids = B.tok.encode(text).ids;
    if (ids.length > B.maxLen) ids = ids.slice(0, B.maxLen - 1).concat(ids[ids.length - 1]);
    return ids;
  }

  async function bertPca(S, text) {
    const k = S.bt.index.get(text);
    if (k !== undefined) return { z: Array.from(S.bt.table.subarray(k * S.bt.k, (k + 1) * S.bt.k)), source: "table" };
    const B = await bert(), ids = await bertIds(text);
    const big = B.o.Tensor;
    const feeds = { input_ids: new big("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]),
                    attention_mask: new big("int64", new BigInt64Array(ids.length).fill(1n), [1, ids.length]) };
    const x = (await B.session.run(feeds)).pooled.data;
    const { mean, comp, k: K, dim } = S.bt, z = new Array(K).fill(0);
    for (let c = 0; c < K; c++) {                                      // PCA.transform = X·Cᵀ − mean·Cᵀ (ลำดับเดียวกับ sklearn)
      let a = 0, m = 0;
      for (let j = 0; j < dim; j++) { a += x[j] * comp[c * dim + j]; m += mean[j] * comp[c * dim + j]; }
      z[c] = a - m;
    }
    return { z, source: "model" };
  }

  // ---------------------------------------------------------------- โมเดลทายสกุล
  const genusSession = (route) => once(`genus_${route}`, async () => {
    const [o, S] = await Promise.all([ort(), spec()]);
    const buf = await cfg.fetchBuffer(cfg.models + S.models[route].file);
    return o.InferenceSession.create(new Uint8Array(buf), { executionProviders: ["wasm"] });
  });

  function classesOf(S, text) {
    if (typeof text !== "string") return new Set();
    const t = text.replace(S.re.notFossil, " ");
    return new Set(Object.keys(S.fossil_classes).filter((c) => S.fossil_classes[c].some((k) => t.includes(k))));
  }
  function specimenClasses(S, fType, comName) {
    const a = classesOf(S, fType);
    return a.size ? a : classesOf(S, comName);
  }
  function otherGroup(S, fType) {
    if (isBlank(fType) || fType === "") return null;
    const c = [...specimenClasses(S, fType, fType)].sort();
    return c.length ? c.map((x) => S.group_names[x] || x).join(" / ") : (fType === "อื่น ๆ" ? "ประเภทอื่น ๆ" : String(fType));
  }

  function encodeRow(S, route, row) {
    // TabularPreprocessor(mode="tree").transform -> float32: ตัวเลขตามเดิม (ค่าว่าง = NaN), หมวดหมู่ = ลำดับในระดับที่ fit ไว้ (ไม่รู้จัก = -1)
    const m = S.models[route], x = new Float32Array(m.num_cols.length + m.cat_cols.length);
    m.num_cols.forEach((c, j) => { const v = row[c]; x[j] = isBlank(v) ? NaN : Number(v); });
    m.cat_cols.forEach((c, j) => { const k = m.levels[c].indexOf(str(row[c])); x[m.num_cols.length + j] = k; });
    if (m.zero_tiny) for (let j = 0; j < x.length; j++) if (Math.abs(x[j]) <= TINY) x[j] = 0;
    return x;
  }

  async function specimenFeatures(S, record) {
    // = Predictor.specimen_features
    const rec = { ...record }, sid = rec.site_OBJECTID;
    const known = !isBlank(sid) && Object.prototype.hasOwnProperty.call(S.known_sites, String(Math.trunc(sid)));
    const row = {};
    if (known) S.site_cols.forEach((c, j) => { row[c] = S.known_sites[String(Math.trunc(sid))][j]; });
    if (!rec.province && !isBlank(rec.lat)) rec.province = await provinceAt(rec.lon, rec.lat);
    for (const c of ["com_name", "f_type", "f_group", "f_part", "province", "district"]) row[c] = str(rec[c]);
    row.com_name_generic = S.vocabSet.has(row.com_name) ? row.com_name : row.f_type;
    const text = specimenText(S, row);
    let textSource = "tfidf";
    if (known) {
      tfidf(S, text).forEach((v, k) => { row[`tfidf_${k}`] = v; });
    } else {
      let fm, gg;
      if (rec.formation) { fm = String(rec.formation); gg = str(rec.geo_group || "Unknown"); }
      else if (!isBlank(sid)) [fm, gg] = S.site_formation[String(Math.trunc(sid))] || ["Unknown", "Unknown"];
      else if (!isBlank(rec.lat)) { fm = S.symbol_formation[await symbolAt(rec.lon, rec.lat)] || "Unknown"; gg = "Unknown"; }
      else { fm = "Unknown"; gg = "Unknown"; }
      row.fm_FORMATION = fm;
      row.fm_GEO_GROUP = gg;
      const fv = S.formations[fm] || [0];                             // ชื่อที่ไม่อยู่ในตาราง = จับคู่ ThaiLex ไม่ได้
      S.formation_cols.forEach((c, j) => { row[c] = j < fv.length ? fv[j] : null; });
      const b = await bertPca(S, text);
      textSource = b.source;
      b.z.forEach((v, k) => { row[`wangchanberta_${k}`] = v; });
    }
    return { row, known, text, textSource, rec };
  }

  async function genusProbs(S, record) {
    // = Predictor._genus_probs: โมเดล known / new + ตัวกรองกลุ่มซาก (consistency_filter)
    const f = await specimenFeatures(S, record);
    const route = f.known ? "known" : "new";
    const session = await genusSession(route), o = await ort(), x = encodeRow(S, route, f.row);
    const out = await session.run({ X: new o.Tensor("float32", x, [1, x.length]) });
    const P = Array.from(out.P.data, Number);
    const groups = specimenClasses(S, f.row.f_type, f.row.com_name_generic);
    if (groups.size) {
      const Q = P.map((p, j) => {
        const g = S.classes[j], gc = S.groupMap[g];
        return g !== S.other && gc && gc.size && ![...gc].some((c) => groups.has(c)) ? 0 : p;
      });
      const s = Q.reduce((a, b) => a + b, 0);
      if (s > 0) for (let j = 0; j < P.length; j++) P[j] = Q[j] / s;
    }
    return { P, row: f.row, known: f.known, groups, text: f.text, textSource: f.textSource };
  }

  function speciesGuess(S, genus, sid) {
    if (genus === S.other) return null;
    if (!isBlank(sid)) {
      const k = `${Math.trunc(sid)}|${genus}`;
      if (Object.prototype.hasOwnProperty.call(S.species_site, k)) return S.species_site[k];
    }
    return Object.prototype.hasOwnProperty.call(S.species_nat, genus) ? S.species_nat[genus] : null;
  }

  function topK(P, k) {
    // เรียงความน่าจะเป็นจากมากไปน้อย (เท่ากัน -> ลำดับคลาส)
    return P.map((p, j) => [p, j]).sort((a, b) => b[0] - a[0] || a[1] - b[1]).slice(0, k).map((e) => e[1]);
  }

  async function predictRegular(S, record) {
    const r = await genusProbs(S, record), sid = record.site_OBJECTID, oth = otherGroup(S, record.f_type);
    const top = topK(r.P, 3).map((j) => [S.classes[j], pyRound(r.P[j], 3), S.classes[j] === S.other ? oth : null]);
    return { route: r.known ? "แหล่งเดิม" : "แหล่งใหม่", formation: r.known ? null : r.row.fm_FORMATION, groups: [...r.groups].sort(),
             top, species: top.map(([g]) => speciesGuess(S, g, sid)), _P: r.P, _text: r.text, _text_source: r.textSource };
  }

  function leafId(S, card, option) {
    const opts = S.appearance.cards[card];
    if (!opts) throw new Error("ไม่รู้จักคำตอบเรื่องหน้าตาซาก");
    if (!opts.length) return card;
    const o = option || "unsure";
    if (!opts.includes(o)) throw new Error("ไม่รู้จักคำตอบเรื่องหน้าตาซาก");
    return `${card}/${o}`;
  }

  async function predictAppearance(S, record, card, option) {
    // = Predictor.predict_specimen_appearance: รวมผลทายของประเภทที่เป็นไปได้ตามน้ำหนัก; "อื่น ๆ" แยกตามกลุ่มซาก
    const leaf = leafId(S, card, option), w = S.appearance.leaves[leaf], oi = S.classes.indexOf(S.other);
    let P = new Array(S.classes.length).fill(0), groups = new Set(), last = null, sources = new Set();
    const others = new Map();
    for (const [t, wt, part] of w) {
      const rec = { ...record, f_type: t, f_group: Object.prototype.hasOwnProperty.call(S.type_group, t) ? S.type_group[t] : null,
                    f_part: record.f_part || part };
      last = await genusProbs(S, rec);
      P = P.map((p, j) => p + wt * last.P[j]);
      last.groups.forEach((g) => groups.add(g));
      sources.add(last.textSource);
      const grp = otherGroup(S, t);
      others.set(grp, (others.get(grp) || 0) + wt * last.P[oi]);
    }
    const entries = S.classes.map((g, j) => [g, P[j], null]).filter((_, j) => j !== oi);
    for (const [g, v] of others) entries.push([S.other, v, g]);
    entries.sort((a, b) => b[1] - a[1]);
    const top = entries.slice(0, 3).map(([g, p, grp]) => [g, pyRound(p, 3), grp]);
    return { route: last.known ? "แหล่งเดิม" : "แหล่งใหม่", formation: last.known ? null : last.row.fm_FORMATION, groups: [...groups].sort(),
             top, species: top.map(([g]) => speciesGuess(S, g, record.site_OBJECTID)), mode: "appearance", leaf,
             types: w.map(([t, wt]) => [t, pyRound(wt, 3)]), _P: P, _text_source: [...sources].sort().join(",") };
  }

  async function predictSpecimen(body) {
    // = POST /api/predict_specimen (ตรวจข้อมูลแบบเดียวกัน)
    const S = await spec();
    const rec = {};
    for (const [k, v] of Object.entries(body)) { const t = typeof v === "string" ? v.trim() : v; rec[k] = t === "" ? null : t; }
    const card = rec.look_card, option = rec.look_option;
    delete rec.look_card;
    delete rec.look_option;
    for (const k of ["com_name", "f_type", "f_group", "f_part", "province", "district", "site_OBJECTID", "lat", "lon"]) if (!(k in rec)) rec[k] = null;
    if (card && rec.f_type) throw new Error("เลือกอย่างใดอย่างหนึ่ง: หน้าตาซาก หรือประเภทซาก");
    if (card) leafId(S, card, option);
    else if (!rec.f_type && !rec.com_name) throw new Error("ต้องบอกหน้าตาซาก ประเภทซาก หรือชื่อสามัญ อย่างน้อยหนึ่งอย่าง");
    if (!isBlank(rec.lat)) {
      if (isBlank(rec.lon)) throw new Error("พิกัดอยู่นอกประเทศไทย");
      checkThai(rec.lon, rec.lat, "พิกัดอยู่นอกประเทศไทย");
    }
    return card ? predictAppearance(S, rec, card, option) : predictRegular(S, rec);
  }

  // ต้องโหลดโมเดลภาษาหรือไม่ (ให้หน้าเว็บบอกผู้ใช้ก่อนโหลด ~210 MB) — true เมื่อทุกข้อความที่จะใช้มีในตารางแล้ว
  async function textCached(body) {
    const S = await spec();
    const sid = body.site_OBJECTID;
    if (!isBlank(sid) && Object.prototype.hasOwnProperty.call(S.known_sites, String(Math.trunc(sid)))) return true;
    const rows = body.look_card
      ? S.appearance.leaves[leafId(S, body.look_card, body.look_option)].map(([t, , part]) => ({ ...body, f_type: t, f_part: body.f_part || part }))
      : [body];
    return rows.every((r) => {
      const row = { com_name: str(r.com_name || null), f_type: str(r.f_type || null), f_part: str(r.f_part || null) };
      row.com_name_generic = S.vocabSet.has(row.com_name) ? row.com_name : row.f_type;
      return S.bt.index.has(specimenText(S, row));
    });
  }

  async function api(path, body) {
    // แทน api() ของ app.js: GET /api/<ชื่อ> -> data/<ชื่อ>.json, POST -> คำนวณในเบราว์เซอร์
    const name = path.replace(/^\/api\//, "");
    if (!body) return cfg.fetchJSON(`${cfg.data}${name}.json`);
    if (name === "predict_location") return predictLocation(body);
    if (name === "predict_specimen") return predictSpecimen(body);
    throw new Error(`ไม่มี ${path} ในเว็บแบบ static`);
  }

  return { cfg, api, predictLocation, predictSpecimen, textCached, bertIds, tfidf: async (t) => tfidf(await spec(), t),
           cleanPart: async (t) => cleanPart(await spec(), t), provinceAt, symbolAt, versions: { ORT_VERSION, TOKENIZERS_VERSION } };
});
