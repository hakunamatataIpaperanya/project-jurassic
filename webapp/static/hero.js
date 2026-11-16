// Project Jurassic — หน้าแรก: ภาพ 3 มิติแบบ "สแกน" (canvas 2D ไม่ใช้ไลบรารีเพิ่ม)
//   ซากตัวอย่างข้ามยุคจากทะเบียน (/api/showcase) วาดเป็นภาพฉายโฮโลแกรมจากจุด 3 มิติของ holo.js (แสงเงาตามทิศผิว + ขอบเรือง + วงสแกน)
//   ลอยเหนือก้อนชั้นหิน: 1 ชั้น = 1 ยุค (ล่างเก่า บนใหม่) ความหนาแน่นของเส้นในชั้น = จำนวนแหล่ง ทธ. ของยุคนั้น
//   ผิวบนของก้อน = แผนที่โอกาสพบซาก 5 กม. ของไทย (/api/grid) + แหล่ง ทธ. + หมุดที่แหล่งที่พบตัวอย่าง
//   เปลี่ยนตัวอย่าง: เส้นแสงพุ่งออกจากจุดกลาง แล้วตัวใหม่พุ่งเข้ามาประกอบ (ยุคที่พบเรืองแสงในก้อนชั้นหิน)
const HERO = (() => {
  const BW = 190, BD = 345, BH = 185;               // ก้อนชั้นหิน (หน่วยฉาก): กว้าง = ลองจิจูด, ลึก = ละติจูด (สัดส่วนประเทศไทย), สูง
  const GAP = 56, SPEC_W = 330, SPEC_H = 190;               // ซากลอยเหนือผิวบน (เดิม 44 / 230 × 150: ขยายและยกสูงขึ้นให้รูปร่างชัด ไม่กลืนกับแผนที่)
  const LON = [97.3, 105.7], LAT = [5.6, 20.5];
  const DIST = 900;
  const T_HOLD = 7200, T_OUT = 850, T_IN = 1150;
  const CY = "24,209,255", HI = "175,242,255", AMB = "238,183,99";
  let S = null;

  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const ease = (u) => u * u * (3 - 2 * u);
  const fmt = (v, p, n) => `${Math.abs(v).toFixed(3)}°${v >= 0 ? p : n}`;
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

  // ---------------------------------------------------------------- เตรียมข้อมูลฉาก
  function prep(m) {
    // จุดของ holo.js (x ขวา, y ลง, z ลึก) -> ฉาก (y ขึ้น; normal พลิกแกน y ตาม) ย่อ / ขยายให้พอดีกรอบ
    const s = Math.min(SPEC_W / m.w, SPEC_H / m.h, 170 / Math.max(m.d, 1)), n = m.n, f = () => new Float32Array(n);
    const X = f(), Y = f(), Z = f(), A = f(), R = f(), NX = f(), NY = f(), NZ = f(), K = new Uint8Array(n);
    for (let j = 0; j < n; j++) {
      const o = j * 7;
      X[j] = m.P[o] * s;
      Y[j] = GAP + (m.bottom - m.P[o + 1]) * s;
      Z[j] = m.P[o + 2] * s;
      A[j] = m.P[o + 3];
      K[j] = m.P[o + 4];
      R[j] = m.P[o + 6];
      NX[j] = m.N[j * 3]; NY[j] = -m.N[j * 3 + 1]; NZ[j] = m.N[j * 3 + 2];
    }
    return { n, X, Y, Z, A, R, K, NX, NY, NZ, top: GAP + m.h * s };
  }

  function sprite(rgb) {
    // จุดแสงนุ่ม (แกนสว่าง + ขอบจาง) สำหรับวาดแบบบวกแสง
    const c = document.createElement("canvas");
    c.width = c.height = 32;
    const g = c.getContext("2d"), gr = g.createRadialGradient(16, 16, 0, 16, 16, 16);
    gr.addColorStop(0, `rgba(${rgb},1)`);
    gr.addColorStop(0.18, `rgba(${rgb},0.85)`);
    gr.addColorStop(0.45, `rgba(${rgb},0.25)`);
    gr.addColorStop(1, `rgba(${rgb},0)`);
    g.fillStyle = gr;
    g.fillRect(0, 0, 32, 32);
    return c;
  }

  function strata(periods) {
    // อนุภาคบนผิวหน้า / ผิวข้างของแต่ละชั้น: จำนวน ∝ จำนวนแหล่งของยุค (อย่างน้อยพอให้เห็นชั้น)
    const L = periods.length, lh = BH / L, out = [];
    periods.forEach((p, i) => {
      const y0 = -BH + i * lh, dens = 0.008 + 0.06 * Math.sqrt(p.sites / 159);      // 159 = แหล่งของยุคที่มากสุด (เพอร์เมียน) โดยประมาณ
      for (const [face, len] of [["front", BW], ["left", BD], ["right", BD]]) {
        const cnt = Math.round(len * lh * dens);
        for (let c = 0; c < cnt; c++) {
          const u = Math.random() - 0.5, v = y0 + 1 + Math.random() * (lh - 2);
          const yq = Math.round(v / 2.2) * 2.2;
          if (face === "front") out.push(i, 0, u * BW, yq, BD / 2, Math.random());
          else out.push(i, face === "left" ? 1 : 2, face === "left" ? -BW / 2 : BW / 2, yq, u * BD, Math.random());
        }
      }
    });
    return { L, lh, P: Float32Array.from(out) };
  }

  const mapXZ = (lon, lat) => [((lon - (LON[0] + LON[1]) / 2) / (LON[1] - LON[0])) * BW * 0.94,
    -((lat - (LAT[0] + LAT[1]) / 2) / (LAT[1] - LAT[0])) * BD * 0.95];

  // ---------------------------------------------------------------- กล้อง
  function camera(t) {
    const reduce = S.reduce;
    const a = (reduce ? 0.22 : 0.28 * Math.sin(t * 0.00011) + 0.1) + S.px * 0.3;
    const p = 0.6 + (reduce ? 0 : 0.05 * Math.sin(t * 0.00007)) - S.py * 0.1;
    return { ca: Math.cos(a), sa: Math.sin(a), cp: Math.cos(p), sp: Math.sin(p), a };
  }

  function specCam(c) {
    // ภาพฉายของซาก: ก้มน้อยกว่ากล้องของก้อนชั้นหิน (เห็นรูปร่างด้านข้าง) โดยฐานยังอยู่ตรงแท่นฉายเดิม
    const p = 0.12 - S.py * 0.08, c2 = { ...c, cp: Math.cos(p), sp: Math.sin(p) };
    const a = proj(c, 0, GAP, 0, [0, 0, 0]), b = proj(c2, 0, GAP, 0, [0, 0, 0]);
    c2.dx = a[0] - b[0];
    c2.dy = a[1] - b[1];
    return c2;
  }

  function projS(c2, x, y, z, out) {
    proj(c2, x, y, z, out);
    out[0] += c2.dx;
    out[1] += c2.dy;
    return out;
  }

  function proj(c, x, y, z, out) {
    const xr = x * c.ca + z * c.sa, zr = -x * c.sa + z * c.ca, y1 = y - S.ty;
    const yr = y1 * c.cp - zr * c.sp, zz = y1 * c.sp + zr * c.cp;
    const d = DIST - zz, k = S.F / d;
    out[0] = S.cx + xr * k;
    out[1] = S.cy - yr * k;
    out[2] = k;
    return out;
  }

  function fit() {
    // ปรับขนาดภาพให้พอดีกรอบทุกมุมกล้องที่เป็นไปได้
    const cv = S.cv, dpr = S.dpr;
    S.W = cv.width = Math.max(1, Math.round(cv.clientWidth * dpr));
    S.H = cv.height = Math.max(1, Math.round(cv.clientHeight * dpr));
    for (const [k, ck, div] of [["sc", "sctx", 1], ["tc", "tctx", 1], ["gl", "gctx", 4]]) {
      S[k] = S[k] || document.createElement("canvas");
      S[k].width = Math.max(1, Math.round(S.W / div));
      S[k].height = Math.max(1, Math.round(S.H / div));
      S[ck] = S[k].getContext("2d");
    }
    const pc = document.createElement("canvas"), ph = Math.max(3, Math.round(3 * dpr));
    pc.width = 1; pc.height = ph;
    const pg = pc.getContext("2d");
    pg.fillStyle = "rgba(0,0,0,0.2)";                         // เส้นสแกนของภาพฉาย: บางลง ไม่ให้รูปร่างขาดเป็นริ้ว
    pg.fillRect(0, ph - Math.max(1, Math.round(dpr)), 1, Math.max(1, Math.round(dpr)));
    S.scanPat = S.ctx.createPattern(pc, "repeat");
    S.ty = (GAP + SPEC_H - BH) / 2 - 10;
    S.F = 1; S.cx = 0; S.cy = 0;
    const pts = [];
    for (const x of [-BW / 2, BW / 2]) for (const z of [-BD / 2, BD / 2]) for (const y of [-BH, 0]) pts.push([x, y, z]);
    const spts = [];
    for (const x of [-SPEC_W / 2, SPEC_W / 2]) for (const z of [-60, 60]) spts.push([x, GAP + SPEC_H, z]);
    let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
    const o = [0, 0, 0], save = [S.px, S.py];
    S.px = 0; S.py = 0;
    for (const t of [0, 7000, 14000, 21000, 28000, 35000, 42000, 49000]) {
      const c = camera(t), c2 = specCam(c);
      for (const [p, f] of [...pts.map((q) => [q, proj]), ...spts.map((q) => [q, (cc, x, y, z, out) => projS(c2, x, y, z, out)])]) {
        f(c, p[0], p[1], p[2], o);
        x0 = Math.min(x0, o[0]); x1 = Math.max(x1, o[0]); y0 = Math.min(y0, o[1]); y1 = Math.max(y1, o[1]);
      }
    }
    [S.px, S.py] = save;
    S.small = S.W / dpr < 480;
    const labW = (S.small ? 70 : 150) * dpr;                       // ที่ว่างด้านขวาสำหรับชื่อยุค
    const F = Math.min((S.W - labW) * 0.9 / (x1 - x0), S.H * 0.84 / (y1 - y0));
    S.F = F;
    S.cx = (S.W - labW) / 2 - ((x0 + x1) / 2) * F;
    S.cy = S.H * 0.53 - ((y0 + y1) / 2) * F;
  }

  // ---------------------------------------------------------------- วาด
  function line(ctx, c, a, b, col, w) {
    const p = proj(c, a[0], a[1], a[2], [0, 0, 0]), q = proj(c, b[0], b[1], b[2], [0, 0, 0]);
    ctx.strokeStyle = col;
    ctx.lineWidth = w;
    ctx.beginPath();
    ctx.moveTo(p[0], p[1]);
    ctx.lineTo(q[0], q[1]);
    ctx.stroke();
  }

  function drawBlock(ctx, c, now, hiIdx) {
    const { L, lh, P } = S.strata, dpr = S.dpr, o = [0, 0, 0];
    const showL = Math.max(0, c.sa) * 2.4, showR = Math.max(0, -c.sa) * 2.4;      // ผิวข้างที่หันหากล้อง
    const pulse = 0.75 + 0.25 * Math.sin(now * 0.004);
    // อนุภาคในชั้น: เส้นแนวนอนสั้น ๆ (ชั้นของยุคที่พบ = สีอำพัน)
    const buckets = [[], [], []];
    for (let j = 0; j < P.length; j += 6) {
      const i = P[j], face = P[j + 1], vis = face === 0 ? 1 : face === 1 ? clamp(showL, 0, 1) : clamp(showR, 0, 1);
      if (vis < 0.05) continue;
      proj(c, P[j + 2], P[j + 3], P[j + 4], o);
      const len = (1.5 + P[j + 5] * 6) * o[2];
      buckets[i === hiIdx ? 2 : (P[j + 5] > 0.6 ? 1 : 0)].push(o[0] - len / 2, o[1], len, vis);
    }
    const styles = [`rgba(${CY},0.30)`, `rgba(${HI},0.42)`, `rgba(${AMB},${(0.75 * pulse).toFixed(3)})`];
    const th = Math.max(1, 1.1 * dpr);
    buckets.forEach((b, k) => {
      ctx.fillStyle = styles[k];
      for (let j = 0; j < b.length; j += 4) {
        ctx.globalAlpha = b[j + 3] < 1 ? b[j + 3] : 1;
        ctx.fillRect(b[j], b[j + 1], b[j + 2], th);
      }
    });
    ctx.globalAlpha = 1;
    // เส้นแบ่งชั้นบนผิวหน้า + ผิวข้าง และขอบก้อน
    const fx = BW / 2, fz = BD / 2;
    for (let i = 0; i <= L; i++) {
      const y = -BH + i * lh, hi = i === hiIdx || i === hiIdx + 1;
      const col = hi ? `rgba(${AMB},0.85)` : `rgba(${CY},0.28)`;
      line(ctx, c, [-fx, y, fz], [fx, y, fz], col, hi ? 1.4 * dpr : dpr);
      if (showL > 0.05) line(ctx, c, [-fx, y, fz], [-fx, y, -fz], `rgba(${hi ? AMB : CY},${(clamp(showL, 0, 1) * (hi ? 0.8 : 0.26)).toFixed(3)})`, dpr);
      if (showR > 0.05) line(ctx, c, [fx, y, fz], [fx, y, -fz], `rgba(${hi ? AMB : CY},${(clamp(showR, 0, 1) * (hi ? 0.8 : 0.26)).toFixed(3)})`, dpr);
    }
    const edge = `rgba(${CY},0.55)`;
    for (const [x, z] of [[-fx, fz], [fx, fz], [-fx, -fz], [fx, -fz]]) line(ctx, c, [x, -BH, z], [x, 0, z], edge, dpr);
    line(ctx, c, [-fx, 0, -fz], [fx, 0, -fz], edge, dpr);
    line(ctx, c, [-fx, 0, fz], [-fx, 0, -fz], edge, dpr);
    line(ctx, c, [fx, 0, fz], [fx, 0, -fz], edge, dpr);
    // ป้ายยุคด้านขวา: ชื่อยุค + จำนวนแหล่ง (แถบยาวตามจำนวน) — ยุคที่พบเน้นสีอำพัน + ช่วงอายุ
    const small = S.small;
    const rowPx = lh * proj(c, fx, -BH / 2, fz, o)[2];
    for (let i = 0; i < L; i++) {
      const y = -BH + (i + 0.5) * lh, hi = i === hiIdx;
      if (small && !hi) continue;
      if (!hi && rowPx < 9 * dpr) continue;
      let ax = -1e9, ay = 0;
      for (const [x, z] of [[fx, fz], [-fx, fz], [fx, -fz], [-fx, -fz]]) {
        proj(c, x, y, z, o);
        if (o[0] > ax) { ax = o[0]; ay = o[1]; }
      }
      const p = S.data.periods[i], x = ax + 12 * dpr;
      ctx.strokeStyle = hi ? `rgba(${AMB},0.9)` : `rgba(${CY},0.35)`;
      ctx.lineWidth = dpr;
      ctx.beginPath(); ctx.moveTo(ax + 3 * dpr, ay); ctx.lineTo(x - 3 * dpr, ay); ctx.stroke();
      ctx.font = `${hi ? 600 : 400} ${(hi ? 12.5 : 10.5) * dpr}px "IBM Plex Sans Thai", sans-serif`;
      ctx.textBaseline = "middle";
      ctx.fillStyle = hi ? `rgb(${AMB})` : "rgba(200,214,220,0.62)";
      const name = I18N.period(p.period), tw = ctx.measureText(name).width;
      if (small) { ctx.fillText(name, Math.min(x, S.W - tw - 4 * dpr), ay); continue; }   // จอเล็ก: ชื่อยุคอย่างเดียว (จำนวนแหล่งอยู่ในป้ายข้อมูล)
      ctx.fillText(name, x, ay);
      const bar = Math.max(2, Math.sqrt(p.sites) * 4.4) * dpr;
      ctx.fillStyle = hi ? `rgba(${AMB},0.9)` : `rgba(${CY},0.45)`;
      ctx.fillRect(x + tw + 8 * dpr, ay - 1.5 * dpr, bar, 3 * dpr);
      if (hi) {                                                    // ยุคที่พบ: บอกจำนวนแหล่งเป็นตัวเลขด้วย (บรรทัดเดียว ไม่ทับป้ายข้างเคียง)
        ctx.font = `600 ${11 * dpr}px "IBM Plex Sans Thai", sans-serif`;
        ctx.fillText(I18N.L(`${p.sites} แหล่ง`, `${p.sites} sites`), x + tw + bar + 14 * dpr, ay);
      }
    }
  }

  function drawMap(ctx, c, item, now) {
    const o = [0, 0, 0], dpr = S.dpr;
    if (S.map) {
      const { X, Z, V } = S.map, sz = Math.max(1, 1.25 * dpr);
      const st = ["0.07", "0.15", "0.28", "0.5"].map((a) => `rgba(${CY},${a})`);   // หรี่ลงให้ซากที่ลอยด้านบนเด่นกว่า (เดิม 0.10–0.75)
      for (let b = 0; b < 4; b++) {
        ctx.fillStyle = st[b];
        for (let j = 0; j < X.length; j++) {
          if (V[j] !== b) continue;
          proj(c, X[j], 0, Z[j], o);
          ctx.fillRect(o[0], o[1], sz, sz);
        }
      }
      if (S.sites) {
        ctx.fillStyle = `rgba(${HI},0.55)`;
        for (let j = 0; j < S.sites.length; j += 2) {
          proj(c, S.sites[j], 0.5, S.sites[j + 1], o);
          ctx.fillRect(o[0] - 0.75 * dpr, o[1] - 0.75 * dpr, 1.6 * dpr, 1.6 * dpr);
        }
      }
    }
    if (!item) return;
    // หมุดที่แหล่งที่พบ: วงกระเพื่อม + แกนเจาะลงไปถึงชั้นของยุค + ลำแสงขึ้นไปหาซาก
    const [px, pz] = S.pin;
    const r = 6 + 10 * ((now * 0.0006) % 1), ring = [];
    for (let k = 0; k <= 28; k++) {
      const t = (k / 28) * Math.PI * 2;
      ring.push(proj(c, px + Math.cos(t) * r, 0.5, pz + Math.sin(t) * r, [0, 0, 0]));
    }
    ctx.strokeStyle = `rgba(${AMB},${(0.9 - 0.8 * ((r - 6) / 10)).toFixed(3)})`;
    ctx.lineWidth = 1.2 * dpr;
    ctx.beginPath();
    ring.forEach((p, k) => (k ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
    ctx.stroke();
    const yl = S.hiY;
    line(ctx, c, [px, 0, pz], [px, yl, pz], `rgba(${AMB},0.9)`, 1.6 * dpr);
    proj(c, px, yl, pz, o);
    ctx.fillStyle = `rgb(${AMB})`;
    ctx.fillRect(o[0] - 2.5 * dpr, o[1] - 2.5 * dpr, 5 * dpr, 5 * dpr);
    const a = proj(c, px, 0, pz, [0, 0, 0]), b = proj(c, 0, GAP - 6, 0, [0, 0, 0]);
    const g = ctx.createLinearGradient(a[0], a[1], b[0], b[1]);
    g.addColorStop(0, `rgba(${AMB},0.85)`);
    g.addColorStop(1, `rgba(${CY},0.05)`);
    ctx.strokeStyle = g;
    ctx.lineWidth = 1.2 * dpr;
    ctx.setLineDash([4 * dpr, 4 * dpr]);
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    ctx.setLineDash([]);
    // แท่นฉาย: วงรีใต้ซาก
    ctx.strokeStyle = `rgba(${CY},0.4)`;
    ctx.beginPath();
    for (let k = 0; k <= 40; k++) {
      const t = (k / 40) * Math.PI * 2;
      proj(c, Math.cos(t) * SPEC_W * 0.42, GAP - 8, Math.sin(t) * 46, o);
      k ? ctx.lineTo(o[0], o[1]) : ctx.moveTo(o[0], o[1]);
    }
    ctx.stroke();
  }

  const LX = -0.48, LY = 0.55, LZ = 0.68;                // แสงหลักจากซ้ายบนด้านหน้า (พิกัดกล้อง: y ขึ้น, z เข้าหากล้อง)
  const TONE = ["20,96,170", "24,209,255", "175,242,255", "240,253,255"];   // เงา -> สีหลัก -> สว่าง -> จุดร้อน

  function drawHolo(ctx, c, sp, now, vis) {
    // วาดซากลงผืนแยก แล้วซ้อนเข้าภาพหลักพร้อมแสงเรือง: ผิวที่หันหากล้องรับแสงตามมุม + ขอบเรือง (fresnel),
    // ผิวด้านหลังมองทะลุได้แต่จาง (โฮโลแกรมโปร่งแสง), วงสแกนแนวนอน, แถบสแกนเลื่อน, เส้นสแกนของภาพฉาย, กะพริบ / สัญญาณรบกวน
    if (!sp || vis <= 0) return;
    const g = S.sctx, dpr = S.dpr, c2 = specCam(c), o = [0, 0, 0], live = !S.reduce;
    const cp = c2.cp, sp2 = c2.sp, ca = c2.ca, sa = c2.sa;
    g.globalCompositeOperation = "source-over";
    g.clearRect(0, 0, S.W, S.H);
    g.globalCompositeOperation = "lighter";
    const scanY = live ? GAP - 10 + (sp.top - GAP + 30) * (1 - ((now * 0.00028) % 1.3)) : -1e9;
    if (live && now > S.glitchAt) {
      S.glitchEnd = now + 90 + Math.random() * 150;
      S.gY = GAP + Math.random() * (sp.top - GAP); S.gH = 4 + Math.random() * 16; S.gS = (Math.random() < 0.5 ? -1 : 1) * (5 + Math.random() * 14) * dpr;
      S.glitchAt = now + 8000 + Math.random() * 8000;
    }
    const glitch = live && now < S.glitchEnd, th = Math.max(1, dpr);
    // (ไม่วาดลำแสงรูปสี่เหลี่ยมคางหมูจากแท่นฉายขึ้นไปหาภาพ — ผู้ใช้ขอเอาออก)
    // ตัวภาพ
    for (let j = 0; j < sp.n; j++) {
      projS(c2, sp.X[j], sp.Y[j], sp.Z[j], o);
      const k = sp.K[j], r = sp.R[j], nx = sp.NX[j], ny = sp.NY[j], nz = sp.NZ[j];
      const nxr = nx * ca + nz * sa, nzr = -nx * sa + nz * ca, nyr = ny * cp - nzr * sp2, face = ny * sp2 + nzr * cp;
      const rim = Math.pow(1 - Math.min(1, Math.abs(face)), 2.2), diff = Math.max(0, nxr * LX + nyr * LY + face * LZ);
      let lit = k === 2 ? 1.1 * sp.A[j] / 0.9 : face >= 0 ? 0.25 + 0.85 * diff + 0.75 * rim : 0.15 + 0.4 * rim;
      lit *= 0.55 + 0.45 * clamp((o[2] * DIST / S.F - 0.86) * 4, 0, 1);         // ไกลกล้อง = จางลง
      if (Math.abs(sp.Y[j] - scanY) < 1.6) lit += 0.9;
      let x = o[0];
      if (glitch && Math.abs(sp.Y[j] - S.gY) < S.gH) x += S.gS;
      const t = lit < 0.4 ? 0 : lit < 0.8 ? 1 : lit < 1.15 ? 2 : 3;
      if (k === 1) {                                  // วงสแกน (ชั้นตัดแนวนอนของรูปทรง) = เส้นสั้นแนวนอน
        const len = (1.2 + 1.8 * r) * o[2];
        g.globalAlpha = Math.min(1, (0.04 + 0.1 * lit) * sp.A[j]);   // จาง: ให้เส้นรอบตัวเด่นกว่า
        g.fillStyle = `rgb(${TONE[t]})`;
        g.fillRect(x - len / 2, o[1], len, th);
      } else {
        // ผิว = จุดแสงใหญ่จางซ้อนกันเป็นเนื้อ, ขอบรูปร่าง = จุดเล็กสว่าง (เส้นรอบตัวคม), รายละเอียด (ตา ฟัน) = ชัด
        const rad = (k === 3 ? 0.9 : k === 2 ? 1.3 : 1.6) * o[2];
        g.globalAlpha = Math.min(1, (k === 3 ? 0.65 : k === 2 ? 0.35 + 0.4 * lit : 0.1 + 0.25 * lit) * sp.A[j] * 1.5);
        g.drawImage(S.spr[k === 3 ? 3 : k === 2 ? Math.max(2, t) : t], x - rad, o[1] - rad, rad * 2, rad * 2);
      }
    }
    // ฝุ่นแสงลอยขึ้นจากแท่นฉาย
    for (const d of S.dust) {
      if (live) { d[1] += d[3] * 0.4; if (d[1] > sp.top + 10) d[1] = GAP - 6; }
      projS(c2, d[0], d[1], d[2], o);
      const life = clamp((d[1] - GAP + 6) / 30, 0, 1) * clamp((sp.top + 10 - d[1]) / 40, 0, 1), rad = 1.4 * o[2];
      g.globalAlpha = 0.35 * life;
      g.drawImage(S.spr[2], o[0] - rad, o[1] - rad, rad * 2, rad * 2);
    }
    // เส้นสแกนของภาพฉาย: ลบแสงเป็นเส้นบาง ๆ ทุก 3 พิกเซล
    g.globalCompositeOperation = "destination-out";
    g.globalAlpha = 1;
    g.fillStyle = S.scanPat;
    g.fillRect(0, 0, S.W, S.H);
    // ซ้อนเข้าภาพหลัก: แสงเรือง (ย่อแล้วขยาย = เบลอ) + ภาพคม; กะพริบเบา ๆ; สีเหลื่อมตอนสัญญาณรบกวน
    const fl = (live ? 0.9 + 0.06 * Math.sin(now * 0.021) + 0.04 * Math.sin(now * 0.0077) : 1) * (glitch ? 0.8 : 1) * vis;
    S.gctx.globalCompositeOperation = "source-over";
    S.gctx.clearRect(0, 0, S.gl.width, S.gl.height);
    S.gctx.drawImage(S.sc, 0, 0, S.gl.width, S.gl.height);
    ctx.globalCompositeOperation = "lighter";
    ctx.globalAlpha = 0.45 * fl;                       // แสงเรืองพอดี ขอบรูปร่างไม่ฟุ้ง
    ctx.drawImage(S.gl, 0, 0, S.W, S.H);
    ctx.globalAlpha = fl;
    ctx.drawImage(S.sc, 0, 0);
    if (glitch) {
      for (const [col, dx] of [["255,60,130", 3], ["70,120,255", -3]]) {
        S.tctx.globalCompositeOperation = "source-over";
        S.tctx.clearRect(0, 0, S.W, S.H);
        S.tctx.drawImage(S.sc, 0, 0);
        S.tctx.globalCompositeOperation = "source-in";
        S.tctx.fillStyle = `rgb(${col})`;
        S.tctx.fillRect(0, 0, S.W, S.H);
        ctx.globalAlpha = 0.45 * fl;
        ctx.drawImage(S.tc, dx * dpr, 0);
      }
    }
    ctx.globalAlpha = 1;
  }

  function drawSpec(ctx, c, sp, now, mode, u) {
    // ตอนเปลี่ยนตัว: "out" / "in" = เส้นแสงพุ่งออก / เข้าหาจุดหายลับ (u 0→1 = ห่างจากรูปร่าง)
    if (!sp) return;
    const dpr = S.dpr, o = [0, 0, 0], n = sp.n, c2 = specCam(c);
    const vc = projS(c2, 0, GAP + (sp.top - GAP) * 0.55, 0, [0, 0, 0]);
    const th = Math.max(1, 1.5 * dpr), alphaAll = 1 - Math.pow(u, 1.6);
    const B = [[], [], []];
    for (let j = 0; j < n; j++) {
      projS(c2, sp.X[j], sp.Y[j], sp.Z[j], o);
      const r = sp.R[j], near = clamp((o[2] * DIST / S.F - 0.86) * 4, 0, 1);
      let x = o[0], y = o[1];
      {
        if (j % 3) continue;
        const dxv = x - vc[0], dyv = y - vc[1], dist = Math.hypot(dxv, dyv) + 1e-3, ux = dxv / dist, uy = dyv / dist;
        const push = u * u * (0.5 + r) * S.W * 0.8;
        x += ux * push; y += uy * push;
        const len = (5 + u * 90 * (0.3 + r)) * dpr;
        B[r > 0.7 ? 1 : 0].push(x, y, ux, uy, len, alphaAll * 0.55 * (0.4 + 0.6 * near));
      }
    }
    const st = [`rgb(${CY})`, `rgb(${HI})`, `rgb(${AMB})`];
    const base = [0.55, 0.85, 1];
    B.forEach((b, k) => {
      {
        ctx.strokeStyle = st[k];
        ctx.lineWidth = th;
        for (let j = 0; j < b.length; j += 6) {
          ctx.globalAlpha = base[k] * b[j + 5];
          ctx.beginPath();
          ctx.moveTo(b[j], b[j + 1]);
          ctx.lineTo(b[j] - b[j + 2] * b[j + 4], b[j + 1] - b[j + 3] * b[j + 4]);
          ctx.stroke();
        }
      }
    });
    ctx.globalAlpha = 1;
  }

  function frame(now) {
    S.raf = requestAnimationFrame(frame);
    if (document.hidden || document.body.dataset.screen !== "index") return;
    const ctx = S.ctx, dt = Math.min(64, now - (S.last || now));
    S.last = now;
    S.px += (S.tx - S.px) * 0.05;
    S.py += (S.tyM - S.py) * 0.05;
    // เวลาในรอบ: hold -> out -> (สลับตัว) -> in -> hold
    if (!S.paused || S.phase !== "hold") S.clock += dt;
    if (S.phase === "hold" && S.clock > (S.reduce ? 12000 : T_HOLD)) next(1);
    if (S.phase === "out" && S.clock > T_OUT) swap();
    if (S.phase === "in" && S.clock > T_IN) { S.phase = "hold"; S.clock = 0; }
    // ยุคที่เรืองแสง / หมุด เลื่อนตามตัวอย่างใหม่อย่างนุ่มนวล
    S.hiY += (S.hiYT - S.hiY) * 0.08;
    S.pin[0] += (S.pinT[0] - S.pin[0]) * 0.08;
    S.pin[1] += (S.pinT[1] - S.pin[1]) * 0.08;

    const c = camera(now);
    ctx.globalCompositeOperation = "source-over";
    ctx.clearRect(0, 0, S.W, S.H);
    ctx.globalCompositeOperation = "lighter";
    const item = S.data.items[S.k];
    drawBlock(ctx, c, now, S.hiIdx);
    drawMap(ctx, c, item, now);
    if (S.phase === "wait") { /* ระหว่างโหลดตัวถัดไป: ไม่วาดซาก */ } else if (S.reduce) {
      drawHolo(ctx, c, S.sp, now, clamp(S.phase === "hold" ? 1 : S.phase === "in" ? S.clock / T_IN : 1 - S.clock / T_OUT, 0, 1));
    } else if (S.phase === "hold") drawHolo(ctx, c, S.sp, now, 1);
    else if (S.phase === "out") {
      const u = ease(clamp(S.clock / T_OUT, 0, 1));
      drawHolo(ctx, c, S.sp, now, 1 - u * 3);
      drawSpec(ctx, c, S.sp, now, "out", u);
    } else if (S.phase === "in") {
      const u = 1 - ease(clamp(S.clock / T_IN, 0, 1));
      drawSpec(ctx, c, S.sp, now, "in", u);
      drawHolo(ctx, c, S.sp, now, 1 - u / 0.35);
    }
    ctx.globalCompositeOperation = "source-over";
  }

  // ---------------------------------------------------------------- เปลี่ยนตัวอย่าง
  function next(step) {
    if (S.phase !== "hold") return;
    S.want = (S.k + step + S.data.items.length) % S.data.items.length;
    S.phase = "out";
    S.clock = 0;
    S.models[S.want] = S.models[S.want] || load(S.want);
  }

  function goto(k) {
    if (k === S.k || S.phase !== "hold") return;
    next(k - S.k);
  }

  function load(k) {
    const it = S.data.items[k];
    // จอใหญ่: จุดหนาแน่นขึ้น; แบบ "clear" เดียวกับหน้าทายชื่อซาก (วงสแกนน้อยลง เส้นรอบตัวคม)
    return HOLO.model(HOLO.shapeFor(it.genus, it.f_type), S.small ? 1 : 1.4, "clear").then(prep);
  }

  async function swap() {
    S.phase = "wait";
    const k = S.want, sp = await S.models[k];
    S.k = k;
    S.sp = sp;
    target(k);
    hud(k);
    S.phase = "in";
    S.clock = 0;
    const nk = (k + 1) % S.data.items.length;
    S.models[nk] = S.models[nk] || load(nk);
  }

  function target(k) {
    const it = S.data.items[k];
    S.hiIdx = S.data.periods.findIndex((p) => p.period === it.period);
    S.hiYT = -BH + (S.hiIdx + 0.5) * S.strata.lh;
    S.pinT = mapXZ(it.site.lon, it.site.lat);
  }

  // ---------------------------------------------------------------- ป้ายข้อมูล (DOM)
  const CONF = { recorded: ["ยุคที่ ทธ. บันทึกไว้", "Period recorded by the DMR"], high: ["ยุคที่ระบบเติม · เชื่อมั่นสูง", "Period filled in by the system · high confidence"],
    medium: ["ยุคที่ระบบเติม · เชื่อมั่นปานกลาง", "Period filled in by the system · medium confidence"],
    low: ["ยุคที่ระบบเติม · เชื่อมั่นต่ำ", "Period filled in by the system · low confidence"] };

  function scramble(el, text) {
    // ข้อความภาษาอังกฤษ / ตัวเลข: ถอดรหัสทีละตัวแบบหน้าจอเครื่องมือ
    if (S.reduce) { el.textContent = text; return; }
    const glyph = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789/.-";
    const t0 = performance.now(), dur = 520;
    const tick = (now) => {
      const u = clamp((now - t0) / dur, 0, 1), n = Math.floor(u * text.length);
      el.textContent = text.slice(0, n) + Array.from(text.slice(n), (ch) => (ch === " " ? " " : glyph[(Math.random() * glyph.length) | 0])).join("");
      if (u < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  function hud(k) {
    const it = S.data.items[k], N = S.data.items.length, el = S.el;
    scramble(el.idx, `SPECIMEN ${String(k + 1).padStart(2, "0")} / ${String(N).padStart(2, "0")} // SCAN`);
    scramble(el.genus, it.genus);
    el.sp.textContent = it.species || `${it.genus} sp.`;
    el.th.textContent = I18N.L([it.com_name, it.f_type !== it.com_name ? it.f_type : null].filter(Boolean).join(" · "), tr(it.f_type));   // ชื่อเรียกภาษาไทยไม่มีคำแปล: ใช้ประเภทซาก
    const per = S.data.periods.find((q) => q.period === it.period);
    const range = S.opt.range ? S.opt.range(it.period) : "";
    el.age.innerHTML = `<b>${esc(I18N.period(it.period))}</b><span>${esc(range)}${per ? I18N.L(` · แหล่ง ทธ. ในยุคนี้ ${per.sites} แหล่ง`, ` · ${per.sites} DMR sites from this period`) : ""}</span>`;
    el.conf.textContent = CONF[it.confidence] ? I18N.L(...CONF[it.confidence]) : "";
    el.site.textContent = it.site.name;
    el.prov.textContent = I18N.L(`จังหวัด${it.site.province}`, `${tr(it.site.province)} Province`);
    scramble(el.xy, `${fmt(it.site.lat, "N", "S")}  ${fmt(it.site.lon, "E", "W")}`);
    el.n.innerHTML = I18N.L(`<b>${it.n.toLocaleString()}</b> ชิ้นในทะเบียน · <b>${it.site.n.toLocaleString()}</b> ชิ้นที่แหล่งนี้`,
      `<b>${it.n.toLocaleString()}</b> registered specimens · <b>${it.site.n.toLocaleString()}</b> at this site`);
    el.note.textContent = HOLO.note(HOLO.shapeFor(it.genus, it.f_type));
    el.segs.forEach((s, i) => s.classList.toggle("on", i === k));
  }

  function build(fig) {
    fig.innerHTML = `<canvas class="scan-cv" aria-hidden="true"></canvas>
      <div class="scan-hud">
        <div class="sh-tl"><div class="sh-k" data-f="idx"></div><div class="sh-genus" data-f="genus"></div>
          <div class="sh-sp" data-f="sp"></div><div class="sh-th" data-f="th"></div></div>
        <div class="sh-tr"><div class="sh-k" data-f="kAge"></div><div class="sh-age" data-f="age"></div><div class="sh-conf" data-f="conf"></div></div>
        <div class="sh-bl"><div class="sh-k" data-f="kLoc"></div><div class="sh-site" data-f="site"></div>
          <div class="sh-prov"><span data-f="prov"></span> <span class="sh-xy" data-f="xy"></span></div><div class="sh-n" data-f="n"></div></div>
      </div>
      <div class="scan-ctrl"><button class="sc-btn" data-d="-1" data-f="prev">‹</button><div class="sc-segs"></div>
        <button class="sc-btn" data-d="1" data-f="next">›</button></div>
      <figcaption><span data-f="note"></span><br><span data-f="cap"></span></figcaption>`;
    const el = {};
    fig.querySelectorAll("[data-f]").forEach((e) => (el[e.dataset.f] = e));
    const segs = fig.querySelector(".sc-segs");
    segs.innerHTML = S.data.items.map((it, i) => `<button class="sc-seg" data-k="${i}" aria-label="${esc(it.genus)}"><i></i></button>`).join("");
    el.segs = [...segs.children];
    el.segs.forEach((b) => b.addEventListener("click", () => goto(+b.dataset.k)));
    fig.querySelectorAll(".sc-btn").forEach((b) => b.addEventListener("click", () => next(+b.dataset.d)));
    fig.addEventListener("pointermove", (e) => {
      const r = fig.getBoundingClientRect();
      S.tx = clamp((e.clientX - r.left) / r.width - 0.5, -0.5, 0.5);
      S.tyM = clamp((e.clientY - r.top) / r.height - 0.5, -0.5, 0.5);
    });
    fig.addEventListener("pointerenter", () => (S.paused = true));
    fig.addEventListener("pointerleave", () => { S.paused = false; S.tx = 0; S.tyM = 0; });
    return el;
  }

  function statics(el) {
    // ข้อความคงที่ของป้าย (เปลี่ยนตามภาษา)
    el.kAge.textContent = I18N.L("AGE // ยุคของแหล่งที่พบ", "AGE // PERIOD OF THE SITE");
    el.kLoc.textContent = I18N.L("LOCALITY // แหล่งที่พบมากสุด", "LOCALITY // MOST FINDS");
    el.prev.setAttribute("aria-label", I18N.L("ตัวอย่างก่อนหน้า", "Previous specimen"));
    el.next.setAttribute("aria-label", I18N.L("ตัวอย่างถัดไป", "Next specimen"));
    el.cap.textContent = I18N.L("ชั้นหิน = ยุคธรณี (ล่างเก่า บนใหม่) · ความหนาแน่นของเส้น / แถบ = จำนวนแหล่ง ทธ. ของยุคนั้น · ผิวบน = แผนที่โอกาสพบซาก 5 กม.",
      "Rock layers = geological periods (oldest at the bottom) · line density / bars = number of DMR sites from each period · top surface = 5 km fossil potential map");
  }

  // ---------------------------------------------------------------- API
  async function start(fig, data, opt = {}) {
    if (!fig || !data || !data.items || !data.items.length) return;
    const k0 = Math.max(0, data.items.findIndex((it) => it.genus === (opt.first || "Phuwiangosaurus")));
    S = { data, opt, k: k0, models: [], phase: "wait", clock: 0, paused: false, px: 0, py: 0, tx: 0, tyM: 0,
      reduce: matchMedia("(prefers-reduced-motion: reduce)").matches, dpr: Math.min(2, window.devicePixelRatio || 1),
      strata: strata(data.periods), map: null, sites: null, hiIdx: 0, hiY: 0, hiYT: 0, pin: [0, 0], pinT: [0, 0],
      spr: TONE.map(sprite), glitchAt: performance.now() + 3000, glitchEnd: 0, gY: 0, gH: 0, gS: 0,
      dust: Array.from({ length: 18 }, () => [(Math.random() - 0.5) * SPEC_W * 0.7, GAP + Math.random() * SPEC_H, (Math.random() - 0.5) * 90, 0.3 + Math.random() * 0.7]) };
    S.el = build(fig);
    statics(S.el);
    S.cv = fig.querySelector(".scan-cv");
    S.ctx = S.cv.getContext("2d");
    if (!S.ctx) return;
    fit();
    new ResizeObserver(fit).observe(S.cv);
    target(k0);
    S.hiY = S.hiYT;
    S.pin = S.pinT.slice();
    hud(k0);
    S.models[k0] = load(k0);
    S.sp = await S.models[k0];
    S.phase = "in";
    S.clock = 0;
    S.models[(k0 + 1) % data.items.length] = load((k0 + 1) % data.items.length);
    S.raf = requestAnimationFrame(frame);
  }

  function setMap(grid, sites) {
    // แผนที่บนผิวบน: ช่อง 5 กม. (ทุกช่องที่ 3) ความสว่างตามอันดับโอกาสพบซาก (bg_bias) 4 ระดับ + แหล่ง ทธ.
    if (!S || !grid) return;
    const X = [], Z = [], V = [];
    const rank = grid.rank.bg_bias;
    for (let i = 0; i < grid.n; i += 3) {
      const [x, z] = mapXZ(grid.lon[i], grid.lat[i]), r = rank[i] / 1000;
      X.push(x); Z.push(z); V.push(r >= 0.9 ? 3 : r >= 0.75 ? 2 : r >= 0.5 ? 1 : 0);
    }
    S.map = { X: Float32Array.from(X), Z: Float32Array.from(Z), V: Uint8Array.from(V) };
    S.sites = sites ? Float32Array.from(sites.flatMap((s) => mapXZ(s.lon, s.lat))) : null;
  }

  function relabel() {
    // เปลี่ยนภาษา: ป้ายข้อความ (ชื่อยุคบนภาพวาดใหม่เองทุกเฟรม)
    if (!S || !S.el) return;
    statics(S.el);
    hud(S.k);
  }

  return { start, setMap, relabel };
})();
