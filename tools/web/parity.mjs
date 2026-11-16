// รัน webapp/static/local.js ใน Node (onnxruntime-web แบบ wasm + @huggingface/tokenizers รุ่นเดียวกับที่หน้าเว็บโหลด)
//   node tools/web/parity.mjs <cases.jsonl> <out.jsonl> [--data DIR] [--models DIR]
//   cases: {"kind": "specimen" | "location" | "tokens" | "tfidf" | "clean" | "province" | "symbol", "body": {...}} ทีละบรรทัด
//   ใช้โดย tests/test_static_site.py เพื่อเทียบผลกับ Python
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..", "..");
const require = createRequire(import.meta.url);
const args = process.argv.slice(2);
const opt = (k, d) => (args.includes(k) ? args[args.indexOf(k) + 1] : d);
const [casesFile, outFile] = args;
const dataDir = opt("--data", path.join(root, "webapp", "data"));
const modelDir = opt("--models", path.join(root, "models", "web"));

const L = require(path.join(root, "webapp", "static", "local.js"));
const pkg = JSON.parse(fs.readFileSync(path.join(here, "package.json"), "utf8"));
if (pkg.dependencies["onnxruntime-web"] !== L.versions.ORT_VERSION || pkg.dependencies["@huggingface/tokenizers"] !== L.versions.TOKENIZERS_VERSION) {
  throw new Error("เวอร์ชันใน tools/web/package.json ไม่ตรงกับ local.js");
}
const file = (url) => {
  for (const [pre, dir] of [["data/", dataDir], ["models/", modelDir]]) if (url.startsWith(pre)) return path.join(dir, url.slice(pre.length));
  throw new Error(url);
};
Object.assign(L.cfg, {
  fetchJSON: async (url) => JSON.parse(fs.readFileSync(file(url), "utf8")),
  fetchBuffer: async (url) => { const b = fs.readFileSync(file(url)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); },
  importOrt: () => import("onnxruntime-web"),
  importTokenizers: () => import("@huggingface/tokenizers"),
});

const out = fs.openSync(outFile, "w");
for (const line of fs.readFileSync(casesFile, "utf8").split("\n")) {
  if (!line.trim()) continue;
  const c = JSON.parse(line);
  let r;
  try {
    if (c.kind === "specimen") r = await L.predictSpecimen(c.body);
    else if (c.kind === "location") r = await L.predictLocation(c.body);
    else if (c.kind === "tokens") r = await L.bertIds(c.body.text);
    else if (c.kind === "tfidf") r = await L.tfidf(c.body.text);
    else if (c.kind === "clean") r = await L.cleanPart(c.body.text);
    else if (c.kind === "province") r = await L.provinceAt(c.body.lon, c.body.lat);
    else if (c.kind === "symbol") r = await L.symbolAt(c.body.lon, c.body.lat);
    else throw new Error(`kind ${c.kind}`);
    fs.writeSync(out, JSON.stringify({ id: c.id, ok: true, result: r }) + "\n");
  } catch (e) {
    fs.writeSync(out, JSON.stringify({ id: c.id, ok: false, error: String(e.message || e) }) + "\n");
  }
}
fs.closeSync(out);
