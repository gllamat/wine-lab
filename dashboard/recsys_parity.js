// node dashboard/recsys_parity.js -> Python recommender vs JS port (same quantised data)
const fs = require("fs"), path = require("path"), { RecIndex } = require("./recsys.js"), D = path.join(__dirname, "data");
const meta = JSON.parse(fs.readFileSync(path.join(D, "recsys_meta.json"))), rd = (f, T) => { const b = fs.readFileSync(path.join(D, f)); return new T(b.buffer, b.byteOffset, b.length / T.BYTES_PER_ELEMENT); };
const R = new RecIndex(meta, rd("recsys_x.bin", Int8Array), rd("recsys_e.bin", Int8Array)), fx = JSON.parse(fs.readFileSync(path.join(__dirname, "../outputs/recsys_parity.json")));
const stat = {}; for (const r of fx) for (const t of ["raw", "zw", "latent", "emb", "hybrid", "rrf", "mmr"]) { const js = R.technique(t, r.i, 10), py = r[t], ov = js.filter(x => py.includes(x)).length / 10; (stat[t] = stat[t] || []).push(ov); }
let bad = 0; for (const [t, a] of Object.entries(stat)) { const m = a.reduce((x, y) => x + y, 0) / a.length, ex = a.filter(v => v === 1).length / a.length; console.log(`${t.padEnd(7)} mean overlap@10 ${m.toFixed(3)} | identical lists ${(100 * ex).toFixed(0)}%`); if (m < 0.97) bad++; }
console.log(bad ? "recsys parity: FAIL" : "recsys parity: OK"); process.exit(bad ? 1 : 0);
