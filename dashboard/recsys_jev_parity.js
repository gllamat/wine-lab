// node dashboard/recsys_jev_parity.js -> Python (src/recsys_jev.py) vs the JS port: Jev vector, taster's rules and hub correction on the same quantised data
const fs = require("fs"), path = require("path"), { RecIndex } = require("./recsys.js"), D = path.join(__dirname, "data");
const meta = JSON.parse(fs.readFileSync(path.join(D, "recsys_meta.json"))), rd = (f, T) => { const b = fs.readFileSync(path.join(D, f)); return new T(b.buffer, b.byteOffset, b.length / T.BYTES_PER_ELEMENT); };
const R = new RecIndex(meta, rd("recsys_x.bin", Int8Array), rd("recsys_e.bin", Int8Array)); R.attachJev(rd("recsys_j.bin", Int8Array), rd("recsys_hub.bin", Uint8Array));
const fx = JSON.parse(fs.readFileSync(path.join(__dirname, "../outputs/recsys_jev_parity.json")));
const CFG = { jev: ["jev", {}], jevh: ["jevh", {}], jev_rules: ["jev", { rules: true }], jevh_rules: ["jevh", { rules: true }], jevh_hub: ["jevh", { hub: true }], jevh_rules_hub: ["jevh", { rules: true, hub: true }], emb_hub: ["emb", { hub: true }] };
let bad = 0;
for (const [key, [t, o]] of Object.entries(CFG)) { const a = fx.map(r => { const js = R.technique(t, r.i, 10, Object.assign({ hub: false }, o)), py = r[key]; return js.filter(x => py.includes(x)).length / 10; }), m = a.reduce((x, y) => x + y, 0) / a.length, ex = a.filter(v => v === 1).length / a.length;
  console.log(`${key.padEnd(15)} mean overlap@10 ${m.toFixed(3)} | identical lists ${(100 * ex).toFixed(0)}%`); if (m < 0.97) bad++; }
console.log(bad ? "recsys jev parity: FAIL" : "recsys jev parity: OK"); process.exit(bad ? 1 : 0);
