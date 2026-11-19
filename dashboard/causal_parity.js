// node dashboard/causal_parity.js -> Python (src/causal.py) vs JS on outputs/causal_parity.json
const fs = require("fs"), path = require("path"), { DAG } = require("./causal.js");
const L = JSON.parse(fs.readFileSync(path.join(__dirname, "data/dags.json"))), fx = JSON.parse(fs.readFileSync(path.join(__dirname, "../outputs/causal_parity.json")));
const G = Object.fromEntries(L.map(l => [l.id, [new DAG(l.nodes, l.edges), l.conditioned]]));
let bad = 0; const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
for (const r of fx) {
  const [g, cond] = G[r.dag];
  if (r.identify) { const o = g.identify(r.x, r.y, cond); if (!same(o.backdoor, r.identify.backdoor) || !same(o.frontdoor, r.identify.frontdoor)) { bad++; console.log("IDENTIFY mismatch", r.dag, JSON.stringify(o), JSON.stringify(r.identify)); } continue; }
  const o = [g.dsep([r.x], [r.y], r.Z), g.backdoorValid(r.x, r.y, r.Z, cond)[0], g.rule1([r.y], [r.x], r.Z.slice(0, 1), r.W), g.rule2([r.y], [r.x], r.Z.slice(0, 1), r.W), g.rule3([r.y], [r.x], r.Z.slice(0, 1), r.W)], p = [r.dsep, r.backdoor, r.r1, r.r2, r.r3];
  if (!same(o, p)) { bad++; if (bad < 5) console.log("MISMATCH", r.dag, r.x, r.y, JSON.stringify(r.Z), JSON.stringify(r.W), JSON.stringify(o), JSON.stringify(p)); }
}
console.log(`causal parity: ${fx.length - bad}/${fx.length} identical`); process.exit(bad ? 1 : 0);
