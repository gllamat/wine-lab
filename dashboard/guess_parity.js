// node dashboard/guess_parity.js -> Python guess model vs JS port on outputs/guess_parity.json
const fs = require("fs"), path = require("path"), { Matcher, resolveRipeness } = require("./matcher.js"), G = require("./guess.js");
const lex = JSON.parse(fs.readFileSync(path.join(__dirname, "data/lexicon.json"))), M = JSON.parse(fs.readFileSync(path.join(__dirname, "data/guess_model.json")));
const fx = JSON.parse(fs.readFileSync(path.join(__dirname, "../outputs/guess_parity.json"))), mt = new Matcher(lex, { embed: false });
let bad = 0, maxd = 0; const cmpTop = (js, py, tol) => py.every(([c, p], i) => js[i][0] === c ? Math.abs(js[i][1] - p) < tol : Math.abs(js[i][1] - p) < tol && Math.abs(js[i][1] - py[i][1]) < tol);
for (const row of fx) {
  const r = mt.match(row.text, null), x = G.featurize(r, M.spec, resolveRipeness), o = G.predict(M, x);
  const chk = [["variety", G.top(o.Pv, M.variety.classes, 3), row.variety], ["place", G.top(o.Pp, M.place.classes, 3), row.place], ["group", G.top(o.Pg, M.variety.groups, 3), row.group], ["producer", G.top(o.Pw, M.producer.wineries, 3), row.producer]];
  let ok = true;
  for (const [n, js, py] of chk) { for (let i = 0; i < 3; i++) { const dp = Math.abs(js[i][1] - py[i][1]); maxd = Math.max(maxd, dp); if (dp > 3e-3) ok = false; } }
  for (const k of ["vintage", "price", "points"]) row[k].forEach((v, i) => { const d = Math.abs(o[k][i] - v); maxd = Math.max(maxd, d / (k === "vintage" ? 10 : 1)); if (d > 0.02) ok = false; });
  const pt = G.placePoint(M, o.Pp); if (Math.abs(pt[0] - row.point[0]) > 0.01 && Math.abs(pt[1] - row.point[1]) > 0.01) ok = false;
  if (!ok) { bad++; if (bad <= 3) console.log("MISMATCH", JSON.stringify(row.text.slice(0, 80)), JSON.stringify(row.variety[0]), JSON.stringify(G.top(o.Pv, M.variety.classes, 1)), JSON.stringify(row.place[0]), JSON.stringify(G.top(o.Pp, M.place.classes, 1))); }
}
console.log(`guess parity: ${fx.length - bad}/${fx.length} within tolerance (max prob diff ${maxd.toExponential(2)})`); process.exit(bad ? 1 : 0);
