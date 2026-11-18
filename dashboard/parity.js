// node dashboard/parity.js  -> compares JS matcher with Python output (outputs/parity.json)
const fs = require("fs"), path = require("path");
const { Matcher } = require("./matcher.js");
const lex = JSON.parse(fs.readFileSync(path.join(__dirname, "data/lexicon.json")));
const fx = JSON.parse(fs.readFileSync(path.join(__dirname, "../outputs/parity.json")));
const mk = { embed: new Matcher(lex, { embed: true }), rule: new Matcher(lex, { embed: false }) };
let bad = 0, total = 0;
for (const mode of ["embed", "rule"]) {
  for (const row of fx) {
    total++;
    const r = mk[mode].match(row.text, row.colour), py = row[mode];
    const hits = r.hits.map(h => [h.kind, h.f1, h.f2, h.surface, h.neg, h.tag, h.scope, h.amb, h.ctiers]);
    const ok = JSON.stringify(r.toks) === JSON.stringify(py.toks) && JSON.stringify(hits) === JSON.stringify(py.hits)
      && JSON.stringify(r.years) === JSON.stringify(py.years);
    if (!ok) { bad++; if (bad <= 5) console.log("MISMATCH", mode, JSON.stringify(row.text.slice(0, 120)), "\n JS:", JSON.stringify(hits.slice(0, 6)), "\n PY:", JSON.stringify(py.hits.slice(0, 6))); }
  }
}
console.log(`parity: ${total - bad}/${total} identical`);
process.exit(bad ? 1 : 0);
