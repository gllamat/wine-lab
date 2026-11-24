// node dashboard/phrases_parity.js -> Python reference vs the JS port of the typed-note path (tokeniser + learned phrases)
const fs = require("fs"), path = require("path"), { phrases, PhraseVec } = require("./phrases.js");
const P = JSON.parse(fs.readFileSync(path.join(__dirname, "data/phrases.json"))), fx = JSON.parse(fs.readFileSync(path.join(__dirname, "../outputs/phrases_parity.json"))), V = new PhraseVec(P);
let tokBad = 0, worst = 0; for (const r of fx) { const t = phrases(r.text).slice(0, 60); if (JSON.stringify(t) !== JSON.stringify(r.tokens)) tokBad++; const { z } = V.vector(r.text, r.cue); r.z.forEach((v, j) => { worst = Math.max(worst, Math.abs(v - z[j])); }); }
console.log(`tokens identical in ${fx.length - tokBad}/${fx.length} texts | max |z difference| ${worst.toExponential(2)}`); const ok = tokBad === 0 && worst < 2e-3; console.log(ok ? "phrases parity: OK" : "phrases parity: FAIL"); process.exit(ok ? 0 : 1);
