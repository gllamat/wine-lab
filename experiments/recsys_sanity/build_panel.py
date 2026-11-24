"""Builds panel_preview.html: a stand-alone preview of the 'Does it make sense?' panel (NOT part of the deployed dashboard).
Reads results.json and sanity_lists.json written by benchmark.py."""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
res, lists = json.load(open(H / "results.json")), json.load(open(H / "sanity_lists.json"))
page = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Does it make sense? Preview</title>
<style>
:root{--bg:#fbf8f3;--fg:#2a2118;--mut:#7a6d5e;--card:#fff;--line:#e4dccf;--ok:#2f7d4f;--bad:#b3392f;--warn:#c58a1b;--acc:#7b2d3b}
@media(prefers-color-scheme:dark){:root{--bg:#1d1814;--fg:#efe7dc;--mut:#a89a88;--card:#272019;--line:#3a3128;--ok:#6cc08a;--bad:#e5766c;--warn:#e0b04c;--acc:#d98a9a}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}main{max-width:1100px;margin:0 auto;padding:20px 16px 60px}
h1{font-size:1.6rem;margin:.2em 0}h2{font-size:1.2rem;margin:1.6em 0 .4em}p,li{max-width:70ch}.mut{color:var(--mut)}
.banner{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--warn);padding:10px 14px;border-radius:6px;margin:12px 0}
table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);border-radius:6px;overflow:hidden;font-size:.92rem}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left}th{font-weight:600;background:rgba(127,127,127,.08)}td.n{font-variant-numeric:tabular-nums}
.bar{display:inline-block;height:9px;border-radius:2px;vertical-align:middle;margin-left:6px}
.pick{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0}select,button{font:inherit;padding:5px 8px;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px}
.q{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin:14px 0}.q h3{margin:0 0 2px;font-size:1.02rem}
.wine{display:grid;grid-template-columns:1.6em 1fr auto;gap:2px 10px;padding:6px 0;border-top:1px solid var(--line);font-size:.92rem}.wine .t{font-weight:600}.wine .s{grid-column:2;color:var(--mut);font-size:.84rem}
.tag{display:inline-block;padding:0 6px;border-radius:9px;font-size:.76rem;margin-left:4px;border:1px solid}.t-bad{color:var(--bad);border-color:var(--bad)}.t-ok{color:var(--ok);border-color:var(--ok)}.t-warn{color:var(--warn);border-color:var(--warn)}
.hd{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap}
</style></head><body><main>
<h1>Does it make sense?</h1>
<div class="banner"><b>Preview only.</b> This panel is not on the website yet. It shows what we would add to the recommender tab: how to test a recommender <i>with a taster's eyes</i>, not just with a number.</div>
<p>A recommender can look clever on a few examples and still be wrong too often. We test it two ways: <b>(1)</b> a measurement on 1,500 queries, using the answers Jev gave to the WSET card as the yardstick (the recommender never saw those answers), and <b>(2)</b> 30 queries you can read yourself, with clashes flagged automatically.</p>
<h2>1. The measurement</h2>
<p class="mut">Each technique recommends 10 wines for each of 1,500 query wines. Colour clashes use the wine's colour; style clashes and taste similarity use Jev's answers, for the recommended wines that Jev has also labelled (about 37%).</p>
<table id="m"></table>
<h2>3. The fix: a taste vector that speaks WSET</h2>
<p>The four <b>Jev vector</b> rows use a new description of every wine: Jev's 216 yes/no answers and the level of eight structure scales (sweetness, acidity, tannin, alcohol, body, flavour intensity, finish, nose intensity), <b>predicted from the words of the note</b> for all 25,000 wines (Jev only labelled 37% of them). This is the distillation idea reused: expert labels give the features a meaning a taster recognises. No wine's own Jev answers are used to recommend it: labelled wines are predicted by models that never saw them.</p>
<h2>4. The taster's rules switch</h2>
<p>Data describes taste but does not know the trade's common sense. The <b>taster's rules</b> are three lines a taster would write down: <b>same colour family</b> (a red is not offered for a white, a sparkling is not offered for a still wine), and <b>no sweetness jump</b> (candidates are at most one sweetness level away from the query; the limit was chosen on 600 other queries, not the ones measured here). The rules filter the candidate wines <i>before</i> the top 10 is picked, so each removed wine is replaced by the next best one. In the list viewer below, the switch shows exactly which wines the rules removed and why.</p>
<h2>What the table teaches</h2><ul id="lessons"></ul>
<h2>2. Read the lists yourself</h2>
<div class="pick"><label>Technique <select id="tech"></select></label><label><input type="checkbox" id="rules" checked> Taster's rules</label><label>Query wine <select id="qs"></select></label><button id="prev">&larr;</button><button id="next">&rarr;</button><span class="mut" id="cnt"></span></div>
<div id="out"></div>
<h2>Limits</h2><ul><li>Jev is a model, not a taster: its answers are a useful yardstick, not the truth.</li><li>A different grape is not automatically a bad recommendation (Albariño can suit a Vermentino lover), which is why grape match is shown but not used as the score.</li><li>Style clashes are measured only where both wines have Jev answers.</li></ul>
</main><script>
const RES=__RES__, LISTS=__LISTS__;
const NAMES={random:"random (same colour)",raw:"raw cues",zw:"zw (rare cues count more)",latent:"latent (SVD)",emb:"text embedding",hybrid:"hybrid",rrf:"rank fusion (rrf)",mmr:"mmr (diverse)",jev_desc:"Jev vector: answers only",jev:"Jev vector: answers + 8 structure levels",jev_hybrid:"Jev vector + text (hybrid)",jev_rrf:"Jev vector + text (rrf)","hybrid+rules":"hybrid + taster's rules","rrf+rules":"rrf + taster's rules","jev+rules":"Jev vector + taster's rules","jev_hybrid+rules":"Jev vector + text + taster's rules"};
const T=RES.techniques, order=Object.keys(NAMES), pct=x=>(100*x).toFixed(0)+"%";
function row(k){const t=T[k];return `<tr><td><b>${NAMES[k]}</b></td><td class=n>${pct(t.lists_with_hard_clash)}${bar(t.lists_with_hard_clash,1,'var(--bad)')}</td><td class=n>${pct(t.style_clash_any)}${bar(t.style_clash_any,.5,'var(--warn)')}</td><td class=n>${t.taste_cosine.toFixed(2)}${bar(t.taste_cosine,.6,'var(--ok)')}</td><td class=n>${pct(t.grape_group)}</td><td class=n>${t.wines_swapped_per_list!==undefined?t.wines_swapped_per_list.toFixed(1):'&ndash;'}</td></tr>`}
function bar(v,mx,c){return `<span class=bar style="width:${Math.max(2,60*v/mx)}px;background:${c}"></span>`}
document.getElementById("m").innerHTML=`<tr><th>Technique</th><th>lists with a red&harr;white or sparkling&harr;still wine (lower is better)</th><th>wine pairs with a sweetness or body gap of 2+ levels (lower is better)</th><th>taste similarity to the query, Jev answers (higher is better)</th><th>same grape group</th><th>wines swapped per list by the rules</th></tr>`+order.map(row).join("");
const b=T.rrf, r=T.random, z=T.zw;
document.getElementById("lessons").innerHTML=[
`<b>Two lists in five have a clash.</b> Even the best technique here (${NAMES.hybrid}) puts at least one red-for-white or sparkling-for-still wine in <b>${pct(T.hybrid.lists_with_hard_clash)}</b> of its 10-wine lists. No average score would have told us: the clashes had to be counted.`,
`<b>The yardstick matters.</b> The cue-based techniques match the query's grape group only ${pct(z.grape_group)} of the time, about the same as random picks (${pct(r.grape_group)}), yet their wines are far more similar in taste to the query (${z.taste_cosine.toFixed(2)} against ${r.taste_cosine.toFixed(2)} for random). Judged by grape they look useless; judged by taste they work. Choose the measure that matches the question.`,
`<b>Each technique trades one thing for another.</b> The text embedding finds the same grape most often (${pct(T.emb.grape_group)}) but is the worst on body clashes and taste similarity; the cue techniques are the reverse. The fusion (hybrid) keeps most of both.`,
`<b>Give the vector a taster's vocabulary and the clashes drop.</b> Replacing the hand-written cue vector by the Jev-aligned vector cuts sweetness-or-body gaps from ${pct(T.hybrid.style_clash_any)} (${NAMES.hybrid}) to ${pct(T.jev.style_clash_any)}. Adding just 8 structure numbers (sweetness, body...) to the 216 answers matters: without them (${NAMES.jev_desc}) the gaps stay at ${pct(T.jev_desc.style_clash_any)}.`,
`<b>Beware of marking your own homework.</b> The "taste similarity" column uses Jev's answers, and the new vector was trained to predict those answers, so its high score there (${T.jev.taste_cosine.toFixed(2)}) is partly built in. The style-gap and colour columns are the fairer evidence, because they compare the predicted vector with Jev's <i>true</i> levels of the recommended wines.`,
`<b>The fusion keeps what each side knows.</b> ${NAMES.jev_hybrid} keeps the text's grape match (${pct(T.jev_hybrid.grape_group)}, against ${pct(T.jev.grape_group)} for the vector alone) while keeping most of the style gain.`,
`<b>Data still needs the taster's rule.</b> Even the best row puts a red-for-white or sparkling-for-still wine in ${pct(T.jev_hybrid.lists_with_hard_clash)} of lists: the vector describes taste but does not know that nobody swaps a Chablis for a Barolo. That is the next rung, the taster's rules switch.`,
`<b>The taster's rules remove every hard clash, and cost almost nothing.</b> They are 0% by construction, which is the point of a rule. What matters is the price: for ${NAMES.jev_hybrid} the taste similarity moves from ${T.jev_hybrid.taste_cosine.toFixed(2)} to ${T["jev_hybrid+rules"].taste_cosine.toFixed(2)}, the sweetness-or-body gaps fall from ${pct(T.jev_hybrid.style_clash_any)} to ${pct(T["jev_hybrid+rules"].style_clash_any)}, and the same-grape match <i>rises</i> from ${pct(T.jev_hybrid.grape_group)} to ${pct(T["jev_hybrid+rules"].grape_group)}.`,
`<b>The counter tells you what the data alone proposed.</b> To satisfy the rules, ${T["hybrid+rules"].wines_swapped_per_list.toFixed(1)} of 10 wines had to be swapped in the old hybrid's lists, and ${T["jev_hybrid+rules"].wines_swapped_per_list.toFixed(1)} even with the Jev vector. Constraints are not a patch for bad data: a better vector only reduces how often they act.`,
`<b>Rules inherit the errors of what they read.</b> Colour comes from the grape (a field of the data), the sweetness level is predicted from the words, so where either is wrong a good wine can be dropped or a bad one kept.`,
`<b>"Random (same colour)" is the honest baseline.</b> It has no clashes by construction, so everything else must be read against it: the real aim is to keep the taste similarity <i>and</i> remove the clashes. A taster's rule (never mix red with white) is exactly what the data alone does not know.`
].map(x=>`<li>${x}</li>`).join("");
const tech=document.getElementById("tech"), qs=document.getElementById("qs"), out=document.getElementById("out"), cnt=document.getElementById("cnt");
["raw","zw","emb","hybrid","rrf","jev","jev_hybrid","random"].forEach(k=>tech.add(new Option(NAMES[k],k)));tech.value="jev_hybrid";const rules=document.getElementById("rules");const RULED=["hybrid","rrf","jev","jev_hybrid"];
LISTS.forEach((e,i)=>qs.add(new Option(`${i+1}. ${e.query.variety} (${e.query.colour}) - ${e.query.title.slice(0,50)}`,i)));
function flag(w){let f=w.flags,s="";if(f.colour_clash)s+='<span class="tag t-bad">red/white clash</span>';if(f.sparkling_still)s+='<span class="tag t-bad">sparkling/still</span>';
 if(f.sweetness_gap!==null&&f.sweetness_gap>=2)s+=`<span class="tag t-warn">sweetness gap ${f.sweetness_gap}</span>`;
 s+=f.same_grape?'<span class="tag t-ok">same grape group</span>':'';return s}
function render(){const e=LISTS[+qs.value],q=e.query,ruled=RULED.includes(tech.value);rules.disabled=!ruled;const useR=ruled&&rules.checked,key=useR?tech.value+'+rules':tech.value,L=e.lists[key],rem=useR?(e.removed[tech.value]||[]):[];const bad=L.filter(w=>w.flags.colour_clash||w.flags.sparkling_still).length;
 cnt.textContent=`${+qs.value+1} / ${LISTS.length}`;
 out.innerHTML=`<div class=q><div class=hd><h3>Query: ${q.title}</h3><span class=mut>${q.variety} &middot; ${q.country} &middot; ${q.points} pts${q.sweetness!==null?` &middot; sweetness level ${q.sweetness}, body ${q.body}`:''}</span></div><div class=mut>${q.note}&hellip;</div>
 <p><b>${NAMES[key]}</b> recommends: <span class="tag ${bad?'t-bad':'t-ok'}">${bad} hard clash${bad==1?'':'es'} in 10</span></p>`+
 L.map((w,i)=>`<div class=wine><span class=mut>${i+1}</span><span><span class=t>${w.title}</span>${flag(w)}</span><span class=mut>${w.points} pts</span><span></span><span class=s>${w.variety} &middot; ${w.country} &middot; ${w.colour}${w.sweetness!==null?` &middot; sweetness ${w.sweetness}`:''}</span></div>`).join("")+`${rem.length?`<div class=mut style="margin-top:8px"><b>The taster's rules removed ${rem.length} wine${rem.length>1?'s':''} from the plain list:</b> ${rem.map(w=>`<span style="text-decoration:line-through">${w.title}</span> (${w.why})`).join('; ')}</div>`:(useR?'<div class=mut style="margin-top:8px">The rules removed nothing here: the plain list already passed.</div>':'')}`+`</div>`}
tech.onchange=qs.onchange=rules.onchange=render;document.getElementById("prev").onclick=()=>{qs.value=(+qs.value+LISTS.length-1)%LISTS.length;render()};document.getElementById("next").onclick=()=>{qs.value=(+qs.value+1)%LISTS.length;render()};render();
</script></body></html>"""
page = page.replace("__RES__", json.dumps(res)).replace("__LISTS__", json.dumps(lists))
(H / "panel_preview.html").write_text(page); print("wrote", H / "panel_preview.html", round(len(page) / 1024), "KB")
