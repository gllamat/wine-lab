"""Causal graph toolkit: d-separation, back-door / front-door criteria, adjustment sets, and the three rules of do-calculus.
A DAG library lives in lexicon/dags/*.yaml (the base graph is lexicon/dag.yaml). The browser has the same algorithms in dashboard/causal.js;
`python src/causal.py fixture` writes a Python-vs-JS parity fixture and `python src/causal.py export` writes dashboard/data/dags.json.
Latent nodes (kind: latent) can never be adjusted for. `conditioned` = nodes we are always conditioned on (e.g. 'selection': we only see reviewed wines)."""
import sys, json, itertools, random
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parent.parent


class DAG:
    def __init__(self, nodes, edges):
        self.nodes = dict(nodes); self.edges = [tuple(e) for e in edges]
        self.par = {n: set() for n in self.nodes}; self.ch = {n: set() for n in self.nodes}
        for a, b in self.edges: self.ch[a].add(b); self.par[b].add(a)

    def sub(self, remove_into=(), remove_out_of=()):
        into, out = set(remove_into), set(remove_out_of)
        return DAG(self.nodes, [(a, b) for a, b in self.edges if b not in into and a not in out])

    def anc(self, S, incl=True):
        seen, stack = set(S) if incl else set(), list(S)
        while stack:
            for p in self.par[stack.pop()]:
                if p not in seen: seen.add(p); stack.append(p)
        return seen

    def desc(self, S, incl=False):
        seen, stack = set(S) if incl else set(), list(S)
        while stack:
            for c in self.ch[stack.pop()]:
                if c not in seen: seen.add(c); stack.append(c)
        return seen

    def dsep(self, X, Y, Z):
        X, Y, Z = set(X), set(Y), set(Z); keep = self.anc(X | Y | Z)
        und = {n: set() for n in keep}
        for b in keep:                                                  # moralise the ancestral graph
            ps = [p for p in self.par[b] if p in keep]
            for p in ps: und[p].add(b); und[b].add(p)
            for p, q in itertools.combinations(ps, 2): und[p].add(q); und[q].add(p)
        seen, stack = set(), [x for x in X if x not in Z]
        while stack:
            n = stack.pop()
            if n in seen or n in Z: continue
            seen.add(n); stack.extend(und[n] - seen)
        return not (seen & Y)

    # ---- paths (for teaching: which paths are open or closed, and why)
    def paths(self, x, y, maxlen=9, cap=4000):
        out = []; nb = {n: self.par[n] | self.ch[n] for n in self.nodes}
        def dfs(p):
            if len(out) >= cap: return
            if p[-1] == y: out.append(list(p)); return
            if len(p) >= maxlen: return
            for m in sorted(nb[p[-1]]):
                if m not in p: p.append(m); dfs(p); p.pop()
        dfs([x]); return out

    def path_status(self, p, Z):
        Z = set(Z); dz = self.desc(Z, incl=True) if Z else set(); blockers = []
        for i in range(1, len(p) - 1):
            a, v, b = p[i - 1], p[i], p[i + 1]; coll = (a in self.par[v]) and (b in self.par[v])
            if coll:
                if v not in dz and not (self.desc([v]) & Z): blockers.append(f"{v} is a collider and neither it nor a descendant is conditioned")
            elif v in Z: blockers.append(f"{v} is conditioned (non-collider)")
        kind = "causal" if all(p[i + 1] in self.ch[p[i]] for i in range(len(p) - 1)) else ("backdoor" if p[1] in self.par[p[0]] else "other")
        return dict(path=p, kind=kind, open=not blockers, blockers=blockers)

    # ---- criteria
    def adjustable(self, x, cond=()):
        return [n for n, v in self.nodes.items() if v.get("kind") in ("observed", "derived", "new_data_acquired") and n != x and n not in set(cond)]

    def backdoor_valid(self, x, y, Z, cond=()):
        Z, cond = set(Z), set(cond); bad = (Z | cond) & self.desc([x])
        if bad: return False, f"descendant of the treatment is conditioned on: {sorted(bad)}"
        ok = self.sub(remove_out_of=[x]).dsep([x], [y], Z | cond)
        return ok, "all back-door paths are blocked" if ok else "an open back-door path remains"

    def adjustment_sets(self, x, y, cand, cond=(), maxsize=3, limit=12):
        cand = [c for c in cand if c not in self.desc([x]) and c not in (x, y)]; found = []
        for k in range(maxsize + 1):
            for S in itertools.combinations(sorted(cand), k):
                if any(set(f) <= set(S) for f in found): continue
                if self.backdoor_valid(x, y, S, cond)[0]: found.append(S)
                if len(found) >= limit: return found
        return found

    def frontdoor_valid(self, x, y, M, cond=()):
        M = set(M)
        if self.desc([x]) & set(cond): return False, "a descendant of X is conditioned on"
        g = DAG({n: v for n, v in self.nodes.items() if n not in M}, [(a, b) for a, b in self.edges if a not in M and b not in M])
        if y in g.desc([x]): return False, "some directed path from X to Y avoids M"
        if not self.sub(remove_out_of=[x]).dsep([x], M, cond): return False, "an open back-door path from X to M"
        if not self.sub(remove_out_of=M).dsep(M, [y], set([x]) | set(cond)): return False, "a back-door path from M to Y is not blocked by X"
        return True, "front-door criterion satisfied"

    def frontdoor_sets(self, x, y, cand, cond=(), maxsize=2):
        out = []
        for k in range(1, maxsize + 1):
            for S in itertools.combinations(sorted(c for c in cand if c not in (x, y)), k):
                if any(set(f) <= set(S) for f in out): continue
                if self.frontdoor_valid(x, y, S, cond)[0]: out.append(S)
        return out

    # ---- do-calculus: each returns (rule applies?, graph description)
    def rule1(self, Y, X, Z, W): return self.sub(remove_into=X).dsep(Y, Z, set(X) | set(W))
    def rule2(self, Y, X, Z, W): return self.sub(remove_into=X, remove_out_of=Z).dsep(Y, Z, set(X) | set(W))
    def rule3(self, Y, X, Z, W):
        gx = self.sub(remove_into=X); zw = set(Z) - gx.anc(W)
        return self.sub(remove_into=set(X) | zw).dsep(Y, Z, set(X) | set(W))

    def identify(self, x, y, cond=(), maxsize=3):
        adj = self.adjustment_sets(x, y, self.adjustable(x, cond), cond, maxsize)
        fd = self.frontdoor_sets(x, y, self.adjustable(x, cond), cond)
        return dict(backdoor=[list(s) for s in adj], frontdoor=[list(s) for s in fd], identified=bool(adj or fd))


def load(path):
    d = yaml.safe_load(open(path)); nodes = d["nodes"]
    return DAG(nodes, [(e["from"], e["to"]) for e in d["edges"]]), d


def library():
    base, meta = load(ROOT / "lexicon/dag.yaml"); out = [dict(id="base", name="Base (draft)", note="the graph drawn in the plan: weather acts through ripeness; critics may select on quality", nodes=base.nodes, edges=base.edges, conditioned=["selection"])]
    for f in sorted((ROOT / "lexicon/dags").glob("*.yaml")):
        d = yaml.safe_load(open(f)); b = json.loads(json.dumps(out[0]))
        nodes = {**b["nodes"], **d.get("add_nodes", {})}; rm = {tuple(e) for e in d.get("remove_edges", [])}
        edges = [list(e) for e in b["edges"] if tuple(e) not in rm] + [[e["from"], e["to"]] for e in d.get("add_edges", [])]
        out.append(dict(id=f.stem, name=d["name"], note=d["note"], nodes=nodes, edges=edges, conditioned=d.get("conditioned", ["selection"])))
    return out


def fixture(n=400, seed=5):
    rnd = random.Random(seed); out = []
    for lib in library():
        g = DAG(lib["nodes"], lib["edges"]); names = sorted(g.nodes)
        for _ in range(n // 3):
            x, y = rnd.sample(names, 2); rest = [c for c in names if c not in (x, y)]; Z = rnd.sample(rest, rnd.randint(0, 4)); W = rnd.sample([c for c in rest if c not in Z], rnd.randint(0, 2))
            out.append(dict(dag=lib["id"], x=x, y=y, Z=Z, W=W, dsep=g.dsep([x], [y], Z), backdoor=g.backdoor_valid(x, y, Z, lib["conditioned"])[0], r1=g.rule1([y], [x], Z[:1], W), r2=g.rule2([y], [x], Z[:1], W), r3=g.rule3([y], [x], Z[:1], W)))
        e1 = g.identify("weather_anomaly", "note_sat", lib["conditioned"]); out.append(dict(dag=lib["id"], identify=e1, x="weather_anomaly", y="note_sat"))
    json.dump(out, open(ROOT / "outputs/causal_parity.json", "w")); return len(out)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "export"
    if cmd == "fixture": print("fixture rows", fixture())
    elif cmd == "export":
        L = library()
        for l in L:
            g = DAG(l["nodes"], l["edges"]); print(f"{l['id']:28s} E1 (weather->note, conditioned on {l['conditioned']}):", g.identify("weather_anomaly", "note_sat", l["conditioned"]))
        json.dump(L, open(ROOT / "dashboard/data/dags.json", "w"), separators=(",", ":"))
