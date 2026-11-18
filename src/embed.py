"""Phase 4a: deterministic embeddings of review text. Usage: embed.py <minilm|bert> <N|all>
Models pinned by revision; CPU; eval mode; no sampling."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).parent))
from wsetkit import ROOT

MODELS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
    "bert": ("google-bert/bert-base-uncased", "86b5e0934494bd15c9632b12f734a8a67f723594"),
}
torch.manual_seed(0)
torch.set_num_threads(4)


def load(name):
    from transformers import AutoTokenizer, AutoModel
    repo, rev = MODELS[name]
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    mdl = AutoModel.from_pretrained(repo, revision=rev).eval()
    return tok, mdl


@torch.no_grad()
def embed_texts(texts, tok, mdl, bs=64, maxlen=128, normalize=True):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=maxlen, return_tensors="pt")
        h = mdl(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1)                     # mean pooling
        if normalize:
            v = torch.nn.functional.normalize(v, dim=1)
        out.append(v.numpy().astype("float32"))
    return np.vstack(out)


def main():
    name, n = sys.argv[1], sys.argv[2]
    vec = pd.read_parquet(ROOT / f"outputs/wset_vectors_{n}.parquet", columns=["id"])
    raw = pd.read_parquet(ROOT.parent / "datasets/wine_reviews_deduped.parquet", columns=["description"])
    from wsetkit import review_id
    raw["id"] = raw.description.map(review_id)
    raw = raw.drop_duplicates("id").set_index("id")
    texts = raw.loc[vec.id, "description"].tolist()
    tok, mdl = load(name)
    t0 = time.time()
    # stable order: sort by length within chunks would change batching numerics slightly -> keep file order
    E = embed_texts(texts, tok, mdl)
    np.save(ROOT / f"outputs/emb_{name}_{n}.npy", E)
    print(name, E.shape, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
