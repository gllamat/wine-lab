# Distilling Jev's WSET answers into a small model

Teaches a small model (RoBERTa by default) to answer the 231 WSET-card questions for a critic's tasting note the way Jev (TypeSafe `jev-1.14.0`) does.
Plan and reasoning: [../docs/DISTILL_PLAN.md](../docs/DISTILL_PLAN.md).

| File | What |
|---|---|
| `make_data_pack.py` | Joins critic notes (`../../datasets/`) with Jev answers (`../outputs/jev_*.jsonl`) -> `pack/wset_distill_pack.parquet` + `pack/wset_distill_schema.json` |
| `wset_distill.ipynb` | The didactic notebook: data -> split -> model -> loss -> training -> evaluation -> web interface. Same cells on Colab and locally |
| `build_notebook.py` | Source of the notebook (edit here, then `python distill/build_notebook.py`) |

## Run it on Google Colab (free tier)
1. Make the pack: `python distill/make_data_pack.py` (needs `pandas`, `pyarrow`). With no arguments it uses the 1,500 notes labelled so far.
2. In Google Drive create the shared drive folder `Lamat/AI_projects/Wine-lab/` and upload the two files from `pack/`.
3. Upload `wset_distill.ipynb` to Colab (File > Upload notebook).
4. `Runtime > Change runtime type`: **T4 GPU** (reliable) or **v5e-1 TPU** (experimental: also set `USE_TPU = True` in the settings cell).
5. `Runtime > Run all`. Allow Drive access when asked. The last cell prints a public link to the interface.

Free sessions can disconnect: a checkpoint is saved to Drive after every epoch and the notebook resumes by itself when you run it again.

## Quick test first
Set `MAX_ROWS = 500` in the settings cell for a 1-minute run that checks everything works. With only 1,500 labelled notes the model is **undertrained**
and the scores are not meaningful (most questions have too few positives); you need roughly 20-30k labelled notes, ideally all 169k.

## Getting more labels (this spends API credit, about $0.17 per 1,000 notes, about $29 for everything)
`src/jev_pilot.py` only samples from the 20,000-note set unless you give it an id list, so write one first (all notes minus the ones already labelled):
```bash
python - <<'EOF'
import json, re, hashlib, pandas as pd
rid = lambda d: hashlib.md5(re.sub(r"\s+", " ", str(d).lower().strip()).encode()).hexdigest()[:12]
ids = pd.read_parquet("../datasets/wine_reviews_deduped.parquet", columns=["description"]).description.map(rid).drop_duplicates()
ids.sample(frac=1, random_state=1).to_frame("id").to_csv("outputs/jev_ids_all.csv", index=False)
EOF
python src/jev_pilot.py 30000 --ids-from outputs/jev_ids_all.csv --out outputs/jev_full.jsonl --model jev-1.14.0
python distill/make_data_pack.py outputs/jev_full.jsonl outputs/jev_1000.jsonl outputs/jev_pilot.jsonl
```
The labelling script is resumable (it skips ids already in `--out`) and the id list is shuffled, so you can start with 30,000 and raise the number later. Run it from `wset_pipeline/`.

## Notes
- The pack contains review text: keep it private. Everything is non-commercial (CC BY-NC-SA 4.0). Check TypeSafe's terms on training from their outputs before the large labelling run.
- Model choice: `roberta-base` is the robust default. `microsoft/deberta-v3-base` is slightly better but fussier on T4 (fp16) and TPU; `sentence-transformers/all-MiniLM-L6-v2` is 5x smaller.
- The TPU path uses Google's TorchTPU (`torch_tpu`, Python 3.11+), since PyTorch/XLA is being archived. It has not been run on a real TPU yet.

## The notebook, explained
See [NOTEBOOK_GUIDE.md](NOTEBOOK_GUIDE.md): what a head is, which notes are used for training / validation / test, and how to read the interactive loss and metrics plots.
