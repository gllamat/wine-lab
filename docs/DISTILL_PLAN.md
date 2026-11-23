# Distilling Jev's WSET answers into a local model (RTX 5060 Ti 16 GB)

## 1. What we are distilling (from the existing code/outputs)
- Teacher: TypeSafe `jev-1.14.0` (System One), called by `src/jev_pilot.py`. One request per review, **231 typed answers** mapped to the WSET L4 SAT card:
  - 216 `noul` (presence probability 0-1): each Lexicon descriptor asked separately for nose and palate, plus tannin nature, texture, appearance extras, "stated__*".
  - 9 `score` (ordinal, 6 sweetness-type levels): sweetness, acidity, tannin, alcohol, body, flavour intensity, finish, nose intensity, quality.
  - 6 `choice` (with probabilities): clarity, colour depth, colour, condition, ripeness, ageing.
- Input is short: Wine Enthusiast notes, mean 40 words (~60 tokens), max 135 words.
- Already labelled: `outputs/jev_1000.jsonl` (1000), `outputs/jev_pilot.jsonl` (500). Cost measured: ~4,000 input tokens/review, **$0.17 per 1,000 reviews, ~$28.6 for all 169,428**. Labelling the whole corpus is cheap, so label budget is not the bottleneck.
- Baseline to beat: the deterministic matcher (`src/wsetkit.py`) and the existing guess model.

**Conclusion: this is multi-label / multi-head classification + ordinal regression, not text generation.** A small encoder fits it far better than a small LLM.

## 2. Model choice

| Option | Fit on 16 GB | Verdict |
|---|---|---|
| **DeBERTa-v3-base (184M) or ModernBERT-base (149M), 231-output heads** | Trivial (~4 GB at bs 64, len 128, bf16) | **Recommended.** Trains in ~20-40 min, inference thousands of notes/s, exportable to ONNX/int8 for the browser dashboard. |
| MiniLM / bge-small (33M) | Trivial | Fast fallback / browser version; expect a few points lower. |
| Qwen3-4B / Llama-3.2-3B + QLoRA emitting JSON | Fits (QLoRA 4-bit) | Not recommended for this task: 231 fields as text is slow, probabilities are lost, parse failures. Only worth it if you later want free-text WSET-style explanations. |

Heads: 216 sigmoid outputs trained on Jev's soft probabilities (BCE with soft targets, or KL); ordinal scores as classification over levels (or CORAL); choices as softmax with KL to Jev's probability vector. Shared encoder, mean-pool + small MLP.

## 3. Plan

**Phase 0 - Check the rules (before spending anything)**
- Read TypeSafe's terms on using outputs to train a model; get it in writing if unclear.
- Licence stays CC BY-NC-SA / non-commercial (review text and everything derived). WSET Lexicon/SAT text is not redistributed. The model card must say so. Add to `DATA_LICENSES.md`.

**Phase 1 - Teacher labels (1-2 days elapsed, ~$30)**
1. Run `jev_pilot.py` over the full corpus (`--out outputs/jev_full.jsonl`; script is resumable). Keep `.env` key handling as is.
2. Pin the model (`jev-1.14.0`, not `jev-latest`) so labels are consistent; log usage.
3. Sanity-check label sparsity: many `noul` descriptors will be ~0 for >99% of notes. Record prevalence per question; questions with <50 positives get dropped from the headline metrics (still trained, but flagged).

**Phase 2 - Splits and gold set**
- Split by winery (not random) to avoid leakage via repeated producers/styles: 90/5/5 train/val/test.
- Hand-check a gold set of ~200 notes (stratified, include odd ones) to measure Jev's own error. The student can only match the teacher; the gold set shows what the teacher itself is worth.
- Add out-of-distribution notes if the target use is people's own tasting notes (WSET exam style, not critic prose): write/collect ~500-2,000 unlabeled ones, label with Jev, include in training and keep a held-out part. Otherwise the model will only be good at Wine Enthusiast prose.

**Phase 3 - Baselines (half a day)**
- TF-IDF + logistic regression per head (strong cheap baseline for sparse lexical labels).
- Existing deterministic matcher outputs vs Jev (already in `compare_jev.py`).
- Frozen MiniLM embeddings (already computed in `outputs/emb_minilm_all.npy`) + linear heads.

**Phase 4 - Fine-tune on the 5060 Ti**
- Env: PyTorch built for CUDA 12.8+ (Blackwell sm_120 needs it), transformers, bf16, `torch.compile` optional. Verify with a 1-minute smoke run before the full one.
- Settings: max_len 160, bs 64, lr 3e-5 encoder / 1e-3 heads, 3-4 epochs, cosine, early stop on val mean AUC. Expect ~20-40 min for DeBERTa-base on 150k notes.
- Loss weighting: rare descriptors need pos-weight or focal loss; ordinal and choice heads weighted so 216 binary heads do not drown the 15 structural ones.
- Run 3 seeds; report mean +/- sd.

**Phase 5 - Evaluate (against Jev as teacher, then against gold)**
- noul heads: per-question AUROC, average precision, F1 at tuned threshold, Brier/ECE. Aggregate by cluster (floral, oak, ...).
- score heads: exact-level accuracy, within-one-level accuracy, MAE, quadratic-weighted kappa.
- choice heads: accuracy, KL to Jev.
- Pass bar (suggested): macro-AUROC >= 0.95 on questions with enough positives, scores within-one >= 95%, and student agrees with gold at least as well as Jev does minus ~2 pts. Compare student-vs-gold and Jev-vs-gold directly.
- Then the downstream test that matters: feed the student's vectors into the existing guess model (grape/country) and check the headline numbers (58% grape, 78% country) do not drop.

**Phase 6 - Package**
- Export: safetensors + ONNX; int8 variant for the browser dashboard (static site, like the existing simple model); calibrated thresholds in JSON.
- Add a parity test vs PyTorch, matching the repo's existing Python/JS parity convention.
- Model card (non-commercial, unofficial WSET interpretation, teacher = Jev 1.14.0, training data = Wine Enthusiast via Kaggle).
- Move to the PC: copy `jev_full.jsonl`, review text (parquet), and the training script; Mac only needs inference.

## 4. Risks
- **Teacher noise is the ceiling.** Rare descriptors and "not stated" boundaries are subjective; show per-question reliability instead of one headline number.
- **Domain shift** (critic prose vs exam notes) - see Phase 2.
- **Terms/licence** of teacher outputs - see Phase 0. This is the one that could block the project.
- **Calibration**: soft-label training helps, but fit a temperature on val.

## 5. Effort
About 3-5 working days end to end: 1 day labelling and QA, 1 day baselines and gold set, 1 day training/tuning, 1 day eval and packaging. Cost: ~$30 in API plus electricity.

## 6. Decisions needed
1. Target input: critic-style notes only, or users' own WSET-style notes too?
2. Is the 5060 Ti on a Windows or Linux box (affects env setup)?
3. Is a browser (ONNX int8) build wanted, or only local Python use?
