# Instructions for the AI agent installing this on the Windows PC

You are setting up and running a small machine-learning project on a **Windows PC with an NVIDIA RTX 5060 Ti (16 GB)**. Do everything inside this folder. Read this file fully first.

## 1. What this project is
Critic tasting notes (Wine Enthusiast) go in; a WSET Level 4 "Systematic Approach to Tasting" card comes out (colour, sweetness, acidity, tannin, body, finish, quality, ageing, and 216 aroma/flavour descriptors for nose and palate).
A large model ("Jev", an external API) already answered these 231 questions for ~1,500 notes. The notebook trains a small model (`roberta-base`, 125M parameters) to **copy Jev's answers** (knowledge distillation).
This package is only the **training/inference** side. It needs **no API key and no Jev access**. Everything it needs is in `pack\`.

Files:
| File | Purpose |
|---|---|
| `wset_distill.ipynb` | The training notebook (didactic; runs end to end) |
| `run_headless.py` | Runs the notebook without a browser and saves the executed copy under `runs\` |
| `check_gpu.py` | Verifies PyTorch can use the GPU |
| `setup_windows.bat`, `run_notebook.bat`, `run_app.bat` | Double-click helpers for the human (they contain `pause`, so as an agent use the commands below instead) |
| `app.py` | Stand-alone web page to try the trained model |
| `pack\wset_distill_pack.parquet`, `pack\wset_distill_schema.json` | Notes + Jev's answers |
| `requirements.txt` | Python packages (PyTorch is installed separately, step 3) |

## 2. Rules
- **Privacy / licence.** `pack\` contains review text (CC BY-NC-SA 4.0). Keep it on this machine. Do not upload it, email it, or put it in a repository or any online service. Non-commercial use only.
- Install only into the `.venv` created inside this folder. Do not change system Python, the NVIDIA driver or system settings without asking the user.
- Do not edit `pack\` files. Do not delete anything outside this folder.
- If a step needs the user (installing Python, a driver update needing admin/reboot), stop and ask them with the exact action needed.

## 3. Install (PowerShell, from the unzipped folder, e.g. `C:\wset_distill\wset_distill_pc`)
```powershell
# 3.0 prerequisites: Python 3.10-3.13 (3.12 recommended) and an NVIDIA driver >= 570
py -3 --version            # if "py" is missing try: python --version
nvidia-smi                 # must list the RTX 5060 Ti; "CUDA Version" should be 12.8 or higher

# 3.1 private environment
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# 3.2 PyTorch built for CUDA 12.8 (REQUIRED for RTX 50-series; the default PyPI build will not work on this card). ~3 GB download
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128

# 3.3 the rest (includes plotly, used for the interactive training plots)
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3.4 verify the GPU
.\.venv\Scripts\python.exe check_gpu.py
```
**Success for 3.4:** the output ends with `test calculation on the GPU worked. The GPU is ready.` and shows `GPU: NVIDIA GeForce RTX 5060 Ti`, `bf16 supported: True`.
Always call `.\.venv\Scripts\python.exe` explicitly (no need to "activate" the venv, which avoids execution-policy problems).

## 4. Smoke test (about 2-5 minutes; the first run downloads `roberta-base`, ~500 MB, so internet access to huggingface.co is needed)
```powershell
.\.venv\Scripts\python.exe run_headless.py --max-rows 500 --epochs 1 --out runs\smoke.ipynb
```
Success: prints `done: runs\smoke.ipynb`, no traceback, and `pack\wset_student.pt` now exists.
Open `runs\smoke.ipynb` (e.g. `.\.venv\Scripts\python.exe -c "import json;nb=json.load(open('runs/smoke.ipynb'));[print(''.join(o.get('text','')) or o.get('data',{}).get('text/plain','')) for c in nb['cells'] for o in c.get('outputs',[])]"`) and confirm:
- the hardware cell printed `device: cuda` and `mixed precision: torch.bfloat16`;
- the training cell printed `epoch 1/1 | train loss ... | validation loss ...`;
- the card example near the end printed a "Structure and levels" list.
Metrics from 500 notes are meaningless (most descriptors have too few positives); the smoke test only proves the pipeline works on the GPU.

## 5. Real run
```powershell
.\.venv\Scripts\python.exe run_headless.py --batch-size 64 --epochs 4 --out runs\full.ipynb
```
It uses all rows in the pack (about 1,500 notes today, so it takes only a few minutes; training will be undertrained until the pack contains many more notes, see section 9).
Outputs: `runs\full.ipynb` (all tables, plots and printed metrics), `pack\wset_student.pt` (the model), `pack\best_student.pt`, `pack\checkpoint.pt`.
Checkpoints are tied to the settings, so a checkpoint left by the smoke test is ignored automatically (the log says "found a checkpoint from a different run ... starting fresh"). If you re-run with the SAME settings after a crash it resumes.
If you hit CUDA out-of-memory: lower `--batch-size` to 32 then 16. Other options: `--model NAME` (e.g. `sentence-transformers/all-MiniLM-L6-v2`, much smaller), `--epochs N`, `--max-rows N`.

### 5-images. Plots that survive re-opening (notebook v6)
A notebook executed headless is "untrusted" when opened later (or on another computer), so interactive plots and Mermaid diagrams show up EMPTY. The notebook therefore now also stores a **PNG picture of every plot inside the notebook file** (and Mermaid diagrams as pictures from mermaid.ink, loaded when the notebook is viewed). This needs:
```powershell
.\.venv\Scripts\python.exe -m pip install kaleido nbconvert      # (both are in requirements.txt now)
.\.venv\Scripts\python.exe -c "import plotly.graph_objects as go; go.Figure(go.Bar(x=[1],y=[1])).to_image(format='png'); print('static images OK')"
```
kaleido drives Chrome. If the test fails with a message about Chrome, install Google Chrome, or run `.\.venv\Scripts\plotly_get_chrome -y` (downloads about 150 MB: tell the user first). If it cannot be made to work, run with `--no-images` and say so in the report (plots will then only be in `pack\figures\*.html`).
After a run you get, besides `runs\NAME.ipynb`: `runs\NAME.html` (the whole run as one web page; opens in any browser, needs internet for plotly.js/mermaid.js), `pack\figures\*.png|html|json` (each figure as picture, page and plotly JSON) and `pack\run_state.json` (the training record: losses, metrics, per epoch, to redraw plots).
Report the number of PNG pictures embedded (the notebook prints a warning "static images skipped" if it could not).

### 5a. New in notebook v4: section 7b compares three ways to adapt the body
After the main training run the notebook trains **two more models** with the same data and epochs: a *frozen body* (only the heads learn) and *LoRA* (tiny adapters), then plots/tabulates them against the main run (full fine-tuning).
That adds about **1.3x the time of the main run** (frozen is fast, LoRA is slightly faster than full). If time is short use `--no-compare` (e.g. for the smoke test): `.\.venv\Scripts\python.exe run_headless.py --max-rows 500 --epochs 1 --no-compare --out runs\smoke.ipynb`.
The comparison is saved to `pack\strategy_comparison.json`; if the run is repeated with the same settings it is reused instead of retrained. Delete that file to retrain.
Mermaid diagrams are drawn by loading mermaid.js from a CDN (internet needed when the notebook is viewed); they are also saved as HTML in `pack\figures\`.

### 5c. New in notebook v8: section 7c compares the fine-tuned student with classic methods (no GPU needed)
Section 7c builds, on the same split, (a) the project's original **deterministic rule system** (loaded from the `rules_pkg\\` folder next to the notebook; needs `pyyaml`), (b) **improved rules** mined from the labelled training notes, and (c) a **random forest** on word counts; section 8 then scores all five methods (plus the word-counting baseline) on the test notes.
It uses the CPU only: about 1-3 minutes for the rules and the learning curve, plus the forest (RF_TREES=150: roughly 15-60 s and about 1.5 GB of RAM with 28,000 training notes; lower `RF_TREES` if memory is short). Settings: `RULE_MIN_SUPPORT`, `RULE_MIN_PRECISION`, `RF_TREES`, `RF_FEATURES`, `RF_MIN_LEAF`, `RF_SAMPLE`.
Extra files written to `pack\\`: `rules_v1.csv` (every mined rule: question, phrase, support, precision), `lexicon_candidates_for_review.csv` (strong phrases the original lexicon lacks, for a WSET-qualified taster), and figures `07d`-`07g`, `08b`, `08c`.
If `rules_pkg\\` is missing the notebook still runs: the original rules are skipped and the mined rules and forest are built anyway.
Add to the report: the table printed at the end of section 8 (mean AUC, level exact / within one, train seconds for all five methods) and the forest's memory use if you saw a problem.

### 5b. What the (updated) notebook produces
The notebook is now didactic: it explains the train/validation/test split, what a "head" is, and draws **interactive plotly figures**. After a run, besides `runs\*.ipynb` (figures are embedded and need internet to render, they load plotly.js from a CDN), these stand-alone files exist and can be opened by double-click in any browser:
about 25 files: `00_*` (data, Parquet, tokens), `01_split`, `02*` (body, heads, encoders, attention), `03_loss`, `03b_weight_change`, `04_metrics`, `05_test_by_family`, `06_head_similarity`, `07*` (strategy comparison).
`03_loss.html` = loss per step and per epoch; `04_metrics.html` = quality metrics (mean AUC, level accuracy, AUC by aroma family) after every epoch, epoch 0 = untrained.
If `pip install -r requirements.txt` was run before this update, run `.\.venv\Scripts\python.exe -m pip install plotly` once.
Split: 80% / 10% / 10% of WINERIES for train / validation / test (settings `VAL_FRAC`, `TEST_FRAC`).

## 6. Check the trained model works as an app
```powershell
.\.venv\Scripts\python.exe -c "import app; print(app.wset_card('Ripe black cherry, vanilla and cedar. Full-bodied with firm tannins and a long finish.'))"
```
Success: prints a "Structure and levels" list and "Nose/Palate descriptors" sections. To start the web page for the human: `.\.venv\Scripts\python.exe app.py` (opens a local address like http://127.0.0.1:7860; it keeps running until stopped, so only start it if asked).
The human can also use `run_notebook.bat` (Jupyter) and `run_app.bat`.

## 7. Troubleshooting
| Symptom | Cause / fix |
|---|---|
| `check_gpu.py` says NO GPU FOUND | CPU-only torch was installed: `pip uninstall torch -y`, repeat step 3.2. Otherwise the driver is too old: ask the user to update NVIDIA driver to >= 570 (reboot) |
| `no kernel image is available for execution on the device` | torch is not a CUDA 12.8+ build; redo step 3.2 |
| `pip` cannot find `torch` on the index | the Python version is too new/old for the wheels; use Python 3.12 |
| `CUDA out of memory` | `--batch-size 32` (or 16) |
| Hugging Face download fails / times out | check internet/proxy; retry; the model downloads once and is then cached |
| `py` not recognised | use `python` instead; if neither exists, ask the user to install Python 3.12 from python.org with "Add python.exe to PATH" |
| Path-too-long errors | move the folder to a short path like `C:\wset_distill` |
| Notebook cell error | the executed copy in `runs\` contains the traceback in the failing cell; report the cell and the last 20 lines |

## 8. Report back to the user
- GPU line from `check_gpu.py`, torch version, Python version.
- Smoke test: pass/fail, time taken.
- Real run: time per epoch, final validation loss, mean AUC (student vs baseline), the level-question table (`exact` / `within_one_level`), and any warnings.
- Section 7b: the comparison table (numbers trained, seconds per epoch, peak GPU memory, best validation AUC) for full / frozen / LoRA.
- The per-epoch table (epoch, train loss, validation loss, mean AUC, level exact, level within one) printed near the end of section 7 of the notebook.
- Paths of `runs\full.ipynb`, `pack\figures\` and `pack\wset_student.pt`.
- Anything you changed or could not do.

## 9. For context (not for you to do now)
The data pack has only ~1,500 labelled notes. For a useful model the human will label 20-30k+ more notes with Jev on their Mac (about $0.17 per 1,000 notes) and send a bigger `pack\` (two files). When they do: replace the two files in `pack\`, then repeat sections 5-6. No code change is needed.
