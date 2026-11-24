"""Generates distill/wset_distill.ipynb. Edit the cell text below, then run:  python distill/build_notebook.py
(The notebook itself is the product; this script just keeps its source readable and diff-friendly.)"""
import nbformat as nbf
from pathlib import Path

cells = []
def md(t):   cells.append(nbf.v4.new_markdown_cell(t.strip("\n")))
def code(t): cells.append(nbf.v4.new_code_cell(t.strip("\n")))

# --------------------------------------------------------------------------------------------------------------------
md("""
# From critic tasting notes to a WSET Systematic Approach card, with a small model

**The idea.** A big model (*Jev*) can read a critic's note, e.g. *"Ripe black cherry and a hint of vanilla, firm tannins, long finish"*, and answer
231 WSET questions about it: *is there black cherry on the nose? how high is the tannin? how long is the finish?*
Asking Jev costs money and needs the internet. We have its answers for many notes, so we can **teach a small model to give the same answers**.
This is called **distillation**: *teacher* (Jev) answers, *student* (our small model) learns to copy.

**What you will do in this notebook**
1. Load the critics' notes and Jev's answers (one file you upload to Google Drive).
2. Look at what the answers look like and how rare most of them are.
3. Split the data honestly, and *see* which notes are used for what (train / validation / test).
4. Build a small student: a pre-trained language model (the **body**) plus two answer **heads**, and open a head to see what it really is.
5. Set a yardstick: a simple word-counting baseline.
6. Train it on a free Colab **T4 GPU** or **v5e-1 TPU** (or your own GPU), watching the loss and the quality metrics **after every epoch** in interactive plots.
7. Measure how well it copies Jev on the untouched test notes.
8. Try it on any note you type.

### The big picture

```
 critic's note ──► tokenizer ──► ENCODER ("body", 125M numbers, pre-trained) ──► one vector for the note (768 numbers)
                                                                                      │
                                                          ┌───────────────────────────┴──────────────────────────┐
                                                          ▼                                                      ▼
                                          YES/NO HEAD  (768 → 216)                              LEVEL HEAD  (768 → one score per level)
                                    "is vanilla on the nose? is there                          "how high is the tannin? what colour?
                                     black cherry on the palate? ..."                           how long is the finish? ..."
```

### Words you will meet (read once, come back when stuck)

| Word | Plain meaning |
|---|---|
| **Teacher / student** | Jev (big, paid, online) answers; our small model learns to give the same answers |
| **Encoder / body** | the part that *reads* the text and summarises it as numbers |
| **Head** | a small layer on top of the body that turns the summary into the *answers to one kind of question* (explained properly in section 5) |
| **Epoch** | one full pass over all the training notes |
| **Batch / step** | a handful of notes (e.g. 32) processed together; after each batch the weights are nudged once: that nudge is a *step* |
| **Loss** | one number: how far the student's answers are from Jev's. Training = pushing it down |
| **Train / validation / test** | notes the model *learns from* / notes used to *monitor and tune* (never learned from) / notes kept sealed until the final exam |
| **AUC** | quality score for a yes/no question: 1.0 = perfect ranking, 0.5 = guessing |
| **Overfitting** | the model memorises the training notes: training loss keeps falling, validation loss stops improving or rises |
| **Activation function** | a fixed formula applied to every number between two matrix products (sigmoid, GELU, softmax). It has no weights; without it, stacked layers would collapse into one matrix (section 6b) |
| **Gradient / back-propagation** | for each trainable number: how much the loss would change if it moved a little. Back-propagation computes all of them by running backwards from the loss (section 6b) |
| **Rule / lexicon** | a fixed phrase-to-answer instruction ("`vanilla` -> yes for the vanilla question"). A *deterministic* rule system has only such rules and learns nothing at run time (section 7c) |
| **Support / precision** | for a phrase and a question: in how many notes the phrase and a Jev "yes" occur together / how often Jev says yes when the phrase is there. Used to choose which phrases become rules |
| **Random forest** | many decision trees (flow charts of yes/no questions about words) voting together (section 7c) |
| **Logit** | the raw score a head outputs before the sigmoid or softmax turns it into a probability |

> **Licence reminder.** The notes are Wine Enthusiast reviews (CC BY-NC-SA 4.0), so everything here is **non-commercial**.
> This project is not affiliated with WSET; the questions are an unofficial reading of the WSET Level 4 tasting card.
""")

# --------------------------------------------------------------------------------------------------------------------
md("""
## 0. The problem, in plain words: what is *distillation*?

**The situation.** We want a machine that reads a critic's tasting note and fills in a WSET-style tasting card: *which aromas, how much tannin, how long a finish?*
A very large language model (Jev) can do it, but:

| | Jev (teacher) | Our small student |
|---|---|---|
| Where it runs | on someone else's servers, over the internet | on your own computer |
| Cost | about 9,000 input tokens per note: **~$0.39 per 1,000 notes, ~$65 for all 169,428 notes**, every time | free once trained |
| Speed | a network call per note | thousands of notes per minute on a GPU |
| Privacy / availability | the text leaves your machine; needs a key | fully offline |

**The idea (distillation).** Pay the teacher once to answer a few thousand notes, keep its answers, and **train a small model to give the same answers.**
The student never needs to be as clever as the teacher in general; it only has to imitate it on *this one task*.

```
 STEP 1 (done, costs money)        STEP 2 (this notebook, free)               STEP 3 (forever, free)
 note ──► Jev ──► 231 answers      note ──► student ──► 231 guesses           new note ──► student ──► WSET card
                    │                                      │
                    └──── compare ◄────────────────────────┘   the gap is the "loss"; training shrinks it
```

**Hard labels versus soft labels.** If Jev says *"vanilla on the nose: 0.83"* we could round that to **yes**. But 0.83 carries more information: *Jev was fairly sure, not certain*.
Training the student on the full number (a **soft label**) lets it learn those shades. This is the heart of distillation (Hinton et al., 2015: the teacher's *probabilities* are richer than its final choices).

| Teacher says | Hard label | Soft label (what we use) |
|---|---|---|
| vanilla on the nose: 0.83 | 1 | 0.83 |
| vanilla on the nose: 0.51 | 1 | 0.51 (*"unsure"*) |
| tannin: Low 0.78, Medium(-) 0.19, Medium 0.03 | Low | the whole distribution (0.78, 0.19, 0.03, 0, 0) |

**What changes during distillation?** Only the *student's* numbers (its weights). Jev never changes; we don't even have its weights. Section 5 shows exactly which of the student's weights move, and by how much.
""")

code("""
# A tiny helper that draws Mermaid diagrams (text -> picture). It loads mermaid.js from a CDN, so it needs internet; every diagram is also saved in figures/.
import html as _html, os
from IPython.display import HTML, display

def mermaid(diagram, name=None, height=330):
    page = ("<html><body style='margin:0;background:white;font-family:sans-serif'><pre class='mermaid'>" + _html.escape(diagram) + "</pre>"
            "<script type='module'>import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs'; mermaid.initialize({startOnLoad:true, theme:'neutral'});</script></body></html>")
    display(HTML('<iframe srcdoc="' + _html.escape(page, quote=True) + '" width="100%" height="' + str(height) + '" style="border:1px solid #ddd;border-radius:6px;background:white"></iframe>'))
    fd = globals().get("FIG_DIR")
    if name and fd is not None:
        try: (fd / f"{name}.html").write_text(page)
        except Exception: pass
    st = globals().get("STATIC_IMAGES", "auto"); st = (os.environ.get("WSET_HEADLESS") == "1") if st == "auto" else bool(st)
    if st and globals().get("MERMAID_IMAGES", True):                          # a plain <img> survives in an "untrusted" re-opened notebook; the iframe above does not
        import base64
        b64 = base64.urlsafe_b64encode(diagram.encode()).decode()
        display(HTML('<img src="https://mermaid.ink/img/' + b64 + '?bgColor=white" alt="diagram" style="max-width:100%;border:1px solid #ddd;border-radius:6px">'))

mermaid('''
flowchart LR
  note["critic tasting note"] --> jev["Jev - the teacher<br/>big, paid, online"]
  jev --> ans["231 answers<br/>soft probabilities"]
  note --> stu["Student - small model<br/>on your own GPU"]
  stu --> guess["231 guesses"]
  ans --> loss{{"compare: the loss"}}
  guess --> loss
  loss -. "nudge the student weights" .-> stu
  classDef t fill:#f3e5ab,stroke:#b58900
  classDef st fill:#d9f0f0,stroke:#009999
  class jev,ans t
  class stu,guess st
''', "00a_distillation_flow", 300)
""")

md("""
## 1. Setup

**On Colab:** `Runtime > Change runtime type` and pick **T4 GPU** (safe choice) or **v5e-1 TPU** (experimental, see the switch below).
Upload the folder `pack/` (two files made by `make_data_pack.py`) to Google Drive to the shared drive folder `Lamat/AI_projects/Wine-lab/`.
""")

code("""
import os, sys, json, time, math, random, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")                      # asks you to allow access to your Drive
    PACK_DIR = Path("/content/drive/Shareddrives/Lamat/AI_projects/Wine-lab")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "gradio", "scikit-learn", "plotly"], check=False)
else:
    PACK_DIR = Path("pack")                            # next to this notebook when running on your own computer

# Interactive plots (plotly): hover for values, click legend entries to hide/show lines, drag to zoom.
# Every figure is also saved as a stand-alone .html file in  <data folder>/figures/  (open it in any browser).
import plotly.graph_objects as go, plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = "colab" if IN_COLAB else "plotly_mimetype+notebook_connected"
FIG_DIR = PACK_DIR / "figures"
try: FIG_DIR.mkdir(exist_ok=True)
except Exception: FIG_DIR = None

# Why static images? Interactive plotly output is a small program (JavaScript). A notebook that was run without a browser (run_headless.py) is "untrusted"
# when you open it later (or on another computer), so Jupyter refuses to run that program and the plot area stays EMPTY. A PNG picture is always shown.
# STATIC_IMAGES = "auto": also embed a PNG of every figure when the notebook runs headless (needs the `kaleido` package and Chrome/Edge); True/False forces it.
_static_state = {"failed": False}
def want_static():
    return (os.environ.get("WSET_HEADLESS") == "1") if STATIC_IMAGES == "auto" else bool(STATIC_IMAGES)

def show(fig, name):
    fig.update_layout(template="plotly_white", font=dict(size=13), margin=dict(l=50, r=20, t=60, b=50))
    fig.show()                                                                   # interactive version (hover, zoom)
    if FIG_DIR is not None:
        try: fig.write_html(FIG_DIR / f"{name}.html", include_plotlyjs="cdn")     # stand-alone page: opens in any browser
        except Exception: pass
        try: fig.write_json(FIG_DIR / f"{name}.json")                             # the figure itself: plotly.io.read_json(...) brings it back
        except Exception: pass
    if want_static() and not _static_state["failed"]:
        try:
            png = fig.to_image(format="png", scale=1.5)
            if FIG_DIR is not None:
                try: (FIG_DIR / f"{name}.png").write_bytes(png)
                except Exception: pass
            from IPython.display import Image as _Img, display as _display
            _display(_Img(data=png))                                              # static copy stored INSIDE the notebook file: survives re-opening
        except Exception as err:
            _static_state["failed"] = True
            print("static images skipped (install `kaleido` and make sure Chrome is installed; `plotly_get_chrome` downloads one):", repr(err)[:150])

print("Running on Colab:", IN_COLAB, "| data folder:", PACK_DIR.resolve())
""")

md("""
### Settings you can change
Everything you might want to tweak is in this one cell.
""")

code("""
# ---- the student model: any Hugging Face encoder works. Smaller = faster, bigger = a bit more accurate.
MODEL_NAME = "roberta-base"          # 125M parameters. Alternatives: "microsoft/deberta-v3-base" (a bit better, fussier on T4/TPU),
                                     # "sentence-transformers/all-MiniLM-L6-v2" (22M, very fast, a few points lower)
MAX_LEN    = 128                     # tokens per note. Notes are ~60 tokens; we pad to a FIXED length (TPUs need fixed shapes)

# ---- training
EPOCHS     = 4                       # passes over the training notes
BATCH_SIZE = 32                      # notes per step. TPU/bigger GPUs can use 64
LR_ENCODER = 3e-5                    # learning rate for the pre-trained part (small: it already knows English)
LR_HEADS   = 1e-3                    # learning rate for the new answer heads (large: they start from nothing)
BIN_WEIGHT = 5.0                     # how much the 216 yes/no questions count relative to the 15 multi-level ones (see section 6)
SEED       = 42

# ---- data split (by winery, see section 3): the rest of the wineries is the training set
VAL_FRAC   = 0.10                    # share of WINERIES kept for validation (monitoring while training)
TEST_FRAC  = 0.10                    # share of WINERIES kept sealed for the final exam
MIN_POS    = 5                       # a yes/no question is only scored in a set if it has at least this many yes AND no examples there

# ---- comparing ways to adapt the body (section 7b): full fine-tuning (the main run) vs freezing the body vs LoRA
COMPARE_STRATEGIES = True            # True = also train the two alternatives with the same data and epochs and compare (adds about 2 extra training runs)
LORA_RANK  = 8                       # LoRA: size of the tiny adapters (smaller = fewer new numbers)
LORA_ALPHA = 16                      # LoRA: how strongly the adapters are mixed in
LR_LORA    = 5e-4                    # LoRA: learning rate of the adapters

# ---- classic alternatives (section 7c): hand-written + mined rules, and a random forest
RULE_MIN_SUPPORT   = 10              # a mined phrase becomes a rule only if it appears (with a Jev "yes") in at least this many training notes...
RULE_MIN_PRECISION = 0.5             # ...and Jev says yes at least this share of the time when the phrase is present
RF_TREES    = 150                    # random forest: number of trees (more = steadier, slower, more memory: about 1.5 GB at 150 trees on 28,000 notes)
RF_FEATURES = 4000                   # random forest: how many of the most frequent words / two-word phrases it looks at
RF_MIN_LEAF = 5                      # random forest: smallest group of notes a tree may end in
RF_SAMPLE   = 0.5                    # random forest: each tree learns from this share of the training notes

# ---- plots: also store a PNG picture of every figure inside the notebook, so the plots still show when the notebook is re-opened after a headless run
STATIC_IMAGES = "auto"               # "auto" = only when run headless (run_headless.py); True = always; False = never
MERMAID_IMAGES = True                # headless only: also add a picture of each Mermaid diagram (fetched from the public mermaid.ink service when the notebook is VIEWED; the text is generic, no data)

# ---- data / hardware
MAX_ROWS   = None                   # e.g. 500 for a 1-minute test run; None = use everything in the pack
USE_TPU    = False                   # set True ONLY on a v5e-1 TPU runtime (experimental, see the hardware cell)
""")

# --------------------------------------------------------------------------------------------------------------------
md("""
### Which hardware are we on?

| Runtime | What the code does |
|---|---|
| **T4 GPU** (Colab) | 16 GB, **no bf16**, so we use *fp16 mixed precision* (+ a gradient scaler, which stops tiny numbers rounding to zero) |
| **v5e-1 TPU** (Colab) | *experimental*: uses Google's new **TorchTPU** (`pip install torch_tpu`), bf16. TPUs compile the training step for one fixed tensor shape, which is why we pad every note to `MAX_LEN` and drop the last partial batch |
| **Your own NVIDIA GPU** | bf16 if the card supports it (RTX 30/40/50 do), else fp16 |
| **CPU** | works, but only for tiny test runs |

If the TPU cell fails, the notebook tells you and falls back to the CPU: switch the runtime to T4 and run again.
""")

code("""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

device, amp_dtype, use_scaler = torch.device("cpu"), None, False

if USE_TPU:
    try:
        if IN_COLAB:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch_tpu"], check=True)
        import torch_tpu                                   # registers the "tpu" device with PyTorch
        device, amp_dtype = torch.device("tpu"), torch.bfloat16
        _ = torch.ones(2, device=device) + 1               # tiny test: does a calculation run on the TPU?
    except Exception as e:                                 # anything goes wrong -> be honest and fall back
        print("TPU setup failed, falling back to CPU. Reason:", repr(e)[:300])
        print("Tip: switch the runtime to T4 GPU and set USE_TPU = False.")
        device, amp_dtype = torch.device("cpu"), None
elif torch.cuda.is_available():
    device = torch.device("cuda")
    if torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    else:
        amp_dtype, use_scaler = torch.float16, True        # T4: fp16 + gradient scaler

print("device:", device, "| mixed precision:", amp_dtype, "| gradient scaler:", use_scaler)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0), "| memory:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
""")

# --------------------------------------------------------------------------------------------------------------------
md("""
## 2. The data: critics' notes and Jev's answers

### 2a. What Jev sends back, exactly as it arrives
For every note we send Jev the text plus **231 typed questions**; it replies with one answer per question. Three question types exist:

* `noul` (yes/no): *"Does the note describe vanilla among the aromas on the nose?"* -> one number, the probability of yes.
* `score` (an ordered scale): *"How high is the tannin?"* with levels Low ... High -> a probability for each level, plus an overall score.
* `choice` (pick one): *"Which fruit-ripeness state does the note give?"* -> a probability for each option, including `not_stated`.

Below: the request we send (one question of each type), and then the **raw JSON** Jev returned for three real notes (shortened to the interesting parts; nothing is edited).
""")

code("""
RAW_FILE = PACK_DIR / "jev_raw_examples.json"
raw_examples = json.load(open(RAW_FILE)) if RAW_FILE.exists() else None
if raw_examples is None:
    print("jev_raw_examples.json not found next to the data pack: skip this section (the rest of the notebook does not need it)")
else:
    print("THE REQUEST WE SEND TO JEV (shortened: the real one carries all 231 questions)\\n")
    print(json.dumps(raw_examples["request_example"], indent=2)[:2600], "...")
""")

code("""
if raw_examples is not None:
    ex = raw_examples["examples"][0]
    print("THE CRITIC'S NOTE:\\n ", ex["description"], f"\\n  ({ex['title'] or ex['winery']}, {ex['points']} points)\\n")
    rec = ex["record"]
    print("WHAT COMES BACK (top level):", {k: (v if k != "answers" else f"<{len(v)} answers>") for k, v in rec.items()}, "\\n")
    for k in ["nose__oak__vanilla", "level__tannin", "ripeness"]:
        print(f"answers['{k}'] =", json.dumps(rec["answers"][k]), "\\n")
    from collections import Counter
    print("answer types in this reply:", dict(Counter(a["type"] for a in rec["answers"].values())))
    u = rec["usage"]; print(f"tokens: {u['input_tokens']:,} in, {u['output_tokens']:,} out  ->  cost of this one note: ${u['input_tokens'] * 0.042 / 1e6:.4f}"
                            f"  ->  for 169,428 notes: ${u['input_tokens'] * 0.042 / 1e6 * 169428:,.0f}")
    print(f"size of the raw JSON for ONE note: {len(json.dumps(rec)) / 1024:.1f} KB (the note itself is {len(ex['description'])} characters)")
""")

md("""
### 2b. Turning the answers into a table (a Parquet file)

Ten thousand JSON replies of 16 KB each are awkward to train from. We flatten them into **one table, one row per note**:

| JSON answer | becomes | column name |
|---|---|---|
| `noul` (yes/no) | one number between 0 and 1 | `b::nose__oak__vanilla` (b = binary) |
| `score` / `choice` | a list of probabilities, one per level | `g::level__tannin` (g = group), e.g. `[0.78, 0.19, 0.03, 0.0, 0.0]` |
| the note and its details | text and numbers | `description`, `title`, `winery`, `variety`, `country`, `points` |

Each row is one review: the note, a few details (winery, grape, points), and Jev's answers in two shapes:

* **Yes/no questions** (`b::...`): a number between 0 and 1, *how sure Jev is that the note mentions it*. e.g. `b::nose__oak__vanilla`.
* **Multi-level questions** (`g::...`): a probability for *each level*. e.g. `g::level__tannin` has five numbers for *Low, Medium(-), Medium, Medium(+), High*.
  The six categorical questions (colour, clarity, condition, fruit ripeness, ageing...) work the same way.
""")

code("""
df_full = pd.read_parquet(PACK_DIR / "wset_distill_pack.parquet")
schema = json.load(open(PACK_DIR / "wset_distill_schema.json"))
df = df_full
if MAX_ROWS: df = df.sample(min(MAX_ROWS, len(df)), random_state=SEED).reset_index(drop=True)

BIN_COLS   = ["b::" + q for q in schema["binary"]]            # 216 yes/no questions
GROUPS     = schema["groups"]                                 # {"level__tannin": ["Low", "Medium(-)", ...], ...}
GROUP_COLS = ["g::" + q for q in GROUPS]
print(f"{len(df):,} notes | {len(BIN_COLS)} yes/no questions | {len(GROUP_COLS)} multi-level questions")
""")

code("""
# Proof that the table row IS the raw JSON, rearranged: rebuild the row of the first example by hand and compare with the file
if raw_examples is not None:
    ex = raw_examples["examples"][0]; ans = ex["record"]["answers"]
    stored = df_full[df_full.id == ex["id"]]
    if len(stored):
        stored = stored.iloc[0]
        ok_b = all(abs(ans[q]["noul"] - stored["b::" + q]) < 1e-9 for q in schema["binary"])
        ok_g = all(np.allclose([ans[q]["probabilities"][k] for k in ans[q]["probabilities"]], stored["g::" + q]) for q in GROUPS)
        print(f"row for note {ex['id']}: 216 yes/no numbers identical to the JSON: {ok_b} | 15 level-lists identical: {ok_g}")
        print("e.g. JSON level__tannin probabilities", ans["level__tannin"]["probabilities"], "-> column g::level__tannin =", list(np.round(stored["g::level__tannin"], 2)))
    else:
        print("this example is not in the pack (the pack was rebuilt without it)")
""")

md("""
### 2c. What *is* a Parquet file?
**Parquet** is a file format for tables, made for data analysis and machine learning. Three ideas make it different from a CSV or JSON file:

1. **Columnar.** A CSV stores row after row (note 1: all its columns, note 2: all its columns...). Parquet stores *column after column* (all 10,000 values of `b::nose__oak__vanilla`, then the next column...).
   If you only need 3 of the 235 columns, only those are read from disk.
2. **Typed and self-describing.** The file carries its own *schema* (this column is text, this one is a decimal number, this one is a list of numbers). Nothing is guessed when reading.
3. **Compressed.** Similar values sit together (a column of mostly-zero probabilities shrinks a lot), so the file is typically several times smaller than the same CSV.

```
 ┌ "PAR1"  (magic bytes)
 ├ ROW GROUP 1  (a block of rows, e.g. 10,000 notes)
 │    ├ column chunk: description      (compressed pages of text)
 │    ├ column chunk: b::nose__oak__vanilla   (compressed pages of numbers)
 │    └ ... one chunk per column
 ├ ROW GROUP 2 ...
 └ FOOTER: schema + where every chunk starts + statistics (min/max) + "PAR1"
```
""")

code("""
import pyarrow.parquet as pq
pf = pq.ParquetFile(PACK_DIR / "wset_distill_pack.parquet")
md_ = pf.metadata
print(f"file: {(PACK_DIR / 'wset_distill_pack.parquet').stat().st_size / 1e6:.1f} MB | rows: {md_.num_rows:,} | columns: {md_.num_columns} | row groups: {md_.num_row_groups} | written by: {md_.created_by}")

# the schema: name -> type, grouped by the naming convention
sch = pd.DataFrame({"column": pf.schema_arrow.names, "type": [str(t) for t in pf.schema_arrow.types]})
sch["kind"] = np.where(sch.column.str.startswith("b::"), "b:: yes/no probability", np.where(sch.column.str.startswith("g::"), "g:: list of level probabilities", "note / details"))
print("\\ncolumns by kind and type:"); print(sch.groupby(["kind", "type"]).size().rename("how many").to_frame())
sch.head(12)
""")

code("""
# What does one row look like inside? Compressed size per column chunk, the biggest ones
rg = md_.row_group(0)
sizes = pd.DataFrame({"column": [rg.column(i).path_in_schema for i in range(rg.num_columns)],
                      "compressed_KB": [rg.column(i).total_compressed_size / 1024 for i in range(rg.num_columns)],
                      "uncompressed_KB": [rg.column(i).total_uncompressed_size / 1024 for i in range(rg.num_columns)],
                      "compression": [rg.column(i).compression for i in range(rg.num_columns)]})
top = sizes.sort_values("compressed_KB", ascending=False).head(12)
fig = make_subplots(rows=1, cols=2, subplot_titles=("Same table, three file formats (size, MB)", "Biggest column chunks inside the Parquet file (KB)"), column_widths=[0.4, 0.6])
csv_mb = len(df_full.to_csv(index=False).encode()) / 1e6
json_mb = len(df_full.to_json(orient="records").encode()) / 1e6
pq_mb = (PACK_DIR / "wset_distill_pack.parquet").stat().st_size / 1e6
fig.add_trace(go.Bar(x=["JSON", "CSV", "Parquet"], y=[json_mb, csv_mb, pq_mb], text=[f"{v:.1f}" for v in (json_mb, csv_mb, pq_mb)], textposition="outside", marker_color=["#b0b0b0", "#b0b0b0", "#009999"], showlegend=False), row=1, col=1)
fig.add_trace(go.Bar(y=top.column[::-1], x=top.uncompressed_KB[::-1], orientation="h", name="uncompressed", marker_color="#b0b0b0"), row=1, col=2)
fig.add_trace(go.Bar(y=top.column[::-1], x=top.compressed_KB[::-1], orientation="h", name="compressed", marker_color="#009999"), row=1, col=2)
fig.update_layout(barmode="overlay", height=460, title="Why Parquet: smaller, and one chunk per column")
show(fig, "00b_parquet")

# columnar in action: read only what you need
t0 = time.time(); _ = pd.read_parquet(PACK_DIR / "wset_distill_pack.parquet", columns=["id", "b::nose__oak__vanilla"]); t_two = time.time() - t0
t0 = time.time(); _ = pd.read_parquet(PACK_DIR / "wset_distill_pack.parquet"); t_all = time.time() - t0
print(f"reading 2 columns: {t_two*1000:.0f} ms | reading all {md_.num_columns} columns: {t_all*1000:.0f} ms")
""")

md("""
### 2d. A first look at the table: one note next to what the teacher said about it
""")

code("""
row = df.sample(1, random_state=3).iloc[0]
print(row.description, "\\n")
print("Wine:", row.title, f"({row.points} pts)\\n")

yes = sorted(((row[c], c[3:]) for c in BIN_COLS), reverse=True)[:6]
print("Jev's strongest yes/no answers:")
for p, q in yes: print(f"  {p:4.2f}  {q}")
print("\\nJev's multi-level answers:")
for q, levels in GROUPS.items():
    p = np.array(row["g::" + q]); print(f"  {q:28s} -> {levels[p.argmax()]:12s} (sure at {p.max():.0%})")
""")

md("""
### Most questions are rare
A critic mentions a few aromas, not all 99. Before training, let's see how rare each yes/no question is.
This matters: a model that always answers "no" would already look 99% right, so **accuracy is the wrong yardstick**; we will use AUC and F1 later.
""")

code("""
B = df[BIN_COLS].to_numpy(dtype=np.float32)
share = (B >= 0.5).mean(0)                                    # fraction of notes where Jev says "yes"
fig = go.Figure(go.Histogram(x=share * 100, nbinsx=40, marker_color="#009999", hovertemplate="%{x}% of notes: %{y} questions<extra></extra>"))
fig.update_layout(title="Most descriptors are rare", xaxis_title="% of notes where Jev says yes", yaxis_title="number of questions (log)", yaxis_type="log", height=340)
show(fig, "00_rarity")
print("questions that fire in fewer than 1% of notes:", int((share < 0.01).sum()), "of", len(share))
""")

# --------------------------------------------------------------------------------------------------------------------
md("""
## 3. An honest split

A school analogy: the **training notes** are the practice questions the student works through (and learns from). The **validation notes** are a mock exam:
we use it to check progress and to make small decisions (which epoch is best? what threshold?), but the student never *learns* from it.
The **test notes** are the real exam: sealed in an envelope, opened **once**, at the very end. If we peeked at them while tuning, the final score would be flattering and untrue.

| Set | Share of wineries | Changes the model's weights? | What we use it for | When it is used |
|---|---|---|---|---|
| **Train** | `1 - VAL_FRAC - TEST_FRAC` (80%) | **yes** | learning | every epoch, in shuffled batches |
| **Validation** | `VAL_FRAC` (10%) | no | watching progress after every epoch, picking the best epoch, tuning thresholds | after every epoch |
| **Test** | `TEST_FRAC` (10%) | no | the final, honest score | once, after training is finished |

**Split by winery, not by note.** A winery's notes sound alike (same critic, same style). If one note from Chateau X is in training and another in test,
the model can look clever just by recognising the winery. Putting all of a winery's notes on one side avoids that.
""")

code("""
rng = np.random.RandomState(SEED)
key = df.winery.fillna(df.id)                                  # notes with no winery count as their own "winery"
wineries = key.unique(); rng.shuffle(wineries)
n = len(wineries); n_val, n_test = int(VAL_FRAC * n), int(TEST_FRAC * n)
val_set  = set(wineries[: n_val])
test_set = set(wineries[n_val: n_val + n_test])
split = key.map(lambda w: "val" if w in val_set else "test" if w in test_set else "train")
train_idx, val_idx, test_idx = [np.where(split == s)[0] for s in ("train", "val", "test")]
print("train / val / test notes:", len(train_idx), len(val_idx), len(test_idx))

# leakage check: no winery may appear on two sides
sets = {s: set(key[split == s]) for s in ("train", "val", "test")}
print("wineries shared between train and val / train and test / val and test:",
      len(sets["train"] & sets["val"]), len(sets["train"] & sets["test"]), len(sets["val"] & sets["test"]), "(must all be 0)")
""")

md("""
### The split, in pictures
Left: how many notes and wineries sit in each set. Right: how often each set is *looked at* and how often it *changes the model* (we will fill the training numbers in section 7).
""")

code("""
cnt = pd.DataFrame({"set": ["train", "validation", "test"], "notes": [len(train_idx), len(val_idx), len(test_idx)],
                    "wineries": [len(sets["train"]), len(sets["val"]), len(sets["test"])]})
cols = {"train": "#009999", "validation": "#d6b02e", "test": "#c0392b"}
fig = make_subplots(rows=1, cols=2, subplot_titles=("Notes and wineries per set", "How each set is used"), column_widths=[0.5, 0.5])
for m in ("notes", "wineries"):
    fig.add_trace(go.Bar(name=m, x=cnt.set, y=cnt[m], text=cnt[m], textposition="outside"), row=1, col=1)
usage = pd.DataFrame({"set": ["train", "validation", "test"], "times each note is READ by the model": [EPOCHS, EPOCHS + 1, 1],
                      "times each note CHANGES the weights": [EPOCHS, 0, 0]})
for c, colr in zip(usage.columns[1:], ("#7fb3d5", "#e67e22")):
    fig.add_trace(go.Bar(name=c, x=usage.set, y=usage[c], marker_color=colr, text=usage[c], textposition="outside"), row=1, col=2)
fig.update_layout(barmode="group", height=420, title="Which notes are used for what (EPOCHS = %d)" % EPOCHS, legend=dict(orientation="h", y=-0.2))
show(fig, "01_split")

# a few example wineries per set, so the idea is concrete
for s in ("train", "val", "test"):
    top = key[split == s].value_counts().head(3)
    print(f"{s:5s} e.g. " + " | ".join(f"{w} ({k} notes)" for w, k in top.items()))
""")

# --------------------------------------------------------------------------------------------------------------------
md("""
## 4. Turning words into numbers

A *tokenizer* chops text into pieces the model knows (`"blackcurrant"` -> `black`, `currant`) and gives each a number.
We pad every note to exactly `MAX_LEN` tokens, so all batches have the same shape (fast on GPU, **required** for TPU),
and a *mask* tells the model which positions are real words and which are padding.
""")

code("""
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
enc = tok(df.description.tolist(), padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
print(tok.convert_ids_to_tokens(enc["input_ids"][0][:15]), "...")
print("notes cut off by MAX_LEN:", int((enc["attention_mask"].sum(1) == MAX_LEN).sum()))

# Targets: yes/no probabilities, and one probability vector per multi-level question (made to sum to exactly 1)
Y_bin = torch.tensor(B)
group_sizes = [len(v) for v in GROUPS.values()]
slices = np.cumsum([0] + group_sizes)                          # where each question starts inside the long vector
Y_grp = np.concatenate([np.stack(df[c].to_numpy()) for c in GROUP_COLS], axis=1).astype(np.float32)
for s, e in zip(slices[:-1], slices[1:]):
    Y_grp[:, s:e] /= Y_grp[:, s:e].sum(1, keepdims=True).clip(min=1e-6)
Y_grp = torch.tensor(Y_grp)
print("yes/no targets:", tuple(Y_bin.shape), "| multi-level targets:", tuple(Y_grp.shape))
""")

md("""
### A visual example: from a sentence to a row of numbers
The journey of one sentence, in four steps (the next cells do steps 1 and 2; step 3 and 4 come in section 5, once the model exists):

```
 "Ripe blackcurrant and vanilla"
   │ 1. TOKENIZE      cut into pieces the model knows           [CLS] ripe black ##currant and vanilla [SEP] <pad> <pad> ...
   │ 2. IDS + MASK    each piece -> a number; mask = 1 real, 0 padding     101  7612  2304  ...         1 1 1 1 1 1 1 0 0
   │ 3. EMBED         the encoder turns EACH token into a vector of 768 numbers that depends on its neighbours (so "cherry" in
   │                  "black cherry" and in "cherry blossom" get different vectors)
   │ 4. AVERAGE       average the vectors of the real tokens (not the padding)  ->  ONE vector of 768 numbers for the whole note
   ▼
 the head(s) read that single vector and answer the questions
```
""")

code("""
sentence = "Ripe blackcurrant and vanilla, with firm tannins."
SHOW_LEN = 24
e1 = tok(sentence, padding="max_length", max_length=SHOW_LEN, truncation=True)
toks = tok.convert_ids_to_tokens(e1["input_ids"])
real = np.array(e1["attention_mask"], dtype=bool)
fig = go.Figure(go.Table(
    header=dict(values=["position"] + list(range(SHOW_LEN)), fill_color="#eeeeee", align="center"),
    cells=dict(values=[["token", "id", "mask"]] + [[toks[i], e1["input_ids"][i], int(real[i])] for i in range(SHOW_LEN)], align="center",
               fill_color=[["#f5f5f5"] * 3] + [["#d9f0f0" if real[i] else "#e0e0e0"] * 3 for i in range(SHOW_LEN)], height=30)))
fig.update_layout(height=230, title="Steps 1-2: tokens, ids and mask (teal = real tokens, grey = padding up to a fixed length)")
show(fig, "00c_tokens")

print("Words the tokenizer does not know are cut into known pieces:")
for w in ["blackcurrant", "minerality", "Gewürztraminer", "petrichor", "Sangiovese", "tannins"]:
    print(f"  {w:15s} -> {tok.tokenize(w)}")
""")

code("""
from torch.utils.data import TensorDataset, DataLoader

def make_loader(idx, shuffle):
    ds = TensorDataset(enc["input_ids"][idx], enc["attention_mask"][idx], Y_bin[idx], Y_grp[idx])
    # drop_last=True while training keeps every batch the same size (TPUs recompile if the shape changes)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=shuffle)

train_dl, val_dl, test_dl = make_loader(train_idx, True), make_loader(val_idx, False), make_loader(test_idx, False)
""")

md("""
### How the training notes are served to the model
The model never sees all notes at once. The *data loader* shuffles the training notes, cuts them into **batches**, and hands them over one batch at a time.
**One batch = one step** (one small nudge to the weights). **One pass through all batches = one epoch.** Next epoch the notes are **shuffled again**, so batches are different each time.
""")

code("""
steps_per_epoch = len(train_dl)
used = steps_per_epoch * BATCH_SIZE
print(f"{len(train_idx):,} training notes / batch of {BATCH_SIZE} = {steps_per_epoch} steps per epoch  "
      f"({used:,} notes used per epoch; the last {len(train_idx) - used} are skipped this epoch, but the reshuffle brings them back in the next)")
print(f"{EPOCHS} epochs = {EPOCHS * steps_per_epoch:,} steps in total; each training note is read {EPOCHS} times\\n")

# Look at the order for three epochs: the first 6 notes of the first batch (positions in the training set) differ every time
g = torch.Generator()
for e in range(3):
    g.manual_seed(SEED + e)
    order = torch.randperm(len(train_idx), generator=g)[:BATCH_SIZE]
    print(f"epoch {e+1}: first batch = notes", [int(train_idx[i]) for i in order[:6]], "...")

# And what ONE batch looks like for the model: a block of token numbers, plus the answers it should reproduce
ids, mask, yb, yg = next(iter(train_dl))
print(f"\\none batch: input {tuple(ids.shape)} (notes x tokens) | yes/no answers {tuple(yb.shape)} (notes x 216) | level answers {tuple(yg.shape)}")
""")

# --------------------------------------------------------------------------------------------------------------------
md("""
## 5. The student model

Three pieces:

1. **Encoder (the "body")**: a pre-trained language model (RoBERTa by default) that already understands English. It turns the note into one vector per token.
2. **Pooling**: average the token vectors (ignoring padding) to get *one vector for the whole note*: a list of 768 numbers that summarises its meaning.
3. **Two heads**: small linear layers that turn that vector into answers:
   * a *yes/no head* with 216 outputs (one score per descriptor),
   * a *multi-level head* with one output per level of every multi-level question.

Only the heads start from random; the encoder starts from what it learned reading millions of pages, which is why a few thousand notes are enough.

### What is a *head*?

Think of the body as a **taster who reads a note and writes a short private summary** (768 numbers). That summary is not about wine specifically: it is a general "what is this text about".
A **head** is a *panel of small judges* who all read that same summary, and each judge only answers **one question**:

* the judge for *"vanilla on the nose?"* has 768 weights, one per summary number, and a bias. It multiplies summary × weights, adds them up, and squashes the result to a probability between 0 and 1 (a *sigmoid*);
* the judge for *"how high is the tannin?"* does the same for five levels, then a *softmax* makes the five scores add up to 1.

So **a head is just a table of weights with one row per question** (a `Linear` layer). *Heads are the part that is specific to our task*; the body is general-purpose and reusable.
That is why the **same body can carry many heads**: swap in a different head and the same reader answers a different set of questions (e.g. predicting the grape variety from the note).

> **Not to be confused with** the *attention heads* inside the transformer body. Same word, different thing: those are internal to how the body reads. Here, "head" means *the output layer that answers our questions*.

During training the body learns only a *gentle adjustment* (small learning rate: it already knows English), while the heads learn fast (large learning rate: they start from random numbers).
""")

md("""
### Which body? A short zoo of encoders
Any **Transformer encoder** can be the body. They share one architecture: a stack of identical *layers*; each layer has an *attention* block (every token looks at all the other tokens
and decides which ones matter for it, using several parallel "attention heads") and a *feed-forward* block (a small network applied to each token). More layers and wider vectors = more capacity, more memory, slower.
Change `MODEL_NAME` in the settings to try another; nothing else in the notebook needs to change.

| Model (`MODEL_NAME`) | Parameters | Layers x vector width | Attention heads | When to pick it |
|---|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 22 M | 6 x 384 | 12 | tiny and fast; a sentence-embedding model; fine for a first test or a CPU |
| `distilbert-base-uncased` | 66 M | 6 x 768 | 12 | a *distilled* BERT: 40% smaller, keeps most of BERT's ability |
| `bert-base-uncased` | 110 M | 12 x 768 | 12 | the classic (2018) |
| `roberta-base` (**default**) | 125 M | 12 x 768 | 12 | BERT trained longer on more text; robust, no surprises |
| `microsoft/deberta-v3-base` | 184 M (86 M layers + 98 M word table) | 12 x 768 | 12 | often the most accurate of the base size; fussier with fp16 |
| `answerdotai/ModernBERT-base` | 149 M | 22 x 768 | 12 | 2024 design, fast, reads long texts (8,192 tokens); needs a recent `transformers` |
| `roberta-large` | 355 M | 24 x 1024 | 16 | a little more accurate, about 3x the memory and time |

(Parameter counts are the published, rounded figures. Our notes are about 60 words, so a long-text model brings little here.)
""")

code("""
ZOO = pd.DataFrame([
    ("all-MiniLM-L6-v2", 22, 6, 384, 12), ("distilbert-base", 66, 6, 768, 12), ("bert-base", 110, 12, 768, 12), ("roberta-base", 125, 12, 768, 12),
    ("deberta-v3-base", 184, 12, 768, 12), ("ModernBERT-base", 149, 22, 768, 12), ("roberta-large", 355, 24, 1024, 16)],
    columns=["model", "params_M", "layers", "width", "attention_heads"])
fig = go.Figure(go.Scatter(x=ZOO.params_M, y=ZOO.layers, mode="markers+text", text=ZOO.model, textposition="top center",
                           marker=dict(size=ZOO.width / 18, color=ZOO.params_M, colorscale="Teal", showscale=False, line=dict(width=1, color="#444")),
                           customdata=ZOO[["width", "attention_heads"]], hovertemplate="%{text}<br>%{x} M parameters<br>%{y} layers<br>vector width %{customdata[0]}, %{customdata[1]} attention heads<extra></extra>"))
fig.update_layout(title="The encoder zoo: more parameters, deeper, wider (bubble = vector width)", xaxis_title="parameters (millions, log scale)", xaxis_type="log",
                  yaxis_title="number of layers", yaxis_range=[0, 28], height=440)
show(fig, "02a_zoo")
""")

md("""
### What does distillation actually change?
Three different "trainings" are involved. Only the middle one happens in this notebook.

| Stage | Who | Which numbers change | Data used |
|---|---|---|---|
| **Pre-training** | the model's authors (Meta, Google, ...) | *all* body weights, starting from random, to learn English | billions of words |
| **Distillation** (this notebook) | us | **body:** every layer is *nudged* a little (tiny learning rate); **heads:** learned **from scratch** (large learning rate) | ~10,000 notes + Jev's answers |
| **The teacher, Jev** | TypeSafe | **nothing**: we never touch it, we only read its answers | |

Section 7 measures this on the real model: *how much did each part move compared with where it started?* Expect the heads to move a lot (they started random) and the body only slightly.

Two other ways to adapt a body are **compared in section 7b** with real training runs: **freezing** the body and training only the heads (fast, a bit less accurate), and **LoRA** (add tiny trainable adapters and leave the original weights untouched).
""")

code("""
class WsetStudent(nn.Module):
    def __init__(self, model_name, n_binary, n_group_outputs):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)               # the BODY
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)                         # randomly silences 10% of numbers while training: stops over-memorising
        self.binary_head = nn.Linear(h, n_binary)              # HEAD 1: 216 yes/no scores (a table of 216 x 768 weights)
        self.group_head  = nn.Linear(h, n_group_outputs)       # HEAD 2: all levels of all multi-level questions, side by side

    def embed(self, input_ids, attention_mask):                # body + pooling: note -> 768 numbers
        tokens = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state   # (batch, tokens, h)
        m = attention_mask.unsqueeze(-1).to(tokens.dtype)                                              # 1 for words, 0 for padding
        return (tokens * m).sum(1) / m.sum(1).clamp(min=1)                                             # average over real words

    def forward(self, input_ids, attention_mask):
        note = self.dropout(self.embed(input_ids, attention_mask))
        return self.binary_head(note), self.group_head(note)

model = WsetStudent(MODEL_NAME, len(BIN_COLS), int(slices[-1])).to(device)
""")

md("""
### Open the box: how big are the body and the heads?
""")

code("""
n_body = sum(p.numel() for p in model.encoder.parameters())
n_bin  = sum(p.numel() for p in model.binary_head.parameters())
n_grp  = sum(p.numel() for p in model.group_head.parameters())
fig = go.Figure(go.Bar(x=["body (encoder)", "yes/no head", "level head"], y=[n_body, n_bin, n_grp], text=[f"{v/1e6:.2f} M" for v in (n_body, n_bin, n_grp)],
                       textposition="outside", marker_color=["#7fb3d5", "#009999", "#d6b02e"]))
fig.update_layout(title=f"Parameters (numbers the model can learn). Heads are {100 * (n_bin + n_grp) / (n_body + n_bin + n_grp):.2f}% of the model", yaxis_title="parameters", yaxis_type="log", height=380)
show(fig, "02_body_and_heads")
print(f"body: {n_body/1e6:.1f} M | yes/no head: {tuple(model.binary_head.weight.shape)} = {n_bin:,} | level head: {tuple(model.group_head.weight.shape)} = {n_grp:,}")
""")

code("""
# Where do the body's numbers live? Group every weight by the part of the transformer it belongs to.
import re
def component(name):
    n = name.lower()
    if "norm" in n: return "layer norms"
    if "pooler" in n: return "other (pooler...)"
    if "embed" in n: return "word & position tables (embeddings)"
    if "attn" in n or "attention" in n: return "attention"
    if any(k in n for k in ("intermediate", "mlp", "ffn", "output.dense", ".wi.", ".wo.", "dense")): return "feed-forward"
    return "other (pooler...)"
def layer_index(name):
    m = re.search(r"(?:layer|layers|blocks|h)\\.(\\d+)\\.", name); return int(m.group(1)) if m else None

body = pd.Series({k: v.numel() for k, v in model.encoder.named_parameters()})
comp = body.groupby([component(k) for k in body.index]).sum().sort_values(ascending=False)
cfg = model.encoder.config
print(f"{MODEL_NAME}: {getattr(cfg, 'num_hidden_layers', '?')} layers x width {cfg.hidden_size}, {getattr(cfg, 'num_attention_heads', '?')} attention heads per layer, vocabulary {getattr(cfg, 'vocab_size', '?'):,} pieces")
fig = go.Figure(go.Bar(x=comp.index, y=comp.values / 1e6, text=[f"{v/1e6:.1f} M" for v in comp.values], textposition="outside", marker_color="#7fb3d5"))
fig.update_layout(title="Inside the body: where its parameters live (millions)", yaxis_title="parameters (M)", height=380)
show(fig, "02b_body_components")
""")

md("""
### "Multi-headed": one body, several heads
Our student is **multi-headed**: one body, **two heads**, each answering a different *kind* of question from the same note summary.

```
                         ┌──► YES/NO HEAD   (216 outputs, each a probability)         loss: binary cross-entropy
 note ─► BODY ─► summary ┤
                         └──► LEVEL HEAD    (76 outputs = 15 questions with their levels)  loss: cross-entropy per question
                                  (a third head could be added: e.g. the grape, the vintage, the points ...)
```
Why share one body instead of training two models?
* the body learns from **both** kinds of question at once (*multi-task learning*): knowing a note says "firm tannins" helps both the descriptor *and* the tannin-level answers;
* one reading of the note serves every head (cheap at prediction time);
* a new head can be added later without retraining from scratch.

How it trains: the two losses are **added** (`BIN_WEIGHT` sets the balance) and the gradient from *both* heads flows back into the same body weights. Below, we measure that pull on one real weight.

> **Two different "heads", don't mix them up.** *Output heads* (this section) are the answer layers on top. *Attention heads* are the parallel "looking" units inside each body layer (12 per layer in RoBERTa).
> A **multi-headed model** has several output heads; **multi-head attention** is how every transformer layer reads.
""")

md("""
### From tokens to one vector (steps 3 and 4 of the journey)
Now that the body exists we can watch it work on the sentence from section 4. Each token becomes a vector; we average the vectors of the real tokens to get the note's summary.
""")

code("""
model.eval()
with torch.no_grad():
    e = tok([sentence], padding="max_length", max_length=SHOW_LEN, truncation=True, return_tensors="pt")
    vecs = model.encoder(input_ids=e["input_ids"].to(device), attention_mask=e["attention_mask"].to(device)).last_hidden_state[0].float().cpu()   # (SHOW_LEN, width)
    m = e["attention_mask"][0].bool()
    summary_manual = vecs[m].mean(0)                                                         # step 4 by hand: average of the REAL tokens only
    summary_model  = model.embed(e["input_ids"].to(device), e["attention_mask"].to(device))[0].float().cpu()
    summary_wrong  = vecs.mean(0)                                                            # what you would get if you forgot the mask (padding included)
print("tokens:", toks[: int(m.sum())], "+", int((~m).sum()), "padding")
print(f"average of real tokens (by hand) vs the model's pooling: identical = {torch.allclose(summary_manual, summary_model, atol=1e-5)}")
print(f"average INCLUDING padding differs by cosine {torch.cosine_similarity(summary_manual, summary_wrong, dim=0):.3f} (1.0 would be identical): this is why we pass a mask")

# a toy to see the arithmetic: 3 tokens, 4 numbers each
toy = torch.tensor([[1., 0., 2., 4.], [3., 2., 0., 4.], [2., 4., 1., 1.]])
print("\\ntoy example, three token vectors:\\n", toy.numpy(), "\\naverage of each column =", toy.mean(0).numpy())

DIMS = 24
rows = [f"{i}: {toks[i]}" for i in range(int(m.sum()))]
Z = torch.vstack([vecs[m][:, :DIMS], summary_manual[:DIMS]]).numpy()
fig = go.Figure(go.Heatmap(z=Z, y=rows + ["AVERAGE = the note's summary"], x=[f"d{i}" for i in range(DIMS)], colorscale="RdBu", zmid=0, colorbar=dict(title="value")))
fig.update_layout(title=f"Steps 3-4: one vector per token (rows), averaged into the summary (last row). First {DIMS} of {vecs.shape[1]} numbers shown",
                  height=60 + 38 * (len(rows) + 1), yaxis=dict(autorange="reversed"), xaxis_title=f"vector dimension (first {DIMS} of {vecs.shape[1]})")
show(fig, "02c_tokens_to_vector")

# the same word gets a different vector in a different sentence ("context")
def token_vec(text, word):
    enc_ = tok([text], return_tensors="pt", truncation=True, max_length=MAX_LEN)
    with torch.no_grad(): h = model.encoder(input_ids=enc_["input_ids"].to(device), attention_mask=enc_["attention_mask"].to(device)).last_hidden_state[0].float().cpu()
    for i, t in enumerate(tok.convert_ids_to_tokens(enc_["input_ids"][0])):
        if word in t.lower(): return h[i]
a, b_, c_ = token_vec("Ripe black cherry and vanilla on the nose.", "cherry"), token_vec("Delicate cherry blossom and white pepper.", "cherry"), token_vec("Ripe black plum and vanilla on the nose.", "plum")
if a is not None and b_ is not None and c_ is not None:
    cs = lambda x, y: float(torch.cosine_similarity(x, y, dim=0))
    print(f"\\nsimilarity of the vector for 'cherry' in 'black cherry' vs 'cherry blossom': {cs(a, b_):.2f}  (same word, different context)")
    print(f"similarity of 'cherry' (black cherry) and 'plum' (black plum): {cs(a, c_):.2f}  (different words, similar context)")
""")

md("""
## 5b. Inside the encoder: layers and attention heads
The body is not a black box. It is a **stack of identical layers**. Each layer lets every token *look at the other tokens* (**attention**), then thinks about what it saw (**feed-forward**).
After the last layer every token's vector has absorbed information from the whole note. Here is the stack for the model you chose:
""")

code("""
cfg = model.encoder.config
N_LAYERS, N_HEADS, WIDTH = getattr(cfg, "num_hidden_layers", 12), getattr(cfg, "num_attention_heads", 12), cfg.hidden_size
mermaid(f'''
flowchart TB
  txt["critic note: Ripe blackcurrant and vanilla ..."] --> tk["tokenizer: pieces and ids"]
  tk --> emb["embeddings: each piece becomes a vector of {WIDTH} numbers<br/>(word table + position)"]
  emb --> L1
  subgraph STACK["{N_LAYERS} identical layers, one after the other"]
    direction TB
    L1["layer 1"] --> L2["layer 2"] --> Ldots["..."] --> LN["layer {N_LAYERS}"]
  end
  LN --> tv["one vector per token ({WIDTH} numbers each)"]
  tv --> pool["average the real tokens (mask out padding)"]
  pool --> sm["note summary: {WIDTH} numbers"]
  sm --> h1["yes/no head"]
  sm --> h2["level head"]
  classDef body fill:#e8f1fa,stroke:#2375ae
  classDef head fill:#d9f0f0,stroke:#009999
  class emb,L1,L2,Ldots,LN,tv body
  class h1,h2 head
''', "02d_encoder_stack", 620)

mermaid(f'''
flowchart LR
  x["token vectors in"] --> att["ATTENTION<br/>{N_HEADS} heads look at the other tokens in parallel"]
  att --> an["add + normalise<br/>(keep the original, stay stable)"]
  an --> ff["FEED-FORWARD<br/>a small network on each token<br/>({WIDTH} to {getattr(cfg, 'intermediate_size', 4 * WIDTH)} to {WIDTH})"]
  ff --> fn["add + normalise"]
  fn --> y["token vectors out<br/>same shape, richer meaning"]
  x -. "shortcut" .-> an
  an -. "shortcut" .-> fn
  classDef a fill:#f3e5ab,stroke:#b58900
  classDef f fill:#d9f0f0,stroke:#009999
  class att a
  class ff f
''', "02e_one_layer", 300)
""")

md("""
### What is an attention head?
Inside the attention block, every token plays three roles at once:

* a **query** ("what am I looking for?"),
* a **key** ("what do I offer?"),
* a **value** ("what do I actually hand over if you pick me?").

A token compares its query with *every* token's key (a dot product = how well they match). The scores go through a *softmax* so they add up to 1: these are the **attention weights**.
The token's new vector is the **weighted mix of the values**. One such "look around" is **one attention head**. A layer has several heads (12 in RoBERTa) that look **in parallel, each with its own query/key/value matrices**,
so each can specialise (one follows the previous word, one links an adjective to its noun, one watches the punctuation...). Their outputs are glued together and mixed. Formula: `softmax(Q·Kᵀ / √d) · V`.
""")

code("""
mermaid('''
flowchart LR
  tok["token vectors"] --> q["Q = what I look for"]
  tok --> k["K = what I offer"]
  tok --> v["V = what I hand over"]
  q --> s["scores = Q x K-transposed / sqrt(d)"]
  k --> s
  s --> sf["softmax: weights add up to 1"]
  sf --> mix["weighted mix of the values"]
  v --> mix
  mix --> out["new vector for the token"]
  out --> cat["glue the heads together<br/>(this is the multi-head part)"]
  classDef qkv fill:#f3e5ab,stroke:#b58900
  class q,k,v qkv
''', "02f_attention_head", 320)

# the arithmetic of ONE head with three tokens and 2-number vectors, by hand
words = ["black", "cherry", "pie"]
Q = np.array([[1., 0.], [0., 1.], [1., 1.]]); K = np.array([[1., 0.], [1., 1.], [0., 1.]]); V = np.array([[1., 0.], [0., 2.], [3., 1.]])
scores = Q @ K.T / np.sqrt(2)
weights = np.exp(scores) / np.exp(scores).sum(1, keepdims=True)
new_vecs = weights @ V
print("scores = Q.K^T / sqrt(2):\\n", scores.round(2), "\\n\\nweights (each row adds to 1):\\n", weights.round(2), "\\n\\nnew vectors = weights . V:\\n", new_vecs.round(2))
fig = go.Figure(go.Heatmap(z=weights, x=words, y=words, colorscale="Blues", text=weights.round(2), texttemplate="%{text}", zmin=0, zmax=1, showscale=False))
fig.update_layout(title="Toy attention head: row = the token that looks, column = the token it looks at", height=330, yaxis=dict(autorange="reversed", title="looks"), xaxis=dict(title="is looked at"))
show(fig, "02g_toy_attention")
""")

md("""
### The real attention of the real body
Below, the model you chose reads our sentence and we draw **what every head of every layer pays attention to**. Rows = the token that is looking, columns = the tokens it looks at; darker = more attention.
Use the **drop-down to change layer**. Typical things to spot: heads that look at the *previous* or *next* word (a bright diagonal just below or above the main one), heads that dump attention on the first and last tokens (the special markers), and heads spread evenly over everything.
(Needs a model that can report attention; if yours cannot, this part is skipped.)
""")

code("""
import gc, math
from transformers import AutoModel as _AutoModel
eager = None
try:
    eager = _AutoModel.from_pretrained(MODEL_NAME, attn_implementation="eager").to(device).eval()
    e_ = tok([sentence], return_tensors="pt")
    with torch.no_grad(): out_ = eager(input_ids=e_["input_ids"].to(device), attention_mask=e_["attention_mask"].to(device), output_attentions=True)
    ATT = torch.stack([a[0] for a in out_.attentions]).float().cpu().numpy()                 # (layers, heads, tokens, tokens)
    labels = tok.convert_ids_to_tokens(e_["input_ids"][0])
    L_, H_ = ATT.shape[0], ATT.shape[1]; cols = 4; rows = math.ceil(H_ / cols)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"head {h}" for h in range(H_)], horizontal_spacing=0.04, vertical_spacing=0.07)
    for l in range(L_):
        for h in range(H_):
            fig.add_trace(go.Heatmap(z=ATT[l, h], x=labels, y=labels, colorscale="Blues", zmin=0, zmax=1, showscale=False, visible=(l == L_ - 1),
                                     hovertemplate="looks: %{y}<br>at: %{x}<br>weight %{z:.2f}<extra>layer " + str(l) + " head " + str(h) + "</extra>"), row=h // cols + 1, col=h % cols + 1)
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=7)); fig.update_xaxes(tickfont=dict(size=7), tickangle=60)
    buttons = [dict(label=f"layer {l}", method="update", args=[{"visible": [(i // H_) == l for i in range(L_ * H_)]}]) for l in range(L_)]
    fig.update_layout(updatemenus=[dict(buttons=buttons, active=L_ - 1, x=0.0, y=1.12, xanchor="left")], height=230 * rows + 60,
                      title=f"Attention of every head, sentence: '{sentence}' (pick the layer)")
    show(fig, "02h_attention_heads")
except Exception as err:
    print("attention maps skipped:", repr(err)[:200])
""")

md("""
### What do the heads specialise in? (measured over many notes)
We let the body read 32 notes and measure, for every head of every layer:

* **focus (entropy):** low = the head concentrates on a few tokens, high = it spreads attention everywhere,
* **previous-token** and **next-token** attention: the share of attention that goes to the word just before / after,
* **start/end markers:** the share that goes to the first and last token (the special [CLS]/[SEP] markers; heads often park their attention there when there is nothing to do).

Each cell is one head. If every cell were the same colour the heads would be redundant; the variety is what multi-head attention buys.
""")

code("""
if eager is not None:
    try:
        texts = df.description.iloc[:32].tolist()
        enc_b = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            att = eager(input_ids=enc_b["input_ids"].to(device), attention_mask=enc_b["attention_mask"].to(device), output_attentions=True).attentions
        A = torch.stack([a.float().cpu() for a in att], 1)                                     # (notes, layers, heads, T, T)
        ent, prv, nxt, spc = [torch.zeros(A.shape[1], A.shape[2]) for _ in range(4)]
        for b in range(A.shape[0]):
            n = int(enc_b["attention_mask"][b].sum()); Ab = A[b, :, :, :n, :n]
            ent += -(Ab * torch.log(Ab + 1e-9)).sum(-1).mean(-1) / math.log(2)
            prv += torch.diagonal(Ab, offset=-1, dim1=-2, dim2=-1).mean(-1); nxt += torch.diagonal(Ab, offset=1, dim1=-2, dim2=-1).mean(-1)
            spc += (Ab[..., 0] + Ab[..., n - 1]).mean(-1)
        S = {k: (v / A.shape[0]).numpy() for k, v in dict(ent=ent, prv=prv, nxt=nxt, spc=spc).items()}
        titles = ("focus: entropy in bits (low = focused)", "attention to the PREVIOUS token", "attention to the NEXT token", "attention to first + last token")
        fig = make_subplots(rows=2, cols=2, subplot_titles=titles, horizontal_spacing=0.12, vertical_spacing=0.14)
        for (r_, c_), k, cs in zip([(1, 1), (1, 2), (2, 1), (2, 2)], ["ent", "prv", "nxt", "spc"], ["Viridis", "Blues", "Blues", "Blues"]):
            fig.add_trace(go.Heatmap(z=S[k], x=list(range(S[k].shape[1])), y=list(range(S[k].shape[0])), colorscale=cs, colorbar=dict(len=0.4, thickness=10, x=0.46 if c_ == 1 else 1.0, y=0.8 if r_ == 1 else 0.2)), row=r_, col=c_)
        fig.update_yaxes(title_text="layer", autorange="reversed"); fig.update_xaxes(title_text="head", dtick=1)
        fig.update_layout(height=700, title="Every head of every layer: what it pays attention to (average over 32 notes)")
        show(fig, "02i_head_roles")
        l_, h_ = np.unravel_index(S["prv"].argmax(), S["prv"].shape); print(f"most 'previous-word' head: layer {l_} head {h_} ({S['prv'][l_, h_]:.0%} of its attention)")
        l_, h_ = np.unravel_index(S["nxt"].argmax(), S["nxt"].shape); print(f"most 'next-word' head: layer {l_} head {h_} ({S['nxt'][l_, h_]:.0%})")
        l_, h_ = np.unravel_index(S["ent"].argmin(), S["ent"].shape); print(f"most focused head: layer {l_} head {h_} ({S['ent'][l_, h_]:.2f} bits)")
    except Exception as err:
        print("head statistics skipped:", repr(err)[:200])
    del eager; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
""")

md("""
### The body already knows some wine words (before any training)
The body was pre-trained on general text, never on wine notes. Yet if we put wine words in a sentence (*"This wine has notes of ..."*), take the vector of each word and squash the 768 (or 384...) dimensions down to 2 with PCA,
words of the same family tend to sit together. That is the knowledge the heads will tap into: distillation does not teach the body what "cherry" means, it teaches the **heads** which parts of the body's summary matter for the WSET questions.
""")

code("""
FAM = {"fruit": ["blackcurrant", "cherry", "raspberry", "plum", "apple", "pear", "lemon", "peach", "apricot", "strawberry"],
       "oak & spice": ["vanilla", "cedar", "toast", "smoke", "clove", "pepper", "cinnamon", "oak", "coconut"],
       "structure": ["tannins", "acidity", "body", "finish", "alcohol", "texture"],
       "earth & savoury": ["earth", "mineral", "leather", "mushroom", "tobacco", "graphite", "gravel"],
       "floral & sweet": ["rose", "violet", "honey", "blossom", "jasmine", "caramel"]}
def word_vec(word):
    text = f"This wine has notes of {word}."
    e = tok([text], return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    off = e.pop("offset_mapping")[0].tolist(); a = text.index(word); b = a + len(word)
    idx = [i for i, (s0, e0) in enumerate(off) if e0 > s0 and s0 < b and e0 > a]
    with torch.no_grad(): h = model.encoder(input_ids=e["input_ids"].to(device), attention_mask=e["attention_mask"].to(device)).last_hidden_state[0].float().cpu()
    return h[idx].mean(0).numpy()
model.eval()
words_, fam_, vecs_ = [], [], []
for f, ws in FAM.items():
    for w in ws: words_.append(w); fam_.append(f); vecs_.append(word_vec(w))
Wm = np.array(vecs_); Wm = Wm - Wm.mean(0); U_, S_, Vt_ = np.linalg.svd(Wm, full_matrices=False); P2 = Wm @ Vt_[:2].T
fig = go.Figure()
pal = {"fruit": "#c0392b", "oak & spice": "#b58900", "structure": "#2375ae", "earth & savoury": "#6e4b2a", "floral & sweet": "#8e44ad"}
for f in FAM:
    m_ = [i for i, x in enumerate(fam_) if x == f]
    fig.add_trace(go.Scatter(x=P2[m_, 0], y=P2[m_, 1], mode="markers+text", text=[words_[i] for i in m_], textposition="top center", name=f, marker=dict(size=11, color=pal[f])))
fig.update_layout(title="Wine words in the body's own space (PCA, before any training; colour = family we chose)", xaxis_title="PC 1", yaxis_title="PC 2", height=520)
show(fig, "02j_word_space")
""")

md("""
## 5c. The architecture as matrices (shapes, not counts)
Everything the body and the heads do is **matrix multiplication**. The useful question is not "how many weights?" but **"what shape does each matrix have, and what goes in and out?"**
Notation (the numbers for *your* model are printed below): `T` = number of tokens, `d` = vector width, `H` = attention heads, `d_h = d / H` = width of one head, `d_ff` = width of the feed-forward block (usually 4d).

| Object | Shape | Meaning |
|---|---|---|
| `X` | `T x d` | the note: one row (a vector of `d` numbers) per token |
| `W_Q, W_K, W_V` | `d x d` each | learned matrices; `Q = X·W_Q`, `K = X·W_K`, `V = X·W_V` are each `T x d` |
| one head | columns `h·d_h ... (h+1)·d_h` of `Q, K, V` | each head takes its own slice of `d_h` columns: `Q_h, K_h, V_h` are `T x d_h` |
| scores | `T x T` per head | `Q_h · K_hᵀ / √d_h`: how well token *i* matches token *j* |
| attention weights | `T x T` per head | softmax of each row: rows add to 1 |
| head output | `T x d_h` | `weights · V_h` |
| all heads glued | `T x d` | the `H` outputs side by side |
| `W_O` | `d x d` | mixes the heads: `T x d` out |
| `W_1`, `W_2` | `d x d_ff`, `d_ff x d` | feed-forward: widen, apply a non-linearity, narrow again |
| pooling | `T x d` to `1 x d` | average of the rows of the real tokens |
| yes/no head `W_bin` | `d x 216` | `summary (1 x d) · W_bin = 216 scores` |
| level head `W_grp` | `d x 76` | `summary · W_grp = 76 scores` (the levels of all multi-level questions side by side) |

Note: PyTorch stores a `Linear` layer as `(out, in)`; the math above writes the matrix as `(in, out)` so that `X·W` reads left to right.
""")

code("""
d_, H_ = model.encoder.config.hidden_size, getattr(model.encoder.config, "num_attention_heads", 12)
dff = getattr(model.encoder.config, "intermediate_size", 4 * d_)
n_tok = len(tok([sentence])["input_ids"][0])
print(f"{MODEL_NAME}\\n  d (width) = {d_} | H (heads) = {H_} | d_h = d/H = {d_ // H_} | d_ff = {dff} | vocabulary = {model.encoder.config.vocab_size:,} | T for our sentence = {n_tok}\\n")

def role(name):
    n = name.lower()
    if "word_embeddings" in n: return "E  (vocabulary x d): the word table"
    if "position_embeddings" in n: return "P  (positions x d)"
    if "norm" in n: return "layer-norm scale / shift (a vector of d)"
    if n.endswith("query.weight") or "q_proj.weight" in n or "query_proj.weight" in n: return "W_Q"
    if n.endswith("key.weight") or "k_proj.weight" in n or "key_proj.weight" in n: return "W_K"
    if n.endswith("value.weight") or "v_proj.weight" in n or "value_proj.weight" in n: return "W_V"
    if "attention.output.dense.weight" in n: return "W_O"
    if "intermediate.dense.weight" in n: return "W_1  (d to d_ff)"
    if "output.dense.weight" in n: return "W_2  (d_ff to d)"
    if n.endswith(".bias"): return "bias (a vector)"
    return ""
rows_ = []
for name, p in model.encoder.named_parameters():
    if layer_index(name) in (None, 0) and p.ndim >= 1:
        sym = lambda n: {d_: "d", dff: "d_ff", model.encoder.config.vocab_size: "vocab"}.get(n, str(n))
        rows_.append((name.replace("encoder.layer.0.", "layer0."), " x ".join(sym(n) for n in p.shape) + f"   ({' x '.join(map(str, p.shape))})", role(name)))
rows_ += [("binary_head.weight", f"216 x d   (216 x {d_})", "W_bin  (stored out x in)"), ("group_head.weight", f"{int(slices[-1])} x d   ({int(slices[-1])} x {d_})", "W_grp  (stored out x in)")]
pd.DataFrame(rows_, columns=["parameter (layer 0 shown; the other layers repeat it)", "shape", "role"])
""")

md("""
### The matrices drawn to scale
One rectangle per weight matrix of **one layer**, plus the two heads, all drawn with the **same scale** (area = number of weights, but now you also see the *shape*).
Notice: the attention matrices are square `d x d`; the feed-forward block is the big one (widen to `d_ff`, narrow back); the heads are thin, tall-and-narrow matrices: `d` rows in, 216 or 76 columns out.
""")

code("""
mats = [("W_Q", d_, d_, "#f3e5ab"), ("W_K", d_, d_, "#f3e5ab"), ("W_V", d_, d_, "#f3e5ab"), ("W_O", d_, d_, "#f3e5ab"),
        ("W_1", d_, dff, "#d9f0f0"), ("W_2", dff, d_, "#d9f0f0"), ("W_bin", d_, 216, "#a8d5a2"), ("W_grp", d_, int(slices[-1]), "#a8d5a2")]
gap = 0.06 * d_; x0 = 0; shapes, notes = [], []
for nm, r, c, col in mats:                                                        # r = rows (inputs), c = columns (outputs); drawn left to right
    shapes.append(dict(type="rect", x0=x0, x1=x0 + c, y0=0, y1=r, fillcolor=col, line=dict(color="#444", width=1.5)))
    notes.append(dict(x=x0 + c / 2, y=-0.07 * dff, text=f"<b>{nm}</b><br>{r} x {c}", showarrow=False, font=dict(size=11)))
    x0 += c + gap
fig = go.Figure(); fig.update_layout(shapes=shapes, annotations=notes, height=470, title=f"The matrices of one layer and the two heads, to scale ({MODEL_NAME})",
                                     xaxis=dict(visible=False, range=[-gap, x0]), yaxis=dict(visible=False, range=[dff * 1.05, -0.2 * dff], scaleanchor="x"))
fig.add_annotation(x=0, y=-0.17 * dff, xanchor="left", showarrow=False, text="yellow = attention (4 square matrices) | blue = feed-forward | green = the two output heads", font=dict(size=12))
show(fig, "02k_matrices_to_scale")
""")

md("""
### One attention head as matrix multiplications (with the model's real numbers)
We take the **real `W_Q, W_K, W_V, W_O` of layer 0**, run the chain of multiplications ourselves for our sentence, and check that we reproduce what the library computes for the attention block of that layer.
Every line prints the **shape** that comes out, so you can follow the data as it is multiplied, split into heads, scored, mixed and put back together.
""")

code("""
try:
    L0 = model.encoder.encoder.layer[0]
    sa, ao = L0.attention.self, L0.attention.output
    model.eval()
    with torch.no_grad():
        e_ = tok([sentence], return_tensors="pt")
        X = model.encoder(input_ids=e_["input_ids"].to(device), attention_mask=e_["attention_mask"].to(device), output_hidden_states=True).hidden_states[0][0].float()   # (T, d): the input of layer 0
        T = X.shape[0]; dh = d_ // H_
        W = lambda lin: lin.weight.float().t()                                         # (in, out): so that X @ W reads left to right
        Q, K, V = X @ W(sa.query) + sa.query.bias, X @ W(sa.key) + sa.key.bias, X @ W(sa.value) + sa.value.bias
        print(f"X                       {tuple(X.shape)}      T x d            the tokens as rows")
        print(f"Q = X.W_Q (+bias)       {tuple(Q.shape)}      T x d            same for K and V")
        Qh, Kh, Vh = [m.reshape(T, H_, dh).transpose(0, 1) for m in (Q, K, V)]
        print(f"split into {H_} heads     {tuple(Qh.shape)}   H x T x d_h      each head gets its own {dh} columns")
        scores = Qh @ Kh.transpose(1, 2) / math.sqrt(dh)
        print(f"scores = Q_h.K_h^T/sqrt(d_h) {tuple(scores.shape)}  H x T x T   token-to-token match, per head")
        weights = torch.softmax(scores, -1)
        print(f"softmax over each row   {tuple(weights.shape)}  H x T x T   rows add to 1: {bool(torch.allclose(weights.sum(-1), torch.ones(H_, T), atol=1e-4))}")
        ctx = weights @ Vh
        print(f"weights.V_h             {tuple(ctx.shape)}   H x T x d_h")
        ctx = ctx.transpose(0, 1).reshape(T, d_)
        print(f"glue the heads together {tuple(ctx.shape)}      T x d")
        mixed = ctx @ W(ao.dense) + ao.dense.bias
        print(f"mix with W_O            {tuple(mixed.shape)}      T x d")
        mine = ao.LayerNorm(mixed + X)
        print(f"add the shortcut X, normalise -> {tuple(mine.shape)}  T x d   (the output of the attention block)")
        res = L0.attention(X.unsqueeze(0).to(device)); theirs = (res[0] if isinstance(res, tuple) else res)[0].float()
        print(f"\\nmatches the library's attention block of layer 0: {torch.allclose(mine.cpu(), theirs.cpu(), atol=1e-4)}  (max difference {float((mine.cpu() - theirs.cpu()).abs().max()):.2e})")
        Wq_view = W(sa.query).abs().cpu()
    # W_Q as a picture: block-average to ~96 x 96; vertical lines mark the H head slices (each head owns d/H columns)
    k = max(1, d_ // 96); img = torch.nn.functional.avg_pool2d(Wq_view[None, None], k)[0, 0].numpy()
    fig = go.Figure(go.Heatmap(z=img, colorscale="Blues", colorbar=dict(title="|weight|")))
    for h in range(1, H_): fig.add_vline(x=h * img.shape[1] / H_ - 0.5, line_color="#c0392b", line_width=1)
    fig.update_layout(title=f"W_Q of layer 0 (|weights|, block-averaged): the {H_} red-bounded column stripes are the {H_} heads", xaxis_title="output column (head h owns stripe h)", yaxis_title="input row (the d numbers of a token)", height=520, yaxis=dict(autorange="reversed"))
    show(fig, "02l_Wq_by_head")
except Exception as err:
    print("matrix walk-through skipped for this model family:", repr(err)[:200])
""")

md("""
### The same view for the heads, and for LoRA
* **Heads:** a batch of `B` notes gives a summary matrix `S` of shape `B x d`. The yes/no head is one matrix product, `S · W_bin + b`, with `W_bin` of shape `d x 216`: **column `j` of `W_bin` is judge `j`** (the 768 or 384 weights from section 5). The output is `B x 216` scores, one row per note.
* **LoRA:** instead of changing `W` (`d x d`) we learn `A` (`d x r`) and `B` (`r x d`) and add their product `A·B`: a `d x d` matrix that has **rank at most `r`**: every one of its `d` rows is a mix of the same `r` rows. Below, such a product next to a full random matrix: how many independent directions does each really contain?
""")

code("""
r = LORA_RANK
Aa, Bb = torch.randn(d_, r), torch.randn(r, d_)
low, full_ = (Aa @ Bb), torch.randn(d_, d_)
sv_low, sv_full = torch.linalg.svdvals(low)[:24].numpy(), torch.linalg.svdvals(full_)[:24].numpy()
fig = make_subplots(rows=1, cols=3, subplot_titles=(f"A ({d_} x {r}) times B ({r} x {d_})", f"= A.B: still {d_} x {d_}, but rank {r}", "Singular values (strength of each independent direction)"), column_widths=[0.25, 0.3, 0.45])
fig.add_trace(go.Heatmap(z=Aa.numpy(), colorscale="RdBu", zmid=0, showscale=False), row=1, col=1)
fig.add_trace(go.Heatmap(z=low[:96, :96].numpy(), colorscale="RdBu", zmid=0, showscale=False), row=1, col=2)
fig.add_trace(go.Bar(x=list(range(1, 25)), y=sv_low, name=f"A.B (rank {r})", marker_color="#d6b02e"), row=1, col=3)
fig.add_trace(go.Bar(x=list(range(1, 25)), y=sv_full, name="a full random matrix", marker_color="#b0b0b0"), row=1, col=3)
fig.update_yaxes(autorange="reversed", row=1, col=1); fig.update_yaxes(autorange="reversed", row=1, col=2)
fig.update_xaxes(title_text="direction number", row=1, col=3); fig.update_layout(height=400, barmode="group", title="LoRA as matrices: a big matrix that only has r independent directions", legend=dict(orientation="h", y=-0.2))
show(fig, "02m_lora_rank")
print(f"entries needed to describe A and B: {d_*r + r*d_:,}  vs the full {d_}x{d_} matrix: {d_*d_:,}  ->  {100*(d_*r + r*d_)/(d_*d_):.1f}%.   Numerical rank of A.B: {int((torch.linalg.svdvals(low) > 1e-3).sum())} (of {d_})")
""")

md("""
### One judge, by hand
We take one note and one question, and compute the head's answer **ourselves** with a dot product. It must equal what the model outputs.
Right now the heads are untrained (random weights), so the answer is meaningless; we repeat the same cell after training in section 8 to see the same judge, now knowing its job.
""")

code("""
DEMO_Q = "nose__oak__vanilla" if "nose__oak__vanilla" in schema["binary"] else schema["binary"][0]
j = schema["binary"].index(DEMO_Q)
demo_note = df.description.iloc[int(np.argmax(B[:, j]))]                  # a note where Jev is most sure the answer is yes

def one_judge(note):
    e = tok([note], padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        summary = model.embed(e["input_ids"].to(device), e["attention_mask"].to(device))[0].float().cpu()     # the 768 numbers
        w, b = model.binary_head.weight[j].detach().float().cpu(), model.binary_head.bias[j].detach().float().cpu()  # this judge's row
        by_hand = torch.sigmoid(summary @ w + b).item()                                                        # dot product + bias -> probability
        by_model = torch.sigmoid(model(e["input_ids"].to(device), e["attention_mask"].to(device))[0][0, j].float()).item()
    return summary, by_hand, by_model

summary, by_hand, by_model = one_judge(demo_note)
print("note:", demo_note[:160], "...\\n")
print(f"the note's summary: {len(summary)} numbers, e.g. {[round(v, 2) for v in summary[:5].tolist()]} ...")
print(f"judge for '{DEMO_Q}': {len(summary)} weights x summary + bias -> probability {by_hand:.4f}   (the model says {by_model:.4f})")
print(f"Jev's answer for this note: {B[int(np.argmax(B[:, j])), j]:.2f}")
""")

md("""
## 5d. All the models we could train, side by side (to scale)
Section 5c drew the matrices of **one** body. Here is the same picture for **every body in the zoo**, on **one common scale** (a rectangle that is twice as wide is twice as wide in every panel), so you can *see* what "more capacity" means.

How to read the figure: each panel is one model. It shows the weight matrices of **one layer** (the model repeats that set of matrices once per layer: 6, 12, 22 or 24 times) and the two output heads we add on top.
Yellow = attention (`W_Q, W_K, W_V, W_O`, square `d x d`), blue = feed-forward (`W_1` wide, `W_2` tall), green = our two heads (`W_bin`: `d x 216`, `W_grp`: `d x 76`).
Two things to notice: (1) the heads are **thin slivers** next to any body; (2) going from MiniLM to roberta-large the matrices grow in *both* directions at once, so the area grows with the **square** of the width.

The **word table** (one row per vocabulary piece, `vocab x d`) is not drawn here because it would be taller than the page: it gets its own bar chart after the figure.
""")

code("""
# The architecture facts of each candidate body (from the published model configurations).
# columns: name, layers, width d, attention heads, feed-forward columns of W_1, feed-forward rows of W_2, vocabulary, positions, published size in millions
SPEC = [
    ("all-MiniLM-L6-v2", 6,  384, 12, 1536, 1536, 30522, 512, 22.7),
    ("distilbert-base",  6,  768, 12, 3072, 3072, 30522, 512, 66),
    ("bert-base",        12, 768, 12, 3072, 3072, 30522, 512, 110),
    ("roberta-base",     12, 768, 12, 3072, 3072, 50265, 514, 125),
    ("deberta-v3-base",  12, 768, 12, 3072, 3072, 128100, 512, 184),     # 128k-piece vocabulary: the word table alone is 98 M
    ("ModernBERT-base",  22, 768, 12, 2304, 1152, 50368, 0,   149),      # gated feed-forward: W_1 is two d x 1152 matrices side by side; positions are rotary (no table)
    ("roberta-large",    24, 1024, 16, 4096, 4096, 50265, 514, 355),
]
ARCH = pd.DataFrame(SPEC, columns=["model", "layers", "d", "heads", "ff1", "ff2", "vocab", "positions", "published_M"])
ARCH["d_h"] = ARCH.d // ARCH.heads
ARCH["word table"] = ARCH.vocab * ARCH.d
ARCH["position table"] = ARCH.positions * ARCH.d
ARCH["attention (all layers)"] = ARCH.layers * 4 * ARCH.d * ARCH.d
ARCH["feed-forward (all layers)"] = ARCH.layers * ARCH.d * (ARCH.ff1 + ARCH.ff2)
ARCH["biases and norms"] = ARCH.layers * (8 * ARCH.d + ARCH.ff1 + ARCH.ff2)
ARCH["our two heads"] = ARCH.d * (216 + 76) + 216 + 76
PARTS = ["word table", "position table", "attention (all layers)", "feed-forward (all layers)", "biases and norms", "our two heads"]
ARCH["body_M"] = ARCH[PARTS[:5]].sum(axis=1) / 1e6

tab = pd.DataFrame({
    "model": ARCH.model, "layers": ARCH.layers, "width d": ARCH.d, "heads x d_h": ARCH.heads.astype(str) + " x " + ARCH.d_h.astype(str),
    "attention matrices (per layer)": ["4 x " + f"{d}x{d}" for d in ARCH.d],
    "feed-forward matrices (per layer)": [f"{d}x{a} and {b}x{d}" for d, a, b in zip(ARCH.d, ARCH.ff1, ARCH.ff2)],
    "word table": [f"{v:,} x {d}" for v, d in zip(ARCH.vocab, ARCH.d)],
    "body, computed (M)": ARCH.body_M.round(1), "published (M)": ARCH.published_M, "word table share": (100 * ARCH["word table"] / (ARCH.body_M * 1e6)).round(0).astype(int).astype(str) + " %",
})
cfg_ = model.encoder.config
same = ARCH[(ARCH.layers == getattr(cfg_, "num_hidden_layers", -1)) & (ARCH.d == cfg_.hidden_size) & (ARCH.vocab == cfg_.vocab_size)]
print(f"your MODEL_NAME ({MODEL_NAME}): {cfg_.num_hidden_layers} layers, d={cfg_.hidden_size}, vocabulary {cfg_.vocab_size:,}  ->  " + (f"matches the table row '{same.model.iloc[0]}'" if len(same) else "not one of the seven in the table (that is fine, the notebook still works)"))
print("(computed sizes come from the shapes alone: the small gap to the published figure is biases, norms and details such as DeBERTa's relative-position matrices)\\n")
tab
""")

code("""
# ONE common scale: every panel has the same data range, and the axes are locked to 1:1 so a square is drawn as a square.
COL = {"attention": "#f3e5ab", "feed-forward": "#bcd9ee", "heads": "#a8d5a2"}
GAP = 150
def layer_rects(d, ff1, ff2):
    S = 2 * d + GAP; xw2 = max(ff1, S) + GAP; xh = xw2 + d + GAP                    # (name, x, y, width, height, colour): rows = inputs (height), columns = outputs (width)
    return [("W_Q", 0, 0, d, d, COL["attention"]), ("W_K", d + GAP, 0, d, d, COL["attention"]), ("W_V", 0, d + GAP, d, d, COL["attention"]), ("W_O", d + GAP, d + GAP, d, d, COL["attention"]),
            ("W_1", 0, S + GAP, ff1, d, COL["feed-forward"]), ("W_2", xw2, 0, d, ff2, COL["feed-forward"]),
            ("W_bin", xh, 0, 216, d, COL["heads"]), ("W_grp", xh + 216 + GAP / 2, 0, 76, d, COL["heads"])]

titles = [f"<b>{r.model}</b><br>{r.layers} layers, d = {r.d}, {r.body_M:.0f} M" for r in ARCH.itertuples()] + ["How to read"]
fig = make_subplots(rows=4, cols=2, subplot_titles=titles, vertical_spacing=0.09, horizontal_spacing=0.04)
for k, r in enumerate(ARCH.itertuples()):
    row, col = k // 2 + 1, k % 2 + 1
    rects = layer_rects(int(r.d), int(r.ff1), int(r.ff2)); cx, cy, tx = [], [], []
    for nm, x, y, w, h, c in rects:
        fig.add_shape(type="rect", x0=x, x1=x + w, y0=y, y1=y + h, fillcolor=c, line=dict(color="#444", width=1), row=row, col=col)
        cx.append(x + w / 2); cy.append(y + h / 2); tx.append(f"{nm}: {h} x {w} = {h*w:,} numbers")
        if w > 600 and h > 600: fig.add_annotation(x=x + w / 2, y=y + h / 2, text=f"<b>{nm}</b><br>{h}x{w}", showarrow=False, font=dict(size=10), row=row, col=col)
    fig.add_trace(go.Scatter(x=cx, y=cy, mode="markers", marker=dict(size=14, opacity=0), text=tx, hoverinfo="text", showlegend=False), row=row, col=col)
for k in range(1, 9):
    xa, ya = ("x" if k == 1 else f"x{k}"), ("y" if k == 1 else f"y{k}")
    fig.update_xaxes(range=[-50, 5900], visible=False, constrain="domain", row=(k - 1) // 2 + 1, col=(k - 1) % 2 + 1)
    fig.update_yaxes(range=[4200, -100], visible=False, scaleanchor=xa, scaleratio=1, constrain="domain", row=(k - 1) // 2 + 1, col=(k - 1) % 2 + 1)
for i, (label, c) in enumerate([("attention: W_Q, W_K, W_V, W_O", COL["attention"]), ("feed-forward: W_1 (wide), W_2 (tall)", COL["feed-forward"]), ("our heads: W_bin, W_grp", COL["heads"])]):
    fig.add_shape(type="rect", x0=300, x1=700, y0=500 + i * 700, y1=900 + i * 700, fillcolor=c, line=dict(color="#444"), row=4, col=2)
    fig.add_annotation(x=850, y=700 + i * 700, text=label, showarrow=False, xanchor="left", font=dict(size=12), row=4, col=2)
fig.add_annotation(x=300, y=3000, text="Same scale in every panel.<br>Each model repeats the yellow + blue<br>set once per layer (6 to 24 times).<br>The word table is not drawn (next figure).", showarrow=False, xanchor="left", yanchor="top", font=dict(size=11), row=4, col=2)
fig.update_layout(height=1500, width=1000, title=dict(text="The weight matrices of one layer of every model in the zoo, to one common scale", y=0.985), margin=dict(t=130, l=20, r=20, b=20))
show(fig, "02n_models_to_scale")
""")

md("""
### Where do the numbers live? The word table versus the layers
Same models, now as **stacked bars of parameters** (millions). The *layers* (attention + feed-forward) are the "thinking"; the **word table** is a dictionary with one row per vocabulary piece.
A small model with a big vocabulary spends most of its numbers on the dictionary (MiniLM: more than half; DeBERTa-v3: more than half too, 98 M of 184 M), which is why parameter count alone is a poor measure of "how smart".
""")

code("""
fig = go.Figure()
colors_ = ["#8d6e63", "#bcaaa4", "#e0b93a", "#5b9bd5", "#9e9e9e", "#4caf50"]
for part, c in zip(PARTS, colors_):
    fig.add_trace(go.Bar(y=ARCH.model, x=ARCH[part] / 1e6, name=part, orientation="h", marker_color=c, hovertemplate="%{y}<br>" + part + ": %{x:.1f} M<extra></extra>"))
fig.update_layout(barmode="stack", height=430, title="Parameters by part (millions): the heads we add are the tiny green sliver", xaxis_title="millions of parameters", yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=-0.25))
show(fig, "02o_param_parts")
""")

md("""
### Will it fit in the GPU? A back-of-the-envelope memory estimate
Training needs memory for the **weights**, and (for every number that is *trained*) a **gradient** and **two AdamW statistics**. With 32-bit numbers (4 bytes each):

| Strategy | Bytes per body parameter | Why |
|---|---|---|
| Full fine-tuning | about **16** | 4 weight + 4 gradient + 8 for AdamW's two running averages |
| LoRA | about **4** (+ 16 for the tiny adapters) | the body is only read; gradient and statistics exist for the adapters only |
| Frozen body | about **4** (+ 16 for the heads) | the body is only read |

This counts **only** weights + gradient + optimiser. The **activations** (the intermediate results kept for back-propagation) come on top and grow with batch size x tokens; they are often several GB. Section 7b prints the *measured* peak for the model you picked.
""")

code("""
n_body = ARCH.body_M * 1e6; n_heads_ = ARCH["our two heads"]; r_ = LORA_RANK
lora_params = ARCH.layers * 2 * (2 * ARCH.d * r_)                                       # A and B for W_Q and W_V in every layer
est = pd.DataFrame({"model": ARCH.model,
    "Full fine-tuning": (16 * (n_body + n_heads_)) / 1e9,
    "LoRA": (4 * n_body + 16 * (lora_params + n_heads_)) / 1e9,
    "Frozen body": (4 * n_body + 16 * n_heads_) / 1e9})
fig = go.Figure()
for c_, col_ in zip(["Full fine-tuning", "LoRA", "Frozen body"], ["#c0392b", "#d6b02e", "#2375ae"]):
    fig.add_trace(go.Bar(x=est.model, y=est[c_], name=c_, marker_color=col_, hovertemplate="%{x}<br>" + c_ + ": %{y:.1f} GB<extra></extra>"))
fig.add_hline(y=16, line_dash="dash", line_color="#444", annotation_text="16 GB (RTX 5060 Ti)", annotation_position="top left")
fig.add_hline(y=15, line_dash="dot", line_color="#888", annotation_text="15 GB (Colab T4)", annotation_position="bottom left")
fig.update_layout(barmode="group", height=420, title="Memory for weights + gradients + optimiser (activations come on top)", yaxis_title="GB")
show(fig, "02p_memory_estimate")
print(est.round(2).to_string(index=False))
""")

# --------------------------------------------------------------------------------------------------------------------
md("""
## 6. How the student is graded while it learns (the loss)

The loss is a single number saying *how far the student's answers are from Jev's*. Training pushes it down.

* **Yes/no questions**: *binary cross-entropy* against Jev's probability. We use Jev's **soft** answer (0.83) instead of rounding to 1: it keeps the information that Jev was only fairly sure.
* **Multi-level questions**: *cross-entropy* between the student's level probabilities (after a *softmax*, which makes them add up to 1) and Jev's. This is the same as minimising the KL divergence between the two distributions.

The two parts are added; `BIN_WEIGHT` says how much the 216 rare yes/no questions count compared with the 15 multi-level ones.
""")

code("""
def wset_loss(bin_logits, grp_logits, y_bin, y_grp):
    loss_bin = F.binary_cross_entropy_with_logits(bin_logits.float(), y_bin)          # average over notes and questions
    loss_grp = 0.0
    for s, e in zip(slices[:-1], slices[1:]):                                          # one softmax per question
        log_p = F.log_softmax(grp_logits[:, s:e].float(), dim=-1)
        loss_grp = loss_grp - (y_grp[:, s:e] * log_p).sum(-1).mean()
    loss_grp = loss_grp / len(GROUPS)
    return BIN_WEIGHT * loss_bin + loss_grp, loss_bin.detach(), loss_grp.detach()
""")

md("""
### Two heads, one body: both pull on the same weights
A direct look at "multi-headed" training. On one batch, we compute the gradient (the direction each weight would like to move) of **one body weight matrix**, separately from the yes/no loss and from the level loss, and compare the two directions.
A cosine near 0 means the heads want unrelated things; positive means they agree; negative, they pull against each other. The real update uses their (weighted) sum.
""")

code("""
model.train()
ids, mask, yb, yg = [t.to(device) for t in next(iter(DataLoader(TensorDataset(enc["input_ids"][train_idx[:64]], enc["attention_mask"][train_idx[:64]], Y_bin[train_idx[:64]], Y_grp[train_idx[:64]]), batch_size=32)))]
probe_name, probe = [(n, p) for n, p in model.encoder.named_parameters() if p.ndim == 2 and component(n) == "attention"][-1]    # an attention weight in the top layer
lb, lg = model(ids, mask)
l_bin = F.binary_cross_entropy_with_logits(lb.float(), yb)
l_grp = sum(-(yg[:, s:e] * F.log_softmax(lg[:, s:e].float(), -1)).sum(-1).mean() for s, e in zip(slices[:-1], slices[1:])) / len(GROUPS)
g_bin = torch.autograd.grad(l_bin, probe, retain_graph=True)[0].flatten().float().cpu()
g_grp = torch.autograd.grad(l_grp, probe)[0].flatten().float().cpu()
print(f"body weight examined: {probe_name} ({probe.numel():,} numbers)")
print(f"pull from the yes/no head: size {g_bin.norm():.4f} | pull from the level head: size {g_grp.norm():.4f} | agreement (cosine) {float(torch.cosine_similarity(g_bin, g_grp, dim=0)):+.3f}")
model.zero_grad(set_to_none=True); model.eval()
""")

md("""
## 6b. Activation functions and back-propagation: what they are and where they act
So far everything was matrix multiplication. A network made *only* of matrix products would be useless: a product of matrices is itself just one matrix, so ten layers would be no cleverer than one. What lets layers stack up into something powerful is a **non-linear step** between the products: an **activation function**, applied to every number separately (sigmoid, GELU, softmax...). **An activation function has no weights of its own**: it is a fixed little formula.

**Back-propagation** is how the student learns. After the *forward pass* (note in, answers out, loss computed), the *backward pass* runs the other way: starting from the loss it uses the chain rule to find, for **every trainable number**, *how much would the loss change if this number changed a little?* That is its **gradient**. The optimiser (AdamW) then nudges each number against its gradient. Each step backwards multiplies by the **slope** of the step it goes through; this is where the activation functions matter again, because their slopes are part of the product.

So the two ideas meet like this: *weights are what learns; activations are fixed formulas between the weights; back-propagation carries the error signal through both and gives the weights their gradient.*

### Where is each used in this notebook?

| Part | Non-linear function? | Has numbers that learn? | Receives a gradient: full fine-tuning | frozen body | LoRA |
|---|---|---|---|---|---|
| **Tokenizer** (words to pieces to ids) | **none**: a fixed rulebook and a lookup | no | no | no | no |
| **Word table E** (id to a row of `d` numbers) | none (a row lookup) | **yes**: the rows | yes, **only the rows of the pieces that occur in the batch** | no | no |
| **Attention** (`W_Q, W_K, W_V, W_O`) | **softmax** over the tokens (weights add to 1) | yes | yes | no | no (the matrices are frozen; LoRA adds its own) |
| **Feed-forward** (`W_1`, `W_2`) | **GELU** between them (ModernBERT: GeGLU, a gated GELU) | yes | yes | no | no |
| **Layer norms** | normalisation (not an activation) | a small scale and shift | yes | no | no |
| **Pooling** (average of the tokens) | none | no | (passes the signal) | (passes it, nothing to update) | passes it |
| **Yes/no head** `W_bin` | **sigmoid** (one probability per question) | yes | yes | yes | yes |
| **Level head** `W_grp` | **softmax** per question (levels add to 1) | yes | yes | yes | yes |
| **LoRA adapters** `A`, `B` | **none**: they are plain matrices, `B·(A·x)`, with no activation inside | yes (the only body numbers that move) | n/a | n/a | **yes** |
| **Loss** | log of the probability (BCE, cross-entropy) | no | the start of the backward pass | the start | the start |

Two details worth remembering. The sigmoid and softmax of the heads are *folded into the loss* for numerical stability (`binary_cross_entropy_with_logits`); the model itself outputs raw **logits** and we apply the sigmoid when we want a probability.
And "frozen" means *"do not update"*, **not** *"do not pass the signal"*: with LoRA, the error signal still has to travel back **through** the frozen layers to reach the adapters (the next sections measure this).
""")

code("""
# What do the activation functions look like, and what are their slopes? (the slope is what back-propagation multiplies by)
x = np.linspace(-5, 5, 400); xt = torch.tensor(x, dtype=torch.float32, requires_grad=True)
funcs = {"sigmoid (the yes/no head)": torch.sigmoid, "GELU (inside every layer)": F.gelu, "ReLU (older networks)": torch.relu, "tanh": torch.tanh}
cols_ = ["#2375ae", "#c0392b", "#888888", "#8e44ad"]
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10, subplot_titles=("the function: what it does to a number", "its slope: what the backward pass multiplies by"))
for (nm, f), c in zip(funcs.items(), cols_):
    y = f(xt); g, = torch.autograd.grad(y.sum(), xt)                                  # autograd finds the slope for us
    fig.add_trace(go.Scatter(x=x, y=y.detach().numpy(), name=nm, legendgroup=nm, line=dict(color=c)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=g.numpy(), name=nm, legendgroup=nm, showlegend=False, line=dict(color=c)), row=2, col=1)
fig.update_xaxes(title_text="input number", row=2, col=1); fig.update_layout(height=620, title="Activation functions and their slopes", legend=dict(orientation="h", y=-0.15))
show(fig, "06a_activations")
print("The sigmoid slope is at most 0.25 (at 0) and almost 0 far from 0: a unit that is already very sure learns very slowly (saturation). GELU and ReLU keep a slope near 1 for positive inputs.")
""")

code("""
# The softmax: scores -> weights that add to 1. It is what attention uses over tokens and what the level head uses over levels.
scores = torch.tensor([3.0, 1.5, 0.2, -1.0, 0.5])
labels_ = ["the", "ripe", "black-currant", "and", "vanilla"]
sm = torch.softmax(scores, 0); sm_hot = torch.softmax(scores * 3, 0)
fig = make_subplots(rows=1, cols=3, subplot_titles=("raw scores (logits)", "softmax: weights add to 1", "scores x 3: sharper"))
fig.add_trace(go.Bar(x=labels_, y=scores.numpy(), marker_color="#9e9e9e", showlegend=False), row=1, col=1)
fig.add_trace(go.Bar(x=labels_, y=sm.numpy(), marker_color="#e0b93a", text=[f"{v:.2f}" for v in sm], textposition="outside", showlegend=False), row=1, col=2)
fig.add_trace(go.Bar(x=labels_, y=sm_hot.numpy(), marker_color="#c0392b", text=[f"{v:.2f}" for v in sm_hot], textposition="outside", showlegend=False), row=1, col=3)
fig.update_layout(height=360, title=f"Softmax on 5 scores. The sum of the weights is {float(sm.sum()):.1f}")
show(fig, "06b_softmax")
z = torch.tensor(1.3)
print(f"sigmoid is a softmax over two scores [z, 0]:  sigmoid(1.3) = {float(torch.sigmoid(z)):.4f}   softmax([1.3, 0])[0] = {float(torch.softmax(torch.stack([z, torch.tensor(0.0)]), 0)[0]):.4f}")
""")

code("""
# Open the real model and list what it is made of: which pieces carry learnable numbers, and which are just formulas?
from collections import defaultdict
agg = defaultdict(lambda: [0, 0, ""])
for n_, m_ in model.named_modules():
    if n_ == "" or type(m_).__name__ == type(model.encoder).__name__ or isinstance(m_, (nn.ModuleList, nn.Sequential)): continue
    a_ = agg[type(m_).__name__]; a_[0] += 1; a_[1] += sum(p_.numel() for p_ in m_.parameters(recurse=False)); a_[2] = a_[2] or n_
def what_is(c):
    c_l = c.lower()
    if c_l == "linear": return "a matrix (learned): the building block of everything above"
    if "embedding" in c_l: return "a lookup table (learned)"
    if "norm" in c_l: return "normalisation (a small learned scale and shift)"
    if "dropout" in c_l: return "randomly hides numbers while training; no parameters"
    if "act" in c_l or "gelu" in c_l or "relu" in c_l or "tanh" in c_l: return "ACTIVATION FUNCTION: a fixed formula; no parameters"
    return "container / other"
mods = pd.DataFrame([(c, v[0], v[1], what_is(c), v[2]) for c, v in agg.items() if v[1] > 0 or what_is(c) != "container / other"], columns=["module type", "how many", "own learnable numbers", "what it is", "first example"]).sort_values("own learnable numbers", ascending=False)
print("(softmax in the attention is written as a function call, not as a module, so it is not in this list; same for the sigmoid, which sits inside the loss)\\n")
mods
""")

md("""
### Back-propagation by hand, on one judge
One yes/no judge (section 5c): `z = w·s + b`, then `p = sigmoid(z)`, then the loss compares `p` with Jev's answer `y`. Going **backwards** with the chain rule, the slope of the sigmoid and the slope of the log in the loss **cancel** and leave something wonderfully simple:

* `dLoss/dz = p - y` (the prediction minus the answer: "how wrong, and in which direction")
* `dLoss/dw = (p - y) · s` (each weight moves in proportion to the number it multiplied)
* `dLoss/db = p - y`
* `dLoss/ds = (p - y) · w` (this is the signal that is **handed down into the body**, to be carried further back through the layers)

We compute these four lines ourselves, and then ask PyTorch's automatic back-propagation for the same thing. They must agree.
""")

code("""
s_ = summary.clone().float().requires_grad_(True)                                    # the note's summary vector (from the 'one judge' cell)
w_ = model.binary_head.weight[j].detach().float().cpu().clone().requires_grad_(True)  # this judge's weights
b_ = model.binary_head.bias[j].detach().float().cpu().clone().requires_grad_(True)
y_ = float(B[int(np.argmax(B[:, j])), j])                                              # Jev's answer for this note
z_ = s_ @ w_ + b_; p_ = torch.sigmoid(z_); loss_ = F.binary_cross_entropy(p_, torch.tensor(y_)); loss_.backward()   # forward, then backward
err = float(p_) - y_
print(f"FORWARD   z = w.s + b = {float(z_):+.4f}   p = sigmoid(z) = {float(p_):.4f}   Jev says y = {y_:.2f}   loss = {float(loss_):.4f}")
print(f"BACKWARD  by hand: p - y = {err:+.4f}")
print(f"  dLoss/dw: by hand {(err * summary.float())[:3].tolist()} ...   autograd {w_.grad[:3].tolist()} ...  equal: {torch.allclose(w_.grad, err * summary.float(), atol=1e-6)}")
print(f"  dLoss/db: by hand {err:+.4f}   autograd {float(b_.grad):+.4f}")
print(f"  dLoss/ds: by hand (p-y).w   equal to autograd: {torch.allclose(s_.grad, err * w_.detach(), atol=1e-6)}   <- this is the signal that goes back into the body")
print(f"\\nThe judge would change its {len(w_)} weights by (learning rate) x -(p - y) x s. If p is already close to y, the error is ~0 and the weights hardly move.")
""")

code("""
mermaid('''
flowchart LR
  tk["tokenizer<br/>fixed rulebook, a lookup<br/>NO activation, NO gradient"] --> E["word table E<br/>row lookup<br/>rows learn"]
  E --> ATT["attention<br/>W_Q W_K W_V W_O<br/>softmax over tokens"]
  ATT --> FF["feed-forward<br/>W_1, GELU, W_2"]
  FF --> POOL["average the tokens<br/>no weights"]
  POOL --> H1["yes/no head<br/>W_bin, sigmoid"]
  POOL --> H2["level head<br/>W_grp, softmax"]
  H1 --> L["loss: compare with Jev"]
  H2 --> L
  L -. "1. error: p - y" .-> H1
  L -. "1. error" .-> H2
  H1 -. "2. gradient to the summary" .-> POOL
  H2 -.-> POOL
  POOL -. "3. back through every layer<br/>(multiply by each slope)" .-> FF
  FF -.-> ATT
  ATT -. "4. only the rows used" .-> E
  classDef fixed fill:#e0e0e0,stroke:#888
  classDef learn fill:#d9f0f0,stroke:#009999
  classDef act fill:#f3e5ab,stroke:#b58900
  class tk,POOL fixed
  class E,H1,H2 learn
  class ATT,FF act
''', "06c_forward_backward", 470)
""")

md("""
### Where does the gradient actually arrive? (measured on the real model)
Now the real thing: one batch of 32 notes goes forward through the untrained student, we back-propagate, and look at **how large the gradient is** in each part of the model and in each layer. A part that gets no gradient never learns.
Look for two things: the gradient is usually **strongest near the heads** (the top layers) and weaker as it travels back; and the **word table** only receives gradient in the rows of the pieces that appeared in the batch.
""")

code("""
def part_of(name):
    if name.startswith(("binary_head", "group_head")): return "output heads"
    if name.endswith((".A", ".B")): return "LoRA adapters"
    c = component(name)
    return {"word & position tables (embeddings)": "word table", "other (pooler...)": "other (pooler)"}.get(c, c)

PROBE_IDX = train_idx[:32]
probe_batch = [enc["input_ids"][PROBE_IDX], enc["attention_mask"][PROBE_IDX], Y_bin[PROBE_IDX], Y_grp[PROBE_IDX]]
def backward_once(m):
    m.zero_grad(set_to_none=True); ids_, mask_, yb_, yg_ = [t.to(device) for t in probe_batch]
    t0 = time.time(); lb_, lg_ = m(ids_, mask_); loss = wset_loss(lb_, lg_, yb_, yg_)[0]; loss.backward()
    if device.type == "cuda": torch.cuda.synchronize()
    secs_ = time.time() - t0
    rows = [(n, p.numel(), p.requires_grad, None if p.grad is None else float(p.grad.float().norm()), layer_index(n)) for n, p in m.named_parameters()]
    return pd.DataFrame(rows, columns=["name", "n", "trainable", "grad_norm", "layer"]), secs_

was_training = model.training; model.eval()
gdf, _ = backward_once(model)
gdf["part"] = gdf.name.map(part_of)
gdf["sq"] = gdf.grad_norm.fillna(0) ** 2
byp = (gdf.groupby("part").sq.sum() ** 0.5).sort_values()
bylayer = gdf[gdf.layer.notna()].pivot_table(index="layer", columns="part", values="sq", aggfunc="sum") ** 0.5
fig = make_subplots(rows=1, cols=2, column_widths=[0.42, 0.58], subplot_titles=("by part of the model", "by layer (0 = words side, last = heads side)"))
fig.add_trace(go.Bar(y=byp.index, x=byp.values, orientation="h", marker_color="#5b9bd5", showlegend=False), row=1, col=1)
for pt, c in zip(["attention", "feed-forward", "layer norms"], ["#e0b93a", "#5b9bd5", "#9e9e9e"]):
    if pt in bylayer: fig.add_trace(go.Scatter(x=bylayer.index, y=bylayer[pt], mode="lines+markers", name=pt, line=dict(color=c)), row=1, col=2)
fig.update_xaxes(title_text="gradient norm", row=1, col=1); fig.update_xaxes(title_text="layer", row=1, col=2); fig.update_yaxes(type="log", title_text="gradient norm", row=1, col=2)
fig.update_layout(height=450, margin=dict(t=110), title="Gradient size measured: one batch, untrained student", legend=dict(orientation="h", y=-0.2))
show(fig, "06d_gradient_arrival")
we_ = [p for n, p in model.encoder.named_parameters() if "word_embeddings" in n][0]
touched = int((we_.grad.abs().sum(1) > 0).sum())
print(f"word table: {touched:,} of {we_.shape[0]:,} rows received a gradient ({100*touched/we_.shape[0]:.2f} %): exactly the distinct word pieces of these 32 notes. The other {we_.shape[0]-touched:,} rows did not move at all.")
print("tokenizer: it has no numbers to learn (a fixed vocabulary and rules), so there is nothing to give a gradient to.")
model.zero_grad(set_to_none=True); _ = model.train(was_training)
""")

# --------------------------------------------------------------------------------------------------------------------
md("""
## 7. Training

Each **step**: take a batch of notes, let the student answer, measure the loss, nudge every weight a little in the direction that lowers it (*AdamW*).
After every **epoch** (a full pass over the training notes) we **stop learning and take a mock exam on the validation notes**: we record the validation loss *and* the quality metrics, so we can watch the student improve.
We also **save a checkpoint** (free Colab sessions can disconnect; just run the notebook again and training resumes).

### First, a yardstick: the word-counting baseline
Before training anything fancy, a simple model: TF-IDF (how characteristic each word is of a note) plus a linear model. If our fine-tuned student can't beat this, it isn't worth the trouble.
It is trained on the training notes only, and scored on validation (to draw the dashed line in the plots) and on test (for the final comparison).
""")

code("""
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
Xtr = tfidf.fit_transform(df.description.iloc[train_idx])
base = Ridge(alpha=1.0).fit(Xtr, Y_bin[train_idx].numpy())
P_base_val  = base.predict(tfidf.transform(df.description.iloc[val_idx]))
P_base_test = base.predict(tfidf.transform(df.description.iloc[test_idx]))

T_bin_val, T_grp_val = Y_bin[val_idx].numpy(), Y_grp[val_idx].numpy()          # Jev's answers on the validation notes
T_bin, T_grp = Y_bin[test_idx].numpy(), Y_grp[test_idx].numpy()                # ... and on the test notes (not used until section 8)
CLUSTER = np.array([q.split("__")[0] + " / " + q.split("__")[1] for q in schema["binary"]])   # e.g. "nose / oak"
print("baseline ready")
""")

md("""
### The report card we will print after every epoch
Four numbers, all measured against Jev's answers on the **validation** notes:

* **mean AUC** over the yes/no questions that have enough examples (1.0 perfect, 0.5 guessing),
* **level exact**: how often the student picks the same level as Jev (tannin, body, colour...),
* **level within one**: for the ordered scales, how often it is at most one step away,
* **per-cluster AUC**: the same AUC averaged by aroma family (oak, red fruit, ...), to see *what is learned first*.
""")

code("""
def auc_per_question(P, T, min_pos):
    out = np.full(P.shape[1], np.nan)
    for j in range(P.shape[1]):
        t = T[:, j] >= 0.5
        if t.sum() >= min_pos and (~t).sum() >= min_pos: out[j] = roc_auc_score(t, P[:, j])
    return out

def level_scores(Pg, Tg):
    ex, w1 = [], []
    for q, s, e in zip(GROUPS, slices[:-1], slices[1:]):
        t, p = Tg[:, s:e].argmax(1), Pg[:, s:e].argmax(1); ex.append((t == p).mean())
        if schema["kind"][q] == "score": w1.append((abs(t - p) <= 1).mean())
    return float(np.mean(ex)), float(np.mean(w1))

@torch.no_grad()
def predict(dl):
    model.eval(); P_bin, P_grp = [], []
    for ids, mask, _, _ in dl:
        with autocast(): lb, lg = model(ids.to(device), mask.to(device))
        P_bin.append(torch.sigmoid(lb.float()).cpu())
        P_grp.append(torch.cat([F.softmax(lg[:, s:e].float(), -1) for s, e in zip(slices[:-1], slices[1:])], 1).cpu())
    return torch.cat(P_bin).numpy(), torch.cat(P_grp).numpy()

def report_card(P_bin, P_grp):
    a = auc_per_question(P_bin, T_bin_val, MIN_POS); ex, w1 = level_scores(P_grp, T_grp_val)
    clu = pd.Series(a).groupby(CLUSTER).mean()
    return dict(auc=float(np.nanmean(a)), level_exact=ex, level_within_one=w1, clusters=clu.to_dict(), n_scored=int((~np.isnan(a)).sum()))

BASE_VAL_AUC = float(np.nanmean(auc_per_question(P_base_val, T_bin_val, MIN_POS)))
print(f"baseline mean AUC on validation: {BASE_VAL_AUC:.3f}")
""")

code("""
head_params = list(model.binary_head.parameters()) + list(model.group_head.parameters())
opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": LR_ENCODER},
                         {"params": head_params,                "lr": LR_HEADS}], weight_decay=0.01)
total_steps = EPOCHS * len(train_dl)
sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)   # warm up, then decay linearly
scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
CKPT = PACK_DIR / "checkpoint.pt"; BEST = PACK_DIR / "best_student.pt"
W0 = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}      # the starting weights, kept to measure what training changed

def autocast():
    return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None and device.type != "cpu")

@torch.no_grad()
def val_loss():
    model.eval(); tot = n = 0
    for ids, mask, yb, yg in val_dl:
        ids, mask, yb, yg = ids.to(device), mask.to(device), yb.to(device), yg.to(device)
        with autocast(): lb, lg = model(ids, mask)
        tot += wset_loss(lb, lg, yb, yg)[0].item() * len(ids); n += len(ids)
    return tot / n

start_epoch, best, history = 0, float("inf"), []
if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
RUN_ID = dict(v=2, model=MODEL_NAME, epochs=EPOCHS, batch=BATCH_SIZE, rows=len(df), max_len=MAX_LEN, split=(VAL_FRAC, TEST_FRAC))   # a checkpoint only counts if it came from the SAME settings
if CKPT.exists():                                              # resume after a disconnect
    ck = torch.load(CKPT, map_location="cpu")
    if ck.get("run_id") == RUN_ID:
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_epoch, best, history = ck["epoch"], ck["best"], ck["history"]
        print("resumed from epoch", start_epoch)
    else:
        print("found a checkpoint from a different run (other settings or data): ignoring it and starting fresh")

if not history:                                                # "epoch 0": the untrained student takes the mock exam first
    Pb, Pg = predict(val_dl)
    history.append(dict(epoch=0, train_loss=None, val_loss=val_loss(), steps=[], **report_card(Pb, Pg)))
    print(f"before any training: validation loss {history[0]['val_loss']:.4f} | mean AUC {history[0]['auc']:.3f} (0.5 = guessing) | baseline {BASE_VAL_AUC:.3f}")
""")

code("""
for epoch in range(start_epoch, EPOCHS):
    model.train(); t0 = time.time(); loss_sum = torch.zeros((), device=device); step_losses = []
    for step, (ids, mask, yb, yg) in enumerate(train_dl, 1):
        ids, mask, yb, yg = ids.to(device), mask.to(device), yb.to(device), yg.to(device)
        with autocast():
            lb, lg = model(ids, mask)
        loss, l_bin, l_grp = wset_loss(lb, lg, yb, yg)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)      # stop one bad batch from wrecking the weights
        scaler.step(opt); scaler.update(); sched.step()
        loss_sum += loss.detach()                               # keep the running total on the device: calling .item() every step would make it wait
        step_losses.append(loss.detach())                       # remember every step's loss for the plot (converted once, at the end of the epoch)
        if step % 50 == 0:
            print(f"epoch {epoch+1} step {step}/{len(train_dl)} avg loss {loss_sum.item() / step:.4f}", end="\\r")
    # ---- end of the epoch: stop learning, take the mock exam on the validation notes
    v = val_loss(); Pb, Pg = predict(val_dl); rc = report_card(Pb, Pg)
    history.append(dict(epoch=epoch + 1, train_loss=loss_sum.item() / len(train_dl), val_loss=v, steps=torch.stack(step_losses).float().cpu().tolist(),
                        secs=time.time() - t0, peak_gb=(torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else None), **rc))
    print(f"epoch {epoch+1}/{EPOCHS} | train loss {history[-1]['train_loss']:.4f} | validation loss {v:.4f} | mean AUC {rc['auc']:.3f} "
          f"(baseline {BASE_VAL_AUC:.3f}) | level exact {rc['level_exact']:.2f} | {time.time()-t0:.0f}s" + " " * 10)
    if v < best:
        best = v; torch.save({"model": model.state_dict()}, BEST)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "epoch": epoch + 1, "best": best, "history": history, "run_id": RUN_ID}, CKPT)

model.load_state_dict(torch.load(BEST, map_location="cpu")["model"]); model.to(device)       # go back to the best epoch
print("best validation loss:", round(best, 4))

# keep everything needed to redraw the training plots later, even if the notebook output is lost
try:
    json.dump(dict(run_id=RUN_ID, history=history, baseline_val_auc=BASE_VAL_AUC), open(PACK_DIR / "run_state.json", "w"))
    print("saved the training record to", PACK_DIR / "run_state.json")
except Exception as err: print("could not save run_state.json:", repr(err)[:100])
""")

md("""
### Watching it learn: the loss, step by step and epoch by epoch
**Top:** the loss on every training step (pale line) with a moving average (bold). The dashed vertical lines mark the end of each epoch; the diamonds are the **validation loss** measured at that moment.
**Bottom:** the same story per epoch: the average training loss and the validation loss.

How to read it:
* Both lines going **down** = the student is learning something that also works on notes it has never trained on.
* Training loss still falling but validation loss flat or rising = **overfitting** (memorising). The saved "best" model is the epoch with the lowest validation loss.
* Training loss is averaged *while* the weights are changing (and with dropout on), so it usually looks a bit worse than the validation loss in epoch 1: that is normal.
""")

code("""
H = [h for h in history if h["epoch"] > 0]
steps_all = [l for h in H for l in h["steps"]]
x_end = np.cumsum([len(h["steps"]) for h in H])
win = max(5, len(steps_all) // 60)
smooth = pd.Series(steps_all).rolling(win, min_periods=1).mean()

fig = make_subplots(rows=2, cols=1, subplot_titles=("Loss at every training step", "Loss at the end of every epoch"), vertical_spacing=0.16)
fig.add_trace(go.Scatter(y=steps_all, mode="lines", name="step loss", line=dict(color="rgba(0,153,153,0.25)", width=1)), row=1, col=1)
fig.add_trace(go.Scatter(y=smooth, mode="lines", name=f"moving average ({win} steps)", line=dict(color="#009999", width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=x_end, y=[h["val_loss"] for h in H], mode="markers+lines", name="validation loss (end of epoch)",
                         marker=dict(symbol="diamond", size=11, color="#d6b02e"), line=dict(color="#d6b02e", dash="dot")), row=1, col=1)
for xe in x_end[:-1]: fig.add_vline(x=xe, line_dash="dash", line_color="grey", row=1, col=1)
fig.add_trace(go.Scatter(x=[h["epoch"] for h in H], y=[h["train_loss"] for h in H], mode="lines+markers", name="train loss (epoch average)", line=dict(color="#009999", width=3)), row=2, col=1)
fig.add_trace(go.Scatter(x=[h["epoch"] for h in H], y=[h["val_loss"] for h in H], mode="lines+markers", name="validation loss", line=dict(color="#d6b02e", width=3)), row=2, col=1)
fig.add_hline(y=history[0]["val_loss"], line_dash="dot", line_color="grey", annotation_text="before training", row=2, col=1)
fig.update_xaxes(title_text="training step", row=1, col=1); fig.update_xaxes(title_text="epoch", dtick=1, row=2, col=1)
fig.update_yaxes(title_text="loss (lower is better)")
fig.update_layout(height=720, title="How the loss falls during training", legend=dict(orientation="h", y=-0.12))
show(fig, "03_loss")

best_ep = min(H, key=lambda h: h["val_loss"])["epoch"]; last = H[-1]
gap = last["val_loss"] - last["train_loss"]
print(f"best epoch by validation loss: {best_ep} of {len(H)}")
print(f"last epoch: train {last['train_loss']:.4f} vs validation {last['val_loss']:.4f} (gap {gap:+.4f}) ->",
      "validation is clearly worse than training: watch for overfitting" if gap > 0.1 * last["val_loss"] else "no strong sign of overfitting")
""")

md("""
### Watching it learn: the quality metrics after each epoch
Loss is what training minimises, but it is hard to read. These are the numbers we actually care about, measured on the validation notes after every epoch
(epoch 0 = before any training; the student starts at about 0.5 AUC, i.e. guessing).

* **Left:** mean AUC over the yes/no questions (the dashed line is the word-counting baseline), and the two level metrics.
* **Right:** AUC by aroma family, one line per family. Families whose line rises early are *easy* for the model (common, clear words); flat lines are *hard* (rare in critic prose).
""")

code("""
ep = [h["epoch"] for h in history]
fig = make_subplots(rows=1, cols=2, subplot_titles=("Quality of the answers after each epoch", "Which aroma families are learned (AUC by family)"), column_widths=[0.45, 0.55])
fig.add_trace(go.Scatter(x=ep, y=[h["auc"] for h in history], mode="lines+markers", name="yes/no questions: mean AUC", line=dict(color="#009999", width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=ep, y=[BASE_VAL_AUC] * len(ep), mode="lines", name="word-counting baseline AUC", line=dict(color="grey", dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=ep, y=[h["level_exact"] for h in history], mode="lines+markers", name="levels: exact match", line=dict(color="#d6b02e", width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=ep, y=[h["level_within_one"] for h in history], mode="lines+markers", name="levels: within one step", line=dict(color="#2375ae", width=3)), row=1, col=1)
fams = sorted({k.split(" / ")[0] for k in history[-1]["clusters"]})
clu = sorted(history[-1]["clusters"], key=lambda k: -history[-1]["clusters"][k])
for k in clu:
    ys = [h["clusters"].get(k, np.nan) for h in history]
    fig.add_trace(go.Scatter(x=ep, y=ys, mode="lines+markers", name=k, legendgroup="fam", line=dict(width=1.5), marker=dict(size=5), visible="legendonly" if len(clu) > 8 and clu.index(k) >= 8 else True), row=1, col=2)
fig.update_xaxes(title_text="epoch (0 = untrained)", dtick=1); fig.update_yaxes(title_text="score (higher is better)", range=[0, 1.02], row=1, col=1); fig.update_yaxes(title_text="mean AUC", row=1, col=2)
fig.update_layout(height=520, title="Report card after each epoch (validation notes)", legend=dict(font=dict(size=10)))
show(fig, "04_metrics")

tbl = pd.DataFrame([dict(epoch=h["epoch"], train_loss=h["train_loss"], val_loss=h["val_loss"], mean_auc=h["auc"], level_exact=h["level_exact"], level_within_one=h["level_within_one"], questions_scored=h["n_scored"]) for h in history]).round(4)
tbl
""")

# --------------------------------------------------------------------------------------------------------------------
md("""
### What did training change? (the starting weights versus the trained weights)
For every weight matrix we compute **how far it moved relative to its own size**: `|trained - start| / |start|`. 0% = untouched. The heads started from random numbers, so they are *not* a fair "pre-trained" comparison: for them we show the change relative to their random start, which is large by construction.
**Left:** the average change by kind of part. **Right:** the change by depth in the body. Remember what to expect: a gentle adjustment of the body, strongest in the upper layers, which are closest to the task.
""")

code("""
W1 = {k: v.detach().cpu() for k, v in model.state_dict().items() if k in W0}
rows = []
for k in W0:
    d = (W1[k].float() - W0[k].float()).norm().item(); n0 = W0[k].float().norm().item()
    if n0 == 0 or W0[k].numel() < 2: continue
    part = "HEAD: " + k.split(".")[0] if not k.startswith("encoder.") else "body: " + component(k[len("encoder."):])
    rows.append((k, part, layer_index(k) if k.startswith("encoder.") else None, 100 * d / n0, W0[k].numel()))
chg = pd.DataFrame(rows, columns=["weight", "part", "layer", "chg_pct", "n"])
by_part = chg.groupby("part").apply(lambda g: np.average(g.chg_pct, weights=g.n)).sort_values()
by_layer = chg[chg.layer.notna() & chg.part.str.startswith("body")].groupby("layer").apply(lambda g: np.average(g.chg_pct, weights=g.n))
fig = make_subplots(rows=1, cols=2, subplot_titles=("Average change by part (%)", "Body: change by layer depth (%)"), column_widths=[0.55, 0.45])
fig.add_trace(go.Bar(y=by_part.index, x=by_part.values, orientation="h", marker_color=["#009999" if p.startswith("HEAD") else "#7fb3d5" for p in by_part.index],
                     text=[f"{v:.2f}%" for v in by_part.values], textposition="outside", showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=by_layer.index, y=by_layer.values, mode="lines+markers", line=dict(color="#7fb3d5", width=3), showlegend=False), row=1, col=2)
fig.update_xaxes(title_text="layer (0 = nearest the words, last = nearest the heads)", row=1, col=2); fig.update_xaxes(title_text="% change", row=1, col=1)
fig.update_layout(height=430, title="Distillation moves the heads a lot and the body gently")
show(fig, "03b_weight_change")
print(f"body overall: {np.average(chg[chg.part.str.startswith('body')]['chg_pct'], weights=chg[chg.part.str.startswith('body')].n):.2f}% | "
      f"heads: {np.average(chg[chg.part.str.startswith('HEAD')]['chg_pct'], weights=chg[chg.part.str.startswith('HEAD')].n):.0f}% (of their random start)")
""")

md("""
## 7b. Three ways to adapt the body: a real comparison
So far we used **full fine-tuning**: every number in the body is allowed to move a little. There are cheaper ways. Same data, same epochs, same heads; only *what is allowed to change* differs:

| Strategy | What is trained | What is left alone | Idea |
|---|---|---|---|
| **A. Full fine-tuning** (the main run above) | every body weight (tiny learning rate) + the heads | nothing | most flexible, most memory and time |
| **B. Frozen body** | **only the heads** | the whole body | the body is a fixed reader; the heads learn to use its summary |
| **C. LoRA** (Low-Rank Adaptation) | the heads + **tiny adapter matrices** added next to the attention weights | the original body weights | learn a small *correction* instead of moving the big matrices |

**How LoRA works.** A weight matrix `W` (say 768 x 768 = 590,000 numbers) is frozen. Next to it we add two thin matrices, `B` (768 x r) and `A` (r x 768) with a small *rank* `r` (e.g. 8). The layer computes
`W·x + (alpha/r) · B·(A·x)`. `B` starts at zero, so at the start the model is *exactly* the pretrained one. Only `A` and `B` are trained: `2 x 768 x 8` = 12,000 numbers, about 2% of `W`.
The correction `B·A` has rank at most `r`: a "low-rank" change, which turns out to be enough for many tasks. After training it can even be merged back into `W`.
""")

code("""
mermaid('''
flowchart LR
  subgraph FULL["A. Full fine-tuning"]
    b1["body: ALL weights move a little"] --> h1["heads: trained"]
  end
  subgraph FROZEN["B. Frozen body"]
    b2["body: frozen"] --> h2["heads: trained"]
  end
  subgraph LORA["C. LoRA"]
    b3["body: original weights W frozen"] --- ad["tiny adapters B x A<br/>next to the attention weights"]
    ad --> h3["heads: trained"]
    b3 --> h3
  end
  classDef tr fill:#d9f0f0,stroke:#009999
  classDef fr fill:#e0e0e0,stroke:#888
  class b1,h1,h2,h3,ad tr
  class b2,b3 fr
''', "07a_strategies", 420)
""")

code("""
# LoRA, written by hand (about 10 lines) so nothing is hidden. W stays frozen; A and B are the only new numbers.
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base, self.scale = base, alpha / r                                     # base = the original frozen Linear layer
        self.A = nn.Parameter(torch.empty(r, base.in_features)); nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))                    # zeros: at the start the layer equals the pretrained one
    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale

LORA_TARGETS = ("query", "value", "q_proj", "v_proj", "query_proj", "value_proj")   # the attention 'question' and 'content' matrices
def add_lora(encoder, r, alpha):
    hits = [(mod, name, child) for modname, mod in encoder.named_modules() for name, child in mod.named_children() if isinstance(child, nn.Linear) and name in LORA_TARGETS]
    if not hits:                                                                   # models with fused attention weights: adapt every attention Linear
        hits = [(mod, name, child) for modname, mod in encoder.named_modules() if ("attn" in modname.lower() or "attention" in modname.lower())
                for name, child in mod.named_children() if isinstance(child, nn.Linear)]
    for mod, name, child in hits: setattr(mod, name, LoRALinear(child, r, alpha))
    return len(hits)

# the arithmetic of one matrix
d = model.encoder.config.hidden_size
print(f"one attention matrix W: {d} x {d} = {d*d:,} numbers (frozen)  |  LoRA adapters at rank {LORA_RANK}: B {d} x {LORA_RANK} + A {LORA_RANK} x {d} = {2*d*LORA_RANK:,} numbers  ({100*2*d*LORA_RANK/(d*d):.1f}% of W)")
""")

md("""
### Where does the gradient go under each strategy? (measured)
Before training the three strategies, one backward pass on the same batch of 32 notes under each. The table-picture shows **which parts of the model receive a gradient** (and so can learn) and how large it is.
What to look for:
* **Frozen body:** nothing in the body needs a gradient, so the backward pass **stops at the heads**. The body only runs forward: this is why it is so fast and light.
* **LoRA:** the adapters sit inside the attention of *every* layer, so the error signal **must travel back through all the frozen layers** to reach them. It is cheaper than full fine-tuning (no gradient and no optimiser memory for the big matrices) but not as cheap as a frozen body.
* **LoRA at step 0:** `B` starts at zero, and the gradient of `A` is proportional to `B`: so `A` gets **exactly zero** gradient on the very first step and only `B` moves. After that first update `B` is no longer zero and `A` starts learning too.
""")

code("""
def grad_map(mode):
    m = WsetStudent(MODEL_NAME, len(BIN_COLS), int(slices[-1])).to(device)
    for p in m.encoder.parameters(): p.requires_grad = (mode == "full")
    if mode == "lora": add_lora(m.encoder, LORA_RANK, LORA_ALPHA); m.to(device)
    m.eval(); gd, secs_ = backward_once(m)
    gd["part"] = gd.name.map(part_of); gd["sq"] = gd.grad_norm.fillna(0) ** 2
    out = {pt: (int(d_.n[d_.trainable].sum()), (float(np.sqrt(d_.sq.sum())) if d_.grad_norm.notna().any() else None)) for pt, d_ in gd.groupby("part")}
    ab = {k: float(np.sqrt(gd[gd.name.str.endswith(k)].sq.sum())) for k in (".A", ".B")} if mode == "lora" else None
    body_has_grad = bool(gd[~gd.name.str.startswith(("binary_head", "group_head"))].grad_norm.notna().any())
    del m; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    return out, secs_, ab, body_has_grad

PARTS_G = ["word table", "attention", "feed-forward", "layer norms", "other (pooler)", "LoRA adapters", "output heads"]
GM = {mode: grad_map(mode) for mode in ("full", "frozen", "lora")}
names_ = {"full": "A. Full fine-tuning", "frozen": "B. Frozen body", "lora": "C. LoRA"}
z, txt = [], []
for mode in GM:
    rowz, rowt = [], []
    for pt in PARTS_G:
        tr, g = GM[mode][0].get(pt, (0, None))
        if g is None: rowz.append(np.nan); rowt.append("no gradient<br>(frozen)" if pt in GM[mode][0] else "-")
        else: rowz.append(np.log10(max(g, 1e-12))); rowt.append(f"{g:.3g}<br>{tr:,} train")
    z.append(rowz); txt.append(rowt)
fig = go.Figure(go.Heatmap(z=z, x=PARTS_G, y=[names_[m] for m in GM], text=txt, texttemplate="%{text}", colorscale="Teal", showscale=False, hoverongaps=False, xgap=3, ygap=3))
fig.update_layout(height=360, margin=dict(t=110, l=150), title="Which parts receive a gradient? (cell: gradient size, trainable numbers)", yaxis=dict(autorange="reversed"), xaxis=dict(side="top", tickangle=0, tickfont=dict(size=10)))
show(fig, "07c_gradient_by_strategy")
for mode in GM:
    out, secs_, ab, through = GM[mode]
    print(f"{names_[mode]:20s} | one forward+backward: {secs_:.2f}s | trainable numbers: {sum(v[0] for v in out.values()):>12,} | gradient travels back through the body: {'YES' if through else 'no, it stops at the heads'}")
ab = GM["lora"][2]
print(f"\\nLoRA at step 0:  gradient of the A matrices = {ab['.A']:.2e}   gradient of the B matrices = {ab['.B']:.2e}   (A is zero because B starts at zero)")
""")

code("""
RES_FILE = PACK_DIR / "strategy_comparison.json"
CMP_ID = dict(RUN_ID, lora=(LORA_RANK, LORA_ALPHA, LR_LORA))

def train_strategy(mode):
    global model
    main_model = model
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    if device.type == "cuda": torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = WsetStudent(MODEL_NAME, len(BIN_COLS), int(slices[-1])).to(device)
    try:
        for p in model.encoder.parameters(): p.requires_grad = False
        extra = []
        if mode == "full":
            for p in model.encoder.parameters(): p.requires_grad = True
        elif mode == "lora":
            n_hit = add_lora(model.encoder, LORA_RANK, LORA_ALPHA); model.to(device)
            extra = [p for n, p in model.encoder.named_parameters() if n.endswith(".A") or n.endswith(".B")]
            print(f"  LoRA adapters added to {n_hit} attention matrices")
        heads = list(model.binary_head.parameters()) + list(model.group_head.parameters())
        groups = [{"params": heads, "lr": LR_HEADS}]
        if mode == "full": groups.append({"params": list(model.encoder.parameters()), "lr": LR_ENCODER})
        if mode == "lora": groups.append({"params": extra, "lr": LR_LORA})
        n_train = sum(p.numel() for g in groups for p in g["params"])
        o = torch.optim.AdamW(groups, weight_decay=0.01); total = EPOCHS * len(train_dl)
        sc = get_linear_schedule_with_warmup(o, int(0.06 * total), total); gs = torch.cuda.amp.GradScaler(enabled=use_scaler)
        Pb, Pg = predict(val_dl); rows = [dict(epoch=0, val_loss=val_loss(), secs=0.0, **report_card(Pb, Pg))]
        for epoch in range(EPOCHS):
            model.train(); t0 = time.time()
            if mode == "frozen": model.encoder.eval()                              # a frozen body should not apply dropout either
            for ids, mask, yb, yg in train_dl:
                ids, mask, yb, yg = ids.to(device), mask.to(device), yb.to(device), yg.to(device)
                with autocast(): lb, lg = model(ids, mask)
                loss = wset_loss(lb, lg, yb, yg)[0]
                o.zero_grad(set_to_none=True); gs.scale(loss).backward(); gs.unscale_(o)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); gs.step(o); gs.update(); sc.step()
            Pb, Pg = predict(val_dl); rows.append(dict(epoch=epoch + 1, val_loss=val_loss(), secs=time.time() - t0, **report_card(Pb, Pg)))
            print(f"  {mode:6s} epoch {epoch+1}/{EPOCHS} | validation loss {rows[-1]['val_loss']:.4f} | mean AUC {rows[-1]['auc']:.3f} | {rows[-1]['secs']:.0f}s")
        peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None
        return dict(mode=mode, trainable=int(n_train), peak_gb=peak, rows=rows)
    finally:
        model = main_model                                                         # always give the main model back
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()

results = {}
if COMPARE_STRATEGIES:
    cached = json.load(open(RES_FILE)) if RES_FILE.exists() else {}
    if cached.get("id") == json.loads(json.dumps(CMP_ID)):
        results = cached["results"]; print("using the saved comparison from the previous run:", list(results))
    for mode in ("frozen", "lora"):
        if mode not in results:
            print(f"training strategy: {mode}"); results[mode] = train_strategy(mode)
            json.dump(dict(id=CMP_ID, results=results), open(RES_FILE, "w"))
else:
    print("COMPARE_STRATEGIES is False: skipping (set it to True in the settings to train the two alternatives)")
""")

md("""
### The comparison
Everything below is measured on the **validation** notes (the test notes stay sealed until we have chosen the model). The *full fine-tuning* numbers are those of the main run.
Look at four things: **how good** (AUC, loss), **how fast it learns**, **how many numbers are trained**, and **how much time and GPU memory it costs**.
""")

code("""
if results:
    full = dict(mode="full", trainable=int(sum(p.numel() for p in model.parameters())), peak_gb=max([h.get("peak_gb") or 0 for h in history]) or None,
                rows=[dict(epoch=h["epoch"], val_loss=h["val_loss"], auc=h["auc"], level_exact=h["level_exact"], secs=h.get("secs", 0.0) or 0.0) for h in history])
    S3 = {"A. full fine-tuning": full, "B. frozen body": results["frozen"], "C. LoRA": results["lora"]}
    col3 = {"A. full fine-tuning": "#009999", "B. frozen body": "#888888", "C. LoRA": "#d6b02e"}
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Validation loss after each epoch", "Mean AUC after each epoch"))
    for name, r in S3.items():
        ep_ = [x["epoch"] for x in r["rows"]]
        fig.add_trace(go.Scatter(x=ep_, y=[x["val_loss"] for x in r["rows"]], mode="lines+markers", name=name, line=dict(color=col3[name], width=3), legendgroup=name), row=1, col=1)
        fig.add_trace(go.Scatter(x=ep_, y=[x["auc"] for x in r["rows"]], mode="lines+markers", name=name, line=dict(color=col3[name], width=3), legendgroup=name, showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=ep_, y=[BASE_VAL_AUC] * len(ep_), mode="lines", name="word-counting baseline", line=dict(color="black", dash="dash")), row=1, col=2)
    fig.update_xaxes(title_text="epoch", dtick=1); fig.update_yaxes(title_text="loss (lower is better)", row=1, col=1); fig.update_yaxes(title_text="AUC (higher is better)", row=1, col=2)
    fig.update_layout(height=430, title="Three ways to adapt the body: learning curves (validation)", legend=dict(orientation="h", y=-0.2))
    show(fig, "07b_strategy_curves")

    names = list(S3)
    tab = pd.DataFrame({"strategy": names, "numbers trained": [S3[n]["trainable"] for n in names],
                        "% of the model": [100 * S3[n]["trainable"] / S3["A. full fine-tuning"]["trainable"] for n in names],
                        "seconds / epoch": [np.mean([x["secs"] for x in S3[n]["rows"] if x["epoch"] > 0]) for n in names],
                        "peak GPU memory (GB)": [S3[n]["peak_gb"] for n in names],
                        "best validation AUC": [max(x["auc"] for x in S3[n]["rows"]) for n in names],
                        "final validation loss": [S3[n]["rows"][-1]["val_loss"] for n in names]}).round(3)
    fig = make_subplots(rows=1, cols=4, subplot_titles=("Numbers trained (log)", "Seconds per epoch", "Peak GPU memory (GB)", "Best validation AUC"))
    for i, k in enumerate(["numbers trained", "seconds / epoch", "peak GPU memory (GB)", "best validation AUC"], 1):
        fig.add_trace(go.Bar(x=["A", "B", "C"], y=tab[k], marker_color=[col3[n] for n in names], text=tab[k].map(lambda v: "n/a" if pd.isna(v) else f"{v:,.3g}"), textposition="outside", showlegend=False), row=1, col=i)
    fig.update_yaxes(type="log", row=1, col=1); fig.update_yaxes(range=[0.4, 1.0], row=1, col=4)
    fig.update_layout(height=400, title="What each strategy costs, and what it gets (A = full, B = frozen, C = LoRA)")
    show(fig, "07c_strategy_costs")
    tab
else:
    print("no comparison to show")
""")

md("""
### How to read it, and which to choose
* **A, full fine-tuning** usually reaches the best scores but trains *every* number: most memory, most time, and a full copy of the body for every task.
* **B, frozen body** is the fastest and lightest (the body is only *read*), and you can keep one shared body for many heads. It is limited to what the pre-trained summary already contains, so it often scores a bit lower.
* **C, LoRA** trains a few percent of the numbers; the original body is untouched (you store only the small adapters per task) and the scores usually land close to full fine-tuning. It is the usual choice when memory is tight or you need many task variants.
* With only ~10,000 notes and a body this small, the gaps can be small and a single run is noisy (try another `SEED`): differences of a point or two of AUC are not conclusive. The honest takeaway is the *trade-off*, not a winner.
* Whichever you adopt, evaluate it **once** on the sealed test notes (section 8), not on the validation notes used here to compare.
""")

md("""
## 7c. Two classic alternatives: hand-written rules and a random forest
Fine-tuning a Transformer is **one** way to answer the 231 questions. Before it was an option, the same job was done in other ways, and they are still the right tool in many situations. Here we build two of them on **exactly the same notes, the same split and the same questions** and compare.

| | **Rules** (deterministic) | **Random forest** | **Fine-tuned student** (sections 7 and 7b) |
|---|---|---|---|
| Idea | a list of phrases written down: *"if the note contains `vanilla` then the answer to the vanilla question is yes"* | many decision trees, each asking a chain of *yes/no questions about words* (does the note contain `tannin`? `grippy`?...), voting together | a neural network that has read a lot of English and is nudged to copy Jev |
| What is learned | **nothing at run time**: the rules are fixed. (Humans wrote the first ones; we can *mine* more from the labelled notes.) | which word questions split the notes best | millions of weights |
| Sees | the exact phrases it was given | words and two-word phrases, **as counts** (no word order, no meaning) | the whole sentence, with context and meaning |
| Explainable? | **fully**: every answer points to the phrase that caused it | partly (which words matter) | hardly |
| Needs a GPU? | no | no | for training, yes |

Same honesty rules as before: everything is built from the **training** notes only; we compare on the **validation** notes now (the test notes stay sealed until section 8).

### A. The original rule system: a lexicon written by hand
The project started with a **fully deterministic** matcher: a *lexicon* (a list of phrases for each WSET descriptor, written from the SAT card; `lexicon/aromas.yaml` and `structure.yaml`) and a small matcher that finds the phrases in a note,
handles **negation** ("no oak" does not count), and notes whether the phrase is in the **nose** or the **palate** part of the sentence. It never learns: same note in, same answer out, forever.
It lives in the open-source repository (`src/wsetkit.py`); in the PC folder it sits in `rules_pkg\\`.
""")

code("""
import re, unicodedata
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sp

RULES_HOME = next((p for p in (Path("rules_pkg"), Path("..")) if (p / "src" / "wsetkit.py").exists() and (p / "lexicon" / "aromas.yaml").exists()), None)
lex = None
if RULES_HOME is not None:
    sys.path.insert(0, str((RULES_HOME / "src").resolve())); import wsetkit; lex = wsetkit.Lexicon()
    print(f"rule system found in {RULES_HOME.resolve()}: {len(lex.entries):,} hand-written phrases | {len(lex.descriptors)} aroma descriptors in {len({c for c, _ in lex.descriptors})} clusters | {len(lex.scales)} structure scales (sweetness, acidity, tannin...)")
    vanilla = sorted(" ".join(k) for k, v in lex.entries.items() if any(t[0] == "aroma" and t[1] == "oak" and t[2] == "vanilla" for t in v))
    print("phrases written for  oak / vanilla :", vanilla)
else:
    print("rules_pkg was not found next to the notebook: the ORIGINAL rule system (v0) is skipped; the rules mined from the data below still work.")
""")

code("""
if lex is not None:
    toks_, cov_, hits_, _ = lex.match(demo_note)
    print("NOTE:", demo_note, "\\n")
    print("what the matcher found (phrase, what it means, negated?, nose or palate part):")
    ex = pd.DataFrame([dict(phrase=h["surface"], means=f"{h['f1']} / {h['f2']}", kind=h["kind"], negated=h["neg"], part=h["scope"]) for h in hits_])
else: ex = None
ex
""")

md("""
To compare with Jev we translate the matcher's findings into the **same 231 questions**: a hit on `vanilla` in the nose part gives a score of 1 for `nose__oak__vanilla`; in an unspecified part it counts at 0.5 for both nose and palate;
a negated phrase counts for nothing; a "firm tannins" cue answers `stated__tannin` *and* picks the tannin level; and so on. Questions the lexicon has **no** phrases for (for example `quality`) get the most common answer.
""")

code("""
t0 = time.time()
G_np = Y_grp.numpy() if hasattr(Y_grp, "numpy") else np.asarray(Y_grp)
GCLS = {q: schema["groups"][q] for q in GROUPS}
MAJ = {q: int(np.bincount(G_np[train_idx][:, s:e].argmax(1), minlength=e - s).argmax()) for q, s, e in zip(GROUPS, slices[:-1], slices[1:])}   # most common answer in training
Q_COL = {q: j for j, q in enumerate(schema["binary"])}

def v0_predict(idx):
    # -> (scores for the 216 yes/no questions, {question: predicted level} for the 15 multi-level ones) for each note
    Sb = np.zeros((len(idx), len(BIN_COLS)), np.float32); preds = []
    for r, i in enumerate(idx):
        toks, cov, hits, _ = lex.match(df.description.iloc[i])
        pos = [h for h in hits if not h["neg"]]; pr = {}
        for h in pos:
            if h["kind"] == "aroma":
                for part in {"nose": ["nose"], "palate": ["palate"], "both": ["nose", "palate"], "unspecified": ["nose", "palate"]}[h["scope"]]:
                    q = f"{part}__{h['f1']}__{h['f2']}"
                    if q in Q_COL: Sb[r, Q_COL[q]] = max(Sb[r, Q_COL[q]], 1.0 if h["scope"] != "unspecified" else 0.5)
            elif h["kind"] == "structure":
                if f"stated__{h['f1']}" in Q_COL: Sb[r, Q_COL[f"stated__{h['f1']}"]] = 1.0
            elif h["kind"] == "cat":
                for q in ({"tannin_nature": "tnature__" + h["f2"], "texture": "texture__" + h["f2"], "petillance": "appother__petillance", "appearance_other": "appother__" + h["f2"].replace("/", "_")}.get(h["f1"]), ):
                    if q in Q_COL: Sb[r, Q_COL[q]] = 1.0
                if h["f1"] == "petillance" and "palate__petillance" in Q_COL: Sb[r, Q_COL["palate__petillance"]] = 1.0
        for s_ in lex.scales:                                                   # a cue for a level of tannin, body... -> that level
            lv = [h["f2"] for h in pos if h["kind"] == "structure" and h["f1"] == s_]
            if lv and "level__" + s_ in GCLS: pr["level__" + s_] = GCLS["level__" + s_][int(np.clip(round(float(np.mean(lv))), 1, len(GCLS["level__" + s_]))) - 1]
        for field, q in (("colour", "app__colour"), ("clarity", "app__clarity"), ("appearance_intensity", "app__intensity"), ("condition", "condition")):
            v = [h["f2"] for h in pos if h["kind"] == "cat" and h["f1"] == field and h["f2"] in GCLS[q]]
            if v: pr[q] = v[0]
        dn = any(h["kind"] == "readiness" and h["f1"] == "drink_now" for h in pos); ca = any(h["kind"] == "readiness" and h["f1"] == "can_age" for h in pos)
        ag = wsetkit.ageing_outcome(dn, ca)
        if ag: pr["ageing"] = ag
        rip, _ = wsetkit.resolve_ripeness([h["f2"] for h in pos if h["kind"] == "aroma" and (h["f1"] == "fruit_ripeness" or (h["f1"] in ("tertiary_red", "tertiary_white") and h["f2"] in ("dried_fruit", "cooked_fruit")))])
        if rip in GCLS["ripeness"]: pr["ripeness"] = rip
        preds.append(pr)
    return Sb, preds

def groups_from(preds_list, fallback=None):
    # one-hot probability vector per multi-level question: the rule's level if it has one, else the fallback, else the most common answer
    Pg = np.zeros((len(preds_list), int(slices[-1])), np.float32)
    for r, pr in enumerate(preds_list):
        for q, s, e in zip(GROUPS, slices[:-1], slices[1:]):
            k = GCLS[q].index(pr[q]) if q in pr else (MAJ[q] if fallback is None else int(fallback[r, s:e].argmax()))
            Pg[r, s + k] = 1.0
    return Pg

def card(Pb, Pg, Tb, Tg, min_pos):
    a = auc_per_question(Pb, Tb, min_pos); ex_, w1_ = level_scores(Pg, Tg)
    return dict(auc=float(np.nanmean(a)), level_exact=ex_, level_within_one=w1_, n_scored=int((~np.isnan(a)).sum()), per_q=a)

SPLIT_IDX = {"train": train_idx, "val": val_idx, "test": test_idx}
if lex is not None:
    V0 = {k: v0_predict(ix) for k, ix in SPLIT_IDX.items()}
    t_v0 = time.time() - t0
    # how much to trust a rule hit: the share of hits that Jev agrees with, measured on TRAINING notes
    Ytr_bin = (B[train_idx] >= 0.5)
    P0_TRUST = np.array([((V0["train"][0][:, j] > 0) & Ytr_bin[:, j]).sum() / max(1, (V0["train"][0][:, j] > 0).sum()) for j in range(len(BIN_COLS))], np.float32)
    S0 = {k: np.where(V0[k][0] > 0, P0_TRUST[None, :] * np.where(V0[k][0] >= 1, 1.0, 0.7), 0).astype(np.float32) for k in V0}      # score = how trustworthy this hit is
    G0 = {k: groups_from(V0[k][1]) for k in V0}
    c0 = card(S0["val"], G0["val"], T_bin_val, T_grp_val, MIN_POS)
    has_rule = (V0["train"][0].sum(0) > 0)
    print(f"original rules (v0), validation notes: mean AUC {c0['auc']:.3f} over {c0['n_scored']} questions | {int(has_rule.sum())} of {len(BIN_COLS)} yes/no questions have at least one rule that fires | level exact {c0['level_exact']:.3f}, within one {c0['level_within_one']:.3f}")
    print(f"matching {sum(len(v) for v in SPLIT_IDX.values()):,} notes took {t_v0:.0f}s on CPU (no GPU, nothing learned)")
else:
    S0 = {k: np.zeros((len(ix), len(BIN_COLS)), np.float32) for k, ix in SPLIT_IDX.items()}; G0 = {k: groups_from([{}] * len(ix)) for k, ix in SPLIT_IDX.items()}
    c0 = card(S0["val"], G0["val"], T_bin_val, T_grp_val, MIN_POS); t_v0 = 0.0
""")

md("""
### B. Improving the rules with the labelled notes (rules v1)
The lexicon only knows the phrases someone thought of. With tens of thousands of notes labelled by Jev we can **find the phrases the lexicon misses** (`creamy`? `lemon curd`? `grippy`?) automatically. The method is plain counting, and the result is still a list of fixed phrases:

1. Cut every **training** note into words and two-word phrases (so `vanilla`, `black cherry`, `firm tannins` are all candidates).
2. For every question and every phrase, count **support** (in how many training notes does the phrase appear *and* Jev says yes?) and **precision** (when the phrase appears, how often does Jev say yes?).
3. **Keep a phrase as a rule** if its support is at least `RULE_MIN_SUPPORT` and its precision at least `RULE_MIN_PRECISION`. For the multi-level questions the same is done per level ("`grippy` -> tannin High").
4. **Prune** redundant two-word rules (if `vanilla` is already a rule, `and vanilla` adds nothing).
5. **Score** a note for a question with the *strongest rule that fires* (the highest precision), or with the original lexicon's trust value if that is higher. Nothing else is learned: no weights, no training loop.

This is the human-in-the-loop idea automated: the mined phrases are **candidates**; the notebook also writes them to a file so a WSET-qualified taster can accept or reject each one (and the good ones can be added to the lexicon).
Caveat: the mined rules do not handle negation (a phrase like `no vanilla` is two words, so it is only a rule if it is frequent and precise enough).
""")

code("""
def lemma_(w):                                                              # tiny plural-stripper: "tannins" -> "tannin", "berries" -> "berry"
    if len(w) > 4 and w.endswith("ies"): return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("ches", "shes", "xes", "zes", "sses")): return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")): return w[:-1]
    return w

def phrases(text):
    # words and two-word phrases, never across a comma / full stop / bracket
    t = unicodedata.normalize("NFKD", str(text).lower()); t = "".join(c for c in t if not unicodedata.combining(c)).replace("'", "").replace("-", " ")
    out = []
    for clause in re.split(r"[.;:,!?()]", t):
        w = [lemma_(x) for x in re.findall(r"[a-z]+", clause)]
        out += w + [w[i] + " " + w[i + 1] for i in range(len(w) - 1)]
    return out

t0 = time.time()
vec_rules = CountVectorizer(analyzer=phrases, min_df=3, binary=True, dtype=np.float32)
X_tr = vec_rules.fit_transform(df.description.iloc[train_idx]).tocsc()
X_va, X_te = vec_rules.transform(df.description.iloc[val_idx]).tocsc(), vec_rules.transform(df.description.iloc[test_idx]).tocsc()
VOCAB = np.array(vec_rules.get_feature_names_out())
print(f"{len(VOCAB):,} candidate phrases from {len(train_idx):,} training notes ({time.time() - t0:.0f}s)")

# a two-word rule is redundant when one of its words is already a rule that is about as precise ("and vanilla" adds nothing to "vanilla")
IDX_ = {w: i for i, w in enumerate(VOCAB)}
BI_ = [(i, IDX_.get(w.split(" ")[0], -1), IDX_.get(w.split(" ")[1], -1)) for i, w in enumerate(VOCAB) if " " in w]
BI_I, BI_A, BI_B = (np.array([b[k] for b in BI_], dtype=int) for k in range(3))
def prune(ok, prec):
    ok = ok.copy()
    for A in (BI_A, BI_B):
        has = A >= 0; ii, aa = BI_I[has], A[has]
        ok[ii] = ok[ii] & ~(ok[aa] & (prec[aa] >= prec[ii] - 0.05))
    return ok

def class_onehot(Tg):
    C = np.zeros_like(Tg)
    for s, e in zip(slices[:-1], slices[1:]): C[np.arange(len(Tg)), s + Tg[:, s:e].argmax(1)] = 1
    return C

def mine(X, Tb, Tg, min_support, min_prec):
    # support / precision of every phrase for every yes/no question and for every level of every multi-level question
    n_with = np.asarray(X.sum(0)).ravel(); out = {}
    for name, T in (("bin", (Tb >= 0.5).astype(np.float32)), ("grp", class_onehot(Tg))):
        both = np.asarray((X.T @ sp.csr_matrix(T)).todense()); prec = both / np.maximum(n_with[:, None], 1)
        out[name] = dict(ok=prune((both >= min_support) & (prec >= min_prec), prec), prec=prec.astype(np.float32), support=both)
    return out

def max_rule_scores(X, ok, prec):
    # score = precision of the strongest rule that fires in the note (0 if none)
    X = X.tocsc(); S = np.zeros((X.shape[0], ok.shape[1]), np.float32)
    for j in np.flatnonzero(ok.any(0)):
        g = np.flatnonzero(ok[:, j]); S[:, j] = X[:, g].multiply(prec[g, j][None, :]).max(axis=1).toarray().ravel()
    return S

def rules_predict(R, X, split):
    Sm = max_rule_scores(X, R["bin"]["ok"], R["bin"]["prec"]); Sb = np.maximum(Sm, S0[split])                         # mined rules, or the original lexicon, whichever is more trusted
    Sg = max_rule_scores(X, R["grp"]["ok"], R["grp"]["prec"]); preds = []
    for r in range(X.shape[0]):
        pr = {}
        for q, s, e in zip(GROUPS, slices[:-1], slices[1:]):
            if Sg[r, s:e].max() > 0: pr[q] = GCLS[q][int(Sg[r, s:e].argmax())]
            elif lex is not None and q in V0[split][1][r]: pr[q] = V0[split][1][r][q]                               # fall back on the original lexicon
        preds.append(pr)
    return Sb, groups_from(preds)

t0 = time.time()
RULES1 = mine(X_tr, B[train_idx], G_np[train_idx], RULE_MIN_SUPPORT, RULE_MIN_PRECISION); t_mine = time.time() - t0
S1_val, G1_val = rules_predict(RULES1, X_va, "val")
c1 = card(S1_val, G1_val, T_bin_val, T_grp_val, MIN_POS)
n_rules = int(RULES1["bin"]["ok"].sum()); n_rules_grp = int(RULES1["grp"]["ok"].sum())
print(f"mined {n_rules:,} yes/no rules and {n_rules_grp:,} level rules in {t_mine:.1f}s  (support >= {RULE_MIN_SUPPORT}, precision >= {RULE_MIN_PRECISION})")
print(f"validation notes:  original rules v0: mean AUC {c0['auc']:.3f}, level exact {c0['level_exact']:.3f}  ->  improved rules v1: mean AUC {c1['auc']:.3f}, level exact {c1['level_exact']:.3f}")
""")

md("""
### What do the mined rules look like?
For a few questions: the strongest mined phrases, with their support and precision, and whether the **original lexicon already had them**. Phrases it did not have are exactly what the labelled notes taught us.
""")

code("""
known = {}
if lex is not None:
    for k, v in lex.entries.items():
        for t in v:
            if t[0] == "aroma": known.setdefault(f"{t[1]}__{t[2]}", set()).add(" ".join(k))
def in_lex(key_, phrase):
    return lex is not None and " ".join(wsetkit.tokenize(phrase)) in known.get(key_, set())
def top_rules(q, n=10):
    j = Q_COL[q]; R = RULES1["bin"]; g = np.flatnonzero(R["ok"][:, j])
    g = g[np.argsort(-(R["prec"][g, j] * np.log1p(R["support"][g, j])))][:n]
    key_ = "__".join(q.split("__")[1:])
    return pd.DataFrame(dict(question=q, phrase=VOCAB[g], support=R["support"][g, j].astype(int), precision=R["prec"][g, j].round(2), in_original_lexicon=[in_lex(key_, VOCAB[i]) for i in g]))
demo_qs = [q for q in (DEMO_Q, "stated__finish", "texture__creamy", "nose__floral__blossom", "nose__tropical_fruit__pineapple") if q in Q_COL][:5]
pd.concat([top_rules(q, 6) for q in dict.fromkeys(demo_qs)], ignore_index=True)
""")

code("""
# per aroma family: original rules versus improved rules (validation notes)
fam = pd.DataFrame({"cluster": CLUSTER, "v0": c0["per_q"], "v1": c1["per_q"]}).dropna().groupby("cluster")[["v0", "v1"]].mean()
fam = fam.sort_values("v1")
fig = go.Figure([go.Bar(y=fam.index, x=fam.v0, name="original rules (v0)", orientation="h", marker_color="#b0b0b0"), go.Bar(y=fam.index, x=fam.v1, name="improved rules (v1, mined from the labelled notes)", orientation="h", marker_color="#d6b02e")])
fig.update_layout(barmode="group", height=max(420, 22 * len(fam)), title="Rules before and after learning phrases from the labelled notes (validation, AUC by family)", xaxis=dict(range=[0.4, 1.0], title="mean AUC"), legend=dict(orientation="h", y=-0.08))
show(fig, "07d_rules_v0_v1")
""")

code("""
# Two knobs: how often a phrase must be seen (support) and how reliable it must be (precision). Looser = more rules = better coverage but more noise.
grid = []
for ms in (3, 5, 10, 20):
    row = []
    for mp in (0.4, 0.5, 0.6, 0.7, 0.8):
        R_ = mine(X_tr, B[train_idx], G_np[train_idx], ms, mp); S_, _ = rules_predict(R_, X_va, "val"); row.append(float(np.nanmean(auc_per_question(S_, T_bin_val, MIN_POS))))
    grid.append(row)
fig = go.Figure(go.Heatmap(z=grid, x=[0.4, 0.5, 0.6, 0.7, 0.8], y=[3, 5, 10, 20], colorscale="YlGn", text=[[f"{v:.3f}" for v in r] for r in grid], texttemplate="%{text}", colorbar=dict(title="val AUC")))
fig.update_layout(height=380, title="Validation AUC of the mined rules for different thresholds", xaxis_title="minimum precision", yaxis_title="minimum support (notes)", yaxis=dict(type="category"), xaxis=dict(type="category"))
show(fig, "07e_rule_thresholds")
""")

code("""
# Save the rules as files: a deterministic artefact you can read, diff, review, or use without Python's machine learning stack.
rows_ = []
for name, qs, R in (("yes/no", [q for q in schema["binary"]], RULES1["bin"]), ("level", [f"{q} = {c}" for q in GROUPS for c in GCLS[q]], RULES1["grp"])):
    ii, jj = np.nonzero(R["ok"])
    rows_ += [(name, qs[j], VOCAB[i], int(R["support"][i, j]), round(float(R["prec"][i, j]), 3)) for i, j in zip(ii, jj)]
rules_df = pd.DataFrame(rows_, columns=["kind", "question", "phrase", "support", "precision"]).sort_values(["question", "precision", "support"], ascending=[True, False, False])
rules_df.to_csv(PACK_DIR / "rules_v1.csv", index=False)
cand = rules_df[(rules_df.kind == "yes/no") & (rules_df.question.str.startswith(("nose__", "palate__"))) & (rules_df.support >= 10) & (rules_df.precision >= 0.7)]
cand = cand[[not in_lex("__".join(q.split("__")[1:]), p) for q, p in zip(cand.question, cand.phrase)]]
cand.to_csv(PACK_DIR / "lexicon_candidates_for_review.csv", index=False)
print(f"saved {len(rules_df):,} rules to {PACK_DIR / 'rules_v1.csv'}")
print(f"saved {len(cand):,} strong aroma phrases the original lexicon does not have to {PACK_DIR / 'lexicon_candidates_for_review.csv'}: for a WSET-qualified taster to accept or reject")
cand.head(8)
""")

md("""
### C. A random forest
A **decision tree** is a flow chart of yes/no questions about the note, learned from data: *"Does it contain `tannin`? -> yes: does it contain `grippy`? -> yes: answer High..."*. One tree is brittle; a **random forest** grows many trees, each on a random part of the notes and looking at random words at each split, and **averages their answers**: the mistakes of different trees cancel out.
We give the forest what the rules saw (words and two-word phrases, as 0/1 counts) and ask it to predict **all 292 numbers at once** (the 216 yes/no probabilities and the 76 level probabilities): it learns from Jev's *soft* answers, like the student.
First, one small tree for a single question, so you can read it.
""")

code("""
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestRegressor

rf_min_df = max(2, min(20, len(train_idx) // 300))
vec_rf = CountVectorizer(analyzer=phrases, min_df=rf_min_df, max_features=RF_FEATURES, binary=True, dtype=np.float32)
Xf_tr = vec_rf.fit_transform(df.description.iloc[train_idx]); Xf_va = vec_rf.transform(df.description.iloc[val_idx]); Xf_te = vec_rf.transform(df.description.iloc[test_idx])
RF_WORDS = np.array(vec_rf.get_feature_names_out())
print(f"the forest looks at {Xf_tr.shape[1]:,} words and phrases (the most frequent ones in the training notes)")

jq = Q_COL[DEMO_Q]
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=SEED).fit(Xf_tr, (B[train_idx][:, jq] >= 0.5).astype(int))
print(f"ONE decision tree (depth 3) for the question '{DEMO_Q}':\\n")
print(export_text(tree, feature_names=[w + " (present)" for w in RF_WORDS], max_depth=3))
print("(each line is a yes/no question about the note; 'class: 1' means 'yes, the note mentions it'.)")
""")

code("""
Y_all = np.concatenate([B, G_np], axis=1)                                         # 216 yes/no + 76 level probabilities: the forest learns them all at once
def rf_groups(P):
    out = P[:, len(BIN_COLS):].copy()
    for s, e in zip(slices[:-1], slices[1:]): out[:, s:e] = out[:, s:e] / np.maximum(out[:, s:e].sum(1, keepdims=True), 1e-9)
    return out

def fit_rf(rows, trees=RF_TREES):
    return RandomForestRegressor(n_estimators=trees, max_features="sqrt", min_samples_leaf=RF_MIN_LEAF, max_samples=RF_SAMPLE, n_jobs=-1, random_state=SEED).fit(Xf_tr[rows], Y_all[train_idx][rows])

t0 = time.time(); rf = fit_rf(np.arange(len(train_idx))); t_rf = time.time() - t0
t1 = time.time(); Prf_val = rf.predict(Xf_va); t_rf_pred = time.time() - t1
P_rf_val_bin, P_rf_val_grp = np.clip(Prf_val[:, :len(BIN_COLS)], 0, 1), rf_groups(Prf_val)
crf = card(P_rf_val_bin, P_rf_val_grp, T_bin_val, T_grp_val, MIN_POS)
print(f"random forest ({RF_TREES} trees) trained in {t_rf:.0f}s on CPU | validation notes: mean AUC {crf['auc']:.3f}, level exact {crf['level_exact']:.3f}, within one {crf['level_within_one']:.3f}")
imp = pd.Series(rf.feature_importances_, index=RF_WORDS).sort_values(ascending=False).head(25).sort_values()
fig = go.Figure(go.Bar(y=imp.index, x=imp.values, orientation="h", marker_color="#5b9bd5"))
fig.update_layout(height=620, title="Which words does the forest rely on most? (all 292 answers together)", xaxis_title="importance (share of the impurity reduction)")
show(fig, "07f_rf_importance")
""")

md("""
### D. Do more labelled notes help? A learning curve
This is where the new samples pay off. We re-build each method using only the first `n` training notes (all the notes in the first `n` are a random winery-wise subset) and score on the **same validation notes**.
The original rules do not learn, so their line is flat; the mined rules, the forest and the word-counting baseline climb as they see more examples. Each extra thousand notes cost real money (about $0.40 per thousand), so this tells you when to stop labelling.
""")

code("""
rng_c = np.random.RandomState(SEED); order = rng_c.permutation(len(train_idx))
sizes = sorted({s for s in (300, 1000, 3000, 10000, len(train_idx)) if s <= len(train_idx)})
curve = []
for n_ in sizes:
    rows = order[:n_]
    Rn = mine(X_tr[rows], B[train_idx][rows], G_np[train_idx][rows], max(2, int(round(RULE_MIN_SUPPORT * n_ / len(train_idx)))) if n_ < len(train_idx) else RULE_MIN_SUPPORT, RULE_MIN_PRECISION)
    Sn, _ = rules_predict(Rn, X_va, "val")
    rfn = fit_rf(rows, trees=max(30, RF_TREES // 3)); Pn = np.clip(rfn.predict(Xf_va)[:, :len(BIN_COLS)], 0, 1)
    rg = Ridge(alpha=1.0).fit(Xtr[rows], Y_bin[train_idx][rows].numpy()); Bn = rg.predict(tfidf.transform(df.description.iloc[val_idx]))
    f = lambda P: float(np.nanmean(auc_per_question(P, T_bin_val, MIN_POS)))
    curve.append(dict(notes=n_, rules_v1=f(Sn), random_forest=f(Pn), word_counting_baseline=f(Bn), rules_v0=c0["auc"]))
    print(f"{n_:>6} training notes | rules v1 {curve[-1]['rules_v1']:.3f} | forest {curve[-1]['random_forest']:.3f} | word-counting baseline {curve[-1]['word_counting_baseline']:.3f} | original rules {c0['auc']:.3f}")
cv_ = pd.DataFrame(curve)
fig = go.Figure()
for col_, nm, c, dash in (("rules_v0", "original rules (v0, hand-written)", "#888888", "dash"), ("rules_v1", "improved rules (v1, mined)", "#d6b02e", "solid"), ("random_forest", "random forest", "#2375ae", "solid"), ("word_counting_baseline", "word-counting baseline (TF-IDF + Ridge)", "#8e44ad", "solid")):
    fig.add_trace(go.Scatter(x=cv_.notes, y=cv_[col_], mode="lines+markers", name=nm, line=dict(color=c, dash=dash, width=3)))
fig.update_layout(height=430, title="More labelled notes -> better (validation mean AUC)", xaxis=dict(title="labelled training notes", type="log", tickvals=list(cv_.notes), ticktext=[f"{int(v):,}" for v in cv_.notes]), yaxis_title="mean AUC", legend=dict(orientation="h", y=-0.25))
show(fig, "07g_learning_curve")
""")

code("""
# keep the predictions for the final exam (section 8): the test notes are only used here to PREDICT with models that are already finished
S1_test, G1_test = rules_predict(RULES1, X_te, "test")
Prf_test = rf.predict(Xf_te); P_rf_test_bin, P_rf_test_grp = np.clip(Prf_test[:, :len(BIN_COLS)], 0, 1), rf_groups(Prf_test)
rg_all = Ridge(alpha=1.0).fit(Xtr, G_np[train_idx]); Gb_val = rg_all.predict(tfidf.transform(df.description.iloc[val_idx])); Gb_test = rg_all.predict(tfidf.transform(df.description.iloc[test_idx]))
cb = card(P_base_val, Gb_val, T_bin_val, T_grp_val, MIN_POS)
val_table = pd.DataFrame([dict(method=nm, mean_auc=c["auc"], level_exact=c["level_exact"], level_within_one=c["level_within_one"], questions_scored=c["n_scored"]) for nm, c in
    (("original rules (v0)", c0), ("improved rules (v1)", c1), ("random forest", crf), ("word-counting baseline", cb))]).round(3)
print("VALIDATION notes (the test notes are used in section 8):")
val_table
""")

md("""
## 8. How well does the student copy Jev?

We ask the student about the **test notes** (never used for training) and compare with Jev.

* **Yes/no questions: AUC.** If you pick one note where Jev says yes and one where it says no, how often does the student give the *yes* note the higher score?
  1.0 is perfect, 0.5 is guessing. AUC ignores the fact that most answers are "no". We only score questions with at least 10 positives in the test notes.
* **Level questions (tannin, body, ...)**: *exact* = same level as Jev's top choice; *within one* = at most one step away (Medium vs Medium(+) is a small miss).

The yardstick is the **word-counting baseline** from section 7. This is the **first and only time** the test notes are used.
""")

code("""
P_bin_test, P_grp_test = predict(test_dl)
P_bin_val,  P_grp_val  = predict(val_dl)

TEST_MIN_POS = max(MIN_POS, 10)
a_stu, a_base = auc_per_question(P_bin_test, T_bin, TEST_MIN_POS), auc_per_question(P_base_test, T_bin, TEST_MIN_POS)
ok = ~np.isnan(a_stu)
auc = pd.DataFrame({"question": np.array(schema["binary"])[ok], "positives": (T_bin[:, ok] >= 0.5).sum(0), "student_auc": a_stu[ok], "baseline_auc": a_base[ok]})
auc["cluster"] = auc.question.str.split("__").str[:2].str.join(" / ")
print(f"yes/no questions scored: {len(auc)} of {len(BIN_COLS)}")
print(f"mean AUC  student {auc.student_auc.mean():.3f} | word-counting baseline {auc.baseline_auc.mean():.3f}")

by = auc.groupby("cluster")[["student_auc", "baseline_auc"]].mean().sort_values("student_auc")
fig = go.Figure([go.Bar(y=by.index, x=by.baseline_auc, name="word-counting baseline", orientation="h", marker_color="#b0b0b0"),
                 go.Bar(y=by.index, x=by.student_auc, name="student", orientation="h", marker_color="#009999")])
fig.update_layout(barmode="group", height=max(400, 22 * len(by)), title="Final exam (test notes): AUC by aroma family", xaxis=dict(range=[0.4, 1.0], title="mean AUC"))
show(fig, "05_test_by_family")
by.round(3)
""")

code("""
# --- multi-level questions
rows = []
for q, s, e in zip(GROUPS, slices[:-1], slices[1:]):
    t, p = T_grp[:, s:e].argmax(1), P_grp_test[:, s:e].argmax(1)
    kl = (T_grp[:, s:e] * (np.log(T_grp[:, s:e] + 1e-6) - np.log(P_grp_test[:, s:e] + 1e-6))).sum(1).mean()
    rows.append((q, schema["kind"][q], (t == p).mean(), (abs(t - p) <= 1).mean() if schema["kind"][q] == "score" else np.nan, kl))
levels = pd.DataFrame(rows, columns=["question", "type", "exact", "within_one_level", "kl_to_jev"]).round(3)
levels
""")

md("""
### Where does the student disagree with Jev?
The notes where the student's yes/no answers are furthest from Jev's. Sometimes the student is wrong; sometimes Jev read something into the note that isn't really there.
Reading a few of these is the best way to see the limits of the method.
""")

code("""
diff = np.abs(P_bin_test - T_bin)
for i in np.argsort(-diff.max(1))[:4]:
    j = diff[i].argmax()
    print(df.description.iloc[test_idx[i]], "\\n   ->", schema["binary"][j], f"| Jev {T_bin[i, j]:.2f} vs student {P_bin_test[i, j]:.2f}\\n")
""")

md("""
### All the methods on the final exam (test notes)
The same sealed test notes, the same questions, five ways of answering. The rule systems and the forest were finished (and their settings chosen on the validation notes) before this point, so this is as fair as for the student.
""")

code("""
methods = {"original rules (v0)": (S0["test"], G0["test"]), "improved rules (v1)": (S1_test, G1_test), "random forest": (P_rf_test_bin, P_rf_test_grp),
           "word-counting baseline": (P_base_test, Gb_test), "fine-tuned student": (P_bin_test, P_grp_test)}
secs_student = sum((h.get("secs") or 0) for h in history)
train_secs = {"original rules (v0)": 0.0, "improved rules (v1)": t_mine, "random forest": t_rf, "word-counting baseline": None, "fine-tuned student": secs_student or None}
explain = {"original rules (v0)": "fully", "improved rules (v1)": "fully", "random forest": "partly", "word-counting baseline": "partly", "fine-tuned student": "hardly"}
needs_gpu = {"original rules (v0)": "no", "improved rules (v1)": "no", "random forest": "no", "word-counting baseline": "no", "fine-tuned student": "to train"}
res, per_q = [], {}
for nm, (Pb, Pg) in methods.items():
    c = card(Pb, Pg, T_bin, T_grp, TEST_MIN_POS); per_q[nm] = c["per_q"]
    res.append(dict(method=nm, mean_auc=c["auc"], level_exact=c["level_exact"], level_within_one=c["level_within_one"], train_seconds=train_secs[nm], explainable=explain[nm], gpu=needs_gpu[nm]))
final = pd.DataFrame(res).round(3)
cols_ = {"original rules (v0)": "#b0b0b0", "improved rules (v1)": "#d6b02e", "random forest": "#2375ae", "word-counting baseline": "#8e44ad", "fine-tuned student": "#009999"}
fig = make_subplots(rows=1, cols=3, subplot_titles=("yes/no: mean AUC", "levels: exact", "levels: within one"), horizontal_spacing=0.07)
for k, m_ in enumerate(("mean_auc", "level_exact", "level_within_one")):
    fig.add_trace(go.Bar(x=final.method, y=final[m_], marker_color=[cols_[x] for x in final.method], text=final[m_].round(3), textposition="outside", showlegend=False), row=1, col=k + 1)
    fig.update_yaxes(range=[0.4 if k == 0 else 0, 1.05], row=1, col=k + 1)
fig.update_xaxes(tickangle=-35); fig.update_layout(height=520, margin=dict(t=110, b=150), title="Five ways to answer the 231 questions: the final exam on the test notes")
show(fig, "08b_all_methods")
fam_t = pd.DataFrame({nm: pd.Series(per_q[nm]).groupby(CLUSTER).mean() for nm in methods}).dropna().sort_values("fine-tuned student")
fig = go.Figure([go.Bar(y=fam_t.index, x=fam_t[nm], name=nm, orientation="h", marker_color=cols_[nm]) for nm in methods])
fig.update_layout(barmode="group", height=max(520, 30 * len(fam_t)), title="Final exam by aroma family (AUC on the test notes)", xaxis=dict(range=[0.4, 1.0], title="mean AUC"), legend=dict(orientation="h", y=-0.08))
show(fig, "08c_all_methods_by_family")
print("best on yes/no AUC:", final.loc[final.mean_auc.idxmax(), "method"], "| best on level exact:", final.loc[final.level_exact.idxmax(), "method"])
final
""")

md("""
### How to read the comparison
* **Rules** are the only methods that give a *reason* for every answer ("the phrase `grippy` fired"), cost nothing to run, never drift, and can be audited and edited by a taster. Their weakness is **vocabulary**: they only know the phrases they were given. The hand-written lexicon is precise but incomplete; mining phrases from labelled notes closes much of the gap, which is why labelled data helps even a "no machine learning" system.
* The **random forest** and the **word-counting baseline** both see words as independent counts. They are fast, run on any laptop, and are a strong yardstick: a Transformer has to *earn* its GPU by beating them.
* The **fine-tuned student** reads the sentence (word order, negation, "not as ripe as it sounds"). That is where it should gain, especially on the **multi-level** questions (tannin, body, finish), which depend on how things are said rather than on a single word.
* Rules of thumb: need an explanation or a guarantee -> rules. Need a quick baseline on a CPU -> forest or linear model. Need the best accuracy and can afford a GPU -> fine-tune. And they can be **combined**: use the student for the score and the rules to explain or to flag disagreements.
""")

# --------------------------------------------------------------------------------------------------------------------
md("""
## 9. Turning probabilities into a card

For each yes/no question we need a *threshold*: above it we say "the note mentions this". 0.5 is a poor choice for rare questions,
so for each question we pick the threshold that gives the best F1 score on the **validation** notes, then check it on the test notes.
""")

code("""
V_bin = Y_bin[val_idx].numpy()
thresholds = np.full(len(BIN_COLS), 0.5, dtype=np.float32)
for j in range(len(BIN_COLS)):
    t = V_bin[:, j] >= 0.5
    if t.sum() >= 5:                                           # need a few positives to tune on
        cand = np.linspace(0.05, 0.95, 19)
        f1 = [2 * ((P_bin_val[:, j] >= c) & t).sum() / max(1, (P_bin_val[:, j] >= c).sum() + t.sum()) for c in cand]
        thresholds[j] = cand[int(np.argmax(f1))]

pred = P_bin_test >= thresholds; true = T_bin >= 0.5
tp = (pred & true).sum(); print(f"overall on test notes: precision {tp / max(1, pred.sum()):.2f} | recall {tp / max(1, true.sum()):.2f}")
""")

md("""
### Back to the judge: what did the head learn?
Same note, same question as in section 5, but now the head's 768 weights have been trained.
Then a map of *all the judges at once*: each row of the yes/no head is one question's weight vector. If two questions have similar rows, the model treats them alike:
related descriptors (the same fruit on the nose and on the palate, or red and black cherry) should end up close together. The heatmap shows the **similarity between the rows** (cosine, 1 = identical direction) for the 40 most common questions.
""")

code("""
summary, by_hand, by_model = one_judge(demo_note)
print(f"judge for '{DEMO_Q}' on the same note: by hand {by_hand:.4f} | model {by_model:.4f} | Jev {B[int(np.argmax(B[:, j])), j]:.2f}   (before training it was random)")

W = model.binary_head.weight.detach().float().cpu().numpy()                      # (216, 768): one row per question
common = np.argsort(-(B >= 0.5)[train_idx].mean(0))[:40]
Wc = W[common] / np.linalg.norm(W[common], axis=1, keepdims=True)
names = [schema["binary"][i].replace("__", " ").replace("_", " ") for i in common]
fig = go.Figure(go.Heatmap(z=Wc @ Wc.T, x=names, y=names, colorscale="RdBu", zmid=0, zmin=-1, zmax=1, colorbar=dict(title="cosine")))
fig.update_layout(height=760, title="Which judges think alike? (similarity between the head's weight rows, 40 most common questions)", yaxis=dict(autorange="reversed"))
show(fig, "06_head_similarity")
""")

# --------------------------------------------------------------------------------------------------------------------
md("""
## 10. Save the student and try it

The saved file holds the weights, the thresholds and the question names, everything `app.py` (the stand-alone interface) needs.
Below is a small web interface: paste a critic note, get a WSET-style card. On Colab, click the public link it prints.
""")

code("""
torch.save({"model": model.state_dict(), "model_name": MODEL_NAME, "max_len": MAX_LEN, "schema": schema,
            "thresholds": thresholds.tolist()}, PACK_DIR / "wset_student.pt")
print("saved", PACK_DIR / "wset_student.pt")
""")

code("""
@torch.no_grad()
def wset_card(note):
    model.eval()
    e = tok([note], padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
    with autocast(): lb, lg = model(e["input_ids"].to(device), e["attention_mask"].to(device))
    pb = torch.sigmoid(lb.float())[0].cpu().numpy(); pg = lg.float()[0].cpu()
    out = ["### Structure and levels"]
    for q, s, e_ in zip(GROUPS, slices[:-1], slices[1:]):
        p = F.softmax(pg[s:e_], -1).numpy()
        if GROUPS[q][p.argmax()] != "not_stated":
            out.append(f"- **{q.replace('__', ': ').replace('_', ' ')}**: {GROUPS[q][p.argmax()]} ({p.max():.0%})")
    for scope in ("nose", "palate"):
        hits = [(pb[j], q) for j, q in enumerate(schema["binary"]) if q.startswith(scope + "__") and pb[j] >= thresholds[j]]
        out.append(f"### {scope.title()} descriptors")
        out += [f"- {q.split('__')[-1].replace('_', ' ')} ({p:.0%})" for p, q in sorted(hits, reverse=True)[:12]] or ["- none found"]
    return "\\n".join(out)

print(wset_card("Aromas of ripe black cherry, vanilla and a touch of cedar. Full-bodied with firm tannins and a long, lingering finish."))
""")

code("""
import gradio as gr
gr.Interface(fn=wset_card, inputs=gr.Textbox(lines=5, label="Critic's tasting note"), outputs=gr.Markdown(),
             title="Critic note -> WSET card (small student model)",
             description="Unofficial. Trained to copy a larger model's reading of Wine Enthusiast notes. Non-commercial.",
             flagging_mode="never").launch(share=IN_COLAB)
""")

md("""
## What next?
* **More data:** the student keeps improving with more labelled notes. Label more with `src/jev_pilot.py` and rebuild the pack with `make_data_pack.py`.
* **Bigger or smaller student:** change `MODEL_NAME` at the top and run again.
* **Weak spots:** check the cluster table in section 8: descriptors that are rare in critic prose are the hardest.
* **Real WSET notes:** this student has only seen critic prose. Trying it on notes written in WSET exam style will show how far it generalises, and is the natural next experiment.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
               "accelerator": "GPU", "colab": {"provenance": []}}
out = Path(__file__).with_name("wset_distill.ipynb")
nbf.write(nb, out); print("wrote", out, f"({len(cells)} cells)")
