"""Run the whole notebook without a browser (for scripts and agents) and save the executed copy with all outputs.

    python run_headless.py                       # full run with the notebook's own settings
    python run_headless.py --max-rows 500 --epochs 1       # quick smoke test
Options: --model NAME  --epochs N  --batch-size N  --max-rows N  --no-compare  --out runs/NAME.ipynb
Settings are applied by editing the settings cell in memory; the notebook file itself is not changed.
The final web-interface cell is skipped (it would wait forever); use run_app.bat afterwards.
Results: the executed notebook (open it to see tables/plots; plots are embedded as pictures too, so they survive re-opening) plus a stand-alone .html copy next to it, pack/wset_student.pt, pack/best_student.pt, pack/checkpoint.pt."""
import argparse, os, re, sys, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--model"); ap.add_argument("--epochs", type=int); ap.add_argument("--batch-size", type=int)
ap.add_argument("--max-rows", type=int); ap.add_argument("--out"); ap.add_argument("--no-compare", action="store_true", help="skip section 7b (the two extra training runs: frozen body and LoRA)")
ap.add_argument("--no-images", action="store_true", help="do not embed PNG pictures of the plots (use if kaleido/Chrome are not available)")
a = ap.parse_args()

nb = nbformat.read(HERE / "wset_distill.ipynb", 4)


def setting(src, name, value):                      # replace e.g.  EPOCHS     = 4   ->  EPOCHS = 2  (keeps the trailing comment)
    new, n = re.subn(rf"^({name}\s*=\s*)[^#\n]*?(\s*(#.*)?)$", lambda m: f"{m.group(1)}{value!r}{m.group(2) or ''}", src, count=1, flags=re.M)
    if n != 1: sys.exit(f"could not find setting {name} in the notebook")
    return new


for c in nb.cells:
    if c.cell_type != "code": continue
    if re.search(r"^MODEL_NAME\s*=", c.source, re.M):                         # the settings cell
        if a.model: c.source = setting(c.source, "MODEL_NAME", a.model)
        if a.epochs: c.source = setting(c.source, "EPOCHS", a.epochs)
        if a.batch_size: c.source = setting(c.source, "BATCH_SIZE", a.batch_size)
        if a.max_rows: c.source = setting(c.source, "MAX_ROWS", a.max_rows)
        if a.no_compare: c.source = setting(c.source, "COMPARE_STRATEGIES", False)
        if a.no_images: c.source = setting(c.source, "STATIC_IMAGES", False)
    if "import gradio" in c.source:
        c.source = "print('web interface skipped in headless mode; run run_app.bat instead')"

out = Path(a.out) if a.out else HERE / "runs" / time.strftime("run_%Y%m%d_%H%M%S.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
print("running; the executed notebook will be saved to", out, flush=True)
os.environ["WSET_HEADLESS"] = "1"                   # tells the notebook it runs without a browser: it then embeds a PNG picture of every plot
client = NotebookClient(nb, timeout=-1, kernel_name="python3", resources={"metadata": {"path": str(HERE)}})
try:
    client.execute()
finally:
    nbformat.write(nb, out)                         # saved even if a cell failed, so the error is visible in it
    try:                                            # mark the executed notebook as trusted on THIS computer, so interactive plots also work when re-opened here
        from nbformat.sign import NotebookNotary
        NotebookNotary().sign(nb)
    except Exception as e:
        print("could not sign the notebook (harmless):", repr(e)[:100])
    try:                                            # a stand-alone web page of the whole run (text, tables, plots, diagrams): opens in any browser, needs internet for plotly.js/mermaid.js
        from nbconvert import HTMLExporter
        body, _ = HTMLExporter().from_notebook_node(nb)
        out.with_suffix(".html").write_text(body, encoding="utf-8"); print("web page:", out.with_suffix(".html"))
    except Exception as e:
        print("could not write the HTML copy (harmless):", repr(e)[:100])
print("done:", out)
