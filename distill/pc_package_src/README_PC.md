# WSET distillation: Windows PC package (NVIDIA RTX GPU)

Trains a small model to give WSET-style answers for a critic's tasting note. Non-commercial; the data are Wine Enthusiast reviews (CC BY-NC-SA 4.0). Unofficial, not affiliated with WSET.

> **Handing this to an AI agent?** Point it at `AGENT_INSTRUCTIONS.md` (full step-by-step install, test and report-back instructions).
> `run_headless.py` runs the whole notebook without a browser.

## First time (about 15 minutes, mostly downloads)
1. Install **Python 3.11 or 3.12** from python.org if you don't have it (tick *Add python.exe to PATH*).
2. Make sure the NVIDIA driver is recent (version 570 or newer for the RTX 5060 Ti).
3. Unzip this folder somewhere simple (e.g. `C:\wset_distill`; avoid very long paths).
4. Double-click **`setup_windows.bat`**. It creates a private Python environment, installs PyTorch with CUDA 12.8 and the other packages, then tests the GPU.
   You should see `GPU: NVIDIA GeForce RTX 5060 Ti ... The GPU is ready.`

## Train
1. Double-click **`run_notebook.bat`**. Jupyter opens in your browser with `wset_distill.ipynb`.
2. In the settings cell leave `USE_TPU = False`. For a 1-minute test set `MAX_ROWS = 500`; set it back to `None` for the real run.
3. `Run > Run All Cells`. The notebook explains each step. It will find the GPU by itself and use bf16.
   On the 5060 Ti you can raise `BATCH_SIZE` to 64.
4. The last cells save `pack\wset_student.pt` and open a small web page where you can paste a note.

A checkpoint is saved after every epoch, so if you close the window just run the notebook again and it resumes.

## Use the trained model without the notebook
Double-click **`run_app.bat`** (needs `pack\wset_student.pt`). A page opens in your browser: paste a critic's note, get a WSET-style card.

## More training data
`pack\` holds the notes plus Jev's answers. Rebuild it with more labelled notes on the Mac (`distill/make_data_pack.py`, see `distill/README.md`) and replace the two files in `pack\`.

## If something goes wrong
- `NO GPU FOUND`: update the NVIDIA driver, then re-run `setup_windows.bat`.
- *Out of memory* in the notebook: lower `BATCH_SIZE` (32, then 16) or `MAX_LEN`.
- Anything else: send the last lines of the error.
