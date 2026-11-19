# Contributing

Thank you for helping. Wine Lab is a **non-commercial, educational** project, and contributions must keep it that way.

## Ground rules (these protect everyone's licences)
1. **Never commit review text**, the raw datasets, the WSET documents, API keys, tokens or `.env` files. The `.gitignore` and `dashboard/.vercelignore` already exclude the usual suspects; check `git status` before you commit.
2. **Data contributions must be compatible with a non-commercial, share-alike project.** When you add a source, add a row to [DATA_LICENSES.md](DATA_LICENSES.md) with its licence and what we publish from it. If a source forbids redistribution, we use it locally and publish only what its licence allows.
3. **Respect the disclaimers.** Do not add claims of affiliation with or endorsement by WSET, Wine Enthusiast or anyone else, and do not present model guesses as advice.
4. By contributing code you agree it is released under the MIT licence; derived data you contribute fall under CC BY-NC-SA 4.0 (see [DATA_LICENSES.md](DATA_LICENSES.md)).

## Setting up
```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements/core.txt            # add requirements/embed.txt and requirements/causal.txt as needed
python scripts/smoke.py
```
Or use Docker: `make build && make smoke`. The corpus is not in the repository: see [docs/DATA.md](docs/DATA.md).

## Before you open a pull request
- Run the parity tests that apply to what you changed: `node dashboard/parity.js` (matcher), `node dashboard/guess_parity.js`, `node dashboard/causal_parity.js`, `node dashboard/recsys_parity.js`. Python and JavaScript must give the same answers.
- Keep results **deterministic** (fixed seeds, pinned models). If you change a model or the lexicon, say how the held-out numbers changed. Never tune on blind-round notes (see `src/blind.py`).
- Prefer plain language in the dashboard: every page serves both a wine lover and a data-science student.
- Be kind and follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ideas that would help
More regions' vineyard and harvest data (see `lexicon/dag.yaml`, section `data_plan`), validation of the WSET mapping by qualified tasters, translations of the explanations, accessibility improvements.
