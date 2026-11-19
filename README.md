# Wine Lab

**Data science applied to wine: what 169,000 critics' tasting notes can tell us, and how the methods work.**
A free, open-source, **non-commercial** demonstration by [Lamat UK](https://lamatuk.com).

[![CI](https://github.com/gllamat/wine-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/gllamat/wine-lab/actions/workflows/ci.yml)
[![Code: MIT](https://img.shields.io/badge/code-MIT-009999.svg)](LICENSE)
[![Data: CC BY-NC-SA 4.0](https://img.shields.io/badge/derived%20data-CC%20BY--NC--SA%204.0-lightgrey.svg)](DATA_LICENSES.md)

> **Not-for-profit and non-commercial.** This project has no ads, no fees and sells nothing. The tasting-note data are CC BY-NC-SA 4.0, so everything derived from them is for **non-commercial use only**, with attribution and share-alike. It is **not affiliated with or endorsed by WSET, Wine Enthusiast, Kaggle, NASA or OpenStreetMap**. No review text is included. Read [DISCLAIMER.md](DISCLAIMER.md) and [DATA_LICENSES.md](DATA_LICENSES.md) before reusing anything.

## What it is
One site, two audiences (switch in the header):

- **Wine lover**: describe a wine in words and see what a model thinks it is; find wines like one you love and steer the search; explore which regions taste alike; see what vintages and warm summers do; read a tasting note the way a WSET taster would.
- **Data science**: a seven-chapter path from raw data to causal claims, each with the mathematics and code behind it: data → words to numbers (a deterministic WSET parser) → vector spaces → place, vintage and weather → prediction (with calibration and sealed blind-test rounds) → recommender systems → causal inference (DAGs, do-calculus, placebo tests).

Data science sits where **domain knowledge, mathematics and computer science** meet; every page says which part of each it uses.

## Headline results (held-out data)
| | Simple model (runs in your browser) | Full model (adds a text embedding) |
|---|---|---|
| Grape right first guess | 36% | 58% (63% on blind round 1) |
| Country right first guess | 60% | 78% |
| Median miss distance | 666 km | 242 km |

Intervals for vintage and price contain the truth about 80% of the time. Producers are close to unguessable. Warmer seasons go with lower acidity in the notes (and this is the one weather effect that passes our placebo test). See the site for details and caveats.

## Quick start
```bash
git clone https://github.com/gllamat/wine-lab.git && cd wine-lab/dashboard
python3 -m http.server 8765        # open http://localhost:8765
```
The dashboard is a static site (no build step) and ships with the derived data it needs. To reproduce the analysis you need to download the public datasets yourself: see [docs/DATA.md](docs/DATA.md). To deploy, point any static host (for example Vercel, with Root Directory `dashboard`) at the `dashboard/` folder.

## Repository map
| Path | What |
|---|---|
| `dashboard/` | the static site: `index.html` (tools), `site.js`/`site.css` (two-audience shell), JavaScript ports of the matcher, guess model, causal tools and recommender, derived data in `data/` |
| `src/` | Python: matcher (`wsetkit.py`), guess model, blind rounds, recommender, causal tools, terroir analysis, corpus and pipeline scripts |
| `lexicon/` | word lists and structure for the parser; the causal graphs (`dag.yaml`, `dags/`) |
| `experiments/` | records of the blind-tasting rounds (ledger, manifests, scores) |
| `docs/` | [DATA.md](docs/DATA.md) (get the data, run the pipeline), [PIPELINE.md](docs/PIPELINE.md) (module notes) |
| `Dockerfile`, `docker-compose.yml`, `Makefile`, `requirements/` | reproducible environments (targets `core`, `causal`, `embed`, `serve`) |

Everything is **deterministic** (pinned requirements, fixed seeds, pinned model revisions), and Python and JavaScript implementations are tested against each other (`node dashboard/parity.js`, `guess_parity.js`, `causal_parity.js`, `recsys_parity.js`).

## Licences and attributions
- **Code**: MIT ([LICENSE](LICENSE)).
- **Derived data and models**: CC BY-NC-SA 4.0; place coordinates © OpenStreetMap contributors (ODbL 1.0) ([DATA_LICENSES.md](DATA_LICENSES.md)).
- **Review data**: Wine Enthusiast reviews via Kaggle "Wine Reviews" (CC BY-NC-SA 4.0). **Weather**: NASA POWER Project (NASA Langley Research Center). **Geography**: © OpenStreetMap contributors.
- **WSET**: the SAT card and Wine-Lexicon belong to the Wine & Spirit Education Trust and are not included; our interpretation is unofficial ([DISCLAIMER.md](DISCLAIMER.md)).
- **Lamat** name and logo are not covered by the open-source licences.
- Third-party software: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Contributing, citing, contact
See [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md). Rights concerns or takedown requests: use the issue template or see [DISCLAIMER.md](DISCLAIMER.md) section 5. Cite with [CITATION.cff](CITATION.cff). More about Lamat UK: <https://lamatuk.com>.
