# Data licences and attributions

This file lists every upstream data source and model, the terms we understand apply, and what we publish. **Please check the original terms yourself; this is a good-faith summary, not legal advice.** The project is **non-commercial** (see [DISCLAIMER.md](DISCLAIMER.md)).

## What is under which licence in this repository
| Part of the repository | Licence |
|---|---|
| Source code and our documentation (`src/`, `scripts/`, `dashboard/*.js|css|html`, `Dockerfile`, `Makefile`, `docs/`, our `lexicon/*.yaml`) | MIT ([LICENSE](LICENSE)) |
| Derived data and models: `dashboard/data/*` (aggregates, statistics, 25k-wine index and its numeric vectors, model weights), `experiments/**`, `reports/**` | **CC BY-NC-SA 4.0**: attribution, non-commercial, share-alike, and the attribution lines below |
| Place coordinates (latitudes and longitudes of regions, wherever they appear: `terroir.json`, `place_attrs.json`, `recsys_meta.json`, `spaces.json`, model files) | derived from OpenStreetMap: **© OpenStreetMap contributors, ODbL 1.0** (attribution; share-alike for the coordinate data) |
| Lamat name and logo (`dashboard/assets/`) | not licensed: all rights reserved by their owner |
| Review text, the WSET documents, the raw datasets | **not included**; see below |

## Upstream sources
| Source | Used for | Terms (as we understand them) | In this repo? |
|---|---|---|---|
| **Wine Enthusiast reviews** via Kaggle "Wine Reviews" (`winemag-data-130k-v2`, `winemag-data_first150k`), scraped June and November 2017 | text → WSET features, embeddings, all analyses | Kaggle lists the dataset as **CC BY-NC-SA 4.0**. Rights in the reviews remain with Wine Enthusiast and the authors. | **No review text.** Derived numbers only. |
| Hugging Face mirror of the same reviews (`gronlp_winemag`, optional) | cross-check only; adds no unique review | check its dataset card | No |
| **WSET** Level 4 Systematic Approach to Tasting Wine (Issue 1.2, 2023) and Wine-Lexicon; Level 3 card (2016) in the archive | taxonomy that our parser refers to | © Wine & Spirit Education Trust; reproduction requires WSET's written permission (printed on the document) | **Not stored.** Our word lists refer to the structure for non-commercial educational analysis; see DISCLAIMER section 2. |
| **NASA POWER** (MERRA-2 reanalysis, 0.5° × 0.625° grid), 2000–2017 | growing-season weather | NASA open data; please acknowledge: *"These data were obtained from the NASA Langley Research Center (LaRC) POWER Project funded through the NASA Earth Science/Applied Science Program."* | Derived anomalies and indices only |
| **OpenStreetMap / Nominatim** | geocoding region names | ODbL 1.0, **© OpenStreetMap contributors**; follow the [Nominatim usage policy](https://operations.osmfoundation.org/policies/nominatim/) | Region coordinates (ODbL) |
| `sentence-transformers/all-MiniLM-L6-v2`, `google-bert/bert-base-uncased` | text embeddings | Apache-2.0 | Not redistributed (downloaded at run time); numeric embeddings of 25,000 notes are included under CC BY-NC-SA 4.0 |
| **TypeSafe AI "Jev"** API | one-off cross-check of the rule-based reader on 1,500 notes | provider's terms of service | **No outputs included** |
| PT Sans (font, via Google Fonts) | typography | SIL Open Font Licence 1.1 | Loaded from Google Fonts |
| Plotly.js (via CDN) | charts | MIT | Loaded from a CDN |

## Required attributions (also shown on the site)
- Review data: Wine Enthusiast reviews as published on Kaggle ("Wine Reviews"), CC BY-NC-SA 4.0.
- Weather: NASA POWER Project (NASA Langley Research Center), as worded above.
- Geography: © OpenStreetMap contributors (ODbL).

## If you fork this project
- Keep the non-commercial restriction for anything derived from the reviews, and keep the attributions.
- Replace the Lamat branding.
- For commercial use of the *code*, retrain on data you are entitled to use; do not reuse our derived data or models.
- If you add a data source, add it to this table with its licence (see [CONTRIBUTING.md](CONTRIBUTING.md)).
