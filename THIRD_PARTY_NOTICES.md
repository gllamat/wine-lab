# Third-party software

Wine Lab builds on open-source software. Each package keeps its own licence; the main ones are listed here (versions are pinned in `requirements/*.txt`). To list every licence in your environment run `pip install pip-licenses && pip-licenses`.

| Package | Licence | | Package | Licence |
|---|---|---|---|---|
| NumPy, pandas, SciPy, scikit-learn | BSD-3-Clause | | PyYAML | MIT |
| statsmodels, patsy | BSD-3 / BSD-2 | | NetworkX | BSD-3-Clause |
| umap-learn, pynndescent | BSD-3 / BSD-2 | | numba, llvmlite | BSD-2-Clause |
| PyArrow | Apache-2.0 | | requests | Apache-2.0 |
| PyTorch | BSD-style | | Transformers, sentence-transformers, huggingface_hub | Apache-2.0 |
| Plotly (Python and JS) | MIT | | folium, openpyxl | MIT |
| Pillow | MIT-CMU (HPND) | | Jinja2, joblib | BSD-3-Clause |

Models downloaded at run time (not redistributed): `sentence-transformers/all-MiniLM-L6-v2` and `google-bert/bert-base-uncased`, both Apache-2.0. Font: PT Sans, SIL Open Font Licence 1.1 (loaded from Google Fonts).

This list is a convenience, not a substitute for the licences themselves. Data sources and their terms are in [DATA_LICENSES.md](DATA_LICENSES.md).
