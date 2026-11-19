# Disclaimer

**Wine Lab is a free, non-commercial, not-for-profit research and educational demonstration.**
It has no advertising, no subscriptions, no paid features, and nobody sells the data or the models. It is published openly to show how data science combines domain knowledge, mathematics and computer science.

## 1. Non-commercial use of the data (please read before reusing anything)
The tasting-note data come from public datasets released under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)** (see [DATA_LICENSES.md](DATA_LICENSES.md)).
Everything derived from those data that this repository publishes (aggregates, statistics, embeddings, trained models, the 25,000-wine index, experiment records) is therefore offered **only for non-commercial use, with attribution, under the same licence**.
- You may study, run, adapt and share this work non-commercially under those terms.
- You may **not** use the derived data or models commercially. The *code* is MIT-licensed and may be reused commercially, but a commercial reuse must retrain on data you have the right to use.
- Anyone who forks this project is responsible for complying with the upstream licences listed in [DATA_LICENSES.md](DATA_LICENSES.md). Nothing here is legal advice.

## 2. No affiliation or endorsement
This project is independent. It is **not** affiliated with, sponsored by, or endorsed by the Wine & Spirit Education Trust (WSET), Wine Enthusiast, Kaggle, Hugging Face, NASA, OpenStreetMap or the OpenStreetMap Foundation, TypeSafe AI, Vercel, or any winery, critic or producer.
All names and trademarks belong to their owners and are used only to identify things factually.

### About WSET
"WSET" and the *Systematic Approach to Tasting Wine* (SAT) card and *Wine-Lexicon* are the intellectual property of the Wine & Spirit Education Trust. The WSET documents state that they may only be reproduced with WSET's written permission.
- **We do not store or distribute the WSET documents.** The repository contains our own parser and word lists that *refer to* the SAT structure (the names of the aroma clusters and scales, and descriptors drawn from the Wine-Lexicon) for non-commercial educational analysis.
- Our mapping of reviews onto the card is **our own interpretation**. It has not been reviewed, validated or approved by WSET or by WSET-qualified examiners, and it must not be used for exam preparation or as an official WSET tool.
- If WSET (or anyone acting for it) asks us to remove or change any element, we will do so promptly (see section 5).

## 3. About the reviews
The underlying reviews were written by critics and published by Wine Enthusiast; the rights in them remain with their authors and publisher. **This repository and the public site do not contain any review text**; they contain statistics, per-wine metadata (grape, place, producer, vintage, price, score) and numerical representations derived from the notes. Reviews are used here for non-commercial research and teaching.

## 4. Accuracy, advice and responsibility
- The guesses, similarity lists and "findings" are **statistical outputs of models trained on critics' notes**. They are often wrong, especially for producers, vintages and prices, and they reflect how those critics write, not what you will taste.
- Most results are **associations**, not proven causes. Only the link between warmer growing seasons and lower acidity in the notes passed our placebo test; the link to critic scores did not.
- Nothing here is professional, financial, investment, medical or purchasing advice. Please drink responsibly and only if you are of legal age.
- The software and data are provided **"as is"**, without warranty of any kind (see [LICENSE](LICENSE)).

## 5. Takedown and corrections
If you are a rights holder, or you believe something here should not be published, please open an issue at <https://github.com/gllamat/wine-lab/issues> using the *Takedown / rights request* template, or contact Lamat UK through <https://lamatuk.com>. We will review requests promptly and remove or amend content where appropriate.

## 6. Trademarks and branding
"Lamat" and the Lamat logo are not covered by the open-source licences. If you fork this project, remove or replace `dashboard/assets/` and the Lamat branding.
