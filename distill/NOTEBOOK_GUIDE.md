# Guide to the training notebook (`wset_distill.ipynb`)

What the notebook does: a small model (the **student**) learns to answer the 231 WSET-card questions about a critic's tasting note the way a big model (**Jev**, the teacher) does.
This guide explains the new teaching parts. Everything is interactive (plotly): hover for values, click a legend entry to hide a line, drag to zoom.
Each figure is also saved as a stand-alone `.html` file in `pack/figures/` (open it in any browser).


## 0. Reading order of the notebook (what each part teaches)
| Section | Teaches | Figures (in `pack/figures/`) |
|---|---|---|
| 0 The problem | what **distillation** is (teacher Jev, student), why (cost ~$65 for all notes, offline, speed), hard vs **soft labels** | |
| 2a Jev's raw output | the **request** we send and the **raw JSON** that comes back (3 answer types: `noul`, `score`, `choice`), tokens and cost of one note | |
| 2b-2c Parquet | how the JSON becomes one table row (`b::` yes/no columns, `g::` lists of probabilities); **what Parquet is** (columnar, typed, compressed), the file structure (row groups, column chunks, footer) and its real metadata; proof that the row equals the JSON | `00b_parquet` (size vs CSV/JSON, biggest column chunks) |
| 2d-3 First look, rarity, split | one note next to the teacher's answers; most descriptors are rare; train / validation / test by winery | `00_rarity`, `01_split` |
| 4 Words to numbers | tokenizer, ids, attention mask, padding; one batch; how batches are served and reshuffled | `00c_tokens` |
| 5 The student | the **zoo of encoders** (size, layers, width, attention heads); **what distillation changes**; what a **head** is; parameter breakdown by part; **multi-headed** models; **tokens -> vectors -> average** (visual); one judge by hand | `02a_zoo`, `02b_body_components`, `02c_tokens_to_vector`, `02_body_and_heads` |
| 5b Inside the encoder | the stack of layers and one layer (Mermaid); **what an attention head is** (query/key/value, toy example by hand); the **real attention maps** of every head of every layer with a layer drop-down; **what each head specialises in** (focus, previous/next word, start/end markers) measured over 32 notes; wine words in the body's own space (PCA) | `02d_encoder_stack`, `02e_one_layer`, `02f_attention_head`, `02g_toy_attention`, `02h_attention_heads`, `02i_head_roles`, `02j_word_space` |
| 6 Loss | how the student is graded; both heads pull on the same body weights (measured) | |
| 7 Training | baseline yardstick; loss and metrics after every epoch (plotly); **what training changed** in the weights (heads vs body, by layer) | `03_loss`, `04_metrics`, `03b_weight_change` |
| 5c Architecture as matrices | every object as a **matrix with a shape** (`X: T x d`, `W_Q/W_K/W_V/W_O: d x d`, `W_1: d x d_ff`, `W_2: d_ff x d`, heads `d x 216` and `d x 76`), the real shapes of the chosen model; all matrices of one layer drawn **to scale**; one attention head computed as matrix products from the **real layer-0 weights**, with the shape after every step, and checked against the library (they agree); `W_Q` split into head stripes; **LoRA as a low-rank matrix product** | `02k_matrices_to_scale`, `02l_Wq_by_head`, `02m_lora_rank` |
| 7b Three ways to adapt the body | **full fine-tuning vs frozen body vs LoRA** trained for real with the same data and epochs; LoRA written by hand; learning curves, numbers trained, time, GPU memory, AUC | `07a_strategies`, `07b_strategy_curves`, `07c_strategy_costs` |
| 8-10 Exam, thresholds, use | the final test, what the head learned, saving and a web page | `05_test_by_family`, `06_head_similarity` |

## A. Distillation in one picture
Jev (big, paid, online) answers 231 questions per note. We keep the answers (including the *probabilities*, the soft labels) and train a small model to reproduce them. Only the student's weights change.

## B. Where the data live: JSON -> Parquet
Jev's reply is a JSON of ~16 KB per note. In the pack, one note is one **row**: columns `b::<question>` (a number, yes/no questions) and `g::<question>` (a list of level probabilities), plus the note and details.
`wset_distill_pack.parquet` is a **Parquet** file: columnar (read only the columns you need), typed (the schema travels with the file) and compressed (several times smaller than CSV/JSON).
`jev_raw_examples.json` (3 notes with the untouched replies) is only used for the teaching cells; the notebook runs without it.

## C. The encoders, and what is changed
All candidates are Transformer encoders (stack of layers = attention + feed-forward). `MODEL_NAME` switches between them (22 M to 355 M parameters). Distillation nudges the body (tiny learning rate) and learns the heads from scratch (large learning rate); the teacher is never touched.
Measured in section 7: heads move by tens of percent (they start random), the body by well under 1%, most in the upper layers.

## D. Multi-headed
One body, two output heads (yes/no and level), losses added, gradients from both flow into the same body weights. Not the same as *multi-head attention* inside each layer.

## E. Tokens -> one vector
tokenize -> ids + mask -> one vector per token (depends on neighbours) -> average the real tokens (mask out the padding) -> one summary vector for the note, read by every head.

## 1. What is a *head*?
The student has two parts.
* **Body (encoder):** a pre-trained language model. It reads a note and writes a private summary: a list of 768 numbers (384 for the small MiniLM option). The summary is about meaning in general, not about wine.
* **Heads:** small layers on top of the body that turn the summary into answers.
  * *Yes/no head*: 216 outputs, one judge per descriptor ("is vanilla on the nose?"). Each judge is **one row of 768 weights + a bias**: multiply the summary by the weights, add up, squash to a probability (sigmoid).
  * *Level head*: one output per level of every multi-level question (tannin: Low ... High). A softmax per question makes the levels add up to 1.

A head is therefore **just a table of weights, one row per question**. The notebook proves it: it computes one judge's answer by hand (a dot product) and shows that it equals the model's output (section 5), then repeats it after training (section 9).
The body is general and reusable; the heads are specific to our task. The same body can carry other heads (for example one that predicts the grape).
During training the body changes gently (small learning rate, it already knows English); the heads change fast (large learning rate, they start from random).
*Not to be confused with the "attention heads" inside the transformer body: same word, different thing.*

Figure `02_body_and_heads`: the body has ~125 M numbers, the heads about 0.2 M (0.1-0.2% of the model).
Figure `06_head_similarity`: similarity between the judges' weight rows for the 40 most common questions. Related descriptors (the same fruit on nose and palate) should point in similar directions.

## 2. Which notes are used for what (train / validation / test)
Think of school: **train** = practice questions the student learns from; **validation** = a mock exam (never learned from, used to watch progress and make small decisions); **test** = the real exam, sealed until the end.

| Set | Share | Changes the weights? | Used for | When |
|---|---|---|---|---|
| Train | 80% of wineries | **yes** | learning | every epoch, in shuffled batches |
| Validation | 10% | no | progress after every epoch, picking the best epoch, tuning the yes/no thresholds | after every epoch (and once before training) |
| Test | 10% | no | the final score | once, at the very end |

* **Split by winery, not by note**: a winery's notes sound alike; having Chateau X in both training and test would flatter the score. The notebook checks that no winery is on two sides.
* Settings: `VAL_FRAC`, `TEST_FRAC` (share of wineries), `MIN_POS` (a question is only scored in a set if it has at least this many yes and no examples there).
* **Batch / step / epoch:** the training notes are shuffled and cut into batches (e.g. 32). One batch = one step = one small nudge of the weights. One pass over all batches = one epoch. The notes are reshuffled every epoch. The last partial batch is skipped (TPU needs fixed shapes), but reshuffling brings those notes back next epoch.
* Figure `01_split`: notes and wineries per set, and how often each set is *read* vs how often it *changes the model*.

## 3. Watching the training
* Figure `03_loss`: **top** loss at every step (pale) with a moving average (bold); dashed lines = end of epochs; diamonds = validation loss at that moment. **Bottom** per-epoch train vs validation loss, with the untrained starting level.
  * both falling = learning something that generalises;
  * train falling but validation flat/rising = **overfitting**; the notebook keeps the epoch with the lowest validation loss.
* Figure `04_metrics`: the report card after every epoch (epoch 0 = untrained, about 0.5 AUC = guessing), on the validation notes:
  * **mean AUC** over the yes/no questions (dashed grey = the word-counting baseline the student must beat),
  * **level exact** and **level within one step** for tannin, body, colour...,
  * **AUC by aroma family** (right): families whose line rises early are easy; flat lines are hard.
* A table with the same numbers is printed under the figure.

## 4. The final exam
After training, section 8 uses the **test** notes once: AUC per question (student vs baseline), by aroma family (figure `05_test_by_family`), the level table, and a few notes where student and Jev disagree most.

## 5. Words
Epoch = one pass over the training notes. Loss = one number for "how far from Jev's answers" (training pushes it down). AUC = chance the model ranks a yes-note above a no-note (1.0 perfect, 0.5 guessing).
Sigmoid = squashes a score to 0-1. Softmax = squashes scores so they add to 1. Overfitting = memorising the training notes.

Licence: the notes are Wine Enthusiast reviews (CC BY-NC-SA 4.0); everything is non-commercial and unofficial (not affiliated with WSET).

## F. Attention heads, in short
Each layer lets every token look at the others. A **head** is one such look: the token makes a *query*, compares it with every token's *key* (dot product), turns the scores into weights that add to 1 (softmax) and takes that weighted mix of the *values*.
A layer has several heads in parallel (12 in RoBERTa), each with its own query/key/value matrices, so they can specialise. Section 5b draws the real attention of the model you chose and measures what each head does.

## G. Three ways to adapt the body (section 7b)
| | trains | leaves alone | typical trade-off |
|---|---|---|---|
| A. Full fine-tuning | whole body (tiny learning rate) + heads | nothing | best scores, most memory and time |
| B. Frozen body | heads only | the whole body | fastest and lightest, a bit lower scores |
| C. LoRA | heads + adapters (`B x A`, rank 8) beside the attention matrices | the original body weights | a few % of the numbers trained, scores usually close to A |
The notebook trains B and C for real (same data, same epochs, same seed) and compares them with the main run on the **validation** notes. The test notes remain sealed for the one strategy you finally choose. Set `COMPARE_STRATEGIES = False` to skip.

## H. Mermaid diagrams
Mermaid turns a few lines of text into a diagram. The notebook has a small helper `mermaid(...)` that loads mermaid.js from a CDN inside the notebook (needs internet) and saves each diagram as HTML in `pack/figures/`.

## I. The architecture as matrices (section 5c)
Read every layer as a chain of matrix products. With `T` tokens, width `d`, `H` heads, `d_h = d/H`:
`X (T x d)` -> `Q, K, V = X.W_Q, X.W_K, X.W_V (each T x d)` -> split the columns into `H` blocks of `d_h` -> per head `scores = Q_h.K_h^T / sqrt(d_h) (T x T)` -> softmax per row -> `weights.V_h (T x d_h)` -> glue the heads (`T x d`) -> `.W_O (d x d)` -> add `X`, normalise -> feed-forward `(d x d_ff)` then `(d_ff x d)` -> after the last layer average the rows -> `summary (1 x d)` -> `.W_bin (d x 216)` and `.W_grp (d x 76)`.
So a "head" in the output sense is literally a matrix whose **column j is judge j**; an attention head is a **block of `d_h` columns** of `W_Q, W_K, W_V`; LoRA adds `A.B` (`d x r` times `r x d`), a `d x d` matrix of rank at most `r`.

## J. Plots after a headless run (why they used to disappear)
Interactive plotly output is a small JavaScript program stored in the notebook. When the notebook is run without a browser (`run_headless.py`) and opened later, Jupyter treats it as *untrusted* and refuses to run that program, so the plot area is empty (the same for the Mermaid diagrams).
Fix, in three layers: (1) every figure is also embedded as a **PNG picture** inside the notebook (setting `STATIC_IMAGES`, automatic in headless runs; needs `kaleido` + Chrome); (2) each figure is saved in `pack/figures/` as `.png`, `.html` (stand-alone interactive page) and `.json` (`plotly.io.read_json` restores it); (3) `run_headless.py` also writes `runs/NAME.html`, the whole run as one web page, and marks the notebook trusted on the machine that ran it.
The training record is saved as `pack/run_state.json` so plots can be redrawn even if outputs are lost.

## K. All the models to scale (section 5d)
Same drawing as section 5c, for the seven bodies in the zoo, on **one common scale**: the matrices of one layer (attention W_Q/W_K/W_V/W_O, feed-forward W_1/W_2) plus our two heads. A table lists every matrix shape per model, the computed size against the published one, and the share taken by the **word table** (vocabulary x d). A stacked bar chart shows where the parameters live (the word table is more than half of MiniLM and DeBERTa-v3), and a memory estimate (weights + gradients + AdamW: ~16 bytes per trained parameter, ~4 for a frozen one) is drawn against the 16 GB of the RTX 5060 Ti. Activations come on top; section 7b prints the measured peak.

## L. Activation functions and back-propagation (sections 6b and 7b)
* **Activation function** = a fixed formula between two matrix products (sigmoid, GELU, softmax); no weights. Without it stacked layers collapse into one matrix.
* **Where:** softmax in attention (over tokens) and in the level head (over levels); GELU inside every feed-forward block; sigmoid in the yes/no head (folded into the loss). **Not in:** the tokenizer (a fixed rulebook and lookup, nothing to learn), pooling, or the LoRA adapters (plain `B.(A.x)`).
* **Back-propagation** = run backwards from the loss and multiply by each step's slope to get a **gradient** for every trainable number. Section 6b does it by hand on one judge (`dLoss/dz = p - y`) and checks it against autograd, then measures the real gradient per part and per layer. Only the word-table rows of pieces present in the batch receive a gradient.
* **Per strategy (7b):** full fine-tuning: gradients everywhere; frozen body: the backward pass stops at the heads (fast, light); LoRA: the signal still travels back through every frozen layer to the adapters, and at step 0 the gradient of `A` is exactly zero because `B` starts at zero.

## M. Rules and a random forest next to the fine-tuned model (sections 7c and 8)
Three ways to answer the same 231 questions, built on the same split:
* **Original rules (v0):** the project's deterministic lexicon matcher (`wsetkit`): hand-written phrases per WSET descriptor, with negation handling and nose/palate scope. Nothing is learned. Its findings are translated into the 231 questions (a phrase hit scores the trust measured on the training notes; the level cues pick a level; questions with no phrases get the most common answer).
* **Improved rules (v1):** count every word and two-word phrase in the training notes; for each question keep the phrases with enough **support** and **precision** against Jev's answers (`RULE_MIN_SUPPORT`, `RULE_MIN_PRECISION`); drop two-word rules a one-word rule already covers; score a note by the strongest rule that fires. Still a fixed list of phrases (saved as `rules_v1.csv`), still fully explainable; the candidates the lexicon lacks are saved for a taster to review (`lexicon_candidates_for_review.csv`).
* **Random forest:** many decision trees on 0/1 word features, learning all 292 numbers (216 yes/no + 76 level probabilities) at once from Jev's soft answers; one small tree is printed so you can read it.
A **learning curve** shows each method's validation AUC against the number of labelled training notes (the original rules are flat; the others climb), and section 8 ends with the five methods and the word-counting baseline side by side on the test notes, with training time, explainability and GPU need.
