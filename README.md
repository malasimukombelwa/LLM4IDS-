# LLM4IDS-
A Multimodal Framework with Lightweight LLMs
Input: network flow table (e.g., NSL‑KDD / CICIDS2017),
Output: intrusion class (multiclass) + confidence + explanation hooks.

Loss‑Aware Textualization
Maps each flow to a compact sentence preserving discriminative attributes (e.g., protocol, src/dst roles, duration, byte/packet stats, flags). Includes a label‑aware weighting to keep features that maximize F1.

Semantic Encoding (Text)
MiniLM/SentenceTransformer encodes the sentence into a dense vector (384–768‑d). Embeddings are cached and deduplicated for speed.

Tabular Encoder
Numerical/categorical features → MLP/CNN/TabTransformer to a latent vector.

Multimodal Fusion
Concatenate or gated‑fusion of text & tabular latents; optional projection to shared space.

QCA Classifier (Query‑Compressed Attention)
Lightweight attention that uses a small query set to reduce compute without sacrificing accuracy.

Latency‑Optimized Inference (LoLI)
Embedding cache, near‑duplicate removal, and fast‑path routing for frequent patterns.

Metrics: Accuracy, Macro‑F1, PR‑AUC, Behavioral‑F1, Threat‑Coverage Score (TCS).
Datasets: NSL‑KDD, CICIDS2017, UNSW‑NB15 (links below).


# LLM4IDS — A Multimodal, Lightweight On‑Prem LLM Framework for Intrusion Detection

**LLM4IDS** is a *semantic‑aware* intrusion detection pipeline that combines:
1) **Textualization** of network flows (NetFlow‑like) into short sentences,  
2) **Embeddings** produced by a **lightweight on‑prem encoder** (*MiniLM / `all‑MiniLM‑L6‑v2`*),  
3) **Multimodal fusion** (text embeddings + standardized numeric features),  
4) **Lightweight classifiers** (a compact Dense‑Transformer head with multi‑head attention) and **baselines** (LogReg, RF+ADASYN, XGBoost, MLP),  
5) **Robust evaluation** under class imbalance (micro/macro/weighted F1, balanced accuracy, MCC, one‑vs‑rest ROC).

> **Status**: public research code (paper under review at EGC 2026).  
> **Goal**: an **end‑to‑end reproducible** pipeline that is **deployable on‑prem** (CPU) and avoids any external LLM service.

---

##  Key features

- **On‑prem, privacy‑first**: MiniLM encoder runs locally (no external APIs).
- **Multimodal by design**: concatenate **embeddings (384‑D)** + **~39 numeric features** → **423‑D vector**.
- **Efficient**: compact attention head + dense layers; **millisecond‑level latencies** on CPU.
- **Reproducible**: fixed seed (`42`), stratified splits, unified evaluation protocol.
- **Baselines included**: Logistic Regression, Random Forest (**ADASYN** oversampling), XGBoost, MLP (scikit‑learn).
- **Interpretability**: **LIME** local explanations; optional **t‑SNE** visualizations.

---

##  Architecture (overview)

```
Flows → row_to_text → MiniLM Embeddings ┐
                                        ├─ Concat → [Dense‑Transformer] → 5‑class predictions
Numeric features → StandardScaler ──────┘
```

- Label space: `['normal', 'DoS', 'Probe', 'R2L', 'U2R']`  
- **Textualization (`row_to_text`)** injects protocol/state context that pure tabular features often miss.  
- **Fusion** makes statistical (tabular) and semantic (text) signals complementary.

---

##  Installation

Requirements: Python **3.9–3.11**, `pip`, CPU or GPU (optional).

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install sentence-transformers scikit-learn imbalanced-learn xgboost tensorflow matplotlib seaborn pandas numpy lime shap
```

**Conda** option:

```bash
conda create -n llm4ids python=3.10 -y
conda activate llm4ids
pip install -U sentence-transformers scikit-learn imbalanced-learn xgboost tensorflow matplotlib seaborn pandas numpy lime shap
```

---

##  Data (NSL‑KDD)

The pipeline loads the official NSL‑KDD splits over HTTP:

- Train: `https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt`  
- Test : `https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt`

Labels are mapped to five classes: `normal`, `DoS`, `Probe`, `R2L`, `U2R` (see code for the exact mapping).

---

##  Quick start

### Option A — Notebook (Colab / local)
Open `notebooks/llm4ids_multiclasses.ipynb` (or the *clean* version) and run cells in order: **load**, **embeddings**, **fusion**, **train**, **evaluate**.

### Option B — Script
Run the default end‑to‑end pipeline (Dense‑Transformer + baselines):

```bash
python scripts/llm4ids_multiclasses.py
```

**Outputs**: classification reports, confusion matrices, one‑vs‑rest ROC curves, and a comparison table (Accuracy / F1 / inference time / model size).

---

##  Reproducibility & protocol

- **Global seed**: `42` (NumPy / TensorFlow / Python).  
- **Split**: `train/val = 80/20`, stratified **within the official train set**.  
- **Normalization**: `StandardScaler` fit on train, applied to val/test.  
- **Textualization**: `row_to_text` → one sentence per row.  
- **Encoder**: `sentence-transformers` → `all‑MiniLM‑L6‑v2` (384‑D).  
- **Fusion**: concat `[embeddings (384)] + [numeric features]` → **423‑D**.  
- **LLM4IDS head**: multi‑head attention (4×64) → Dense(128) → Dense(64) → Softmax.  
- **Baselines**:
  - **MLP (sklearn)**: hidden layers `(256, 128)`, `relu`, `adam`.
  - **RF (+ADASYN)**: randomized search (e.g., `n_estimators ∈ {200, 300}`, …), `scoring='f1_weighted'`.
  - **XGBoost**: `multi:softprob`, `eval_metric=mlogloss`, `n_estimators=200`, `max_depth=6`.
  - **LogReg**: multinomial (`lbfgs`, `max_iter=1000`).

**Metrics**: micro/macro/weighted F1, balanced accuracy, **MCC**, confusion matrices, **OVR‑ROC**.  
*Note*: **micro‑F1** and **balanced accuracy** serve different purposes; avoid conflating them.

---

## What to expect (NSL‑KDD guidelines)

LLM4IDS is typically **competitive** with strong MLP baselines and **clearly stronger** than classical baselines, while improving **recall on rare classes** (**R2L/U2R**). Figures are rendered with an **A4‑friendly** and **consistent palette**:  
`normal` `#00b894`, `DoS` `#d63031`, `Probe` `#0984e3`, `R2L` `#fdcb6e`, `U2R` `#6c5ce7`.

Results may vary with environment and seeds.

---

## Options & good practices

- **t‑SNE** is expensive → enable only on a small sample.  
- **CNN/GRU** are **optional** demos and are not required for the main contribution.  
- **On‑prem ops**: use **quantization**, **embedding cache**, and **ANN reuse** to minimize latency and cost in production.

---

## Suggested repository layout

```
.
├── notebooks/
│   └── llm4ids_multiclasses.ipynb
├── scripts/
│   └── llm4ids_multiclasses.py
├── artifacts/                 # models, metrics, figures
├── requirements.txt
├── LICENSE
└── README.md
```

---

##  Traceability

For an auditable run, **pin a tag/commit** (e.g., `v1.0`) and record: seed, encoder (`all‑MiniLM‑L6‑v2`), split, scaler, hyper‑parameters, library versions. Example commit note:

```
seed=42; encoder=all-MiniLM-L6-v2; scaler=StandardScaler; split=train/val 80/20 strat;
head=attn(4x64)+dense(128,64); epochs=25; batch=64; sklearn=1.4+; tf=2.15+
```

---

##  License

This repository is released under the **MIT** license (or another permissive license; adapt if needed).

---

##  Maintainer / Contact

**Jean‑Jarcke Malasi Mukombelwa** (UQO) — Issues and PRs are welcome.

---

##  Citation

If you use LLM4IDS in academic work, please cite:

---

##  Acknowledgments

- NSL‑KDD maintainers and the communities behind **sentence-transformers**, **scikit‑learn**, **TensorFlow**, **XGBoost**, **imbalanced‑learn**, and **LIME**.  
-  readability, reproducibility, and graphical consistency.

— Last updated: 2025-11-07 21:56
