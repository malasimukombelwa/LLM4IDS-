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
