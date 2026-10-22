import os
import re
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

# Ensure utf-8 output encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Import ConG-MoE components from existing pipeline
from tanglish_sentiment_pipeline import (
    load_dataset,
    demojize,
    normalize_text,
    naive_clean,
    CongMoeClassifier,
    TARGET_CLASSES,
    resolve_path
)

LABEL_TO_IDX = {label: i for i, label in enumerate(TARGET_CLASSES)}
IDX_TO_LABEL = {i: label for i, label in enumerate(TARGET_CLASSES)}

# -------------------------------------------------------------
# 1. P2 Preprocessing (Shanmugavadivel et al., 2022)
# Emojis removed, repeated chars reduced, punctuations/numbers removed
# -------------------------------------------------------------
def p2_clean_text(text: str) -> str:
    text = str(text)
    try:
        import emoji
        text = emoji.replace_emoji(text, replace="")
    except Exception:
        text = re.sub(r"[\U00010000-\U0010ffff]", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text.lower()).strip()


# -------------------------------------------------------------
# 2. Vocabulary & PyTorch Dataset for P2 Neural Models
# -------------------------------------------------------------
class P2Vocab:
    def __init__(self, max_features=20000):
        self.max_features = max_features
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def fit(self, texts):
        freq = {}
        for text in texts:
            for word in text.split():
                freq[word] = freq.get(word, 0) + 1
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        for w, _ in sorted_words[: self.max_features - 2]:
            idx = len(self.word2idx)
            self.word2idx[w] = idx
            self.idx2word[idx] = w

    def transform(self, texts, max_len=40):
        seqs = []
        for text in texts:
            tokens = text.split()
            ids = [self.word2idx.get(w, 1) for w in tokens[:max_len]]
            if len(ids) < max_len:
                ids += [0] * (max_len - len(ids))
            seqs.append(ids)
        return np.array(seqs, dtype=np.int64)


class TextDataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        return self.sequences[idx]


# -------------------------------------------------------------
# 3. Neural Architecture 1 from P2: BiLSTM (Section 4.2.2)
# -------------------------------------------------------------
class P2_BiLSTM(nn.Module):
    def __init__(self, vocab_size=20000, embed_dim=128, hidden_dim=196, num_classes=4, dropout=0.2):
        super(P2_BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        out, (hn, _) = self.bilstm(embedded)
        feat = torch.cat([hn[-2], hn[-1]], dim=1)
        logits = self.fc(feat)
        return logits


# -------------------------------------------------------------
# 4. Neural Architecture 2 from P2: Hybrid CNN+BiLSTM (Section 4.4.3)
# (The best proposed model in P2.pdf)
# -------------------------------------------------------------
class P2_CNN_BiLSTM(nn.Module):
    def __init__(self, vocab_size=20000, embed_dim=128, conv_filters=100, lstm_hidden=196, num_classes=4, dropout=0.2):
        super(P2_CNN_BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.conv_kernels = [2, 3, 4, 5]
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=conv_filters, kernel_size=k, padding="same")
            for k in self.conv_kernels
        ])
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bilstm = nn.LSTM(
            input_size=conv_filters * 4,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        transposed = embedded.transpose(1, 2)
        conv_outputs = []
        for conv in self.convs:
            c = self.relu(conv(transposed))
            p = self.pool(c)
            conv_outputs.append(p)
        cnn_concat = torch.cat(conv_outputs, dim=1)
        cnn_seq = cnn_concat.transpose(1, 2)
        lstm_out, (hn, _) = self.bilstm(cnn_seq)
        feat = torch.cat([hn[-2], hn[-1]], dim=1)
        logits = self.fc(feat)
        return logits


def train_neural_model(model, train_loader, epochs=4, lr=0.001, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model


def predict_neural_model(model, loader, device="cpu"):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            seqs = batch[0] if isinstance(batch, (list, tuple)) else batch
            seqs = seqs.to(device)
            outputs = model(seqs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
    return np.array(all_preds)


def compute_metrics_table(y_true, predictions_dict, latencies_dict, train_times_dict, split_name="20000 Comments"):
    summary_rows = []
    cm_dict = {}

    for m_name, yp in predictions_dict.items():
        acc = accuracy_score(y_true, yp)
        macro_f1 = f1_score(y_true, yp, average="macro", zero_division=0)
        macro_prec = precision_score(y_true, yp, average="macro", zero_division=0)
        macro_rec = recall_score(y_true, yp, average="macro", zero_division=0)

        class_f1s = f1_score(y_true, yp, average=None, labels=TARGET_CLASSES, zero_division=0)
        class_rec = recall_score(y_true, yp, average=None, labels=TARGET_CLASSES, zero_division=0)
        class_prec = precision_score(y_true, yp, average=None, labels=TARGET_CLASSES, zero_division=0)

        cm = confusion_matrix(y_true, yp, labels=TARGET_CLASSES)
        cm_dict[m_name] = cm

        mix_idx = TARGET_CLASSES.index("Mixed_feelings")
        pos_idx = TARGET_CLASSES.index("Positive")
        unk_idx = TARGET_CLASSES.index("Unknown_state")

        mix_total = cm[mix_idx].sum()
        mix_as_pos = cm[mix_idx, pos_idx]
        mix_fp_rate = (mix_as_pos / mix_total * 100) if mix_total > 0 else 0

        summary_rows.append({
            "Architecture": m_name,
            "Accuracy": acc,
            "Macro-F1": macro_f1,
            "Macro-Recall": macro_rec,
            "Macro-Precision": macro_prec,
            "Mixed F1": class_f1s[mix_idx],
            "Mixed Recall (%)": class_rec[mix_idx] * 100,
            "Unknown Recall (%)": class_rec[unk_idx] * 100,
            "Mixed->Pos Collapse (%)": mix_fp_rate,
            "Latency (ms)": latencies_dict.get(m_name, 0.0),
            "Train Time (s)": train_times_dict.get(m_name, 0.0)
        })

    df_out = pd.DataFrame(summary_rows)
    return df_out, cm_dict


# -------------------------------------------------------------
# 5. Benchmark Execution: Tested with Full 20,000 Comments
# -------------------------------------------------------------
def run_benchmark(filepath="data/tanglish_test_20000_with_labels.csv"):
    filepath = resolve_path(filepath)
    print("=" * 95)
    print("      SCALED 20,000-COMMENT BENCHMARK: 3 P2.pdf MODELS VS. PROPOSED ConG-MoE v2")
    print("=" * 95)
    print(f"[*] Dataset: {filepath}")

    df, text_col, label_col = load_dataset(filepath)
    total_samples = len(df)
    print(f"[*] Total Benchmark Scale: {total_samples:,} Annotated Comments")
    print("    Class distribution:")
    for cls_name, count in df[label_col].value_counts().items():
        print(f"      - {cls_name:<16}: {count:>6,} ({count / total_samples * 100:5.2f}%)")

    # Stratified split: 75% Train (15,000), 25% Test (5,000)
    X_train_raw, X_test_raw, y_train_str, y_test_str = train_test_split(
        df[text_col],
        df[label_col],
        test_size=0.25,
        random_state=42,
        stratify=df[label_col]
    )

    y_train_idx = np.array([LABEL_TO_IDX[y] for y in y_train_str])
    y_test_idx = np.array([LABEL_TO_IDX[y] for y in y_test_str])
    y_full_str = df[label_col].values
    y_full_idx = np.array([LABEL_TO_IDX[y] for y in y_full_str])

    print(f"\n[*] Training Support: {len(X_train_raw):,} samples | Test Support: {len(X_test_raw):,} samples")
    print(f"[*] Full Corpus Evaluation Scale: {len(df):,} samples\n")

    # Preprocessing
    print("[*] Preprocessing text using P2.pdf rules (emojis stripped, repeated chars compressed)...")
    X_train_p2 = [p2_clean_text(t) for t in X_train_raw]
    X_test_p2 = [p2_clean_text(t) for t in X_test_raw]
    X_full_p2 = [p2_clean_text(t) for t in df[text_col]]

    print("[*] Preprocessing text using ConG-MoE v2 rules (affective demojization, char compression)...")
    X_train_cong = [normalize_text(t) for t in X_train_raw]
    X_test_cong = [normalize_text(t) for t in X_test_raw]
    X_full_cong = [normalize_text(t) for t in df[text_col]]

    train_times = {}
    latencies_test = {}
    latencies_20k = {}
    preds_5k = {}
    preds_20k = {}

    # -------------------------------------------------------------
    # MODEL 1: P2 Traditional ML Baseline (TF-IDF + Logistic Regression)
    # -------------------------------------------------------------
    m1_name = "1. TF-IDF + Logistic Regression (P2 Baseline)"
    print("\n" + "-" * 75)
    print(f"1/4. Training {m1_name}...")
    tfidf = TfidfVectorizer(ngram_range=(1, 1), max_df=0.95, min_df=2)
    t0 = time.time()
    X_train_vec = tfidf.fit_transform(X_train_p2)
    model_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model_lr.fit(X_train_vec, y_train_str)
    train_times[m1_name] = time.time() - t0

    # 5k evaluation
    t0 = time.time()
    X_test_vec = tfidf.transform(X_test_p2)
    preds_5k[m1_name] = model_lr.predict(X_test_vec)
    latencies_test[m1_name] = ((time.time() - t0) / len(X_test_raw)) * 1000

    # 20k full evaluation
    t0 = time.time()
    X_full_vec = tfidf.transform(X_full_p2)
    preds_20k[m1_name] = model_lr.predict(X_full_vec)
    latencies_20k[m1_name] = ((time.time() - t0) / total_samples) * 1000
    print(f"     Done. Train: {train_times[m1_name]:.2f}s | 20k Latency: {latencies_20k[m1_name]:.4f} ms/sample")

    # -------------------------------------------------------------
    # Build Vocabulary & Loaders for Neural Models
    # -------------------------------------------------------------
    vocab = P2Vocab(max_features=20000)
    vocab.fit(X_train_p2)
    train_seqs = vocab.transform(X_train_p2, max_len=40)
    test_seqs = vocab.transform(X_test_p2, max_len=40)
    full_seqs = vocab.transform(X_full_p2, max_len=40)

    train_loader = DataLoader(TextDataset(train_seqs, y_train_idx), batch_size=128, shuffle=True)
    test_loader = DataLoader(TextDataset(test_seqs, y_test_idx), batch_size=256, shuffle=False)
    full_loader = DataLoader(TextDataset(full_seqs, y_full_idx), batch_size=256, shuffle=False)

    # -------------------------------------------------------------
    # MODEL 2: P2 Deep Learning Baseline (BiLSTM)
    # -------------------------------------------------------------
    m2_name = "2. BiLSTM (P2 Deep Learning)"
    print("\n" + "-" * 75)
    print(f"2/4. Training {m2_name}...")
    bilstm_model = P2_BiLSTM(vocab_size=len(vocab.word2idx), embed_dim=128, hidden_dim=196, num_classes=4)
    t0 = time.time()
    train_neural_model(bilstm_model, train_loader, epochs=4, lr=0.001)
    train_times[m2_name] = time.time() - t0

    # 5k evaluation
    t0 = time.time()
    pred_idx_5k = predict_neural_model(bilstm_model, test_loader)
    preds_5k[m2_name] = [IDX_TO_LABEL[i] for i in pred_idx_5k]
    latencies_test[m2_name] = ((time.time() - t0) / len(X_test_raw)) * 1000

    # 20k full evaluation
    t0 = time.time()
    pred_idx_20k = predict_neural_model(bilstm_model, full_loader)
    preds_20k[m2_name] = [IDX_TO_LABEL[i] for i in pred_idx_20k]
    latencies_20k[m2_name] = ((time.time() - t0) / total_samples) * 1000
    print(f"     Done. Train: {train_times[m2_name]:.2f}s | 20k Latency: {latencies_20k[m2_name]:.4f} ms/sample")

    # -------------------------------------------------------------
    # MODEL 3: P2 Best Proposed Hybrid Model (CNN + BiLSTM)
    # -------------------------------------------------------------
    m3_name = "3. CNN + BiLSTM (P2 Flagship Best)"
    print("\n" + "-" * 75)
    print(f"3/4. Training {m3_name}...")
    cnn_bilstm_model = P2_CNN_BiLSTM(
        vocab_size=len(vocab.word2idx),
        embed_dim=128,
        conv_filters=100,
        lstm_hidden=196,
        num_classes=4
    )
    t0 = time.time()
    train_neural_model(cnn_bilstm_model, train_loader, epochs=4, lr=0.001)
    train_times[m3_name] = time.time() - t0

    # 5k evaluation
    t0 = time.time()
    pred_idx_cnn_5k = predict_neural_model(cnn_bilstm_model, test_loader)
    preds_5k[m3_name] = [IDX_TO_LABEL[i] for i in pred_idx_cnn_5k]
    latencies_test[m3_name] = ((time.time() - t0) / len(X_test_raw)) * 1000

    # 20k full evaluation
    t0 = time.time()
    pred_idx_cnn_20k = predict_neural_model(cnn_bilstm_model, full_loader)
    preds_20k[m3_name] = [IDX_TO_LABEL[i] for i in pred_idx_cnn_20k]
    latencies_20k[m3_name] = ((time.time() - t0) / total_samples) * 1000
    print(f"     Done. Train: {train_times[m3_name]:.2f}s | 20k Latency: {latencies_20k[m3_name]:.4f} ms/sample")

    # -------------------------------------------------------------
    # MODEL 4: Proposed ConG-MoE v2
    # -------------------------------------------------------------
    m4_name = "4. ConG-MoE v2 (Our Proposed Model)"
    print("\n" + "-" * 75)
    print(f"4/4. Training {m4_name}...")
    cong_moe = CongMoeClassifier(confidence_threshold=0.65, C=0.5, random_state=42)
    t0 = time.time()
    cong_moe.fit(X_train_cong, y_train_str, X_raw=X_train_raw)
    train_times[m4_name] = time.time() - t0

    # 5k evaluation
    t0 = time.time()
    preds_5k[m4_name] = cong_moe.predict(X_test_cong, X_raw=X_test_raw)
    latencies_test[m4_name] = ((time.time() - t0) / len(X_test_raw)) * 1000

    # 20k full evaluation
    t0 = time.time()
    preds_20k[m4_name] = cong_moe.predict(X_full_cong, X_raw=df[text_col].values)
    latencies_20k[m4_name] = ((time.time() - t0) / total_samples) * 1000
    print(f"     Done. Train: {train_times[m4_name]:.2f}s | 20k Latency: {latencies_20k[m4_name]:.4f} ms/sample")

    # -------------------------------------------------------------
    # 6. Evaluation Reports: 20k Full Corpus & 5k Held-Out Test
    # -------------------------------------------------------------
    df_20k, cm_20k = compute_metrics_table(y_full_str, preds_20k, latencies_20k, train_times, split_name="20,000 Comments")
    df_5k, cm_5k = compute_metrics_table(y_test_str, preds_5k, latencies_test, train_times, split_name="5,000 Test Set")

    print("\n" + "=" * 98)
    print("                     EVALUATION REPORT ACROSS ALL 20,000 COMMENTS")
    print("=" * 98)
    header_str = (
        f"{'Model Architecture':<40} | {'Acc':<6} | {'Macro-F1':<8} | {'Mix Rec%':<8} | "
        f"{'Unk Rec%':<8} | {'Mix->Pos%':<9} | {'Latency':<8}"
    )
    print(header_str)
    print("-" * len(header_str))
    for _, row in df_20k.iterrows():
        print(
            f"{row['Architecture']:<40} | "
            f"{row['Accuracy']*100:5.2f}% | "
            f"{row['Macro-F1']:8.4f} | "
            f"{row['Mixed Recall (%)']:7.2f}% | "
            f"{row['Unknown Recall (%)']:7.2f}% | "
            f"{row['Mixed->Pos Collapse (%)']:8.2f}% | "
            f"{row['Latency (ms)']:6.3f} ms"
        )
    print("-" * len(header_str))

    # Save CSVs
    os.makedirs("docs", exist_ok=True)
    csv_20k_path = "docs/p2_comparison_20000_full_benchmark_results.csv"
    csv_5k_path = "docs/p2_comparison_5000_test_benchmark_results.csv"
    df_20k.to_csv(csv_20k_path, index=False)
    df_5k.to_csv(csv_5k_path, index=False)
    print(f"\n[+] Saved 20,000 comments benchmark to: {csv_20k_path}")
    print(f"[+] Saved 5,000 test comments benchmark to: {csv_5k_path}")

    # -------------------------------------------------------------
    # 7. Generate High-Res 20k Comparison Chart
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    models_labels = [
        "P2: TF-IDF+LR",
        "P2: BiLSTM",
        "P2: CNN+BiLSTM",
        "Proposed:\nConG-MoE v2"
    ]
    colors = ["#7f8c8d", "#3498db", "#e67e22", "#27ae60"]

    # 1. Macro-F1
    macro_f1_vals = df_20k["Macro-F1"].values
    bars1 = axes[0].bar(models_labels, macro_f1_vals, color=colors, edgecolor="black", width=0.55)
    axes[0].set_title("Overall Balanced Metric: Macro-F1 (20k Comments)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Macro-F1 Score", fontsize=11)
    axes[0].set_ylim(0, 1.0)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

    # 2. Mixed Recall
    mix_rec_vals = df_20k["Mixed Recall (%)"].values
    bars2 = axes[1].bar(models_labels, mix_rec_vals, color=colors, edgecolor="black", width=0.55)
    axes[1].set_title("Minority Class: Mixed Feelings Recall (%) (20k Comments)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Recall Rate (%)", fontsize=11)
    axes[1].set_ylim(0, 105)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)

    # 3. Collapse Rate
    collapse_vals = df_20k["Mixed->Pos Collapse (%)"].values
    bars3 = axes[2].bar(models_labels, collapse_vals, color=["#c0392b", "#c0392b", "#e74c3c", "#27ae60"], edgecolor="black", width=0.55)
    axes[2].set_title("Majority Collapse Rate: Mixed Error -> Positive (%)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Error Rate (%) (Lower is Better)", fontsize=11)
    axes[2].set_ylim(0, 105)
    for bar in bars3:
        yval = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)

    plt.suptitle("Tanglish Sentiment Benchmark Evaluated on Complete 20,000 Comments (P2.pdf vs ConG-MoE v2)", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    chart_path = "docs/tanglish_20k_full_benchmark_chart.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved 20,000-comment comparison chart to: {chart_path}\n")

    return df_20k, df_5k


if __name__ == "__main__":
    run_benchmark()
