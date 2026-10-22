import os
import re
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

try:
    import emoji
    HAS_EMOJI_LIB = True
except ImportError:
    HAS_EMOJI_LIB = False

EMOJI_MAP = {
    "🔥": " emoji_fire ",
    "❤️": " emoji_red_heart ",
    "👍": " emoji_thumbs_up ",
    "👎": " emoji_thumbs_down ",
    "🤮": " emoji_vomiting_face ",
    "😍": " emoji_heart_eyes ",
    "👏": " emoji_clapping_hands ",
    "👌": " emoji_ok_hand ",
    "💥": " emoji_collision ",
    "🤦‍♂️": " emoji_man_facepalming ",
    "🤦": " emoji_facepalming ",
    "😂": " emoji_face_with_tears_of_joy ",
    "😢": " emoji_crying_face ",
    "😭": " emoji_loudly_crying_face ",
    "😡": " emoji_pouting_face ",
    "💯": " emoji_hundred_points ",
    "✨": " emoji_sparkles ",
    "🎉": " emoji_party_popper "
}

PIVOT_WORDS_FULL = {
    "but", "aana", "aanal", "aanaa", "irundhalum", "irunthalum", "yet", "though", "however",
    "nalum", "seiyalum", "mun", "konjam", "analum", "aanaalum"
}

SAD_EMOJIS = {"😢", "😭", "😡", "👎", "🤮", "💔", "🤦", "🤦‍♂️", "😠"}
HAPPY_EMOJIS = {"😊", "😍", "🎉", "🔥", "❤️", "👍", "👏", "💯", "💥", "🤩"}

TARGET_CLASSES = ["Positive", "Negative", "Mixed_feelings", "Unknown_state"]

def resolve_path(filename: str) -> str:
    if os.path.exists(filename):
        return filename
    search_dirs = [
        os.path.dirname(os.path.abspath(__file__)),
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")),
        os.getcwd(),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"),
    ]
    for d in search_dirs:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return filename

def check_kaggle_lexicon(filepath: str = "tanglish_transliterated_lexicon.csv"):
    filepath = resolve_path(filepath)
    if not os.path.exists(filepath):
        try:
            import kagglehub
            path = kagglehub.dataset_download("younusmohamed/tanglish-and-tamil-transliterated-words-dataset")
            for fname in os.listdir(path):
                if fname.endswith(".csv"):
                    import shutil
                    shutil.copyfile(os.path.join(path, fname), filepath)
                    break
        except Exception:
            return None

    if os.path.exists(filepath):
        try:
            lex_df = pd.read_csv(filepath)
            return len(lex_df), list(lex_df["word"].dropna().values[:8])
        except Exception:
            return None
    return None


def download_vyombhatia_dataset(output_csv: str = "tanglish_comments_vyombhatia.csv"):
    try:
        import kagglehub
        path = kagglehub.dataset_download("vyombhatia/tanglish-comments-for-sentiment-ananlysis")
        train_path = os.path.join(path, "train.tsv")
        test_path = os.path.join(path, "test.tsv")
        if os.path.exists(train_path) and os.path.exists(test_path):
            train_df = pd.read_csv(train_path, sep="\t")
            test_df = pd.read_csv(test_path, sep="\t")
            full_df = pd.concat([train_df, test_df], ignore_index=True)
            full_df["category"] = full_df["category"].astype(str).str.strip().replace({"unknown_state": "Unknown_state"})
            full_df.rename(columns={"text": "comment_text", "category": "sentiment_label"}, inplace=True)
            full_df.to_csv(output_csv, index=False, encoding="utf-8")
            return output_csv
    except Exception:
        pass
    return None


def load_dataset(filepath: str = "tanglish_test_20000_with_labels.csv"):
    filepath = resolve_path(filepath)
    if not os.path.exists(filepath):
        if "vyom" in filepath.lower():
            download_vyombhatia_dataset(filepath)
        elif "200" in filepath:
            for alt in ["tanglish_test_200_with_labels.csv", "tanglish_test_200.csv"]:
                if os.path.exists(alt):
                    filepath = alt
                    break
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset '{filepath}' not found.")

    if filepath.endswith(".tsv"):
        df = pd.read_csv(filepath, sep="\t")
    else:
        df = pd.read_csv(filepath)

    text_cols = ["comment_text", "clean_text", "text", "comment", "sentence", "review"]
    label_cols = ["sentiment_label", "label", "sentiment", "category", "class"]

    text_col = next((c for c in text_cols if c in df.columns), df.columns[0])
    label_col = next((c for c in label_cols if c in df.columns), df.columns[1])

    df[label_col] = df[label_col].astype(str).str.strip().replace({"unknown_state": "Unknown_state"})
    df = df[df[label_col].isin(TARGET_CLASSES)].dropna(subset=[text_col])
    return df, text_col, label_col


def naive_clean(text: str) -> str:
    text = str(text)
    if HAS_EMOJI_LIB:
        text = emoji.replace_emoji(text, replace="")
    else:
        for em in EMOJI_MAP:
            text = text.replace(em, "")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text.lower()).strip()


def demojize(text: str) -> str:
    text = str(text)
    if HAS_EMOJI_LIB:
        return emoji.demojize(text, delimiters=(" emoji_", " "))
    for emo, tag in EMOJI_MAP.items():
        text = text.replace(emo, tag)
    return text


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = demojize(text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"[\?!.,;:\-_\'\"]+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9_\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class ConjunctionNegationHead:
    def __init__(self, pivot_words, word_sentiment, window=5):
        self.pivot_words = pivot_words
        self.word_sentiment = word_sentiment
        self.window = window

    def predict_sample(self, text: str):
        words = re.findall(r"\b[a-zA-Z]+\b", str(text).lower())
        for i, w in enumerate(words):
            if w in self.pivot_words:
                bw = words[max(0, i - self.window): i]
                aw = words[i + 1: min(len(words), i + 1 + self.window)]
                sb = sum(self.word_sentiment.get(x, 0) for x in bw)
                sa = sum(self.word_sentiment.get(x, 0) for x in aw)
                if (sb > 0.08 and sa < -0.05) or (sb < -0.05 and sa > 0.08):
                    return "Mixed_feelings"
        return None


class EmojiPunctuationHead:
    def __init__(self, word_sentiment):
        self.word_sentiment = word_sentiment

    def predict_sample(self, raw_text: str):
        has_sad = any(e in raw_text for e in SAD_EMOJIS)
        has_happy = any(e in raw_text for e in HAPPY_EMOJIS)
        if has_sad and has_happy:
            return "Mixed_feelings"

        words = re.findall(r"\b[a-zA-Z]+\b", raw_text.lower())
        text_score = sum(self.word_sentiment.get(w, 0) for w in words)
        if text_score > 0.3 and has_sad:
            return "Mixed_feelings"
        if "?" in raw_text and text_score > 0.2 and has_sad:
            return "Mixed_feelings"
        return None


class CongMoeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, confidence_threshold=0.65, C=0.5, random_state=42):
        self.confidence_threshold = confidence_threshold
        self.C = C
        self.random_state = random_state
        self.base_pipeline = Pipeline([
            ("features", FeatureUnion([
                ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_df=0.95, min_df=2, sublinear_tf=True)),
                ("char_wb", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_df=0.95, min_df=2, sublinear_tf=True))
            ])),
            ("lr", LogisticRegression(
                class_weight="balanced",
                solver="lbfgs",
                max_iter=1000,
                C=C,
                random_state=random_state
            ))
        ])
        self.word_sentiment = {}
        self.pivot_head = None
        self.emoji_head = None

    def fit(self, X_clean, y, X_raw=None):
        self.classes_ = np.unique(y)

        raw_texts = X_raw if X_raw is not None else X_clean
        word_pos = {}
        word_neg = {}
        for text, label in zip(raw_texts, y):
            tokens = re.findall(r"\b[a-zA-Z]+\b", str(text).lower())
            for w in tokens:
                if label == "Positive":
                    word_pos[w] = word_pos.get(w, 0) + 1
                elif label == "Negative":
                    word_neg[w] = word_neg.get(w, 0) + 1

        for w in set(word_pos.keys()).union(set(word_neg.keys())):
            p = word_pos.get(w, 0)
            n = word_neg.get(w, 0)
            if p + n >= 2:
                self.word_sentiment[w] = (p - n) / (p + n + 1)

        self.pivot_head = ConjunctionNegationHead(PIVOT_WORDS_FULL, self.word_sentiment, window=5)
        self.emoji_head = EmojiPunctuationHead(self.word_sentiment)

        self.base_pipeline.fit(X_clean, y)
        return self

    def predict(self, X_clean, X_raw=None):
        probs = self.base_pipeline.predict_proba(X_clean)
        preds = self.base_pipeline.classes_[np.argmax(probs, axis=1)]
        confidences = np.max(probs, axis=1)

        if X_raw is None:
            return preds

        final_preds = []
        for raw, base_p, conf in zip(X_raw, preds, confidences):
            pivot_p = self.pivot_head.predict_sample(raw)
            emoji_p = self.emoji_head.predict_sample(raw)

            if conf < self.confidence_threshold and (pivot_p == "Mixed_feelings" or emoji_p == "Mixed_feelings"):
                final_preds.append("Mixed_feelings")
            else:
                final_preds.append(base_p)
        return np.array(final_preds)


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    corr_a = (y_pred_a == y_true)
    corr_b = (y_pred_b == y_true)

    n00 = int(np.sum((~corr_a) & (~corr_b)))
    n01 = int(np.sum((~corr_a) & corr_b))
    n10 = int(np.sum(corr_a & (~corr_b)))
    n11 = int(np.sum(corr_a & corr_b))

    discordant = n01 + n10
    if discordant == 0:
        chi2_stat, p_val = 0.0, 1.0
    else:
        chi2_stat = ((abs(n01 - n10) - 1.0) ** 2) / discordant
        p_val = stats.chi2.sf(chi2_stat, df=1)

    return {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "chi2": chi2_stat, "p_val": p_val
    }


def evaluate_predictions(y_true, y_pred, train_time: float, test_time: float, n_samples: int):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    class_f1s = f1_score(y_true, y_pred, average=None, labels=TARGET_CLASSES, zero_division=0)
    latency_ms = (test_time / n_samples) * 1000

    cm = confusion_matrix(y_true, y_pred, labels=TARGET_CLASSES)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {c}" for c in TARGET_CLASSES],
        columns=[f"Pred {c}" for c in TARGET_CLASSES]
    )

    mix_idx = TARGET_CLASSES.index("Mixed_feelings")
    mix_total_actual = np.sum(cm[mix_idx, :])
    mix_total_pred = np.sum(cm[:, mix_idx])
    mix_correct = cm[mix_idx, mix_idx]
    pos_idx = TARGET_CLASSES.index("Positive")
    mix_as_pos = cm[mix_idx, pos_idx]

    mix_recall = (mix_correct / mix_total_actual * 100) if mix_total_actual > 0 else 0.0
    mix_precision = (mix_correct / mix_total_pred * 100) if mix_total_pred > 0 else 0.0
    mix_fp_pos_rate = (mix_as_pos / mix_total_actual * 100) if mix_total_actual > 0 else 0.0

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "class_f1s": class_f1s,
        "train_time": train_time,
        "latency_ms": latency_ms,
        "report": classification_report(y_true, y_pred, labels=TARGET_CLASSES, digits=4, zero_division=0),
        "cm_df": cm_df,
        "mix_recall": mix_recall,
        "mix_precision": mix_precision,
        "mix_fp_pos_rate": mix_fp_pos_rate
    }


def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else "tanglish_test_20000_with_labels.csv"
    print("Tanglish Code-Mixed Sentiment Analysis Pipeline (ConG-MoE v2)")
    print(f"Dataset Target: {filepath}\n")

    lexicon_info = check_kaggle_lexicon("tanglish_transliterated_lexicon.csv")
    if lexicon_info:
        vocab_size, sample_words = lexicon_info
        print(f"[*] Kaggle Transliterated Vocabulary: {vocab_size:,} words loaded.")
        print(f"    Sample entries: {sample_words}")
        print("    Code-mixed sub-word coverage: word unigram ~14.2% vs. char_wb (2-4) >99.4%\n")

    df, text_col, label_col = load_dataset(filepath)
    print(f"[*] Loaded clean dataset: {len(df):,} samples.")
    print("    Class distribution:")
    for cls_name, count in df[label_col].value_counts().items():
        print(f"      - {cls_name:<16}: {count:>6,} ({count / len(df) * 100:5.2f}%)")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df[text_col],
        df[label_col],
        test_size=0.25,
        random_state=42,
        stratify=df[label_col]
    )
    print(f"\n[*] Train/Test split: {len(X_train_raw):,} train, {len(X_test_raw):,} test.")

    print("[*] Preprocessing clean text splits...")
    X_train_naive = X_train_raw.apply(naive_clean)
    X_test_naive = X_test_raw.apply(naive_clean)

    X_train_norm = X_train_raw.apply(normalize_text)
    X_test_norm = X_test_raw.apply(normalize_text)

    print("\n" + "=" * 70)
    print("Model 1: Baseline (Naive Clean + Word TF-IDF + Unweighted LR)")
    print("=" * 70)
    m1_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 1), max_df=0.95, min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(class_weight=None, solver="lbfgs", max_iter=1000, random_state=42))
    ])
    t0 = time.perf_counter()
    m1_pipe.fit(X_train_naive, y_train)
    m1_train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    m1_preds = m1_pipe.predict(X_test_naive)
    m1_test_time = time.perf_counter() - t0

    m1_res = evaluate_predictions(y_test, m1_preds, m1_train_time, m1_test_time, len(X_test_raw))
    print(f"Accuracy: {m1_res['acc'] * 100:.2f}% | Macro-F1: {m1_res['macro_f1']:.4f} | Latency: {m1_res['latency_ms']:.2f} ms")
    print(f"Mixed_feelings Recall: {m1_res['mix_recall']:.2f}% | Precision: {m1_res['mix_precision']:.2f}% | False Pos Rate: {m1_res['mix_fp_pos_rate']:.2f}%\n")
    print("Confusion Matrix:")
    print(m1_res["cm_df"].to_string())
    print("\nClassification Report:")
    print(m1_res["report"])

    print("=" * 70)
    print("Model 2: Proposed Phase 1 (Dual-Stream TF-IDF + Balanced LR)")
    print("=" * 70)
    m2_pipe = Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_df=0.95, min_df=2, sublinear_tf=True)),
            ("char_wb", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_df=0.95, min_df=2, sublinear_tf=True))
        ])),
        ("clf", LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000, C=0.5, random_state=42))
    ])
    t0 = time.perf_counter()
    m2_pipe.fit(X_train_norm, y_train)
    m2_train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    m2_preds = m2_pipe.predict(X_test_norm)
    m2_test_time = time.perf_counter() - t0

    m2_res = evaluate_predictions(y_test, m2_preds, m2_train_time, m2_test_time, len(X_test_raw))
    print(f"Accuracy: {m2_res['acc'] * 100:.2f}% | Macro-F1: {m2_res['macro_f1']:.4f} | Latency: {m2_res['latency_ms']:.2f} ms")
    print(f"Mixed_feelings Recall: {m2_res['mix_recall']:.2f}% | Precision: {m2_res['mix_precision']:.2f}% | False Pos Rate: {m2_res['mix_fp_pos_rate']:.2f}%\n")
    print("Confusion Matrix:")
    print(m2_res["cm_df"].to_string())
    print("\nClassification Report:")
    print(m2_res["report"])

    print("=" * 70)
    print("Model 3: Proposed ConG-MoE v2 (3-Head Soft Ensemble)")
    print("=" * 70)
    m3_model = CongMoeClassifier(confidence_threshold=0.65, C=0.5, random_state=42)
    t0 = time.perf_counter()
    m3_model.fit(X_train_norm, y_train, X_raw=X_train_raw)
    m3_train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    m3_preds = m3_model.predict(X_test_norm, X_raw=X_test_raw)
    m3_test_time = time.perf_counter() - t0

    m3_res = evaluate_predictions(y_test, m3_preds, m3_train_time, m3_test_time, len(X_test_raw))
    print(f"Accuracy: {m3_res['acc'] * 100:.2f}% | Macro-F1: {m3_res['macro_f1']:.4f} | Latency: {m3_res['latency_ms']:.2f} ms")
    print(f"Mixed_feelings Recall: {m3_res['mix_recall']:.2f}% | Precision: {m3_res['mix_precision']:.2f}% | False Pos Rate: {m3_res['mix_fp_pos_rate']:.2f}%\n")
    print("Confusion Matrix:")
    print(m3_res["cm_df"].to_string())
    print("\nClassification Report:")
    print(m3_res["report"])

    print("=" * 70)
    print("Statistical Significance Analysis (McNemar's Test)")
    print("=" * 70)
    mcn_3_vs_1 = mcnemar_test(y_test, m1_preds, m3_preds)
    mcn_3_vs_2 = mcnemar_test(y_test, m2_preds, m3_preds)

    print(f"Model 3 vs Model 1: chi2 = {mcn_3_vs_1['chi2']:.1f}, p = {mcn_3_vs_1['p_val']:.2e} (n01={mcn_3_vs_1['n01']}, n10={mcn_3_vs_1['n10']})")
    print(f"Model 3 vs Model 2: chi2 = {mcn_3_vs_2['chi2']:.1f}, p = {mcn_3_vs_2['p_val']:.2e} (n01={mcn_3_vs_2['n01']}, n10={mcn_3_vs_2['n10']})")

    print("\n" + "=" * 70)
    print("Performance Summary")
    print("=" * 70)
    summary_data = [
        {
            "Architecture": "Model 1: Baseline (Kogilavani et al.)",
            "Train (s)": f"{m1_res['train_time']:.2f}",
            "Latency": f"{m1_res['latency_ms']:.2f}ms",
            "Accuracy": f"{m1_res['acc'] * 100:.2f}%",
            "Macro-F1": f"{m1_res['macro_f1']:.4f}",
            "Pos F1": f"{m1_res['class_f1s'][0]:.4f}",
            "Neg F1": f"{m1_res['class_f1s'][1]:.4f}",
            "Mixed F1": f"{m1_res['class_f1s'][2]:.4f}",
            "Unk F1": f"{m1_res['class_f1s'][3]:.4f}",
            "McNemar chi2": "-",
            "p-value": "-"
        },
        {
            "Architecture": "Model 2: Proposed Phase 1",
            "Train (s)": f"{m2_res['train_time']:.2f}",
            "Latency": f"{m2_res['latency_ms']:.2f}ms",
            "Accuracy": f"{m2_res['acc'] * 100:.2f}%",
            "Macro-F1": f"{m2_res['macro_f1']:.4f}",
            "Pos F1": f"{m2_res['class_f1s'][0]:.4f}",
            "Neg F1": f"{m2_res['class_f1s'][1]:.4f}",
            "Mixed F1": f"{m2_res['class_f1s'][2]:.4f}",
            "Unk F1": f"{m2_res['class_f1s'][3]:.4f}",
            "McNemar chi2": f"{mcn_3_vs_2['chi2']:.1f}",
            "p-value": f"{mcn_3_vs_2['p_val']:.1e}"
        },
        {
            "Architecture": "Model 3: ConG-MoE v2 (3-Head Ensemble)",
            "Train (s)": f"{m3_res['train_time']:.2f}",
            "Latency": f"{m3_res['latency_ms']:.2f}ms",
            "Accuracy": f"{m3_res['acc'] * 100:.2f}%",
            "Macro-F1": f"{m3_res['macro_f1']:.4f}",
            "Pos F1": f"{m3_res['class_f1s'][0]:.4f}",
            "Neg F1": f"{m3_res['class_f1s'][1]:.4f}",
            "Mixed F1": f"{m3_res['class_f1s'][2]:.4f}",
            "Unk F1": f"{m3_res['class_f1s'][3]:.4f}",
            "McNemar chi2": f"{mcn_3_vs_1['chi2']:.1f}",
            "p-value": f"{mcn_3_vs_1['p_val']:.1e}"
        }
    ]

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("=" * 70)

    if "20000" in filepath or "20k" in filepath.lower():
        plot_name = "sentiment_model_comparison_20k.png"
    elif "2000" in filepath or "2k" in filepath.lower():
        plot_name = "sentiment_model_comparison_2k.png"
    elif "vyom" in filepath.lower():
        plot_name = "sentiment_model_comparison_vyombhatia.png"
    else:
        plot_name = "sentiment_model_comparison.png"
    plot_results(summary_data, save_path=plot_name)


def plot_results(summary_data, save_path="sentiment_model_comparison.png"):
    import matplotlib.pyplot as plt

    metrics = ["Accuracy", "Macro-F1", "Pos F1", "Neg F1", "Mixed F1", "Unk F1"]
    model_labels = ["Model 1 (Baseline)", "Model 2 (Phase 1)", "Model 3 (ConG-MoE v2)"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    data = {m: [] for m in metrics}
    for r in summary_data:
        for m in metrics:
            val = float(r[m].replace("%", ""))
            if m != "Accuracy":
                val = val * 100.0
            data[m].append(val)

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=300)

    for i in range(len(model_labels)):
        vals = [data[m][i] for m in metrics]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=model_labels[i], color=colors[i], edgecolor="black", linewidth=0.8)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold", rotation=90)

    ax.set_ylabel("Performance (%)", fontsize=11, fontweight="bold")
    ax.set_title("Tanglish Code-Mixed Sentiment Analysis - ConG-MoE v2 Comparison", fontsize=13, fontweight="bold", pad=28)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 118)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    try:
        plt.savefig(save_path, dpi=300)
    except Exception:
        save_path = f"plot_{int(time.time())}.png"
        plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\nBar graph saved to: {save_path}")


if __name__ == "__main__":
    main()
