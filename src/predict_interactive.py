import os
import sys
import time
import pickle
import numpy as np
import pandas as pd

from tanglish_sentiment_pipeline import (
    resolve_path,
    load_dataset,
    normalize_text,
    CongMoeClassifier,
    TARGET_CLASSES,
    EMOJI_MAP
)

MODEL_FILE = resolve_path("tanglish_congmoe_model.pkl")
DATASET_FILE = resolve_path("tanglish_test_20000_with_labels.csv")

STATE_EMOJIS = {
    "Positive": "🌟 POSITIVE (Praise / High Excitement)",
    "Negative": "💔 NEGATIVE (Disappointment / Criticism)",
    "Mixed_feelings": "⚖️  MIXED FEELINGS (Conflicted / Nuanced Review)",
    "Unknown_state": "❓ UNKNOWN STATE (Question / Inquiry / Neutral)"
}

def get_or_train_model():
    """Loads the pre-trained model or trains and saves it if not already present."""
    if os.path.exists(MODEL_FILE):
        print(f"[*] Loading pre-trained ConG-MoE v2 model from '{MODEL_FILE}'...")
        t0 = time.perf_counter()
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        print(f"[*] Model loaded in {(time.perf_counter() - t0)*1000:.1f} ms!\n")
        return model

    print(f"[*] Pre-trained model not found. Training once on {DATASET_FILE}...")
    df, text_col, label_col = load_dataset(DATASET_FILE)
    X_train_raw = df[text_col]
    y_train = df[label_col]

    print("[*] Normalizing text representations...")
    X_train_norm = X_train_raw.apply(normalize_text)

    print("[*] Training ConG-MoE v2 architecture...")
    t0 = time.perf_counter()
    model = CongMoeClassifier(confidence_threshold=0.65, C=0.5, random_state=42)
    model.fit(X_train_norm, y_train, X_raw=X_train_raw)
    train_time = time.perf_counter() - t0
    print(f"[*] Training finished in {train_time:.2f} seconds.")

    print(f"[*] Saving model to '{MODEL_FILE}' for instant future startup...")
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    print("[*] Model successfully cached!\n")
    return model

def predict_comment(raw_text: str, model: CongMoeClassifier):
    """Predicts the sentiment state of a user comment using ConG-MoE v2."""
    t0 = time.perf_counter()

    clean_text = normalize_text(raw_text)

    # 1. Base statistical pipeline prediction & probabilities
    probs = model.base_pipeline.predict_proba([clean_text])[0]
    classes = model.base_pipeline.classes_
    prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}

    base_pred = classes[probs.argmax()]
    confidence = float(probs.max())

    # 2. Check Expert Heads & Linguistic Markers
    # Extended pivot check
    extended_pivots = {"but", "aana", "aanal", "aanaa", "anaa", "ana", "irundhalum", "irunthalum", "yet", "though", "however", "analum"}
    has_pivot = any(p in clean_text.split() for p in extended_pivots)

    # Counterfactual regret markers ('could have been better')
    regret_markers = {"irunthurukalam", "irunthirukalam", "irundhurukalam", "irundhirukalam", "irukalam", "irunthrukalam", "panniirukalam"}
    has_regret = any(m in clean_text for m in regret_markers) or ("inno" in clean_text and "best" in clean_text) or ("innum" in clean_text and "nalla" in clean_text)

    pivot_pred = model.pivot_head.predict_sample(raw_text)
    emoji_pred = model.emoji_head.predict_sample(raw_text)

    # 3. Gating Decision Logic
    override_reason = None

    if confidence < model.confidence_threshold and (pivot_pred == "Mixed_feelings" or emoji_pred == "Mixed_feelings"):
        final_state = "Mixed_feelings"
        triggers = []
        if pivot_pred == "Mixed_feelings":
            triggers.append("Contrastive pivot (e.g. 'aana'/'but') detected a polarity flip")
        if emoji_pred == "Mixed_feelings":
            triggers.append("Affective dissonance detected between text and emojis")
        override_reason = " + ".join(triggers)
    elif (has_pivot and has_regret) or (has_regret and "acting" in clean_text):
        final_state = "Mixed_feelings"
        override_reason = "Counterfactual suggestion ('could have been better') detected alongside praise"
    else:
        final_state = base_pred

    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "raw_text": raw_text,
        "clean_text": clean_text,
        "final_state": final_state,
        "base_prediction": base_pred,
        "confidence": confidence,
        "override_reason": override_reason,
        "prob_dict": prob_dict,
        "latency_ms": latency_ms
    }

def print_prediction_result(res: dict):
    state_str = STATE_EMOJIS.get(res['final_state'], res['final_state'])
    print("\n" + "─" * 70)
    print(f"  Input Comment : \"{res['raw_text']}\"")
    print(f"  Processed As  : \"{res['clean_text']}\"")
    print(f"  Predicted State: {state_str}")
    print("─" * 70)
    print(f"  Base Model Prediction : {res['base_prediction']} (Confidence: {res['confidence']*100:.1f}%)")

    if res['override_reason']:
        print(f"  Specialist Intervention: [YES] Overridden to Mixed feelings")
        print(f"  Reason                : {res['override_reason']}")
    else:
        print(f"  Specialist Intervention: [NO] Base prediction accepted (Confidence >= 65% or no conflict)")

    print(f"  Inference Latency     : {res['latency_ms']:.3f} milliseconds")
    print("\n  Class Probability Breakdown:")
    for cls in TARGET_CLASSES:
        p = res['prob_dict'].get(cls, 0.0)
        bar = "█" * int(p * 30)
        print(f"    - {cls:<16}: {p*100:5.1f}%  |{bar:<30}|\n")
    print("─" * 70 + "\n")

def main():
    model = get_or_train_model()

    # If the user passed a comment directly as a command-line argument:
    if len(sys.argv) > 1:
        user_comment = " ".join(sys.argv[1:])
        res = predict_comment(user_comment, model)
        print_prediction_result(res)
        return

    # Otherwise, run interactive prompt loop:
    print("=" * 70)
    print("     TANGLISH SENTIMENT PREDICTOR (ConG-MoE v2 Interactive CLI)")
    print("=" * 70)
    print("Type any Tanglish comment below and hit Enter.")
    print("Commands: 'exit' or 'q' to quit, 'samples' to test built-in examples.\n")

    sample_comments = [
        "Thalaivar entry vera level mass 🔥 padam blockbuster confirmed!",
        "Padam romba mokka da timing waste don't go theatre 🤮 worst movie",
        "Acting sema mass aana screenplay romba worst",
        "first half nalla irukku aana climax mokka 🤮",
        "GOAT trailer vera level bro... eppo ott release date???"
    ]

    while True:
        try:
            user_input = input("Enter Tanglish Comment > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Vanakkam!")
            break

        if not user_input:
            continue
        if user_input.lower() in ["q", "quit", "exit"]:
            print("Exiting predictor. Vanakkam!")
            break
        if user_input.lower() == "samples":
            print("\n--- Running Built-In Sample Comments ---")
            for sample in sample_comments:
                res = predict_comment(sample, model)
                print_prediction_result(res)
            continue

        res = predict_comment(user_input, model)
        print_prediction_result(res)

if __name__ == "__main__":
    main()
