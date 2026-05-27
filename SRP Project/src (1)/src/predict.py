import sys
import joblib
import pandas as pd
import numpy as np

from preprocess import load_and_preprocess_audio
from feature_extraction import extract_features


MODEL_PATH = "../models/best_model.joblib"
FEATURE_COLUMNS_PATH = "../models/feature_columns.joblib"

CLASS_NAMES = {
    0: "Normal",
    1: "Slight Fault",
    2: "Severe Fault"
}


def predict_audio(file_path):
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    signal, sr = load_and_preprocess_audio(file_path)
    features = extract_features(signal, sr)

    X = pd.DataFrame([features], columns=feature_columns)

    pred = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
    else:
        probs = np.zeros(3)

    print("\n===== PREDICTION RESULT =====")
    print(f"Input File: {file_path}")
    print(f"Predicted Class: {CLASS_NAMES[pred]}")
    print("Probability Scores:")

    for i, class_name in CLASS_NAMES.items():
        if i < len(probs):
            print(f"{class_name}: {probs[i]:.4f}")
        else:
            print(f"{class_name}: N/A")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_audio.wav>")
        sys.exit(1)

    input_file = sys.argv[1]
    predict_audio(input_file)