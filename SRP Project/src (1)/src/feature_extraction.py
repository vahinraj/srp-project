import os
import numpy as np
import pandas as pd
import librosa

from preprocess import load_and_preprocess_audio, get_audio_files_from_directory


LABEL_MAP = {
    "normal": 0,
    "slight_fault": 1,
    "severe_fault": 2
}


def extract_features(signal, sr, n_mfcc=13):
    """
    Extract a single feature vector from one audio signal.
    Features:
    - MFCC mean (13)
    - MFCC variance (13)
    - ZCR mean, variance
    - RMS mean, variance
    - Spectral Centroid mean, variance
    - Spectral Bandwidth mean, variance
    - Spectral Roll-off mean, variance
    """
    features = []

    # MFCC
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_var = np.var(mfcc, axis=1)
    features.extend(mfcc_mean)
    features.extend(mfcc_var)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(signal)
    features.append(np.mean(zcr))
    features.append(np.var(zcr))

    # RMS Energy
    rms = librosa.feature.rms(y=signal)
    features.append(np.mean(rms))
    features.append(np.var(rms))

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    features.append(np.mean(centroid))
    features.append(np.var(centroid))

    # Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    features.append(np.mean(bandwidth))
    features.append(np.var(bandwidth))

    # Spectral Roll-off
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
    features.append(np.mean(rolloff))
    features.append(np.var(rolloff))

    return np.array(features, dtype=np.float32)


def get_feature_names(n_mfcc=13):
    names = []
    for i in range(n_mfcc):
        names.append(f"mfcc_{i+1}_mean")
    for i in range(n_mfcc):
        names.append(f"mfcc_{i+1}_var")

    names += [
        "zcr_mean", "zcr_var",
        "rms_mean", "rms_var",
        "centroid_mean", "centroid_var",
        "bandwidth_mean", "bandwidth_var",
        "rolloff_mean", "rolloff_var"
    ]
    return names


def infer_label_from_path(file_path):
    parts = os.path.normpath(file_path).split(os.sep)
    for label_name in LABEL_MAP.keys():
        if label_name in parts:
            return LABEL_MAP[label_name], label_name
    raise ValueError(f"Could not infer label from path: {file_path}")


def build_dataset(data_dir="../data", output_csv="../outputs/features_dataset.csv"):
    audio_files = get_audio_files_from_directory(data_dir, extensions=(".wav",))
    rows = []
    feature_names = get_feature_names()

    print(f"Found {len(audio_files)} audio files.")

    for file_path in audio_files:
        try:
            label_id, label_name = infer_label_from_path(file_path)
            signal, sr = load_and_preprocess_audio(file_path)
            features = extract_features(signal, sr)

            row = dict(zip(feature_names, features))
            row["label"] = label_id
            row["label_name"] = label_name
            row["file_path"] = file_path

            rows.append(row)
            print(f"Processed: {file_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Dataset saved to {output_csv}")
    return df


if __name__ == "__main__":
    build_dataset()