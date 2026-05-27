import os
import numpy as np
import librosa
from scipy.signal import butter, lfilter


TARGET_SR = 44100
FIXED_DURATION = 5.0  # seconds
USE_BANDPASS = True
LOWCUT = 300.0
HIGHCUT = 15000.0


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass_filter(signal, sr, lowcut=300.0, highcut=15000.0, order=5):
    b, a = butter_bandpass(lowcut, highcut, sr, order=order)
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal


def normalize_audio(signal):
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        return signal
    return signal / max_val


def fix_length(signal, sr, duration=5.0):
    target_length = int(sr * duration)
    if len(signal) > target_length:
        signal = signal[:target_length]
    else:
        padding = target_length - len(signal)
        signal = np.pad(signal, (0, padding), mode='constant')
    return signal


def load_and_preprocess_audio(file_path, target_sr=TARGET_SR, duration=FIXED_DURATION,
                              use_bandpass=USE_BANDPASS, trim_silence=True):
    """
    Load and preprocess an audio file.

    Steps:
    1. Load audio
    2. Resample to target_sr
    3. Trim silence
    4. Normalize amplitude
    5. Apply optional bandpass filter
    6. Pad/trim to fixed duration
    """
    signal, sr = librosa.load(file_path, sr=target_sr, mono=True)

    if trim_silence:
        signal, _ = librosa.effects.trim(signal, top_db=20)

    signal = normalize_audio(signal)

    if use_bandpass:
        signal = apply_bandpass_filter(signal, sr, lowcut=LOWCUT, highcut=HIGHCUT)

    signal = fix_length(signal, sr, duration)
    signal = normalize_audio(signal)

    return signal, sr


def get_audio_files_from_directory(root_dir, extensions=(".wav",)):
    audio_files = []
    for current_root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(extensions):
                audio_files.append(os.path.join(current_root, file))
    return sorted(audio_files)