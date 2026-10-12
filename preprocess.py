import numpy as np
from scipy.signal import butter, filtfilt, welch

def bandpass(x, low=4, high=45, fs=128, order=4):
    b, a = butter(order, [low, high], btype="band", fs=fs)
    return filtfilt(b, a, x, axis=-1)

def preprocess_subject(d, win=512, step=256):
    data = d["data"][:, :32, 384:]
    data = bandpass(data)
    mean = data.mean(axis=(0, 2), keepdims=True)
    std = data.std(axis=(0, 2), keepdims=True)
    data = ((data - mean) / std).astype(np.float32)

    y_trial = (d["labels"][:, :2] > 5).astype(int)

    X, y, trial_id = [], [], []
    for t in range(data.shape[0]):
        for start in range(0, data.shape[2] - win + 1, step):
            X.append(data[t, :, start:start + win])
            y.append(y_trial[t])
            trial_id.append(t)
    return np.array(X), np.array(y), np.array(trial_id)

def band_power_features(X, fs=128):
    bands = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
    f, psd = welch(X, fs=fs, nperseg=128, axis=-1)
    feats = []
    for lo, hi in bands.values():
        idx = (f >= lo) & (f < hi)
        feats.append(np.log(psd[:, :, idx].mean(axis=-1) + 1e-10))
    return np.concatenate(feats, axis=1)
