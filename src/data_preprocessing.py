import numpy as np
import pandas as pd
from scipy.signal import welch

SENSOR_CHANNELS = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
)


def time_domain_features(window):
    centered = window - window.mean()
    return {
        "mean": float(np.mean(window)),
        "std": float(np.std(window)),
        "rms": float(np.sqrt(np.mean(window**2))),
        "min": float(np.min(window)),
        "max": float(np.max(window)),
        "zero_crossing_rate": float(
            ((centered[:-1] * centered[1:]) < 0).sum()
        ),
    }


def freq_domain_features(window, sampling_rate=50):
    freqs, psd = welch(window, fs=sampling_rate, nperseg=len(window))
    psd_sum = float(psd.sum())
    psd_norm = psd / psd_sum if psd_sum else np.zeros_like(psd)
    return {
        "dominant_freq": float(freqs[np.argmax(psd)]),
        "spectral_energy": float(np.sum(psd**2)),
        "spectral_entropy": float(
            -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        ),
    }


def safe_corr(first, second):
    if first.std() == 0 or second.std() == 0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def gravity_features(total_acc_x, total_acc_y, total_acc_z):
    return {
        "grav_mean_x": float(total_acc_x.mean()),
        "grav_mean_y": float(total_acc_y.mean()),
        "grav_mean_z": float(total_acc_z.mean()),
        "grav_std_x": float(total_acc_x.std()),
        "grav_std_y": float(total_acc_y.std()),
        "grav_std_z": float(total_acc_z.std()),
    }


def build_feature_row(data):
    row = {}
    channels = {
        name: np.asarray(data[name], dtype=float) for name in SENSOR_CHANNELS
    }
    for name, window in channels.items():
        for feature_name, value in time_domain_features(window).items():
            row[f"{name}_{feature_name}"] = value
        for feature_name, value in freq_domain_features(window).items():
            row[f"{name}_{feature_name}"] = value

    total_acc = {
        axis: np.asarray(data[f"total_acc_{axis}"], dtype=float)
        for axis in ("x", "y", "z")
    }
    row.update(
        gravity_features(total_acc["x"], total_acc["y"], total_acc["z"])
    )
    row.update(
        {
            "corr_xy": safe_corr(
                channels["body_acc_x"], channels["body_acc_y"]
            ),
            "corr_xz": safe_corr(
                channels["body_acc_x"], channels["body_acc_z"]
            ),
            "corr_yz": safe_corr(
                channels["body_acc_y"], channels["body_acc_z"]
            ),
        }
    )
    return row


def build_feature_frame(data, feature_columns):
    row = build_feature_row(data)
    missing = [column for column in feature_columns if column not in row]
    if missing:
        raise ValueError(f"Missing expected features: {missing}")
    values = [[row[column] for column in feature_columns]]
    return pd.DataFrame(values, columns=feature_columns)


def validate_sensor_window(data, window_length=128):
    required = (*SENSOR_CHANNELS, "total_acc_x", "total_acc_y", "total_acc_z")
    missing = [channel for channel in required if channel not in data]
    if missing:
        raise ValueError(f"Missing sensor channels: {missing}")
    invalid = [
        channel
        for channel in required
        if len(data[channel]) != window_length
        or not np.isfinite(np.asarray(data[channel], dtype=float)).all()
    ]
    if invalid:
        raise ValueError(
            "Each sensor channel must contain "
            f"{window_length} finite numeric values: {invalid}"
        )
