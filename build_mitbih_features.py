"""
MIT-BIH PSG -> Feature Dataset Builder (Portable Version)

Goal:
Convert MIT-BIH PSG records (WFDB: .hea + .dat + annotations)
into a final epoch-based dataset with EEG features,
compatible with NeuroSleep Analytics.

Output:
1. Individual CSV per subject
2. Final merged CSV:
   sleep_features_b.csv

USAGE:
  py -3.12 build_mitbih_features.py

REQUIREMENTS:
  py -3.12 -m pip install wfdb numpy pandas scipy

EXPECTED STRUCTURE:
  project/
  ├── build_mitbih_features.py
  ├── data_mitbih/
  │   ├── slp01a.hea
  │   ├── slp01a.dat
  │   ├── slp01a.st
  │   ├── slp01b.hea
  │   ├── slp01b.dat
  │   └── ...
  ├── output/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import welch
from scipy.stats import kurtosis, skew


# =========================================================
# CONFIGURATION (PORTABLE)
# =========================================================

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data_mitbih"
FOLDERS = [DATA_DIR] if DATA_DIR.exists() else []

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_SEC = 30

# Preferred EEG channels
PREFERRED_EEG_CHANNELS = [
    "EEG (C4-A1)",
    "EEG C4-A1",
    "C4-A1",
    "EEG",
    "EEG(sec)",
    "EEG2",
]

# Stage names
STAGE_NAME_MAP = {
    "W": "Wake",
    "1": "N1",
    "2": "N2",
    "3": "N3",
    "4": "N3",
    "R": "REM",
}

# Final numeric mapping compatible with the project
STAGE_NUM_MAP = {
    "W": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 3,
    "R": 4,
}

VALID_STAGE_CODES = set(STAGE_NUM_MAP.keys())


# =========================================================
# UTILITIES
# =========================================================

def find_records(folders: list[Path]) -> list[tuple[Path, str]]:
    records = []

    for p in folders:
        if not p.exists():
            print(f"[!] Folder not found: {p}")
            continue

        for hea in sorted(p.glob("*.hea")):
            record_name = hea.stem
            dat_file = hea.with_suffix(".dat")

            if dat_file.exists():
                records.append((p, record_name))

    return records


def read_sleep_annotations(record_path: Path, record_name: str) -> tuple[pd.DataFrame | None, str | None]:
    """
    Read sleep annotations and extract stage from aux_note.

    Examples:
      '4 LA LA' -> stage_code = '4'
      '3 HA'    -> stage_code = '3'
      '2 LA'    -> stage_code = '2'
      'W ...'   -> stage_code = 'W'
      'R ...'   -> stage_code = 'R'
    """
    ann = None
    used_ext = None

    for ext in ["st", "slp", "hypnogram", "atr"]:
        try:
            ann = wfdb.rdann(str(record_path / record_name), ext)
            used_ext = ext
            break
        except Exception:
            continue

    if ann is None:
        return None, None

    samples = ann.sample
    aux_notes = ann.aux_note if hasattr(ann, "aux_note") else []

    rows = []

    for i in range(len(samples)):
        sample = samples[i]
        aux = aux_notes[i] if i < len(aux_notes) else ""

        if not isinstance(aux, str):
            continue

        aux = aux.strip()
        if not aux:
            continue

        first_char = aux[0].upper()

        if first_char in VALID_STAGE_CODES:
            stage_code = first_char
            rows.append({
                "annotation_sample": sample,
                "stage_code": stage_code,
                "stage_name": STAGE_NAME_MAP[stage_code],
                "sleep_stage": STAGE_NUM_MAP[stage_code],
            })

    if not rows:
        return None, used_ext

    df = pd.DataFrame(rows)
    return df, used_ext


def select_eeg_channel(sig_names: list[str]) -> tuple[int | None, str | None]:
    """
    Find the best available EEG channel.
    """
    for preferred in PREFERRED_EEG_CHANNELS:
        for i, ch in enumerate(sig_names):
            if ch.strip().lower() == preferred.strip().lower():
                return i, ch

    for i, ch in enumerate(sig_names):
        if "eeg" in ch.lower():
            return i, ch

    return None, None


def safe_log(x: float, eps: float = 1e-8) -> float:
    return float(np.log10(max(x, eps)))


def bandpower(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> float:
    mask = (freqs >= fmin) & (freqs < fmax)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def spectral_entropy_from_psd(psd: np.ndarray) -> float:
    psd = np.asarray(psd, dtype=float)
    psd = np.clip(psd, 1e-12, None)
    psd_norm = psd / np.sum(psd)
    return float(-np.sum(psd_norm * np.log2(psd_norm)))


def hjorth_parameters(signal: np.ndarray) -> tuple[float, float, float]:
    signal = np.asarray(signal, dtype=float)

    if len(signal) < 3:
        return 0.0, 0.0, 0.0

    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)

    var_zero = np.var(signal)
    var_d1 = np.var(first_deriv)
    var_d2 = np.var(second_deriv)

    activity = float(var_zero)
    mobility = float(np.sqrt(var_d1 / var_zero)) if var_zero > 1e-12 else 0.0
    complexity = (
        float(np.sqrt(var_d2 / var_d1) / mobility)
        if var_d1 > 1e-12 and mobility > 1e-12
        else 0.0
    )

    return activity, mobility, complexity


def compute_epoch_features(epoch_signal: np.ndarray, fs: float) -> dict | None:
    """
    Extract EEG features for one epoch.
    """
    x = np.asarray(epoch_signal, dtype=float)

    if len(x) < 10:
        return None

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    feat_skewness = float(skew(x))
    feat_kurtosis = float(kurtosis(x))
    feat_peak = float(np.max(np.abs(x)))

    nperseg = min(256, len(x))
    if nperseg < 16:
        return None

    freqs, psd = welch(x, fs=fs, nperseg=nperseg)

    delta = bandpower(freqs, psd, 0.5, 4.0)
    theta = bandpower(freqs, psd, 4.0, 8.0)
    alpha = bandpower(freqs, psd, 8.0, 13.0)
    beta = bandpower(freqs, psd, 13.0, 30.0)

    total_power = delta + theta + alpha + beta + 1e-8

    delta_ratio = delta / total_power
    theta_ratio = theta / total_power
    alpha_ratio = alpha / total_power
    beta_ratio = beta / total_power

    theta_alpha_ratio = theta / (alpha + 1e-8)
    delta_beta_ratio = delta / (beta + 1e-8)

    fft_peak_freq = float(freqs[np.argmax(psd)]) if len(freqs) > 0 else 0.0
    spectral_entropy = spectral_entropy_from_psd(psd)

    hjorth_activity, hjorth_mobility, hjorth_complexity = hjorth_parameters(x)

    return {
        "skewness": feat_skewness,
        "kurtosis": feat_kurtosis,
        "peak": feat_peak,
        "log_delta": safe_log(delta),
        "log_theta": safe_log(theta),
        "log_alpha": safe_log(alpha),
        "log_beta": safe_log(beta),
        "delta_ratio": float(delta_ratio),
        "theta_ratio": float(theta_ratio),
        "alpha_ratio": float(alpha_ratio),
        "beta_ratio": float(beta_ratio),
        "theta_alpha_ratio": float(theta_alpha_ratio),
        "delta_beta_ratio": float(delta_beta_ratio),
        "fft_peak_freq": fft_peak_freq,
        "spectral_entropy": spectral_entropy,
        "hjorth_activity": float(hjorth_activity),
        "hjorth_mobility": float(hjorth_mobility),
        "hjorth_complexity": float(hjorth_complexity),
    }


def build_subject_feature_rows(record_path: Path, record_name: str) -> pd.DataFrame | None:
    """
    Build epoch-level rows for one subject.
    """
    record_full_path = str(record_path / record_name)

    record = wfdb.rdrecord(record_full_path)
    fs = float(record.fs)
    sig_names = record.sig_name

    eeg_idx, eeg_name = select_eeg_channel(sig_names)
    if eeg_idx is None:
        print(f"  [!] {record_name}: no compatible EEG channel found")
        return None

    ann_df, ann_ext = read_sleep_annotations(record_path, record_name)
    if ann_df is None or ann_df.empty:
        print(f"  [!] {record_name}: no valid sleep annotations found")
        return None

    eeg_signal = record.p_signal[:, eeg_idx]
    epoch_samples = int(fs * EPOCH_SEC)

    rows = []

    for epoch_idx, row in enumerate(ann_df.itertuples(index=False), start=0):
        start = int(row.annotation_sample)
        end = start + epoch_samples

        if end > len(eeg_signal):
            continue

        stage_code = row.stage_code
        if stage_code not in VALID_STAGE_CODES:
            continue

        epoch_signal = eeg_signal[start:end]
        features = compute_epoch_features(epoch_signal, fs)

        if features is None:
            continue

        rows.append({
            "subject_id": record_name,
            "epoch": epoch_idx,
            "sleep_stage": int(STAGE_NUM_MAP[stage_code]),
            "stage_name": STAGE_NAME_MAP[stage_code],
            **features,
        })

    if not rows:
        print(f"  [!] {record_name}: no valid epochs could be constructed")
        return None

    df_features = pd.DataFrame(rows)
    print(f"  [✓] {record_name}: EEG={eeg_name} | valid epochs={len(df_features)} | ann=.{ann_ext}")
    return df_features


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    print("=" * 60)
    print("MIT-BIH PSG -> Feature Dataset Builder (Portable)")
    print("=" * 60)

    if not DATA_DIR.exists():
        print(f"\n[ERROR] Data folder not found: {DATA_DIR}")
        print("👉 Create a folder named 'data_mitbih' next to this script and place your dataset files inside it.")
        sys.exit(1)

    records = find_records(FOLDERS)
    if not records:
        print("\n[ERROR] No WFDB records were found.")
        print("👉 Make sure your .hea and .dat files are directly inside 'data_mitbih'.")
        sys.exit(1)

    print(f"\nRecords found: {len(records)}")
    for folder, name in records:
        print(f" - {name}  [{folder.name}]")

    all_subjects = []
    processed = 0
    skipped = 0

    for folder, record_name in records:
        print(f"\n[→] Processing subject: {record_name}")

        try:
            df_subject = build_subject_feature_rows(folder, record_name)

            if df_subject is None or df_subject.empty:
                skipped += 1
                continue

            subject_csv = OUTPUT_DIR / f"{record_name}_features.csv"
            df_subject.to_csv(subject_csv, index=False, encoding="utf-8")
            print(f"  [✓] Saved individual CSV: {subject_csv.name}")

            all_subjects.append(df_subject)
            processed += 1

        except Exception as e:
            print(f"  [ERROR] {record_name}: {e}")
            skipped += 1

    if not all_subjects:
        print("\n[ERROR] No feature dataset was generated.")
        sys.exit(1)

    final_df = pd.concat(all_subjects, ignore_index=True)

    final_csv = OUTPUT_DIR / "sleep_features_b.csv"
    final_df.to_csv(final_csv, index=False, encoding="utf-8")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Subjects processed successfully : {processed}")
    print(f"Subjects skipped / failed      : {skipped}")
    print(f"Total final rows               : {len(final_df)}")
    print(f"Total columns                  : {len(final_df.columns)}")
    print(f"Final file generated           : {final_csv}")
    print("=" * 60)

    print("\nFinal columns:")
    print(", ".join(final_df.columns))


if __name__ == "__main__":
    main()
