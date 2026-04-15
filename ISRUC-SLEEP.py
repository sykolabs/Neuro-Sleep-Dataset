"""
ISRUC-SLEEP -> Feature Dataset Builder
--------------------------------------
Convierte registros ISRUC (.rec + 1_1 / *_1) a un dataset por epochs de 30 s
con features de un único canal EEG.

Salida:
1. CSV individual por sujeto
2. CSV unificado final:
   isruc_sleep_features.csv

Requisitos:
    pip install mne numpy pandas scipy

Uso:
    py -3.12 build_isruc_features.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis

try:
    import mne
    mne.set_log_level("ERROR")
except ImportError:
    raise SystemExit("Falta mne. Instala con: py -3.12 -m pip install mne")


# =========================================================
# CONFIGURACIÓN
# =========================================================

# Carpeta que contiene subcarpetas 1, 2, 3, ...
DATA_DIR = Path(__file__).parent
OUTPUT_DIR = Path(r"./output_isruc")
EPOCH_SEC = 30

# Un solo canal EEG
PREFERRED_EEG_CHANNELS = [
    "C3-A2",
    "C3A2",
    "EEG C3-A2",
    "EEG (C3-A2)",
    "C4-A1",
    "C4A1",
    "EEG C4-A1",
    "EEG (C4-A1)",
]

# ISRUC final compatible con tu proyecto
# 0=Wake, 1=N1, 2=N2, 3=N3, 5=REM
ISRUC_STAGE_MAP = {
    0: 0,  # Wake
    1: 1,  # N1
    2: 2,  # N2
    3: 3,  # N3
    5: 4,  # REM
}

ISRUC_STAGE_NAME = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3",
    5: "REM",
}


# =========================================================
# FEATURES
# =========================================================

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


# =========================================================
# HELPERS
# =========================================================

def normalize_channel_name(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .replace("(", "")
        .replace(")", "")
    )


def select_eeg_channel(ch_names: list[str]) -> tuple[int | None, str | None]:
    normalized_map = [(i, ch, normalize_channel_name(ch)) for i, ch in enumerate(ch_names)]

    preferred = [
        "c3a2",
        "eegc3a2",
        "c4a1",
        "eegc4a1",
        "c3",
        "c4",
    ]

    for pref in preferred:
        for i, ch, norm in normalized_map:
            if norm == pref:
                return i, ch

    for i, ch, norm in normalized_map:
        if "c3" in norm and "a2" in norm:
            return i, ch

    for i, ch, norm in normalized_map:
        if "c4" in norm and "a1" in norm:
            return i, ch

    for i, ch, norm in normalized_map:
        if "eeg" in norm and "c3" in norm:
            return i, ch

    for i, ch, norm in normalized_map:
        if "eeg" in norm and "c4" in norm:
            return i, ch

    if len(ch_names) > 0:
        return 0, ch_names[0]

    return None, None


def open_rec_as_edf(rec_path: Path):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_edf = Path(tmpdir) / f"{rec_path.stem}.edf"
        shutil.copyfile(rec_path, tmp_edf)
        raw = mne.io.read_raw_edf(str(tmp_edf), preload=True, verbose=False)
        return raw.copy()


def resolve_scoring_file(subject_dir: Path, subject_name: str) -> Path | None:
    """
    Usa SOLO el anotador 1.
    Acepta:
    - 1_1
    - 1_1.txt
    """
    candidates = [
        subject_dir / f"{subject_name}_1.txt",
        subject_dir / f"{subject_name}_1",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def find_subjects(data_dir: Path) -> list[tuple[str, Path]]:
    """
    Busca subcarpetas 1, 2, 3... y verifica que tengan:
    - 1.rec
    - 1_1 o 1_1.txt
    """
    subjects = []

    for subject_dir in sorted([p for p in data_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        subject_name = subject_dir.name

        rec_file = subject_dir / f"{subject_name}.rec"
        score_file = resolve_scoring_file(subject_dir, subject_name)

        if rec_file.exists() and score_file is not None:
            subjects.append((subject_name, subject_dir))

    return subjects


def read_stage_file(stage_path: Path) -> pd.DataFrame | None:
    """
    Lee 1_1 o 1_1.txt.
    Formato real observado: un número por línea. :contentReference[oaicite:1]{index=1}
    """
    try:
        lines = stage_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None

    values = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.isdigit():
            code = int(s)
            values.append(code)

    if not values:
        return None

    rows = []
    for epoch_idx, code in enumerate(values):
        if code not in ISRUC_STAGE_MAP:
            continue

        rows.append({
            "epoch": epoch_idx,
            "stage_code_raw": code,
            "stage_name": ISRUC_STAGE_NAME[code],
            "sleep_stage": ISRUC_STAGE_MAP[code],
        })

    out = pd.DataFrame(rows)
    return out if not out.empty else None


# =========================================================
# PIPELINE
# =========================================================

def build_subject_feature_rows(subject_name: str, subject_dir: Path) -> pd.DataFrame | None:
    rec_path = subject_dir / f"{subject_name}.rec"
    stage_path = resolve_scoring_file(subject_dir, subject_name)

    if not rec_path.exists() or stage_path is None:
        print(f"  [!] {subject_name}: faltan archivos requeridos")
        return None

    try:
        raw = open_rec_as_edf(rec_path)
    except Exception as e:
        print(f"  [!] {subject_name}: error leyendo .rec como EDF -> {e}")
        return None

    fs = float(raw.info["sfreq"])
    ch_names = raw.ch_names

    print(f"  [DEBUG] Canales disponibles en {subject_name}: {ch_names}")

    eeg_idx, eeg_name = select_eeg_channel(ch_names)
    if eeg_idx is None:
        print(f"  [!] {subject_name}: no se encontró canal EEG compatible")
        return None

    stage_df = read_stage_file(stage_path)
    if stage_df is None or stage_df.empty:
        print(f"  [!] {subject_name}: scoring file vacío o inválido")
        return None

    eeg_signal = raw.get_data(picks=[eeg_idx])[0]
    epoch_samples = int(fs * EPOCH_SEC)

    rows = []

    for row in stage_df.itertuples(index=False):
        epoch_idx = int(row.epoch)
        start = epoch_idx * epoch_samples
        end = start + epoch_samples

        if end > len(eeg_signal):
            continue

        epoch_signal = eeg_signal[start:end]
        features = compute_epoch_features(epoch_signal, fs)
        if features is None:
            continue

        rows.append({
            "dataset": "ISRUC_Sleep",
            "subject_id": subject_name,
            "epoch": epoch_idx,
            "sleep_stage": int(row.sleep_stage),
            "stage_name": row.stage_name,
            **features,
        })

    if not rows:
        print(f"  [!] {subject_name}: no se pudieron construir epochs válidos")
        return None

    df_features = pd.DataFrame(rows)
    print(f"  [✓] {subject_name}: EEG={eeg_name} | epochs válidos={len(df_features)}")
    return df_features


def main():
    print("=" * 60)
    print("ISRUC-SLEEP -> Feature Dataset Builder")
    print("Single EEG channel | 30 s epochs | scorer 1 only")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    subjects = find_subjects(DATA_DIR)
    if not subjects:
        print("\n[ERROR] No se encontraron sujetos válidos.")
        print("Verifica la estructura: ISRUC/1/1.rec + 1_1")
        return

    print(f"\nSujetos encontrados: {len(subjects)}")
    for subject_name, _ in subjects:
        print(f" - {subject_name}")

    all_subjects = []
    processed = 0
    skipped = 0

    for subject_name, subject_dir in subjects:
        print(f"\n[→] Procesando sujeto: {subject_name}")

        try:
            df_subject = build_subject_feature_rows(subject_name, subject_dir)

            if df_subject is None or df_subject.empty:
                skipped += 1
                continue

            subject_csv = OUTPUT_DIR / f"{subject_name}_features.csv"
            df_subject.to_csv(subject_csv, index=False, encoding="utf-8")
            print(f"  [✓] Guardado CSV individual: {subject_csv.name}")

            all_subjects.append(df_subject)
            processed += 1

        except Exception as e:
            print(f"  [ERROR] {subject_name}: {e}")
            skipped += 1

    if not all_subjects:
        print("\n[ERROR] No se generó ningún dataset de features.")
        return

    final_df = pd.concat(all_subjects, ignore_index=True)

    final_csv = OUTPUT_DIR / "isruc_sleep_features.csv"
    final_df.to_csv(final_csv, index=False, encoding="utf-8")

    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    print(f"Sujetos procesados correctamente : {processed}")
    print(f"Sujetos omitidos / fallidos     : {skipped}")
    print(f"Total de filas finales          : {len(final_df)}")
    print(f"Total de columnas               : {len(final_df.columns)}")
    print(f"Archivo final generado          : {final_csv}")
    print("=" * 60)

    print("\nColumnas finales:")
    print(", ".join(final_df.columns))


if __name__ == "__main__":
    main()
