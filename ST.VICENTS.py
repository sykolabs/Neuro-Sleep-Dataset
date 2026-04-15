"""
ST. VINCENT (UCDDB) -> Feature Dataset Builder
----------------------------------------------
Convierte registros St. Vincent / UCDDB (.rec + _stage/_stage.txt + _respevt/_respevt.txt)
a un dataset por epochs de 30 s con features de un único canal EEG.

Salida:
1. CSV individual por sujeto
2. CSV unificado final:
   st_vincent_sleep_features.csv

Requisitos:
    pip install mne numpy pandas scipy

Uso:
    py -3.12 build_st_vincent_features.py
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

DATA_DIR = Path(r"C:\Users\orlyj\OneDrive\Documents\Pro1Dataset\Pro1Dataset\ST. Vicents dataset")
OUTPUT_DIR = Path(r"./output_st_vincent")
EPOCH_SEC = 30

# Un solo canal EEG
PREFERRED_EEG_CHANNELS = [
    "EEG (C3-A2)",
    "EEG C3-A2",
    "C3-A2",
    "C3A2",
    "EEG (C4-A1)",
    "EEG C4-A1",
    "C4-A1",
    "C4A1",
]

# UCDDB original:
# 0 Wake, 1 REM, 2 Stage1, 3 Stage2, 4 Stage3, 5 Stage4, 6 Artifact, 7 Indeterminate
STAGE_NAME_MAP_RAW = {
    0: "Wake",
    1: "REM",
    2: "N1",
    3: "N2",
    4: "N3",
    5: "N3",   # Stage 4 colapsado a N3
    6: "Artifact",
    7: "Indeterminate",
}

# Mapeo final compatible con tu proyecto
# Wake=0, N1=1, N2=2, N3=3, REM=4
STAGE_NUM_MAP = {
    0: 0,  # Wake
    1: 4,  # REM
    2: 1,  # Stage 1 -> N1
    3: 2,  # Stage 2 -> N2
    4: 3,  # Stage 3 -> N3
    5: 3,  # Stage 4 -> N3
}

VALID_STAGE_CODES = set(STAGE_NUM_MAP.keys())


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
# HELPERS DE ARCHIVOS
# =========================================================

def resolve_optional_txt(base_dir: Path, stem_name: str) -> Path | None:
    candidates = [
        base_dir / f"{stem_name}.txt",
        base_dir / stem_name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def find_subjects(data_dir: Path) -> list[str]:
    subjects = []
    for rec_file in sorted(data_dir.glob("ucddb*.rec")):
        subject_id = rec_file.stem

        stage_candidates = [
            data_dir / f"{subject_id}_stage.txt",
            data_dir / f"{subject_id}_stage",
        ]

        if any(p.exists() for p in stage_candidates):
            subjects.append(subject_id)

    return subjects


# =========================================================
# IO / PARSERS
# =========================================================

def open_rec_as_edf(rec_path: Path):
    """
    Los .rec se leen como EDF haciendo una copia temporal con extensión .edf.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_edf = Path(tmpdir) / f"{rec_path.stem}.edf"
        shutil.copyfile(rec_path, tmp_edf)
        raw = mne.io.read_raw_edf(str(tmp_edf), preload=True, verbose=False)
        return raw.copy()


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
    """
    Selecciona un único canal EEG.

    Prioridad:
    1) C3-A2 / C3A2
    2) C4-A1 / C4A1
    3) Match flexible
    4) Fallback al primer canal
    """
    normalized_map = [(i, ch, normalize_channel_name(ch)) for i, ch in enumerate(ch_names)]

    preferred = [
        "eegc3a2",
        "c3a2",
        "eegc4a1",
        "c4a1",
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


def read_stage_file(stage_path: Path) -> pd.DataFrame | None:
    """
    Lee _stage o _stage.txt.
    Formato real: un número por línea.
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
            if 0 <= code <= 7:
                values.append(code)

    if not values:
        return None

    rows = []
    for epoch_idx, code in enumerate(values):
        if code not in VALID_STAGE_CODES:
            continue

        rows.append({
            "epoch": epoch_idx,
            "stage_code_raw": code,
            "stage_name": STAGE_NAME_MAP_RAW[code],
            "sleep_stage": STAGE_NUM_MAP[code],
        })

    out = pd.DataFrame(rows)
    return out if not out.empty else None


def read_respevt_file(respevt_path: Path) -> list[tuple[float, float]]:
    """
    Parser específico para ST. Vincent:
    HH:MM:SS  TYPE  DURATION
    """
    if not respevt_path.exists():
        return []

    intervals = []

    try:
        lines = respevt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 3:
            continue

        time_str = parts[0]
        event_type = parts[1]
        duration_str = parts[2]

        if not ("APNEA" in event_type or "HYP" in event_type):
            continue

        try:
            h, m, s = map(int, time_str.split(":"))
            start_sec = h * 3600 + m * 60 + s
        except Exception:
            continue

        try:
            duration_sec = float(duration_str)
        except Exception:
            continue

        end_sec = start_sec + duration_sec
        intervals.append((start_sec, end_sec))

    return intervals


def epoch_has_apnea(epoch_start_sec: float, epoch_end_sec: float, resp_intervals: list[tuple[float, float]]) -> int:
    for start_sec, end_sec in resp_intervals:
        overlap = min(epoch_end_sec, end_sec) - max(epoch_start_sec, start_sec)
        if overlap > 0:
            return 1
    return 0


# =========================================================
# MAIN PIPELINE
# =========================================================

def build_subject_feature_rows(data_dir: Path, subject_id: str) -> pd.DataFrame | None:
    rec_path = data_dir / f"{subject_id}.rec"
    stage_path = resolve_optional_txt(data_dir, f"{subject_id}_stage")
    respevt_path = resolve_optional_txt(data_dir, f"{subject_id}_respevt")

    if not rec_path.exists() or stage_path is None:
        print(f"  [!] {subject_id}: faltan archivos requeridos")
        return None

    try:
        raw = open_rec_as_edf(rec_path)
    except Exception as e:
        print(f"  [!] {subject_id}: error leyendo .rec como EDF -> {e}")
        return None

    fs = float(raw.info["sfreq"])
    ch_names = raw.ch_names

    print(f"  [DEBUG] Canales disponibles en {subject_id}: {ch_names}")

    eeg_idx, eeg_name = select_eeg_channel(ch_names)
    if eeg_idx is None:
        print(f"  [!] {subject_id}: no se encontró canal EEG compatible")
        return None

    stage_df = read_stage_file(stage_path)
    if stage_df is None or stage_df.empty:
        print(f"  [!] {subject_id}: stage file vacío o inválido")
        return None

    resp_intervals = read_respevt_file(respevt_path) if respevt_path is not None else []

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

        epoch_start_sec = start / fs
        epoch_end_sec = end / fs
        apnea_label = epoch_has_apnea(epoch_start_sec, epoch_end_sec, resp_intervals)

        rows.append({
            "dataset": "ST_Vincent",
            "subject_id": subject_id,
            "epoch": epoch_idx,
            "sleep_stage": int(row.sleep_stage),
            "stage_name": row.stage_name,
            "apnea_label": apnea_label,
            **features,
        })

    if not rows:
        print(f"  [!] {subject_id}: no se pudieron construir epochs válidos")
        return None

    df_features = pd.DataFrame(rows)
    print(f"  [✓] {subject_id}: EEG={eeg_name} | epochs válidos={len(df_features)} | apnea_events={len(resp_intervals)}")
    return df_features


def main():
    print("=" * 60)
    print("ST. VINCENT -> Feature Dataset Builder")
    print("Single EEG channel | 30 s epochs | sleep + apnea")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    subjects = find_subjects(DATA_DIR)
    if not subjects:
        print("\n[ERROR] No se encontraron sujetos válidos.")
        print("Verifica la ruta y la presencia de .rec + _stage")
        return

    print(f"\nSujetos encontrados: {len(subjects)}")
    for s in subjects:
        print(f" - {s}")

    all_subjects = []
    processed = 0
    skipped = 0

    for subject_id in subjects:
        print(f"\n[→] Procesando sujeto: {subject_id}")

        try:
            df_subject = build_subject_feature_rows(DATA_DIR, subject_id)

            if df_subject is None or df_subject.empty:
                skipped += 1
                continue

            subject_csv = OUTPUT_DIR / f"{subject_id}_features.csv"
            df_subject.to_csv(subject_csv, index=False, encoding="utf-8")
            print(f"  [✓] Guardado CSV individual: {subject_csv.name}")

            all_subjects.append(df_subject)
            processed += 1

        except Exception as e:
            print(f"  [ERROR] {subject_id}: {e}")
            skipped += 1

    if not all_subjects:
        print("\n[ERROR] No se generó ningún dataset de features.")
        return

    final_df = pd.concat(all_subjects, ignore_index=True)

    final_csv = OUTPUT_DIR / "st_vincent_sleep_features.csv"
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
