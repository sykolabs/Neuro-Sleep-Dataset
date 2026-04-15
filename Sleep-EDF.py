"""
extract_features.py  v3 (PORTABLE VERSION)
==========================================

Mejoras:
- Rutas relativas (funciona en cualquier computadora)
- Creación automática de carpetas
- Compatible con GitHub y Docker
- Features avanzadas EEG (log power, Hjorth, entropy, ratios)

USO:
  py -3.12 extract_features.py

REQUISITOS:
  py -3.12 -m pip install mne pandas numpy

ESTRUCTURA ESPERADA:
  project/
  ├── extract_features.py
  ├── data_sleep_edf/
  │   ├── SC4001E0-PSG.edf
  │   ├── SC4001EC-Hypnogram.edf
  ├── output/
"""

import os, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    import mne
    mne.set_log_level('ERROR')
except ImportError:
    print("Falta MNE: py -3.12 -m pip install mne"); exit(1)

# =========================================================
# 🔥 RUTAS PORTABLES
# =========================================================

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data_sleep_edf"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "sleep_features.csv"

# =========================================================

EPOCH_SEC  = 30
EEG_CHANNEL = 'EEG Fpz-Cz'

STAGE_MAP = {
    'Sleep stage W':  0, 'Sleep stage 1':  1, 'Sleep stage 2':  2,
    'Sleep stage 3':  3, 'Sleep stage 4':  3, 'Sleep stage R':  4,
    'Sleep stage ?': -1, 'Movement time': -1,
}

STAGE_NAMES = {0:'Wake', 1:'N1', 2:'N2', 3:'N3', 4:'REM'}

BANDS = {
    'delta':(0.5,4.0),
    'theta':(4.0,8.0),
    'alpha':(8.0,13.0),
    'beta':(13.0,30.0)
}

# =========================================================
# FEATURES
# =========================================================

def band_power(sig, fs, lo, hi):
    freqs = np.fft.rfftfreq(len(sig), 1.0/fs)
    fft_v = np.abs(np.fft.rfft(sig))**2
    idx   = np.where((freqs>=lo)&(freqs<=hi))[0]
    return float(np.mean(fft_v[idx])) if len(idx) else 1e-12

def spectral_entropy(sig, fs):
    fft_v = np.abs(np.fft.rfft(sig))**2
    fft_v = fft_v/(fft_v.sum()+1e-12)
    fft_v = fft_v[fft_v>0]
    return float(-np.sum(fft_v*np.log2(fft_v)))

def hjorth(sig):
    act  = float(np.var(sig))
    d1   = np.diff(sig)
    mob  = float(np.sqrt(np.var(d1)/(act+1e-12)))
    d2   = np.diff(d1)
    comp = float(np.sqrt(np.var(d2)/(np.var(d1)+1e-12))/(mob+1e-12))
    return act, mob, comp

def safe_skew(x):
    s=np.std(x); return float(np.mean(((x-np.mean(x))/s)**3)) if s>1e-10 else 0.0

def safe_kurt(x):
    s=np.std(x); return float(np.mean(((x-np.mean(x))/s)**4)-3) if s>1e-10 else 0.0

def extract_features(epoch, fs):
    f = {}
    f['skewness'] = safe_skew(epoch)
    f['kurtosis'] = safe_kurt(epoch)
    f['peak']     = float(np.max(np.abs(epoch)))

    powers = {b: band_power(epoch, fs, lo, hi) for b,(lo,hi) in BANDS.items()}
    for b, bp in powers.items():
        f[f'log_{b}'] = float(np.log10(bp+1e-12))

    total = sum(powers.values())+1e-12
    for b in BANDS:
        f[f'{b}_ratio'] = round(powers[b]/total, 6)

    f['theta_alpha_ratio'] = round(powers['theta']/(powers['alpha']+1e-12), 4)
    f['delta_beta_ratio']  = round(powers['delta']/(powers['beta']+1e-12),  4)

    freqs = np.fft.rfftfreq(len(epoch), 1.0/fs)
    fft_v = np.abs(np.fft.rfft(epoch))
    idx30 = np.where(freqs<=30)[0]

    f['fft_peak_freq'] = float(freqs[idx30[np.argmax(fft_v[idx30])]]) if len(idx30) else 0.0
    f['spectral_entropy'] = spectral_entropy(epoch, fs)

    act, mob, comp = hjorth(epoch)
    f['hjorth_activity']   = float(np.log10(act+1e-12))
    f['hjorth_mobility']   = mob
    f['hjorth_complexity'] = comp

    return f

# =========================================================
# DATA LOADING
# =========================================================

def find_pairs(d):
    files = os.listdir(str(d))
    psgs  = sorted([f for f in files if f.endswith('-PSG.edf')])
    hyps  = sorted([f for f in files if 'Hypnogram' in f and f.endswith('.edf')])

    pairs = []
    for psg in psgs:
        m = re.match(r'(SC\d{4})', psg)
        if not m: continue

        code  = m.group(1)
        hypno = next((h for h in hyps if h.startswith(code)), None)

        if not hypno:
            print(f"  Sin hypnogram para {psg}")
            continue

        pairs.append((
            int(code.replace('SC','')),
            d / psg,
            d / hypno
        ))

    return pairs

def process(sid, psg_path, hyp_path):
    print(f"  PSG:   {psg_path.name}")
    print(f"  Hypno: {hyp_path.name}")

    try:
        raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)
    except Exception as e:
        print(f"  ERROR PSG: {e}")
        return None

    avail = raw.ch_names

    ch = EEG_CHANNEL if EEG_CHANNEL in avail else \
         ('EEG Pz-Oz' if 'EEG Pz-Oz' in avail else \
         next((c for c in avail if 'EEG' in c.upper()), None))

    if not ch:
        print(f"  Sin canal EEG")
        return None

    if ch != EEG_CHANNEL:
        print(f"  Usando canal alternativo: {ch}")

    raw.pick([ch])

    fs  = int(raw.info['sfreq'])
    sig = raw.get_data()[0]

    try:
        annot = mne.read_annotations(str(hyp_path))
    except Exception as e:
        print(f"  ERROR hypno: {e}")
        return None

    spe  = fs * EPOCH_SEC
    rows = []

    for i, ann in enumerate(annot):
        stage = STAGE_MAP.get(ann['description'], -1)
        if stage == -1:
            continue

        s = int(ann['onset']*fs)
        e = s + spe

        if e > len(sig):
            break

        epoch = sig[s:e].astype(np.float64)

        if np.std(epoch) < 1e-8:
            continue

        feat = extract_features(epoch, fs)

        feat.update({
            'subject_id': sid,
            'epoch': i,
            'sleep_stage': stage,
            'stage_name': STAGE_NAMES[stage]
        })

        rows.append(feat)

    if not rows:
        print(f"  Sin epocas validas")
        return None

    return pd.DataFrame(rows)

# =========================================================
# MAIN
# =========================================================

def main():
    print("="*60)
    print("  DV Projects - EEG Feature Extraction v3 (Portable)")
    print("="*60)

    if not DATA_DIR.exists():
        print(f"Carpeta no encontrada: {DATA_DIR}")
        print("👉 Crea la carpeta 'data_sleep_edf' y pon los .edf dentro")
        return

    pairs = find_pairs(DATA_DIR)

    if not pairs:
        print("Sin pares PSG+Hypnogram")
        return

    print(f"\n{len(pairs)} sujetos encontrados\n")

    all_dfs = []

    for sid, psg, hyp in pairs:
        print(f"-- Sujeto {sid} --")
        df = process(sid, psg, hyp)

        if df is not None:
            all_dfs.append(df)

        print()

    if not all_dfs:
        print("Sin resultados")
        return

    final = pd.concat(all_dfs, ignore_index=True)

    feat_cols = [c for c in final.columns if c not in ('subject_id','epoch','sleep_stage','stage_name')]

    final = final[['subject_id','epoch'] + feat_cols + ['sleep_stage','stage_name']]

    for c in feat_cols:
        final[c] = final[c].round(6)

    final.to_csv(OUTPUT_CSV, index=False)

    print("\nCSV generado:", OUTPUT_CSV)

# =========================================================

if __name__ == '__main__':
    main()
