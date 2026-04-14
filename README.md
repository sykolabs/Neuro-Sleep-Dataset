# 🧠 NeuroSleep Data Processing Pipeline

This project was developed to **convert raw polysomnography (PSG) sleep data into labeled, machine learning-ready datasets** using a single EEG channel.

The main objective is to standardize multiple public sleep datasets into a unified format for:

* Sleep stage classification
* EEG signal analysis
* Machine Learning applications
* Apnea detection (where available)

---

# 🎯 Purpose of the Project

Each script in this repository was created to solve the same problem across different datasets:

👉 **Raw sleep data is complex and not directly usable for Machine Learning.**

Therefore, each dataset required a custom pipeline to:

* Extract EEG signals
* Segment data into 30-second epochs
* Convert annotations into labels
* Compute meaningful features
* Export structured CSV datasets

---

# 📊 Supported Datasets & Description

## 1. Sleep-EDF Dataset

### 📌 Purpose

This script was created to process Sleep-EDF data, which contains EEG recordings and hypnogram annotations, and convert them into labeled feature datasets.

### ⚙️ What it does

* Reads `.edf` EEG signals
* Loads hypnogram annotations
* Segments signals into 30-second epochs
* Extracts EEG features
* Maps sleep stages into standardized labels

### 📤 Output

* `sleep_features.csv`

---

## 2. MIT-BIH PSG Dataset

### 📌 Purpose

This script was designed to handle WFDB-format data (`.hea + .dat + annotations`) and convert annotation-based sleep stages into structured labels.

### ⚙️ What it does

* Reads raw EEG signals using WFDB
* Extracts sleep stages from annotation files (`.st`, `.slp`, etc.)
* Processes epochs of 30 seconds
* Computes statistical and spectral features

### 📤 Output

* `sleep_features_b.csv`

---

## 3. ST. Vincent Dataset (UCDDB)

### 📌 Purpose

This script was developed to process clinical sleep data that includes both sleep stages and respiratory events (apnea).

### ⚙️ What it does

* Converts `.rec` files into readable EEG format
* Extracts sleep stages from `_stage` files
* Detects apnea events from `_respevt` files
* Generates features per epoch
* Adds apnea classification (binary)

### 📤 Output

* `st_vincent_sleep_features.csv`

### 🔥 Extra Feature

Includes:

* `apnea_label` → 0 (no apnea), 1 (apnea present)

---

## 4. ISRUC-SLEEP Dataset

### 📌 Purpose

This script was created to process ISRUC dataset recordings using a single EEG channel and scorer annotations.

### ⚙️ What it does

* Reads `.rec` EEG files
* Uses only primary scorer annotations (`1_1.txt`)
* Segments into epochs
* Extracts EEG features
* Maps sleep stages into standardized labels

### 📤 Output

* `isruc_sleep_features.csv`

---

# 🧠 Feature Extraction

All datasets are processed using the same feature extraction pipeline:

### 📈 Time-domain features

* Skewness
* Kurtosis
* Peak amplitude

### 🌊 Frequency-domain features

* Delta (0.5–4 Hz)
* Theta (4–8 Hz)
* Alpha (8–13 Hz)
* Beta (13–30 Hz)
* Log-scaled band power

### 🔄 Ratios

* Theta / Alpha
* Delta / Beta
* Band power ratios

### 🧠 Advanced features

* Spectral entropy
* Hjorth parameters:

  * Activity
  * Mobility
  * Complexity

---

# 🏷️ Label Standardization

All datasets are unified into the same labeling system:

| Label | Stage |
| ----- | ----- |
| 0     | Wake  |
| 1     | N1    |
| 2     | N2    |
| 3     | N3    |
| 4     | REM   |

👉 Note: Stage 4 is merged into N3 (modern sleep scoring standard)

---

# 📁 Project Structure

```
project/
│
├── build_sleep_edf_features.py      # Sleep-EDF processor
├── build_mitbih_features.py        # MIT-BIH processor
├── build_st_vincent_features.py    # ST Vincent + apnea processor
├── build_isruc_features.py         # ISRUC processor
│
├── output/
│   ├── sleep_features.csv
│   ├── sleep_features_b.csv
│   ├── st_vincent_sleep_features.csv
│   └── isruc_sleep_features.csv
│
├── app.py                          # Streamlit app
├── requirements.txt                # Dependencies
└── README.md
```

---

# ⚙️ Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install mne numpy pandas scipy wfdb
```

---

# ▶️ Usage

Run any dataset processor:

```bash
py -3.12 build_sleep_edf_features.py
py -3.12 build_mitbih_features.py
py -3.12 build_st_vincent_features.py
py -3.12 build_isruc_features.py
```

---

# 📊 Output Format

Each dataset is exported as a CSV file:

```
subject_id | epoch | features... | sleep_stage
```

---

# 🚀 Applications

This project enables:

* Sleep stage classification models
* EEG-based Machine Learning pipelines
* Apnea detection systems
* Data preprocessing for AI research

---

# 👨‍💻 Author
Orlando Figueroa 

---

# 📄 License

For academic and research purposes only.
