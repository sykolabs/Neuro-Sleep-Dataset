# 🧠 NeuroSleep Data Processing Pipeline

This project was developed to convert raw polysomnography (PSG) sleep data into labeled, machine learning-ready datasets using a single EEG channel.

The main objective is to standardize multiple public sleep datasets into a unified format for:

* Sleep stage classification
* EEG signal analysis
* Machine Learning applications
* Apnea detection (where available)

---

# 🎯 Purpose of the Project

Each script in this repository was created to solve the same problem across different datasets:

👉 Raw sleep data is complex and not directly usable for Machine Learning.

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

Processes EEG recordings and hypnogram annotations into labeled datasets.

### ⚙️ What it does

* Reads `.edf` EEG signals
* Loads hypnogram annotations
* Segments signals into 30-second epochs
* Extracts EEG features
* Maps sleep stages into standardized labels

### 📤 Output

`sleep_features.csv`

---

## 2. MIT-BIH PSG Dataset

### 📌 Purpose

Handles WFDB-format sleep data and converts annotation-based sleep stages into structured labels.

### ⚙️ What it does

* Reads EEG signals using WFDB
* Extracts stages from `.st`, `.slp`, `.atr`, etc.
* Processes 30-second epochs
* Computes statistical and spectral features

### 📤 Output

`sleep_features_b.csv`

---

## 3. ST. Vincent Dataset (UCDDB)

### 📌 Purpose

Processes clinical sleep data including respiratory events (apnea).

### ⚙️ What it does

* Converts `.rec` → EEG-readable format
* Extracts stages from `_stage` files
* Detects apnea from `_respevt` files
* Generates epoch-based features
* Adds apnea classification

### 📤 Output

`st_vincent_sleep_features.csv`

### 🔥 Extra Feature

* `apnea_label` → 0 (no apnea), 1 (apnea present)

---

## 4. ISRUC-SLEEP Dataset

### 📌 Purpose

Processes ISRUC EEG recordings using a single channel and scorer annotations.

### ⚙️ What it does

* Reads `.rec` EEG files
* Uses scorer 1 annotations (`1_1.txt`)
* Segments into 30-second epochs
* Extracts EEG features
* Maps sleep stages into standardized labels

### 📤 Output

`isruc_sleep_features.csv`

---

# 🧠 Feature Extraction

All datasets share the same feature extraction pipeline:

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

All datasets are unified into:

| Label | Stage |
| ----- | ----- |
| 0     | Wake  |
| 1     | N1    |
| 2     | N2    |
| 3     | N3    |
| 4     | REM   |

👉 Stage 4 is merged into N3 (modern AASM standard)

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
├── data_sleep_edf/                 # Sleep-EDF dataset
├── data_mitbih/                   # MIT-BIH dataset
├── data_st_vincent/               # ST Vincent dataset
├── data_isruc/                    # ISRUC dataset
│
├── output/                        # Sleep-EDF + MIT outputs
├── output_st_vincent/             # ST Vincent outputs
├── output_isruc/                  # ISRUC outputs
│
├── app.py                         # Streamlit app
├── requirements.txt
└── README.md
```

---

# 📂 Data Setup

Each dataset must be placed in its corresponding folder:

**Sleep-EDF**

```
data_sleep_edf/
→ .edf PSG + Hypnogram files
```

**MIT-BIH PSG**

```
data_mitbih/
→ .hea + .dat + annotation files
```

**ST Vincent (UCDDB)**

```
data_st_vincent/
→ .rec + _stage + _respevt files
```

**ISRUC-SLEEP**

```
data_isruc/
→ subject folders containing:
   - .rec file
   - 1_1.txt (labels)
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

# 🚀 Portable Design

All scripts use relative paths, allowing the project to run on any machine without modifying file paths.

Simply place the datasets in the correct folders and execute the scripts.

---

# 📊 Output Format

Each dataset is exported as:

```
subject_id | epoch | features... | sleep_stage
```

---

# 🚀 Applications

* Sleep stage classification
* EEG-based Machine Learning pipelines
* Apnea detection systems
* Data preprocessing for AI research

---

# 👨‍💻 Author

Orlando Figueroa

---

# 📄 License

For academic and research purposes only.

