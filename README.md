# 🌱 Krishi Sahayak — AI-Powered Crop Advisory System

A multi-modal agricultural advisory platform that helps a farmer decide **what to grow** and **how to grow it** — combining a trained machine learning model, real historical government crop-production data, and a multilingual crop-care reference guide, all through an interactive Streamlit app.

This isn't a single-model demo. It's three distinct systems, each answering the same "what should I grow" question with a different, deliberately-chosen mechanism — and each one exists because the previous approach was tested, found lacking for a specific real reason, and replaced rather than patched over.

---

## 📌 Why Three Modes, Not One

| Mode | Mechanism | Best for |
|---|---|---|
| 🌾 **Parameter-Based** | Trained ML model (Random Forest) on soil + climate | When you have real soil test values (N, P, K, pH) and climate readings |
| 🌍 **State-Based** | Real historical government crop-production statistics, ranked by district + season | When you don't have soil data, but want to know what's actually, historically been grown in your district |
| 🌱 **Crop Care Guide** | Dataset-derived N/P/K/temperature requirements + fertilizer guidance + farming process, in 12 Indian languages | Once you've picked a crop, and need to know how to actually grow it |


---

## 🚀 Features

### 🌾 Parameter-Based Mode
- Enter N, P, K, temperature, humidity, pH, and rainfall directly
- Random Forest model (99.31% test accuracy) predicts the best-fit crop
- Shows prediction confidence

### 🌍 State-Based Mode
- Select Country → State → District → Season (Kharif/Rabi)
- Ranks the top 3 crops by their real historical share of recorded production/area for that district and season, sourced from government crop statistics 
- Does **not** use the ML model — this is a statistics lookup against real records, not a prediction

### 🌱 Crop Care Guide
- Pick any of the 22 crops in the dataset
- See the crop's real average required N, P, K, and temperature (computed live from the same training data the ML model uses — not a separate invented table)
- Get derived fertilizer guidance (Urea/DAP/MOP) based on those N/P/K levels
- See a general step-by-step farming process (sowing, spacing, irrigation, key pest to watch, harvest signal)
- Available in **English + 12 major Indian languages** (Hindi, Bengali, Telugu, Marathi, Tamil, Gujarati, Kannada, Malayalam, Punjabi, Odia, Urdu, Assamese) via machine translation

---

## 🧠 Machine Learning Model (Parameter-Based Mode)

| Model | Accuracy |
|---|---|
| **Random Forest** (deployed) | 99.31% |
| ANN (comparison) | 97.95% |

Trained on a 2,200-row dataset (22 crops, 100 samples each, no class imbalance) with 7 features: N, P, K, temperature, humidity, pH, rainfall.

---

## 🛠️ Tech Stack

- **Python**, **Streamlit** — app framework
- **scikit-learn** — Random Forest model, preprocessing
- **pandas**, **NumPy** — data handling
- **deep-translator** — machine translation (Google Translate, no API key required)
- **Matplotlib**, **Seaborn** — model comparison visualization (in notebook)

---

## 📂 Project Structure

```
Krishi-Sahayak/
│
├── app.py                              # Main Streamlit app (all 3 modes)
├── crop_care.py                        # Crop Care Guide: N/P/K/temp lookup, fertilizer guidance, farming steps
├── crop_history.py                     # State-Based mode: historical crop-production lookup
├── locations.py                        # Country/State data
├── translation.py                      # Multilingual support for Crop Care Guide
│
├── best_model.pkl                      # Trained Random Forest model
├── scaler.pkl                          # Feature scaler
├── label_encoder.pkl                   # Crop label encoder
│
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/
│   ├── Crop_recommendation.csv         # ML training data (2,200 rows, 22 crops)
│   └── district_season_crop_production.csv   # Historical crop stats (download separately — see Setup)
│
├── notebooks/
│   └── Crop_Recommendation_System_Model.ipynb   # Model training, comparison, evaluation
│
│
└── venv/
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/nandi-boy/Krishi-Sahayak.git
cd Krishi-Sahayak
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the historical crop-production dataset (required for State-Based mode)

State-Based mode needs a real government dataset that isn't bundled in this repo (it's too large / better fetched fresh):

- Download **"Crop Production in India"** from Kaggle: https://www.kaggle.com/datasets/abhinand05/crop-production-in-india
  *(This mirrors the official data.gov.in "District-wise, season-wise crop production statistics" dataset, 1997–2015.)*
- Rename the downloaded CSV to `district_season_crop_production.csv`
- Place it in the `dataset/` folder

If this file isn't present, State-Based mode will show a clear error rather than fail silently — Parameter-Based and Crop Care Guide modes work without it.

### 5. Run the app
```bash
python -m streamlit run app.py
```

---

## 📚 Data Sources

| Data | Source |
|---|---|
| Soil/climate → crop training data | Crop Recommendation Dataset (Kaggle) |
| Historical crop production by district/season | data.gov.in, mirrored via Kaggle ("Crop Production in India" by abhinand05) |
| Crop care N/P/K/temperature | Derived from the same training dataset above (real averages, not invented) |
| Fertilizer guidance | General N/P/K → common-fertilizer (Urea/DAP/MOP) mapping |
| Farming process steps | General, widely-practiced agronomic steps |
| Translations | Google Translate, via the `deep-translator` library |

---

## ⚠️ Known Limitations & Lessons Learned

Being upfront about these — they're arguably the most interesting part of this project:

- **Data leakage in the training notebook.** The feature scaler is fit on the *entire* dataset before the train/test split, rather than fit only on the training set. Separately, the ANN's early-stopping watches validation loss on the *test set itself*, meaning the reported ANN accuracy reflects a model partly selected using test data. The Random Forest's 99.31% figure is not affected by the second issue, but both are worth fixing before fully trusting either number. *(Not yet fixed — noted here rather than silently left out.)*
- **State-Based mode's original design failed for a specific, diagnosable reason.** Averaging soil nutrients to a single dataset-wide constant for every location destroyed roughly half the model's discriminative power (soil and climate contribute close to equally to the model's decisions), causing real rice-growing districts to never predict rice. This is why State-Based mode now uses real historical statistics instead of the ML model.
- **Crop Care Guide's fertilizer/process guidance is general reference information**, not soil-test-based or region-specific. A real soil test from a local Krishi Vigyan Kendra (KVK) will always be more accurate.
- **Translations are unverified machine translation** (Google Translate via an unofficial wrapper). Quality is expected to be strong for major languages, but hasn't been checked against a fluent speaker for every language/crop combination.
- **No pesticide/chemical-dosage guidance is included**, deliberately — incorrect chemical recommendations carry real safety risk, so this was scoped out rather than guessed at.

---

## 🗺️ Roadmap

- [ ] Fix the two data leakage issues above; add k-fold cross-validation and a full classification report
- [ ] Explainability: surface feature importance per prediction in Parameter-Based mode
- [ ] Crop disease detection (image-based)
- [ ] Pesticide/pest guidance (general IPM guidance first, chemical-specific later, done carefully)
- [ ] Deploy publicly (Streamlit Community Cloud)
- [ ] Automated test suite

---

## 👨‍💻 Author

**Ayan Nandi**

- GitHub: https://github.com/nandi-boy
- LinkedIn: https://www.linkedin.com/in/ayan-nandi/
