# 🌱 Crop Recommendation System using Machine Learning

An AI-powered crop recommendation system that predicts the most suitable crop based on soil nutrients and environmental conditions.

This project compares **Random Forest** and **Artificial Neural Network (ANN)** models and deploys the best-performing model through an interactive **Streamlit web application**.

---

## 📌 Project Overview

Agriculture productivity highly depends on selecting the right crop for specific soil and climate conditions.

This application helps users predict the best crop by analyzing:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- pH Value
- Rainfall

The system recommends the most suitable crop and shows prediction confidence.

---

## 🚀 Features

✅ Crop Prediction using Machine Learning 

✅ ANN vs Random Forest model comparison  

✅ Real time prediction through Streamlit UI 

✅ Prediction confidence score  

✅ Clean and responsive interface  

✅ Preprocessing using scaler and label encoder 
  
✅ Model persistence with pickle files  

---

## 🧠 Machine Learning Models Used

### 1. Random Forest Classifier
- Best performing model
- Accuracy: **99.31%**

### 2. Artificial Neural Network (ANN)
- Deep learning comparison model
- Accuracy: **97.95%**

---

## 📊 Model Accuracy Comparison

| Model | Accuracy |
|----|----|
| Random Forest | 99.31% |
| ANN | 97.95% |


---

## 📂 Project Structure

```bash
Crop-Recommendation-System/
│
├── app.py
├── best_model.pkl
├── scaler.pkl
├── label_encoder.pkl
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/
│   └── Crop_recommendation.csv
│
├── notebooks/
│   └── Crop_Recommendation_System_Model.ipynb
│
├── images/
│   ├── UI.png
│   └── Accuracy_comparison_ANN_vs_Random_Forest.png
│
└── venv/
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/nandi-boy/Crop-Recommendation-System.git
cd Crop-Recommendation-System
```

Create virtual environment:

```bash
python -m venv venv
```

Activate virtual environment:

### Windows
```bash
venv\Scripts\activate
```

### Linux/Mac
```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python -m streamlit run app.py
```

---

## 📦 Required Libraries

- streamlit
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- pickle

---

## 🧪 Input Parameters

Users provide:

- Nitrogen
- Phosphorus
- Potassium
- Temperature
- Humidity
- pH
- Rainfall

Output:
- Recommended Crop
- Prediction Confidence

---

## 📚 Dataset

Dataset contains agricultural and soil parameters such as:

- N
- P
- K
- temperature
- humidity
- ph
- rainfall
- label



---

## 🛠️ Tech Stack

- Python
- Machine Learning
- Scikit-learn
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Google Colab
- VS Code

---

## 👨‍💻 Author

**Ayan Nandi**

GitHub: https://github.com/nandi-boy  
LinkedIn: https://www.linkedin.com/in/ayan-nandi/

---
