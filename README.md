# Predictive Maintenance AI — GUI

A Streamlit web application for predictive maintenance using machine sensor readings.

## 🚀 Live Demo

**[Open the Deployed Application](YOUR_STREAMLIT_APP_URL)**

> Replace `YOUR_STREAMLIT_APP_URL` with the URL of the deployed Streamlit application.

## 📌 Overview

This project provides a user-friendly web interface for a predictive maintenance machine-learning system.

The application takes machine sensor readings as input and provides three predictions:

* **Failure Risk Within 24 Hours** — Classification
* **Most Likely Failure Type** — Classification
* **Estimated Repair Cost** — Regression

The GUI is built using Streamlit and uses the machine-learning pipeline developed in `final_day_project.ipynb`.

> **Note:** Remaining Useful Life (RUL) prediction is not included because an RUL model was not trained in the original notebook.

---

## 📂 Project Structure

```text
pm_project/
│
├── app/
│   └── app.py
│
├── data/
│   └── predictive_maintenance_v3.csv
│
├── models/
│   ├── scaler.joblib
│   ├── operating_mode_encoder.joblib
│   ├── failure_type_label_encoder.joblib
│   ├── model_failure_within_24h.joblib
│   ├── model_failure_type.joblib
│   ├── model_estimated_repair_cost.joblib
│   └── config.json
│
├── preprocessing/
│   └── train_and_save.py
│
├── utils/
│   └── predict.py
│
├── final_day_project.ipynb
│
├── requirements.txt
│
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd pm_project
```

### 2. Install the required dependencies

```bash
py -m pip install -r requirements.txt
```

---

## ▶️ Run the Application Locally

From the project root directory, run:

```bash
py -m streamlit run app/app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open the URL in your browser to use the application.

---

## 🤖 Machine Learning Models

The project uses **XGBoost** models for the three prediction tasks:

### 1. Failure Risk Within 24 Hours

A classification model that predicts whether the machine is likely to experience a failure within the next 24 hours.

### 2. Most Likely Failure Type

A classification model that predicts the most likely type of machine failure.

### 3. Estimated Repair Cost

A regression model that estimates the expected repair cost based on the machine's sensor readings and operating conditions.

---

## 🔄 Data Preprocessing

The machine-learning pipeline includes:

* Missing-value imputation
* Outlier handling during training
* Categorical feature encoding
* Numerical feature scaling
* Feature preparation for model prediction

The main categorical features include:

* `machine_type`
* `operating_mode`

Sensor measurements are processed before being passed to the trained models.

---

## 🧠 Model Pipeline

The application uses persisted model artifacts stored in the `models/` directory.

These files include:

* Trained XGBoost models
* Feature scaler
* Categorical encoders
* Model configuration
* Feature information and prediction settings

This allows the Streamlit application to load the trained models directly without retraining them every time the application starts.

---

## 🏗️ Model Training

The original machine-learning pipeline was developed in:

```text
final_day_project.ipynb
```

The `preprocessing/train_and_save.py` script can be used to reproduce the preprocessing and training pipeline and save the trained model artifacts.

Run:

```bash
py preprocessing/train_and_save.py
```

This step is only required if the models need to be regenerated or the training data/pipeline is changed.

---

## ✅ Pipeline Validation

The project also includes a validation script:

```bash
py preprocessing/validate.py
```

This can be used to verify that the persisted prediction pipeline produces results consistent with the original notebook pipeline.

---

## 📊 Dataset

The project uses:

```text
predictive_maintenance_v3.csv
```

The dataset contains machine sensor readings and maintenance-related information used to train the predictive maintenance models.

---

## 🛠️ Technologies

* **Python**
* **Streamlit**
* **Scikit-learn**
* **XGBoost**
* **Pandas**
* **Joblib**
* **Jupyter Notebook**

---

## 🚀 Deployment

The application can be deployed as a Streamlit web application.

After deployment, the live application can be accessed through the **Live Demo** link at the top of this README.

---

## ⚠️ RUL Prediction

Remaining Useful Life (**RUL**) prediction is currently not included in this deployment.

The original notebook contained `rul_hours` as a dataset column, but it was not used as a trained regression target. Therefore, no RUL prediction model was included in this version of the application.

---

## 👥 Project

**Predictive Maintenance AI**

Machine Learning Project — NTI

The project demonstrates the use of machine-learning models to support predictive maintenance by analyzing machine sensor data and estimating potential failures and maintenance costs.
