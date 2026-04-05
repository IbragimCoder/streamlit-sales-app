# 📈 Sales Analytics & Prediction Dashboard

An end-to-end machine learning web application for retail sales analytics and revenue prediction, built with Streamlit and Scikit-learn.

## 🎯 Overview

This project combines a **Random Forest regression model** with an interactive **Streamlit dashboard** to help analyze retail sales data and predict sale amounts based on store, product, and customer inputs.

---

## ✨ Features

- **🤖 ML-based Sales Prediction** — predict revenue for any store/product/customer combination in real time
- **📊 Business KPI Dashboard** — total revenue, items sold, unique customers at a glance
- **🏆 Sales Leaders** — top-5 products and stores ranked by revenue with bar charts
- **🔍 In-depth Product Analysis** — drill down into any product's sales performance across stores
- **⚡ Optimized Performance** — cached data loading and model inference via `@st.cache_data` / `@st.cache_resource`

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| ML Model | `RandomForestRegressor` (Scikit-learn) |
| Preprocessing | `ColumnTransformer`, `StandardScaler`, `OneHotEncoder` |
| Pipeline | `sklearn.pipeline.Pipeline` |
| Web App | `Streamlit` |
| Data Processing | `Pandas` |
| Model Serialization | `Joblib` |

---

## 🗂️ Project Structure

```
sales-analytics/
├── app.py                    # Streamlit dashboard & prediction UI
├── train_model.py            # Model training & pipeline serialization
├── regressor_pipeline.joblib # Saved trained pipeline
├── last_satis.csv            # Sales dataset
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/IbragimCoder/sales-analytics.git
cd sales-analytics
```

### 2. Install dependencies

```bash
pip install streamlit pandas scikit-learn joblib
```

### 3. Train the model

```bash
python train_model.py
```

This will preprocess the data and save the trained pipeline as `regressor_pipeline.joblib`.

### 4. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🧠 Model Details

The ML pipeline consists of two stages:

**Preprocessing (ColumnTransformer):**
- Numerical features → `StandardScaler`
- Categorical features → `OneHotEncoder` (with `handle_unknown='ignore'` for robustness)

**Model:**
- `RandomForestRegressor` with 100 estimators, trained with `n_jobs=-1` for parallel computation

**Target variable:** `Ümumi satış` (Total sale amount)

**Input features:** Store (`Mağaza`), Card number (`Kart_nomresi`), Product ID (`Məhsul_nomresi`), Product name (`Məhsul_adi`), Quantity (`Məhsul sayi`)

---

## 📸 App Preview

| Section | Description |
|---|---|
| **Prediction Panel** | Select store & product, enter quantity → get predicted revenue |
| **KPI Metrics** | Total revenue, items sold, unique customers |
| **Bar Charts** | Top-5 products and stores by revenue |
| **Product Drilldown** | Filter full dataset by product with store-level breakdown |

---

## 📄 License

MIT License — feel free to use and modify.
