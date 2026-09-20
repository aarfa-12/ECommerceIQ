# E-CommerceIQ — E-Commerce Customer Analytics & Purchase Prediction System


---

## Project Overview

**E-CommerceIQ** is an end-to-end data analytics and machine learning project that analyses
Indian e-commerce customer behaviour, predicts purchase completion using strictly **leak-free**
pre-session features, segments customers into behavioural groups, and delivers product
recommendations — all through an interactive Streamlit dashboard.

---

## Problem Statement

E-commerce businesses may benefit from identifying sessions with different levels of purchase
intent, enabling more targeted personalisation, retargeting, and promotional strategies.

---

## Objectives

1. Analyse customer behavioural patterns across 25,000 sessions from an Indian e-commerce platform.
2. Build a leak-free binary classifier to predict purchase completion from pre-purchase session signals.
3. Compare Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting using correct
   CV-based model selection (no test-set leakage).
4. Perform customer-level K-Means segmentation (8,442 customers) using aggregated behavioural profiles.
5. Provide popularity-based product recommendations per category.
6. Deploy all findings in an interactive Streamlit dashboard.

---

## Dataset

| Attribute | Value |
|-----------|-------|
| Name | Indian E-Commerce Customer Behavior & Purchase |
| Source | [Kaggle](https://www.kaggle.com/datasets/kundanbedmutha/indian-e-commerce-customer-behavior-and-purchase) |
| Rows | 25,000 |
| Columns | 29 |
| Unique Customers | 8,442 |
| Target | `purchased` (0 = not purchased, 1 = purchased) |
| Class ratio | ~77.5 % not purchased / ~22.5 % purchased |

### Dataset Placement

Place `Ecommerce.csv` inside the `data/` folder:

```
ECommerceIQ/
└── data/
    └── Ecommerce.csv
```

---

## Data Leakage Audit

The following columns are **excluded** from prediction features (post-purchase, target-derived,
or ambiguous):

| Column | Reason |
|--------|--------|
| `added_to_cart` | `purchased=1` ↔ `added_to_cart=1` in 100 % of cases — perfect predictor |
| `cart_abandoned` | Logically = `added_to_cart=1 AND purchased=0` — derived from target |
| `rating` | All non-purchasers have `rating=4` exactly — post-purchase placeholder |
| `review_text` | Constant for all non-purchasers — post-purchase |
| `review_helpful_votes` | 0 for every non-purchaser — post-purchase |
| `revenue` | 0 for every non-purchaser — direct target signal |
| `revenue_normalized` | Scaled revenue — same direct leak |
| `discount_amount` | Exact derivation of `unit_price × qty × discount_pct` (redundant) |
| `payment_method` | Ambiguous checkout/payment-stage field; present even for sessions with no cart add; excluded (precautionary — low correlation does not establish pre-purchase availability) |

**Retained prediction features (15):**
`device_type`, `user_type`, `marketing_channel`, `product_category`, `unit_price`,
`quantity`, `discount_percent`, `pages_viewed`, `time_on_site_sec`, `visit_day`,
`visit_month`, `visit_weekday`, `visit_season`, `location`, `session_duration_bucket`

> Including post-purchase or target-derived features can substantially inflate apparent
> predictive performance and therefore introduce data leakage.

---

## Technologies

- **Python 3.10+**
- **Streamlit** — interactive dashboard
- **scikit-learn** — ML models, preprocessing, CV, tuning
- **pandas / NumPy** — data manipulation
- **Plotly** — interactive charts
- **Matplotlib / Seaborn** — notebook charts
- **joblib** — model persistence

---

## ML Techniques

| Step | Method |
|------|--------|
| Preprocessing | `ColumnTransformer` → `StandardScaler` (numeric) + `OrdinalEncoder` (categorical session-duration bucket) |
| Models | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting |
| Selection | 5-fold stratified CV ROC-AUC on **training set only** |
| Tuning | `RandomizedSearchCV` (n_iter=20, 5-fold CV, training set only) |
| Final evaluation | **Single evaluation** on held-out 20 % test set |
| Segmentation | K-Means on customer-level aggregates (Elbow + Silhouette, K=2–8) |
| Recommendations | Popularity-based: purchase count (50 %) + avg rating (30 %) + session frequency (20 %) |

---

## Application Features

| Page | Description |
|------|-------------|
| 📊 Dashboard | KPIs, monthly trend, category/device/channel/season breakdown, auto-generated key insights |
| 👤 Customer Analytics | Purchase funnel, session behaviour, weekday patterns, buyer vs non-buyer profile |
| 🔍 Exploratory Data Analysis | Distributions, correlation heatmap, missing values, statistical summary |
| 🎯 Purchase Prediction | Interactive 15-feature form → prediction + probability gauge |
| 🗂️ Customer Segmentation | Customer-level K-Means, PCA 2D visualisation, cluster profiles, radar chart |
| 🛍️ Product Recommendations | Popularity-based recommendations per category with score breakdown |
| 📈 Model Performance | Metrics table, ROC curves, confusion matrices, feature importance, best hyperparameters |
| ℹ️ About Project | Methodology, full leakage audit table, technology stack |

---

## Folder Structure

```
ECommerceIQ/
├── AarfaFatima_ECommerceIQ.py      ← Streamlit application
├── AarfaFatima_ECommerceIQ.ipynb   ← Jupyter notebook
├── requirements.txt
├── README.md

```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## How to Run

### Streamlit Application

```bash
streamlit run AarfaFatima_ECommerceIQ.py
```

### Jupyter Notebook

```bash
jupyter notebook AarfaFatima_ECommerceIQ.ipynb
```

Run all cells top-to-bottom after installing the dependencies listed in `requirements.txt`.

---

## Key Results

The dashboard computes model metrics and analytical results from the dataset at runtime.
The values below are representative results from the project run.

- **Overall purchase rate:** ~22.4 % of sessions
- **Final model:** selected by CV ROC-AUC on training set; evaluated once on test set
- **Test ROC-AUC:** approximately 0.55–0.57 (depending on the final selected model and run)
- **Model selection methodology:** CV on training set only — no test-set peeking
- **Customer clusters:** optimal K determined by Silhouette Score

> **Note on AUC:** After removing post-purchase and ambiguous fields, the model relies only
> on retained pre-purchase session signals, which provide relatively limited predictive
> information in this dataset. The AUC (~0.55–0.57) reflects the predictive signal available
> from those retained features.

---

## Limitations

- Column semantic meaning is inferred from data patterns; no external data dictionary is available.
- The AUC reflects the available predictive signal in the retained pre-purchase features for this dataset.
- Recommendations are popularity-based; collaborative filtering would require explicit user–item history.
- Customer segments are based on behavioural aggregates; demographic data is not available.

---

## Future Scope

- Integrate richer session signals (scroll depth, hover events, referral path).
- Explore neural network–based tabular models (e.g., TabNet, SAINT).
- Build a collaborative-filtering recommendation engine with proper user–item matrices.
- Add SHAP-based local feature explanations per prediction.
- Deploy on cloud infrastructure with a CI/CD pipeline.

---

## Author

**Aarfa Fatima**  

