"""
E-CommerceIQ: E-Commerce Customer Analytics & Purchase Prediction System
Author : Aarfa Fatima
Program: IBM SkillsBuild Data Analytics with AI Academic Internship (2026)

════════════════════════════════════════════════════════════════════════════════
DATA LEAKAGE AUDIT  (all decisions documented)
════════════════════════════════════════════════════════════════════════════════

EXCLUDED – post-purchase / derived-from-target:
  added_to_cart        purchased=1 ↔ added_to_cart=1 (100% of cases) → perfect predictor
  cart_abandoned       logically = added_to_cart=1 AND purchased=0 → derived from target
  rating               all non-purchasers have exactly rating=4 → post-purchase placeholder
  review_text          constant value (1) for all non-purchasers → post-purchase
  review_helpful_votes 0 for every non-purchaser → post-purchase
  revenue              0 for every non-purchaser → direct target signal
  revenue_normalized   scaled revenue → same direct leak
  discount_amount      exact derivation: unit_price × quantity × discount_percent/100

EXCLUDED – ambiguous checkout/payment-stage field:
  payment_method       Present for sessions where added_to_cart=0 (7,354 sessions),
                       suggesting it may be a server-recorded checkout attribute.
                       Distribution is statistically independent of purchase outcome
                       (χ²=6.44, p=0.27), but because the data-generation logic is
                       not externally documented we apply the precautionary principle
                       and exclude it.  A low correlation does not establish pre-purchase
                       availability; we require positive evidence of pre-purchase
                       availability, which is absent here.

KEPT – genuine pre-purchase session signals (15 features):
  device_type, user_type, marketing_channel, product_category,
  unit_price, quantity, discount_percent,
  pages_viewed, time_on_site_sec,
  visit_day, visit_month, visit_weekday, visit_season, location,
  session_duration_bucket  (ordinal-encoded via OrdinalEncoder inside Pipeline)

════════════════════════════════════════════════════════════════════════════════
MODEL SELECTION METHODOLOGY
════════════════════════════════════════════════════════════════════════════════
1. 80 / 20 stratified train / test split (test set never touched during selection).
2. 5-fold stratified CV on training set → select best base model by mean CV ROC-AUC.
3. RandomizedSearchCV (n_iter=20, 5-fold CV) on training set → tune best base model.
4. If tuned CV AUC ≥ base CV AUC, the tuned pipeline is the final model; otherwise
   the base pipeline (already fitted on full training set) is the final model.
5. The selected final model is evaluated ONCE on the held-out test set (reported only,
   never fed back to influence model or hyperparameter selection).
════════════════════════════════════════════════════════════════════════════════
"""

import warnings
warnings.filterwarnings("ignore")

import pathlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import streamlit as st

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROOT       = pathlib.Path(__file__).parent
DATA_DIR   = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUT_DIR    = ROOT / "outputs" / "figures"
for _d in [DATA_DIR, MODELS_DIR, OUT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN MAPPINGS  (verified against actual dataset integer codes)
# ─────────────────────────────────────────────────────────────────────────────
DEVICE_MAP      = {0: "Desktop", 1: "Mobile",      2: "Tablet"}
USER_TYPE_MAP   = {0: "New User", 1: "Returning User"}
MARKETING_MAP   = {0: "Direct",  1: "Email",       2: "Social Media",
                   3: "Affiliate", 4: "Paid Search", 5: "SEO"}
PRODUCT_CAT_MAP = {0: "Electronics", 1: "Clothing",  2: "Home & Kitchen",
                   3: "Sports",      4: "Beauty",     5: "Books",
                   6: "Toys",        7: "Other"}
SEASON_MAP      = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Autumn"}
WEEKDAY_MAP     = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
DURATION_ORDER  = ["Very Short", "Short", "Long", "Very Long"]

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
NUM_FEATURES = [
    "device_type", "user_type", "marketing_channel", "product_category",
    "unit_price", "quantity", "discount_percent",
    "pages_viewed", "time_on_site_sec",
    "visit_day", "visit_month", "visit_weekday", "visit_season", "location",
]
CAT_FEATURES = ["session_duration_bucket"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES  # 15 total
TARGET       = "purchased"

# Single shared preprocessor (used inside every Pipeline)
_PREPROCESSOR = ColumnTransformer(transformers=[
    ("num", StandardScaler(), NUM_FEATURES),
    ("cat", OrdinalEncoder(
        categories=[DURATION_ORDER],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    ), CAT_FEATURES),
], remainder="drop")


def _make_preprocessor():
    """Return a fresh (unfitted) ColumnTransformer instance."""
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OrdinalEncoder(
            categories=[DURATION_ORDER],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        ), CAT_FEATURES),
    ], remainder="drop")


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    csv_path = DATA_DIR / "Ecommerce.csv"
    if not csv_path.exists():
        st.error(
            f"Dataset not found at `{csv_path}`. "
            "Place `Ecommerce.csv` inside the `data/` folder and restart."
        )
        st.stop()
    df = pd.read_csv(csv_path)
    df["visit_date"]      = pd.to_datetime(df["visit_date"], dayfirst=True, errors="coerce")
    df["device_label"]    = df["device_type"].map(DEVICE_MAP)
    df["user_type_label"] = df["user_type"].map(USER_TYPE_MAP)
    df["marketing_label"] = df["marketing_channel"].map(MARKETING_MAP)
    df["category_label"]  = df["product_category"].map(PRODUCT_CAT_MAP)
    df["season_label"]    = df["visit_season"].map(SEASON_MAP)
    df["weekday_label"]   = df["visit_weekday"].map(WEEKDAY_MAP)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING  (correct CV-based selection; test set used only for reporting)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models… (cached after first run)")
def train_models(df: pd.DataFrame):
    X = df[ALL_FEATURES].copy()
    y = df[TARGET]

    # ── 1. Train / test split ──────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ── 2. Base model CV on training set ─────────────────────────────────
    base_defs = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=10, random_state=RANDOM_STATE,
            class_weight="balanced", n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=5,
            random_state=RANDOM_STATE,
        ),
    }

    base_cv_results: dict[str, dict] = {}
    for name, clf in base_defs.items():
        pipe = Pipeline([("prep", _make_preprocessor()), ("clf", clf)])
        scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
        base_cv_results[name] = {"mean": scores.mean(), "std": scores.std()}

    # ── 3. Select best base model by CV AUC (training set only) ──────────
    best_base_name = max(base_cv_results, key=lambda k: base_cv_results[k]["mean"])

    # ── 4. Tune best base model with RandomizedSearchCV on training set ──
    param_grids: dict[str, dict] = {
        "Random Forest": {
            "clf__n_estimators":      [100, 200, 300],
            "clf__max_depth":         [8, 10, 12, None],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf":  [1, 2, 4],
            "clf__max_features":      ["sqrt", "log2"],
        },
        "Gradient Boosting": {
            "clf__n_estimators":      [100, 200, 300],
            "clf__learning_rate":     [0.05, 0.1, 0.2],
            "clf__max_depth":         [3, 5, 7],
            "clf__min_samples_leaf":  [1, 2, 4],
            "clf__subsample":         [0.7, 0.85, 1.0],
        },
        "Logistic Regression": {
            "clf__C":       [0.01, 0.1, 1.0, 10.0],
            "clf__solver":  ["lbfgs", "saga"],
            "clf__penalty": ["l2"],
        },
        "Decision Tree": {
            "clf__max_depth":         [5, 8, 12, None],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf":  [1, 2, 4],
        },
    }

    tuned_pipe = Pipeline([("prep", _make_preprocessor()), ("clf", base_defs[best_base_name])])
    search = RandomizedSearchCV(
        tuned_pipe,
        param_distributions=param_grids[best_base_name],
        n_iter=20,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,          # refit on full training set with best params
    )
    search.fit(X_tr, y_tr)

    tuned_cv_auc  = search.best_score_
    base_cv_auc   = base_cv_results[best_base_name]["mean"]

    # ── 5. Choose final model (CV-based, no test peeking) ─────────────────
    if tuned_cv_auc >= base_cv_auc:
        final_pipeline = search.best_estimator_
        final_label    = f"Tuned {best_base_name}"
        best_params    = search.best_params_
    else:
        # Refit base model on full training set
        base_pipe = Pipeline([("prep", _make_preprocessor()), ("clf", base_defs[best_base_name])])
        base_pipe.fit(X_tr, y_tr)
        final_pipeline = base_pipe
        final_label    = best_base_name
        best_params    = {}

    # ── 6. Single evaluation on held-out test set (reporting only) ────────
    # Build ALL model results for comparison table (each fit on X_tr, evaluated on X_te)
    all_results: dict[str, dict] = {}
    for name, clf in base_defs.items():
        pipe = Pipeline([("prep", _make_preprocessor()), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        yp = pipe.predict(X_te)
        yproba = pipe.predict_proba(X_te)[:, 1]
        all_results[name] = {
            "pipeline": pipe,
            "cv_auc":   base_cv_results[name]["mean"],
            "cv_std":   base_cv_results[name]["std"],
            "accuracy": accuracy_score(y_te, yp),
            "precision":precision_score(y_te, yp, zero_division=0),
            "recall":   recall_score(y_te, yp, zero_division=0),
            "f1":       f1_score(y_te, yp, zero_division=0),
            "roc_auc":  roc_auc_score(y_te, yproba),
            "y_pred":   yp,
            "y_proba":  yproba,
            "y_te":     y_te,
            "cm":       confusion_matrix(y_te, yp),
        }

    # Add tuned model
    tuned_yp     = final_pipeline.predict(X_te)
    tuned_yproba = final_pipeline.predict_proba(X_te)[:, 1]
    all_results[final_label] = {
        "pipeline":   final_pipeline,
        "cv_auc":     tuned_cv_auc,
        "cv_std":     0.0,
        "accuracy":   accuracy_score(y_te, tuned_yp),
        "precision":  precision_score(y_te, tuned_yp, zero_division=0),
        "recall":     recall_score(y_te, tuned_yp, zero_division=0),
        "f1":         f1_score(y_te, tuned_yp, zero_division=0),
        "roc_auc":    roc_auc_score(y_te, tuned_yproba),
        "y_pred":     tuned_yp,
        "y_proba":    tuned_yproba,
        "y_te":       y_te,
        "cm":         confusion_matrix(y_te, tuned_yp),
        "best_params": best_params,
    }

    # Persist
    joblib.dump(final_pipeline, MODELS_DIR / "best_model.pkl")
    joblib.dump({
        "name": final_label, "features": ALL_FEATURES,
        "num_features": NUM_FEATURES, "cat_features": CAT_FEATURES,
    }, MODELS_DIR / "meta.pkl")

    # Feature importances from Random Forest (tree-based; no scaling artefact)
    rf_clf = all_results["Random Forest"]["pipeline"].named_steps["clf"]
    importances = pd.Series(
        rf_clf.feature_importances_,
        index=NUM_FEATURES + ["session_duration_bucket"],
    ).sort_values(ascending=False)

    return all_results, final_label, best_params, X_tr, X_te, y_tr, y_te, importances


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOMER SEGMENTATION  (customer-level aggregation first)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Building customer profiles and clustering…")
def run_segmentation(df: pd.DataFrame):
    cust = df.groupby("customer_id").agg(
        total_sessions   = ("session_id",      "count"),
        purchase_rate    = ("purchased",        "mean"),
        total_purchases  = ("purchased",        "sum"),
        avg_pages_viewed = ("pages_viewed",     "mean"),
        avg_time_on_site = ("time_on_site_sec", "mean"),
        avg_unit_price   = ("unit_price",       "mean"),
        avg_quantity     = ("quantity",          "mean"),
        avg_discount_pct = ("discount_percent", "mean"),
        returning_user   = ("user_type",         "max"),
    ).reset_index()

    seg_features = [
        "total_sessions", "purchase_rate", "avg_pages_viewed",
        "avg_time_on_site", "avg_unit_price", "avg_quantity",
        "avg_discount_pct", "returning_user",
    ]
    X_seg    = cust[seg_features].fillna(0).values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_seg)

    K_range     = range(2, 9)
    inertias    = []
    silhouettes = []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(
            X_scaled, km.labels_, sample_size=3000, random_state=RANDOM_STATE
        )
        silhouettes.append(sil)

    best_k   = list(K_range)[int(np.argmax(silhouettes))]
    km_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    km_final.fit(X_scaled)
    cust["cluster"] = km_final.labels_

    pca    = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    cust["pca1"] = coords[:, 0]
    cust["pca2"] = coords[:, 1]

    return cust, best_k, list(K_range), inertias, silhouettes, seg_features


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATIONS  (popularity-based, per category)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Building recommendations…")
def build_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    rec = df.groupby(["product_category", "product_id"]).agg(
        purchase_count = ("purchased",        "sum"),
        avg_rating     = ("rating",           "mean"),
        avg_price      = ("unit_price",       "mean"),
        avg_discount   = ("discount_percent", "mean"),
        session_count  = ("session_id",       "count"),
    ).reset_index()
    max_p = rec["purchase_count"].max() or 1
    max_s = rec["session_count"].max()  or 1
    rec["score"] = (
        0.50 * rec["purchase_count"] / max_p
        + 0.30 * (rec["avg_rating"] - 1) / 4.0
        + 0.20 * rec["session_count"] / max_s
    )
    rec["category_label"] = rec["product_category"].map(PRODUCT_CAT_MAP)
    return rec.sort_values(["product_category", "score"], ascending=[True, False])


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC INSIGHTS  (generated from actual data; no hard-coded values)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Computing insights…")
def compute_insights(df: pd.DataFrame) -> dict:
    ins: dict = {}
    ins["n_sessions"]          = len(df)
    ins["n_customers"]         = int(df["customer_id"].nunique())
    ins["purchase_rate"]       = float(df[TARGET].mean())
    ins["total_revenue"]       = float(df["revenue"].sum())
    ins["avg_time_min"]        = float(df["time_on_site_sec"].mean() / 60)
    ins["avg_pages"]           = float(df["pages_viewed"].mean())

    cat_pur = df.groupby("category_label")[TARGET].mean()
    ins["best_category"]       = cat_pur.idxmax()
    ins["best_category_rate"]  = float(cat_pur.max())

    mkt_pur = df.groupby("marketing_label")[TARGET].mean()
    ins["best_channel"]        = mkt_pur.idxmax()
    ins["best_channel_rate"]   = float(mkt_pur.max())

    ins["returning_pur_rate"]  = float(df[df["user_type"] == 1][TARGET].mean())
    ins["new_pur_rate"]        = float(df[df["user_type"] == 0][TARGET].mean())

    dev_pur = df.groupby("device_label")[TARGET].mean()
    ins["best_device"]         = dev_pur.idxmax()
    ins["best_device_rate"]    = float(dev_pur.max())
    return ins


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-CommerceIQ",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
[data-testid="stSidebar"]{background:#1a1a2e;}
[data-testid="stSidebar"] *{color:#e0e0e0 !important;}
.kpi-box{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
         border-radius:10px;padding:18px 12px;text-align:center;color:white;height:100%;}
.kpi-val{font-size:1.9rem;font-weight:700;}
.kpi-lbl{font-size:.82rem;opacity:.85;margin-top:3px;}
</style>
""", unsafe_allow_html=True)


def _kpi(col, label: str, value: str, delta: str = "") -> None:
    with col:
        extra = (f'<div style="font-size:.75rem;margin-top:4px">{delta}</div>'
                 if delta else "")
        st.markdown(
            f'<div class="kpi-box"><div class="kpi-val">{value}</div>'
            f'<div class="kpi-lbl">{label}</div>{extra}</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    df  = load_data()
    ins = compute_insights(df)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🛒 E-CommerceIQ")
        st.markdown("*Customer Analytics & Purchase Prediction*")
        st.markdown("---")
        page = st.radio("Navigate", [
            "📊 Dashboard",
            "👤 Customer Analytics",
            "🔍 Exploratory Data Analysis",
            "🎯 Purchase Prediction",
            "🗂️ Customer Segmentation",
            "🛍️ Product Recommendations",
            "📈 Model Performance",
            "ℹ️ About Project",
        ])
        st.markdown("---")
        st.caption(f"Records: **{ins['n_sessions']:,}** | "
                   f"Customers: **{ins['n_customers']:,}**")
        st.caption(f"Prediction features: **{len(ALL_FEATURES)}**")
        st.caption("IBM SkillsBuild Internship · 2026")
        st.caption("By: **Aarfa Fatima**")

    # =========================================================================
    # PAGE: DASHBOARD
    # =========================================================================
    if page == "📊 Dashboard":
        st.title("📊 E-CommerceIQ — Dashboard")

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        _kpi(c1, "Total Sessions",   f"{ins['n_sessions']:,}")
        _kpi(c2, "Unique Customers", f"{ins['n_customers']:,}")
        _kpi(c3, "Purchase Rate",    f"{ins['purchase_rate']:.1%}")
        _kpi(c4, "Total Revenue",    f"₹{ins['total_revenue']/1e6:.2f}M")
        _kpi(c5, "Avg Session",      f"{ins['avg_time_min']:.1f} min")
        _kpi(c6, "Avg Pages Viewed", f"{ins['avg_pages']:.1f}")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            monthly = (
                df.assign(month_year=df["visit_date"].dt.to_period("M").astype(str))
                  .groupby("month_year")
                  .agg(sessions=("session_id", "count"), purchases=(TARGET, "sum"))
                  .reset_index()
            )
            monthly["rate"] = monthly["purchases"] / monthly["sessions"]
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=monthly["month_year"], y=monthly["sessions"],
                                 name="Sessions", marker_color="#667eea"), secondary_y=False)
            fig.add_trace(go.Scatter(x=monthly["month_year"], y=monthly["rate"],
                                     name="Purchase Rate",
                                     line=dict(color="#f59e0b", width=3)), secondary_y=True)
            fig.update_layout(title="Monthly Sessions & Purchase Rate", height=360,
                              legend=dict(orientation="h"))
            fig.update_yaxes(title_text="Sessions", secondary_y=False)
            fig.update_yaxes(title_text="Purchase Rate", tickformat=".0%", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            cat_perf = (
                df.groupby("category_label")
                  .agg(purchases=(TARGET, "sum"), sessions=("session_id", "count"))
                  .reset_index()
            )
            cat_perf["rate"] = cat_perf["purchases"] / cat_perf["sessions"]
            st.plotly_chart(
                px.bar(cat_perf.sort_values("purchases", ascending=True),
                       x="purchases", y="category_label", orientation="h",
                       color="rate", color_continuous_scale="Viridis",
                       title="Purchases by Product Category",
                       labels={"category_label": "", "purchases": "Purchases",
                               "rate": "Purchase Rate"},
                       height=360),
                use_container_width=True,
            )

        c3, c4, c5 = st.columns(3)
        with c3:
            dev = df.groupby("device_label")[TARGET].sum().reset_index()
            st.plotly_chart(
                px.pie(dev, values=TARGET, names="device_label",
                       title="Purchases by Device", hole=0.4, height=300),
                use_container_width=True,
            )
        with c4:
            mkt = df.groupby("marketing_label")[TARGET].sum().reset_index()
            st.plotly_chart(
                px.bar(mkt.sort_values(TARGET), x=TARGET, y="marketing_label",
                       orientation="h", title="Purchases by Marketing Channel",
                       color=TARGET, color_continuous_scale="Blues", height=300),
                use_container_width=True,
            )
        with c5:
            sea = (
                df.groupby("season_label")
                  .agg(sessions=("session_id", "count"), purchases=(TARGET, "sum"))
                  .reset_index()
            )
            sea["rate"] = sea["purchases"] / sea["sessions"]
            fig_s = px.bar(sea, x="season_label", y="rate", color="season_label",
                           title="Purchase Rate by Season",
                           labels={"rate": "Purchase Rate", "season_label": "Season"},
                           height=300)
            fig_s.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_s, use_container_width=True)

        st.markdown("---")
        st.subheader("📌 Key Insights (auto-generated from data)")
        fi1, fi2, fi3, fi4 = st.columns(4)
        fi1.info(f"**Best Category**\n\n{ins['best_category']} — "
                 f"{ins['best_category_rate']:.1%} purchase rate")
        fi2.info(f"**Best Channel**\n\n{ins['best_channel']} — "
                 f"{ins['best_channel_rate']:.1%} purchase rate")
        fi3.info(f"**Returning vs New Users**\n\n"
                 f"Returning: {ins['returning_pur_rate']:.1%}  |  "
                 f"New: {ins['new_pur_rate']:.1%}")
        fi4.info(f"**Best Device**\n\n{ins['best_device']} — "
                 f"{ins['best_device_rate']:.1%} purchase rate")

    # =========================================================================
    # PAGE: CUSTOMER ANALYTICS
    # =========================================================================
    elif page == "👤 Customer Analytics":
        st.title("👤 Customer Analytics")
        tab1, tab2, tab3 = st.tabs(["User Behaviour", "Purchase Funnel", "Buyer Profile"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                ut = df.groupby("user_type_label")[TARGET].mean().reset_index()
                fig = px.bar(ut, x="user_type_label", y=TARGET, color="user_type_label",
                             title="Purchase Rate: New vs Returning",
                             labels={TARGET: "Purchase Rate",
                                     "user_type_label": "User Type"}, height=340)
                fig.update_yaxes(tickformat=".1%")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                bins = [0, 5, 10, 15, 20, 30, 60]
                lbls = ["0-5 m", "5-10 m", "10-15 m", "15-20 m", "20-30 m", "30-60 m"]
                df2 = df.copy()
                df2["time_bin"] = pd.cut(df2["time_on_site_sec"] / 60,
                                         bins=bins, labels=lbls)
                tp = df2.groupby("time_bin", observed=True)[TARGET].mean().reset_index()
                fig2 = px.line(tp, x="time_bin", y=TARGET, markers=True,
                               title="Purchase Rate vs Session Duration",
                               labels={TARGET: "Purchase Rate", "time_bin": "Time"},
                               height=340)
                fig2.update_yaxes(tickformat=".1%")
                st.plotly_chart(fig2, use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                pb = pd.cut(df["pages_viewed"], bins=[0, 5, 10, 15, 20, 25],
                            labels=["1-5", "6-10", "11-15", "16-20", "21-25"])
                pp = df.groupby(pb, observed=True)[TARGET].mean().reset_index()
                pp.columns = ["Pages", "Rate"]
                fig3 = px.bar(pp, x="Pages", y="Rate",
                              title="Purchase Rate by Pages Viewed", height=340)
                fig3.update_yaxes(tickformat=".1%")
                st.plotly_chart(fig3, use_container_width=True)
            with c4:
                wd = df.groupby("weekday_label")[TARGET].mean().reset_index()
                order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                wd["weekday_label"] = pd.Categorical(
                    wd["weekday_label"], categories=order, ordered=True
                )
                fig4 = px.bar(wd.sort_values("weekday_label"),
                              x="weekday_label", y=TARGET,
                              title="Purchase Rate by Weekday",
                              labels={TARGET: "Purchase Rate",
                                      "weekday_label": "Day"}, height=340)
                fig4.update_yaxes(tickformat=".1%")
                st.plotly_chart(fig4, use_container_width=True)

        with tab2:
            st.subheader("Purchase Funnel")
            n_s = len(df)
            n_p = int(df[TARGET].sum())
            n_buying_cust = int(df[df[TARGET] == 1]["customer_id"].nunique())
            fd = pd.DataFrame({"Stage": ["Total Sessions", "Purchase Completed"],
                               "Count": [n_s, n_p]})
            st.plotly_chart(
                px.funnel(fd, x="Count", y="Stage",
                          title="Session-to-Purchase Funnel", height=380),
                use_container_width=True,
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Sessions", f"{n_s:,}")
            m2.metric("Purchases", f"{n_p:,}", f"{n_p/n_s:.1%} of sessions")
            m3.metric("Buying Customers", f"{n_buying_cust:,}",
                      f"{n_buying_cust/ins['n_customers']:.1%} of customers")

        with tab3:
            st.subheader("Purchaser vs Non-Purchaser Profile")
            b  = df[df[TARGET] == 1]
            nb = df[df[TARGET] == 0]
            pm = {
                "Avg Pages Viewed":       (b["pages_viewed"].mean(),
                                           nb["pages_viewed"].mean()),
                "Avg Time on Site (min)": (b["time_on_site_sec"].mean() / 60,
                                           nb["time_on_site_sec"].mean() / 60),
                "Avg Unit Price (₹)":     (b["unit_price"].mean(),
                                           nb["unit_price"].mean()),
                "Avg Quantity":           (b["quantity"].mean(),
                                           nb["quantity"].mean()),
                "Avg Discount %":         (b["discount_percent"].mean(),
                                           nb["discount_percent"].mean()),
            }
            pdf = pd.DataFrame(pm, index=["Purchasers", "Non-Purchasers"]).T.reset_index()
            pdf.columns = ["Metric", "Purchasers", "Non-Purchasers"]
            st.plotly_chart(
                px.bar(pdf.melt(id_vars="Metric"),
                       x="Metric", y="value", color="variable", barmode="group",
                       title="Behavioural Comparison",
                       color_discrete_map={"Purchasers": "#10b981",
                                           "Non-Purchasers": "#667eea"},
                       labels={"value": "Average", "variable": ""}, height=420),
                use_container_width=True,
            )

    # =========================================================================
    # PAGE: EDA
    # =========================================================================
    elif page == "🔍 Exploratory Data Analysis":
        st.title("🔍 Exploratory Data Analysis")
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Distributions", "Correlations", "Missing Values", "Statistics"]
        )

        with tab1:
            feat = st.selectbox("Numeric feature",
                ["unit_price", "quantity", "discount_percent",
                 "pages_viewed", "time_on_site_sec"])
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    px.histogram(df, x=feat, color=TARGET, barmode="overlay",
                                 opacity=0.7,
                                 color_discrete_map={0: "#667eea", 1: "#f59e0b"},
                                 nbins=40,
                                 title=f"{feat} by Purchase Outcome", height=370),
                    use_container_width=True,
                )
            with c2:
                st.plotly_chart(
                    px.box(df, x=df[TARGET].astype(str), y=feat,
                           color=df[TARGET].astype(str),
                           color_discrete_map={"0": "#667eea", "1": "#f59e0b"},
                           title=f"{feat} distribution",
                           labels={"x": "Purchased (0/1)", "color": ""}, height=370),
                    use_container_width=True,
                )

            cat_feat = st.selectbox("Categorical feature",
                ["device_label", "user_type_label", "marketing_label",
                 "category_label", "season_label", "session_duration_bucket"])
            cp = df.groupby([cat_feat, TARGET]).size().reset_index(name="n")
            cp[TARGET] = cp[TARGET].map({0: "Not Purchased", 1: "Purchased"})
            st.plotly_chart(
                px.bar(cp, x=cat_feat, y="n", color=TARGET, barmode="group",
                       color_discrete_map={"Not Purchased": "#667eea",
                                           "Purchased": "#f59e0b"},
                       title=f"Purchase Outcome by {cat_feat}", height=390),
                use_container_width=True,
            )

        with tab2:
            corr_cols = ["unit_price", "quantity", "discount_percent",
                         "pages_viewed", "time_on_site_sec", "user_type",
                         "marketing_channel", "visit_weekday", "visit_month",
                         "visit_season", TARGET]
            cm_mat = df[corr_cols].corr()
            st.plotly_chart(
                px.imshow(cm_mat, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                          title="Feature Correlation Heatmap", aspect="auto", height=520),
                use_container_width=True,
            )
            tc = cm_mat[TARGET].drop(TARGET).sort_values()
            st.plotly_chart(
                px.bar(x=tc.values, y=tc.index, orientation="h",
                       color=tc.values, color_continuous_scale="RdBu",
                       title="Pearson Correlation with Target (purchased)",
                       labels={"x": "Pearson r", "y": "Feature"}, height=400),
                use_container_width=True,
            )

        with tab3:
            miss = df.isnull().sum()
            mdf  = pd.DataFrame({
                "Feature": miss.index,
                "Missing": miss.values,
                "Pct":     miss.values / len(df) * 100,
            })
            mdf = mdf[mdf["Missing"] > 0]
            if mdf.empty:
                st.success("✅ No missing values found in the dataset.")
            else:
                st.dataframe(mdf, use_container_width=True)
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Total Records",  f"{len(df):,}")
            cc2.metric("Total Columns",  str(len(df.columns)))
            cc3.metric("Duplicate Rows", str(df.duplicated().sum()))

        with tab4:
            num = ["unit_price", "quantity", "discount_percent",
                   "pages_viewed", "time_on_site_sec"]
            st.dataframe(df[num].describe().round(3), use_container_width=True)
            n = st.slider("Sample rows", 5, 50, 10)
            st.dataframe(
                df[["customer_id", "visit_date", "device_label", "user_type_label",
                     "marketing_label", "category_label", "unit_price", "quantity",
                     "discount_percent", "pages_viewed", "time_on_site_sec",
                     "session_duration_bucket", TARGET]].head(n),
                use_container_width=True,
            )

    # =========================================================================
    # PAGE: PURCHASE PREDICTION
    # =========================================================================
    elif page == "🎯 Purchase Prediction":
        st.title("🎯 Purchase Prediction")
        st.markdown(
            "Enter pre-purchase session details to predict whether a customer "
            "will complete a purchase."
        )

        with st.spinner("Loading / training model…"):
            all_results, final_label, best_params, X_tr, X_te, y_tr, y_te, imps = train_models(df)

        res = all_results[final_label]
        st.info(
            f"**Active model (selected by CV ROC-AUC on training set):** {final_label}  |  "
            f"CV ROC-AUC: **{res['cv_auc']:.4f}**  |  "
            f"Test ROC-AUC: **{res['roc_auc']:.4f}**  |  "
            f"Test Accuracy: **{res['accuracy']:.4f}**"
        )
        st.caption(
            "ℹ️ After removing all post-purchase features (added_to_cart, cart_abandoned, "
            "rating, revenue, reviews, payment_method) only genuine pre-session signals "
            "remain. The modest AUC (~0.55–0.57) reflects the true signal in those 15 features "
            "and is the scientifically honest result."
        )

        with st.form("pred_form"):
            st.subheader("Session Details (15 pre-purchase features)")
            c1, c2, c3 = st.columns(3)
            with c1:
                device_in = st.selectbox("Device Type",      list(DEVICE_MAP.values()))
                user_in   = st.selectbox("User Type",         list(USER_TYPE_MAP.values()))
                mkt_in    = st.selectbox("Marketing Channel", list(MARKETING_MAP.values()))
                cat_in    = st.selectbox("Product Category",  list(PRODUCT_CAT_MAP.values()))
            with c2:
                price_in  = st.number_input("Unit Price (₹)", 0.0, 5000.0, 500.0, 50.0)
                qty_in    = st.number_input("Quantity", 1, 10, 1)
                disc_in   = st.slider("Discount %", 0, 50, 10)
                pages_in  = st.slider("Pages Viewed", 1, 30, 10)
            with c3:
                time_in   = st.slider("Time on Site (min)", 1, 60, 15)
                season_in = st.selectbox("Visit Season",    list(SEASON_MAP.values()))
                dur_in    = st.selectbox("Session Duration", DURATION_ORDER)
                wday_in   = st.selectbox("Visit Weekday",
                    ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"])
            c4, c5 = st.columns(2)
            with c4:
                day_in   = st.slider("Visit Day (1–31)",  1, 31, 15)
                month_in = st.slider("Visit Month (1–12)", 1, 12, 6)
            with c5:
                loc_in   = st.number_input("Location Code", 0, 500, 100)

            submitted = st.form_submit_button("🔮 Predict Purchase", use_container_width=True)

        if submitted:
            wday_num = ["Monday", "Tuesday", "Wednesday",
                        "Thursday", "Friday", "Saturday", "Sunday"].index(wday_in)
            input_row = pd.DataFrame([{
                "device_type":             {v: k for k, v in DEVICE_MAP.items()}[device_in],
                "user_type":               {v: k for k, v in USER_TYPE_MAP.items()}[user_in],
                "marketing_channel":       {v: k for k, v in MARKETING_MAP.items()}[mkt_in],
                "product_category":        {v: k for k, v in PRODUCT_CAT_MAP.items()}[cat_in],
                "unit_price":              price_in,
                "quantity":                qty_in,
                "discount_percent":        disc_in,
                "pages_viewed":            pages_in,
                "time_on_site_sec":        time_in * 60,
                "visit_day":               day_in,
                "visit_month":             month_in,
                "visit_weekday":           wday_num,
                "visit_season":            {v: k for k, v in SEASON_MAP.items()}[season_in],
                "location":                loc_in,
                "session_duration_bucket": dur_in,
            }])

            pipe   = all_results[final_label]["pipeline"]
            pred   = int(pipe.predict(input_row)[0])
            prob   = float(pipe.predict_proba(input_row)[0][1])

            st.markdown("---")
            st.subheader("Prediction Result")
            r1, r2, r3 = st.columns(3)
            with r1:
                if pred == 1:
                    st.success("✅ **PURCHASE PREDICTED**")
                else:
                    st.error("❌ **NO PURCHASE PREDICTED**")
                st.metric("Predicted Purchase Probability", f"{prob:.1%}")

            with r2:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(prob * 100, 1),
                    title={"text": "Predicted Purchase Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": "#667eea"},
                        "steps": [
                            {"range": [0,  30], "color": "#fee2e2"},
                            {"range": [30, 60], "color": "#fef3c7"},
                            {"range": [60, 100], "color": "#d1fae5"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75, "value": 50,
                        },
                    },
                ))
                gauge.update_layout(height=240, margin=dict(t=30, b=0, l=10, r=10))
                st.plotly_chart(gauge, use_container_width=True)

            with r3:
                st.markdown("**Input Summary**")
                for k, v in [
                    ("Device",    device_in), ("User Type",  user_in),
                    ("Channel",   mkt_in),    ("Category",   cat_in),
                    ("Price",     f"₹{price_in:,.0f}"), ("Qty",  qty_in),
                    ("Discount",  f"{disc_in}%"), ("Pages",  pages_in),
                    ("Session",   dur_in),    ("Model",      final_label),
                ]:
                    st.markdown(f"- **{k}:** {v}")

    # =========================================================================
    # PAGE: CUSTOMER SEGMENTATION
    # =========================================================================
    elif page == "🗂️ Customer Segmentation":
        st.title("🗂️ Customer Segmentation")
        st.markdown(
            f"Sessions are **aggregated per customer** ({ins['n_customers']:,} unique customers), "
            "then K-Means clusters customers by their overall behavioural profile."
        )

        cust, best_k, K_range, inertias, silhouettes, seg_feat = run_segmentation(df)

        c1, c2 = st.columns(2)
        with c1:
            fig_e = make_subplots(specs=[[{"secondary_y": True}]])
            fig_e.add_trace(go.Scatter(x=K_range, y=inertias, name="Inertia",
                                       line=dict(color="#667eea", width=2),
                                       mode="lines+markers"), secondary_y=False)
            fig_e.add_trace(go.Scatter(x=K_range, y=silhouettes, name="Silhouette",
                                       line=dict(color="#f59e0b", width=2),
                                       mode="lines+markers"), secondary_y=True)
            fig_e.add_vline(x=best_k, line_dash="dash", line_color="red",
                            annotation_text=f"Optimal K={best_k}")
            fig_e.update_layout(title="Elbow Curve + Silhouette Score", height=360)
            fig_e.update_yaxes(title_text="Inertia", secondary_y=False)
            fig_e.update_yaxes(title_text="Silhouette Score", secondary_y=True)
            st.plotly_chart(fig_e, use_container_width=True)
        with c2:
            cd = (cust["cluster"].value_counts()
                                 .reset_index()
                                 .rename(columns={"cluster": "Cluster",
                                                  "count": "Customers"}))
            cd["Cluster"] = cd["Cluster"].astype(str)
            st.plotly_chart(
                px.pie(cd, values="Customers", names="Cluster",
                       title=f"Customer Distribution (K={best_k})",
                       hole=0.4, height=360),
                use_container_width=True,
            )

        st.subheader("Cluster Visualisation (PCA 2D — customer level)")
        sample = cust.sample(min(5000, len(cust)), random_state=RANDOM_STATE)
        st.plotly_chart(
            px.scatter(sample, x="pca1", y="pca2",
                       color=sample["cluster"].astype(str),
                       title="Customer Clusters in PCA Space",
                       labels={"color": "Cluster", "pca1": "PC 1", "pca2": "PC 2"},
                       opacity=0.6, height=450),
            use_container_width=True,
        )

        st.subheader("Cluster Profiles (mean per customer)")
        prof = (cust.groupby("cluster")[seg_feat + ["total_purchases"]]
            .mean()
            .round(3))
        prof.index = [f"Cluster {i}" for i in prof.index]
        st.dataframe(prof, use_container_width=True)

        st.subheader("Purchase Rate by Cluster")
        pr = cust.groupby("cluster")["purchase_rate"].mean().reset_index()
        pr["cluster"] = pr["cluster"].astype(str)
        fig_pr = px.bar(pr, x="cluster", y="purchase_rate", color="cluster",
                        title="Average Customer Purchase Rate per Cluster",
                        labels={"purchase_rate": "Avg Purchase Rate",
                                "cluster": "Cluster"}, height=340)
        fig_pr.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig_pr, use_container_width=True)

        st.subheader("Cluster Radar — normalised behavioural features")
        radar_f = ["total_sessions", "avg_pages_viewed", "avg_time_on_site",
                   "avg_unit_price", "avg_quantity", "avg_discount_pct"]
        np_prof = prof[radar_f].copy()
        for col in np_prof.columns:
            r = np_prof[col].max() - np_prof[col].min()
            np_prof[col] = (np_prof[col] - np_prof[col].min()) / (r + 1e-9)
        fig_rad = go.Figure()
        for i, (idx, row) in enumerate(np_prof.iterrows()):
            vals = list(row) + [row.iloc[0]]
            cats = radar_f + [radar_f[0]]
            fig_rad.add_trace(go.Scatterpolar(r=vals, theta=cats, name=idx,
                                              fill="toself", opacity=0.55))
        fig_rad.update_layout(
            polar=dict(radialaxis=dict(range=[0, 1])),
            title="Normalised Cluster Profiles", height=450,
        )
        st.plotly_chart(fig_rad, use_container_width=True)

    # =========================================================================
    # PAGE: RECOMMENDATIONS
    # =========================================================================
    elif page == "🛍️ Product Recommendations":
        st.title("🛍️ Product Recommendations")
        st.markdown(
            "Popularity-based recommendations within each category: weighted by "
            "purchase count (50%), average rating (30%), and session frequency (20%)."
        )
        rec_df = build_recommendations(df)
        c1, c2 = st.columns([1, 3])
        with c1:
            sel_cat = st.selectbox("Product Category", list(PRODUCT_CAT_MAP.values()))
            n_recs  = st.slider("# Recommendations", 3, 15, 8)

        cat_code = {v: k for k, v in PRODUCT_CAT_MAP.items()}[sel_cat]
        top = rec_df[rec_df["product_category"] == cat_code].head(n_recs).reset_index(drop=True)
        top.index += 1

        with c2:
            show = top[["product_id", "purchase_count", "avg_rating",
                         "avg_price", "avg_discount", "score"]].copy()
            show.columns = ["Product ID", "Purchases", "Avg Rating",
                            "Avg Price (₹)", "Avg Discount %", "Score"]
            show = show.round({"Avg Rating": 2, "Avg Price (₹)": 2,
                               "Avg Discount %": 1, "Score": 4})
            st.dataframe(show, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(
                px.bar(top, x="product_id", y="score", color="avg_rating",
                       color_continuous_scale="YlOrRd",
                       title=f"Recommendation Score — {sel_cat}",
                       labels={"product_id": "Product ID", "score": "Score",
                               "avg_rating": "Avg Rating"}, height=370),
                use_container_width=True,
            )
        with c4:
            st.plotly_chart(
                px.scatter(top, x="avg_price", y="avg_rating",
                           size="purchase_count", color="score",
                           color_continuous_scale="Viridis",
                           hover_data=["product_id", "avg_discount"],
                           title="Price vs Rating vs Popularity",
                           labels={"avg_price": "Avg Price (₹)",
                                   "avg_rating": "Avg Rating"}, height=370),
                use_container_width=True,
            )

        st.markdown("---")
        cat_sum = (
            rec_df.groupby("category_label")
                  .agg(total_purchases=("purchase_count", "sum"),
                       avg_score=("score", "mean"))
                  .reset_index()
                  .sort_values("total_purchases", ascending=False)
        )
        st.plotly_chart(
            px.bar(cat_sum, x="category_label", y="total_purchases",
                   color="avg_score", color_continuous_scale="Blues",
                   title="Total Purchases by Category",
                   labels={"category_label": "Category",
                           "total_purchases": "Total Purchases"}, height=370),
            use_container_width=True,
        )

    # =========================================================================
    # PAGE: MODEL PERFORMANCE
    # =========================================================================
    elif page == "📈 Model Performance":
        st.title("📈 Model Performance")
        with st.spinner("Training / loading models…"):
            all_results, final_label, best_params, X_tr, X_te, y_tr, y_te, imps = train_models(df)

        st.subheader("Model Comparison")
        st.caption(
            "**Selection rule:** best model chosen by 5-fold CV ROC-AUC on the "
            "training set. Test metrics are reported once for the selected model "
            "and are shown here for all models for reference only."
        )
        rows = []
        for name, res in all_results.items():
            rows.append({
                "Model":      name,
                "CV AUC (selection)": f"{res['cv_auc']:.4f} ± {res['cv_std']:.4f}",
                "Test Accuracy":  round(res["accuracy"],  4),
                "Test Precision": round(res["precision"], 4),
                "Test Recall":    round(res["recall"],    4),
                "Test F1":        round(res["f1"],        4),
                "Test ROC-AUC":   round(res["roc_auc"],  4),
                "Selected": "✅" if name == final_label else "",
            })
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

        if best_params:
            with st.expander(f"Best Hyperparameters for {final_label} (RandomizedSearchCV)"):
                st.json(best_params)

        bar_data = [{"Model": name, "Test F1": r["f1"], "Test ROC-AUC": r["roc_auc"]}
                    for name, r in all_results.items()]
        st.plotly_chart(
            px.bar(pd.DataFrame(bar_data).melt(id_vars="Model"),
                   x="Model", y="value", color="variable", barmode="group",
                   title="Test-Set Performance Comparison",
                   labels={"value": "Score", "variable": "Metric"}, height=400)
              .update_yaxes(range=[0, 1]),
            use_container_width=True,
        )

        st.subheader("ROC Curves (test set)")
        colours = ["#667eea", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6"]
        fig_r = go.Figure()
        for (name, res), col in zip(all_results.items(), colours):
            fpr, tpr, _ = roc_curve(res["y_te"], res["y_proba"])
            fig_r.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f"{name} (AUC={res['roc_auc']:.3f})",
                line=dict(color=col, width=2),
            ))
        fig_r.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random (AUC=0.5)",
                                   line=dict(dash="dash", color="gray")))
        fig_r.update_layout(xaxis_title="False Positive Rate",
                            yaxis_title="True Positive Rate",
                            title="ROC Curves", height=450)
        st.plotly_chart(fig_r, use_container_width=True)

        st.subheader("Confusion Matrices (test set)")
        n_models  = len(all_results)
        cm_cols   = st.columns(min(n_models, 5))
        for (name, res), col in zip(all_results.items(), cm_cols):
            with col:
                st.plotly_chart(
                    px.imshow(res["cm"], text_auto=True,
                              color_continuous_scale="Blues",
                              title=name,
                              labels=dict(x="Predicted", y="Actual"),
                              x=["No", "Yes"], y=["No", "Yes"], height=270),
                    use_container_width=True,
                )

        st.subheader("Feature Importance (Random Forest — base model)")
        top_n = st.slider("Top N features", 5, len(imps), 12)
        fi = imps.head(top_n).reset_index()
        fi.columns = ["Feature", "Importance"]
        st.plotly_chart(
            px.bar(fi.sort_values("Importance"), x="Importance", y="Feature",
                   orientation="h", color="Importance",
                   color_continuous_scale="Viridis",
                   title=f"Top {top_n} Feature Importances", height=420),
            use_container_width=True,
        )

    # =========================================================================
    # PAGE: ABOUT
    # =========================================================================
    elif page == "ℹ️ About Project":
        st.title("ℹ️ About E-CommerceIQ")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
## 🛒 E-Commerce Customer Analytics & Purchase Prediction System

**E-CommerceIQ** is an end-to-end data analytics and ML project developed for the
**IBM SkillsBuild Data Analytics with AI Academic Internship (2026)**.

### Objective
Analyse Indian e-commerce customer behaviour, predict purchase completion using
**strictly leak-free** session features, segment customers into behavioural groups,
and deliver product recommendations — all through an interactive Streamlit dashboard.

### Dataset
**Indian E-Commerce Customer Behavior & Purchase** — Kaggle
25,000 sessions · 29 raw columns · 8,442 unique customers

### Prediction Feature Set (15 — all pre-purchase)
`device_type` · `user_type` · `marketing_channel` · `product_category` ·
`unit_price` · `quantity` · `discount_percent` · `pages_viewed` ·
`time_on_site_sec` · `visit_day` · `visit_month` · `visit_weekday` ·
`visit_season` · `location` · `session_duration_bucket`

### ML Pipeline
| Step | Detail |
|------|--------|
| Preprocessing | `ColumnTransformer` → `StandardScaler` (numeric) + `OrdinalEncoder` (duration) |
| Models | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting |
| Tuning | `RandomizedSearchCV` (n_iter=20, 5-fold stratified CV, training set only) |
| Selection | Best base model by CV ROC-AUC → tune → pick if CV AUC improves |
| Evaluation | Single evaluation on held-out 20 % test set (never used for selection) |
| Segmentation | K-Means on customer-level aggregates (Elbow + Silhouette, K=2–8) |
| Recommendations | Popularity-based within-category (purchase count + rating + frequency) |

### Why is the AUC ~0.55–0.57?
Removing all post-purchase leaks (`added_to_cart`, `cart_abandoned`, `rating`,
`review_*`, `revenue`, `payment_method`) leaves only genuine pre-session signals
with inherently low individual correlation with purchase outcome.
This is the **correct and scientifically honest result**.

### Technology Stack
`Python 3.10+` · `Streamlit` · `scikit-learn` · `pandas` · `NumPy` · `Plotly` · `joblib`
""")
        with col2:
            st.markdown("""
### Project Details
- **Author:** Aarfa Fatima
- **Internship:** IBM SkillsBuild
- **Program:** Data Analytics with AI
- **Year:** 2026
---
### Deliverables
- `AarfaFatima_ECommerceIQ.py`
- `AarfaFatima_ECommerceIQ.ipynb`
- `requirements.txt`
- `README.md`
- `AarfaFatima_ECommerceIQ_ProjectReport.docx`
---
### Run
```bash
streamlit run AarfaFatima_ECommerceIQ.py
```
""")

        st.markdown("---")
        st.subheader("Full Leakage Audit")
        audit = pd.DataFrame([
            {"Column": "added_to_cart",       "Decision": "EXCLUDED",
             "Reason": "purchased=1 ↔ added_to_cart=1 (100%) — perfect predictor"},
            {"Column": "cart_abandoned",       "Decision": "EXCLUDED",
             "Reason": "Defined as added_to_cart=1 AND purchased=0 — derived from target"},
            {"Column": "rating",               "Decision": "EXCLUDED",
             "Reason": "All non-purchasers have rating=4 exactly — post-purchase placeholder"},
            {"Column": "review_text",          "Decision": "EXCLUDED",
             "Reason": "Constant value for all non-purchasers — post-purchase"},
            {"Column": "review_helpful_votes", "Decision": "EXCLUDED",
             "Reason": "0 for every non-purchaser — post-purchase"},
            {"Column": "revenue",              "Decision": "EXCLUDED",
             "Reason": "0 for every non-purchaser — direct target signal"},
            {"Column": "revenue_normalized",   "Decision": "EXCLUDED",
             "Reason": "Scaled revenue — same direct leak"},
            {"Column": "discount_amount",      "Decision": "EXCLUDED",
             "Reason": "Exact derivation of unit_price × qty × discount_pct (redundant)"},
            {"Column": "payment_method",       "Decision": "EXCLUDED",
             "Reason": ("Ambiguous checkout/payment-stage field. "
                        "Present for sessions where added_to_cart=0 (7,354 sessions) "
                        "but χ²=6.44 p=0.27 vs target. Low correlation does not "
                        "establish pre-purchase availability; excluded (precautionary).")},
        ])
        st.dataframe(audit.set_index("Column"), use_container_width=True)


if __name__ == "__main__":
    main()
