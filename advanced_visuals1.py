# advanced_crime_dashboard.py
# Streamlit app: Advanced Crime Analytics & Governance Suite

import json
import os
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

from fpdf import FPDF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
try:
    import joblib
except Exception:
    joblib = None

st.set_page_config(page_title="Advanced Crime Analytics", layout="wide")
st.title("🚓 Advanced Crime Analytics & Governance Suite")

# ---------------------------- Utilities ----------------------------
PII_COLUMNS = [
    "INCIDENT_NUMBER",  
    "CASE_NUMBER",
    "Location",
    "StreetREET",  # typo preserved
]

DATE_COL = "OCCURRED_ON_DATE"
TIME_COL = "TIME"
LAT_COL = "Lat"
LON_COL = "Long"
CAT_COL = "OFFENSE_CODE_GROUP"
DESC_COL = "OFFENSE_DESCRIPTION"
DISTRICT_COL = "DISTRICT"
DOW_COL = "DAY_OF_WEEK"

def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    date_str = df[DATE_COL].astype(str).str.strip()
    time_str = df[TIME_COL].astype(str).str.strip() if TIME_COL in df.columns else "00:00:00"
    combo = (date_str + " " + time_str).str.strip()
    dt = pd.to_datetime(combo, errors="coerce", dayfirst=True, infer_datetime_format=True)
    missing = dt.isna()
    if missing.any():
        dt.loc[missing] = pd.to_datetime(date_str.loc[missing], errors="coerce", dayfirst=True)
    df["dt"] = dt
    return df

def load_dataset(uploaded) -> pd.DataFrame:
    df = pd.read_csv(uploaded)
    df.columns = [c.strip() for c in df.columns]
    required = [DATE_COL, LAT_COL, LON_COL, DESC_COL, DISTRICT_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}. Please provide those fields.")
        st.stop()
    df = parse_datetime(df)
    df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors="coerce")
    df[LON_COL] = pd.to_numeric(df[LON_COL], errors="coerce")
    if CAT_COL not in df.columns:
        df[CAT_COL] = df[DESC_COL]
    if DOW_COL not in df.columns:
        df[DOW_COL] = df["dt"].dt.day_name()
    df["region"] = df[DISTRICT_COL].astype(str)
    return df.dropna(subset=["dt", LAT_COL, LON_COL])

def apply_privacy(df: pd.DataFrame, anonymize: bool, aggregate_only: bool) -> pd.DataFrame:
    df = df.copy()
    if anonymize:
        for c in PII_COLUMNS:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)
        df[LAT_COL] = df[LAT_COL].round(4)
        df[LON_COL] = df[LON_COL].round(4)
    if aggregate_only:
        agg = (
            df.groupby([pd.Grouper(key="dt", freq="D"), "region", CAT_COL])
              .size()
              .reset_index(name="count")
        )
        return agg
    return df

def save_config(config: dict) -> bytes:
    return json.dumps(config, indent=2).encode("utf-8")

def load_config(bytestr: bytes) -> dict:
    return json.loads(bytestr.decode("utf-8"))

# ---------------------------- Sidebar ----------------------------
with st.sidebar:
    st.header(" Data")
    uploaded = st.file_uploader("Upload Crime CSV", type=["csv"])    
    st.caption("Expected columns: OCCURRED_ON_DATE, TIME, Lat, Long, OFFENSE_DESCRIPTION, OFFENSE_CODE_GROUP (optional), DISTRICT, DAY_OF_WEEK (optional)")

    st.header(" Privacy")
    anonymize = st.checkbox("Anonymize sensitive data (drop PII, round coords)", value=True)
    aggregate_only = st.checkbox("Show aggregated view only", value=False)

    st.header(" Map Performance")
    max_points = st.slider("Max points on map (sampling)", 500, 10000, 2500, 500)
    show_markers = st.checkbox("Show sample markers (slower)", value=False)

    st.header(" Save/Load Config")
    if st.button("Save current config"):
        cfg = {
            "anonymize": anonymize,
            "aggregate_only": aggregate_only,
            "max_points": max_points,
            "show_markers": show_markers,
        }
        st.download_button("Download config.json", data=save_config(cfg), file_name="dashboard_config.json")
    cfg_upload = st.file_uploader("Load config.json", type=["json"], key="cfg")

# ---------------------------- Main flow ----------------------------
params = st.query_params
api_mode = params.get("api", [None])[0]

if uploaded is None and not api_mode:
    st.info("Upload a CSV to begin.")
    st.stop()

if cfg_upload is not None:
    try:
        cfg = load_config(cfg_upload.getvalue())
        anonymize = cfg.get("anonymize", anonymize)
        aggregate_only = cfg.get("aggregate_only", aggregate_only)
        max_points = cfg.get("max_points", max_points)
        show_markers = cfg.get("show_markers", show_markers)
        st.success("Config loaded.")
    except Exception as e:
        st.warning(f"Failed to load config: {e}")

if uploaded is not None:
    df_raw = load_dataset(uploaded)
else:
    st.error("API mode currently requires an uploaded dataset in this session.")
    st.stop()

work_df = apply_privacy(df_raw, anonymize=anonymize, aggregate_only=False)

# ---------------------------- Filters ----------------------------
st.subheader("Filters")

min_dt = pd.to_datetime(work_df["dt"].min()).date()
max_dt = pd.to_datetime(work_df["dt"].max()).date()
start_date, end_date = st.date_input("Date Range", [min_dt, max_dt])

cats = ["All"] + sorted(work_df[CAT_COL].dropna().unique().tolist())
cat_sel = st.multiselect("Crime Category", cats, default=["All"]) 

regions = ["All"] + sorted(work_df["region"].dropna().unique().tolist())
reg_sel = st.multiselect("Region (District)", regions, default=["All"]) 

fdf = work_df[(work_df["dt"] >= pd.to_datetime(start_date)) & (work_df["dt"] <= pd.to_datetime(end_date) + pd.Timedelta(days=1))]
if "All" not in cat_sel:
    fdf = fdf[fdf[CAT_COL].isin(cat_sel)]
if "All" not in reg_sel:
    fdf = fdf[fdf["region"].isin(reg_sel)]

# ---------------------------- Map & Trends ----------------------------
if aggregate_only:
    st.info("Aggregated privacy view enabled: showing daily region/category counts.")
    agg = (
        fdf.groupby([pd.Grouper(key="dt", freq="D"), "region", CAT_COL])
           .size()
           .reset_index(name="count")
    )
    st.dataframe(agg)
    st.download_button("Download aggregated CSV", agg.to_csv(index=False), file_name="aggregated_counts.csv")
else:
    st.subheader(" Interactive Map (fast)")
    map_df = fdf[[LAT_COL, LON_COL, CAT_COL, "dt", "region"]].dropna().copy()
    if len(map_df) > max_points:
        map_df = map_df.sample(max_points, random_state=42)

    if map_df.empty:
        st.warning("No points to display with current filters.")
    else:
        center = [map_df[LAT_COL].mean(), map_df[LON_COL].mean()]
        m = folium.Map(location=center, zoom_start=12)
        HeatMap(map_df[[LAT_COL, LON_COL]].values.tolist(), radius=8, blur=10).add_to(m)
        if show_markers:
            mc = MarkerCluster().add_to(m)
            for _, r in map_df.iterrows():
                folium.Marker([r[LAT_COL], r[LON_COL]], popup=f"{r[CAT_COL]}<br>{r['dt']}<br>{r['region']}").add_to(mc)
        st_folium(m, height=500, width=None)

    st.subheader(" Trends & Seasonality")
    if not fdf.empty:
        daily = fdf.groupby(fdf["dt"].dt.date).size().reset_index(name="count").rename(columns={"dt":"date"})
        fig_ts = px.line(daily, x="date", y="count", title="Daily Incident Counts")
        st.plotly_chart(fig_ts, use_container_width=True, key="trend_daily")

        fdf["hour"] = fdf["dt"].dt.hour
        fdf["weekday"] = fdf["dt"].dt.day_name()
        pivot = fdf.pivot_table(index="hour", columns="weekday", values=CAT_COL, aggfunc="count").fillna(0)
        fig_hm = px.imshow(pivot, labels={"x":"Weekday", "y":"Hour", "color":"Incidents"}, title="Hourly vs. Weekday Heatmap")
        st.plotly_chart(fig_hm, use_container_width=True, key="trend_heatmap")

# ---------------------------- Compare Regions ----------------------------
st.header("Compare Regions / Periods")
colA, colB = st.columns(2)
with colA:
    st.subheader("Panel A")
    regA = st.selectbox("Region A", sorted(df_raw["region"].unique()), key="regA")
    startA, endA = st.date_input("Date Range A", [min_dt, max_dt], key="dateA")
    A = df_raw[(df_raw["region"]==regA) & (df_raw["dt"].between(pd.to_datetime(startA), pd.to_datetime(endA)+pd.Timedelta(days=1)))]
    a_daily = A.groupby(A["dt"].dt.date).size().reset_index(name="count").rename(columns={"dt":"date"})
    st.plotly_chart(px.line(a_daily, x="date", y="count", title=f"Daily — {regA}"), use_container_width=True, key=f"panelA_{regA}")

with colB:
    st.subheader("Panel B")
    regB = st.selectbox("Region B", sorted(df_raw["region"].unique()), key="regB")
    startB, endB = st.date_input("Date Range B", [min_dt, max_dt], key="dateB")
    B = df_raw[(df_raw["region"]==regB) & (df_raw["dt"].between(pd.to_datetime(startB), pd.to_datetime(endB)+pd.Timedelta(days=1)))]
    b_daily = B.groupby(B["dt"].dt.date).size().reset_index(name="count").rename(columns={"dt":"date"})
    st.plotly_chart(px.line(b_daily, x="date", y="count", title=f"Daily — {regB}"), use_container_width=True, key=f"panelB_{regB}")

# ---------------------------- Model Training ----------------------------
st.header(" Predictive Model — Random Forest")
with st.expander("Train / Update Model"):
    st.caption("Target: OFFENSE_CODE_GROUP (falls back to OFFENSE_DESCRIPTION if missing).")
    target_col = CAT_COL if CAT_COL in df_raw.columns else DESC_COL
    f_cols_num = [LAT_COL, LON_COL]
    f_cols_cat = ["region", DOW_COL]

    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    n_estimators = st.slider("Trees", 100, 800, 400, 50)
    max_depth = st.select_slider("Max depth", options=[None, 10, 20, 30, 40], value=None)

    mdl_df = df_raw[[target_col] + f_cols_num + f_cols_cat + ["dt"]].dropna().copy()
    mdl_df["hour"] = mdl_df["dt"].dt.hour
    f_cols_num2 = f_cols_num + ["hour"]

    X = mdl_df[f_cols_num2 + f_cols_cat]
    y = mdl_df[target_col].astype(str)

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), f_cols_cat),
        ("num", "passthrough", f_cols_num2),
    ])

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    print(y.value_counts())
    counts = Counter(y)
    mask = y.map(counts) > 1  # keep only classes with >1 sample
    X, y = X[mask], y[mask]
    pipe = Pipeline([("pre", pre), ("rf", rf)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    if st.button("Train / Update Model"):
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"Accuracy: {acc*100:.2f}%")
        st.text(classification_report(y_test, y_pred, zero_division=0))

        cm = confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique()))
        fig_cm, ax = plt.subplots(figsize=(7,5))
        sns.heatmap(cm, ax=ax, cmap="Blues", cbar=True, xticklabels=sorted(y_test.unique()), yticklabels=sorted(y_test.unique()))
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig_cm, use_container_width=True)

        os.makedirs("models", exist_ok=True)
        if joblib is not None:
            joblib.dump(pipe, "models/random_forest_crime.joblib")
            st.caption("Model saved to models/random_forest_crime.joblib")

        os.makedirs("outputs", exist_ok=True)
        pred_out = X_test.copy()
        pred_out[target_col + "_true"] = y_test.values
        pred_out[target_col + "_pred"] = y_pred
        pred_path = "outputs/test_predictions.csv"
        pred_out.to_csv(pred_path, index=False)
        st.download_button("Download test_predictions.csv", data=pred_out.to_csv(index=False), file_name="test_predictions.csv")
