import os
import json
import re
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

from fpdf import FPDF
from sklearn.cluster import KMeans

# -------------------- PAGE CONFIG & THEME --------------------
st.set_page_config(page_title="Crime Analytics Dashboard", page_icon="📊", layout="wide")
CUSTOM_CSS = """
<style>
    .block-container {padding-top: 1rem; padding-bottom: 1rem; max-width: 1200px;}
    h1, h2, h3 { color: #0F6FFF; }
    .stMetric { background: #f7faff; border-radius: 12px; padding: 0.75rem; }
    .smallnote { color:#6b6b6b; font-size: 0.875rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown("<h2>📊 Advanced Crime Analytics & Governance Suite</h2>", unsafe_allow_html=True)

# -------------------- CONSTANTS --------------------
DATE_COL = "OCCURRED_ON_DATE"
TIME_COL = "TIME"
LAT_COL = "Lat"
LON_COL = "Long"
CAT_COL = "OFFENSE_CODE_GROUP"   # fallback to OFFENSE_DESCRIPTION if missing
DESC_COL = "OFFENSE_DESCRIPTION"
DISTRICT_COL = "DISTRICT"
DOW_COL = "DAY_OF_WEEK"
PII_COLUMNS = ["INCIDENT_NUMBER", "CASE_NUMBER", "Location", "StreetREET"]

# -------------------- DATETIME PARSING HELPERS --------------------
DATE_FORMATS = [
    "%d-%m-%Y", "%m-%d-%Y", "%Y-%m-%d",
    "%d/%m/%Y", "%m/%d/%Y",
]
TIME_FORMATS = [
    "%H:%M:%S", "%H:%M",
    "%I:%M:%S %p", "%I:%M %p"
]

def fix_weird_ampm(t: str) -> str:
    """Fix cases like '13:00:00 PM' → '1:00:00 PM'; '00:15:00 AM' → '12:15:00 AM'."""
    if not isinstance(t, str):
        return ""
    ts = t.strip()
    if ts == "":
        return ts
    # extract hour:min:sec and am/pm if present
    m = re.match(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([AaPp][Mm])?\s*$", ts)
    if not m:
        return ts  # leave as-is; parser will try formats
    hh = int(m.group(1))
    mm = m.group(2)
    ss = m.group(3) or "00"
    ampm = m.group(4)
    if ampm:
        ampm = ampm.upper()
        if hh == 0:
            hh = 12  # 00:xx AM → 12:xx AM
        if 13 <= hh <= 23:
            hh -= 12  # 13–23 with AM/PM → 1–11 with same AM/PM
        return f"{hh}:{mm}:{ss} {ampm}"
    # no AM/PM, leave 24h string intact
    return f"{hh:02d}:{mm}:{ss}"

def try_parse_dt(datestr: str, timestr: str) -> pd.Timestamp:
    """Try multiple format combinations; if all fail, return NaT."""
    ds = (datestr or "").strip()
    ts = fix_weird_ampm((timestr or "").strip())
    # replace dots with slashes/dashes just in case
    ds = ds.replace(".", "-").replace("\\", "/")
    # try date+time combinations
    for dfmt in DATE_FORMATS:
        for tfmt in TIME_FORMATS:
            try:
                dt = datetime.strptime(f"{ds} {ts}".strip(), f"{dfmt} {tfmt}")
                return pd.Timestamp(dt)
            except Exception:
                pass
    # try date-only
    for dfmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(ds, dfmt)
            return pd.Timestamp(dt)
        except Exception:
            pass
    # last resort: dateutil (no warning here since it's per-row)
    try:
        from dateutil import parser
        return pd.Timestamp(parser.parse(f"{ds} {ts}".strip(), dayfirst=True, fuzzy=True))
    except Exception:
        return pd.NaT

# -------------------- LOAD & PREP --------------------
@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]

    # Required columns check
    req = [DATE_COL, LAT_COL, LON_COL, DESC_COL, DISTRICT_COL]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    # Parse dt (row-wise robust parsing; no global infer-warning)
    if TIME_COL not in df.columns:
        df[TIME_COL] = ""

    df["dt"] = df.apply(lambda r: try_parse_dt(str(r[DATE_COL]), str(r[TIME_COL])), axis=1)

    # Coords numeric
    df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors="coerce")
    df[LON_COL] = pd.to_numeric(df[LON_COL], errors="coerce")

    # Category fallback
    if CAT_COL not in df.columns:
        df[CAT_COL] = df[DESC_COL].astype(str)

    # Day-of-week
    if DOW_COL not in df.columns:
        df[DOW_COL] = df["dt"].dt.day_name()

    # Region
    df["region"] = df[DISTRICT_COL].astype(str)

    # Clean drop NaNs and sort by dt
    df = df.dropna(subset=["dt", LAT_COL, LON_COL]).sort_values("dt").reset_index(drop=True)
    return df

@st.cache_data
def filter_df(df, start_date, end_date, categories, regions):
    out = df[(df["dt"] >= pd.to_datetime(start_date)) &
             (df["dt"] <= pd.to_datetime(end_date) + pd.Timedelta(days=1))].copy()
    if categories and "All" not in categories:
        out = out[out[CAT_COL].isin(categories)]
    if regions and "All" not in regions:
        out = out[out["region"].isin(regions)]
    return out

def anonymize_df(df: pd.DataFrame, enable: bool) -> pd.DataFrame:
    if not enable:
        return df
    df = df.copy()
    for c in PII_COLUMNS:
        if c in df.columns:
            df.drop(columns=c, inplace=True)
    df[LAT_COL] = df[LAT_COL].round(4)
    df[LON_COL] = df[LON_COL].round(4)
    return df

# -------------------- MAP --------------------
def fast_density_map(df: pd.DataFrame, show_markers: bool, max_points: int, key: str):
    mdf = df[[LAT_COL, LON_COL, CAT_COL, "dt", "region"]].dropna()
    if len(mdf) == 0:
        st.info("No points to show with current filters.")
        return
    if len(mdf) > max_points:
        mdf = mdf.sample(max_points, random_state=42)

    center = [mdf[LAT_COL].mean(), mdf[LON_COL].mean()]
    fmap = folium.Map(location=center, zoom_start=12)
    HeatMap(mdf[[LAT_COL, LON_COL]].values.tolist(), radius=8, blur=10).add_to(fmap)
    if show_markers:
        mc = MarkerCluster().add_to(fmap)
        for _, r in mdf.iterrows():
            folium.Marker([r[LAT_COL], r[LON_COL]],
                          popup=f"{r[CAT_COL]} | {r['region']} | {r['dt']}").add_to(mc)
    st_folium(fmap, height=450, key=f"map_{key}")

# -------------------- PDF (Unicode-safe) --------------------
def _find_font_candidate():
    candidates = [
        "./DejaVuSans.ttf", "./DejaVuSansCondensed.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialuni.ttf", "C:/Windows/Fonts/Arial.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def _sanitize(s: str) -> str:
    try:
        return s.encode("latin-1").decode("latin-1")
    except Exception:
        return s.encode("ascii", errors="ignore").decode("ascii")

def generate_pdf_report(filtered: pd.DataFrame, notes: str, charts_png: dict) -> bytes:
    pdf = FPDF()
    font_path = _find_font_candidate()
    if font_path:
        try:
            pdf.add_font("Custom", "", font_path, uni=True)
            pdf.set_font("Custom", size=13)
        except Exception:
            pdf.set_font("Arial", size=13)
    else:
        pdf.set_font("Arial", size=13)

    pdf.add_page()
    title = "Crime Analytics Report"
    pdf.set_font_size(16)
    try:
        pdf.cell(0, 10, title, ln=True, align="C")
    except Exception:
        pdf.cell(0, 10, _sanitize(title), ln=True, align="C")

    # Summary
    pdf.set_font_size(11)
    if len(filtered):
        timerange = f"Date Range: {filtered['dt'].min().date()} to {filtered['dt'].max().date()}"
    else:
        timerange = "Date Range: —"
    meta = [
        f"Total Records: {len(filtered)}",
        f"Unique Categories: {filtered[CAT_COL].nunique() if len(filtered) else 0}",
        f"Regions: {', '.join(sorted(filtered['region'].astype(str).unique())[:10])}{' ...' if len(filtered) and filtered['region'].nunique()>10 else ''}",
        timerange,
    ]
    for line in meta:
        try: pdf.multi_cell(0, 8, line)
        except Exception: pdf.multi_cell(0, 8, _sanitize(line))

    # Insights
    pdf.ln(2); pdf.set_font_size(12)
    try: pdf.cell(0, 8, "Key Insights", ln=True)
    except Exception: pdf.cell(0, 8, _sanitize("Key Insights"), ln=True)
    pdf.set_font_size(11)
    if len(filtered):
        try:
            top_cat = filtered[CAT_COL].mode().iloc[0]
            top_reg = filtered['region'].mode().iloc[0]
            peak_day = filtered['dt'].dt.day_name().mode().iloc[0]
            bullets = [
                f"Most common category: {top_cat}",
                f"Region with most incidents: {top_reg}",
                f"Peak day of week: {peak_day}",
            ]
        except Exception:
            bullets = []
        for b in bullets:
            try: pdf.multi_cell(0, 8, f"- {b}")
            except Exception: pdf.multi_cell(0, 8, _sanitize(f"- {b}"))
    else:
        pdf.multi_cell(0, 8, "- No records in current filter")

    # Notes
    if notes:
        pdf.ln(2); pdf.set_font_size(12)
        try: pdf.cell(0, 8, "Analyst Notes", ln=True)
        except Exception: pdf.cell(0, 8, _sanitize("Analyst Notes"), ln=True)
        pdf.set_font_size(11)
        try: pdf.multi_cell(0, 6, notes)
        except Exception: pdf.multi_cell(0, 6, _sanitize(notes))

    # Charts (if provided)
    for label, png_bytes in charts_png.items():
        if not png_bytes:
            continue
        try:
            pdf.ln(2); pdf.set_font_size(12)
            pdf.cell(0, 8, label, ln=True)
            tmp_path = f"_tmp_{re.sub(r'[^A-Za-z0-9]+','_',label)}.png"
            with open(tmp_path, "wb") as f:
                f.write(png_bytes)
            pdf.image(tmp_path, w=180)
            try: os.remove(tmp_path)
            except Exception: pass
        except Exception:
            continue

    try:
        return pdf.output(dest="S").encode("latin-1")
    except Exception:
        # very rare: fallback
        return pdf.output(dest="S").encode("latin-1", errors="ignore")

# Kaleido available?
KALEIDO_OK = True
try:
    import plotly.io as pio  # noqa: F401
except Exception:
    KALEIDO_OK = False

# -------------------- SIDEBAR --------------------
st.sidebar.header("📂 Data")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.sidebar.header("🔒 Privacy")
anonymize = st.sidebar.checkbox("Anonymize (drop PII, round coords)", value=True)

st.sidebar.header("🗺️ Map Options")
max_points = st.sidebar.slider("Max points (sampling)", 500, 10000, 2500, 500)
show_markers = st.sidebar.checkbox("Show markers (slower)", value=False)

st.sidebar.header("🔗 Share / API")
st.sidebar.caption("Query param mode: ?api=aggregates or ?api=daily_counts")

# -------------------- LOAD --------------------
params = st.query_params
api_mode = params.get("api", [None])[0]

if not uploaded and not api_mode:
    st.info("Upload a CSV to start.")
    st.stop()

if uploaded:
    try:
        df_raw = load_data(uploaded)
    except Exception as e:
        st.error(f"Failed to load: {e}")
        st.stop()
else:
    st.error("API mode still requires a dataset upload in this session.")
    st.stop()

# Privacy + clean output for download
work_df = anonymize_df(df_raw, anonymize)

# -------------------- FILTERS --------------------
st.subheader("Filters")
min_dt = pd.to_datetime(work_df["dt"].min()).date()
max_dt = pd.to_datetime(work_df["dt"].max()).date()
start_date, end_date = st.date_input("Date Range", [min_dt, max_dt])

cats = ["All"] + sorted(work_df[CAT_COL].dropna().unique().tolist())
cat_sel = st.multiselect("Crime Category", cats, default=["All"], key="cats")
regions = ["All"] + sorted(work_df["region"].dropna().unique().tolist())
reg_sel = st.multiselect("Region", regions, default=["All"], key="regs")

fdf = filter_df(work_df, start_date, end_date, cat_sel, reg_sel)

# -------------------- METRICS --------------------
colm1, colm2, colm3, colm4 = st.columns(4)
with colm1:
    st.metric("Records", len(fdf))
with colm2:
    st.metric("Categories", int(fdf[CAT_COL].nunique()) if len(fdf) else 0)
with colm3:
    st.metric("Regions", int(fdf['region'].nunique()) if len(fdf) else 0)
with colm4:
    peak = fdf["dt"].dt.day_name().mode().iloc[0] if len(fdf) else "—"
    st.metric("Peak Day", peak)

st.markdown("<p class='smallnote'>Tip: Use fewer markers and the heatmap for faster rendering on large files.</p>", unsafe_allow_html=True)

# -------------------- MAP --------------------
st.subheader("🗺️ Density Map & Hotspots")
fast_density_map(fdf, show_markers=show_markers, max_points=max_points, key="main")
st.caption("Heatmap shows incident density. Toggle markers to inspect specific points.")

# -------------------- VISUALS --------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Daily Trend (Line)")
    if len(fdf):
        daily = (
            fdf.groupby(fdf["dt"].dt.date)
               .size().reset_index(name="count")
               .rename(columns={"dt": "date"})
        )
        fig_line = px.line(daily, x="OCCURRED_ON_DATE" if "OCCURRED_ON_DATE" in daily.columns else "date",
                           y="count", markers=True, title="Daily Incident Counts")
        st.plotly_chart(fig_line, use_container_width=True, key="line")
    else:
        st.info("No data to plot.")

with col2:
    st.subheader("📊 By Category (Bar)")
    if len(fdf):
        bar = fdf[CAT_COL].value_counts().reset_index()
        bar.columns = ["Category", "Count"]
        fig_bar = px.bar(bar, x="Category", y="Count", color="Category", title="Incidents by Category")
        st.plotly_chart(fig_bar, use_container_width=True, key="bar")
    else:
        st.info("No data to plot.")

st.subheader("📊 Additional Interactive Visuals")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Incidents by Day of Week**")
    if len(fdf):
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow_counts = fdf[DOW_COL].value_counts().reindex(order)
        fig_dow = px.bar(dow_counts, x=dow_counts.index, y=dow_counts.values,
                         title="Crimes by Day of Week", labels={"x":"Day", "y":"Count"})
        st.plotly_chart(fig_dow, use_container_width=True, key="dow")
with col_b:
    st.markdown("**Incidents by Hour of Day**")
    if len(fdf):
        temp = fdf.copy()
        temp["hour"] = temp["dt"].dt.hour
        fig_hour = px.histogram(temp, x="hour", nbins=24, title="Crimes by Hour of Day",
                                labels={"hour":"Hour of Day", "count":"Incidents"})
        st.plotly_chart(fig_hour, use_container_width=True, key="hour")

st.markdown("**Category Trends (Stacked Area)**")
if len(fdf):
    top_cats = fdf[CAT_COL].value_counts().nlargest(5).index
    area_df = fdf[fdf[CAT_COL].isin(top_cats)].groupby(
        [fdf["dt"].dt.to_period("M"), CAT_COL]
    ).size().reset_index(name="count")
    area_df["dt"] = area_df["dt"].dt.to_timestamp()
    fig_area = px.area(area_df, x="dt", y="count", color=CAT_COL,
                       title="Top 5 Crime Categories Over Time (Monthly)")
    st.plotly_chart(fig_area, use_container_width=True, key="stacked_area")

st.markdown("**Heatmap: Regions vs Categories**")
if len(fdf):
    heatmap_df = fdf.groupby(["region", CAT_COL]).size().reset_index(name="count")
    pivot = heatmap_df.pivot(index="region", columns=CAT_COL, values="count").fillna(0)
    fig_heat = px.imshow(pivot, text_auto=True, aspect="auto",
                         title="Crime Intensity: Region vs Category",
                         labels=dict(x="Category", y="Region", color="Count"))
    st.plotly_chart(fig_heat, use_container_width=True, key="heatmap")

# -------------------- CLUSTER ANALYSIS --------------------
st.subheader("🔎 Region Clusters (K-Means)")
if len(fdf):
    reg_counts = fdf.groupby("region").size().reset_index(name="crime_count")
    if len(reg_counts) >= 3:
        k = st.slider("Number of clusters", 2, min(8, len(reg_counts)), 3, 1)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        reg_counts["cluster"] = kmeans.fit_predict(reg_counts[["crime_count"]])
        fig_cluster = px.scatter(reg_counts, x="region", y="crime_count", color="cluster",
                                 size="crime_count", title="Region Clusters by Volume")
        st.plotly_chart(fig_cluster, use_container_width=True, key="clusters")
        st.caption("Clusters group regions with similar incident volumes (not geographic distance).")
    else:
        st.info("Need at least 3 regions for clustering.")
else:
    st.info("No data for clustering.")

# -------------------- TABLE --------------------
st.subheader("📑 Records (Filtered)")
st.dataframe(
    fdf[["dt", CAT_COL, DESC_COL, "region", LAT_COL, LON_COL]]
    .sort_values("dt", ascending=False),
    use_container_width=True
)

# -------------------- ANNOTATIONS --------------------
st.subheader("📝 Analyst Notes")
user_note = st.text_area("Add observations/hypotheses for this view (will be included in the report)", height=100)

# -------------------- EXPORTS --------------------
st.subheader("⬇️ Export & API")
colx, coly, colz, colw = st.columns(4)

with colx:
    # NEW: Download the cleaned base dataset (after parsing, sorting, and optional anonymization)
    cleaned_csv = work_df.sort_values("dt").to_csv(index=False)
    st.download_button("Download Cleaned Dataset (CSV)", cleaned_csv,
                       file_name="cleaned_crimes.csv", mime="text/csv")

with coly:
    st.download_button("Download Filtered CSV",
                       fdf.to_csv(index=False),
                       file_name="filtered_crimes.csv",
                       mime="text/csv")

with colz:
    # Prepare chart PNGs if kaleido present
    charts = {}
    if KALEIDO_OK and len(fdf):
        try:
            import plotly.io as pio
            if "fig_line" in locals():
                charts["Daily Trend"] = pio.to_image(fig_line, format="png", width=1000, height=500)
            if "fig_bar" in locals():
                charts["By Category"] = pio.to_image(fig_bar, format="png", width=1000, height=500)
            if "fig_cluster" in locals():
                charts["Clusters"] = pio.to_image(fig_cluster, format="png", width=1000, height=500)
        except Exception:
            charts = {}
    else:
        st.caption("Install kaleido for PNG chart embedding: pip install -U kaleido")

    pdf_bytes = generate_pdf_report(fdf, user_note, charts)
    st.download_button("📄 Download Full PDF Report",
                       data=pdf_bytes,
                       file_name="crime_report.pdf",
                       mime="application/pdf")

with colw:
    # Save notes (optional)
    if st.button("Save Note"):
        os.makedirs("user_notes", exist_ok=True)
        rec = {
            "ts": datetime.utcnow().isoformat(),
            "start": str(start_date),
            "end": str(end_date),
            "cats": cat_sel,
            "regs": reg_sel,
            "note": user_note,
        }
        with open("user_notes/notes.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        st.success("Note saved.")

# Simple API-style previews
st.subheader("🔌 API Preview")
api_choice = st.selectbox("Choose sample endpoint", ["None", "aggregates", "daily_counts"], index=0)
if api_choice == "aggregates":
    api_agg = fdf.groupby(["region", CAT_COL]).size().reset_index(name="count")
    st.json(json.loads(api_agg.to_json(orient="records")))
elif api_choice == "daily_counts":
    api_daily = (fdf.groupby(fdf["dt"].dt.date)
                    .size().reset_index(name="count")
                    .rename(columns={"dt":"date"}))
    st.json(json.loads(api_daily.to_json(orient="records")))

st.markdown("---")
st.caption("Fixes: robust mixed-format datetime parsing (no warnings), Unicode-safe PDF, compact colorful UI, cleaned dataset download, and stable chart/image export.")                     add the above login page with this dashboard and give me full code   
