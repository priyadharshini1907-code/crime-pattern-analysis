import os
import json
import re
from io import BytesIO
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.io as pio
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
from fpdf import FPDF
from sklearn.cluster import KMeans

# ====================== PAGE CONFIG & THEME ======================
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

# ====================== LOGIN SETUP ======================
USER_CREDENTIALS = {
    "admin": "1234",
    "priya": "crime2025",
}

def login_view():
    st.title("🔐 Crime Dashboard Login")
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "show_forgot" not in st.session_state:
        st.session_state.show_forgot = False
    if "pwd_reset_success" not in st.session_state:
        st.session_state.pwd_reset_success = False

    if st.session_state.show_forgot:
        st.subheader("🔑 Forgot Password")
        user = st.text_input("Enter your username", key="fp_user")
        new_pwd = st.text_input("Enter new password", type="password", key="fp_pwd")
        reset = st.button("Reset Password", key="fp_reset_btn")
        if reset:
            if user in USER_CREDENTIALS:
                USER_CREDENTIALS[user] = new_pwd
                st.session_state.show_forgot = False
                st.session_state.pwd_reset_success = True
                st.success("Password reset successful! Please log in.")
            else:
                st.error("Username not found.")
        if st.button("Back to Login", key="fp_back_btn"):
            st.session_state.show_forgot = False
    else:
        if st.session_state.pwd_reset_success:
            st.success("Password reset successful! Please log in.")
            st.session_state.pwd_reset_success = False
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pwd")
        col1, col2 = st.columns(2)
        with col1:
            login_btn = st.button("Login", key="login_btn")
        with col2:
            forgot_btn = st.button("Forgot Password?", key="forgot_btn")
        if login_btn:
            if u in USER_CREDENTIALS and USER_CREDENTIALS[u] == p:
                st.session_state.logged_in = True
                st.session_state.username = u
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
        if forgot_btn:
            st.session_state.show_forgot = True
    if st.session_state.logged_in:
        st.success(f"Welcome {st.session_state.username} 👋")
        if st.button("Logout", key="logout_btn"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()

# ====================== CONSTANTS ======================
DATE_COL = "OCCURRED_ON_DATE"
TIME_COL = "TIME"
LAT_COL = "Lat"
LON_COL = "Long"
CAT_COL = "OFFENSE_CODE_GROUP"
DESC_COL = "OFFENSE_DESCRIPTION"
DISTRICT_COL = "DISTRICT"
# Use a consistent column name for day-of-week derived in load_data
DOW_COL = "day_name"
PII_COLUMNS = ["INCIDENT_NUMBER", "CASE_NUMBER", "Location", "StreetREET"]

# ====================== DATETIME HELPERS ======================
DATE_FORMATS = [
    "%d-%m-%Y", "%m-%d-%Y", "%Y-%m-%d",
    "%d/%m/%Y", "%m/%d/%Y",
]
TIME_FORMATS = [
    "%H:%M:%S", "%H:%M", "%I:%M:%S %p", "%I:%M %p"
]
def fix_weird_ampm(t: str) -> str:
    if not isinstance(t, str):
        return ""
    ts = t.strip()
    if ts == "":
        return ts
    m = re.match(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([AaPp][Mm])?\s*$", ts)
    if not m:
        return ts
    hh = int(m.group(1)); mm = m.group(2); ss = m.group(3) or "00"; ampm = m.group(4)
    if ampm:
        ampm = ampm.upper()
        if hh == 0: hh = 12
        if 13 <= hh <= 23: hh -= 12
        return f"{hh}:{mm}:{ss} {ampm}"
    return f"{hh:02d}:{mm}:{ss}"

def try_parse_dt(datestr: str, timestr: str) -> pd.Timestamp:
    ds = (datestr or "").strip()
    ts = fix_weird_ampm((timestr or "").strip())
    # normalize a few separators
    ds = ds.replace(".", "-")
    for dfmt in DATE_FORMATS:
        for tfmt in TIME_FORMATS:
            try:
                dt = datetime.strptime(f"{ds} {ts}".strip(), f"{dfmt} {tfmt}")
                return pd.Timestamp(dt)
            except Exception:
                pass
    for dfmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(ds, dfmt)
            return pd.Timestamp(dt)
        except Exception:
            pass
    try:
        from dateutil import parser
        return pd.Timestamp(parser.parse(f"{ds} {ts}".strip(), dayfirst=True, fuzzy=True))
    except Exception:
        return pd.NaT

# ====================== LOAD & PREP ======================
@st.cache_data
def load_data(file) -> pd.DataFrame:
    usecols = [DATE_COL, LAT_COL, LON_COL, DESC_COL, DISTRICT_COL, TIME_COL]
    # read with flexible column selection
    df = pd.read_csv(file, dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    # Ensure the columns we expect exist (create defaults where needed)
    if TIME_COL not in df.columns:
        df[TIME_COL] = ""
    if DATE_COL not in df.columns:
        st.warning(f"Expected date column '{DATE_COL}' not present. Trying to use first column as date.")
        df[DATE_COL] = df.iloc[:, 0].astype(str)
    # parse dt
    df["dt"] = df.apply(lambda r: try_parse_dt(str(r.get(DATE_COL, "")), str(r.get(TIME_COL, ""))), axis=1)
    # numeric lat/lon
    df[LAT_COL] = pd.to_numeric(df.get(LAT_COL, pd.Series(np.nan)), errors="coerce")
    df[LON_COL] = pd.to_numeric(df.get(LON_COL, pd.Series(np.nan)), errors="coerce")
    # categories fallback
    if CAT_COL not in df.columns:
        df[CAT_COL] = df[DESC_COL].astype(str)
    # region/district fallback
    df["region"] = df.get(DISTRICT_COL, "").astype(str)
    # drop missing
    df = df.dropna(subset=["dt", LAT_COL, LON_COL]).sort_values("dt").reset_index(drop=True)
    df["dt_str"] = df["dt"].dt.strftime("%Y-%m-%d %H:%M")
    df["day_name"] = df["dt"].dt.day_name()
    # ensure DOW_COL exists (we set it above to "day_name")
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

# ====================== MAP ======================
def fast_density_map(df: pd.DataFrame, show_markers: bool, max_points: int, key: str):
    mdf = df[[LAT_COL, LON_COL, CAT_COL, "dt_str", "region"]].dropna()
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
                          popup=f"{r[CAT_COL]} | {r['region']} | {r['dt_str']}").add_to(mc)
    # set unique key for each map instance
    st_folium(fmap, height=450, key=f"map_{key}")

# ====================== FILE HISTORY UTILS ======================
def register_uploaded_file(uploaded_file):
    if "file_history" not in st.session_state:
        st.session_state.file_history = []
    new_file = {
        "name": uploaded_file.name,
        "size_kb": round(len(uploaded_file.getvalue()) / 1024, 2),
        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    existing = [f for f in st.session_state.file_history if f['name']==new_file['name'] and f['size_kb']==new_file['size_kb']]
    if not existing:
        st.session_state.file_history.append(new_file)

def show_file_history():
    if "file_history" in st.session_state and st.session_state.file_history:
        st.markdown("### 🕑 Upload History This Session")
        hist_df = pd.DataFrame(st.session_state.file_history)
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.info("No CSV upload history yet this session.")

# ====================== PDF (Unicode-safe) ======================
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
        return pdf.output(dest="S").encode("latin-1", errors="ignore")

# detect Kaleido
KALEIDO_OK = True
try:
    _ = pio.to_image
except Exception:
    KALEIDO_OK = False

# ====================== DASHBOARD ======================
def dashboard_view():
    st.markdown("<h2>📊 Crime Analytics Suite (Fast)</h2>", unsafe_allow_html=True)
    st.sidebar.header("📂 Data")
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="uploader")
    st.sidebar.header("🔒 Privacy")
    anonymize = st.sidebar.checkbox("Anonymize (drop PII, round coords)", value=True, key="anon_chk")
    st.sidebar.header("🗺️ Map Options")
    max_points = st.sidebar.slider("Max points (sampling)", 500, 5000, 1500, 500, key="max_points")
    show_markers = st.sidebar.checkbox("Show markers (slower)", value=False, key="show_markers")
    st.sidebar.header("Visual Window")
    recent_days = st.sidebar.slider("Recent days (visualization)", 10, 180, 30, 10, key="recent_days")
    
    show_file_history()

    if not uploaded:
        st.info("Upload a CSV to begin.")
        return
    register_uploaded_file(uploaded)

    df_raw = load_data(uploaded)
    work_df = anonymize_df(df_raw, anonymize)
    max_dt = pd.to_datetime(work_df["dt"]).max()
    min_dt = max_dt - pd.Timedelta(days=recent_days)
    # Top quick-range date_input (unique key)
    start_date, end_date = st.date_input("Date Range (quick)", [min_dt.date(), max_dt.date()], key="date_range_top")
    cats = ["All"] + sorted(work_df[CAT_COL].dropna().unique().tolist())
    cat_sel = st.multiselect("Crime Category (quick)", cats, default=["All"], key="cats_filter1")
    regions = ["All"] + sorted(work_df["region"].dropna().unique().tolist())
    reg_sel = st.multiselect("Region (quick)", regions, default=["All"], key="regs1")
    fdf = filter_df(work_df, start_date, end_date, cat_sel, reg_sel)

    colm1, colm2 = st.columns(2)
    with colm1:
        st.metric("Records (quick)", len(fdf), key="metric_records_top_1")
        st.metric("Categories (quick)", int(fdf[CAT_COL].nunique()) if len(fdf) else 0, key="metric_categories_top")
    with colm2:
        st.metric("Regions (quick)", int(fdf['region'].nunique()) if len(fdf) else 0, key="metric_regions_top")
        peak = fdf["day_name"].mode()[0] if len(fdf) else "—"
        st.metric("Peak Day (quick)", peak, key="metric_peak_top_2")

    st.subheader("🗺️ Map & Hotspots")
    fast_density_map(fdf, show_markers=show_markers, max_points=max_points, key="main_top")
    st.subheader("📊 Category Distribution")
    if len(fdf):
        pie_df = fdf[CAT_COL].value_counts().nlargest(10).reset_index()
        pie_df.columns = ["Category", "Count"]
        fig_pie = px.pie(pie_df, names="Category", values="Count", title="Top Categories (Pie)")
        st.plotly_chart(fig_pie, use_container_width=True, key="fig_pie_top_3")
        bar_df = pie_df
        fig_bar = px.bar(bar_df, x="Category", y="Count", color="Category", title="Top Categories (Bar)")
        st.plotly_chart(fig_bar, use_container_width=True, key="fig_bar_top_4")
    else:
        st.info("No data to plot.")
    st.subheader("📈 Recent Daily Trend")
    if len(fdf):
        daily = fdf.groupby(fdf["dt_str"].str[:10]).size().reset_index(name="count")
        daily.columns = ["date", "count"]
        fig_line = px.line(daily, x="date", y="count", markers=True, title="Incident Counts - Recent Days")
        st.plotly_chart(fig_line, use_container_width=True, key="fig_line_top_5")
    else:
        st.info("No data to plot.")
    st.subheader("📑 Records (Filtered)")
    st.dataframe(
        fdf[["dt_str", CAT_COL, DESC_COL, "region", LAT_COL, LON_COL]].tail(100),
        use_container_width=True,
        key="df_preview_top"
    )
    st.caption("Showing up to last 100 records for speed.")

    # ---------- FILTERS ----------
    st.subheader("Filters")
    min_dt = pd.to_datetime(work_df["dt"].min()).date()
    max_dt = pd.to_datetime(work_df["dt"].max()).date()
    # second date input (unique key)
    start_date, end_date = st.date_input("Date Range (full)", [min_dt, max_dt], key="date_range_bottom")
    cats = ["All"] + sorted(work_df[CAT_COL].dropna().unique().tolist())
    cat_sel = st.multiselect("Crime Category (filters)", cats, default=["All"], key="cats_filter2")
    regions = ["All"] + sorted(work_df["region"].dropna().unique().tolist())
    reg_sel = st.multiselect("Region (filters)", regions, default=["All"], key="regs2")
    fdf = filter_df(work_df, start_date, end_date, cat_sel, reg_sel)

    # ---------- METRICS ----------
    colm1, colm2, colm3, colm4 = st.columns(4)
    with colm1:
        st.metric("Records (full)", len(fdf), key="metric_records_bottom")
    with colm2:
        st.metric("Categories (full)", int(fdf[CAT_COL].nunique()) if len(fdf) else 0, key="metric_categories_bottom")
    with colm3:
        st.metric("Regions (full)", int(fdf['region'].nunique()) if len(fdf) else 0, key="metric_regions_bottom")
    with colm4:
        peak = fdf["dt"].dt.day_name().mode().iloc[0] if len(fdf) else "—"
        st.metric("Peak Day (full)", peak, key="metric_peak_bottom")
    st.markdown("<p class='smallnote'>Tip: Use fewer markers and the heatmap for faster rendering on large files.</p>", unsafe_allow_html=True)

    # ---------- MAP ----------
    st.subheader("🗺️ Density Map & Hotspots")
    fast_density_map(fdf, show_markers=show_markers, max_points=max_points, key="main_bottom")
    st.caption("Heatmap shows incident density. Toggle markers to inspect specific points.")

    # ---------- VISUALS ----------
    # Pie + Bar (side-by-side)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🥧 Category Share (Pie)")
        if len(fdf):
            pie_df = fdf[CAT_COL].value_counts().reset_index()
            pie_df.columns = ["Category", "Count"]
            fig_pie = px.pie(
                pie_df.head(12),
                names="Category",
                values="Count",
                title="Top Category Share"
            )
            st.plotly_chart(fig_pie, use_container_width=True, key="pie_bottom")
        else:
            fig_pie = None
            st.info("No data to plot.")
    with col2:
        st.subheader("📊 By Category (Bar)")
        if len(fdf):
            bar = fdf[CAT_COL].value_counts().reset_index()
            bar.columns = ["Category", "Count"]
            fig_bar = px.bar(bar.head(20), x="Category", y="Count", color="Category", title="Incidents by Category")
            st.plotly_chart(fig_bar, use_container_width=True, key="bar_bottom")
        else:
            fig_bar = None
            st.info("No data to plot.")

    # Time series + Daily line
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📈 Daily Trend (Line)")
        if len(fdf):
            daily = fdf.groupby(fdf["dt"].dt.date).size().reset_index(name="count")
            daily.rename(columns={daily.columns[0]:"date"}, inplace=True)
            fig_line = px.line(daily, x="date", y="count", markers=True, title="Daily Incident Counts")
            st.plotly_chart(fig_line, use_container_width=True, key="line_bottom")
        else:
            fig_line = None
            st.info("No data to plot.")
    with col_b:
        st.subheader("🗓️ Monthly Time Series (Top 5 categories)")
        if len(fdf):
            top_cats = fdf[CAT_COL].value_counts().nlargest(5).index
            area_df = fdf[fdf[CAT_COL].isin(top_cats)].groupby(
                [fdf["dt"].dt.to_period("M"), CAT_COL]
            ).size().reset_index(name="count")
            area_df.rename(columns={area_df.columns[0]:"dt_period"}, inplace=True)
            area_df["dt"] = area_df["dt_period"].dt.to_timestamp()
            fig_area = px.area(area_df, x="dt", y="count", color=CAT_COL,
                               title="Top 5 Crime Categories Over Time (Monthly)")
            st.plotly_chart(fig_area, use_container_width=True, key="stacked_area_bottom")
        else:
            fig_area = None
            st.info("No data to plot.")

    # Day of week + Hour
    col_d, col_h = st.columns(2)
    with col_d:
        st.subheader("📅 Incidents by Day of Week")
        if len(fdf):
            order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            dow_counts = fdf[DOW_COL].value_counts().reindex(order).fillna(0)
            fig_dow = px.bar(
                x=dow_counts.index, y=dow_counts.values,
                title="Crimes by Day of Week",
                labels={"x":"Day", "y":"Count"}
            )
            st.plotly_chart(fig_dow, use_container_width=True, key="dow_bottom")
        else:
            fig_dow = None
            st.info("No data to plot.")
    with col_h:
        st.subheader("⏱️ Incidents by Hour of Day")
        if len(fdf):
            temp = fdf.copy()
            temp["hour"] = temp["dt"].dt.hour
            fig_hour = px.histogram(temp, x="hour", nbins=24, title="Crimes by Hour of Day",
                                    labels={"hour":"Hour of Day", "count":"Incidents"})
            st.plotly_chart(fig_hour, use_container_width=True, key="hour_bottom")
        else:
            fig_hour = None
            st.info("No data to plot.")

    # Heatmap Region vs Category + Comparison grouped bar
    st.subheader("🔥 Heatmap & Comparisons")
    col_hm, col_cmp = st.columns(2)
    with col_hm:
        if len(fdf):
            heatmap_df = fdf.groupby(["region", CAT_COL]).size().reset_index(name="count")
            pivot = heatmap_df.pivot(index="region", columns=CAT_COL, values="count").fillna(0)
            fig_heat = px.imshow(pivot, text_auto=True, aspect="auto",
                                 title="Crime Intensity: Region vs Category",
                                 labels=dict(x="Category", y="Region", color="Count"))
            st.plotly_chart(fig_heat, use_container_width=True, key="heatmap_bottom")
        else:
            fig_heat = None
            st.info("No data to plot.")
    with col_cmp:
        if len(fdf):
            top5 = fdf[CAT_COL].value_counts().nlargest(5).index
            cmp_df = fdf[fdf[CAT_COL].isin(top5)].groupby(["region", CAT_COL]).size().reset_index(name="count")
            fig_cmp = px.bar(cmp_df, x="region", y="count", color=CAT_COL, barmode="group",
                             title="Comparison: Top Categories across Regions")
            st.plotly_chart(fig_cmp, use_container_width=True, key="cmp_bottom")
        else:
            fig_cmp = None
            st.info("No data to plot.")

    # Clusters
    st.subheader("🔎 Region Clusters (K-Means)")
    if len(fdf):
        reg_counts = fdf.groupby("region").size().reset_index(name="crime_count")
        if len(reg_counts) >= 3:
            k = st.slider("Number of clusters", 2, min(8, len(reg_counts)), 3, 1, key="kmeans_k")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            reg_counts["cluster"] = kmeans.fit_predict(reg_counts[["crime_count"]])
            fig_cluster = px.scatter(reg_counts, x="region", y="crime_count", color="cluster",
                                     size="crime_count", title="Region Clusters by Volume")
            st.plotly_chart(fig_cluster, use_container_width=True, key="clusters_bottom")
            st.caption("Clusters group regions with similar incident volumes (not geographic distance).")
        else:
            fig_cluster = None
            st.info("Need at least 3 regions for clustering.")
    else:
        fig_cluster = None
        st.info("No data for clustering.")

    # Table
    st.subheader("📑 Records (Filtered)")
    st.dataframe(
        fdf[["dt", CAT_COL, DESC_COL, "region", LAT_COL, LON_COL]]
        .sort_values("dt", ascending=False),
        use_container_width=True,
        key="df_full_table"
    )

    # Notes
    st.subheader("📝 Analyst Notes")
    user_note = st.text_area("Add observations/hypotheses for this view (will be included in the report)", height=100, key="user_note")

    # ---------- EXPORTS ----------
    st.subheader("⬇️ Export & API")
    colx, coly, colz, colw = st.columns(4)
    with colx:
        cleaned_csv = work_df.sort_values("dt").to_csv(index=False)
        st.download_button("Download Cleaned Dataset (CSV)", cleaned_csv,
                           file_name="cleaned_crimes.csv", mime="text/csv", key="dl_cleaned")
    with coly:
        st.download_button("Download Filtered CSV",
                           fdf.to_csv(index=False),
                           file_name="filtered_crimes.csv",
                           mime="text/csv",
                           key="dl_filtered")
    with colz:
        charts = {}
        if KALEIDO_OK and len(fdf):
            try:
                if 'fig_line_top' in st.session_state and fig_line:   charts["Daily Trend"] = pio.to_image(fig_line, format="png", width=1100, height=500)
            except Exception:
                # try to generate from bottom fig_line if top wasn't present
                try:
                    if fig_line: charts["Daily Trend"] = pio.to_image(fig_line, format="png", width=1100, height=500)
                except Exception:
                    pass
            try:
                if fig_area:   charts["Monthly Series (Top 5)"] = pio.to_image(fig_area, format="png", width=1100, height=500)
                if fig_bar:    charts["By Category"] = pio.to_image(fig_bar, format="png", width=1100, height=500)
                if fig_pie:    charts["Category Share"] = pio.to_image(fig_pie, format="png", width=900, height=700)
                if fig_dow:    charts["By Day of Week"] = pio.to_image(fig_dow, format="png", width=900, height=500)
                if fig_hour:   charts["By Hour"] = pio.to_image(fig_hour, format="png", width=900, height=500)
                if fig_heat:   charts["Region vs Category Heatmap"] = pio.to_image(fig_heat, format="png", width=1100, height=600)
                if fig_cmp:    charts["Regions Comparison (Top Cat)"] = pio.to_image(fig_cmp, format="png", width=1100, height=600)
                if fig_cluster:charts["Region Clusters"] = pio.to_image(fig_cluster, format="png", width=1100, height=600)
            except Exception:
                charts = {}
        else:
            st.caption("Install kaleido for PNG chart embedding: pip install -U kaleido")
        pdf_bytes = generate_pdf_report(fdf, user_note, charts)
        st.download_button("📄 Download Full PDF Report",
                           data=pdf_bytes,
                           file_name="crime_report.pdf",
                           mime="application/pdf",
                           key="dl_pdf")
    with colw:
        if st.button("Save Note", key="save_note_btn"):
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
    api_choice = st.selectbox("Choose sample endpoint", ["None", "aggregates", "daily_counts"], index=0, key="api_choice")
    if api_choice == "aggregates":
        api_agg = fdf.groupby(["region", CAT_COL]).size().reset_index(name="count")
        st.json(json.loads(api_agg.to_json(orient="records")), key="api_agg")
    elif api_choice == "daily_counts":
        api_daily = (fdf.groupby(fdf["dt"].dt.date)
                        .size().reset_index(name="count")
                        .rename(columns={"dt":"date"}))
        st.json(json.loads(api_daily.to_json(orient="records")), key="api_daily")
    st.markdown("---")
    st.caption("Includes: login gate → dashboard with pie/bar/line/time-series/cluster/heatmap/folium map + PDF export of visuals.")

# ====================== APP ENTRY ======================
def main():
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        login_view()
    else:
        dashboard_view()

if __name__ == "__main__":
    main()
