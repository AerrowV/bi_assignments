import os
import io
import contextlib
from typing import List, Optional, Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# Local modules
import hosp_dataloader as dl
import hosp_clean as hc
import hosp_hist as hh
import hosp_boxplot as hb
import hosp_heatmap as ht
import hosp_lineplot as hl
import hosp_scatp as hs

import matplotlib.pyplot as plt  # add once at top with other imports

def _render_current_matplotlib():
    """Render whatever the last plotting call created."""
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "hospital_capacity_for_surgery.ipynb"

st.set_page_config(
    page_title="Hospital Capacity for Surgery — Interactive",
    layout="wide",
)

# ---------- Utilities ----------

@st.cache_data(show_spinner=False)
def _load_default_frames():
    frames = {}
    candidates = {
        "kirurgi_operationer": DATA_DIR / "kirurgi_operationer.xlsx",
        "kirurgi_sengepladser": DATA_DIR / "kirurgi_sengepladser.xlsx",
        "kirurgi_ventetider": DATA_DIR / "kirurgi_ventetider.xlsx",
    }
    for name, path in candidates.items():
        if path.exists():
            try:
                frames[name] = dl.load_data(str(path))
            except Exception as e:
                st.warning(f"Could not load {name} from {path}: {e}")
    return frames

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return hc.clean(df)
    except Exception as e:
        st.error(f"Cleaning failed: {e}")
        return df

def _df_info(df: pd.DataFrame) -> pd.DataFrame:
    info_buf = io.StringIO()
    with contextlib.redirect_stdout(info_buf):
        df.info()
    dtypes = df.dtypes.value_counts()
    return pd.DataFrame({
        "dtype": dtypes.index.astype(str),
        "count": dtypes.values
    })

def _get_numeric_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    exclude = set(exclude or [])
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

def _region_column_guess(df: pd.DataFrame) -> Optional[str]:
    for cand in ["Region", "region", "REGION"]:
        if cand in df.columns:
            return cand
    return None

def _date_column_guess(df: pd.DataFrame) -> Optional[str]:
    for cand in ["Dato", "Date", "date"]:
        if cand in df.columns:
            return cand
    return None

# ---------- Notebook import (optional) ----------

@st.cache_resource(show_spinner=False)
def load_notebook_namespace(nb_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load a .ipynb file and execute its code cells into a namespace dict.
    Returns a dict of variables/functions or None if loading fails.
    """
    if not nb_path.exists():
        return None
    try:
        import nbformat
        ns: Dict[str, Any] = {}
        # Provide common libs for the notebook code
        ns.update({
            "pd": pd,
            "np": np,
        })
        with nb_path.open("r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        for cell in nb.cells:
            if cell.cell_type == "code":
                code = cell.source
                try:
                    exec(code, ns)
                except Exception:
                    # Skip failing cells but keep others
                    pass
        return ns
    except Exception:
        return None

def find_plot_func(ns: Dict[str, Any], keywords: List[str]) -> Optional[str]:
    """Return the name of the first callable in ns whose name contains all keywords."""
    if not ns:
        return None
    for name, obj in ns.items():
        lname = name.lower()
        if callable(obj) and all(k in lname for k in keywords):
            return name
    return None

def _frames_from_notebook(ns: Optional[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """
    Extract dataframes provided by the notebook.
    Supported patterns (in this priority order):
      1) A function `provide_streamlit_data()` returning dict[str, DataFrame]
      2) Variables named like core tables: kirurgi_operationer / kirurgi_sengepladser / kirurgi_ventetider
      3) A merged dataframe named all_kir_cleaned (or a generic df)
    """
    frames: Dict[str, pd.DataFrame] = {}
    if not ns:
        return frames

    # 1) Preferred explicit export function
    if "provide_streamlit_data" in ns and callable(ns["provide_streamlit_data"]):
        try:
            maybe = ns["provide_streamlit_data"]()
            if isinstance(maybe, dict):
                for k, v in maybe.items():
                    if isinstance(v, pd.DataFrame):
                        frames[str(k)] = v
                if frames:
                    return frames
        except Exception:
            pass

    # 2) Named tables commonly used in this project
    for key in ["kirurgi_operationer", "kirurgi_sengepladser", "kirurgi_ventetider"]:
        if key in ns and isinstance(ns[key], pd.DataFrame):
            frames[key] = ns[key]

    # 3) Common merged/single df fallbacks
    for key in ["all_kir_cleaned", "all_kir", "df"]:
        if key in ns and isinstance(ns[key], pd.DataFrame):
            frames[key] = ns[key]
            break

    return frames

st.sidebar.header("Controls")

# Notebook hookup
st.sidebar.subheader("Notebook integration")
st.sidebar.caption(f"Notebook path: {NOTEBOOK_PATH}")
use_nb = st.sidebar.checkbox("Use notebook functions/data (if available)", value=True)
nb_ns = load_notebook_namespace(NOTEBOOK_PATH) if use_nb else None
nb_frames = _frames_from_notebook(nb_ns) if nb_ns else {}

# If you want to completely STOP using the /data folder, keep this False
USE_LOCAL_DATA_FALLBACK = False
default_frames = _load_default_frames() if USE_LOCAL_DATA_FALLBACK else {}

# Build choices (prefer notebook datasets)
choices = ["(Upload your own)"]
if nb_frames:
    choices += [f"[Notebook] {k}" for k in nb_frames.keys()]
if default_frames:
    choices += [f"[Local] {k}" for k in default_frames.keys()]

# If there are notebook frames, default to the first notebook one; otherwise default to upload
default_index = 1 if nb_frames else 0

st.sidebar.subheader("Data source")
data_choice = st.sidebar.selectbox(
    "Choose a dataset",
    options=choices,
    index=min(default_index, len(choices)-1) if choices else 0
)

# Resolve df_raw based on choice
uploaded = None
df_raw = pd.DataFrame()

if data_choice == "(Upload your own)":
    uploaded = st.sidebar.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
    )
    if uploaded is not None:
        if uploaded.name.lower().endswith(".csv"):
            df_raw = pd.read_csv(uploaded)
        else:
            df_raw = pd.read_excel(uploaded)

elif data_choice.startswith("[Notebook] "):
    key = data_choice.replace("[Notebook] ", "", 1)
    df_raw = nb_frames.get(key, pd.DataFrame())

elif data_choice.startswith("[Local] "):
    key = data_choice.replace("[Local] ", "", 1)
    df_raw = default_frames.get(key, pd.DataFrame())

apply_clean = st.sidebar.checkbox("Apply cleaning (hosp_clean.clean)", value=True)

# Friendly status messages
if use_nb and nb_ns is None:
    st.sidebar.info("Notebook not found or could not be loaded; using other sources.")
if use_nb and nb_ns and not nb_frames:
    st.sidebar.warning("Notebook loaded, but no dataframes were found. "
                       "Expose DFs or add a function `provide_streamlit_data()` in the notebook.")

# ---------- Main layout ----------

st.title("Hospital Capacity for Surgery — Interactive Dashboard")
st.caption("Accessible, explainable analysis for non-technical users.")

# Tabs (no Usability Feedback tab)
tab_data, tab_viz, tab_methods, tab_explain = st.tabs(
    ["Data", "Visualize", "Methods & Process", "Explain & Interpret"]
)

with tab_data:
    st.subheader("Source data")
    if df_raw.empty:
        st.info("Load a dataset from the sidebar to begin.")
    else:
        df = _clean(df_raw) if apply_clean else df_raw

        c1, c2, c3 = st.columns([2,1,1])
        with c1:
            st.write("Preview")
            st.dataframe(df.head(50), use_container_width=True)
        with c2:
            st.write("Shape")
            st.metric("Rows", f"{len(df):,}")
            st.metric("Columns", f"{df.shape[1]:,}")
        with c3:
            st.write("Dtypes summary")
            st.dataframe(_df_info(df), use_container_width=True, hide_index=True)

        st.write("---")
        st.write("Describe (numeric)")
        st.dataframe(df.describe(include=[np.number]).T, use_container_width=True)

with tab_viz:
    st.subheader("Interactive visualizations")

    if df_raw.empty:
        st.info("Load a dataset from the sidebar to access visualizations.")
    else:
        df = _clean(df_raw) if apply_clean else df_raw

        # Controls
        plot_type = st.selectbox(
            "Choose a visualization",
            ["Histogram", "Boxplot", "Heatmap (Correlation)", "Scatter", "Capacity over time (Line)"]
        )

        # Common options
        excluded = ["År", "Måned"]
        num_cols = _get_numeric_columns(df, exclude=excluded)

        if plot_type == "Histogram":
            st.caption("Source: notebook function if found, else hosp_hist.plot_hosp_histogram")
            color = st.color_picker("Color", "#1f77b4")
            alpha = st.slider("Alpha", 0.2, 1.0, 0.75, 0.05)
            edgecolor = st.color_picker("Edge color", "#000000")
            font_size = st.slider("Title font size", 10, 32, 16)
            show_skewer = st.checkbox("Show mean/median lines", value=True)

            with st.spinner("Rendering histogram…"):
                called = False
                if nb_ns:
                    nb_func_name = find_plot_func(nb_ns, ["hist"])
                    if nb_func_name:
                        try:
                            nb_func = nb_ns[nb_func_name]
                            nb_func(
                                hosp=df,
                                title="Histogram of Hospital Data",
                                x_label="Value",
                                y_label="Frequency",
                                drop_columns=excluded,
                                color=color,
                                alpha=alpha,
                                edgecolor=edgecolor,
                                font_size=font_size,
                            )
                            called = True
                        except TypeError:
                            try:
                                nb_func(df)
                                called = True
                            except Exception:
                                called = False
                        except Exception:
                            called = False
                if not called:
                    try:
                        hh.plot_hosp_histogram(
                            hosp=df,
                            title="Histogram of Hospital Data",
                            x_label="Value",
                            y_label="Frequency",
                            drop_columns=excluded,
                            color=color,
                            alpha=alpha,
                            edgecolor=edgecolor,
                            font_size=font_size,
                            rect=[0, 0, 1, 0.96],
                            show_skewer=show_skewer,
                        )
                    except TypeError:
                        hh.plot_hosp_histogram(
                            hosp=df,
                            title="Histogram of Hospital Data",
                            x_label="Value",
                            y_label="Frequency",
                            drop_columns=excluded,
                            color=color,
                            alpha=alpha,
                            edgecolor=edgecolor,
                            font_size=font_size,
                        )
                _render_current_matplotlib()

        elif plot_type == "Boxplot":
            st.caption("Source: notebook function if found, else hosp_boxplot.plot_hosp_boxplots")
            with st.spinner("Rendering boxplots…"):
                called = False
                if nb_ns:
                    nb_func_name = find_plot_func(nb_ns, ["box", "plot"])
                    if nb_func_name:
                        try:
                            nb_ns[nb_func_name](df.select_dtypes(include=[np.number]))
                            called = True
                        except Exception:
                            called = False
                if not called:
                    hb.plot_hosp_boxplots(df.select_dtypes(include=[np.number]))
                _render_current_matplotlib()

        elif plot_type == "Heatmap (Correlation)":
            st.caption("Source: notebook function if found, else hosp_heatmap.plot_hosp_heatmap")
            title = st.text_input("Title", "Correlation Heatmap")
            with st.spinner("Rendering heatmap…"):
                called = False
                if nb_ns:
                    nb_func_name = find_plot_func(nb_ns, ["heatmap"])
                    if nb_func_name:
                        try:
                            nb_ns[nb_func_name](df, title=title)
                            called = True
                        except Exception:
                            called = False
                if not called:
                    ht.plot_hosp_heatmap(df, title=title)
                _render_current_matplotlib()

        elif plot_type == "Scatter":
            st.caption("Source: notebook function if found, else hosp_scatp.plot_hosp_scatter")
            if len(num_cols) < 2:
                st.warning("Need at least two numeric columns for scatter.")
            else:
                x_col = st.selectbox("X axis", num_cols, index=0)
                y_col = st.selectbox("Y axis", [c for c in num_cols if c != x_col], index=0)
                hue_col = st.selectbox("Hue (optional)", ["(none)"] + df.columns.tolist(), index=0)
                title = st.text_input("Title", f"{y_col} vs {x_col}")
                alpha = st.slider("Alpha", 0.2, 1.0, 0.6, 0.05)
                palette = st.text_input("Seaborn palette", "Set1")

                with st.spinner("Rendering scatter…"):
                    called = False
                    if nb_ns:
                        nb_func_name = find_plot_func(nb_ns, ["scatter"])
                        if nb_func_name:
                            try:
                                nb_ns[nb_func_name](
                                    df=df,
                                    x_col=x_col,
                                    y_col=y_col,
                                    hue=None if hue_col == "(none)" else hue_col,
                                    palette=palette,
                                    title=title,
                                    alpha=alpha
                                )
                                called = True
                            except Exception:
                                called = False
                    if not called:
                        hs.plot_hosp_scatter(
                            df=df,
                            x_col=x_col,
                            y_col=y_col,
                            hue=None if hue_col == "(none)" else hue_col,
                            palette=palette,
                            title=title,
                            alpha=alpha
                        )
                    _render_current_matplotlib()

        elif plot_type == "Capacity over time (Line)":
            st.caption("Source: notebook function if found, else hosp_lineplot.plot_capacity_over_time")
            date_col = _date_column_guess(df) or st.text_input("Date column", "Dato")
            region_col = _region_column_guess(df) or st.text_input("Region column", "Region")

            default_y = ["Disponible_senge", "Normerede_senge"]
            y_cols = [c for c in default_y if c in df.columns] or _get_numeric_columns(df, exclude=[date_col])
            y_sel = st.multiselect("Y columns", options=y_cols, default=y_cols[:2] if len(y_cols) >= 2 else y_cols[:1])

            with st.spinner("Rendering time series…"):
                called = False
                if nb_ns:
                    nb_func_name = find_plot_func(nb_ns, ["line", "plot"])
                    if nb_func_name:
                        try:
                            nb_ns[nb_func_name](
                                hosp=df,
                                date_col=date_col,
                                region_col=region_col,
                                y_cols=y_sel or y_cols[:1],
                                title="Development of Surgical Capacity Over Time",
                            )
                            called = True
                        except Exception:
                            called = False
                if not called:
                    try:
                        hl.plot_capacity_over_time(
                            hosp=df,
                            date_col=date_col,
                            region_col=region_col,
                            y_cols=y_sel or y_cols[:1],
                            title="Development of Surgical Capacity Over Time",
                            date_format="%Y-%m",
                            figsize=(18, 9),
                        )
                    except Exception as e:
                        st.error(f"Failed to render time series: {e}")
                _render_current_matplotlib()

with tab_methods:
    st.subheader("Analysis process and methods")
    st.markdown(
        """
        **Data pipeline**
        1. Load source files with `hosp_dataloader.load_data` (CSV/Excel auto-detected).
        2. Clean tables with `hosp_clean.clean` (drops Excel index columns, normalises headers, constructs `Dato` from year/month).
        3. Visualise with modules or, if available, notebook functions:
           - `hosp_hist.plot_hosp_histogram`
           - `hosp_boxplot.plot_hosp_boxplots`
           - `hosp_heatmap.plot_hosp_heatmap`
           - `hosp_scatp.plot_hosp_scatter`
           - `hosp_lineplot.plot_capacity_over_time`
        4. Interpret results on the next tab with plain-language explanations.
        """
    )

    with st.expander("Show code import graph"):
        st.code(
            "streamlit_app.py -> hosp_dataloader, hosp_clean, hosp_hist, hosp_boxplot, hosp_heatmap, hosp_scatp, hosp_lineplot (+ optional notebook functions)",
            language="text",
        )

    st.info("Tip: Keep functions pure and reusable to simplify testing and Streamlit integration.")

with tab_explain:
    st.subheader("Explain and interpret results")
    st.markdown(
        """
        This section provides plain-language takeaways from the chosen visualisation:
        - **Histograms:** Skewed distributions and long tails can indicate capacity constraints or reporting anomalies.
        - **Boxplots:** Wide IQR or many outliers may point to inconsistent availability across time or regions.
        - **Heatmaps:** Strong correlations can suggest drivers for capacity (e.g., staffing vs. available beds).
        - **Scatter:** Patterns and clusters can reveal trade-offs (e.g., wait time vs. occupied beds).
        - **Time series:** Seasonal cycles or structural breaks can inform planning and staffing decisions.

        **Benefits of visualisation and explanation**
        - Improves trust and transparency for non-technical stakeholders.
        - Speeds up decision-making by highlighting the most actionable patterns.
        - Makes model assumptions and data quality issues visible.
        """
    )

st.sidebar.write("---")
st.sidebar.caption("Built with Streamlit · Uses your hosp_* modules and optionally your notebook")