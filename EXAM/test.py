import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as sm 
import graphviz
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
from sklearn import model_selection, tree, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import re
from pathlib import Path


# --- Clean ---


def clean_beds(df: pd.DataFrame, drop_blandet: bool = True, add_date: bool = True) -> pd.DataFrame:
    df = df.copy()

    # Fix headers if Excel exported extra "Unnamed" columns
    if any(str(c).lower().startswith("unnamed") for c in df.columns):
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)

    # Clean column names (remove spaces/dashes)
    df.columns = [str(c).strip().replace(" ", "").replace("-", "") for c in df.columns]

    df["År"] = pd.to_numeric(df["År"], errors="coerce")
    df["Måned"] = pd.to_numeric(df["Måned"], errors="coerce")
    df = df.dropna(subset=["År","Måned"])
    df["År"] = df["År"].astype(int)
    df["Måned"] = df["Måned"].astype(int)
    df = df[df["Måned"].between(1, 12)]

riget_disp_clean_df = clean_beds(riget_disp)
riget_norm_clean_df = clean_beds(riget_norm)
aarhus_disp_clean_df = clean_beds(aarhus_disp)
aarhus_norm_clean_df = clean_beds(aarhus_norm)

riget_disp_clean_df.info()
print(riget_disp_clean_df.sample(5))

def clean_wp(df: pd.DataFrame, drop_blandet: bool = True) -> pd.DataFrame:
    df = df.copy()

    # Fix headers if Excel exported an extra header row with "Unnamed" columns
    if any(str(c).lower().startswith("unnamed") for c in df.columns):
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)

    # Normalize column names minimally (trim spaces)
    df.columns = [str(c).strip() for c in df.columns]

    # Rename Year/Month -> År/Måned (case-insensitive)
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "year":
            rename_map[c] = "År"
        elif lc == "month":
            rename_map[c] = "Måned"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Identify bucket columns like "0-30 dage", "31-60 dage", ..., "90+ dage"
    def is_bucket(col: str) -> bool:
        s = str(col).strip().lower()
        return bool(
            re.fullmatch(r"\d+\s*-\s*\d+\s*dage", s) or
            re.fullmatch(r"\d+\s*\+\s*dage", s)
        )

    bucket_cols = [c for c in df.columns if is_bucket(c)]

    # Build keep list: År, Måned, and any bucket columns we found
    keep = [c for c in ["År", "Måned"] if c in df.columns] + bucket_cols
    df = df[keep].copy()

    # Convert numeric columns
    if "År" in df.columns:
        df["År"] = pd.to_numeric(df["År"], errors="coerce")
    if "Måned" in df.columns:
        # Convert to numeric -> int to remove any leading zeros
        df["Måned"] = pd.to_numeric(df["Måned"], errors="coerce")

    for c in bucket_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows missing year/month and enforce valid month 1..12 as integers
    df = df.dropna(subset=[c for c in ["År", "Måned"] if c in df.columns])
    if "År" in df.columns:
        df["År"] = df["År"].astype(int)
    if "Måned" in df.columns:
        df["Måned"] = df["Måned"].astype(int)
        df = df[df["Måned"].between(1, 12)]
    return df

wp_hstaden_clean_df = clean_wp(wp_hstaden)
wp_mjylland_clean_df = clean_wp(wp_mjylland)

wp_hstaden_clean_df.info()

all_hosp_disp = pd.concat([riget_disp_clean_df, aarhus_disp_clean_df], ignore_index=True)
all_hosp_norm = pd.concat([riget_norm_clean_df, aarhus_norm_clean_df], ignore_index=True)
all_hosp_wp = pd.concat([wp_hstaden_clean_df, wp_mjylland_clean_df], ignore_index=True)

print("Waiting Period",all_hosp_wp.shape)
print("Number of Beds (available)", all_hosp_disp.shape)
print("Number of Beds (occuipied)", all_hosp_norm.shape)

# --- Histograms ---
def plot_hosp_histogram(hosp, title="Histogram of Hospital Data"):
    hosp_num = hosp.drop(columns=["År", "Måned"], errors="ignore")
    
    axes = hosp_num.hist(bins=30, figsize=(20, 12))
    plt.suptitle(title, fontsize=16)
    for row in np.atleast_2d(axes):
        for ax in row:
            if ax:
                ax.set_xlabel("Number of Beds")
                ax.set_ylabel("Frequency (Months)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

plot_hosp_histogram(riget_disp_clean_df,title="Available Beds Riget")
plot_hosp_histogram(riget_norm_clean_df,title="Occuipied Beds Riget")
plot_hosp_histogram(aarhus_disp_clean_df,title="Available Beds Aarhus")
plot_hosp_histogram(aarhus_norm_clean_df,title="Occuipied Beds Aarhus")
plot_hosp_histogram(all_hosp_disp,title="All Available")
plot_hosp_histogram(all_hosp_norm,title="All Occuipied")
# plot_hosp_histogram(all_hosp_wp, title="All Waiting Period")

def plot_beds_over_time(df, title="Beds over Time"):
    plt.figure(figsize=(12,6))
    for col in ["Kirurgi", "Medicin", "Onkologi", "Øvrige"]:
        if col in df.columns:
            plt.plot(df["År"].astype(str) + "-" + df["Måned"].astype(str), df[col], marker="o", label=col)
    plt.xticks(rotation=90)
    plt.ylabel("Number of Beds")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_beds_over_time(riget_disp_clean_df, title="Available Beds Riget")
plot_beds_over_time(riget_norm_clean_df, title="Occupied Beds Riget")
plot_beds_over_time(aarhus_disp_clean_df, title="Available Beds Aarhus")
plot_beds_over_time(aarhus_norm_clean_df, title="Occupied Beds Aarhus")

# --- Heatmaps ---
def plot_correlation_heatmap(df, title="Correlation Heatmap"):
    plt.figure(figsize=(20,12))

    df_num = df.select_dtypes(include=[np.number]).drop(columns=["År", "Måned"], errors="ignore")
    corr = df_num.corr()

    # Heatmap
    sb.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title(title)
    plt.show()

plot_correlation_heatmap(all_hosp_disp, title="Hospital Bed Data (available) Correlation Heatmap")
plot_correlation_heatmap(all_hosp_norm, title="Hospital Bed Data (occuipied) Correlation Heatmap")

# --- Scatterplots ---

# -> IMPORTANT Data - Analysis notes

# Based on our scatterplots, we can conclude that scatterplots are not suitable for comparing surgery and medicine,
# since they involve two very different types of procedures. This results in obscure/misleading data. 
# This also applies too Onkologi and Surgery.
"""
def plot_catagories_scatter(df, x_col, y_col, hue=None, palette="Set1", title="", alpha=0.6):
    df_num = df.drop(columns=["År", "Måned"], errors="ignore")
    plt.figure(figsize=(12,6))
    sb.scatterplot(data=df_num, x=x_col, y=y_col, hue=hue, palette=palette, alpha=alpha)
    
    if title is None:
        title = f"{y_col.capitalize()} vs {x_col.capitalize()}"
    plt.title(title)
    plt.show()

plot_catagories_scatter(all_hosp_disp, x_col="Kirurgi", y_col="Medicin", title ="Scatterplot of available beds")
plot_catagories_scatter(all_hosp_disp, x_col="Kirurgi", y_col="Onkologi", title = "Scatterplot of available beds")
plot_catagories_scatter(all_hosp_norm, x_col="Kirurgi", y_col="Medicin", title = "Scatterplot of occuipied beds")
plot_catagories_scatter(all_hosp_norm, x_col="Kirurgi", y_col="Onkologi", title = "Scatterplot of occuipied beds")

"""
# Tag regions on available beds
riget_disp_reg  = riget_disp_clean_df.assign(hospital="Rigshospitalet", region="Region Hovedstaden")
aarhus_disp_reg = aarhus_disp_clean_df.assign(hospital="Aarhus Universitetshospital", region="Region Midtjylland")

# Total available beds per region-month
def beds_total_by_region_month(df):
    dept_cols = [c for c in ["Kirurgi","Medicin","Onkologi","Øvrige"] if c in df.columns]
    out = df.copy()
    out["TotalBeds"] = out[dept_cols].sum(axis=1)  # sum across departments
    grp = out.groupby(["region","År","Måned"], as_index=False)["TotalBeds"].sum()  # sum across hospitals within region
    grp["Date"] = pd.to_datetime(grp["År"].astype(str) + "-" + grp["Måned"].astype(str) + "-01")
    return grp

beds_disp_reg = beds_total_by_region_month(pd.concat([riget_disp_reg, aarhus_disp_reg], ignore_index=True))

# Melt waiting buckets to long
def melt_wait(df, region_name: str):
    d = df.copy()
    d["region"] = region_name
    bucket_cols = [c for c in d.columns if c not in ["År", "Måned", "region"]]
    long = d.melt(id_vars=["År","Måned","region"], value_vars=bucket_cols,
                  var_name="bucket", value_name="patients")
    long = long.dropna(subset=["patients"])
    long["Date"] = pd.to_datetime(long["År"].astype(str) + "-" + long["Måned"].astype(str) + "-01")
    return long

wp_h_long = melt_wait(wp_hstaden_clean_df,  "Region Hovedstaden")
wp_m_long = melt_wait(wp_mjylland_clean_df, "Region Midtjylland")
wp_long   = pd.concat([wp_h_long, wp_m_long], ignore_index=True)

# Merge waiting with beds on region + month
wait_vs_beds = wp_long.merge(
    beds_disp_reg[["region","Date","TotalBeds"]],
    on=["region","Date"],
    how="inner"
)

# Scatter: patients vs available beds
def plot_wait_vs_capacity_scatter(df, title="Waiting Patients vs Available Beds"):
    plt.figure(figsize=(10,6))
    sb.scatterplot(data=df, x="TotalBeds", y="patients", hue="bucket", style="region", alpha=0.8)
    plt.xlabel("Available Beds")
    plt.ylabel("Patients Waiting")
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_wait_vs_capacity_scatter(wait_vs_beds)

# --- Barplot ---
def plot_avg_beds_per_year(df, group_col="År", title="Average Beds per Year"):
    dept_cols = [c for c in ["Kirurgi","Medicin","Onkologi","Øvrige"] if c in df.columns]
    melted = df.melt(id_vars=[group_col], value_vars=dept_cols,
                     var_name="Department", value_name="Beds")

    plt.figure(figsize=(12,6))
    sb.barplot(data=melted, x=group_col, y="Beds", hue="Department", ci=None)
    plt.title(title)
    plt.ylabel("Average Beds")
    plt.xlabel(group_col)
    plt.legend(title="Department")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_avg_beds_per_year(riget_disp_clean_df, group_col="År", title="Riget - Avg Available Beds per Year")
plot_avg_beds_per_year(aarhus_disp_clean_df, group_col="År", title="Aarhus - Avg Available Beds per Year")

# --- Boxplots ---
def plot_department_boxplots(df, title="Beds variation by department"):
    hosp_num = df.drop(columns=["År", "Måned"], errors="ignore")
    melted = hosp_num.melt(var_name="Department", value_name="Beds")

    plt.figure(figsize=(10,6))
    sb.boxplot(x="Department", y="Beds", data=melted)
    plt.title(title)
    plt.ylabel("Number of Beds")
    plt.xlabel("Department")
    plt.show()

plot_department_boxplots(all_hosp_disp, "Available Beds - Boxplot per Department")
plot_department_boxplots(all_hosp_norm, "Occupied Beds - Boxplot per Department")


