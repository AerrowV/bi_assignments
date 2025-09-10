# Data Exploration and Visualisation

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import shapiro

# --- Parameters ---

"""
    Plot histograms for numeric columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data.
    cols : list, optional
        List of column names to plot. If None, all numeric columns are used.
    bins : int, optional
        Number of bins for the histogram.
    kde : bool, optional
        Whether to overlay a KDE curve.
    exclude : list, optional
        List of column names to exclude (useful for 'quality' or derived bins).
"""

# --- Task 1 + 4: Load ---
def load_wine(path):
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, sep=",")
    else:
        df = pd.read_excel(path)
        if any(str(c).lower().startswith("unnamed") for c in df.columns):
            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)
        return df

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
DATA_DIR = os.path.join(BASE_DIR, "data")             

red_df = load_wine(os.path.join(DATA_DIR, "winequality-red.xlsx"))
white_df = load_wine(os.path.join(DATA_DIR, "winequality-white.xlsx"))
wineqt_df = load_wine(os.path.join(DATA_DIR, "WineQT.csv"))

# --- Task 2 + 4: Clean ---
def clean(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    cols = [
        "fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides","free_sulfur_dioxide","total_sulfur_dioxide",
        "density","ph","sulphates","alcohol","quality"
    ]
    df = df[[c for c in cols if c in df.columns]]

    for c in df.columns:
        # Values that cant become numeric becomes 'NaN'
        df[c] = pd.to_numeric(df[c], errors="coerce")
        # Here rows with 'NaN' gets deleted
    df = df.dropna()
    # Removes duplicates found in the datasets
    df = df.drop_duplicates().reset_index(drop=True)

    return df

red_df = clean(red_df)
white_df = clean(white_df)
wineqt_df = clean(wineqt_df)

red_df["type"] = "Red"
white_df["type"] = "White"
wineqt_df["type"] = "Public Wine"

# --- Task 3 + 4: Aggregate ---
all_wine = pd.concat([red_df, white_df, wineqt_df], ignore_index=True)

print("Red:", red_df.shape)
print("White:", white_df.shape)
print("Public Wine Quality", wineqt_df.shape)
print("All:", all_wine.shape)
# print(all_wine.head()) // Shows the first 5 rows of the DataFrame, so you can confirm it merged correctly.

# --- Task 5: Explore Features and Visualisation---

def plot_correlation_heatmap(df, title="Correlation Heatmap"):

    plt.figure(figsize=(12,8))
    corr = df.drop(columns=["type"], errors="ignore").corr()  # safely drop if 'type' exists
    sb.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title(title)
    plt.show()


plot_correlation_heatmap(red_df)
plot_correlation_heatmap(white_df)
plot_correlation_heatmap(wineqt_df)
plot_correlation_heatmap(all_wine)

# --- Boxplots for Dependent Features ---

def plot_feature_boxplot(df, x_col, y_col, palette="Set1", title=None, hue=None):
  
    plt.figure(figsize=(12,6))
    sb.boxplot(data=df, x=x_col, y=y_col, hue=hue, palette=palette)
    if title is None:
        if hue:
            title = f"{y_col.capitalize()} vs {x_col.capitalize()} by {hue.capitalize()}"
        else:
            title = f"{y_col.capitalize()} vs {x_col.capitalize()}"
    plt.title(title)
    plt.show()

# ---  Scatterplots for Dependent Features ---

def plot_feature_scatter(df, x_col, y_col, hue=None, palette="Set1", title=None, alpha=0.6):
  
    plt.figure(figsize=(12,6))
    sb.scatterplot(data=df, x=x_col, y=y_col, hue=hue, palette=palette, alpha=alpha)
    if title is None:
        title = f"{y_col.capitalize()} vs {x_col.capitalize()}"
    plt.title(title)
    plt.show()

# --- Task 6: Data Transformation (Encoding & Discretization)
def transform_wine_data(df):
    df = df.copy()

    # --- Encode categorical ---
    if "type" in df.columns:
        df["type_code"] = df["type"].astype("category").cat.codes

    # --- Discretize numeric features ---
    # Example: alcohol into 5 bins
    df["alcohol_bin"] = pd.cut(
        df["alcohol"], bins=5,
        labels=["Very Low", "Low", "Medium", "High", "Very High"]
    )

    # Example: quality into 3 custom bins
    df["quality_bin"] = pd.cut(
        df["quality"], bins=[0, 4, 6, 10],
        labels=["Low", "Medium", "High"]
    )

    return df

all_wine = transform_wine_data(all_wine)
print(all_wine.head())


# --- Task 7: Descriptive Statistics & Normality Check ---

# 1. Descriptive statistics
print(all_wine.info())
desc = all_wine.groupby("type").describe()
desc.columns = ["_".join(col).strip() for col in desc.columns.values]
print(desc.head())
# desc.to_excel("wine_descriptive_stats.xlsx") // This would save the descriptive stats to an Excel file for reporting.

# 2. Histograms of all numeric features (visual normality check)
def plot_histograms(df, cols=None, bins=30, kde=True, exclude=None):
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()
    if exclude:
        cols = [c for c in cols if c not in exclude]

    for col in cols:
        plt.figure(figsize=(7,4))
        sb.histplot(df[col].dropna(), bins=bins, kde=kde)
        plt.title(f"Histogram of {col.replace('_',' ').title()}")
        plt.xlabel(col.replace("_"," ").title())
        plt.ylabel("Count")
        plt.show()

plot_histograms(all_wine)

# 3. Shapiro-Wilk test for selected features
from scipy.stats import shapiro

def shapiro_report(series: pd.Series, name: str):
    x = series.dropna().astype(float)
    if len(x) > 5000:  # Shapiro requires <= 5000
        x = x.sample(5000, random_state=42)
    stat, p = shapiro(x)
    conclusion = "NOT normal" if p < 0.05 else "approximately normal"
    print(f"{name:>18s}: W={stat:.3f}, p={p:.3g} -> {conclusion}")

for col in all_wine.select_dtypes(include="number").columns:
    shapiro_report(all_wine[col], col)

# --- Task 8: ---

# A:

plot_feature_boxplot(red_df, "fixed_acidity", "density", palette="Set1", hue="type")
# The first boxplot shows that a higher density increases the fixed acidity.
# The distribution exhibits a cluster of outliers around the central values, suggesting that deviations are not confined to the extremes.

plot_feature_boxplot(red_df, "fixed_acidity", "ph", palette="Set1", hue="type")
# Lower pH values are associated with higher fixed acidity, reflecting the expected negative correlation between acidity and pH.
# The middle portion of the chart shows the highest density of outliers, suggesting irregular variation in that region.

plot_feature_boxplot(white_df, "alcohol", "density", palette="Set3", hue="type")
# The boxplot indicates that low-density wines have higher average alcohol levels.
# The distribution of alcohol shows that outliers are concentrated in the upper range, indicating that some wines have unusually high alcohol content relative to the majority of samples.

plot_feature_boxplot(white_df, "density", "residual_sugar", palette="Set3", hue="type")
# Higher residual sugar corresponds to higher density.
# The outliers are dispersed throughout all sugar levels, suggesting consistent variability across the dataset.

# B:
# White wine has a higher average level of quality compared to red wine. Where white wine is 5,85483463771775 and red win being 5,62325239146431

# C:
# White wine has a higher average level of alcohol compared to red wine. White wine being 10,5893579062526 and red wine being 10,4323154280108

# D:
# White wine has a way higher average quantity of redisual sugar compared to red wine. White wine being 5,91481949002777 and red wine being 2,5233995584989

# E:
plot_feature_scatter(all_wine, "alcohol", "quality", palette="Set1", hue="type")
# The higher the alcohol content, the better the perceived quality.
plot_feature_scatter(all_wine, "residual_sugar", "quality", palette="Set1", hue="type")
# Lower residual sugar levels corresponds to lower quality

# --- Task 9: Which other questions might be of interest for the wine consumers and which of wine distributers? ---

# Consumers:
# 1. If price data were available, is paying more strongly correlated with higher quality? (Price vs quality)
# 2. Which chemical features influence taste the most? 
# 3. Which wines tend to have lower sulphates / lower alcohol? Is there a “healthier” option within red vs white? (Health related)
# 4. Which country does the wine stem from and does the origin of the wine have a correlation with the quality of the wine?

# Wine distributers:
# 1. What attributes drive higher-rated wines? Should they focus production on wines with higher alcohol and lower volatile acidity?
# 2. Is there more variation in white wine quality vs red? (helps standardize production). 

# --- Task 10: Binning pH and checking densities ---

# Split into 5 bins
all_wine["ph_bin_5"] = pd.cut(all_wine["ph"], bins=5)

# For each bin, calculate average density
density_by_ph5 = all_wine.groupby("ph_bin_5")["density"].mean()
print("Mean density by 5 pH bins:\n", density_by_ph5)

# Which bin has highest density?
highest_5 = density_by_ph5.idxmax(), density_by_ph5.max()
print("Highest (5 bins):", highest_5)

# Split into 10 bins
all_wine["ph_bin_10"] = pd.cut(all_wine["ph"], bins=10)

density_by_ph10 = all_wine.groupby("ph_bin_10")["density"].mean()
print("Mean density by 10 pH bins:\n", density_by_ph10)

# Which bin has highest density?
highest_10 = density_by_ph10.idxmax(), density_by_ph10.max()
print("Highest (10 bins):", highest_10)

# --- Task 11: ---
# Select only numeric columns
num_df = all_wine.select_dtypes(include="number")

# Correlation matrix (Pearson)
corr = num_df.corr()

# Show as dataframe sorted by correlation with quality
print("Correlation with Quality:\n", corr["quality"].sort_values(ascending=False))

# Heatmap
plt.figure(figsize=(12,8))
sb.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap of Wine Attributes")
plt.show()

# --- Task 12: Explore and remove outliers. ---


