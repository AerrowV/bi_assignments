# Data Exploration and Visualisation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import shapiro

# --- Load ---
def load_wine(path):
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, sep=",")
    else:
        df = pd.read_excel(path)
        if any(str(c).lower().startswith("unnamed") for c in df.columns):
            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)
        return df

red_df = load_wine("C:/Users/Asim/Downloads/MP1/MP2/winequality-red.xlsx")
white_df = load_wine("C:/Users/Asim/Downloads/MP1/MP2/winequality-white.xlsx")
wineqt_df = load_wine("C:/Users/Asim/Downloads/MP1/MP2/WineQT.csv")

# ---Clean ---
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
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()

    df = df.drop_duplicates().reset_index(drop=True)

    return df

red_df = clean(red_df)
white_df = clean(white_df)
wineqt_df = clean(wineqt_df)

red_df["type"] = "Red"
white_df["type"] = "White"
wineqt_df["type"] = "Public Wine"

# --- Aggregate ---
all_wine = pd.concat([red_df, white_df, wineqt_df], ignore_index=True)

print("Red:", red_df.shape)
print("White:", white_df.shape)
print("Public Wine Quality", wineqt_df.shape)
print("All:", all_wine.shape)
print(all_wine.head())

# --- Basic info ---
print(all_wine.info())
desc = all_wine.groupby("type").describe()
desc.columns = ["_".join(col).strip() for col in desc.columns.values]
print(desc.head())
# desc.to_excel("wine_descriptive_stats.xlsx")

# --- Distribution of target variable ---
plt.figure(figsize=(8,5))
sb.countplot(data=all_wine, x="quality", hue="type", palette="Set2")
plt.title("Distribution of Wine Quality Scores")
plt.show()

# --- Visual histogram for key features (positively skewed)
all_wine[["alcohol", "residual_sugar", "volatile_acidity"]].hist(bins = 30, figsize = (10,5))
plt.suptitle("Histograms of Key Features")
plt.show

# --- Correlation heatmap ---
plt.figure(figsize=(12,8))
corr = all_wine.drop(columns=["type"]).corr()
sb.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap of Features")
plt.show()

# --- Boxplots for key predictors vs quality ---
plt.figure(figsize=(12,6))
sb.boxplot(data=all_wine, x="quality", y="alcohol", hue="type", palette="Set3")
plt.title("Alcohol vs Quality by Type")
plt.show()

plt.figure(figsize=(12,6))
sb.boxplot(data=all_wine, x="quality", y="volatile_acidity", hue="type", palette="Set1")
plt.title("Volatile Acidity vs Quality by Type")
plt.show()

# The dataset was already numeric and suitable for analysis. Apart from column name cleaning and removal of duplicates/IDs, no additional transformations were necessary.

# Discretize alcohol into bins
all_wine["alcohol_bin"] = pd.cut(all_wine["alcohol"], bins=5,
                                 labels=["Very Low","Low","Medium","High","Very High"])

# Discretize quality into categories (Low: 3-4, Medium: 5-6, High: 7+)
all_wine["quality_bin"] = pd.cut(all_wine["quality"],
                                 bins=[0,4,6,10],
                                 labels=["Low","Medium","High"])

for col in ["alcohol","residual_sugar","volatile_acidity"]:
    stat, p = shapiro(all_wine[col])
    print(f"{col}: stat={stat:.3f}, p={p:.3f}")
