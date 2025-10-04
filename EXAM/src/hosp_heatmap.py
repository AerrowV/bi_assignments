import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np 

def plot_hosp_heatmap(df, title="Correlation Heatmap"):
    plt.figure(figsize=(20,12))

    df_num = df.select_dtypes(include=[np.number]).drop(columns=["År", "Måned"], errors="ignore")
    corr = df_num.corr()

    # Heatmap
    sb.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title(title)
    plt.show()