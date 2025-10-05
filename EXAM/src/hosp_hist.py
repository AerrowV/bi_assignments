import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_hosp_histogram(
    hosp: pd.DataFrame,
    title: str = "Histogram of Hospital Data",
    x_label: str = "X-Axis",
    y_label: str = "Y-Axis",
    drop_columns: list = None,
    color: str = None,
    alpha: float = 0.75,
    edgecolor: str = "black",
    font_size: int = 16,
    rect: list = [0, 0, 1, 0.96],
    show_skewer: bool = True,
):
    if drop_columns:
        hosp_num = hosp.drop(columns=drop_columns, errors="ignore")
    else:
        hosp_num = hosp.copy()

    hosp_num = hosp_num.select_dtypes(include=[np.number])

    axes = hosp_num.hist(color=color, alpha=alpha, edgecolor=edgecolor, bins=20)
    plt.suptitle(title, fontsize=font_size)

    for i, col in enumerate(hosp_num.columns):
        ax = np.ravel(axes)[i]
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Mean/Median vertical lines
        if show_skewer:
            mean_val = hosp_num[col].mean()
            median_val = hosp_num[col].median()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')

        ax.legend(fontsize=10)

    plt.tight_layout(rect=rect)
    plt.show()
