import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
):
  
    # Drop unwanted columns if specified
    if drop_columns:
        hosp_num = hosp.drop(columns=drop_columns, errors="ignore")
    else:
        hosp_num = hosp.copy()

    # Create histograms for all numeric columns
    axes = hosp_num.hist(color=color, alpha=alpha, edgecolor=edgecolor, bins=20)

    # Set overall figure title
    plt.suptitle(title, fontsize=font_size)

    # Customize all subplots
    for row in np.atleast_2d(axes):
        for ax in row:
            if ax:
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)

    plt.tight_layout(rect=rect)
    plt.show()


