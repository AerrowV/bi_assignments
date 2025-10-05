import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.dates as mdates

def plot_capacity_over_time(
    hosp,
    date_col="Dato",
    region_col="Region",
    y_cols=["Disponible_senge", "Normerede_senge"],
    title="Development of Surgical Capacity Over Time",
    date_format="%Y-%m",
    figsize=(18, 9)
):
    df = hosp.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    plt.figure(figsize=figsize)

    for y_col in y_cols:
        sns.lineplot(
            data=df,
            x=date_col,
            y=y_col,
            hue=region_col,
            marker="o"
        )

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45, ha="right")

    plt.title(title)
    plt.xlabel("Date (Month-Year)")
    plt.ylabel(y_cols)
    plt.legend(title=region_col, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
