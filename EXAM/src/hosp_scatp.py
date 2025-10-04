import matplotlib.pyplot as plt
import seaborn as sb 

def plot_hosp_scatter(df, x_col, y_col, hue=None, palette="Set1", title="", alpha=0.6):
    df_num = df.drop(columns=["År", "Måned"], errors="ignore")
    plt.figure(figsize=(12,6))
    sb.scatterplot(data=df_num, x=x_col, y=y_col, hue=hue, palette=palette, alpha=alpha)
    
    if title is None:
        title = f"{y_col.capitalize()} vs {x_col.capitalize()}"
    plt.title(title)
    plt.show()