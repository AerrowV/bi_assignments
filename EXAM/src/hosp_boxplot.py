import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_hosp_boxplots(df):
    df.plot(kind='box', subplots=True, layout=(6,6), sharex=False, sharey=False,figsize=(18,36))
    plt.show()


