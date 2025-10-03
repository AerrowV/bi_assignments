import os
import pandas as pd

def load_data(path):
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, sep=",")
    else:
        df = pd.read_excel(path)
        if any(str(c).lower().startswith("unnamed") for c in df.columns):
            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)
        return df

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  
DATA_DIR = os.path.join(BASE_DIR, "data")

kir_op = load_data(os.path.join(DATA_DIR, "kirurgi_operationer.xlsx"))
kir_sp = load_data(os.path.join(DATA_DIR, "kirurgi_sengepladser.xlsx"))
kir_vt = load_data(os.path.join(DATA_DIR, "kirurgi_ventetider.xlsx"))
