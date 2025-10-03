import pandas as pd

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop Excel's extra index columns like "Unnamed: 0"
    df = df.loc[:, ~df.columns.astype(str).str.lower().str.startswith("unnamed")]

    # Clean column names (remove surrounding spaces + spaces/dashes inside)
    df.columns = [str(c).strip().replace(" ", "").replace("-", "") for c in df.columns]

    # Map common variants to canonical Danish headers
    cols_lower = {c.lower(): c for c in df.columns}
    def find_col(candidates):
        for cand in candidates:
            key = cand.lower()
            if key in cols_lower:
                return cols_lower[key]
        return None

    col_year  = find_col(["År", "Aar", "Ar"])
    col_month = find_col(["Måned", "Maned", "Maaned"])

    if col_year is None or col_month is None:
        raise KeyError(
            "Kolonnerne 'År'/'Måned' (eller varianter som Aar/Ar og Maned/Maaned) blev ikke fundet."
        )

    # Ensure numeric year/month and keep valid months
    df[col_year]  = pd.to_numeric(df[col_year], errors="coerce")
    df[col_month] = pd.to_numeric(df[col_month], errors="coerce")
    df = df.dropna(subset=[col_year, col_month])
    df[col_year]  = df[col_year].astype(int)
    df[col_month] = df[col_month].astype(int)
    df = df[df[col_month].between(1, 12)]

    # Build date
    df["Dato"] = pd.to_datetime(
        {"year": df[col_year], "month": df[col_month], "day": 1},
        errors="coerce"
    )
    df = df.dropna(subset=["Dato"]).sort_values("Dato").reset_index(drop=True)

    # Put Dato first
    other_cols = [c for c in df.columns if c != "Dato"]
    df = df[["Dato"] + other_cols]

    return df
