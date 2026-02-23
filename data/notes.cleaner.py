import pandas as pd


def clean_notes(df: pd.DataFrame) -> pd.DataFrame:
    df["note_text"] = df["note_text"].fillna("")
    df["note_text"] = df["note_text"].str.strip()

    df = df[df["note_text"] != ""].copy()

    df = df.reset_index(drop=True)

    return df
