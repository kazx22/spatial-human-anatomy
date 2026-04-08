import re
import unicodedata


def normalize_text(df):

    df["note_text"] = df["note_text"].apply(lambda x: unicodedata.normalize("NFKC", x))

    df["note_text"] = df["note_text"].str.lower()

    df["note_text"] = df["note_text"].apply(lambda x: re.sub(r"\s+", " ", x))

    return df
