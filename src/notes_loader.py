from pathlib import Path
import pandas as pd


def load_notes(path: str = "data/raw/mtsamples.csv") -> pd.DataFrame:
    note_path = Path(path)

    if not note_path.is_file():
        raise FileNotFoundError(f"Notes file not found at {note_path}")

    df = pd.read_csv(note_path)
    print(f"Loaded {len(df)} notes from {note_path}")
    print(f"shape: {df.shape}")
    print(f"columns: {df.columns.tolist()}")

    if "transcription" not in df.columns:
        raise ValueError("Expected 'transcription' column not found in the notes data")

    df = df.rename(columns={"transcription": "note_text"})
    df = df[["note_text"]].copy()

    return df
