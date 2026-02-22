from pathlib import Path
from pandas import pd


def load_notes(path: str = "data/raw/mtsamples.csv") -> pd.DataFrame:
    note_path = Path(path)

    if not note_path.is_file():
        raise FileNotFoundError(f"Notes file not found at {note_path}")

    df = pd.read_csv(note_path)
    print(f"Loaded {len(df)} notes from {note_path}")
    print(f"shape: {df.shape}")
    print(f"columns: {df.columns.tolist()}")

    return df
