from pathlib import Path

from src.notes_loader import load_notes
from src.notes_cleaner import clean_notes
from src.notes_normalize import normalize_text


def main():
    df = load_notes("data/raw/mtsamples.csv")
    df = clean_notes(df)
    df = normalize_text(df)

    output_path = Path("data/processed/cleaned_notes.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Saved cleaned dataset to: {output_path}")


if __name__ == "__main__":
    main()
