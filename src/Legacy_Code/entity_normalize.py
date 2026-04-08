from src.utils import clean_text
import re

DISEASE_MAP = {
    "htn": "hypertension",
    "high blood pressure": "hypertension",
    "heart attack": "myocardial infarction",
    "mi": "myocardial infarction",
}


# def normalize_disease_name(disease_name: str) -> str:
#     cleaned_name = clean_text(disease_name)
#     return DISEASE_MAP.get(cleaned_name, cleaned_name)
def normalize_disease_name(disease_name: str) -> dict:
    cleaned_name = clean_disease_text(disease_name)

    if cleaned_name in DISEASE_MAP:
        return {
            "original": disease_name,
            "canonical": DISEASE_MAP[cleaned_name],
            "method": "dictionary",
            "confidence": 0.95,
        }

    return {
        "original": disease_name,
        "canonical": cleaned_name,
        "method": "fallback",
        "confidence": 0.60,
    }


def clean_disease_text(disease_name: str) -> str:
    cleaned_name = clean_text(disease_name)

    cleaned_name = re.sub(r"^complaint of\s+", "", cleaned_name)
    cleaned_name = re.sub(r",\s*past$", "", cleaned_name)
    cleaned_name = re.sub(r"\.+", "", cleaned_name)

    return cleaned_name
