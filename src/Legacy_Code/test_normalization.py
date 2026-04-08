from src.entity_normalize import normalize_disease_name

samples = ["HTN", "heart attack", "asthma"]

for item in samples:
    print(item, "->", normalize_disease_name(item))
