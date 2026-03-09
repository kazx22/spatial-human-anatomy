def filter_entities(entities):
    filtered = []

    for ent in entities:
        text = ent["word"].strip()

        if not text:
            continue

        if text in {".", ",", ";", ":", "(", ")"}:
            continue

        if len(text) < 3:
            continue

        filtered.append(ent)

    return filtered
