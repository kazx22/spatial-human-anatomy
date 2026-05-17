def filter_by_confidence(entities, threshold=0.6):
    filtered = []

    for entity in entities:
        confidence = entity.get("confidence", 1.0)

        if confidence >= threshold:
            filtered.append(entity)

    return filtered


def label_aware_filter(
    entities,
    disease_threshold=0.5,
    chemical_threshold=0.7,
):
    filtered = []

    for entity in entities:
        label = entity["label"]
        confidence = entity.get("confidence", 1.0)

        if label == "DISEASE" and confidence >= disease_threshold:
            filtered.append(entity)

        elif label == "CHEMICAL" and confidence >= chemical_threshold:
            filtered.append(entity)

    return filtered
