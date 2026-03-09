def merge_word(entities):
    if not entities:
        return []

    merged = []
    curr = entities[0].copy()

    for entity in entities[1:]:
        if entity["word"].startswith("##") and entity["start"] == curr["end"]:
            curr["word"] += entity["word"].replace("##", "")
            curr["end"] = entity["end"]
            curr["score"] = max(curr["score"], entity["score"])
        else:
            merged.append(curr)
            curr = entity.copy()

    merged.append(curr)
    return merged
