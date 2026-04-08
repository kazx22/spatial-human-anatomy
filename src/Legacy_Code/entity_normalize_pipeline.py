import json
from src.entity_normalize import normalize_disease_name


def normalize_entities(input_file: str, output_file: str):

    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:

        for line in infile:
            entity = json.loads(line)

            if entity["label"] == "DISEASE":
                norm = normalize_disease_name(entity["text"])

                entity["canonical"] = norm["canonical"]
                entity["norm_method"] = norm["method"]
                entity["norm_confidence"] = norm["confidence"]

            outfile.write(json.dumps(entity) + "\n")
