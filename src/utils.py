import json


def clean_text(txt: str) -> str:
    return txt.strip().lower()


def remove_TEST(input_file: str, output_file: str):
    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:

        for line in infile:
            entity = json.loads(line)

            if entity["label"] != "TEST":
                outfile.write(json.dumps(entity) + "\n")
