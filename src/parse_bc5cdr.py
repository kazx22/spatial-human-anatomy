"""
parse_bc5cdr.py — parses the raw BC5CDR PubTator files into JSONL.

BC5CDR is distributed in PubTator format: each document is a block of
tab/pipe-delimited lines separated by blank lines.  The first two lines are
the title and abstract; subsequent lines are entity annotations.

This script converts all three splits (train, dev, test) into two JSONL files
each:
  bc5cdr_{split}_docs.jsonl     — one record per document: {row_id, full_text}
  bc5cdr_{split}_entities.jsonl — one record per entity span

full_text is title + " " + abstract.  Character offsets in the annotation
lines are relative to this concatenated string, which is how BC5CDR defines
them, so no offset adjustment is needed.

Pipeline position: first step; all other scripts consume the output of this one.
"""

from pathlib import Path
import json

LABEL_MAP = {
    "Disease": "DISEASE",
    "Chemical": "CHEMICAL",
}


def save_jsonl(records, output_file):
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(records)} records to {output_file}...")
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_bc5cdr(file_path):
    """
    Parse a single BC5CDR PubTator file into doc and entity lists.

    PubTator format per document:
      {pmid}|t|{title}
      {pmid}|a|{abstract}
      {pmid}\t{start}\t{end}\t{text}\t{label}\t{mesh_id}
      (blank line)

    full_text is constructed as title + " " + abstract, which matches the
    offset convention used by the BC5CDR annotations — the offsets in the
    annotation lines refer to this concatenated string.

    Entity lines with fewer than 5 tab-separated fields are skipped; these
    are typically relation annotations (CID lines) rather than entity spans.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    documents = content.strip().split("\n\n")
    all_entities = []
    all_docs = []

    for doc in documents:
        lines = doc.strip().split("\n")
        if len(lines) < 2:
            continue

        title_line = lines[0]
        abstract_line = lines[1]

        pmid = title_line.split("|")[0]
        title = title_line.split("|t|")[1]
        abstract = abstract_line.split("|a|")[1]

        # The single space separator between title and abstract is intentional —
        # BC5CDR's character offsets assume exactly this join.
        full_text = title + " " + abstract
        doc_record = {
            "row_id": int(pmid),
            "full_text": full_text,
        }
        all_docs.append(doc_record)

        for line in lines[2:]:
            parts = line.split("\t")
            if len(parts) < 5:
                # CID (chemical-disease relation) lines have fewer fields; skip them
                continue

            start_char = int(parts[1])
            end_char = int(parts[2])
            entity_text = parts[3]
            label = parts[4]

            # Normalise label casing to match the rest of the pipeline
            label = LABEL_MAP.get(label, label)

            entity = {
                "row_id": int(pmid),
                "text": entity_text,
                "start_char": start_char,
                "end_char": end_char,
                "label": label,
            }
            all_entities.append(entity)

    print(f"Parsed {len(all_entities)} entities from {file_path}")
    return all_docs, all_entities


if __name__ == "__main__":
    base_input = Path("data/raw/bc5cdr")
    base_output = Path("data/processed/bc5cdr")

    files = {
        "train": "CDR_TrainingSet.PubTator.txt",
        "dev": "CDR_DevelopmentSet.PubTator.txt",
        "test": "CDR_TestSet.PubTator.txt",
    }

    for key, filename in files.items():
        input_file = base_input / filename

        print(f"\n--- Processing {key.upper()} ---")
        print("Input exists:", input_file.exists())
        print("Input path:", input_file)

        docs, entities = parse_bc5cdr(input_file)

        docs_output_file = base_output / f"bc5cdr_{key}_docs.jsonl"
        entities_output_file = base_output / f"bc5cdr_{key}_entities.jsonl"

        save_jsonl(docs, docs_output_file)
        save_jsonl(entities, entities_output_file)

        print(f"Saved {len(docs)} docs to {docs_output_file}")
        print(f"Saved {len(entities)} entities to {entities_output_file}")
