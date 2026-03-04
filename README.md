# Spatial Human Anatomy – Clinical Text to Knowledge Graph

This project explores how clinical narratives can be transformed into structured representations that support spatial reasoning about the human body.

The goal is to convert unstructured clinical notes into machine-readable medical entities that can later be used to construct body-centric knowledge graphs and support graph-based learning methods.

The project currently focuses on building a reproducible biomedical NLP pipeline.

---

## Project Goals

Clinical notes contain valuable medical information but are typically stored as unstructured text. This project aims to:

- Extract biomedical entities from clinical narratives
- Normalize and structure medical text
- Prepare data for knowledge graph construction
- Enable future graph-based learning methods for healthcare data

---

## Project Pipeline

The current pipeline processes raw clinical notes and extracts structured biomedical entities.


Raw Clinical Notes
↓
Text Cleaning
↓
Text Normalization
↓
Biomedical Named Entity Recognition
↓
Structured Entity Output



## Data Processing

The preprocessing pipeline loads raw clinical notes and performs multiple normalization steps including:

- Unicode normalization
- Lowercasing
- Whitespace standardization

The cleaned dataset is saved to:


data/processed/cleaned_notes.csv


Run the preprocessing pipeline:


python -m src.run_pipeline


---

## Biomedical Named Entity Recognition

The project currently uses **SciSpacy** biomedical models to identify medical entities from clinical text.

Model used:


en_ner_bc5cdr_md


This model identifies biomedical entities such as:

- Diseases
- Chemicals
- Medical conditions

NER results are saved as structured JSON lines:


data/processed/ner/scispacy/entities.jsonl


Each entity record includes:

- row_id
- entity text
- entity label
- character offsets

Example:


{
"row_id": 0,
"text": "diabetes",
"label": "DISEASE",
"start_char": 34,
"end_char": 42
}


---

## Transformer-Based NER Experiments

The repository also includes experimental support for transformer-based biomedical NER using BioBERT.

Model:


Ishan0612/biobert-ner-disease-ncbi


This allows comparison between:

- SciSpacy biomedical NER
- Transformer-based NER models

---

## Future Work

This project is the first stage of a larger research direction focused on spatial representations of human anatomy.

Future development will include:

- Entity normalization to biomedical ontologies
- Relationship extraction between symptoms and anatomical regions
- Construction of body-centric medical knowledge graphs
- Graph-based machine learning methods (Graph Neural Networks)

---

## Research Direction

The long-term goal of this project is to explore how clinical narratives can be transformed into structured representations that capture spatial relationships within the human body, enabling new approaches for machine learning in healthcare.

---

## Author

Kazi Alif  
MSc Computer Science (Distinction)  
Research interests: Machine Learning, Biomedical NLP, Knowledge Graphs
