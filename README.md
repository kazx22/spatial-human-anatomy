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

The repository also includes experimental support for transformer-based biomedical NER using models from the HuggingFace Transformers library.

### Models Used

- Disease detection:  
  `sarahmiller137/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-ft-ncbi-disease`

- Chemical / medication detection:  
  `OpenMed/OpenMed-NER-ChemicalDetect-PubMed-335M`

- Additional comparison model:  
  `samrawal/bert-base-uncased_clinical-ner`

Because no single transformer model reliably detects both disease and chemical entities in clinical notes, separate models are used for each entity type. The outputs of these models are then combined and post-processed to produce a unified list of biomedical entities.

This setup allows comparison between:

- **SciSpacy biomedical NER** (`en_ner_bc5cdr_md`)
- **Transformer-based biomedical NER** (PubMedBERT + chemical detection models)
- **Clinical-domain transformer models** (ClinicalBERT)



## Experimental Findings

SciSpacy was more stable on long clinical notes

transformer models needed more preprocessing/post-processing

transformers were slower

SciSpacy is the main pipeline, transformers are comparison baselines

## How to Run

All scripts are executed from the root of the repository.

### 1. Run the Data Processing Pipeline

This step loads and cleans the clinical notes dataset and prepares the processed file used by the NER pipelines.

```bash
python -m src.run_pipeline
```
this will generate 

`data/processed/cleaned_notes.csv`

Run the SciSpacy NER Pipeline

This extracts biomedical entities using the SciSpacy model.

`python -m src.load_scispacy`

Output will be saved to:

`data/processed/ner/scispacy/entities.jsonl`

3. Run the BioBERT / PubMedBERT Transformer Pipeline

This pipeline combines a disease detection model and a chemical detection model to extract biomedical entities.

`python -m src.load_biobert`

Output will be saved to:

`data/processed/ner/transformers/biobert/biobert_entities.jsonl`

4. Run the ClinicalBERT NER Pipeline

This runs the ClinicalBERT-based NER model for comparison with SciSpacy and BioBERT pipelines.

`python -m src.load_clinicalbert`

Output will be saved to:

`data/processed/ner/transformers/clinicalbert/clinicalbert_entities.jsonl`

Notes

The pipelines expect the processed dataset located at:

`data/processed/cleaned_notes.csv`

# Project Report

A detailed report describing the NER pipeline implementation, model comparison, and experimental findings is available here:

[NER Model Comparison Report](docs/ner_model_comparison.pdf)

The report discusses:

- SciSpacy biomedical NER pipeline
- Transformer-based NER experiments (BioBERT / PubMedBERT / ClinicalBERT)
- Implementation challenges such as token limits and subword tokenization
- Runtime performance comparison between models

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
