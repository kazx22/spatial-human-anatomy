[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18928352.svg)](https://doi.org/10.5281/zenodo.18928352)

# Spatial Human Anatomy — Biomedical NER Benchmarking for Clinical Knowledge Extraction

This repository presents a **biomedical named entity recognition (NER) benchmarking pipeline** designed to extract structured clinical knowledge from text and support future **clinical knowledge graph construction**.

The project focuses on identifying **disease** and **chemical** entities from biomedical text using a combination of classical and transformer-based NER models. It also introduces a **candidate gold** strategy based on model agreement and evaluates system outputs against the **human-annotated BC5CDR benchmark dataset**.

The long-term aim of this work is to support **structured biomedical knowledge extraction**, which can later be extended toward **entity normalization, relation extraction, and spatial human anatomy knowledge graphs**.

---

## Overview

Clinical and biomedical narratives contain valuable information, but most of that information is stored as **unstructured text**. Before building any downstream system such as a knowledge graph, reasoning engine, or decision-support pipeline, the text must first be converted into structured entities.

This project builds a reproducible workflow for:

- loading and preparing biomedical text data
- extracting disease and chemical entities with multiple NER models
- aggregating predictions across systems
- constructing a **candidate gold** dataset from model agreement
- benchmarking model performance using BIO-tag evaluation with `seqeval`
- comparing results against both **pseudo-gold** and **human gold** references

---

## Project Goals

The main goals of this repository are:

- benchmark multiple biomedical NER systems on a public dataset
- compare classical and transformer-based approaches
- examine the usefulness of **candidate gold labels** created through agreement voting
- measure model behaviour using precision, recall, and F1-score
- prepare a structured pipeline that can later support **clinical knowledge graph construction**

---

## Dataset

### BC5CDR (BioCreative V Chemical-Disease Relation Dataset)

This project uses the **BC5CDR dataset**, a widely used benchmark in biomedical NLP.

BC5CDR provides:

- manually annotated **disease** entities
- manually annotated **chemical** entities
- a reliable gold-standard benchmark for evaluation

Using BC5CDR allows the project to move beyond weakly curated or scraped clinical data and evaluate against a **published human-annotated biomedical dataset**.

---

## Models Included

The current benchmarking pipeline compares four biomedical NER approaches.

### 1. SciSpacy
Model: `en_ner_bc5cdr_md`

Characteristics:
- fast inference
- strong recall
- lower precision than stricter transformer-based systems

### 2. PubMedBERT
Model: `BiomedNLP-PubMedBERT`

Characteristics:
- transformer-based biomedical language model
- context-aware predictions
- stronger precision than recall in some settings
- slower inference than lightweight alternatives

### 3. ClinicalBERT
Model: `bert-base-uncased_clinical-ner`

Characteristics:
- trained for clinical-style text
- strong recall
- tends to over-predict entities in this pipeline

### 4. BioELECTRA
Model: `d4data/biomedical-ner-all`

Characteristics:
- balanced precision-recall behaviour
- efficient transformer inference
- strong overall performance in this comparison

---

## Candidate Gold Construction

A key contribution of this repository is the construction of a **candidate gold** dataset.

In many real-world biomedical or clinical settings, fully annotated data is limited or expensive to obtain. To simulate a useful supervision signal, this project aggregates predictions from multiple models and accepts only entities with sufficient agreement.

### Voting strategy

Predictions from the following four systems are combined:

- SciSpacy
- PubMedBERT
- ClinicalBERT
- BioELECTRA

An entity is represented by the tuple:

```text
(row_id, text, start_char, end_char, label)
```

An entity is accepted into the candidate gold file only if:

```text
at least 3 models agree
```

This produces:

```text
candidate_gold_entities.jsonl
```

This strategy prioritises **high-confidence labels**, even if some true entities are missed.

---

## Evaluation Method

Evaluation follows a standard biomedical NER benchmarking workflow.

### Steps

1. Load documents and model predictions
2. Group predictions by `row_id`
3. Convert character spans into BIO labels
4. Evaluate predictions with `seqeval`
5. Compare outputs against:
   - candidate gold labels
   - human gold labels from BC5CDR

### BIO conversion example

Text:

```text
The patient has asthma and takes aspirin
```

Tokens:

```text
["The", "patient", "has", "asthma", "and", "takes", "aspirin"]
```

Labels:

```text
["O", "O", "O", "B-DISEASE", "O", "O", "B-CHEMICAL"]
```

---

## Pipeline Summary

```text
BC5CDR Documents
↓
Text Preparation
↓
Biomedical Named Entity Recognition
├── SciSpacy
├── PubMedBERT
├── ClinicalBERT
└── BioELECTRA
↓
Prediction Aggregation
↓
Candidate Gold Construction (≥ 3 of 4 models agree)
↓
BIO Conversion
↓
Benchmark Evaluation (seqeval)
↓
Structured Biomedical Entities
```

---

## Results

### Candidate Gold vs Human Gold

| Comparison | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Candidate Gold vs Human Gold | ~0.99 | ~0.29 | ~0.45 |

### Interpretation

These results suggest that the candidate gold labels are:

- **highly precise** — most accepted entities are correct
- **low in recall** — many true entities are not captured

This is expected from a majority-vote agreement strategy. In other words, candidate gold works well as a **high-confidence pseudo-label source**, but it does not replace a complete human-annotated benchmark.

---

### Model Performance vs Candidate Gold

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| SciSpacy | 0.32 | 0.99 | 0.48 |
| PubMedBERT | 0.15 | 0.81 | 0.25 |
| ClinicalBERT | 0.22 | 0.88 | 0.35 |
| BioELECTRA | 0.31 | 0.81 | 0.45 |

### Key observations

- **SciSpacy** achieves the highest recall but lower precision.
- **PubMedBERT** underperforms in this setup, especially for disease detection.
- **ClinicalBERT** captures many entities but tends to over-predict.
- **BioELECTRA** provides the best overall balance between precision and recall.

---

### Per-label Performance

| Model | Label | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| SciSpacy | DISEASE | 0.2792 | 0.9843 | 0.4350 |
| SciSpacy | CHEMICAL | 0.3555 | 0.9939 | 0.5237 |
| PubMedBERT | DISEASE | 0.0595 | 0.5702 | 0.1078 |
| PubMedBERT | CHEMICAL | 0.3590 | 0.9834 | 0.5260 |
| ClinicalBERT | DISEASE | 0.1581 | 0.8622 | 0.2672 |
| ClinicalBERT | CHEMICAL | 0.2917 | 0.8968 | 0.4402 |
| BioELECTRA | DISEASE | 0.2482 | 0.8073 | 0.3797 |
| BioELECTRA | CHEMICAL | 0.3829 | 0.8145 | 0.5209 |

---

## Runtime Analysis

| Model | Average time per note |
|---|---:|
| SciSpacy | 0.0238 s |
| PubMedBERT | 2.7074 s |
| ClinicalBERT | 0.1502 s |
| BioELECTRA | 0.1324 s |

### Runtime insight

- **SciSpacy** is the fastest model by a large margin.
- **PubMedBERT** is the slowest in this benchmark.
- **ClinicalBERT** and **BioELECTRA** provide a more practical trade-off between runtime and predictive performance.

---

## Visualisations

### Candidate Gold vs Human Gold
![Candidate vs Human](figure/candidate_vs_human_performance.png)

### Model Performance Comparison
![Model Performance](figure/model_performance_comparison.png)

### Runtime Comparison
![Runtime Comparison](figure/model_runtime_comparison.png)

### Per-label Performance
![Per-label Performance](figure/per_label_performance.png)

---

## Confusion Matrices

### Candidate Gold vs Human Gold
![Candidate Gold vs Human Gold](figure/confusion_matrix_candidate_vs_human.png)

### SciSpacy
![SciSpacy Confusion Matrix](figure/confusion_matrix_scispacy.png)

### PubMedBERT
![PubMedBERT Confusion Matrix](figure/confusion_matrix_pubmedbert.png)

### ClinicalBERT
![ClinicalBERT Confusion Matrix](figure/confusion_matrix_clinicalbert.png)

### BioELECTRA
![BioELECTRA Confusion Matrix](figure/confusion_matrix_bioelectra.png)

---

## Repository Structure

```text
data/
├── raw/
│   └── bc5cdr/
├── processed/
│   └── bc5cdr/
│       ├── docs.jsonl
│       ├── gold_bio.jsonl
│       ├── candidate_gold_entities.jsonl
│       └── model_outputs/

figure/

src/
├── graph.py
├── evaluate_bc5cdr.py
├── candidate_gold_bc5cdr.py
├── ner_pipelines/
```

---

## Installation

Install the project dependencies:

```bash
pip install -r requirements/base.txt
```

If additional environment-specific requirements are needed, install those as required by your setup.

---

## How to Run

### 1. Run the preprocessing / pipeline stage

```bash
python -m src.pipeline
```

### 2. Run each NER system

```bash
python -m src.nerspacey
python -m src.biobert_bc5cdr
python -m src.clinicalbert_bc5cdr
python -m src.bioelectra_bc5cdr
```

### 3. Build candidate gold

```bash
python -m src.candidate_gold_bc5cdr
```

### 4. Run evaluation

```bash
python -m src.evaluate_bc5cdr
```

---

## Why This Project Matters

This repository demonstrates several important ideas in biomedical NLP:

- how to benchmark multiple biomedical NER systems in a reproducible way
- how pseudo-labels can be built through agreement voting
- how precision and recall trade-offs differ across biomedical NER models
- how structured entity extraction can serve as the first stage of a broader clinical knowledge graph pipeline

---

## Future Work

Planned extensions include:

- entity normalization with resources such as **UMLS** or **SNOMED CT**
- relation extraction between biomedical entities
- knowledge graph construction for structured clinical reasoning
- graph-based learning over extracted biomedical concepts
- multimodal biomedical learning pipelines

---

## Citation

If you use this repository, dataset preparation, or benchmarking workflow in your work, please cite the associated Zenodo record:

```bibtex
@misc{spatial_human_anatomy_zenodo,
  title        = {Spatial Human Anatomy},
  author       = {Kazi Alif},
  year         = {2026},
  doi          = {10.5281/zenodo.18928352},
  url          = {https://doi.org/10.5281/zenodo.18928352}
}
```

---

## Acknowledgements

This repository builds on publicly available biomedical NLP tools and benchmark datasets, including:

- **BC5CDR**
- **SciSpacy**
- **PubMedBERT**
- **ClinicalBERT**
- **BioELECTRA**

