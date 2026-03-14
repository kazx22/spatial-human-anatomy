
# Spatial Human Anatomy — Biomedical NER from Clinical Notes

This project explores how **clinical narratives can be transformed into structured biomedical information** that can later support **knowledge graph construction and spatial reasoning over the human body**.

Clinical notes contain large amounts of valuable medical knowledge, but they are typically stored as **unstructured text**. Extracting biomedical entities such as **diseases** and **chemicals** is a key step toward building structured medical knowledge systems.

This repository implements a **reproducible biomedical NLP pipeline** that processes clinical notes, extracts biomedical entities using multiple NER models, and evaluates their performance through benchmarking.

---

# Project Goals

The main objectives of this project are:

- Extract biomedical entities from clinical narratives
- Compare different biomedical Named Entity Recognition (NER) systems
- Construct a candidate gold dataset using model agreement
- Benchmark NER model performance
- Prepare structured biomedical data for future knowledge graph construction

The project currently focuses on **biomedical entity extraction and model evaluation**, which forms the foundation for future work on **clinical knowledge graphs and graph-based learning**.

---

# Project Pipeline

The repository implements the following processing pipeline:

```

Raw Clinical Notes
↓
Text Cleaning
↓
Text Normalization
↓
Biomedical Named Entity Recognition
├── SciSpacy
├── BioBERT
└── ClinicalBERT
↓
Entity Aggregation
↓
Candidate Gold Construction
↓
Benchmark Evaluation
↓
Structured Biomedical Entities

```

Each stage of the pipeline converts raw clinical text into progressively more structured information.

---

# Biomedical NER Models

Three biomedical NER systems are evaluated in this project.

---

## SciSpacy

Model:

```

en_ner_bc5cdr_md

```

SciSpacy is a biomedical NLP model trained to recognize **disease** and **chemical** entities.

Characteristics:

- Fast inference
- Handles long clinical notes well
- Often achieves high recall but lower precision

SciSpacy is well suited for large clinical datasets due to its efficiency.

---

## BioBERT

Model:

```

Ishan0612/biobert-ner-disease-ncbi

```

BioBERT is a transformer-based biomedical language model pretrained on **PubMed biomedical literature**.

Characteristics:

- Context-aware entity recognition
- Higher precision compared to rule-based models
- Slower inference due to transformer computation

BioBERT is particularly useful when contextual understanding of biomedical text is important.

---

## ClinicalBERT

Model:

```

samrawal/bert-base-uncased_clinical-ner

```

ClinicalBERT is a BERT variant trained specifically on **clinical narratives**.

Characteristics:

- Strong recall
- Tends to over-predict entities
- Requires preprocessing support for long clinical notes

ClinicalBERT is designed for clinical environments where medical terminology appears frequently.

---

# Candidate Gold Dataset Construction

The dataset used in this project does not contain manual annotations.

To enable model evaluation, a **candidate gold dataset** was created using **majority voting across models**.

### Process

1. Load entity predictions from:

- SciSpacy
- BioBERT
- ClinicalBERT

2. Clean ClinicalBERT output by removing `TEST` labels.

3. Construct a voting table using entity spans:

```

(row_id, entity_text, start_char, end_char, label)

```

4. Accept entities predicted by **at least two models**.

The resulting dataset:

```

candidate_gold_entities.jsonl

```

This dataset serves as a **pseudo-gold standard** for benchmarking.

---

# Benchmarking Method

The benchmark measures how closely each model matches the candidate gold dataset.

### Evaluation Steps

1. Load clinical notes dataset
2. Load entity predictions from each model
3. Load candidate gold entities
4. Group entities by `row_id`
5. Convert entity spans into **BIO label sequences**
6. Evaluate predictions using **seqeval**

---

# BIO Label Conversion

NER models output entities as **character spans**:

```

start_char
end_char

```

For sequence evaluation, spans must be converted to **BIO token labels**.

### Example

Clinical text:

```

The patient has asthma and takes aspirin

```

Tokenized form:

```

["The", "patient", "has", "asthma", "and", "takes", "aspirin"]

```

BIO labels:

```

["O", "O", "O", "B-DISEASE", "O", "O", "B-CHEMICAL"]

```

This format allows standard sequence evaluation metrics to be applied.

---

# Evaluation Metrics

The benchmark uses the **seqeval** library to compute standard NER evaluation metrics.

Metrics include:

- Precision
- Recall
- F1-score
- Classification report

These metrics provide a clear comparison between biomedical NER models.

---

# Benchmark Results

Example benchmark output:

## SciSpacy

Precision: **0.3679**

Recall: **0.8152**

F1-score: **0.5070**

---

## BioBERT

Precision: **0.4871**

Recall: **0.6485**

F1-score: **0.5563**

---

## ClinicalBERT

Precision: **0.1946**

Recall: **0.8342**

F1-score: **0.3155**

---

### Observations

- **BioBERT achieved the highest overall F1-score**
- **SciSpacy achieved strong recall but lower precision**
- **ClinicalBERT produced many false positives**

These results highlight the trade-off between **precision, recall, and computational efficiency** across biomedical NER systems.

---

# Model Runtime Comparison

Runtime performance was also measured during entity extraction.

### SciSpacy

Total time taken:

```

262.47 seconds

```

Average time per clinical note:

```

0.0529 seconds

```

SciSpacy is the fastest model and scales well to large clinical datasets.

---

### BioBERT

Average time per clinical note:

```

6.34 seconds

```

BioBERT provides strong contextual understanding but has slower inference due to transformer computations.

---

### ClinicalBERT

Total time taken:

```

1144.22 seconds

```

Average time per clinical note:

```

0.2186 seconds

````

ClinicalBERT falls between SciSpacy and BioBERT in runtime depending on preprocessing constraints.

---

# Running the Project

Install dependencies:

```bash
pip install -r requirements/base.txt
````

Run the NER pipeline:

```bash
python -m src.pipeline
```

Run the benchmark:

```bash
python -m src.benchmark
```

The benchmark script prints evaluation metrics for each model.

---

# Repository Structure

```
data/
└── processed/
    ├── cleaned_notes.csv
    └── ner/
        ├── candidate_gold_entities.jsonl
        ├── scispacy/
        │   └── scispacy_entities.jsonl
        └── transformers/
            ├── biobert/
            │   └── biobert_entities.jsonl
            └── clinicalbert/
                └── clinicalbert_entities_clean.jsonl

src/
├── pipeline.py
├── benchmark.py


docs/
└── Biomedical_NER_Clinical_Notes.pdf
```

---

# Research Documentation

A research report describing the biomedical NER experiments is included in the repository:

```
docs/Biomedical_NER_Clinical_Notes.pdf
```

The document summarizes:

* clinical text preprocessing
* biomedical NER pipeline design
* experimental setup
* model comparison results

This document represents the **initial research report for the project** and forms the basis for potential journal publication.

---

# Future Work

Future extensions of this project include:

* Entity normalization to biomedical ontologies
* Clinical knowledge graph construction
* Graph-based reasoning over medical relationships
* Spatial modeling of anatomical relationships
* Comorbidity prediction using graph learning

These steps will enable more advanced applications such as **clinical decision support and medical knowledge discovery**.

---

# Project Status

This repository currently focuses on:

* Biomedical Named Entity Recognition
* Model comparison and benchmarking
* Candidate gold dataset construction

The next phase of the project will extend the pipeline toward **clinical knowledge graph generation and graph-based healthcare analytics**.




