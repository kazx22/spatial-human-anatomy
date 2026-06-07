# Biomedical NER — Comparative Evaluation on BC5CDR

This repository contains the code for a comparative evaluation of five off-the-shelf biomedical named entity recognition models on the BC5CDR corpus, along with a weighted content-aware pseudo-gold majority voting framework for ensemble annotation quality assessment.

All models are evaluated **zero-shot** — no fine-tuning was performed.

---

## Models and Checkpoints

| Model | Checkpoint(s) | Architecture |
|---|---|---|
| **SciSpacy** | `en_ner_bc5cdr_md` | spaCy pipeline |
| **BioBERT** | `alvaroalon2/biobert_diseases_ner` + `alvaroalon2/biobert_chemical_ner` | BERT (dual-model) |
| **PubMedBERT** | `sarahmiller137/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-ft-ncbi-disease` + `OpenMed/OpenMed-NER-ChemicalDetect-PubMed-335M` | BERT (dual-model) |
| **ClinicalBERT** | `samrawal/bert-base-uncased_clinical-ner` | BERT (single-model) |
| **BioELECTRA** | `d4data/biomedical-ner-all` | ELECTRA (single-model) |

BioBERT and PubMedBERT use separate checkpoints for disease and chemical entity types. ClinicalBERT's native label space (`problem`, `treatment`, `test`) is mapped to BC5CDR equivalents at inference; `test` spans are dropped as they have no BC5CDR counterpart.

---

## Dataset

This study uses the **BC5CDR corpus** (BioCreative V Chemical-Disease Relation dataset), which contains PubMed abstracts annotated for diseases and chemicals.

**The dataset is not included in this repository.** It must be obtained separately from the official source:

> BioCreative V CDR Task Corpus: https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/

Place the three PubTator files in `data/raw/bc5cdr/`:

```
data/raw/bc5cdr/
    CDR_TrainingSet.PubTator.txt
    CDR_DevelopmentSet.PubTator.txt
    CDR_TestSet.PubTator.txt
```

---

## Weighted Pseudo-Gold Framework

Each model is weighted by its measured F1 score against human gold on BC5CDR:

| Model | F1 (vs Human Gold) | Weight |
|---|---|---|
| SciSpacy | 0.896 | 0.8959 |
| BioBERT | 0.726 | 0.7259 |
| PubMedBERT | 0.568 | 0.5677 |
| BioELECTRA | 0.547 | 0.5470 |
| ClinicalBERT | 0.421 | 0.4213 |

A span is admitted to the pseudo-gold set when the sum of the F1 weights of models that predicted it meets or exceeds the voting threshold (default **1.5**). This threshold sits above any single model's weight, so no single model can unilaterally admit a span. A threshold sensitivity sweep (0.45–2.00) confirmed that F1 peaks at 1.5 (F1 = 0.848, precision = 0.989, recall = 0.742).

---

## Repository Structure

```
.
├── data/
│   ├── raw/bc5cdr/                  # BC5CDR source files (not included)
│   ├── processed/bc5cdr/            # Parsed JSONL files (generated)
│   └── gold/                        # Gold BIO files and pseudo-gold sets (generated)
│       └── sensitivity/             # Threshold sweep outputs
├── figure/                          # Output figures (generated)
│   ├── confusion/                   # Per-model confusion matrices
│   └── sensitivity/                 # Per-threshold bar charts
├── src/
│   ├── parse_bc5cdr.py              # Parse raw BC5CDR PubTator files → JSONL
│   ├── build_gold_bc5cdr.py         # Convert entity spans → BIO gold standard
│   ├── scispacy_bc5cdr.py           # SciSpacy inference
│   ├── biobert_bc5cdr.py            # BioBERT inference (dual-model)
│   ├── pubmed_bc5cdr.py             # PubMedBERT inference (dual-model)
│   ├── clinicalbert_bc5cdr.py       # ClinicalBERT inference
│   ├── bioelectra_bc5cdr.py         # BioELECTRA inference
│   ├── candidate_gold.py            # Weighted pseudo-gold construction
│   ├── entity_filtering.py          # Label-aware confidence filtering
│   ├── bc5cdr_evaluation.py         # Central evaluation harness
│   ├── bootstrap_significance.py    # Paired bootstrap significance testing
│   ├── cohen_kappa.py               # Inter-model and model-vs-gold kappa
│   ├── threshold_sensitivity.py     # Threshold sweep evaluation and figures
│   ├── graph.py                     # Figure generation
│   └── utils.py                     # Shared utilities
└── README.md
```

---

## Setup

Python 3.9+ recommended. Install dependencies:

```bash
pip install spacy transformers seqeval scikit-learn numpy matplotlib
```

---

## Running the Pipeline

All scripts should be run from the project root.

### 1. Parse the raw BC5CDR files

```bash
python -m src.parse_bc5cdr
```

Outputs `data/processed/bc5cdr/bc5cdr_{train,dev,test}_{docs,entities}.jsonl`.

### 2. Build the human gold BIO standard

```bash
python -m src.build_gold_bc5cdr
```

Outputs `data/gold/bc5cdr_train_gold_bio.jsonl`.

### 3. Run the five model inference scripts

```bash
python -m src.scispacy_bc5cdr
python -m src.biobert_bc5cdr
python -m src.pubmed_bc5cdr
python -m src.clinicalbert_bc5cdr
python -m src.bioelectra_bc5cdr
```

Each writes a per-model JSONL file to `data/processed/bc5cdr/`.

### 4. Build the weighted pseudo-gold set

```bash
python -m src.candidate_gold
```

Outputs `data/gold/weighted_candidate_gold_train_entities_bc5cdr.jsonl` and threshold sensitivity files under `data/gold/sensitivity/`.

### 5. Run the main evaluation

```bash
python -m src.bc5cdr_evaluation
```

Scores all models against human gold and both pseudo-gold sets, and writes all figures to `figure/`.

### 6. Analysis scripts (optional)

```bash
python -m src.bootstrap_significance     # Paired bootstrap p-values and 95% CIs
python -m src.cohen_kappa                # Inter-model and model-vs-gold kappa
python -m src.threshold_sensitivity      # Threshold sensitivity curve
```

---

## Results Summary

Performance against BC5CDR human gold annotations (BC5CDR train split, zero-shot):

| Model | Precision | Recall | F1 |
|---|---|---|---|
| SciSpacy | 0.93 | 0.86 | **0.90** |
| BioBERT | 0.63 | 0.86 | 0.73 |
| PubMedBERT | 0.49 | 0.67 | 0.57 |
| BioELECTRA | 0.59 | 0.51 | 0.55 |
| ClinicalBERT | 0.36 | 0.50 | 0.42 |

Pseudo-gold quality against human gold:

| Framework | Precision | Recall | F1 |
|---|---|---|---|
| Majority Candidate Gold | 0.931 | 0.540 | 0.683 |
| Weighted Candidate Gold (threshold = 1.5) | 0.989 | 0.741 | **0.848** |

This is a comparative study. Results reflect off-the-shelf performance and should not be compared directly to fine-tuned baselines.

---

## Analysis Results

### Bootstrap Significance Testing

Paired bootstrap significance testing (1000 resamples, seed = 42) on 500 BC5CDR training documents. All pairwise F1 differences are statistically significant at p < 0.05 except PubMedBERT vs BioELECTRA (p = 0.041, F1 gap = 0.021).

**95% Bootstrap Confidence Intervals on F1:**

| Model | F1 | 95% CI |
|---|---|---|
| SciSpacy | 0.8959 | [0.8868, 0.9049] |
| BioBERT | 0.7259 | [0.7134, 0.7382] |
| PubMedBERT | 0.5677 | [0.5475, 0.5863] |
| BioELECTRA | 0.5470 | [0.5294, 0.5638] |
| ClinicalBERT | 0.4213 | [0.4051, 0.4362] |

**Pairwise Significance (paired bootstrap, p < 0.05 threshold):**

| Pair | F1 diff | p-value | Significant | Better model |
|---|---|---|---|---|
| SciSpacy vs BioBERT | +0.1701 | 0.0000 | Yes | SciSpacy |
| SciSpacy vs PubMedBERT | +0.3283 | 0.0000 | Yes | SciSpacy |
| SciSpacy vs ClinicalBERT | +0.4747 | 0.0000 | Yes | SciSpacy |
| SciSpacy vs BioELECTRA | +0.3490 | 0.0000 | Yes | SciSpacy |
| BioBERT vs PubMedBERT | +0.1582 | 0.0000 | Yes | BioBERT |
| BioBERT vs ClinicalBERT | +0.3046 | 0.0000 | Yes | BioBERT |
| BioBERT vs BioELECTRA | +0.1789 | 0.0000 | Yes | BioBERT |
| PubMedBERT vs ClinicalBERT | +0.1464 | 0.0000 | Yes | PubMedBERT |
| PubMedBERT vs BioELECTRA | +0.0207 | 0.0410 | Yes (marginal) | PubMedBERT |
| ClinicalBERT vs BioELECTRA | −0.1257 | 0.0000 | Yes | BioELECTRA |

---

### Cohen's Kappa
![Cohen's Kappa](figure/kappa_heatmap.png)
Token-level BIO label agreement across 93,024 tokens (500 documents). Interpreted using the Landis & Koch (1977) scale.

**Model vs Human Gold:**

| Model | κ | Interpretation |
|---|---|---|
| SciSpacy | 0.9219 | Almost perfect |
| BioELECTRA | 0.5663 | Moderate |
| BioBERT | 0.4694 | Moderate |
| ClinicalBERT | 0.4439 | Moderate |
| PubMedBERT | 0.4363 | Moderate |

SciSpacy achieves near-expert agreement with BC5CDR human gold (κ = 0.922). The remaining four models cluster in the Moderate range (0.436–0.566).

**Pairwise Inter-Model Kappa:**

| | SciSpacy | BioBERT | PubMedBERT | ClinicalBERT | BioELECTRA |
|---|---|---|---|---|---|
| SciSpacy | 1.000 | 0.430 | 0.404 | 0.437 | 0.552 |
| BioBERT | 0.430 | 1.000 | 0.763 | 0.257 | 0.264 |
| PubMedBERT | 0.404 | 0.763 | 1.000 | 0.253 | 0.280 |
| ClinicalBERT | 0.437 | 0.257 | 0.253 | 1.000 | 0.359 |
| BioELECTRA | 0.552 | 0.264 | 0.280 | 0.359 | 1.000 |

Inter-model kappa mean = 0.400 (Moderate), std = 0.153. The high BioBERT–PubMedBERT agreement (κ = 0.763, Substantial) reflects their shared BERT pretraining on biomedical text — they make similar mistakes even though they use different checkpoints. ClinicalBERT has the lowest pairwise agreement with all other models (κ = 0.253–0.437), consistent with its i2b2 training domain. The moderate inter-model disagreement overall supports the rationale for the weighted ensemble framework — if the models agreed, majority voting would add nothing.

---

## Figures

All figures are written to `figure/` when `bc5cdr_evaluation.py` and `threshold_sensitivity.py` are run.

### Figure 1 — Model Performance Against Human Gold

![Model performance against human gold annotations (BC5CDR)](figure/fig1_models_vs_human_gold.png)

Grouped bar chart (precision / recall / F1) for all five base models. SciSpacy leads with F1 = 0.90 and the highest precision (0.93). BioBERT has the highest recall (0.86) but low precision (0.63), reflecting over-prediction. PubMedBERT, BioELECTRA, and ClinicalBERT trail significantly, with F1 scores of 0.57, 0.55, and 0.42 respectively.

---

### Figure 2 — Pseudo-Gold Quality Against Human Gold

![Pseudo-gold annotation quality against human gold (BC5CDR)](figure/fig2_candidate_gold_comparison.png)

Compares majority voting and weighted voting pseudo-gold sets against human gold. Majority voting achieves high precision (0.931) but poor recall (0.540), giving F1 = 0.683. Weighted voting improves recall substantially (0.741) while maintaining near-perfect precision (0.989), lifting F1 to 0.848. This shows the F1-weighted scheme retains far more legitimate spans without introducing noise.

---

### Figure 3 — Per-Label F1

![Per-label F1-score against human gold (BC5CDR)](figure/fig3_per_label_f1.png)

Shows DISEASE and CHEMICAL F1 separately for each model. The chemical detection gap is striking: BioBERT and PubMedBERT both achieve F1 = 0.94 on CHEMICAL despite low overall scores, because their disease checkpoints underperform badly (BioBERT DISEASE F1 = 0.55, PubMedBERT DISEASE F1 = 0.25). SciSpacy is the only model with balanced performance across both types (DISEASE = 0.91, CHEMICAL = 0.88). BioELECTRA shows the reverse of BioBERT — stronger on chemicals (0.63) than diseases (0.46).

---

### Figure 4 — SciSpacy Confusion Matrix

![SciSpacy confusion matrix vs human gold (BC5CDR)](figure/fig4_scispacy_confusion_matrix.png)

Token-level confusion matrix for the best-performing model. The diagonal is strong: 3801 correct B-DISEASE, 2425 correct I-DISEASE, 4389 correct B-CHEMICAL, 516 correct I-CHEMICAL. The main failure mode is false negatives into O (569 B-CHEMICAL and 126 B-DISEASE missed entirely), and 94 B-DISEASE tokens misclassified as I-CHEMICAL — a label boundary confusion rather than entity-type confusion.

---

### Figure 5 — Weighted Candidate Gold Confusion Matrix

![Weighted candidate gold confusion matrix vs human gold (BC5CDR)](figure/fig5_weighted_gold_confusion_matrix.png)

Pseudo-gold vs human gold at the operating threshold (1.5). Errors are almost entirely false negatives — 825 B-DISEASE and 1562 B-CHEMICAL spans predicted as O, consistent with the recall = 0.741 in Figure 2. Cross-type confusion is essentially zero (no B-DISEASE predicted as B-CHEMICAL or vice versa), confirming the framework does not introduce label noise, only recall loss.

---

### Figure 6 — All Model Confusion Matrices

![All model confusion matrices vs human gold (BC5CDR)](figure/fig6_all_model_confusion_matrices.png)

2×3 grid showing all five models side by side. Key observations: BioBERT has a large I-DISEASE false positive block (16,874 true-O tokens predicted as I-DISEASE), suggesting the disease model is generating spurious multi-token continuations. PubMedBERT shows a similar but smaller pattern (13,301). ClinicalBERT has broad off-diagonal spread across all classes, consistent with domain mismatch from i2b2 training. BioELECTRA has the cleanest off-diagonal outside SciSpacy, with most errors being true false negatives rather than cross-type confusion.

---

### Threshold Sensitivity Curve

![Pseudo-gold threshold sensitivity vs human gold](figure/threshold_sensitivity_curve.png)

Precision, recall, and F1 of the weighted pseudo-gold set versus human gold across thresholds 0.45–2.00. Precision climbs steeply from 0.40 at threshold 0.45 to 0.989 at 1.5, then plateaus near 1.0. Recall peaks around 0.79 at low thresholds and falls to 0.57 at 2.0 as fewer spans are retained. F1 peaks at **threshold = 1.5 (F1 = 0.848)**, marked with a dashed vertical line, which is the operating threshold used throughout the paper.

---

### Per-Threshold Bar Charts

Individual precision/recall/F1 bar charts for each of the seven thresholds tested:

| Threshold | Precision  | Recall     | F1         | Figure                                                |
| --------- | ---------- | ---------- | ---------- | ----------------------------------------------------- |
| 0.45      | 0.4029     | 0.6943     | 0.5099     | ![t=0.45](figure/sensitivity/threshold_0.45_bars.png) |
| 0.70      | 0.5838     | 0.7931     | 0.6726     | ![t=0.70](figure/sensitivity/threshold_0.7_bars.png)  |
| 0.90      | 0.6929     | 0.7829     | 0.7351     | ![t=0.90](figure/sensitivity/threshold_0.9_bars.png)  |
| 1.20      | 0.7611     | 0.7944     | 0.7774     | ![t=1.20](figure/sensitivity/threshold_1.2_bars.png)  |
| **1.50**  | **0.9893** | **0.7415** | **0.8476** | ![t=1.50](figure/sensitivity/threshold_1.5_bars.png)  |
| 1.80      | 0.9929     | 0.6150     | 0.7596     | ![t=1.80](figure/sensitivity/threshold_1.8_bars.png)  |
| 2.00      | 0.9951     | 0.5658     | 0.7214     | ![t=2.00](figure/sensitivity/threshold_2.0_bars.png)  |

---

### Per-Model Confusion Matrices

Full-size individual confusion matrices for all five models, equivalent to the panels in Figure 6 but at higher resolution for supplementary use.

| Model        | Confusion Matrix                                                       |
| ------------ | ---------------------------------------------------------------------- |
| SciSpacy     | ![SciSpacy confusion matrix](figure/confusion/cm_scispacy.png)         |
| BioBERT      | ![BioBERT confusion matrix](figure/confusion/cm_biobert.png)           |
| PubMedBERT   | ![PubMedBERT confusion matrix](figure/confusion/cm_pubmedbert.png)     |
| ClinicalBERT | ![ClinicalBERT confusion matrix](figure/confusion/cm_clinicalbert.png) |
| BioELECTRA   | ![BioELECTRA confusion matrix](figure/confusion/cm_bioelectra.png)     |

---

## Compute

Experiments were run on an NVIDIA RTX 3070 Ti Mobile (8 GB VRAM). Approximate inference times per document:

| Model | Avg. time/doc |
|---|---|
| SciSpacy | 0.024 s |
| BioELECTRA | 0.13 s |
| ClinicalBERT | 0.21 s |
| BioBERT | 0.46 s |
| PubMedBERT | 3.48 s |

---

## Citation

If you use this code or framework, please cite the associated paper (details to follow on publication).

---

## License

Code: MIT. BC5CDR dataset: subject to BioCreative data use terms. The dataset is not redistributed here.
