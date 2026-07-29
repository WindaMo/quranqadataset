
# Quran QA Dataset Construction Pipeline

This repository contains the **source code, generation pipeline, validation framework, and dataset artifacts**
used in the study:

**Single-Path and Multi-Path Thematic Prompting for Grounded Question–Answer Dataset Construction**

The project automatically constructs a **verse-grounded QA dataset from the Qur'an** using Large Language Models (LLMs),
and evaluates different prompting strategies and validation protocols.

The repository also contains **human evaluation scripts and analysis tools** used to assess agreement between
LLM validators and human annotators.

---

# Overview

The pipeline follows four main stages:

1. **Canonical Verse Preparation**  
   Qur’anic verses are combined with thematic paths derived from a hierarchical thematic index.

2. **QA Generation**  
   Two prompting strategies are used:
   
   - **SPTP** — Single-Path Thematic Prompting  
   - **MPTP** — Multi-Path Thematic Prompting

3. **LLM Validation**  
   Generated QA pairs are evaluated by **three independent LLM validators** to verify whether answers are
   strictly supported by the verse text.

4. **Gold Dataset Construction**  
   Validated QA pairs are consolidated into a final **canonical grounded QA dataset**.

A **human evaluation subset** is also included to measure agreement between automated validators
and human annotators.

---

# Repository Structure

## Dataset Files

### Core Dataset Inputs

| File | Description |
|-----|-------------|
| ayat.json | Qur’anic verse text used for dataset construction |
| tematik.json | Hierarchical thematic index |
| canonical_verses_with_theme.json | Verses paired with thematic paths |

### Filtered Dataset Inputs

| File | Description |
|-----|-------------|
| filtered_verses.json | Filtered verses used for SPTP generation |
| filtered_verses_multipath.json | Filtered verses used for MPTP generation |

### Generated QA Datasets

| File | Description |
|-----|-------------|
| canonical_gold_qas_SPTP.json | QA pairs generated using SPTP |
| canonical_gold_qas_MPTP.json | QA pairs generated using MPTP |

### Gold Dataset Variants

| File | Description |
|-----|-------------|
| gold_majority_supported_SPTP.json | QA pairs supported by majority validators |
| gold_majority_supported_MPTP.json | QA pairs supported by majority validators |
| gold_strict_supported_SPTP.json | QA pairs supported by all validators |
| gold_strict_supported_MPTP.json | QA pairs supported by all validators |
| gold_consensus_supported.json | Final consensus dataset |

### Final Verse Coverage

| File | Description |
|-----|-------------|
| canonical_verses_for_gold.json | Verses included in the final dataset |

---

# Generation Pipeline

## Dataset Preparation

Prepare canonical dataset:

```
python prepare_canonical_dataset.py
python prepare_canonical_dataset_with_theme.py
```

---

## QA Generation

### Single-Path Thematic Prompting

```
python generate_canonical_gold_gpt52_SPTP.py
```

### Multi-Path Thematic Prompting

```
python generate_canonical_gold_gpt52_MPTP.py
```

### Context Generation

```
python generate_canonical_context_gold_gpt52.py
```

---

# LLM Validation

The repository implements **automatic validation using multiple LLM judges**.

Each QA pair is evaluated by **three independent validators**.

Scripts:

```
dual_llm_validation.py
triple_llm_validation.py
triple_llm_validation_multipath.py
```

Validation outputs:

```
validation_log_3models_SPTP.jsonl
validation_log_3models_MPTP.jsonl
```

Each entry records validator decisions:

```
SUPPORTED
NOT_SUPPORTED
```

---

# Human Evaluation

A subset of **415 QA pairs** was manually evaluated by two annotators.

Files:

| File | Description |
|-----|-------------|
| human_validation_set.json | Human evaluation subset |
| evaluation_values.csv | Annotator labels |
| human_annotation_multipath_non_strict.xlsx | Annotation sheet |

Scripts for human evaluation:

```
generate_human_annotation.py
export_human_annotation_excel.py
build_human_validation_set.py
```

---

# Agreement Analysis

The repository contains scripts used to compute agreement statistics.

Metrics include:

- raw agreement
- Cohen's κ
- Fleiss' κ
- Krippendorff's α
- confusion matrices

Scripts:

```
evaluate_llm_human_agreement.py
evaluate_llm_human_agreement_Krippendorff.py
confusion_matrix_human_llm.py
```

Additional scripts analyze disagreement cases:

```
extract_disagreement_cases.py
disagreement_cases.json
```

---

# Human–LLM Combined Validation

To merge LLM and human annotations:

```
combine_human_llm_validation.py
validation_log_3models_human_MPTP.jsonl
```

---

# Reproducing the Experiments

### Step 1 — Prepare dataset

```
python prepare_canonical_dataset_with_theme.py
```

### Step 2 — Generate QA pairs

SPTP:

```
python generate_canonical_gold_gpt52_SPTP.py
```

MPTP:

```
python generate_canonical_gold_gpt52_MPTP.py
```

### Step 3 — Run LLM validation

```
python triple_llm_validation.py
```

or

```
python triple_llm_validation_multipath.py
```

### Step 4 — Compute agreement statistics

```
python evaluate_llm_human_agreement.py
```

or

```
python evaluate_llm_human_agreement_Krippendorff.py
```

---

# Dataset Summary

| Property | Value |
|--------|------|
| Total verses attempted | 4685 |
| Final QA pairs | 4644 |
| Verse coverage | ~99% |
| LLM validator majority acceptance | 98.8% |
| LLM validator strict consensus | 91.0% |
| Human evaluation subset | 415 QA pairs |
| LLM–human agreement | 83.4% |

---

# Citation

If you use this dataset or code, please cite the associated paper.

@article{nasution2025quranqa,
title={Single-Path and Multi-Path Thematic Prompting for Grounded QA Dataset Construction},
author={Nasution, Arbi Haza},
year={2025}
}

---

# License

Specify the license appropriate for your repository (e.g., MIT License or CC BY 4.0).

---

# Contact

**Arbi Haza Nasution**  
Universitas Islam Riau  

Repository: https://gitlab.com/arbihaza/quranqadataset
