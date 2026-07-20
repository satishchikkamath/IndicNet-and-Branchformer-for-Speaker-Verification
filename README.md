# IndicNet and Branchformer for Speaker Verification

## Overview

This repository contains the implementation of multilingual speaker verification experiments conducted on the Kathbath dataset. The repository includes implementations of:

- ECAPA-TDNN (Training from Scratch)
- ECAPA-TDNN (Zero-shot Evaluation)
- Conformer (Training from Scratch)
- E-Branchformer (Training from Scratch)

The project evaluates different deep speaker embedding architectures for multilingual Indian speaker verification.

---

## Repository Structure

```
.
├── ALL_FINAL_PLOTS/
│   ├── Dataset Plots/
│   ├── kathbath/
│       ├── eer/
│       └── tsne/
│
├── conformer scrath/
│   ├── conformer_scratch.py
│   ├── conformer_scratch_a1.py
│   ├── conformer_scratch_a2.py
│   └── ...
│
├── ebrachformer scrath/
│   ├── ebranchformer_scratch.py
│   ├── ebranchformer_scratch_a1.py
│   ├── ebranchformer_scratch_a2.py
│   └── ...
│
├── ecapa_kath_scratch.py
├── zeroshot_ecapa.py
├── zeroshot_ecapa1.py
└── README.md
```

---

# Models

The repository contains implementations of:

- ECAPA-TDNN
- Zero-shot ECAPA-TDNN
- Conformer
- E-Branchformer

---

# Dataset

Experiments were conducted using the AI4Bharat Kathbath Dataset.

Dataset consists of multilingual Indian speech collected for speaker recognition research.

Supported languages include:

- Hindi
- Kannada
- Telugu
- Tamil
- Bengali
- Marathi
- Gujarati
- Malayalam
- Punjabi
- Odia
- Assamese
- Urdu

Dataset Link

https://huggingface.co/datasets/ai4bharat/Kathbath

---

# Experimental Tasks

The repository includes experiments for

## Experiment 1

ECAPA-TDNN trained from scratch.

---

## Experiment 2

Zero-shot ECAPA-TDNN evaluation.

---

## Experiment 3

Conformer trained from scratch.

---

## Experiment 4

E-Branchformer trained from scratch.

---

# Evaluation

Performance is evaluated using

- Equal Error Rate (EER)
- t-SNE visualization
- Embedding analysis

Plots are available inside

```
ALL_FINAL_PLOTS/
```

---

# Requirements

Python 3.10+

Main libraries

```
torch
torchaudio
numpy
scikit-learn
matplotlib
tqdm
```

Install dependencies

```bash
pip install torch torchaudio numpy scikit-learn matplotlib tqdm
```

---

# Running Experiments

## ECAPA Scratch

```bash
python ecapa_kath_scratch.py
```

---

## Zero-shot ECAPA

```bash
python zeroshot_ecapa.py
```

---

## Conformer

```bash
python "conformer scrath/conformer_scratch.py"
```

---

## E-Branchformer

```bash
python "ebrachformer scrath/ebranchformer_scratch.py"
```

---

# Results

The repository contains

- Training curves
- EER plots
- t-SNE visualizations
- Evaluation outputs

---


This repository is intended for academic and research purposes.
