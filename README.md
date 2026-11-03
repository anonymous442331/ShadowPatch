# ShadowPatch — Artifact

**Paper:** ShadowPatch: Taint-Guided Structural Evasion Against Vulnerability
Detection Models

---

## Overview

ShadowPatch is a taint-guided structural evasion framework for ML-based
vulnerability detectors. It restricts transformations to regions outside the
identified vulnerability-relevant code, then independently validates each
variant with three post-hoc checks. It operates in four phases:

| Phase | Component | Description |
|---|---|---|
| 1 | `ShadowPatch_Attack/pdg_taint.py` | Dependence-based taint analysis — partitions each function into protected region T and safe region S |
| 2 | `ShadowPatch_Attack/code_transformer.py` | 18 structural transforms (T1–T18) applied only within S |
| 3 | `ShadowPatch_Attack/genetic_optimizer.py` | Black-box genetic search (P=20, G=50, B=500) to reduce detector score |
| 4 | `ShadowPatch_Attack/posthoc_checks.py` | Three post-hoc checks: protected-region identity, edit legality, differential testing |

---


## Directory Structure

```
ShadowPatch/
├── data/
│   └── dataset_loader.py        # BigVul + Devign + DiverseVul + PrimeVul loaders
├── models/
│   └── vulnerability_detector.py # LineVul, ReVeal, IVDetect, GraphCodeBERT
├── ShadowPatch_Attack/
│   ├── pdg_taint.py             # Phase 1: taint propagation (region T / region S)
│   ├── code_transformer.py      # Phase 2: T1–T18 structural transforms
│   ├── genetic_optimizer.py     # Phase 3: tournament selection, crossover, mutation
│   ├── posthoc_checks.py        # Phase 4: protected-region identity, edit legality, differential testing
│   └── cvss_fitness.py          # Optional CVSS-weighted fitness (not used in paper)
├── defense/
│   └── defense.py               # D1 adv. training, D2 ensemble, D3 rand. smoothing,
│                                #   D4 input norm., D5 PDG-Taint MLP (proposed)
├── evaluation/
│   └── evaluator.py             # Reproduces the paper tables and figures
├── utils/
│   └── utils.py
├── run_experiment.py            # End-to-end runner
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Reproducing Paper Results

### Quick smoke test

```bash
python run_experiment.py \
    --models linevul reveal \
    --max_bigvul 2000 --max_devign 1000 --max_diversevul 500 --max_primevul 500\
    --train_epochs 2 --attack_samples 50 --query_budget 100 \
    --output_dir results_test/
```

### Full experiment (exact paper settings)

```bash
python run_experiment.py \
    --models linevul reveal ivdetect graphcodebert \
    --max_bigvul 10000 --max_devign 5000 --max_diversevul 3000 --max_primevul 500\
    --train_epochs 5 --attack_samples 500 --query_budget 500 \
    --run_ablation \
    --output_dir results/
```

Post-hoc validation (Phase 4) runs by default. Use `--no_validation` to skip it
or `--no_differential` to skip only Check 3 (differential testing, which
requires a C compiler and self-contained functions).


---

## Transformation Library

| ID | Category | Description |
|---|---|---|
| T1 | Loop | for → while |
| T2 | Loop | while → do-while |
| T3 | Loop | Loop unrolling (factor 2) |
| T4 | Loop | Loop splitting |
| T5 | Pointer | Add alias declaration |
| T6 | Pointer | Array subscript → pointer arithmetic |
| T7 | Pointer | Pointer arithmetic → subscript |
| T8 | Pointer | Split pointer declaration |
| T9 | Control-flow | Insert unreachable goto |
| T10 | Control-flow | Insert opaque predicate if(1\|0) |
| T11 | Control-flow | Statement reordering |
| T12 | Control-flow | Compound condition split |
| T13 | Control-flow | Insert dead-code block |
| T14 | Dead code | Rename local identifier |
| T15 | Dead code | Insert dead computation |
| T16 | Dead code | Split variable declaration |
| T17 | Dead code | Insert volatile counter |
| T18 | Dead code | Add redundant type cast |

All transforms are applied exclusively within the safe region S = L \ T.

---

## Datasets

| Dataset | Source | Used (vulnerable) |
|---|---|---|
| DiverseVul | Chen et al., RAID 2023 | 18,945 (5.7%) |
| PrimeVul | Ding et al., ICSE 2025 | 6,968 (3.0%) |
| BigVul | GitHub CVE | 10,895 (5.0%) |
| Devign | FFmpeg/QEMU | 11,888 (43.5%) |

Each dataset is split 70/15/15 into train/validation/test. 

---

## Attack Configuration

| Parameter | Value |
|---|---|
| Population size P | 20 |
| Max generations G | 50 |
| Mutation rate p_m | 0.4 |
| Query budget B | 500 per function |
| Decision threshold τ | 0.5 |
| Random seed | 42 |

---

## Post-hoc Validation (Phase 4)

Each successful variant is validated by three independent checks that do not
query the detector:

| Check | Verdicts |
|---|---|
| Protected-region identity | IDENTICAL / MODIFIED / NO-REGION |
| Edit legality | LEGAL / VIOLATION |
| Differential testing | PASS / FAIL / UNSUPPORTED / UNKNOWN |

Checks 1 and 2 assess structural preservation. Check 3 provides behavioral
evidence where executable testing is feasible; for isolated functions without
full build dependencies it returns UNSUPPORTED. Together these checks assess
transformation preservation without establishing formal semantic equivalence.

---


