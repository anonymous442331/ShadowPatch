# ShadowPatch — Artifact

**Paper:** ShadowPatch: Semantic-Preserving Adversarial Code Patches Against
Machine Learning-Based Vulnerability Detection Models
*ACM CCS 2026*

---

## Overview

ShadowPatch is a structure-aware evasion framework for ML-based vulnerability
detectors. It operates in four phases:

| Phase | Component | Description |
|---|---|---|
| 1 | `ShadowPatch_Attack/pdg_taint.py` | PDG-based taint analysis — partitions each function into taint region T and safe region S |
| 2 | `ShadowPatch_Attack/code_transformer.py` | 18 semantics-preserving structural transforms (T1–T18) applied only within S |
| 3 | `ShadowPatch_Attack/genetic_optimizer.py` | Black-box genetic search (P=20, G=50, B=500) to minimize detector score |
| 4 | `ShadowPatch_Attack/smt_verifier.py` | Z3 SMT bounded equivalence verification (30 s timeout, loop unroll=10) |

---

## Key Results (paper tables)

| Metric | Value |
|---|---|
| Peak ASR | 69.9% (Devign / LineVul) |
| Combined-split ASR — LineVul | 31.3% |
| Combined-split ASR — GraphCodeBERT | 15.3% |
| Combined-split ASR — ReVeal | 8.5% |
| Combined-split ASR — IVDetect | 5.7% |
| SMT-verified equivalence | 99.1–100% across all models |
| Peak black-box transfer ASR | 57.1% (IVDetect → ReVeal) |
| D5 detection rate | 88.0% at 4.0% false positives |
| Avg queries to success | 71–150 (transformer models) |

All results are on the Combined split (17,664 samples; 2,651 test)
unless otherwise noted. See Tables 4–8 in the paper.

---

## Directory Structure

```
shadowpatch_v2/
├── data/
│   └── dataset_loader.py        # BigVul + Devign + SARD + PrimeVul loaders
│                                #   merge → deduplicate → CWE filter → 70/15/15 split
├── models/
│   └── vulnerability_detector.py # LineVul, ReVeal, IVDetect, GraphCodeBERT
├── ShadowPatch_Attack/
│   ├── pdg_taint.py             # Phase 1: Joern CPG extraction + taint propagation
│   ├── code_transformer.py      # Phase 2: T1–T18 via Clang LibTooling + GCC syntax check
│   ├── genetic_optimizer.py     # Phase 3: tournament selection, crossover, mutation
│   ├── smt_verifier.py          # Phase 4: LLVM-IR → Z3 UNSAT equivalence check
│   
├── defense/
│   └── defense.py               # D1 adv. training, D2 ensemble, D3 rand. smoothing,
│                                #   D4 input norm., D5 PDG-Taint MLP (proposed)
├── evaluation/
│   └── evaluator.py             # Reproduces Tables 1–8 and Figures 1–6
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
    --max_bigvul 2000 --max_devign 1000 --max_sard 500 \
    --train_epochs 2 --attack_samples 50 --query_budget 100 \
    --output_dir results_test/
```

### Full experiment (exact paper settings)

```bash
python run_experiment.py \
    --models linevul reveal ivdetect graphcodebert \
    --max_bigvul 10000 --max_devign 5000 --max_sard 3000 \
    --train_epochs 5 --attack_samples 500 --query_budget 500 \
    --smt_verify \
    --run_ablation \
    --output_dir results/
```


**Hardware used in paper:** Tesla V100 16 GB GPU, 8 CPU cores, 64 GB RAM.


### Optional: CVSS-weighted objective

The implementation includes optional support for a CVSS-weighted fitness
function in the genetic optimizer (Phase 3). When enabled, the attack
objective incorporates vulnerability severity (CVSS score) alongside
detector evasion:

```bash
python run_experiment.py \
    --splits combined \
    --attack_samples 500 --query_budget 500 \
    --smt_verify --cvss_lambda 0.3 \
    --output_dir results_cvss/
```

`--cvss_lambda` controls the severity weighting (0.0 = disabled, default).
**This feature is not used in any experiment reported in the paper.**
All paper results use `--cvss_lambda 0.0` (or omit the flag entirely).
This option is provided for exploratory analysis and future extensions
and does not affect reproducibility of the reported results.

---

## Output Files

```
results/
├── tables/
│   ├── table1_clean_performance.csv      # Table 5 in paper (clean detector perf.)
│   ├── table2_attack_success.csv         # Table 4 (ASR + AvgQ per model/dataset)
│   ├── table3_transferability.csv        # Table 7 (4×4 cross-model transfer)
│   ├── table4_cwe_specific.csv           # Figure 3 data (per-CWE ASR)
│   ├── table5_defenses.csv               # Table 8 (D1–D5 defense evaluation)
│   └── table6_ablation.csv              # Table 10 (ablation, requires --run_ablation)
├── figures/
│   ├── fig1_asr_vs_budget.pdf            # Figure 4 (query efficiency)
│   ├── fig2_transform_heatmap.pdf        # Figure 5 (transform usage)
│   ├── fig3_prob_shift.pdf               # Figure 6 (probability-shift distributions)
│   ├── fig4_cwe_asr.pdf                  # Figure 3 (per-CWE ASR bar chart)
│   └── fig5_ablation.pdf                # Ablation figure (requires --run_ablation)
├── raw_results_linevul.json
├── raw_results_reveal.json
├── raw_results_ivdetect.json
└── raw_results_graphcodebert.json
```

---

## Transformation Library (Table 1 in paper)

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
| T11 | Control-flow | Condition negation + branch swap |
| T12 | Control-flow | Switch → if-else chain |
| T13 | Control-flow | Insert dead-code block |
| T14 | Dead code | Rename local identifier |
| T15 | Dead code | Insert dead computation |
| T16 | Dead code | Split variable declaration |
| T17 | Dead code | Insert volatile counter |
| T18 | Dead code | Add redundant type cast |

All transforms are applied exclusively within the safe region S = L \ T,
verified syntactically by GCC (`-O0 -fsyntax-only`) and semantically by Z3.

---

## Datasets (Table 2 in paper)

| Dataset | Source | Raw | Used |
|---|---|---|---|
| BigVul | GitHub CVE | 188,636 | 7,664 |
| Devign | FFmpeg/QEMU | 27,318 | 5,000 |
| SARD | NIST Juliet | 54,484 | 3,000 |
| PrimeVul | Human-verified | 224,300 | 5,000 |
| Combined | Merged + dedup | — | 17,664 |

All datasets are publicly available; none are redistributed in this repository.

---

## Attack Configuration (Section 4.1 in paper)

| Parameter | Value |
|---|---|
| Population size P | 20 |
| Max generations G | 50 |
| Mutation rate p_m | 0.4 |
| Query budget B | 500 per function |
| Decision threshold τ | 0.5 |
| Z3 timeout | 30 s |
| Z3 loop-unroll bound | 10 |
| Random seed | 42 |

---

## Defense Summary (Table 8 in paper)

| Defense | ASR (%) | Det. (%) | FP (%) | F1 (%) |
|---|---|---|---|---|
| No defense | 31.3 | 0.0 | 0.0 | — |
| D1: Adversarial Training | 1.5 | 95.2 | 20.6 | 88.2 |
| D2: Ensemble | 1.0 | 96.8 | 24.6 | 87.5 |
| D3: Randomized Smoothing | 0.7 | 97.6 | 26.2 | 87.2 |
| D4: Input Normalization | 14.9 | 52.4 | 5.6 | 66.3 |
| **D5: PDG-Taint (proposed)** | **3.2** | **88.0** | **4.0** | **91.7** |

D5 achieves the best detection–false-positive tradeoff (Section 4.6).

---

