#!/usr/bin/env python3
"""
run_experiment.py  —  ShadowPatch 
=====================================


  [1] PRETRAINED MODELS  (no wasted training time)
      ┌──────────────────┬────────────────────────────────────┬────────────────────┐
      │ Detector         │ HuggingFace checkpoint              │ Falls back to      │
      ├──────────────────┼────────────────────────────────────┼────────────────────┤
      │ LineVul          │ michaelfu1998/linevul (fine-tuned)  │ fine-tune CodeBERT │
      │ ReVeal           │ —  (no public fine-tune)            │ fine-tune CodeBERT │
      │ IVDetect         │ —  (no public fine-tune)            │ fine-tune CodeBERT │
      │ GraphCodeBERT    │ microsoft/graphcodebert-base        │ fine-tune backbone │
      └──────────────────┴────────────────────────────────────┴────────────────────┘
      Resolution order: (1) local checkpoint  →  (2) HF fine-tuned weights
      →  (3) fine-tune from backbone ONCE, then cache.
      --force_retrain bypasses steps 1 & 2.

  [2] ALL DATASETS SEPARATELY + COMBINED
      --splits bigvul devign sard combined
      Runs the full attack pipeline on each split independently,
      then again on the merged set.  Table 2 shows ASR per split × model.

  [3] COMPARISON FRAMEWORKS  (Table 7)
      • ALERT      – naturalness-constrained renaming (ICSE 2022)
      • MHM        – Metropolis-Hastings MCMC renaming (ACL 2020)
      • Dead+Rename – T13+T14 joint, no GA
      • Random     – random transform sequences, no fitness

  [4] BOTH GA MODES
      Flat fitness (default) + CVSS-weighted fitness (--cvss_lambda 0.3)

  [5] FULL CHECKPOINT SYSTEM
      Every expensive step saved to disk immediately after completion.
      Re-running the same command resumes from last saved step.
      State auto-invalidated when key args change.

Full pipeline
─────────────
  Step 1   Load datasets (per split + combined)
  Step 2   Load / download pretrained models
  Step 3   Clean performance          → Table 1
  Step 4a  GA flat fitness attacks    → Table 2  (splits × models)
  Step 4b  GA CVSS-weighted attacks   → Table 2b [if --cvss_lambda > 0]
  Step 5   Transferability            → Table 3
  Step 6   CWE-specific analysis      → Table 4
  Step 7   Defense evaluation D1–D5  → Table 5
  Step 8   Ablation                   → Table 6
  Step 9   Comparison frameworks      → Table 7
  Step 10  All CCS-style figures

Quick start
───────────
  # Full experiment — downloads pretrained weights, attacks all splits
  python run_experiment.py \\
      --splits bigvul devign sard combined \\
      --attack_samples 500 --query_budget 500 \\
      --smt_verify --output_dir results/

  # Add CVSS weighting in a separate output dir
  python run_experiment.py \\
      --splits combined \\
      --attack_samples 500 --query_budget 500 \\
      --smt_verify --cvss_lambda 0.3 --output_dir results_cvss/

  # HPC  (no internet after first download — use cached NVD)
  python run_experiment.py \\
      --splits bigvul devign sard combined \\
      --smt_verify --cvss_lambda 0.3 --cvss_offline \\
      --output_dir results/

  # Force re-download / retrain everything
  python run_experiment.py --force_retrain ...

"""

from __future__ import annotations

import argparse, hashlib, json, logging, math, os, pickle
import random, re, sys, time
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from utils.utils                 import set_seed, configure_logging
from data.dataset_loader         import (BigVulLoader, DevignLoader,
                                         SARDLoader, PrimeVulLoader,
                                         load_dataset, get_attack_subset,
                                         VulnSample, TARGET_CWES)
from models.vulnerability_detector import VulnDetector, VulnDetectorTrainer
from ShadowPatch_Attack.genetic_optimizer    import (GeneticAttacker, AttackResult,
                                         summarise_results)
from ShadowPatch_Attack.exploit_verifier     import ExploitVerifier
from ShadowPatch_Attack.cvss_fitness         import (CVSSTable, CVSSFitnessWrapper,
                                         cvss_imputation_report)
from defense.defense             import (AdversarialTrainer, EnsembleDefense,
                                         RandomizedSmoothingDefense,
                                         InputNormalizationDefense,
                                         PDGTaintDefense)
from evaluation.evaluator        import ShadowPatchEvaluator

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Pretrained model registry
# ══════════════════════════════════════════════════════════════════════════════

# If a community-fine-tuned HF checkpoint exists for the vuln-detection task,
# list it here.  None → we fine-tune from backbone and cache locally.
FINETUNED_HF: Dict[str, Optional[str]] = {
    "linevul":       "michaelfu1998/linevul",   # ready-to-use, no training
    "reveal":        None,                       # fine-tune from backbone
    "ivdetect":      None,
    "graphcodebert": None,
}

# HuggingFace backbone IDs (used when fine-tuned checkpoint is unavailable)
BACKBONE_HF: Dict[str, str] = {
    "linevul":       "microsoft/codebert-base",
    "reveal":        "microsoft/codebert-base",
    "ivdetect":      "microsoft/codebert-base",
    "graphcodebert": "microsoft/graphcodebert-base",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Checkpoint / State Manager
# ══════════════════════════════════════════════════════════════════════════════

class CheckpointManager:
    """
    Saves and restores every expensive pipeline step so no work is ever
    repeated after a crash or interruption.

    Directory layout  (output_dir/checkpoints/):
      state.json                        step completion flags
      args_hash.txt                     invalidates cache when key args change
      model_<name>.pt                   per-model fine-tuned weights
      attack_<split>_<model>.pkl        GA flat-fitness AttackResult lists
      attack_cvss_<split>_<model>.pkl   GA CVSS AttackResult lists
      comp_<fw>_<model>.pkl             comparison framework results
      d1_<model>_adv.pt                 D1 adversarially retrained weights
      d5_mlp.pt                         D5 PDG-Taint MLP weights
      cvss_table.pkl                    serialised CVSSTable for offline resume
    """

    def __init__(self, output_dir: str, args):
        self.ckpt_dir  = os.path.join(output_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.args_hash = self._hash_args(args)
        self.state     = self._load_state()

    # ── hashing ───────────────────────────────────────────────────────────────
    def _hash_args(self, args) -> str:
        key = {
            "models":         sorted(getattr(args, "models",  [])),
            "splits":         sorted(getattr(args, "splits",  [])),
            "attack_samples": getattr(args, "attack_samples", 0),
            "query_budget":   getattr(args, "query_budget",   0),
            "smt_verify":     getattr(args, "smt_verify",     False),
            "cvss_lambda":    getattr(args, "cvss_lambda",    0.0),
            "seed":           getattr(args, "seed",           42),
        }
        return hashlib.md5(
            json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]

    # ── state I/O ─────────────────────────────────────────────────────────────
    def _state_path(self) -> str:
        return os.path.join(self.ckpt_dir, "state.json")

    def _hash_path(self) -> str:
        return os.path.join(self.ckpt_dir, "args_hash.txt")

    def _load_state(self) -> dict:
        if os.path.exists(self._state_path()):
            if os.path.exists(self._hash_path()):
                saved = open(self._hash_path()).read().strip()
                if saved != self.args_hash:
                    logger.warning(
                        "Checkpoint: key args changed (%s → %s). "
                        "Clearing all saved state — all steps will re-run.",
                        saved, self.args_hash)
                    return self._fresh()
            state = json.load(open(self._state_path()))
            done  = [k for k, v in state.items() if v]
            logger.info("Checkpoint: %d completed steps found: %s",
                        len(done), done)
            return state
        return self._fresh()

    def _fresh(self) -> dict:
        with open(self._hash_path(), "w") as f:
            f.write(self.args_hash)
        state: dict = {}
        self._flush(state)
        return state

    def _flush(self, state: dict):
        with open(self._state_path(), "w") as f:
            json.dump(state, f, indent=2)

    # ── public API ────────────────────────────────────────────────────────────
    def step_done(self, key: str) -> bool:
        return bool(self.state.get(key, False))

    def mark_done(self, key: str):
        self.state[key] = True
        self._flush(self.state)
        logger.info("Checkpoint ✓  '%s'", key)

    def status_summary(self) -> str:
        done = [k for k, v in self.state.items() if v]
        return (f"{len(done)} completed steps: {done}"
                if done else "No checkpoints yet — starting fresh.")

    # ── model weights ─────────────────────────────────────────────────────────
    def model_path(self, name: str) -> str:
        return os.path.join(self.ckpt_dir, f"model_{name}.pt")

    def model_cached(self, name: str) -> bool:
        return os.path.exists(self.model_path(name))

    # ── GA attack results ─────────────────────────────────────────────────────
    def _atk_key(self, split: str, model: str, cvss: bool) -> str:
        return f"{'cvss_' if cvss else ''}attack_{split}_{model}"

    def _atk_path(self, split: str, model: str, cvss: bool) -> str:
        return os.path.join(self.ckpt_dir,
                            f"{'attack_cvss' if cvss else 'attack'}"
                            f"_{split}_{model}.pkl")

    def attack_done(self, split: str, model: str, cvss: bool = False) -> bool:
        return self.step_done(self._atk_key(split, model, cvss))

    def save_attack(self, split: str, model: str,
                    results: list, cvss: bool = False):
        with open(self._atk_path(split, model, cvss), "wb") as f:
            pickle.dump(results, f)
        self.mark_done(self._atk_key(split, model, cvss))
        logger.info("  Attack results saved: %s (%d samples)",
                    self._atk_key(split, model, cvss), len(results))

    def load_attack(self, split: str, model: str,
                    cvss: bool = False) -> list:
        with open(self._atk_path(split, model, cvss), "rb") as f:
            res = pickle.load(f)
        logger.info("  Loaded %s (%d samples) — skipping re-attack.",
                    self._atk_key(split, model, cvss), len(res))
        return res

    # ── comparison framework results ──────────────────────────────────────────
    def comp_done(self, fw: str, model: str) -> bool:
        return self.step_done(f"comp_{fw}_{model}")

    def save_comp(self, fw: str, model: str, results: list):
        p = os.path.join(self.ckpt_dir, f"comp_{fw}_{model}.pkl")
        with open(p, "wb") as f:
            pickle.dump(results, f)
        self.mark_done(f"comp_{fw}_{model}")

    def load_comp(self, fw: str, model: str) -> list:
        p = os.path.join(self.ckpt_dir, f"comp_{fw}_{model}.pkl")
        with open(p, "rb") as f:
            return pickle.load(f)

    # ── per-split comparison checkpoints ─────────────────────────────────────
    def comp_split_done(self, fw: str, model: str, split: str) -> bool:
        return self.step_done(f"comp_{fw}_{model}_{split}")

    def save_comp_split(self, fw: str, model: str, split: str, results: list):
        p = os.path.join(self.ckpt_dir, f"comp_{fw}_{model}_{split}.pkl")
        with open(p, "wb") as f:
            pickle.dump(results, f)
        self.mark_done(f"comp_{fw}_{model}_{split}")

    def load_comp_split(self, fw: str, model: str, split: str) -> list:
        p = os.path.join(self.ckpt_dir, f"comp_{fw}_{model}_{split}.pkl")
        with open(p, "rb") as f:
            return pickle.load(f)

    # ── defense checkpoints ───────────────────────────────────────────────────
    def d1_path(self, model: str) -> str:
        return os.path.join(self.ckpt_dir, f"d1_{model}_adv.pt")

    def save_d5(self, d5):
        import torch
        torch.save(d5.model.state_dict(),
                   os.path.join(self.ckpt_dir, "d5_mlp.pt"))
        self.mark_done("d5_trained")

    def load_d5(self, d5):
        import torch
        path = os.path.join(self.ckpt_dir, "d5_mlp.pt")
        d5.model.load_state_dict(torch.load(path, map_location="cpu"))
        d5.model.eval()
        return d5

    # ── CVSS table ────────────────────────────────────────────────────────────
    def save_cvss_table(self, table: CVSSTable):
        p = os.path.join(self.ckpt_dir, "cvss_table.pkl")
        with open(p, "wb") as f:
            pickle.dump(table, f)
        self.mark_done("cvss_table")

    def load_cvss_table(self) -> CVSSTable:
        p = os.path.join(self.ckpt_dir, "cvss_table.pkl")
        with open(p, "rb") as f:
            return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════════════
#  Pretrained model loader
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_name: str,
               ckpt: CheckpointManager,
               args,
               train_codes:  List[str],
               train_labels: List[int],
               val_codes:    List[str],
               val_labels:   List[int]) -> VulnDetector:
    """
    Resolution order per model:

    LineVul (michaelfu1998/linevul)
      1. local .pt  →  load state_dict into LineVulDetector
      2. HF fine-tuned weights (michaelfu1998/linevul) →
            load encoder weights into LineVulDetector.encoder,
            save as local .pt
      3. fine-tune LineVulDetector from microsoft/codebert-base

    ReVeal / IVDetect  (no public fine-tuned checkpoint)
      1. local .pt  →  load state_dict into custom architecture
      2. (skipped — no HF checkpoint exists)
      3. fine-tune custom architecture from scratch using CodeBERT tokenizer

    GraphCodeBERT  (microsoft/graphcodebert-base)
      1. local .pt  →  load state_dict into GraphCodeBERTDetector
      2. GraphCodeBERTDetector._build() ALREADY downloads graphcodebert-base
         from HF automatically on first use — nothing extra needed
      3. fine-tune GraphCodeBERTDetector

    --force_retrain skips steps 1 and 2 for all models.
    """
    det      = VulnDetector(model_name)   # _build() runs here
    local_pt = ckpt.model_path(model_name)

    # ── Step 1: local cached weights (fastest — no internet, no GPU) ──────────
    if ckpt.model_cached(model_name) and not args.force_retrain:
        det.load(local_pt)
        logger.info("  [%s] Loaded from local cache: %s", model_name, local_pt)
        return det

    # ── Step 2: HF fine-tuned weights (only LineVul has a public one) ─────────
    # NOTE: GraphCodeBERT already has its backbone loaded in _build(), so step 2
    # is only meaningful for LineVul where the community fine-tuned checkpoint
    # (michaelfu1998/linevul) contains weights that map onto LineVulDetector.encoder.
    if not args.force_retrain:
        if _try_hf_load_linevul(det, model_name):
            det.save(local_pt)
            logger.info("  [%s] HF pretrained weights loaded and cached → %s",
                        model_name, local_pt)
            return det

    # ── Step 3: fine-tune (runs once, result cached as local_pt) ─────────────
    # VulnDetector._build() already loaded the correct backbone weights:
    #   LineVul       → microsoft/codebert-base encoder (random head)
    #   ReVeal        → random BiGRU (CodeBERT tokenizer only)
    #   IVDetect      → random LSTM+CNN (CodeBERT tokenizer only)
    #   GraphCodeBERT → microsoft/graphcodebert-base (pretrained, just needs head)
    # So fine-tuning here is always starting from the right backbone.
    logger.info("  [%s] Fine-tuning …", model_name)
    trainer = VulnDetectorTrainer(det, lr=args.lr)
    trainer.train(
        train_codes, train_labels,
        val_codes,   val_labels,
        epochs     = args.train_epochs,
        batch_size = args.batch_size,
        max_len    = args.max_len,
        save_path  = local_pt,
    )
    logger.info("  [%s] Fine-tune complete → cached to %s", model_name, local_pt)
    return det


def _try_hf_load_linevul(det: VulnDetector, model_name: str) -> bool:
    """
    Load the community fine-tuned LineVul checkpoint from HuggingFace
    (michaelfu1998/linevul) into the existing LineVulDetector.

    What this does:
      - Downloads the HF model (microsoft/codebert-base backbone + fine-tuned
        head from michaelfu1998/linevul)
      - Copies the encoder (BERT) weights into det.model.encoder
      - Copies the classifier head weights if the shapes match

    Only applies to LineVul.  For all other models returns False immediately
    so the caller falls through to fine-tuning.

    Why this is safe:
      LineVulDetector.encoder   = AutoModel (CodeBERT, 12 layers, 768-dim)
      michaelfu1998/linevul     = same CodeBERT backbone, fine-tuned on BigVul
      → weight keys are identical; load_state_dict(strict=False) handles any
        mismatched head shapes gracefully.
    """
    if model_name != "linevul":
        return False          # ReVeal/IVDetect: custom arch, no HF ckpt exists
                              # GraphCodeBERT: backbone already loaded in _build()

    hf_id = FINETUNED_HF.get("linevul")
    if not hf_id:
        return False

    try:
        from transformers import AutoModel as _AutoModel, AutoTokenizer as _AutoTok
        logger.info("  [linevul] Downloading %s …", hf_id)

        # Download the HF model — this is the fine-tuned CodeBERT encoder
        hf_model = _AutoModel.from_pretrained(hf_id)
        hf_state = hf_model.state_dict()

        # Our LineVulDetector stores the encoder under det.model.encoder
        # Key mapping: HF keys are bare (e.g. "embeddings.word_embeddings.weight")
        # Our keys are prefixed with "encoder." 
        # We remap and load with strict=False so the classifier head is skipped
        # if it doesn't exist in the HF checkpoint (it often doesn't).
        remapped = {"encoder." + k: v for k, v in hf_state.items()}
        missing, unexpected = det.model.load_state_dict(remapped, strict=False)

        # Only warn if truly unexpected keys appear (not just the missing head)
        truly_unexpected = [k for k in unexpected if "encoder" in k]
        if truly_unexpected:
            logger.warning("  [linevul] Unexpected keys after HF load: %s",
                           truly_unexpected[:5])

        encoder_loaded = sum(1 for k in remapped if k not in missing)
        logger.info("  [linevul] Loaded %d / %d encoder weight tensors from %s",
                    encoder_loaded, len(remapped), hf_id)

        det.model.to(det.device)
        det.model.eval()
        return encoder_loaded > 0

    except Exception as exc:
        logger.warning("  [linevul] HF load failed (%s) — will fine-tune instead.",
                       exc)
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset split builder
# ══════════════════════════════════════════════════════════════════════════════

def build_splits(args) -> Dict[str, dict]:
    """
    Builds one dict entry per requested split.  Each entry has:
      train_codes / train_labels / val_codes / val_labels /
      test_codes  / test_labels  / attack_samples
    """
    cache = args.cache_dir
    seed  = args.seed

    def _partition(samples: list) -> Tuple[list, list, list]:
        random.seed(seed)
        random.shuffle(samples)
        n     = len(samples)
        n_tr  = int(n * 0.70)
        n_val = int(n * 0.15)
        return (samples[:n_tr],
                samples[n_tr: n_tr + n_val],
                samples[n_tr + n_val:])

    def _pack(tr, va, te) -> dict:
        atk = get_attack_subset(te, n=args.attack_samples, seed=seed)
        return {
            "train_codes":    [s.code  for s in tr],
            "train_labels":   [s.label for s in tr],
            "val_codes":      [s.code  for s in va],
            "val_labels":     [s.label for s in va],
            "test_codes":     [s.code  for s in te],
            "test_labels":    [s.label for s in te],
            "attack_samples": atk,
        }

    out: Dict[str, dict] = {}

    if "bigvul" in args.splits:
        logger.info("Loading BigVul split …")
        raw = BigVulLoader(cache).load(max_samples=args.max_bigvul,
                                       balance=True, cwe_filter=TARGET_CWES)
        out["bigvul"] = _pack(*_partition(raw))
        _log_split("bigvul", out["bigvul"])

    if "devign" in args.splits:
        logger.info("Loading Devign split …")
        raw = DevignLoader(cache).load(max_samples=args.max_devign,
                                       balance=True)
        out["devign"] = _pack(*_partition(raw))
        _log_split("devign", out["devign"])

    if "sard" in args.splits:
        logger.info("Loading SARD split …")
        raw = SARDLoader(cache).load(max_samples=args.max_sard,
                                     balance=True, cwe_filter=TARGET_CWES)
        out["sard"] = _pack(*_partition(raw))
        _log_split("sard", out["sard"])

    if "primevul" in args.splits:
        logger.info("Loading PrimeVul split …")
        raw = PrimeVulLoader(cache).load(
            max_samples=args.max_primevul, balance=True)
        out["primevul"] = _pack(*_partition(raw))
        _log_split("primevul", out["primevul"])

    if "combined" in args.splits:
        logger.info("Loading combined (BigVul + Devign + SARD + PrimeVul) split …")
        tr, va, te = load_dataset(cache_dir=cache,
                                   max_bigvul=args.max_bigvul,
                                   max_devign=args.max_devign,
                                   max_sard=args.max_sard,
                                   max_primevul=args.max_primevul)
        atk = get_attack_subset(te, n=args.attack_samples, seed=seed)
        out["combined"] = {
            "train_codes":    [s.code  for s in tr],
            "train_labels":   [s.label for s in tr],
            "val_codes":      [s.code  for s in va],
            "val_labels":     [s.label for s in va],
            "test_codes":     [s.code  for s in te],
            "test_labels":    [s.label for s in te],
            "attack_samples": atk,
        }
        _log_split("combined", out["combined"])

    return out


def _log_split(name: str, d: dict):
    logger.info("  %-10s  train=%d  val=%d  test=%d  attack=%d",
                name,
                len(d["train_codes"]),  len(d["val_codes"]),
                len(d["test_codes"]),   len(d["attack_samples"]))


# ══════════════════════════════════════════════════════════════════════════════
#  GA runner  (shared for flat and CVSS modes)
# ══════════════════════════════════════════════════════════════════════════════

def _safe_pop_size(args) -> int:
    """Cap pop_size so the GA always gets at least 5 generations."""
    n_elite   = max(1, int(args.pop_size * 0.2))
    offspring = args.pop_size - n_elite
    return min(
        args.pop_size,
        max(4, (args.query_budget - 1) // max(offspring, 1) // 5)
        if args.query_budget < args.pop_size * 6 else args.pop_size,
    )


def run_ga(samples:          List[VulnSample],
           detector:         VulnDetector,
           args,
           exploit_verifier: ExploitVerifier,
           cvss_table:       Optional[CVSSTable] = None) -> List[AttackResult]:
    """
    Run the GA attack.  Filters out samples the detector does not detect
    (nothing to evade), then runs the GA.  If cvss_table is provided the
    GA uses CVSS-weighted fitness; otherwise flat fitness.
    """
    detected = [s for s in samples if detector.predict(s.code) >= 0.5]
    if not detected:
        logger.warning("  Detector flags 0 / %d samples — skipping.", len(samples))
        return []
    logger.info("  Detector flags %d / %d samples.",
                len(detected), len(samples))

    attacker = GeneticAttacker(
        predict_fn       = detector.predict,
        pop_size         = _safe_pop_size(args),
        max_gens         = args.max_gens,
        query_budget     = args.query_budget,
        mutation_rate    = args.mutation_rate,
        smt_verify       = args.smt_verify,
        exploit_verify   = args.exploit_verify,
        exploit_verifier = exploit_verifier,
        seed             = args.seed,
    )

    if cvss_table is not None:
        active = CVSSFitnessWrapper(attacker, cvss_table)
        mode   = f"CVSS-GA (λ={args.cvss_lambda:.2f})"
    else:
        active = attacker
        mode   = "ShadowPatch-GA (flat)"

    logger.info("  [%s] running on %d samples …", mode, len(detected))
    results = active.batch_attack(detected, show_progress=True,
                                   desc=f"  {mode[:20]}")

    if cvss_table is not None:
        logger.info("\n%s", cvss_imputation_report(results))

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Comparison frameworks  (inline, self-contained implementations)
# ══════════════════════════════════════════════════════════════════════════════

_C_KEYWORDS = {
    "int", "char", "void", "return", "if", "else", "for", "while",
    "struct", "sizeof", "NULL", "free", "malloc", "memcpy", "printf",
    "static", "const", "unsigned", "long", "short", "double", "float",
    "do", "switch", "case", "break", "continue", "include", "define",
    "typedef", "enum", "union", "extern", "register", "volatile",
}

def _comp_result(success, orig_prob, adv_prob, queries, method) -> dict:
    return {"success": success, "original_prob": orig_prob,
            "adversarial_prob": adv_prob,
            "prob_shift": round(orig_prob - adv_prob, 4),
            "queries": queries, "method": method}


def run_alert(samples: List[VulnSample],
              detector: VulnDetector,
              budget: int = 50) -> List[dict]:
    """
    ALERT  (ICSE 2022) — naturalness-constrained identifier renaming.
    Greedily renames one identifier at a time, picking the rename that
    most reduces the vulnerability probability.
    Reference: Yefet et al., ALERT, ICSE 2022.
    """
    results = []
    bar = tqdm(samples, desc="  ALERT", unit="sample",
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    for s in bar:
        op = detector.predict(s.code)
        if op < 0.5:
            results.append(_comp_result(False, op, op, 0, "ALERT"))
            continue

        idents = [i for i in set(
                      re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b', s.code))
                  if i not in _C_KEYWORDS][:budget]

        best_code, best_prob, queries, success = s.code, op, 0, False
        for ident in idents:
            for sfx in ("_val", "_tmp", "_buf", "_ptr", "_data", "_len"):
                candidate = re.sub(r'\b' + re.escape(ident) + r'\b',
                                   ident + sfx, best_code)
                p = detector.predict(candidate)
                queries += 1
                if p < best_prob:
                    best_prob, best_code = p, candidate
                if p < 0.5:
                    success = True
                    break
            if success:
                break

        results.append(_comp_result(success, op, best_prob, queries, "ALERT"))
    return results


def run_mhm(samples: List[VulnSample],
            detector: VulnDetector,
            n_steps: int = 100) -> List[dict]:
    """
    MHM  (ACL 2020) — Metropolis-Hastings MCMC identifier renaming.
    Uses a random walk with a Metropolis acceptance criterion to explore
    the space of identifier renames.
    Reference: Zhang et al., MHM, ACL 2020.
    """
    SYNONYMS = {
        "buf": ["buffer", "tmp_buf", "data_buf", "rbuf"],
        "len": ["length", "size", "n_bytes", "count"],
        "ptr": ["p", "tmp_ptr", "addr", "ref"],
        "src": ["source", "input", "in_buf", "src_data"],
        "dst": ["dest", "output", "out_buf", "dst_data"],
        "i":   ["idx", "iter", "pos", "k"],
        "n":   ["num", "total", "cnt", "amount"],
        "ret": ["result", "retval", "status", "rv"],
        "tmp": ["temp", "scratch", "intermediate", "aux"],
    }
    T = 0.1  # temperature

    def _propose(code: str) -> str:
        for orig, reps in SYNONYMS.items():
            if re.search(r'\b' + orig + r'\b', code):
                return re.sub(r'\b' + re.escape(orig) + r'\b',
                              random.choice(reps), code, count=1)
        ids = re.findall(r'\b[a-z][a-z0-9]{1,3}\b', code)
        if ids:
            orig = random.choice(ids)
            return re.sub(r'\b' + re.escape(orig) + r'\b',
                          orig + "_r", code, count=1)
        return code

    results = []
    bar = tqdm(samples, desc="  MHM  ", unit="sample",
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    for s in bar:
        op = detector.predict(s.code)
        if op < 0.5:
            results.append(_comp_result(False, op, op, 0, "MHM"))
            continue

        cur_code, cur_prob = s.code, op
        best_code, best_prob, success = s.code, op, False

        for step in range(n_steps):
            proposal   = _propose(cur_code)
            prop_prob  = detector.predict(proposal)
            delta      = prop_prob - cur_prob
            if delta < 0 or random.random() < math.exp(-delta / T):
                cur_code, cur_prob = proposal, prop_prob
                if cur_prob < best_prob:
                    best_prob, best_code = cur_prob, cur_code
            if best_prob < 0.5:
                success = True
                break

        results.append(_comp_result(success, op, best_prob, n_steps, "MHM"))
    return results


def run_dead_rename(samples: List[VulnSample],
                    detector: VulnDetector) -> List[dict]:
    """
    Dead+Rename baseline — applies T13 (dead branch) + T14 (rename)
    jointly in a single pass, no GA, no iteration.
    """
    from ShadowPatch_Attack.code_transformer import apply_sequence
    from ShadowPatch_Attack.pdg_taint        import get_taint_set
    results = []
    bar = tqdm(samples, desc="  Dead+Rename", unit="sample",
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    for s in bar:
        op = detector.predict(s.code)
        if op < 0.5:
            results.append(_comp_result(False, op, op, 1, "Dead+Rename"))
            continue
        _, safe = get_taint_set(s.code, hops=2)
        adv     = apply_sequence(s.code, ["T13", "T14"], safe)
        p       = detector.predict(adv)
        results.append(_comp_result(p < 0.5, op, p, 2, "Dead+Rename"))
    return results


def run_random(samples: List[VulnSample],
               detector: VulnDetector,
               budget: int = 150) -> List[dict]:
    """
    Random baseline — random transform sequences with no fitness guidance.
    Uses the same transform operators as ShadowPatch but no GA.
    """
    from ShadowPatch_Attack.code_transformer import TRANSFORM_IDS, apply_sequence
    from ShadowPatch_Attack.pdg_taint        import get_taint_set
    tids = list(TRANSFORM_IDS)
    results = []
    bar = tqdm(samples, desc="  Random", unit="sample",
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    for s in bar:
        op = detector.predict(s.code)
        if op < 0.5:
            results.append(_comp_result(False, op, op, 0, "Random"))
            continue
        _, safe   = get_taint_set(s.code, hops=2)
        best_prob = op
        success   = False
        q         = 0
        for q in range(budget):
            seq  = random.choices(tids, k=random.randint(1, 5))
            adv  = apply_sequence(s.code, seq, safe)
            p    = detector.predict(adv)
            if p < best_prob:
                best_prob = p
            if p < 0.5:
                success = True
                break
        results.append(_comp_result(success, op, best_prob, q + 1, "Random"))
    return results


def _asr(results: List[dict]) -> float:
    return sum(1 for r in results if r["success"]) / max(len(results), 1)

def _avg_q(results: List[dict]) -> float:
    if not results:
        return 0.0
    return sum(r["queries"] for r in results) / len(results)


# ══════════════════════════════════════════════════════════════════════════════
#  Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="ShadowPatch v3 — full experiment runner")

    # ── Datasets ──────────────────────────────────────────────────────────────
    p.add_argument("--cache_dir",  default="data/cache")
    p.add_argument("--max_bigvul", type=int, default=10_000)
    p.add_argument("--max_devign", type=int, default=5_000)
    p.add_argument("--max_sard",     type=int, default=3_000)
    p.add_argument("--max_primevul",  type=int, default=5_000,
                   help="Max PrimeVul samples (0 to exclude).")
    p.add_argument("--splits", nargs="+",
                   default=["combined"],
                   choices=["bigvul", "devign", "sard", "primevul", "combined"],
                   help="Dataset splits to evaluate independently. "
                        "Pass all four for the full per-dataset analysis.")

    # ── Models ────────────────────────────────────────────────────────────────
    p.add_argument("--models", nargs="+",
                   default=["linevul", "reveal", "ivdetect", "graphcodebert"],
                   choices=["linevul", "reveal", "ivdetect", "graphcodebert"])
    p.add_argument("--force_retrain", action="store_true",
                   help="Ignore all cached weights and re-download / retrain.")
    p.add_argument("--train_epochs",  type=int,   default=5)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=2e-5)
    p.add_argument("--max_len",       type=int,   default=512)
    p.add_argument("--model_dir",     default="checkpoints/")

    # ── Attack ────────────────────────────────────────────────────────────────
    p.add_argument("--attack_samples", type=int,   default=500)
    p.add_argument("--query_budget",   type=int,   default=500)
    p.add_argument("--pop_size",       type=int,   default=20)
    p.add_argument("--max_gens",       type=int,   default=50)
    p.add_argument("--mutation_rate",  type=float, default=0.4)
    p.add_argument("--smt_verify",     action="store_true",
                   help="Enable Z3 SMT semantic-equivalence verification.")
    p.add_argument("--exploit_verify", action="store_true",
                   help="Enable ASan / AFL++ exploit-survivability check.")
    p.add_argument("--use_aflpp",      action="store_true")
    p.add_argument("--afl_timeout",    type=float, default=60.0)

    # ── CVSS-weighted fitness ─────────────────────────────────────────────────
    p.add_argument("--cvss_lambda",  type=float, default=0.0,
                   help="Severity weight λ.  0.0 = flat fitness (default). "
                        "0.3 = recommended.  Must be > 0 to activate.")
    p.add_argument("--cvss_offline", action="store_true",
                   help="Skip NVD HTTP calls (use CWE-mean / median only). "
                        "Required on HPC compute nodes without internet.")

    # ── Comparison frameworks ─────────────────────────────────────────────────
    p.add_argument("--run_comparison", action="store_true", default=True,
                   help="Run ALERT, MHM, Dead+Rename, Random baselines.")
    p.add_argument("--comp_budget",    type=int, default=100,
                   help="Query budget for each comparison framework.")

    # ── Defense ───────────────────────────────────────────────────────────────
    p.add_argument("--eval_defenses",    action="store_true", default=True)
    p.add_argument("--adv_train_epochs", type=int, default=3)

    # ── Ablation ──────────────────────────────────────────────────────────────
    p.add_argument("--run_ablation", action="store_true")

    # ── Output / misc ─────────────────────────────────────────────────────────
    p.add_argument("--output_dir", default="results/")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--log_level",  default="INFO")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    configure_logging(args.log_level)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir,  exist_ok=True)

    logger.info("=" * 72)
    logger.info("ShadowPatch v3")
    logger.info("  Models  : %s", args.models)
    logger.info("  Splits  : %s", args.splits)
    logger.info("  CVSS λ  : %.2f", args.cvss_lambda)
    logger.info("=" * 72)

    ckpt = CheckpointManager(args.output_dir, args)
    logger.info("Checkpoint: %s", ckpt.status_summary())

    # ══════════════════════════════════════════════════════════════════════════
    #  Step 1 — Datasets
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n── Step 1: Loading datasets ──────────────────────────────")
    splits_data = build_splits(args)

    # Training data: prefer combined (most diverse); fall back to first split
    train_key    = "combined" if "combined" in splits_data else args.splits[0]
    sd_train     = splits_data[train_key]
    train_codes  = sd_train["train_codes"]
    train_labels = sd_train["train_labels"]
    val_codes    = sd_train["val_codes"]
    val_labels   = sd_train["val_labels"]
    test_codes   = sd_train["test_codes"]
    test_labels  = sd_train["test_labels"]

    # ══════════════════════════════════════════════════════════════════════════
    #  Step 2 — Models
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n── Step 2: Loading models (pretrained where available) ───")
    detectors: Dict[str, VulnDetector] = {}
    for mname in args.models:
        logger.info("  [%s] resolving weights …", mname)
        det = load_model(mname, ckpt, args,
                         train_codes, train_labels,
                         val_codes,   val_labels)
        logger.info("  [%s] ready  (%.1f M params)", mname,
                    det.param_count() / 1e6)
        detectors[mname] = det

    evaluator = ShadowPatchEvaluator(output_dir=args.output_dir)

    # ══════════════════════════════════════════════════════════════════════════
    #  Step 3 — Clean performance  →  Table 1
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n── Step 3: Clean model performance ──────────────────────")
    t1 = evaluator.eval_clean_performance(detectors, test_codes, test_labels)
    logger.info("\n%s", t1.to_string(index=False))

    # ══════════════════════════════════════════════════════════════════════════
    #  Step 4a — ShadowPatch GA (flat fitness)  →  Table 2
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n── Step 4a: GA flat fitness attack ─────────────────────")
    ev = ExploitVerifier(use_aflpp=args.use_aflpp,
                         afl_timeout=args.afl_timeout)

    # results_flat[split][model] = List[AttackResult]
    results_flat: Dict[str, Dict[str, List[AttackResult]]] = defaultdict(dict)

    split_model_pairs = [(s, m) for s in splits_data for m in args.models]
    outer_bar = tqdm(split_model_pairs, desc="Step 4a",
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} splits×models [{elapsed}<{remaining}]")
    for sname, sdata in splits_data.items():
        logger.info("  === Split: %s ===", sname)
        for mname, det in detectors.items():
            outer_bar.set_postfix(split=sname, model=mname)
            outer_bar.update(1)
            if ckpt.attack_done(sname, mname, cvss=False):
                results_flat[sname][mname] = ckpt.load_attack(
                    sname, mname, cvss=False)
                continue

            res = run_ga(sdata["attack_samples"], det, args, ev)
            results_flat[sname][mname] = res
            ckpt.save_attack(sname, mname, res, cvss=False)

            s = summarise_results(res)
            logger.info("  ✓ [%s / %s]  ASR=%.1f%%  AvgQ=%.0f  "
                        "SMT-OK=%d  Expl-OK=%d",
                        sname, mname, s["asr"] * 100, s["avg_queries"],
                        s["smt_verified"], s["exploit_preserved"])

    # Build Table 2  (multi-split ASR matrix)
    t2 = evaluator.eval_attack_per_split(results_flat, args.splits, args.models)
    logger.info("\n%s", t2.to_string(index=False))

    # ══════════════════════════════════════════════════════════════════════════
    #  Step 4b — CVSS-weighted GA  (only when --cvss_lambda > 0)
    # ══════════════════════════════════════════════════════════════════════════
    results_cvss: Dict[str, Dict[str, List[AttackResult]]] = defaultdict(dict)
    cvss_table: Optional[CVSSTable] = None

    if args.cvss_lambda > 0.0:
        logger.info("\n── Step 4b: GA CVSS-weighted (λ=%.2f) ─────────────",
                    args.cvss_lambda)

        # Build / restore CVSS table
        if ckpt.step_done("cvss_table"):
            cvss_table = ckpt.load_cvss_table()
            logger.info("  CVSS table restored from checkpoint.")
        else:
            all_atk = [s for sd in splits_data.values()
                         for s in sd["attack_samples"]]
            cvss_table = CVSSTable.build(
                samples    = all_atk,
                cache_path = os.path.join(args.cache_dir, "cvss_cache.json"),
                lambda_    = args.cvss_lambda,
                offline    = args.cvss_offline,
            )
            ckpt.save_cvss_table(cvss_table)
            logger.info("  CVSS table built and saved.")

        for sname, sdata in splits_data.items():
            logger.info("  === Split: %s (CVSS) ===", sname)
            for mname, det in detectors.items():
                if ckpt.attack_done(sname, mname, cvss=True):
                    results_cvss[sname][mname] = ckpt.load_attack(
                        sname, mname, cvss=True)
                    continue

                res = run_ga(sdata["attack_samples"], det, args, ev,
                             cvss_table=cvss_table)
                results_cvss[sname][mname] = res
                ckpt.save_attack(sname, mname, res, cvss=True)

                s = summarise_results(res)
                logger.info("  ✓ [CVSS/%s/%s]  ASR=%.1f%%  AvgQ=%.0f",
                            sname, mname, s["asr"] * 100, s["avg_queries"])

        t2_cvss = evaluator.eval_attack_per_split(
            results_cvss, args.splits, args.models, tag="CVSS")
        logger.info("\n%s", t2_cvss.to_string(index=False))

    # ══════════════════════════════════════════════════════════════════════════
    #  Step 5 — Transferability  →  Table 3  (combined split)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n── Step 5: Transferability ──────────────────────────────")
    best_split = "combined" if "combined" in results_flat else args.splits[0]
    active_flat = {k: v for k, v in results_flat[best_split].items() if v}
    if len(active_flat) > 1:
        t3 = evaluator.eval_transferability(detectors, active_flat)
        logger.info("\n%s", t3.to_string())
    else:
        logger.info("  Skipping — need >= 2 models with results.")

    # ══════════════════════════════════════════════════════════════════════════
    #  Step 6 — CWE-specific  →  Table 4
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n── Step 6: CWE-specific analysis ────────────────────────")
    all_flat = [r for m_res in results_flat.values()
                  for res  in m_res.values()
                  for r    in res]
    t4 = None
    if all_flat:
        t4 = evaluator.eval_cwe_specific(all_flat)
        logger.info("\n%s", t4.to_string(index=False))
    else:
        logger.warning("  No results available — check model loading.")

    # ══════════════════════════════════════════════════════════════════════════
    #  Step 7 — Defense evaluation  →  Table 5
    # ══════════════════════════════════════════════════════════════════════════
    if args.eval_defenses:
        logger.info("\n── Step 7: Defense evaluation ───────────────────────")
        pm     = args.models[0]
        pdet   = detectors[pm]
        pres   = results_flat.get(best_split, {}).get(pm, [])

        if not pres:
            logger.warning("  No attack results for primary model — skipping.")
        else:
            adv_ok = [r.adversarial_code for r in pres if r.success]
            if not adv_ok:
                logger.warning("  Zero successful attacks — defenses skipped.")
            else:
                # balanced eval set: adversarial (label=1) + clean (label=0)
                clean_fp = [c for c, l in zip(
                    splits_data[best_split]["test_codes"],
                    splits_data[best_split]["test_labels"]) if l == 0
                ][:len(adv_ok)]
                eval_codes  = adv_ok + clean_fp
                eval_labels = [1] * len(adv_ok) + [0] * len(clean_fp)

                # D1 — adversarial retraining
                if ckpt.step_done("d1_trained"):
                    logger.info("  D1: loading from checkpoint …")
                    adv_trainer = AdversarialTrainer(pdet)
                    d1_pt = ckpt.d1_path(pm)
                    if os.path.exists(d1_pt):
                        pdet.load(d1_pt)
                else:
                    logger.info("  D1: retraining with %d adv examples …",
                                len(adv_ok))
                    adv_trainer = AdversarialTrainer(pdet)
                    adv_trainer.augment_and_retrain(
                        train_codes, train_labels, adv_ok,
                        val_codes, val_labels,
                        epochs=args.adv_train_epochs,
                        batch_size=args.batch_size)
                    pdet.save(ckpt.d1_path(pm))
                    ckpt.mark_done("d1_trained")

                # D2–D4 — stateless, build each time
                d2 = EnsembleDefense(list(detectors.values()), mode="soft")
                d3 = RandomizedSmoothingDefense(pdet, n_samples=25)
                d4 = InputNormalizationDefense(pdet)

                # D5 — PDG-Taint MLP
                d5 = PDGTaintDefense()
                if ckpt.step_done("d5_trained"):
                    logger.info("  D5: loading from checkpoint …")
                    ckpt.load_d5(d5)
                else:
                    logger.info("  D5: training PDG-Taint MLP …")
                    d5.train(train_codes, train_labels,
                             val_codes, val_labels,
                             epochs=20, batch_size=64)
                    ckpt.save_d5(d5)

                defenses = {
                    "D1: Adv. Training":    adv_trainer,
                    "D2: Ensemble":         d2,
                    "D3: Rand. Smoothing":  d3,
                    "D4: Input Norm.":      d4,
                    "D5: PDG-Taint (ours)": d5,
                }
                base_asr = summarise_results(pres)["asr"]
                t5 = evaluator.eval_defenses(
                    defenses, eval_codes, eval_labels,
                    baseline_asr=base_asr)
                logger.info("\n%s", t5.to_string(index=False))
                ckpt.mark_done("defenses")

    # ══════════════════════════════════════════════════════════════════════════
    #  Step 8 — Ablation  →  Table 6
    # ══════════════════════════════════════════════════════════════════════════
    if args.run_ablation:
        logger.info("\n── Step 8: Ablation study ───────────────────────────")
        atk_ab = splits_data.get("combined",
                                  list(splits_data.values())[0])
        t6 = evaluator.eval_ablation(
            detectors[args.models[0]],
            atk_ab["attack_samples"],
            query_budget=args.query_budget)
        logger.info("\n%s", t6.to_string(index=False))
        evaluator.plot_ablation(t6)

    # ══════════════════════════════════════════════════════════════════════════
    #  Step 9 — Comparison frameworks  →  Table 7 (per-split × per-model)
    # ══════════════════════════════════════════════════════════════════════════
    # comp_results_split[split][model][fw_name] = List[dict]
    comp_results_split: Dict[str, Dict[str, Dict[str, List[dict]]]] =         defaultdict(lambda: defaultdict(dict))

    # Keep original comp_results for backward-compat (best_split, first model)
    comp_results: Dict[str, List[dict]] = {}
    pm   = args.models[0]
    pdet = detectors[pm]

    if args.run_comparison:
        logger.info("\n── Step 9: Comparison frameworks (per-split × per-model) ───")

        def _make_frameworks(budget):
            return {
                "ALERT":       lambda s, d: run_alert(s, d, budget=budget),
                "MHM":         lambda s, d: run_mhm(s, d, n_steps=budget),
                "Dead+Rename": lambda s, d: run_dead_rename(s, d),
                "Random":      lambda s, d: run_random(s, d, budget=budget),
            }

        n_combos = len(splits_data) * len(args.models) * 4  # 4 frameworks
        step9_bar = tqdm(total=n_combos, desc="Step 9",
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        for sname, sdata in splits_data.items():
            logger.info("  === Split: %s ===", sname)
            for mname, det in detectors.items():
                # detect vulnerable samples for this split+model
                samp = sdata["attack_samples"]
                detected = [s for s in samp if det.predict(s.code) >= 0.5]
                if not detected:
                    logger.info("    [%s/%s] no detected samples — skip",
                                sname, mname)
                    step9_bar.update(4)
                    continue

                frameworks = _make_frameworks(args.comp_budget)
                for fw_name, fw_fn in frameworks.items():
                    # Check per-split checkpoint
                    if ckpt.comp_split_done(fw_name, mname, sname):
                        res = ckpt.load_comp_split(fw_name, mname, sname)
                        comp_results_split[sname][mname][fw_name] = res
                        logger.info("    [%s/%s/%s] ckpt  ASR=%.1f%%",
                                    sname, mname, fw_name, _asr(res) * 100)
                        step9_bar.set_postfix(split=sname, model=mname, fw=fw_name, status="ckpt")
                        step9_bar.update(1)
                        # Also populate comp_results for best_split/pm
                        if sname == best_split and mname == pm:
                            comp_results[fw_name] = res
                        continue

                    logger.info("    Running %s [%s/%s] (%d samples)…",
                                fw_name, sname, mname, len(detected))
                    res = fw_fn(detected, det)
                    comp_results_split[sname][mname][fw_name] = res
                    ckpt.save_comp_split(fw_name, mname, sname, res)

                    # Backward-compat: keep best_split/pm results
                    if sname == best_split and mname == pm:
                        comp_results[fw_name] = res
                        ckpt.save_comp(fw_name, pm, res)

                    logger.info("    ✓ [%s/%s/%s]  ASR=%.1f%%  AvgQ=%.0f",
                                sname, mname, fw_name,
                                _asr(res) * 100, _avg_q(res))
                    step9_bar.set_postfix(split=sname, model=mname, fw=fw_name,
                                          asr=f"{_asr(res)*100:.1f}%")
                    step9_bar.update(1)

        # Table 7: summary for best_split / primary model
        if comp_results:
            sp_res = results_flat.get(best_split, {}).get(pm, [])
            sp_asr = summarise_results(sp_res)["asr"] if sp_res else 0.0
            sp_q   = summarise_results(sp_res)["avg_queries"] if sp_res else 0.0
            t7 = evaluator.eval_comparison(comp_results, sp_asr, sp_q, pm)
            logger.info("\n%s", t7.to_string(index=False))

            # Table 7b: full per-split × per-model comparison
            try:
                t7b = evaluator.eval_comparison_per_split(
                    comp_results_split, results_flat,
                    args.splits, args.models)
                logger.info("\n── Per-split comparison ──\n%s",
                            t7b.to_string(index=False))
            except Exception as _exc:
                logger.warning("  Per-split comparison skipped: %s", _exc)

    # ══════════════════════════════════════════════════════════════════════════
    #  Step 10 — All figures
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n── Step 10: Generating CCS-style figures ────────────────")

    primary_flat = results_flat.get(best_split, {})
    active_pf    = {k: v for k, v in primary_flat.items() if v}

    # Fig 1: ASR vs query budget
    if active_pf:
        evaluator.plot_asr_vs_budget(active_pf)

    # Fig 2: Transform usage heatmap
    if active_pf:
        evaluator.plot_transform_heatmap(active_pf)

    # Fig 3: Probability shift distribution
    if all_flat:
        evaluator.plot_prob_shift(all_flat)

    # Fig 4: CWE-specific ASR
    if t4 is not None:
        evaluator.plot_cwe_asr(t4)

    # Fig 5: Per-split ASR comparison (BigVul / Devign / SARD / Combined)
    if len(splits_data) > 1:
        try:
            evaluator.plot_per_split_asr(results_flat, args.splits,
                                          args.models)
        except Exception as exc:
            logger.warning("  Per-split figure skipped: %s", exc)

    # Fig 6: Flat GA vs CVSS GA comparison
    if args.cvss_lambda > 0.0 and results_cvss:
        try:
            evaluator.plot_flat_vs_cvss(results_flat, results_cvss,
                                         args.splits, args.models,
                                         args.cvss_lambda)
        except Exception as exc:
            logger.warning("  Flat-vs-CVSS figure skipped: %s", exc)

    # Fig 7: ShadowPatch vs comparison frameworks
    if comp_results:
        try:
            sp_res = results_flat.get(best_split, {}).get(pm, [])
            sp_asr = summarise_results(sp_res)["asr"] if sp_res else 0.0
            evaluator.plot_comparison(comp_results, sp_asr, pm)
        except Exception as exc:
            logger.warning("  Comparison figure skipped: %s", exc)

    # Fig 8: CVSS severity-bucketed ASR
    if args.cvss_lambda > 0.0:
        all_cvss = [r for m_res in results_cvss.values()
                      for res  in m_res.values()
                      for r    in res]
        if all_cvss:
            try:
                evaluator.plot_cvss_weighted_asr(all_cvss, args.output_dir)
            except Exception as exc:
                logger.warning("  CVSS figure skipped: %s", exc)

    # ══════════════════════════════════════════════════════════════════════════
    #  Summary printout
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("All done!  Results in: %s", args.output_dir)
    logger.info("=" * 72)

    print("\n" + "=" * 72)
    print("SHADOWPATCH v3 — SUMMARY")
    print("=" * 72)

    for sname in args.splits:
        print(f"\n  Dataset split: {sname}")
        for mname in args.models:
            res = results_flat.get(sname, {}).get(mname, [])
            s   = summarise_results(res) if res else {}
            print(f"    {mname:20s}  "
                  f"ASR={s.get('asr', 0)*100:5.1f}%  "
                  f"AvgQ={s.get('avg_queries', 0):5.0f}")

    if args.cvss_lambda > 0.0 and results_cvss:
        print(f"\n  CVSS-weighted (λ={args.cvss_lambda})")
        for sname in args.splits:
            for mname in args.models:
                res = results_cvss.get(sname, {}).get(mname, [])
                s   = summarise_results(res) if res else {}
                print(f"    [{sname}] {mname:20s}  "
                      f"ASR={s.get('asr', 0)*100:5.1f}%")

    if comp_results:
        print("\n  Comparison frameworks:")
        sp_res = results_flat.get(best_split, {}).get(pm, [])
        sp_asr = summarise_results(sp_res)["asr"] if sp_res else 0.0
        print(f"    {'ShadowPatch-GA':20s}  ASR={sp_asr*100:5.1f}%")
        for fw, res in comp_results.items():
            print(f"    {fw:20s}  ASR={_asr(res)*100:5.1f}%  "
                  f"AvgQ={_avg_q(res):5.0f}")

    print("=" * 72)
    print(f"  Output: {args.output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
