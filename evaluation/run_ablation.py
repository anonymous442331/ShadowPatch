"""
run_ablation.py  —  ShadowPatch ablation study
===============================================
Mother path : /work/ShadowPatch_v2_Codebase/shadowpatch_v2

NO RETRAINING NEEDED.

If your checkpoint filename is different, set DET_CKPT at the top.
If the job is killed mid-run, just resubmit — finished configs are skipped.

Output:
    results/ablation/table_ablation.csv    numbers for the paper
    results/ablation/table_ablation.txt    human readable
    results/ablation/latex_rows.txt        paste into tab:ablation
    results/ablation/ablation_ckpt_*.pkl   per-config resume checkpoints
"""
from __future__ import annotations

import argparse, csv, logging, os, pickle, random, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS  —  only edit this block if your layout differs
# ══════════════════════════════════════════════════════════════════════════════
MOTHER   = Path("./")

# Your LineVul checkpoint — the one used for the combined-split attack results.
# Common names: linevul.pt  /  linevul_combined.pt  /  model_linevul.pt
# Check:  ls results/checkpoints/
DET_CKPT = MOTHER / "linevul.pt"

# Where your raw dataset files live (BigVul CSV, Devign JSON, SARD dir, PrimeVul JSONL)
CACHE    = MOTHER / "data/cache"

# Where results are written
OUT_DIR  = MOTHER / "results/ablation"
# ══════════════════════════════════════════════════════════════════════════════

sys.path.insert(0, str(MOTHER))

try:
    from attack.code_transformer import TRANSFORM_IDS, apply_sequence
    from attack.pdg_taint        import get_taint_set
    from attack.genetic_optimizer import AttackResult, Individual
except ImportError as e:
    logger.error("Cannot import ShadowPatch attack modules: %s", e)
    logger.error("Is MOTHER path correct? Is conda env cvmn active?")
    sys.exit(1)

ALL_T: List[str] = [
    "T1","T2","T3","T4",
    "T5","T6","T7","T8",
    "T9","T10","T11","T12","T13",
    "T14","T15","T16","T17","T18",
]

CONFIGS: Dict[str, Dict] = {
    "Full ShadowPatch":            {"allowed": ALL_T,
                                    "use_random": False, "skip_taint": False},
    "w/o loop (T1-T4)":           {"allowed": [t for t in ALL_T if t not in {"T1","T2","T3","T4"}],
                                    "use_random": False, "skip_taint": False},
    "w/o pointer (T5-T8)":        {"allowed": [t for t in ALL_T if t not in {"T5","T6","T7","T8"}],
                                    "use_random": False, "skip_taint": False},
    "w/o control-flow (T9-T13)":  {"allowed": [t for t in ALL_T if t not in {"T9","T10","T11","T12","T13"}],
                                    "use_random": False, "skip_taint": False},
    "w/o dead-code (T14-T18)":    {"allowed": [t for t in ALL_T if t not in {"T14","T15","T16","T17","T18"}],
                                    "use_random": False, "skip_taint": False},
    "w/o genetic search (random)":{"allowed": ALL_T,
                                    "use_random": True,  "skip_taint": False},
    "w/o PDG taint constraint":   {"allowed": ALL_T,
                                    "use_random": False, "skip_taint": True},
}


# ══════════════════════════════════════════════════════════════════════════════
#  Attacker  —  properly implements all three ablation dimensions
# ══════════════════════════════════════════════════════════════════════════════

class AblationAttacker:

    def __init__(self,
                 predict_fn:         Callable[[str], float],
                 allowed_transforms: List[str],
                 use_random:         bool  = False,
                 skip_taint:         bool  = False,
                 pop_size:           int   = 20,
                 max_gens:           int   = 50,
                 max_seq_len:        int   = 10,
                 query_budget:       int   = 500,
                 mutation_rate:      float = 0.4,
                 elite_frac:         float = 0.2,
                 tournament_k:       int   = 3,
                 threshold:          float = 0.5,
                 seed:               int   = 42):
        assert allowed_transforms
        self.predict_fn    = predict_fn
        self.allowed       = list(allowed_transforms)
        self.use_random    = use_random
        self.skip_taint    = skip_taint
        self.pop_size      = pop_size
        self.max_gens      = max_gens
        self.max_seq_len   = max_seq_len
        self.budget        = query_budget
        self.mutation_rate = mutation_rate
        self.elite_frac    = elite_frac
        self.tournament_k  = tournament_k
        self.threshold     = threshold
        random.seed(seed)
        np.random.seed(seed)

    def attack(self, code: str, cwe: str = "", source: str = "") -> AttackResult:
        t0        = time.time()
        orig_prob = self.predict_fn(code)
        queries   = 1

        if orig_prob < self.threshold:
            return AttackResult(
                original_code=code, adversarial_code=code,
                success=False, original_prob=orig_prob,
                adversarial_prob=orig_prob, prob_shift=0.0,
                queries_used=1, transforms_applied=[],
                cwe=cwe, time_sec=time.time()-t0, source=source)

        safe_set = (set(range(len(code.splitlines()))) if self.skip_taint
                    else get_taint_set(code, hops=2)[1])

        if self.use_random:
            bc, bp, queries, seq = self._random_search(code, safe_set, queries)
        else:
            bc, bp, queries, seq = self._genetic_search(code, safe_set, queries)

        success = bp < self.threshold
        return AttackResult(
            original_code=code, adversarial_code=bc if success else code,
            success=success, original_prob=orig_prob, adversarial_prob=bp,
            prob_shift=orig_prob - bp, queries_used=queries,
            transforms_applied=seq, cwe=cwe,
            time_sec=time.time()-t0, source=source)

    def _genetic_search(self, code, safe_set, queries):
        pop   = self._init_pop(code, safe_set)
        probs = [self.predict_fn(i.code) for i in pop]
        queries += len(pop)
        for ind, p in zip(pop, probs):
            ind.vuln_prob = p; ind.fitness = 1.0 - p
        best = max(pop, key=lambda x: x.fitness).clone()
        for _ in range(self.max_gens):
            if queries >= self.budget or best.vuln_prob < self.threshold:
                break
            pop   = self._evolve(pop, code, safe_set)
            probs = [self.predict_fn(i.code) for i in pop]
            queries += len(pop)
            for ind, p in zip(pop, probs):
                ind.vuln_prob = p; ind.fitness = 1.0 - p
            g = max(pop, key=lambda x: x.fitness)
            if g.fitness > best.fitness:
                best = g.clone()
        return best.code, best.vuln_prob, queries, best.sequence

    def _random_search(self, code, safe_set, queries):
        best_code, best_prob, best_seq = code, self.predict_fn(code), []
        while queries < self.budget:
            seq = random.choices(self.allowed, k=random.randint(1, self.max_seq_len))
            c   = apply_sequence(code, seq, safe_set)
            p   = self.predict_fn(c)
            queries += 1
            if p < best_prob:
                best_prob, best_code, best_seq = p, c, seq
            if best_prob < self.threshold:
                break
        return best_code, best_prob, queries, best_seq

    def _init_pop(self, code, safe_set):
        pop = [Individual(sequence=[t],
                          code=apply_sequence(code, [t], safe_set))
               for t in self.allowed]
        while len(pop) < self.pop_size:
            seq = random.choices(self.allowed, k=random.randint(1, self.max_seq_len))
            pop.append(Individual(sequence=seq,
                                  code=apply_sequence(code, seq, safe_set)))
        return pop[:self.pop_size]

    def _evolve(self, pop, orig, safe_set):
        n_e      = max(1, int(self.pop_size * self.elite_frac))
        elites   = sorted(pop, key=lambda x: x.fitness, reverse=True)[:n_e]
        offspring = [e.clone() for e in elites]
        while len(offspring) < self.pop_size:
            p1    = self._tour(pop)
            child = (self._cross(p1, self._tour(pop)) if random.random() < 0.5
                     else p1.clone())
            if random.random() < self.mutation_rate:
                child.sequence = self._mutate(child.sequence)
            child.code = apply_sequence(orig, child.sequence, safe_set)
            offspring.append(child)
        return offspring

    def _tour(self, pop):
        return max(random.choices(pop, k=self.tournament_k),
                   key=lambda x: x.fitness).clone()

    def _cross(self, p1, p2):
        s1, s2 = p1.sequence, p2.sequence
        if not s1: return p2.clone()
        if not s2: return p1.clone()
        seq = (s1[:random.randint(0,len(s1))] +
               s2[random.randint(0,len(s2)):])[:10] or s1
        return Individual(sequence=seq)

    def _mutate(self, seq):
        if not seq: return [random.choice(self.allowed)]
        seq = seq.copy()
        op  = random.choice(["substitute","insert","delete"])
        if   op == "substitute":
            seq[random.randrange(len(seq))] = random.choice(self.allowed)
        elif op == "insert" and len(seq) < self.max_seq_len:
            seq.insert(random.randint(0, len(seq)), random.choice(self.allowed))
        elif op == "delete" and len(seq) > 1:
            seq.pop(random.randrange(len(seq)))
        return seq


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset  —  reconstruct combined test split from raw files (no saved file needed)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Sample:
    code:   str
    cwe:    str = ""
    source: str = ""


def build_combined_test(cache: Path, seed: int = 42) -> List[Sample]:
    """
    Calls load_dataset() from your existing dataset_loader.py.
    Uses the same four raw files and seed=42, so the split is identical
    to the one used in the main experiment.
    """
    from data.dataset_loader import load_dataset
    logger.info("Building combined test split from %s ...", cache)
    _, _, test = load_dataset(
        cache_dir    = str(cache),
        max_bigvul   = 10_000,
        max_devign   = 5_000,
        max_sard     = 3_000,
        max_primevul = 5_000,
        seed         = seed,
    )
    samples = [Sample(code=s.code,
                      cwe=getattr(s, "cwe", ""),
                      source=getattr(s, "source", ""))
               for s in test if s.label == 1]
    logger.info("Vulnerable samples in combined test: %d", len(samples))
    return samples


def filter_detected(samples: List[Sample], predict_fn,
                    threshold: float = 0.5, n: int = 500) -> List[Sample]:
    from tqdm import tqdm
    detected = []
    for s in tqdm(samples, desc="  Filtering to detected", leave=False):
        if predict_fn(s.code) >= threshold:
            detected.append(s)
        if len(detected) >= n:
            break
    logger.info("Attack pool: %d detected / %d total vulnerable",
                len(detected), len(samples))
    return detected


# ══════════════════════════════════════════════════════════════════════════════
#  Detector loader  —  tries your project's class first, then generic fallback
# ══════════════════════════════════════════════════════════════════════════════

def load_detector(ckpt: Path, device: str = "cuda") -> Callable[[str], float]:
    """
    Uses the same pattern as run_experiment.py:
        det = VulnDetector("linevul")
        det.load(path)
        det.predict(code) -> float
    """
    logger.info("Loading detector from %s ...", ckpt)

    try:
        from models.vulnerability_detector import VulnDetector
        det = VulnDetector("linevul")
        det.load(str(ckpt))
        logger.info("Loaded via VulnDetector('linevul').load()")
        return det.predict
    except Exception as e:
        logger.error("Failed to load detector: %s", e)
        logger.error("")
        logger.error("Expected:  from models.vulnerability_detector import VulnDetector")
        logger.error("           det = VulnDetector('linevul')")
        logger.error("           det.load('%s')", ckpt)
        logger.error("")
        logger.error("Check that models/vulnerability_detector.py exists and")
        logger.error("VulnDetector has .load() and .predict() methods.")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  Summary helper
# ══════════════════════════════════════════════════════════════════════════════

def summarise(results: List[AttackResult]) -> Dict:
    n      = len(results)
    n_succ = sum(1 for r in results if r.success)
    return {
        "n":      n,
        "n_succ": n_succ,
        "asr":    n_succ / n * 100 if n else 0.0,
        "avg_q":  sum(r.queries_used for r in results) / n if n else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    predict_fn  = load_detector(DET_CKPT, device=args.device)
    all_samples = build_combined_test(CACHE, seed=args.seed)
    pool        = filter_detected(all_samples, predict_fn, n=args.n_samples)

    if not pool:
        logger.error("No detected samples. Check DET_CKPT and CACHE paths.")
        sys.exit(1)

    rows:     List[Dict]      = []
    base_asr: Optional[float] = None

    for cfg_name, cfg in CONFIGS.items():
        safe_name = (cfg_name.replace(" ","_").replace("/","_")
                              .replace("(","").replace(")",""))
        ckpt_file = OUT_DIR / f"ablation_ckpt_{safe_name}.pkl"

        # ── resume from checkpoint if available ──────────────────────────────
        if ckpt_file.exists():
            with open(ckpt_file, "rb") as f:
                results: List[AttackResult] = pickle.load(f)
            s = summarise(results)
            logger.info("[SKIP] %-40s  ASR=%5.1f%%  AvgQ=%5.0f",
                        cfg_name, s["asr"], s["avg_q"])

        else:
            logger.info("Running %-40s  allowed=%d  random=%s  skip_taint=%s",
                        cfg_name, len(cfg["allowed"]),
                        cfg["use_random"], cfg["skip_taint"])

            attacker = AblationAttacker(
                predict_fn         = predict_fn,
                allowed_transforms = cfg["allowed"],
                use_random         = cfg["use_random"],
                skip_taint         = cfg["skip_taint"],
                query_budget       = args.query_budget,
                seed               = args.seed,
            )

            from tqdm import tqdm
            results = []
            for sample in tqdm(pool, desc=f"  {cfg_name[:36]:<36}"):
                results.append(
                    attacker.attack(sample.code,
                                    cwe=sample.cwe,
                                    source=sample.source))

            with open(ckpt_file, "wb") as f:
                pickle.dump(results, f)

            s = summarise(results)
            logger.info("[DONE] %-40s  ASR=%5.1f%%  AvgQ=%5.0f  (%d/%d)",
                        cfg_name, s["asr"], s["avg_q"], s["n_succ"], s["n"])

        s = summarise(results)
        if base_asr is None:
            base_asr = s["asr"]

        rows.append({
            "Configuration": cfg_name,
            "ASR (%)":       round(s["asr"],   1),
            "AvgQ":          int(round(s["avg_q"])),
            "Delta (pp)":    round(s["asr"] - base_asr, 1),
            "n_succ":        s["n_succ"],
            "n_total":       s["n"],
        })

    # ── print ─────────────────────────────────────────────────────────────────
    sep = "-" * 72
    hdr = f"{'Configuration':<40}  {'ASR%':>6}  {'AvgQ':>5}  {'Delta':>7}  Succ/N"
    lines = ["", sep, hdr, sep]
    for r in rows:
        lines.append(
            f"{r['Configuration']:<40}  "
            f"{r['ASR (%)']:>6.1f}  "
            f"{r['AvgQ']:>5d}  "
            f"{r['Delta (pp)']:>+7.1f}  "
            f"{r['n_succ']}/{r['n_total']}")
    lines.append(sep)
    table_str = "\n".join(lines)
    print(table_str)

    # ── save CSV ──────────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "table_ablation.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["Configuration","ASR (%)","AvgQ",
                           "Delta (pp)","n_succ","n_total"])
        w.writeheader()
        w.writerows(rows)

    (OUT_DIR / "table_ablation.txt").write_text(table_str + "\n")

    # ── LaTeX rows ────────────────────────────────────────────────────────────
    latex = ["% paste into tab:ablation in the paper"]
    for r in rows:
        name, asr, delta = r["Configuration"], r["ASR (%)"], r["Delta (pp)"]
        if name == "Full ShadowPatch":
            latex.append(
                f"Full \\sys{{}}                         "
                f"& \\textbf{{{asr:.1f}}} & --- \\\\")
        else:
            sign = "$-$" if delta < 0 else "$+$"
            latex.append(f"{name:<44} & {asr:.1f} & {sign}{abs(delta):.1f} \\\\")

    latex_str = "\n".join(latex)
    (OUT_DIR / "latex_rows.txt").write_text(latex_str + "\n")
    print("\n" + latex_str + "\n")

    logger.info("All results saved to %s", OUT_DIR)
    logger.info("  CSV   -> %s", csv_path)
    logger.info("  LaTeX -> %s", OUT_DIR / "latex_rows.txt")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="ShadowPatch ablation study — no retraining needed")
    p.add_argument("--n_samples",    type=int,   default=500,
                   help="Detected samples to attack per config (default 500)")
    p.add_argument("--query_budget", type=int,   default=500,
                   help="Oracle query budget per sample (default 500)")
    p.add_argument("--device",       default="cuda",
                   help="PyTorch device (default cuda)")
    p.add_argument("--seed",         type=int,   default=42,
                   help="Random seed — must match main experiment (default 42)")
    args = p.parse_args()

    logger.info("Mother      : %s", MOTHER)
    logger.info("Detector    : %s", DET_CKPT)
    logger.info("Data cache  : %s", CACHE)
    logger.info("Output      : %s", OUT_DIR)
    logger.info("n_samples   : %d", args.n_samples)
    logger.info("query_budget: %d", args.query_budget)
    logger.info("")

    if not DET_CKPT.exists():
        logger.error("Checkpoint not found: %s", DET_CKPT)
        logger.error("")
        logger.error("Fix: check what files are in results/checkpoints/ and")
        logger.error("     update DET_CKPT at the top of this script.")
        sys.exit(1)

    if not CACHE.exists():
        logger.error("Data cache not found: %s", CACHE)
        sys.exit(1)

    main(args)
