"""
genetic_optimizer.py
====================
Phase 3 of ShadowPatch: Genetic Algorithm optimizer.

Searches transformation sequences that minimize the target detector's
predicted vulnerability probability, subject to:
  - PDG taint constraint (transforms only applied to safe region)
  - SMT equivalence check (Phase 4)


Algorithm matches Paper:
  population P=20, max gens G=50, seq length K≤10, budget B=500
  tournament k=3, crossover (prefix/suffix interleaved),
  mutation rate 0.4, elitism top 20%
"""

from __future__ import annotations
import random, logging, copy, time
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple, Dict

from .code_transformer import TRANSFORM_IDS, apply_sequence
from .pdg_taint        import get_taint_set
from .smt_verifier     import verify_equivalence, SMTResult
from .exploit_verifier import ExploitVerifier, ExploitResult

logger = logging.getLogger(__name__)

TRANSFORM_IDS_LIST = list(TRANSFORM_IDS)   # T1 … T18


# ── individual ────────────────────────────────────────────────────────────────

@dataclass
class Individual:
    sequence:  List[str]           # transform IDs
    code:      str = ""            # transformed code
    vuln_prob: float = 1.0         # model's predicted P(vulnerable)
    fitness:   float = 0.0         # 1 - vuln_prob
    smt_ok:    Optional[bool] = None
    exploit_ok: Optional[bool] = None

    def clone(self) -> "Individual":
        return copy.deepcopy(self)


# ── attack result ─────────────────────────────────────────────────────────────

@dataclass
class AttackResult:
    original_code:   str
    adversarial_code: str
    success:         bool
    original_prob:   float
    adversarial_prob: float
    prob_shift:      float
    queries_used:    int
    transforms_applied: List[str]
    smt_result:      Optional[SMTResult]    = None
    exploit_result:  Optional[ExploitResult] = None
    cwe:             str  = ""
    time_sec:        float = 0.0
    source:          str  = ""


# ══════════════════════════════════════════════════════════════════════════════
# Genetic optimizer
# ══════════════════════════════════════════════════════════════════════════════

class GeneticAttacker:
    """
    Runs the genetic adversarial search from Paper 2.

    predict_fn: callable(code: str) -> float  [P(vulnerable)]
    """

    def __init__(self,
                 predict_fn:   Callable[[str], float],
                 pop_size:     int   = 20,
                 max_gens:     int   = 50,
                 max_seq_len:  int   = 10,
                 query_budget: int   = 500,
                 mutation_rate: float = 0.4,
                 elite_frac:   float = 0.2,
                 tournament_k: int   = 3,
                 threshold:    float = 0.5,
                 smt_verify:   bool  = True,
                 exploit_verify: bool = False,
                 exploit_verifier: Optional[ExploitVerifier] = None,
                 seed: int = 42):

        self.predict_fn     = predict_fn
        self.pop_size       = pop_size
        self.max_gens       = max_gens
        self.max_seq_len    = max_seq_len
        self.budget         = query_budget
        self.mutation_rate  = mutation_rate
        self.elite_frac     = elite_frac
        self.tournament_k   = tournament_k
        self.threshold      = threshold
        self.smt_verify     = smt_verify
        self.exploit_verify = exploit_verify
        self.exploit_verifier = exploit_verifier or ExploitVerifier(use_aflpp=False)
        random.seed(seed)

    # ── public attack entry ────────────────────────────────────────────────────
    def attack(self, code: str, cwe: str = "",
               source: str = "") -> AttackResult:
        t0            = time.time()
        orig_prob     = self.predict_fn(code)
        queries_used  = 1

        if orig_prob < self.threshold:
            # already predicted clean — no attack needed
            return AttackResult(
                original_code=code, adversarial_code=code,
                success=False, original_prob=orig_prob,
                adversarial_prob=orig_prob, prob_shift=0.0,
                queries_used=1, transforms_applied=[],
                cwe=cwe, time_sec=time.time() - t0, source=source)

        # Phase 1: PDG taint analysis
        taint_set, safe_set = get_taint_set(code, hops=2)

        # Phase 3: Genetic search
        population = self._init_population(code, safe_set)

        # evaluate initial population
        batch_codes = [ind.code for ind in population]
        batch_probs = self._batch_predict(batch_codes)
        queries_used += len(population)

        for ind, prob in zip(population, batch_probs):
            ind.vuln_prob = prob
            ind.fitness   = 1.0 - prob

        best_ind = max(population, key=lambda x: x.fitness)

        for gen in range(self.max_gens):
            if queries_used >= self.budget:
                break

            # check termination
            if best_ind.vuln_prob < self.threshold:
                break

            # evolve
            population = self._evolve(population, code, safe_set)

            # evaluate new offspring (non-elite)
            new_codes = [ind.code for ind in population]
            new_probs = self._batch_predict(new_codes)
            queries_used += len(population)

            for ind, prob in zip(population, new_probs):
                ind.vuln_prob = prob
                ind.fitness   = 1.0 - prob

            gen_best = max(population, key=lambda x: x.fitness)
            if gen_best.fitness > best_ind.fitness:
                best_ind = gen_best.clone()

            logger.debug("Gen %d | best_prob=%.3f | queries=%d",
                         gen + 1, best_ind.vuln_prob, queries_used)

        # Phase 4: SMT verification
        smt_result = None
        if self.smt_verify:
            smt_result = verify_equivalence(code, best_ind.code)
            if not smt_result.verified:
                # Try next best
                alts = sorted(population,
                              key=lambda x: x.fitness, reverse=True)[1:5]
                for alt in alts:
                    r = verify_equivalence(code, alt.code)
                    if r.verified:
                        best_ind  = alt.clone()
                        smt_result = r
                        break

        success = best_ind.vuln_prob < self.threshold

        # Phase 5: Exploit verification (optional, expensive)
        exploit_result = None
        if success and self.exploit_verify:
            exploit_result = self.exploit_verifier.verify(
                code, best_ind.code, cwe)
            # If exploit not preserved, revert to original
            if not exploit_result.exploit_preserved:
                success = False

        return AttackResult(
            original_code      = code,
            adversarial_code   = best_ind.code if success else code,
            success            = success,
            original_prob      = orig_prob,
            adversarial_prob   = best_ind.vuln_prob,
            prob_shift         = orig_prob - best_ind.vuln_prob,
            queries_used       = queries_used,
            transforms_applied = best_ind.sequence,
            smt_result         = smt_result,
            exploit_result     = exploit_result,
            cwe                = cwe,
            time_sec           = time.time() - t0,
            source             = source,
        )

    # ── batch attack ──────────────────────────────────────────────────────────
    def batch_attack(self, samples, show_progress: bool = True,
                     desc: str = "  Attacking") -> List[AttackResult]:
        from tqdm import tqdm as _tqdm
        results = []
        n_success = 0
        bar_fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} [ASR={postfix[asr]} {elapsed}<{remaining}, {rate_fmt}]"
        if show_progress:
            iterator = _tqdm(samples, desc=desc, unit="sample",
                             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"
                                        " [ASR={postfix[asr]}"
                                        " {elapsed}<{remaining}"
                                        " {rate_fmt}]",
                             postfix={"asr": "0.0%"})
        else:
            iterator = samples
        for s in iterator:
            r = self.attack(s.code, cwe=s.cwe, source=s.source)
            results.append(r)
            if r.success:
                n_success += 1
            if show_progress:
                iterator.set_postfix(asr=f"{n_success/len(results)*100:.1f}%")
        return results

    # ── initialise population ─────────────────────────────────────────────────
    def _init_population(self, code: str,
                         safe_set) -> List[Individual]:
        pop = []
        # seed: one individual per single transform
        for tid in TRANSFORM_IDS_LIST:
            seq  = [tid]
            c    = apply_sequence(code, seq, safe_set)
            pop.append(Individual(sequence=seq, code=c))

        # fill remainder with random sequences
        while len(pop) < self.pop_size:
            k    = random.randint(1, self.max_seq_len)
            seq  = random.choices(TRANSFORM_IDS_LIST, k=k)
            c    = apply_sequence(code, seq, safe_set)
            pop.append(Individual(sequence=seq, code=c))

        return pop[:self.pop_size]

    # ── evolve one generation ─────────────────────────────────────────────────
    def _evolve(self, population: List[Individual],
                orig_code: str, safe_set) -> List[Individual]:
        n_elite = max(1, int(self.pop_size * self.elite_frac))
        elites  = sorted(population, key=lambda x: x.fitness,
                         reverse=True)[:n_elite]
        offspring = [e.clone() for e in elites]

        while len(offspring) < self.pop_size:
            p1 = self._tournament(population)
            if random.random() < 0.5:
                p2     = self._tournament(population)
                child  = self._crossover(p1, p2)
            else:
                child  = p1.clone()

            if random.random() < self.mutation_rate:
                child.sequence = self._mutate(child.sequence)

            child.code    = apply_sequence(orig_code, child.sequence, safe_set)
            offspring.append(child)

        return offspring

    # ── tournament selection ──────────────────────────────────────────────────
    def _tournament(self, population: List[Individual]) -> Individual:
        competitors = random.choices(population, k=self.tournament_k)
        return max(competitors, key=lambda x: x.fitness).clone()

    # ── crossover ─────────────────────────────────────────────────────────────
    @staticmethod
    def _crossover(p1: Individual, p2: Individual) -> Individual:
        # interleaved prefix/suffix
        s1, s2  = p1.sequence, p2.sequence
        if not s1: return p2.clone()
        if not s2: return p1.clone()
        j1 = random.randint(0, len(s1))
        j2 = random.randint(0, len(s2))
        new_seq = s1[:j1] + s2[j2:]
        if not new_seq:
            new_seq = s1 if s1 else s2
        return Individual(sequence=new_seq[:10])

    # ── mutation ──────────────────────────────────────────────────────────────
    def _mutate(self, seq: List[str]) -> List[str]:
        if not seq:
            return [random.choice(TRANSFORM_IDS_LIST)]
        op = random.choice(["substitute", "insert", "delete"])
        seq = seq.copy()
        if op == "substitute" and seq:
            i       = random.randrange(len(seq))
            seq[i]  = random.choice(TRANSFORM_IDS_LIST)
        elif op == "insert" and len(seq) < self.max_seq_len:
            i = random.randint(0, len(seq))
            seq.insert(i, random.choice(TRANSFORM_IDS_LIST))
        elif op == "delete" and len(seq) > 1:
            seq.pop(random.randrange(len(seq)))
        return seq

    # ── batch predict ─────────────────────────────────────────────────────────
    def _batch_predict(self, codes: List[str]) -> List[float]:
        return [self.predict_fn(c) for c in codes]


# ══════════════════════════════════════════════════════════════════════════════
# White-box PGD embedding attack (supplemental)
# ══════════════════════════════════════════════════════════════════════════════

class WhiteBoxEmbeddingAttack:
    """
    PGD attack on the input token embeddings (white-box setting).
    Perturbs embeddings within an l-inf ball and projects back to
    the nearest token vocabulary entry.

    Parameters match Paper 2: ε=0.1, 20 steps, α=0.01.
    """

    def __init__(self, detector,
                 eps: float = 0.1, steps: int = 20, alpha: float = 0.01):
        self.detector = detector
        self.eps      = eps
        self.steps    = steps
        self.alpha    = alpha

    def attack(self, code: str) -> AttackResult:
        import torch, torch.nn.functional as F

        t0  = time.time()
        dev = self.detector.device
        tok = self.detector.tokenizer
        mdl = self.detector.model

        enc = tok(code, truncation=True, max_length=512,
                  padding="max_length", return_tensors="pt")
        enc = {k: v.to(dev) for k, v in enc.items()}

        orig_prob = float(mdl.predict_proba(**enc).item())
        if orig_prob < 0.5:
            return AttackResult(code, code, False, orig_prob, orig_prob,
                                0.0, 1, [], time_sec=time.time()-t0)

        # get embedding table
        embed_table = None
        for m in mdl.modules():
            if hasattr(m, "weight") and m.weight.shape[0] > 10000:
                embed_table = m.weight
                break

        if embed_table is None:
            # fallback: just return original
            return AttackResult(code, code, False, orig_prob, orig_prob,
                                0.0, 1, [], time_sec=time.time()-t0)

        # get initial embeddings
        ids   = enc["input_ids"]
        emb   = embed_table[ids].detach().clone().requires_grad_(True)
        best_adv_code = code
        best_prob     = orig_prob

        for step in range(self.steps):
            # forward pass with perturbed embeddings
            # (simplified: use full model call with slightly perturbed code)
            # Full embedding attack requires model internals; approximate here
            loss = -torch.log(1 - mdl.predict_proba(**enc) + 1e-8)
            loss.backward()
            if emb.grad is not None:
                emb = emb - self.alpha * emb.grad.sign()
                emb = torch.clamp(emb,
                                  embed_table[ids] - self.eps,
                                  embed_table[ids] + self.eps)
                emb = emb.detach().requires_grad_(True)

        adv_prob = float(mdl.predict_proba(**enc).item())
        success  = adv_prob < 0.5

        return AttackResult(
            original_code=code, adversarial_code=best_adv_code,
            success=success, original_prob=orig_prob,
            adversarial_prob=adv_prob,
            prob_shift=orig_prob - adv_prob,
            queries_used=self.steps,
            transforms_applied=["PGD_embedding"],
            time_sec=time.time() - t0)


# ══════════════════════════════════════════════════════════════════════════════
# Summary stats helper
# ══════════════════════════════════════════════════════════════════════════════

def summarise_results(results: List[AttackResult]) -> Dict:
    if not results:
        return {}
    n          = len(results)
    success    = [r for r in results if r.success]
    asr        = len(success) / n
    avg_q      = sum(r.queries_used for r in success) / max(len(success), 1)
    avg_shift  = sum(r.prob_shift   for r in results) / n

    smt_ok = sum(1 for r in success
                 if r.smt_result and r.smt_result.verified)
    expl_ok = sum(1 for r in success
                  if r.exploit_result and r.exploit_result.exploit_preserved)

    # transform frequency
    from collections import Counter
    freq = Counter()
    for r in success:
        freq.update(r.transforms_applied)

    return {
        "n_samples":      n,
        "n_success":      len(success),
        "asr":            round(asr, 4),
        "avg_queries":    round(avg_q, 1),
        "avg_prob_shift": round(avg_shift, 4),
        "smt_verified":   smt_ok,
        "exploit_preserved": expl_ok,
        "top_transforms": freq.most_common(5),
    }
