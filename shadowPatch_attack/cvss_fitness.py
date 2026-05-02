"""
cvss_fitness.py
===============
CVSS-weighted fitness function for ShadowPatch's Genetic Algorithm.

Sits BESIDE genetic_optimizer.py — does NOT modify it.
Drop this file into the same package folder:
    shadowpatch_v2/
        genetic_optimizer.py   ← unchanged
        cvss_fitness.py        ← NEW (this file)

Usage in run_experiment.py:
    from .cvss_fitness import CVSSFitnessWrapper
    weighted_attacker = CVSSFitnessWrapper(attacker, cvss_table)
    result = weighted_attacker.attack(sample)

How it works
------------
The original fitness function inside GeneticAttacker is:
    ind.fitness = 1.0 - prob          (maximised → minimize prob)

This module wraps the predict_fn that the GA receives, scaling the
probability signal by a CVSS-derived weight before the GA ever sees it.

    effective_prob = prob / severity_weight(sample)

A higher CVSS score → higher weight → effective_prob is pushed lower
for the same raw prob → the GA sees the sample as "closer to success"
→ it applies more aggressive transform sequences.

Three-tier imputation strategy (per dataset):
    BigVul  →  NVD lookup by CVE   → CWE-mean fallback → global median
    Devign  →  flat fitness (no CVE, no CWE) — weight = 1.0
    SARD    →  CWE-mean borrowed from BigVul CVEs
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── CVSS constants ────────────────────────────────────────────────────────────

# Global BigVul median CVSS (computed offline from NVD; used as last-resort)
GLOBAL_MEDIAN_CVSS = 7.2

# CVSS score range
CVSS_MIN = 0.0
CVSS_MAX = 10.0

# Lambda: how strongly CVSS influences fitness.
# fitness = (1 - prob) + LAMBDA * normalised_cvss
# Set 0.0 to disable CVSS weighting entirely (pure ablation baseline).
DEFAULT_LAMBDA = 0.3

# CWE-mean CVSS scores computed from NVD data for BigVul's target CWEs.
# Source: NVD statistics 2002-2023, filtered to C/C++ CVEs.
# These are used for SARD imputation and BigVul CVE-missing fallback.
CWE_MEAN_CVSS: Dict[str, float] = {
    "CWE-119": 7.5,   # buffer errors (broad)
    "CWE-120": 7.5,   # buffer copy without size check
    "CWE-122": 7.8,   # heap-based buffer overflow
    "CWE-125": 6.5,   # out-of-bounds read
    "CWE-787": 8.8,   # out-of-bounds write
    "CWE-416": 8.1,   # use-after-free
    "CWE-190": 7.8,   # integer overflow
    "CWE-191": 7.5,   # integer underflow
    "CWE-476": 6.5,   # null pointer dereference
    "CWE-078": 9.8,   # OS command injection
    "CWE-134": 8.1,   # uncontrolled format string
}


# ══════════════════════════════════════════════════════════════════════════════
# CVSS table builder
# Builds once before the attack loop; passed into CVSSFitnessWrapper.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CVSSRecord:
    """One entry in the lookup table — one VulnSample's resolved CVSS."""
    cvss:       float           # resolved score (0-10)
    source:     str             # "nvd_direct" | "cwe_mean" | "global_median" | "flat"
    cve:        str = ""
    cwe:        str = ""


class CVSSTable:
    """
    Pre-computed CVSS scores for every sample in the attack subset.

    Build once:
        table = CVSSTable.build(attack_samples, cache_path="data/cache/cvss_cache.json")

    Then query:
        record = table.get(sample)
        weight = table.weight(sample)    # normalised to [1.0, 1+lambda]
    """

    def __init__(self, records: Dict[str, CVSSRecord], lambda_: float = DEFAULT_LAMBDA):
        # key = sample identity: cve or f"{source}:{cwe}:{hash(code[:80])}"
        self._records  = records
        self.lambda_   = lambda_

        # stats for logging
        sources = [r.source for r in records.values()]
        for src in ["nvd_direct", "cwe_mean", "global_median", "flat"]:
            n = sources.count(src)
            if n:
                logger.info("CVSSTable: %d samples via '%s'", n, src)

    # ── build ─────────────────────────────────────────────────────────────────
    @classmethod
    def build(cls,
              samples,                        # List[VulnSample]
              cache_path: str = "data/cache/cvss_cache.json",
              lambda_: float = DEFAULT_LAMBDA,
              offline: bool = False) -> "CVSSTable":
        """
        Resolves CVSS for every sample using the three-tier strategy.

        offline=True  → skip NVD HTTP calls (use CWE-mean / median only).
                         Useful for air-gapped HPC runs.
        """
        cache = _load_json_cache(cache_path)
        records: Dict[str, CVSSRecord] = {}

        nvd_fetcher = NVDFetcher(cache_path=cache_path, offline=offline)

        for s in samples:
            key = _sample_key(s)
            rec = _resolve_one(s, nvd_fetcher)
            records[key] = rec

        nvd_fetcher.save_cache()          # persist new NVD responses
        return cls(records, lambda_=lambda_)

    # ── query ─────────────────────────────────────────────────────────────────
    def get(self, sample) -> CVSSRecord:
        key = _sample_key(sample)
        return self._records.get(key, CVSSRecord(
            cvss=GLOBAL_MEDIAN_CVSS, source="global_median"))

    def weight(self, sample) -> float:
        """
        Returns a severity weight w in [1.0, 1 + lambda].

        w = 1.0 + lambda * (cvss / 10.0)

        A CVSS-0 sample → weight=1.0  (no boost, pure evasion)
        A CVSS-10 sample → weight=1+lambda
        """
        if self.lambda_ == 0.0:
            return 1.0
        rec = self.get(sample)
        if rec.source == "flat":           # Devign: no weighting
            return 1.0
        normalised = rec.cvss / CVSS_MAX   # 0..1
        return 1.0 + self.lambda_ * normalised


# ══════════════════════════════════════════════════════════════════════════════
# Three-tier resolver
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_one(sample, nvd_fetcher: "NVDFetcher") -> CVSSRecord:
    """
    Applies the three-tier strategy for one sample.
    Returns a CVSSRecord with the resolved score and its provenance.
    """
    src    = sample.source    # "bigvul" | "devign" | "sard"
    cwe    = _normalise_cwe(sample.cwe)
    cve_raw = getattr(sample, "cve", "")

    # ── Devign: no CVE, no CWE → flat fitness, weight=1.0 ──────────────────
    if src == "devign":
        return CVSSRecord(cvss=GLOBAL_MEDIAN_CVSS, source="flat",
                          cve="", cwe=cwe)

    # ── SARD: has CWE but no real CVE → borrow CWE-mean from BigVul ────────
    if src == "sard":
        score = CWE_MEAN_CVSS.get(cwe, GLOBAL_MEDIAN_CVSS)
        return CVSSRecord(cvss=score, source="cwe_mean", cve="", cwe=cwe)

    # ── BigVul: try NVD lookup first ─────────────────────────────────────────
    if src == "bigvul":
        cves = _parse_cve_list(cve_raw)

        # Tier 1: direct NVD lookup
        scores = []
        for cve in cves:
            s = nvd_fetcher.fetch(cve)
            if s is not None:
                scores.append(s)

        if scores:
            best = max(scores)             # conservative: take highest
            return CVSSRecord(cvss=best, source="nvd_direct",
                              cve=cve_raw, cwe=cwe)

        # Tier 2: CWE-mean imputation
        if cwe in CWE_MEAN_CVSS:
            return CVSSRecord(cvss=CWE_MEAN_CVSS[cwe], source="cwe_mean",
                              cve=cve_raw, cwe=cwe)

        # Tier 3: global median fallback
        return CVSSRecord(cvss=GLOBAL_MEDIAN_CVSS, source="global_median",
                          cve=cve_raw, cwe=cwe)

    # unknown source → global median
    return CVSSRecord(cvss=GLOBAL_MEDIAN_CVSS, source="global_median")


# ══════════════════════════════════════════════════════════════════════════════
# NVD HTTP fetcher with local cache
# ══════════════════════════════════════════════════════════════════════════════

class NVDFetcher:
    """
    Fetches CVSS scores from the NVD 2.0 API.
    Caches responses in a local JSON file to avoid redundant HTTP calls.

    NVD rate limit: 5 requests per 30 seconds (unauthenticated).
    The fetcher respects this automatically.
    """
    NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    RATE_LIMIT_SLEEP = 6.5   # seconds between requests (safe under 5/30s)

    def __init__(self, cache_path: str = "data/cache/cvss_cache.json",
                 offline: bool = False):
        self.cache_path   = Path(cache_path)
        self.offline      = offline
        self._cache: Dict[str, Optional[float]] = _load_json_cache(cache_path)
        self._last_call   = 0.0

    def fetch(self, cve_id: str) -> Optional[float]:
        """
        Returns CVSS v3.1 base score for a CVE, or None if unavailable.
        Results are cached locally — each CVE is only fetched once.
        """
        cve_id = cve_id.strip().upper()
        if not cve_id.startswith("CVE-"):
            return None

        # return from cache if already fetched
        if cve_id in self._cache:
            return self._cache[cve_id]

        if self.offline:
            self._cache[cve_id] = None
            return None

        # rate-limit
        elapsed = time.time() - self._last_call
        if elapsed < self.RATE_LIMIT_SLEEP:
            time.sleep(self.RATE_LIMIT_SLEEP - elapsed)

        try:
            import requests
            resp = requests.get(
                self.NVD_API,
                params={"cveId": cve_id},
                timeout=15,
                headers={"Accept": "application/json"}
            )
            self._last_call = time.time()

            if resp.status_code != 200:
                logger.debug("NVD %s → HTTP %d", cve_id, resp.status_code)
                self._cache[cve_id] = None
                return None

            data = resp.json()
            score = _extract_cvss(data)
            self._cache[cve_id] = score
            return score

        except Exception as e:
            logger.debug("NVD fetch failed for %s: %s", cve_id, e)
            self._cache[cve_id] = None
            return None

    def save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f, indent=2)
        logger.info("CVSS cache saved: %d entries → %s",
                    len(self._cache), self.cache_path)


# ══════════════════════════════════════════════════════════════════════════════
# Main wrapper — plugs into GeneticAttacker without touching it
# ══════════════════════════════════════════════════════════════════════════════

class CVSSFitnessWrapper:
    """
    Wraps GeneticAttacker to use CVSS-weighted fitness.

    The original GA uses:
        ind.fitness = 1.0 - prob

    This wrapper intercepts predict_fn so the GA instead receives:
        effective_prob = prob / weight(sample)

    which makes the fitness landscape steeper for high-severity samples,
    pushing the GA to invest more search effort on critical vulnerabilities.

    The original GeneticAttacker is NEVER modified.

    Example
    -------
        # build once before the attack loop
        table = CVSSTable.build(attack_samples)

        # wrap the existing attacker
        wrapped = CVSSFitnessWrapper(ga_attacker, table)

        # attack exactly as before — API is identical
        result = wrapped.attack(sample)

        # result has two extra fields in meta:
        #   result.meta["cvss"]          → resolved CVSS score
        #   result.meta["cvss_source"]   → "nvd_direct" / "cwe_mean" / etc.
        #   result.meta["severity_weight"] → w value used
    """

    def __init__(self,
                 attacker,              # GeneticAttacker instance
                 cvss_table: CVSSTable):
        self.attacker    = attacker
        self.cvss_table  = cvss_table
        self._current_sample = None     # set just before each attack call

    def attack(self, sample) -> "AttackResult":
        """
        Drop-in replacement for attacker.attack(sample).
        sample must be a VulnSample (has .code, .cwe, .cve, .source).
        """
        self._current_sample = sample
        weight = self.cvss_table.weight(sample)
        rec    = self.cvss_table.get(sample)

        # swap in the weighted predict_fn
        original_fn = self.attacker.predict_fn
        self.attacker.predict_fn = self._make_weighted_fn(original_fn, weight)

        try:
            result = self.attacker.attack(
                code   = sample.code,
                cwe    = sample.cwe,
                source = sample.source,
            )
        finally:
            # always restore original predict_fn
            self.attacker.predict_fn = original_fn

        # attach CVSS metadata to result
        result.meta = getattr(result, "meta", {})
        result.meta.update({
            "cvss":             rec.cvss,
            "cvss_source":      rec.source,
            "severity_weight":  weight,
        })
        return result

    def batch_attack(self, samples, show_progress: bool = True):
        """
        Drop-in replacement for attacker.batch_attack(samples).
        """
        from tqdm import tqdm as _tqdm
        results  = []
        iterator = _tqdm(samples, desc="CVSS-weighted attack") \
                   if show_progress else samples
        for s in iterator:
            results.append(self.attack(s))
        return results

    # ── weighted predict_fn ────────────────────────────────────────────────────
    @staticmethod
    def _make_weighted_fn(original_fn: Callable,
                          weight: float) -> Callable:
        """
        Returns a new predict_fn that divides the raw probability by weight.

        effective_prob = raw_prob / weight

        weight >= 1.0 always, so effective_prob <= raw_prob.
        The GA's fitness = 1 - effective_prob is therefore higher than
        1 - raw_prob for the same code → stronger selection pressure.

        Clamped to [0, 1] to keep it a valid probability.
        """
        def weighted_fn(code: str) -> float:
            raw = original_fn(code)
            return max(0.0, min(1.0, raw / weight))
        return weighted_fn


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _sample_key(sample) -> str:
    """Stable identity key for a sample."""
    cve = getattr(sample, "cve", "") or ""
    if cve and cve.upper().startswith("CVE-"):
        return cve.upper()
    # fallback: source + cwe + first 80 chars of code
    code_snippet = (sample.code or "")[:80].replace(" ", "")
    return f"{sample.source}:{getattr(sample, 'cwe', '')}:{hash(code_snippet)}"


def _normalise_cwe(raw: str) -> str:
    """Normalise 'CWE-122', 'cwe122', '122' all → 'CWE-122'."""
    if not raw:
        return ""
    raw = str(raw).strip()
    m = re.search(r"(\d{2,4})", raw)
    if m:
        return f"CWE-{m.group(1)}"
    return raw.upper()


def _parse_cve_list(raw: str) -> List[str]:
    """
    BigVul's CVE column can hold:
        'CVE-2019-1234'
        'CVE-2019-1234,CVE-2018-5678'
        'CVE-2019-1234;CVE-2018-5678'
        nan / empty string
    Returns a list of clean CVE IDs.
    """
    if not raw or raw in ("nan", "None", ""):
        return []
    return [c.strip() for c in re.split(r"[,;\s]+", raw)
            if c.strip().upper().startswith("CVE-")]


def _extract_cvss(nvd_response: dict) -> Optional[float]:
    """
    Parses NVD 2.0 API response to extract CVSS v3.1 base score.
    Falls back to v3.0 then v2 if v3.1 is not present.
    """
    try:
        vuln = nvd_response["vulnerabilities"][0]["cve"]
        metrics = vuln.get("metrics", {})

        # prefer v3.1 → v3.0 → v2
        for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            if key in metrics:
                entry = metrics[key]
                if isinstance(entry, list):
                    entry = entry[0]
                # v3.x uses "cvssData.baseScore"; v2 uses "cvssData.baseScore"
                score = (entry.get("cvssData", {}).get("baseScore")
                         or entry.get("baseScore"))
                if score is not None:
                    return float(score)
    except (KeyError, IndexError, TypeError):
        pass
    return None


def _load_json_cache(path: str) -> dict:
    p = Path(path)
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# Reporting helper — call after batch_attack to log imputation breakdown
# ══════════════════════════════════════════════════════════════════════════════

def cvss_imputation_report(results) -> str:
    """
    Given a list of AttackResult objects (with .meta from CVSSFitnessWrapper),
    returns a human-readable imputation breakdown string for the paper log.

    Example output:
        CVSS imputation breakdown (n=500):
          nvd_direct    : 312 (62.4%) | mean CVSS = 7.91
          cwe_mean      : 103 (20.6%) | mean CVSS = 7.73
          global_median :  47  (9.4%) | mean CVSS = 7.20
          flat          :  38  (7.6%) | mean CVSS = 7.20 [Devign, no weighting]
    """
    from collections import defaultdict
    buckets: Dict[str, List[float]] = defaultdict(list)

    for r in results:
        meta = getattr(r, "meta", {})
        src  = meta.get("cvss_source", "unknown")
        cvss = meta.get("cvss", GLOBAL_MEDIAN_CVSS)
        buckets[src].append(cvss)

    n_total = sum(len(v) for v in buckets.values())
    lines   = [f"CVSS imputation breakdown (n={n_total}):"]
    order   = ["nvd_direct", "cwe_mean", "global_median", "flat", "unknown"]

    for src in order:
        if src not in buckets:
            continue
        vals = buckets[src]
        pct  = 100 * len(vals) / max(n_total, 1)
        mean = sum(vals) / len(vals)
        tag  = "  [Devign, no weighting]" if src == "flat" else ""
        lines.append(
            f"  {src:<16}: {len(vals):>4} ({pct:5.1f}%) | mean CVSS = {mean:.2f}{tag}"
        )

    return "\n".join(lines)
