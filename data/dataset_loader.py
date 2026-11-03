
import os, re, json, random, logging, requests, zipfile, io
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── CWE categories used in experiments ─────────────────────────────────────
TARGET_CWES = {
    "CWE-119", "CWE-120", "CWE-122", "CWE-125", "CWE-787",  # buffer
    "CWE-416",                                                  # use-after-free
    "CWE-190", "CWE-191",                                       # integer overflow
    "CWE-476",                                                  # null deref
    "CWE-078", "CWE-134",                                       # injection / format
}

BIGVUL_URL  = ("https://raw.githubusercontent.com/ZeoVan/MSR_20_Code_vulnerability"
               "_CSV_Dataset/master/MSR_data_cleaned.csv")
# DiverseVul - Chen et al., RAID 2023.  Primary HF repo.
DIVERSEVUL_HF = "bstee615/diversevul"
# PrimeVul — ICSE 2025 (Ding et al.).  Primary HF repo; fallback mirror below.
PRIMEVUL_HF        = "DLVulDet/PrimeVul"
PRIMEVUL_HF_MIRROR = "starsofchance/PrimeVul"


# ── Data record ─────────────────────────────────────────────────────────────

@dataclass
class VulnSample:
    code:      str
    label:     int          # 1 = vulnerable, 0 = clean
    cwe:       str = ""
    cve:       str = ""
    source:    str = ""     # bigvul | devign | diversevul | primevul
    func_name: str = ""
    meta:      Dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# BigVul
# ══════════════════════════════════════════════════════════════════════════════

class BigVulLoader:
   

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir  = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path   = self.cache_dir / "MSR_data_cleaned.csv"

    # ── download ──────────────────────────────────────────────────────────────
    def _download(self):
        if self.csv_path.exists():
            logger.info("BigVul CSV already cached.")
            return
        logger.info("Downloading BigVul CSV (~180 MB) …")
        r = requests.get(BIGVUL_URL, stream=True, timeout=300)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        buf = b""
        with tqdm(total=total, unit="B", unit_scale=True, desc="BigVul") as pbar:
            for chunk in r.iter_content(1 << 20):
                buf += chunk
                pbar.update(len(chunk))
        self.csv_path.write_bytes(buf)
        logger.info("BigVul saved to %s", self.csv_path)

    # ── load ──────────────────────────────────────────────────────────────────
    def load(self,
             max_samples: int = 10_000,
             balance: bool = True,
             cwe_filter: Optional[set] = None) -> List[VulnSample]:
        self._download()
        df = pd.read_csv(self.csv_path, low_memory=False)

        # rename columns defensively
        col_map = {}
        for c in df.columns:
            lc = c.lower().replace(" ", "_")
            if "func_before" in lc or "func" == lc:
                col_map[c] = "func"
            elif lc in ("vul", "label", "vulnerable"):
                col_map[c] = "label"
            elif "cwe" in lc:
                col_map[c] = "cwe"
            elif "cve" in lc:
                col_map[c] = "cve"
        df = df.rename(columns=col_map)

        required = {"func", "label"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"BigVul CSV missing columns: {missing}. "
                             f"Available: {list(df.columns)[:10]}")

        df = df.dropna(subset=["func", "label"])
        df["label"] = df["label"].astype(int)
        if "cwe" not in df.columns:
            df["cwe"] = ""
        if "cve" not in df.columns:
            df["cve"] = ""

        # optional CWE filter
        if cwe_filter:
            mask = df["cwe"].apply(
                lambda x: any(c in str(x) for c in cwe_filter))
            df = df[mask]

        vuln  = df[df["label"] == 1]
        clean = df[df["label"] == 0]

        if balance:
            n = min(len(vuln), len(clean), max_samples // 2)
            vuln  = vuln.sample(n, random_state=42)
            clean = clean.sample(n, random_state=42)
            df    = pd.concat([vuln, clean]).sample(frac=1, random_state=42)
        else:
            df = df.head(max_samples)

        samples = []
        for _, row in df.iterrows():
            samples.append(VulnSample(
                code      = str(row["func"]),
                label     = int(row["label"]),
                cwe       = str(row.get("cwe", "")),
                cve       = str(row.get("cve", "")),
                source    = "bigvul",
                func_name = str(row.get("func_name", "")),
            ))
        logger.info("BigVul: loaded %d samples", len(samples))
        return samples


# ══════════════════════════════════════════════════════════════════════════════
# Devign
# ══════════════════════════════════════════════════════════════════════════════

class DevignLoader:
    """
    Loads the Devign dataset via HuggingFace datasets.
    Falls back to a local JSON file if network is unavailable.
    """

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self,
             max_samples: int = 5_000,
             balance: bool = True) -> List[VulnSample]:
        try:
            from datasets import load_dataset as hf_load
            logger.info("Loading Devign from HuggingFace …")
            ds = hf_load(DEVIGN_HF, split="train",
                         cache_dir=str(self.cache_dir))
            rows = [{"func": r["func"], "label": r["target"]} for r in ds]
        except Exception as e:
            logger.warning("HF load failed (%s). Trying local JSON …", e)
            local = self.cache_dir / "devign.json"
            if not local.exists():
                raise FileNotFoundError(
                    f"Devign: HF load failed and no local file at {local}. "
                    "Download from https://drive.google.com/file/d/"
                    "1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF") from e
            with open(local) as f:
                raw = json.load(f)
            rows = [{"func": r["func"], "label": r["target"]} for r in raw]

        df = pd.DataFrame(rows).dropna(subset=["func", "label"])
        df["label"] = df["label"].astype(int)

        if balance:
            vuln  = df[df["label"] == 1]
            clean = df[df["label"] == 0]
            n     = min(len(vuln), len(clean), max_samples // 2)
            df    = pd.concat([
                vuln.sample(n, random_state=42),
                clean.sample(n, random_state=42)
            ]).sample(frac=1, random_state=42)
        else:
            df = df.head(max_samples)

        samples = [
            VulnSample(code=row["func"], label=int(row["label"]),
                       source="devign")
            for _, row in df.iterrows()
        ]
        logger.info("Devign: loaded %d samples", len(samples))
        return samples


# ══════════════════════════════════════════════════════════════════════════════
# DiverseVul  (Chen et al., RAID 2023)
# ══════════════════════════════════════════════════════════════════════════════

class DiverseVulLoader:
    """Loads the DiverseVul dataset (Chen et al., RAID 2023) from HuggingFace."""

    def __init__(self, cache_dir: str = "data/cache"):
        """Initialize the loader with a local cache directory for the downloaded split."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self,
             max_samples: int = 10_000,
             balance: bool = True,
             cwe_filter: Optional[set] = None) -> List["VulnSample"]:
        """Load DiverseVul functions, optionally CWE-filtered and label-balanced."""
        from datasets import load_dataset
        try:
            ds = load_dataset(DIVERSEVUL_HF, split="train",
                              cache_dir=str(self.cache_dir))
        except Exception as e:
            logger.warning("DiverseVul download failed: %s. Using empty loader.", e)
            return []

        rows = []
        for ex in ds:
            code = ex.get("func") or ex.get("function") or ""
            label = int(ex.get("target", ex.get("label", 0)))
            cwe_field = ex.get("cwe", "")
            if isinstance(cwe_field, (list, tuple)):
                cwe = cwe_field[0] if cwe_field else ""
            else:
                cwe = str(cwe_field)
            if not code:
                continue
            if cwe_filter and not any(c in str(cwe) for c in cwe_filter):
                continue
            rows.append((code, label, cwe))

        vuln  = [r for r in rows if r[1] == 1]
        clean = [r for r in rows if r[1] == 0]
        import random
        rng = random.Random(42)
        if balance:
            n = min(len(vuln), len(clean), max_samples // 2)
            vuln  = rng.sample(vuln, n) if n <= len(vuln) else vuln
            clean = rng.sample(clean, n) if n <= len(clean) else clean
            rows = vuln + clean
            rng.shuffle(rows)
        else:
            rows = rows[:max_samples]

        samples = []
        for code, label, cwe in rows:
            samples.append(VulnSample(
                code   = str(code),
                label  = int(label),
                cwe    = str(cwe),
                cve    = "",
                source = "diversevul",
            ))
        logger.info("DiverseVul: loaded %d samples", len(samples))
        return samples


# ══════════════════════════════════════════════════════════════════════════════
# PrimeVul  (ICSE 2025)
# ══════════════════════════════════════════════════════════════════════════════

class PrimeVulLoader:
    


    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # PrimeVul ships pre-split; we merge then re-split for consistency
        self.cache_jsonl = self.cache_dir / "primevul_merged.jsonl"

    # ── HF download ───────────────────────────────────────────────────────────
    def _load_from_hf(self, hf_id: str) -> List[dict]:
        from datasets import load_dataset as hf_load
        logger.info("Loading PrimeVul from HuggingFace (%s) …", hf_id)
        rows = []
        for split_name in ("train", "validation", "test"):
            try:
                ds = hf_load(hf_id, split=split_name,
                             cache_dir=str(self.cache_dir))
                for r in ds:
                    rows.append(dict(r))
                logger.info("  PrimeVul split '%s': %d rows", split_name, len(ds))
            except Exception as e:
                logger.warning("  PrimeVul split '%s' failed: %s", split_name, e)
        return rows

    def _ensure_cached(self):
       
        if self.cache_jsonl.exists():
            logger.info("PrimeVul: using cached file %s", self.cache_jsonl)
            return

        # 2. Manual placement (HPC / no internet)
        manual = {
            "train": self.cache_dir / "primevul_train.jsonl",
            "valid": self.cache_dir / "primevul_valid.jsonl",
            "test":  self.cache_dir / "primevul_test.jsonl",
        }
        manual_found = [p for p in manual.values() if p.exists()]
        if manual_found:
            logger.info("PrimeVul: merging %d manually placed JSONL files …",
                        len(manual_found))
            total = 0
            with open(self.cache_jsonl, "w") as out:
                for split_name, path in manual.items():
                    if not path.exists():
                        logger.warning("  PrimeVul: %s not found, skipping.", path.name)
                        continue
                    count = 0
                    with open(path) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                out.write(line + "\n")
                                count += 1
                    logger.info("  %s: %d rows", split_name, count)
                    total += count
            logger.info("PrimeVul: merged %d total rows → %s", total, self.cache_jsonl)
            return

        # 3. HuggingFace auto-download
        rows = []
        for hf_id in (PRIMEVUL_HF, PRIMEVUL_HF_MIRROR):
            try:
                rows = self._load_from_hf(hf_id)
                if rows:
                    break
            except Exception as e:
                logger.warning("PrimeVul HF load failed (%s): %s", hf_id, e)

        if rows:
            with open(self.cache_jsonl, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            logger.info("PrimeVul: cached %d rows to %s", len(rows), self.cache_jsonl)
            return

        # 4. Nothing worked
        raise RuntimeError(
            "PrimeVul dataset not found. Do ONE of the following:\n\n"
            "  OPTION A — manual download (recommended for HPC):\n"
            "    1. Go to: https://drive.google.com/drive/folders/1KuIYgdbwe3OS8qeMHxWxUSPSJ7hNnKDG\n"
            "    2. Download: primevul_train.jsonl  primevul_valid.jsonl  primevul_test.jsonl\n"
            f"   3. Place all three files in: {self.cache_dir}/\n\n"
            "  OPTION B — auto-download (needs internet + HuggingFace datasets):\n"
            "    pip install datasets\n"
            "    python run_experiment.py --splits primevul ...\n"
        )

    @staticmethod
    def _parse_row(r: dict) -> Optional[VulnSample]:
        """Normalise one PrimeVul row to VulnSample."""
        # code field: PrimeVul uses 'func_before' or 'func'
        code = (r.get("func_before") or r.get("func") or
                r.get("code") or "").strip()
        if len(code) < 30:
            return None

        # label field: 'target' (int) or 'label'
        raw_label = r.get("target", r.get("label", -1))
        try:
            label = int(raw_label)
        except (TypeError, ValueError):
            return None
        if label not in (0, 1):
            return None

        # CWE: may be a list or string
        raw_cwe = r.get("cwe", r.get("CWE", ""))
        if isinstance(raw_cwe, list):
            cwe = raw_cwe[0] if raw_cwe else ""
        else:
            cwe = str(raw_cwe)
        # Normalise to "CWE-NNN"
        m = re.search(r"CWE-?\d+", cwe, re.IGNORECASE)
        cwe = m.group(0).upper().replace("CWE", "CWE-").replace("CWE--", "CWE-") if m else ""

        cve = str(r.get("cve_id", r.get("CVE", r.get("cve", "")))).strip()
        func_name = str(r.get("func_name", r.get("function_name", ""))).strip()

        return VulnSample(
            code      = code,
            label     = label,
            cwe       = cwe,
            cve       = cve,
            source    = "primevul",
            func_name = func_name,
        )

    def load(self,
             max_samples: int = 5_000,
             balance:     bool = True,
             cwe_filter:  Optional[set] = None) -> List[VulnSample]:
        
        self._ensure_cached()

        all_vuln:  List[VulnSample] = []
        all_clean: List[VulnSample] = []

        with open(self.cache_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                s = self._parse_row(row)
                if s is None:
                    continue
                if cwe_filter and s.cwe and s.cwe not in cwe_filter:
                    continue
                if s.label == 1:
                    all_vuln.append(s)
                else:
                    all_clean.append(s)

        logger.info("PrimeVul raw: %d vulnerable, %d benign",
                    len(all_vuln), len(all_clean))

        if not all_vuln:
            logger.warning("PrimeVul: 0 vulnerable samples — "
                           "check cwe_filter or cache file.")
            return []

        if balance:
            n = min(len(all_vuln), len(all_clean), max_samples // 2)
            random.seed(42)
            samples = (random.sample(all_vuln,  n) +
                       random.sample(all_clean, n))
        else:
            # keep natural imbalance but cap total
            samples = (all_vuln + all_clean)[:max_samples]

        random.shuffle(samples)
        logger.info("PrimeVul: loaded %d samples  "
                    "(vuln=%d  benign=%d)",
                    len(samples),
                    sum(1 for s in samples if s.label == 1),
                    sum(1 for s in samples if s.label == 0))
        return samples


# ══════════════════════════════════════════════════════════════════════════════
# Combined loader + splits
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(
    cache_dir:    str   = "data/cache",
    max_bigvul:   int   = 10_000,
    max_devign:   int   = 5_000,
    max_diversevul: int = 3_000,
    max_primevul: int   = 5_000,
    train_ratio:  float = 0.70,
    val_ratio:    float = 0.15,
    seed:         int   = 42,
) -> Tuple[List[VulnSample], List[VulnSample], List[VulnSample]]:
    
    all_samples: List[VulnSample] = []

    # BigVul
    try:
        all_samples += BigVulLoader(cache_dir).load(
            max_samples=max_bigvul, balance=True,
            cwe_filter=TARGET_CWES)
    except Exception as e:
        logger.error("BigVul load error: %s", e)

    # Devign
    try:
        all_samples += DevignLoader(cache_dir).load(
            max_samples=max_devign, balance=True)
    except Exception as e:
        logger.error("Devign load error: %s", e)

    # DiverseVul
    try:
        all_samples += DiverseVulLoader(cache_dir).load(
            max_samples=max_diversevul, balance=True,
            cwe_filter=TARGET_CWES)
    except Exception as e:
        logger.error("DiverseVul load error: %s", e)

    # PrimeVul  (skip if max_primevul=0)
    if max_primevul > 0:
        try:
            all_samples += PrimeVulLoader(cache_dir).load(
                max_samples=max_primevul, balance=True)
        except Exception as e:
            logger.error("PrimeVul load error: %s", e)

    if not all_samples:
        raise RuntimeError("No samples loaded from any dataset.")

    random.seed(seed)
    random.shuffle(all_samples)

    n     = len(all_samples)
    n_tr  = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = all_samples[:n_tr]
    val   = all_samples[n_tr: n_tr + n_val]
    test  = all_samples[n_tr + n_val:]

    logger.info("Dataset splits — train: %d | val: %d | test: %d",
                len(train), len(val), len(test))
    return train, val, test


def get_attack_subset(
    test_set: List[VulnSample],
    n: int = 500,
    seed: int = 42,
) -> List[VulnSample]:
    """Return up to n vulnerable samples from the test set for attacking."""
    vuln = [s for s in test_set if s.label == 1]
    random.seed(seed)
    random.shuffle(vuln)
    return vuln[:n]
