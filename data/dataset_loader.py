"""
dataset_loader.py
=================
Loads four real-world C/C++ vulnerability datasets:
  1. BigVul   – MSR 2020, 188 k CVE-annotated functions
  2. Devign   – NeurIPS 2019, 27 k functions from FFmpeg/QEMU/LibTiff/VLC
  3. SARD     – NIST Juliet Test Suite C/C++ 1.3, 81 CWE categories
  4. PrimeVul – ICSE 2025, ~7k vuln / ~229k benign; human-level label accuracy,
                chronological split, paired vuln/patch samples.
                The hardest realistic benchmark — models that score ~68% F1
                on BigVul often drop to <5% F1 here.
                HuggingFace: DLVulDet/PrimeVul  (or starsofchance/PrimeVul)

"""

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
SARD_URL    = ("https://samate.nist.gov/SARD/downloads/test-suites/"
               "2022-08-11-juliet-test-suite-for-c-cplusplus-v1-3.zip")
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
    source:    str = ""     # bigvul | devign | sard
    func_name: str = ""
    meta:      Dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# BigVul
# ══════════════════════════════════════════════════════════════════════════════

class BigVulLoader:
    """
    Loads the MSR 2020 BigVul dataset.
    CSV columns used: func_before, vul, CWE ID, CVE ID
    """

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
# SARD / Juliet
# ══════════════════════════════════════════════════════════════════════════════

class SARDLoader:
    """
    Loads NIST Juliet Test Suite C/C++ 1.3.

    The zip extracts to:
        juliet/C/testcases/CWE119_Buffer_Overread/s01/.../*.c
                           CWE416_Use_After_Free/s01/.../*.c
                           ...
    So the CWE folder sits at depth 3 inside juliet/, not at the root.
    We scan all .c files with rglob and extract the CWE from whichever
    path component starts with "CWE", normalising "CWE119" → "CWE-119".

    label: 1 if filename contains "bad"  (vulnerable)
           0 if filename contains "good" (safe)

    cwe_filter: if None, ALL CWEs are loaded (no filtering).
    When called from build_splits() with cwe_filter=TARGET_CWES, only
    the 11 target CWEs are kept.  If that still yields 0 samples we
    automatically fall back to no filter so SARD is never empty.
    """

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sard_dir  = self.cache_dir / "juliet"
        # Juliet extracts into juliet/C/testcases/ — search from there if present
        tc = self.sard_dir / "C" / "testcases"
        self.search_root = tc if tc.exists() else self.sard_dir

    def _refresh_search_root(self):
        """Re-resolve search root after extraction."""
        tc = self.sard_dir / "C" / "testcases"
        self.search_root = tc if tc.exists() else self.sard_dir

    # ── download ──────────────────────────────────────────────────────────────
    def _download(self):
        if self.sard_dir.exists() and any(self.sard_dir.rglob("*.c")):
            logger.info("SARD already cached at %s", self.sard_dir)
            self._refresh_search_root()
            return
        logger.info("Downloading SARD Juliet C/C++ suite (~150 MB) …")
        try:
            r = requests.get(SARD_URL, stream=True, timeout=300)
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(str(self.sard_dir))
            logger.info("SARD extracted to %s", self.sard_dir)
            self._refresh_search_root()
        except Exception as e:
            logger.warning("SARD download failed: %s. Using empty loader.", e)

    # ── CWE extraction ─────────────────────────────────────────────────────────
    @staticmethod
    def _cwe_from_path(path: Path) -> str:
        """
        Walk path components and return the first one that starts with CWE,
        normalised to "CWE-NNN" format.

        Examples:
          CWE119_Buffer_Overread   → CWE-119
          CWE-119_Buffer_Overread  → CWE-119
          CWE416_Use_After_Free    → CWE-416
        """
        for part in path.parts:
            if part.upper().startswith("CWE"):
                digits = "".join(c for c in part if c.isdigit())
                return ("CWE-" + digits) if digits else part.split("_")[0].upper()
        return ""

    # ── parse one .c file ──────────────────────────────────────────────────────
    @staticmethod
    def _extract_functions(path: Path, cwe: str) -> List[Tuple[str, int, str]]:
        """
        Juliet files contain multiple functions per file. Each file looks like:

            void CWE114_..._01_bad()       ← sig on one line
            {                              ← brace on next line (Allman style)
                ...
            }

            static void goodG2B()
            {
                ...
            }

            void CWE114_..._01_good()
            {
                goodG2B();
            }

        Label rule (applied to the function name before the opening paren):
          ends with "_bad" or contains "_bad_"  → 1  (vulnerable)
          contains "good"                        → 0  (safe)
          anything else (main, printLine, etc.)  → skip
        """
        try:
            src = path.read_text(errors="replace")
        except Exception:
            return []

        funcs  = []
        lines  = src.splitlines(keepends=True)
        n      = len(lines)
        i      = 0

        RETURN_TYPES = ("void ", "int ", "char ", "static ", "unsigned ",
                        "size_t ", "long ", "float ", "double ", "short ")

        while i < n:
            stripped = lines[i].strip()

            # Skip blanks, comments, preprocessor, struct/typedef lines
            if (not stripped
                    or stripped.startswith("//")
                    or stripped.startswith("*")
                    or stripped.startswith("/*")
                    or stripped.startswith("#")
                    or stripped.startswith("typedef")
                    or stripped == "{" or stripped == "}"):
                i += 1
                continue

            # A function signature must:
            #   1. contain a known return type keyword
            #   2. contain "(" (parameter list)
            #   3. NOT end with ";" (that would be a prototype)
            #   4. next non-blank line is exactly "{"  (Allman — confirmed by debug)
            if ("(" in stripped
                    and not stripped.endswith(";")
                    and any(stripped.startswith(k) or (" " + k) in stripped
                            for k in RETURN_TYPES)):

                # find the opening brace line
                j = i + 1
                while j < n and lines[j].strip() == "":
                    j += 1

                if j < n and lines[j].strip() == "{":
                    # this is a function definition — determine label
                    # extract name: token before the first "("
                    name_token = stripped.split("(")[0].split()[-1].lower()
                    # strip pointer stars e.g. "*foo" -> "foo"
                    name_token = name_token.lstrip("*")

                    if name_token.endswith("_bad") or "_bad_" in name_token:
                        label = 1
                    elif "good" in name_token or name_token.endswith("_good"):
                        label = 0
                    else:
                        label = -1   # helper/main/sink — skip

                    # collect body via brace matching starting at "{"
                    buf   = []
                    depth = 0
                    k     = j       # start from the "{" line
                    while k < n:
                        buf.append(lines[k])
                        depth += lines[k].count("{") - lines[k].count("}")
                        k     += 1
                        if depth <= 0:
                            break

                    if label != -1:
                        # prepend signature so the code makes sense
                        code = lines[i].rstrip() + "\n" + "".join(buf)
                        if len(code) >= 50:
                            funcs.append((code, label, cwe))

                    i = k   # jump past the function body
                    continue

            i += 1

        return funcs

    # ── load ──────────────────────────────────────────────────────────────────
    def load(self,
             max_samples: int = 3_000,
             balance: bool = True,
             cwe_filter: Optional[set] = None) -> List[VulnSample]:
        self._download()

        logger.info("SARD: scanning %s …", self.search_root)
        all_c = list(self.search_root.rglob("*.c"))
        logger.info("SARD: found %d .c files", len(all_c))

        all_vuln, all_clean = [], []
        for cfile in all_c:
            cwe = self._cwe_from_path(cfile)
            if cwe_filter and cwe not in cwe_filter:
                continue
            for code, label, cwe_out in self._extract_functions(cfile, cwe):
                sample = VulnSample(code=code, label=label,
                                    cwe=cwe_out, source="sard")
                if label == 1:
                    all_vuln.append(sample)
                else:
                    all_clean.append(sample)

        # ── fallback: if cwe_filter excluded everything, load without filter ──
        if not all_vuln and not all_clean and cwe_filter:
            logger.warning(
                "SARD: cwe_filter excluded all %d files — "
                "retrying without CWE filter.", len(all_c))
            for cfile in all_c:
                cwe = self._cwe_from_path(cfile)
                for code, label, cwe_out in self._extract_functions(cfile, cwe):
                    sample = VulnSample(code=code, label=label,
                                        cwe=cwe_out, source="sard")
                    if label == 1:
                        all_vuln.append(sample)
                    else:
                        all_clean.append(sample)

        if not all_vuln and not all_clean:
            logger.warning("SARD: still 0 samples — returning empty list.")
            return []

        logger.info("SARD raw: %d vulnerable, %d clean",
                    len(all_vuln), len(all_clean))

        if balance:
            n = min(len(all_vuln), len(all_clean), max_samples // 2)
            random.seed(42)
            samples = (random.sample(all_vuln, n) +
                       random.sample(all_clean, n))
        else:
            samples = (all_vuln + all_clean)[:max_samples]

        random.shuffle(samples)
        logger.info("SARD: loaded %d samples", len(samples))
        return samples


# ══════════════════════════════════════════════════════════════════════════════
# PrimeVul  (ICSE 2025)
# ══════════════════════════════════════════════════════════════════════════════

class PrimeVulLoader:
    """
    Loads the PrimeVul dataset (Ding et al., ICSE 2025).

    PrimeVul is the hardest and most realistic C/C++ vuln-detection benchmark:
      • ~7k vulnerable functions, ~229k benign — heavily imbalanced like reality
      • Labels verified to human-level accuracy (vs. automated heuristics in
        BigVul / Devign which have known noise)
      • Chronological train/val/test split → no temporal data leakage
      • Paired samples (vuln function + its patch) for subtle-pattern evaluation
      • 140+ CWEs, real CVE IDs, NVD links, and commit metadata

    Why include it: models that score ~68% F1 on BigVul drop to <5% F1 on
    PrimeVul (Ding et al. 2024).  Showing ShadowPatch still evades detectors
    on PrimeVul is a much stronger claim than BigVul alone.

    HuggingFace:  DLVulDet/PrimeVul  (primary)
                  starsofchance/PrimeVul  (mirror with pre-split structure)
    GitHub:       https://github.com/DLVulDet/PrimeVul

    JSONL columns used:
      func_before / func   — function source code
      target / label       — 1 = vulnerable, 0 = benign
      cwe                  — CWE ID string (e.g. "CWE-119")
      cve_id / CVE         — CVE identifier (for CVSS lookup)
    """

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
        """
        Resolution order:
          1. Merged cache already exists           → done, load it
          2. Manual JSONL files in cache_dir       → merge them into cache
             (primevul_train.jsonl, primevul_valid.jsonl, primevul_test.jsonl)
          3. HuggingFace auto-download             → merge into cache
          4. Raise with clear instructions
        """
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
        """
        Args:
            max_samples: total cap (balanced: max_samples/2 vuln + max_samples/2 benign)
            balance:     if True, undersample the majority class (benign)
                         PrimeVul is ~1:33 imbalanced so balance=True is important
            cwe_filter:  optional set of CWE IDs to restrict to (e.g. TARGET_CWES)
                         If None, all CWEs are included.
        """
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
    max_sard:     int   = 3_000,
    max_primevul: int   = 5_000,
    train_ratio:  float = 0.70,
    val_ratio:    float = 0.15,
    seed:         int   = 42,
) -> Tuple[List[VulnSample], List[VulnSample], List[VulnSample]]:
    """
    Returns (train, val, test) splits from all four datasets combined.
    Set max_primevul=0 to exclude PrimeVul from the combined split.
    """
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

    # SARD
    try:
        all_samples += SARDLoader(cache_dir).load(
            max_samples=max_sard, balance=True,
            cwe_filter=TARGET_CWES)
    except Exception as e:
        logger.error("SARD load error: %s", e)

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
