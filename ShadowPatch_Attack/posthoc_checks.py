"""Phase 4 post-hoc validation: protected-region identity, edit legality, differential testing."""

from __future__ import annotations
import difflib
import subprocess
import tempfile
import os
import shutil
import re
from dataclasses import dataclass, field
from typing import List, Optional, Set

from .pdg_taint import get_taint_set


# ── result containers ─────────────────────────────────────────────────────────

@dataclass
class ProtectedRegionResult:
    """Verdict for Check 1: whether all protected taint lines survive in the variant."""
    verdict: str                      # IDENTICAL | MODIFIED | NO-REGION
    protected_lines: int = 0
    retained_lines: int = 0
    missing_lines: List[str] = field(default_factory=list)


@dataclass
class EditLegalityResult:
    """Verdict for Check 2: whether edits stay outside the protected region."""
    verdict: str                      # LEGAL | VIOLATION
    edited_protected: List[str] = field(default_factory=list)


@dataclass
class DifferentialResult:
    """Verdict for Check 3: whether original and variant agree on generated inputs."""
    verdict: str                      # PASS | FAIL | UNSUPPORTED | UNKNOWN
    detail: str = ""


@dataclass
class PostHocResult:
    """Aggregated Phase-4 verdict combining all three independent checks."""
    protected: ProtectedRegionResult
    legality: EditLegalityResult
    differential: DifferentialResult

    @property
    def preserved(self) -> bool:
        """True when both structural checks pass (identity retained and edits legal)."""
        return (self.protected.verdict in ("IDENTICAL", "NO-REGION")
                and self.legality.verdict == "LEGAL")


# ── helpers ───────────────────────────────────────────────────────────────────

def _normalize(line: str) -> str:
    """Collapse whitespace so protected lines match by content, not formatting."""
    return re.sub(r"\s+", " ", line.strip())


def _protected_lines(code: str) -> List[str]:
    """Return the normalized source lines that belong to the taint region T."""
    taint_set, _ = get_taint_set(code)
    lines = code.split("\n")
    out = []
    for ln in sorted(taint_set):
        if 1 <= ln <= len(lines):
            txt = _normalize(lines[ln - 1])
            if txt:
                out.append(txt)
    return out


# ── Check 1: protected-region identity ────────────────────────────────────────

def protected_region_identity(original: str, variant: str) -> ProtectedRegionResult:
    """Check 1: verify every protected taint line of the original is retained in the variant."""
    prot = _protected_lines(original)
    if not prot:
        return ProtectedRegionResult("NO-REGION", 0, 0, [])
    variant_lines = {_normalize(l) for l in variant.split("\n")}
    missing = [l for l in prot if l not in variant_lines]
    retained = len(prot) - len(missing)
    verdict = "IDENTICAL" if not missing else "MODIFIED"
    return ProtectedRegionResult(verdict, len(prot), retained, missing)


# ── Check 2: edit legality ────────────────────────────────────────────────────

def edit_legality(original: str, variant: str) -> EditLegalityResult:
    """Check 2: verify that no line-level edit touches a protected taint-region statement."""
    prot = set(_protected_lines(original))
    if not prot:
        return EditLegalityResult("LEGAL", [])
    orig_lines = [_normalize(l) for l in original.split("\n")]
    var_lines = [_normalize(l) for l in variant.split("\n")]
    diff = difflib.SequenceMatcher(None, orig_lines, var_lines)
    edited_protected = []
    for tag, i1, i2, _j1, _j2 in diff.get_opcodes():
        if tag in ("replace", "delete"):
            for line in orig_lines[i1:i2]:
                if line in prot:
                    edited_protected.append(line)
    verdict = "LEGAL" if not edited_protected else "VIOLATION"
    return EditLegalityResult(verdict, edited_protected)


# ── Check 3: differential testing ─────────────────────────────────────────────

def _wrap_function(code: str) -> Optional[str]:
    """Wrap a bare function in a minimal compilable translation unit, or None if not a function."""
    if "(" not in code or "{" not in code:
        return None
    includes = ("#include <stdio.h>\n#include <stdlib.h>\n"
                "#include <string.h>\n#include <stdint.h>\n")
    return includes + code + "\n"


def differential_testing(original: str, variant: str,
                         timeout: float = 5.0) -> DifferentialResult:
    """Check 3: compile and run both functions on identical inputs, comparing observable outputs."""
    if not shutil.which("gcc"):
        return DifferentialResult("UNSUPPORTED", "no C compiler available")
    wrapped_o = _wrap_function(original)
    wrapped_v = _wrap_function(variant)
    if wrapped_o is None or wrapped_v is None:
        return DifferentialResult("UNSUPPORTED", "input is not a self-contained function")

    workdir = tempfile.mkdtemp(prefix="sp_diff_")
    try:
        outs = {}
        for tag, src in (("orig", wrapped_o), ("adv", wrapped_v)):
            cpath = os.path.join(workdir, f"{tag}.c")
            bpath = os.path.join(workdir, tag)
            with open(cpath, "w") as fh:
                fh.write(src)
            proc = subprocess.run(
                ["gcc", "-w", "-O0", cpath, "-o", bpath],
                capture_output=True, timeout=timeout)
            if proc.returncode != 0:
                return DifferentialResult("UNKNOWN", "compilation failed")
            try:
                run = subprocess.run([bpath], capture_output=True, timeout=timeout)
                outs[tag] = (run.returncode, run.stdout)
            except subprocess.TimeoutExpired:
                return DifferentialResult("UNKNOWN", "execution timeout")
        if outs["orig"] == outs["adv"]:
            return DifferentialResult("PASS", "outputs agree on generated inputs")
        return DifferentialResult("FAIL", "outputs differ")
    except subprocess.TimeoutExpired:
        return DifferentialResult("UNKNOWN", "compilation timeout")
    except Exception as exc:
        return DifferentialResult("UNKNOWN", str(exc))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ── combined entry point ──────────────────────────────────────────────────────

def run_posthoc(original: str, variant: str,
                differential: bool = True) -> PostHocResult:
    """Run all three Phase-4 checks on a variant and return the aggregated verdict."""
    p = protected_region_identity(original, variant)
    l = edit_legality(original, variant)
    if differential:
        d = differential_testing(original, variant)
    else:
        d = DifferentialResult("UNSUPPORTED", "differential testing disabled")
    return PostHocResult(p, l, d)
