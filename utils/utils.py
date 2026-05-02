"""utils.py — shared helpers."""

from __future__ import annotations
import random, os, logging, re
from typing import List

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def jaccard_similarity(code1: str, code2: str) -> float:
    tokens1 = set(re.findall(r'\w+', code1))
    tokens2 = set(re.findall(r'\w+', code2))
    if not tokens1 and not tokens2:
        return 1.0
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)


class CodeNaturalnessMeasurer:
    """
    Measures code naturalness using CodeGPT perplexity.
    Falls back to a simple n-gram perplexity estimate if model unavailable.
    """

    def __init__(self, model_id: str = "microsoft/CodeGPT-small-java"):
        self._model_id = model_id
        self._model    = None
        self._tok      = None
        self._device   = torch.device("cuda" if torch.cuda.is_available()
                                       else "cpu")
        self._loaded   = False

    def _lazy_load(self):
        if self._loaded:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self._tok   = AutoTokenizer.from_pretrained(self._model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_id).to(self._device)
            self._model.eval()
            self._loaded = True
        except Exception as e:
            logger.warning("CodeGPT not loadable (%s); using n-gram fallback.", e)

    def perplexity(self, code: str) -> float:
        self._lazy_load()
        if not self._loaded:
            return self._ngram_perplexity(code)
        try:
            enc = self._tok(code, return_tensors="pt",
                            truncation=True, max_length=512).to(self._device)
            with torch.no_grad():
                out = self._model(**enc, labels=enc["input_ids"])
            import math
            return math.exp(float(out.loss.item()))
        except Exception:
            return self._ngram_perplexity(code)

    @staticmethod
    def _ngram_perplexity(code: str, n: int = 3) -> float:
        """Simple n-gram perplexity estimate (fallback)."""
        tokens = re.findall(r'\w+|[^\w\s]', code)
        if len(tokens) < n:
            return 1.0
        from collections import Counter
        ngrams  = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        counts  = Counter(ngrams)
        total   = len(ngrams)
        log_p   = sum(np.log(c/total) for c in counts.values())
        return float(np.exp(-log_p / max(total, 1)))


def configure_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S")
