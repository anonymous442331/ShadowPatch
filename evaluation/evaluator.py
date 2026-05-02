"""
evaluator.py  —  ShadowPatch
================================
All evaluation tables and figures.

Tables
──────
  Table 1   Clean model performance              (eval_clean_performance)
  Table 2   Attack success per split × model     (eval_attack_per_split)
  Table 3   Cross-model transferability          (eval_transferability)
  Table 4   CWE-specific ASR                     (eval_cwe_specific)
  Table 5   Defense evaluation D1–D5             (eval_defenses)
  Table 6   Ablation study                       (eval_ablation)
  Table 7   ShadowPatch vs comparison frameworks (eval_comparison)

Figures
───────
  Fig 1   ASR vs Query Budget            (plot_asr_vs_budget)
  Fig 2   Transform usage heatmap        (plot_transform_heatmap)
  Fig 3   Probability shift histogram    (plot_prob_shift)
  Fig 4   CWE-specific ASR bar           (plot_cwe_asr)
  Fig 5   Ablation bar                   (plot_ablation)
  Fig 6   Per-dataset-split ASR          (plot_per_split_asr)      
  Fig 7   Flat GA vs CVSS GA             (plot_flat_vs_cvss)        
  Fig 8   ShadowPatch vs frameworks      (plot_comparison)          
  Fig 9   CVSS severity-bucketed ASR     (plot_cvss_weighted_asr)   

CCS style guide
───────────────
  • Font: Type-1 / PDF (no bitmap fonts)
  • Width: single-col = 3.33 in, double-col = 6.97 in
  • Height: ≤ 2.8 in for single-row figures (keeps column ratio)
  • Font size: axis labels 8 pt, tick labels 7 pt, legend 6.5 pt
  • Line width: 1.0 pt for axes, 0.8 pt for data
  • No seaborn default themes; use manual rcParams below
  • All figures exported as both PDF (vector) and PNG (300 dpi)
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from attack.genetic_optimizer  import AttackResult, GeneticAttacker, summarise_results
from attack.code_transformer   import TRANSFORM_IDS, apply_sequence
from attack.pdg_taint          import get_taint_set

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# (applied globally once)
# ══════════════════════════════════════════════════════════════════════════════

# Column widths in inches (ACM CCS two-column template)
FIG_SINGLE = 3.33    # one column
FIG_DOUBLE = 6.97    # two columns (full width)
FIG_HEIGHT = 2.4     # standard row height

# Palette — colourblind-friendly, works in greyscale
PALETTE = {
    "linevul":       "#1f77b4",   # blue
    "reveal":        "#d62728",   # red
    "ivdetect":      "#2ca02c",   # green
    "graphcodebert": "#9467bd",   # purple
    # comparison frameworks
    "ShadowPatch-GA": "#1f77b4",
    "ALERT":          "#ff7f0e",
    "MHM":            "#2ca02c",
    "Dead+Rename":    "#d62728",
    "Random":         "#8c564b",
    # splits
    "bigvul":         "#1f77b4",
    "devign":         "#ff7f0e",
    "sard":           "#2ca02c",
    "combined":       "#9467bd",
}

CWE_ORDER = [
    "CWE-119", "CWE-120", "CWE-122", "CWE-125",
    "CWE-787", "CWE-416", "CWE-190", "CWE-476",
]

TRANSFORM_CATEGORIES = {
    "Loop\n(T1–T4)":    ["T1", "T2", "T3", "T4"],
    "Pointer\n(T5–T8)": ["T5", "T6", "T7", "T8"],
    "CF\n(T9–T13)":     ["T9",  "T10", "T11", "T12", "T13"],
    "Dead\n(T14–T18)":  ["T14", "T15", "T16", "T17", "T18"],
}

SPLIT_LABELS = {
    "bigvul":   "BigVul",
    "devign":   "Devign",
    "sard":     "SARD",
    "primevul": "PrimeVul",
    "combined": "Combined",
}


def _ccs_style():
    """Apply CCS-compliant rcParams.  Call once at module level."""
    plt.rcParams.update({
        "font.family":         "DejaVu Sans",
        "font.size":           8,
        "axes.labelsize":      8,
        "axes.titlesize":      8.5,
        "axes.linewidth":      0.8,
        "xtick.labelsize":     7,
        "ytick.labelsize":     7,
        "xtick.major.width":   0.6,
        "ytick.major.width":   0.6,
        "legend.fontsize":     6.5,
        "legend.framealpha":   0.88,
        "legend.edgecolor":    "0.75",
        "lines.linewidth":     1.0,
        "lines.markersize":    4.5,
        "patch.linewidth":     0.5,
        "grid.linewidth":      0.4,
        "grid.alpha":          0.5,
        "figure.dpi":          300,
        "savefig.dpi":         300,
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.02,
        "pdf.fonttype":        42,    # Type-1 compatible for ACM
        "ps.fonttype":         42,
    })

_ccs_style()


def _colour(name: str) -> str:
    return PALETTE.get(name, "#7f7f7f")


# ══════════════════════════════════════════════════════════════════════════════
#  ShadowPatchEvaluator
# ══════════════════════════════════════════════════════════════════════════════

class ShadowPatchEvaluator:

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, "tables"),  exist_ok=True)
        os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)


    # ══════════════════════════════════════════════════════════════════════════
    #  Tables
    # ══════════════════════════════════════════════════════════════════════════

    # ── Table 1: clean performance ─────────────────────────────────────────────
    def eval_clean_performance(self,
                                detectors:   Dict,
                                test_codes:  List[str],
                                test_labels: List[int]) -> pd.DataFrame:
        rows = []
        for name, det in detectors.items():
            probs = det.predict_batch(test_codes)
            preds = [1 if p >= 0.5 else 0 for p in probs]
            rows.append({
                "Model":    name,
                "Acc (%)":  round(accuracy_score(test_labels, preds) * 100, 1),
                "Prec (%)": round(precision_score(test_labels, preds,
                                                   zero_division=0) * 100, 1),
                "Rec (%)":  round(recall_score(test_labels, preds,
                                               zero_division=0) * 100, 1),
                "F1 (%)":   round(f1_score(test_labels, preds,
                                           zero_division=0) * 100, 1),
                "AUC":      round(roc_auc_score(test_labels, probs)
                                  if len(set(test_labels)) > 1 else 0.5, 3),
            })
        df = pd.DataFrame(rows)
        self._save_table(df, "table1_clean_performance.csv")
        return df

    # ── Table 2: attack success per split × model ──────────────────────────────
    def eval_attack_per_split(self,
                               results_flat: Dict[str, Dict[str, List[AttackResult]]],
                               splits:       List[str],
                               models:       List[str],
                               tag:          str = "") -> pd.DataFrame:
        """
        Returns a wide table:  rows = models,  columns = splits.
        Each cell = "ASR% (AvgQ)".
        """
        rows = []
        for mname in models:
            row: dict = {"Model": mname}
            for sname in splits:
                res = results_flat.get(sname, {}).get(mname, [])
                if res:
                    s = summarise_results(res)
                    row[SPLIT_LABELS.get(sname, sname)] = (
                        f"{s['asr']*100:.1f}%  ({s['avg_queries']:.0f}q)")
                else:
                    row[SPLIT_LABELS.get(sname, sname)] = "—"
            rows.append(row)
        df = pd.DataFrame(rows)
        fname = f"table2{'_' + tag if tag else ''}_attack_per_split.csv"
        self._save_table(df, fname)
        return df

    # ── Table 3: transferability ───────────────────────────────────────────────
    def eval_transferability(self,
                              detectors: Dict,
                              results_by_source: Dict[str, List[AttackResult]]
                              ) -> pd.DataFrame:
        model_names = list(detectors.keys())
        matrix = {src: {} for src in model_names}
        for src, results in results_by_source.items():
            adv_codes = [r.adversarial_code for r in results if r.success]
            for tgt, tgt_det in detectors.items():
                if not adv_codes:
                    matrix[src][tgt] = 0.0
                    continue
                probs   = tgt_det.predict_batch(adv_codes)
                evasion = sum(1 for p in probs if p < 0.5)
                matrix[src][tgt] = round(evasion / len(adv_codes) * 100, 1)
        df = pd.DataFrame(matrix).T.reindex(model_names)[model_names]
        for name in model_names:
            r = results_by_source.get(name)
            if r:
                df.loc[name, name] = round(
                    summarise_results(r)["asr"] * 100, 1)
        self._save_table(
            df.reset_index().rename(columns={"index": "Source \\ Target"}),
            "table3_transferability.csv")
        return df

    # ── Table 4: CWE-specific ──────────────────────────────────────────────────
    def eval_cwe_specific(self, all_results: List[AttackResult]) -> pd.DataFrame:
        cwe_data: Dict[str, Dict] = defaultdict(
            lambda: {"n": 0, "success": 0, "exploit_ok": 0, "smt_ok": 0})
        for r in all_results:
            cwe = r.cwe if r.cwe else "Unknown"
            cwe_data[cwe]["n"] += 1
            if r.success:
                cwe_data[cwe]["success"] += 1
            if r.exploit_result and r.exploit_result.exploit_preserved:
                cwe_data[cwe]["exploit_ok"] += 1
            if r.smt_result and r.smt_result.verified:
                cwe_data[cwe]["smt_ok"] += 1
        rows = []
        for cwe, d in sorted(cwe_data.items()):
            n = max(d["n"], 1)
            rows.append({
                "CWE":         cwe,
                "N":           d["n"],
                "ASR (%)":     round(d["success"] / n * 100, 1),
                "SMT-OK (%)":  round(d["smt_ok"]  / max(d["success"], 1) * 100, 1),
                "Expl-OK (%)": round(d["exploit_ok"] /
                                     max(d["success"], 1) * 100, 1),
            })
        df = pd.DataFrame(rows)
        self._save_table(df, "table4_cwe_specific.csv")
        return df

    # ── Table 5: defense ──────────────────────────────────────────────────────
    def eval_defenses(self,
                       defenses:     Dict,
                       adv_codes:    List[str],
                       true_labels:  List[int],
                       baseline_asr: float = 0.0) -> pd.DataFrame:
        rows = []
        for dname, defense in defenses.items():
            metrics = defense.eval_defense(adv_codes, true_labels)
            rows.append({
                "Defense":       dname,
                "ASR After (%)": round(baseline_asr *
                                        (1 - metrics["detection_rate"]) * 100, 1),
                "ASR Drop (pp)": round(metrics["detection_rate"] *
                                        baseline_asr * 100, 1),
                "Det. Rate (%)": round(metrics["detection_rate"] * 100, 1),
                "FP Rate (%)":   round(metrics["fp_rate"]         * 100, 1),
                "F1 (%)":        round(metrics["f1"]              * 100, 1),
            })
        df = pd.DataFrame(rows)
        self._save_table(df, "table5_defenses.csv")
        return df

    # ── Table 6: ablation ─────────────────────────────────────────────────────
    def eval_ablation(self, detector, attack_samples,
                       query_budget: int = 500) -> pd.DataFrame:
        all_ids = list(TRANSFORM_IDS)
        configs = {
            "Full ShadowPatch":    all_ids,
            "w/o Loop (T1–T4)":   [t for t in all_ids
                                    if t not in ["T1","T2","T3","T4"]],
            "w/o Pointer (T5–T8)":[t for t in all_ids
                                    if t not in ["T5","T6","T7","T8"]],
            "w/o CF (T9–T13)":    [t for t in all_ids
                                    if t not in ["T9","T10","T11","T12","T13"]],
            "w/o Dead (T14–T18)": [t for t in all_ids
                                    if t not in ["T14","T15","T16","T17","T18"]],
            "w/o GA (random)":    all_ids,
            "w/o PDG taint":      all_ids,
        }
        base_asr = None
        rows     = []
        for cname, tid_list in configs.items():
            use_random = "random" in cname.lower()
            use_taint  = "taint"  not in cname.lower()
            attacker = GeneticAttacker(
                predict_fn    = detector.predict,
                query_budget  = query_budget,
                smt_verify    = False,
                exploit_verify= False,
            )
            attacker._transform_ids_override = tid_list
            attacker._use_random  = use_random
            attacker._skip_taint  = not use_taint
            results  = [attacker.attack(s.code, cwe=s.cwe)
                        for s in attack_samples[:100]]
            s   = summarise_results(results)
            asr = round(s["asr"] * 100, 1)
            if base_asr is None:
                base_asr = asr
            rows.append({"Configuration": cname, "ASR (%)": asr,
                         "Delta (pp)": round(asr - base_asr, 1)})
        df = pd.DataFrame(rows)
        self._save_table(df, "table6_ablation.csv")
        return df

    # ── Table 7: comparison frameworks ────────────────────────────────────────
    def eval_comparison(self,
                         comp_results:  Dict[str, List[dict]],
                         sp_asr:        float,
                         sp_avg_q:      float,
                         primary_model: str) -> pd.DataFrame:
        rows = [{
            "System":      f"ShadowPatch-GA ({primary_model})",
            "ASR (%)":     round(sp_asr * 100, 1),
            "Avg Queries": round(sp_avg_q, 0),
            "Method":      "Genetic (PDG-guided)",
        }]
        for fw_name, res in comp_results.items():
            rows.append({
                "System":      fw_name,
                "ASR (%)":     round(sum(1 for r in res if r["success"]) /
                                      max(len(res), 1) * 100, 1),
                "Avg Queries": round(sum(r["queries"] for r in res) /
                                      max(len(res), 1), 0),
                "Method":      {
                    "ALERT":       "Greedy rename (naturalness)",
                    "MHM":         "MCMC rename",
                    "Dead+Rename": "T13+T14 (single pass)",
                    "Random":      "Random transforms",
                }.get(fw_name, fw_name),
            })
        df = pd.DataFrame(rows)
        self._save_table(df, "table7_comparison.csv")
        return df


    # ══════════════════════════════════════════════════════════════════════════
    #  Original figures  (Figs 1–5, unchanged)
    # ══════════════════════════════════════════════════════════════════════════

    # ── Fig 1: ASR vs Query Budget ────────────────────────────────────────────
    def plot_asr_vs_budget(self,
                            results_by_model: Dict[str, List[AttackResult]],
                            checkpoints: Optional[List[int]] = None):
        if checkpoints is None:
            checkpoints = [50, 100, 150, 200, 300, 400, 500]

        fig, ax = plt.subplots(figsize=(FIG_DOUBLE * 0.52, FIG_HEIGHT))

        for mname, results in results_by_model.items():
            asrs = [
                sum(1 for r in results if r.success and r.queries_used <= b)
                / max(len(results), 1) * 100
                for b in checkpoints
            ]
            ax.plot(checkpoints, asrs, marker="o", label=mname,
                    color=_colour(mname), linewidth=1.0, markersize=4)

        ax.set_xlabel("Query Budget")
        ax.set_ylabel("ASR (%)")
        ax.set_title("(a) ASR vs. Query Budget", pad=3)
        ax.set_xlim(0, max(checkpoints) + 25)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle="--")
        ax.legend(loc="lower right")
        plt.tight_layout(pad=0.4)
        self._save(fig, "fig1_asr_vs_budget")
        return fig

    # ── Fig 2: Transform Heatmap ──────────────────────────────────────────────
    def plot_transform_heatmap(self,
                                results_by_model: Dict[str, List[AttackResult]]):
        tids        = list(TRANSFORM_IDS)
        model_names = list(results_by_model.keys())
        matrix      = np.zeros((len(model_names), len(tids)))

        for mi, (mname, results) in enumerate(results_by_model.items()):
            ctr   = Counter()
            total = 0
            for r in results:
                if r.success:
                    ctr.update(r.transforms_applied)
                    total += len(r.transforms_applied)
            total = max(total, 1)
            for ti, tid in enumerate(tids):
                matrix[mi, ti] = ctr.get(tid, 0) / total

        fig, ax = plt.subplots(figsize=(FIG_DOUBLE, FIG_HEIGHT + 0.3))
        sns.heatmap(matrix, ax=ax,
                    xticklabels=tids, yticklabels=model_names,
                    cmap="YlOrRd", annot=True, fmt=".2f",
                    linewidths=0.4,
                    annot_kws={"size": 6},
                    cbar_kws={"label": "Usage Rate", "shrink": 0.85})
        ax.set_title("(b) Transform Usage in Successful Attacks", pad=3)
        ax.set_xlabel("Transform ID")
        ax.set_ylabel("Target Model")
        plt.tight_layout(pad=0.4)
        self._save(fig, "fig2_transform_heatmap")
        return fig

    # ── Fig 3: Probability Shift Distribution ─────────────────────────────────
    def plot_prob_shift(self, all_results: List[AttackResult],
                         model_name: str = ""):
        shifts    = [r.prob_shift for r in all_results]
        success   = [r.success   for r in all_results]
        s_shifts  = [s for s, ok in zip(shifts, success) if ok]
        f_shifts  = [s for s, ok in zip(shifts, success) if not ok]

        fig, ax = plt.subplots(figsize=(FIG_SINGLE, FIG_HEIGHT))
        bins = np.linspace(min(shifts) - 0.01, max(shifts) + 0.01, 30)
        ax.hist(f_shifts, bins=bins, color="#aec7e8", edgecolor="white",
                alpha=0.85, label="Failed")
        ax.hist(s_shifts, bins=bins, color=_colour("linevul"),
                edgecolor="white", alpha=0.85, label="Success")
        ax.axvline(np.mean(shifts), color="#d62728", linewidth=0.9,
                   linestyle="--",
                   label=f"μ={np.mean(shifts):.3f}")
        ax.set_xlabel("Prob. Shift  Δf = f(orig) − f(adv)")
        ax.set_ylabel("Count")
        ax.set_title(f"(c) Prob. Shift Distribution"
                     + (f" ({model_name})" if model_name else ""), pad=3)
        ax.legend()
        ax.grid(True, linestyle="--")
        plt.tight_layout(pad=0.4)
        fname = f"fig3_prob_shift{'_' + model_name if model_name else ''}"
        self._save(fig, fname)
        return fig

    # ── Fig 4: CWE-specific ASR ───────────────────────────────────────────────
    def plot_cwe_asr(self, cwe_df: pd.DataFrame):
        df = cwe_df[cwe_df["CWE"].isin(CWE_ORDER)].copy()
        df["CWE"] = pd.Categorical(df["CWE"],
                                   categories=CWE_ORDER, ordered=True)
        df = df.sort_values("CWE")

        fig, ax = plt.subplots(figsize=(FIG_DOUBLE * 0.60, FIG_HEIGHT))
        colors = [_colour(c) for c in
                  ["linevul", "reveal", "ivdetect", "graphcodebert",
                   "bigvul", "devign", "sard", "combined"]][:len(df)]
        bars = ax.bar(df["CWE"], df["ASR (%)"],
                      color=colors, edgecolor="0.5",
                      linewidth=0.5, alpha=0.88)
        for bar, val in zip(bars, df["ASR (%)"]):
            if val > 3:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.8, f"{val:.1f}",
                        ha="center", va="bottom", fontsize=6.5)
        ax.set_xlabel("CWE Category")
        ax.set_ylabel("ASR (%)")
        ax.set_title("(d) Per-CWE Attack Success Rate", pad=3)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linestyle="--")
        plt.tight_layout(pad=0.4)
        self._save(fig, "fig4_cwe_asr")
        return fig

    # ── Fig 5: Ablation ───────────────────────────────────────────────────────
    def plot_ablation(self, ablation_df: pd.DataFrame):
        df      = ablation_df.copy()
        colors  = ["#1f77b4" if "Full" in c else "#d62728"
                   for c in df["Configuration"]]

        fig, ax = plt.subplots(figsize=(FIG_DOUBLE * 0.60, FIG_HEIGHT))
        bars    = ax.barh(df["Configuration"], df["ASR (%)"],
                          color=colors, edgecolor="0.5",
                          linewidth=0.5, alpha=0.88)
        for bar, val in zip(bars, df["ASR (%)"]):
            ax.text(bar.get_width() + 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=6.5)
        ax.set_xlabel("ASR (%)")
        ax.set_title("(e) Ablation: Transform Category Contributions", pad=3)
        ax.set_xlim(0, 105)
        ax.grid(axis="x", linestyle="--")
        plt.tight_layout(pad=0.4)
        self._save(fig, "fig5_ablation")
        return fig


    # ══════════════════════════════════════════════════════════════════════════
    #  New figures  (Figs 6–9)
    # ══════════════════════════════════════════════════════════════════════════

    # ── Fig 6: Per-dataset-split ASR comparison ───────────────────────────────
    def plot_per_split_asr(self,
                            results_flat: Dict[str, Dict[str, List[AttackResult]]],
                            splits:       List[str],
                            models:       List[str]):
        """
        Grouped bar chart: x-axis = model, groups = split.
        Shows how ASR varies across BigVul / Devign / SARD / Combined
        for each target model side by side.
        """
        n_splits = len(splits)
        n_models = len(models)
        x        = np.arange(n_models)
        width    = 0.72 / max(n_splits, 1)

        fig, ax = plt.subplots(figsize=(FIG_DOUBLE, FIG_HEIGHT + 0.2))

        for si, sname in enumerate(splits):
            asrs = []
            for mname in models:
                res = results_flat.get(sname, {}).get(mname, [])
                asrs.append(
                    summarise_results(res)["asr"] * 100 if res else 0.0)
            offset = (si - n_splits / 2 + 0.5) * width
            bars = ax.bar(x + offset, asrs, width=width * 0.88,
                          label=SPLIT_LABELS.get(sname, sname),
                          color=_colour(sname),
                          edgecolor="0.5", linewidth=0.4, alpha=0.88)
            for bar, val in zip(bars, asrs):
                if val > 3:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.6, f"{val:.1f}",
                            ha="center", va="bottom", fontsize=5.5)

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=7)
        ax.set_ylabel("ASR (%)")
        ax.set_ylim(0, 105)
        ax.set_title("(f) ASR per Dataset Split", pad=3)
        ax.legend(title="Split", title_fontsize=6.5, ncol=min(n_splits, 4))
        ax.grid(axis="y", linestyle="--")
        plt.tight_layout(pad=0.4)
        self._save(fig, "fig6_per_split_asr")
        return fig

    # ── Fig 7: Flat GA vs CVSS-weighted GA ────────────────────────────────────
    def plot_flat_vs_cvss(self,
                           results_flat: Dict[str, Dict[str, List[AttackResult]]],
                           results_cvss: Dict[str, Dict[str, List[AttackResult]]],
                           splits:       List[str],
                           models:       List[str],
                           cvss_lambda:  float = 0.3):
        """
        Two-panel figure:
          Left:  grouped bar — flat vs CVSS ASR per model  (combined split)
          Right: line — ASR improvement (CVSS − flat) per split
        """
        best_split = "combined" if "combined" in splits else splits[0]

        flat_asrs = [
            summarise_results(results_flat.get(best_split, {}).get(m, []))["asr"] * 100
            if results_flat.get(best_split, {}).get(m) else 0.0
            for m in models
        ]
        cvss_asrs = [
            summarise_results(results_cvss.get(best_split, {}).get(m, []))["asr"] * 100
            if results_cvss.get(best_split, {}).get(m) else 0.0
            for m in models
        ]

        x     = np.arange(len(models))
        width = 0.33

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_DOUBLE, FIG_HEIGHT))

        # Panel 1: grouped bars
        b1 = ax1.bar(x - width / 2, flat_asrs, width=width,
                     label="Flat GA",  color="#1f77b4",
                     edgecolor="0.5",  linewidth=0.5, alpha=0.88)
        b2 = ax1.bar(x + width / 2, cvss_asrs, width=width,
                     label=f"CVSS-GA (λ={cvss_lambda})",
                     color="#d62728",  edgecolor="0.5",
                     linewidth=0.5,    alpha=0.88)
        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                if h > 3:
                    ax1.text(bar.get_x() + bar.get_width() / 2,
                             h + 0.6, f"{h:.1f}",
                             ha="center", va="bottom", fontsize=6)
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, fontsize=7)
        ax1.set_ylabel("ASR (%)")
        ax1.set_ylim(0, 105)
        ax1.set_title(f"(g) Flat vs CVSS-GA  [{best_split}]", pad=3)
        ax1.legend()
        ax1.grid(axis="y", linestyle="--")

        # Panel 2: improvement per split as lines
        for mname in models:
            improvements = []
            for sname in splits:
                f_res = results_flat.get(sname, {}).get(mname, [])
                c_res = results_cvss.get(sname, {}).get(mname, [])
                f_asr = summarise_results(f_res)["asr"] * 100 if f_res else 0.0
                c_asr = summarise_results(c_res)["asr"] * 100 if c_res else 0.0
                improvements.append(c_asr - f_asr)
            ax2.plot([SPLIT_LABELS.get(s, s) for s in splits],
                     improvements, marker="o", label=mname,
                     color=_colour(mname), linewidth=1.0)

        ax2.axhline(0, color="0.5", linewidth=0.7, linestyle="--")
        ax2.set_ylabel("ΔASR (pp)  CVSS − Flat")
        ax2.set_title("(h) CVSS Gain per Split", pad=3)
        ax2.legend(fontsize=6)
        ax2.grid(True, linestyle="--")

        plt.tight_layout(pad=0.4)
        self._save(fig, "fig7_flat_vs_cvss")
        return fig

    # ── Fig 8: ShadowPatch vs comparison frameworks ───────────────────────────
    def plot_comparison(self,
                         comp_results:  Dict[str, List[dict]],
                         sp_asr:        float,
                         primary_model: str):
        """
        Horizontal bar chart comparing ShadowPatch-GA against ALERT, MHM,
        Dead+Rename, and Random, ordered by descending ASR.
        """
        labels = [f"ShadowPatch-GA\n({primary_model})"] + list(comp_results.keys())
        asrs   = [sp_asr * 100] + [
            sum(1 for r in res if r["success"]) / max(len(res), 1) * 100
            for res in comp_results.values()
        ]
        avg_qs = [0] + [
            sum(r["queries"] for r in res) / max(len(res), 1)
            for res in comp_results.values()
        ]

        # sort descending by ASR
        order  = sorted(range(len(asrs)), key=lambda i: asrs[i], reverse=True)
        labels = [labels[i] for i in order]
        asrs   = [asrs[i]   for i in order]
        avg_qs = [avg_qs[i] for i in order]

        raw_names = ["ShadowPatch-GA"] + list(comp_results.keys())
        raw_order  = [raw_names[i] for i in order]
        colors = [_colour(n.split("\n")[0]) for n in raw_order]

        fig, ax = plt.subplots(figsize=(FIG_DOUBLE * 0.55, FIG_HEIGHT))
        bars = ax.barh(labels[::-1], asrs[::-1],
                       color=colors[::-1],
                       edgecolor="0.5", linewidth=0.5, alpha=0.88)
        for bar, val in zip(bars, asrs[::-1]):
            ax.text(bar.get_width() + 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=6.5)
        ax.set_xlabel("ASR (%)")
        ax.set_xlim(0, 110)
        ax.set_title(f"(i) ShadowPatch vs. Comparison Frameworks", pad=3)
        ax.grid(axis="x", linestyle="--")
        plt.tight_layout(pad=0.4)
        self._save(fig, "fig8_comparison")
        return fig

    # ── Fig 9: CVSS severity-bucketed ASR ────────────────────────────────────
    def plot_cvss_weighted_asr(self,
                                results:    List[AttackResult],
                                output_dir: str = ""):
        """
        Two-panel figure:
          Left:  scatter  CVSS score vs prob-shift, coloured by success
          Right: bar      ASR per CVSS severity tier
                          (Low 0–3.9 / Medium 4–6.9 / High 7–8.9 / Crit 9–10)
        """
        tagged = [r for r in results if getattr(r, "meta", {}).get("cvss")]
        if len(tagged) < 5:
            logger.warning("plot_cvss_weighted_asr: < 5 tagged results — skipped.")
            return None

        cvss_sc   = [r.meta["cvss"]                        for r in tagged]
        shifts    = [r.prob_shift                           for r in tagged]
        successes = [r.success                              for r in tagged]
        sources   = [r.meta.get("cvss_source", "unknown")  for r in tagged]

        tier_edges  = [0, 4.0, 7.0, 9.0, 10.1]
        tier_labels = ["Low\n(0–3.9)", "Medium\n(4–6.9)",
                       "High\n(7–8.9)", "Critical\n(9–10)"]
        tier_colors = ["#4575b4", "#74add1", "#f46d43", "#d73027"]
        tier_bg     = ["#f0f0f0", "#fffde7", "#fff3e0", "#fce4ec"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_DOUBLE, FIG_HEIGHT))

        # Panel 1: scatter
        src_markers = {"nvd_direct": "o", "cwe_mean": "s",
                       "global_median": "^", "flat": "D"}
        for src, mk in src_markers.items():
            idx = [i for i, s in enumerate(sources) if s == src]
            if not idx:
                continue
            sc = [cvss_sc[i]  for i in idx]
            sh = [shifts[i]   for i in idx]
            cl = ["#d62728" if successes[i] else "#aec7e8" for i in idx]
            ax1.scatter(sc, sh, c=cl, marker=mk, s=18, alpha=0.7,
                        linewidths=0.3, edgecolors="0.4",
                        label=src.replace("_", " "))

        ax1.axhline(0, color="0.6", linewidth=0.6, linestyle="--")
        for i, (lo, hi) in enumerate(zip(tier_edges, tier_edges[1:])):
            ax1.axvspan(lo, hi, alpha=0.12, color=tier_bg[i], zorder=0)
        ax1.set_xlabel("CVSS Base Score")
        ax1.set_ylabel("Prob. Shift")
        ax1.set_xlim(0, 10.5)
        ax1.set_title("(j) CVSS Score vs. Prob. Shift", pad=3)
        ax1.legend(fontsize=5.5, loc="upper left")

        # Panel 2: ASR by tier
        tier_asrs, tier_ns = [], []
        for lo, hi in zip(tier_edges, tier_edges[1:]):
            bucket = [r for r, c in zip(tagged, cvss_sc) if lo <= c < hi]
            n_suc  = sum(1 for r in bucket if r.success)
            tier_asrs.append(100 * n_suc / max(len(bucket), 1))
            tier_ns.append(len(bucket))

        bars = ax2.bar(tier_labels, tier_asrs,
                       color=tier_colors, edgecolor="0.4",
                       linewidth=0.5, alpha=0.88)
        for bar, n, asr in zip(bars, tier_ns, tier_asrs):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     asr + 1.2, f"{asr:.1f}%\n(n={n})",
                     ha="center", va="bottom", fontsize=6)
        ax2.set_ylabel("ASR (%)")
        ax2.set_ylim(0, max(tier_asrs or [0]) * 1.35 + 5)
        ax2.set_title("(k) ASR by CVSS Severity Tier", pad=3)
        ax2.grid(axis="y", linestyle="--")

        plt.tight_layout(pad=0.4)
        out = output_dir or self.output_dir
        fig.savefig(os.path.join(out, "figures", "fig9_cvss_asr.pdf"),
                    bbox_inches="tight")
        fig.savefig(os.path.join(out, "figures", "fig9_cvss_asr.png"),
                    bbox_inches="tight", dpi=300)
        plt.close(fig)
        logger.info("Figure saved: fig9_cvss_asr.pdf / .png")
        return fig


    # ══════════════════════════════════════════════════════════════════════════
    #  Save helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _save_table(self, df: pd.DataFrame, filename: str):
        path = os.path.join(self.output_dir, "tables", filename)
        df.to_csv(path, index=False)
        logger.info("Table  → %s", path)

    def _save(self, fig, stem: str):
        """Save figure as both PDF and PNG."""
        for ext in ("pdf", "png"):
            path = os.path.join(self.output_dir, "figures",
                                f"{stem}.{ext}")
            fig.savefig(path, bbox_inches="tight",
                        dpi=300 if ext == "png" else None)
            logger.info("Figure → %s", path)
        plt.close(fig)

    # Legacy alias used by run_experiment.py
    def _save_figure(self, fig, filename: str):
        path = os.path.join(self.output_dir, "figures", filename)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        logger.info("Figure → %s", path)

    # ── Raw JSON dump ─────────────────────────────────────────────────────────
    def save_raw_results(self, results: List[AttackResult],
                          model_name: str = ""):
        fname = f"raw_results{'_' + model_name if model_name else ''}.json"
        path  = os.path.join(self.output_dir, fname)
        data  = []
        for r in results:
            data.append({
                "success":            r.success,
                "original_prob":      r.original_prob,
                "adversarial_prob":   r.adversarial_prob,
                "prob_shift":         r.prob_shift,
                "queries_used":       r.queries_used,
                "transforms":         r.transforms_applied,
                "smt_verified":       r.smt_result.verified if r.smt_result else None,
                "smt_status":         r.smt_result.status   if r.smt_result else None,
                "exploit_preserved":  (r.exploit_result.exploit_preserved
                                       if r.exploit_result else None),
                "cwe":                r.cwe,
                "source":             r.source,
                "time_sec":           r.time_sec,
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Raw results → %s", path)
