"""
pdg_taint.py
============
Phase 1 of ShadowPatch: PDG + taint analysis.

Extracts:
  - taint_set T(v): AST/token positions reachable from vulnerable statements
  - safe_region S(v): positions NOT in the 2-hop neighbourhood of T(v)

Used to constrain transforms so they are never applied to tainted nodes,
guaranteeing vulnerability preservation.

Uses tree-sitter AST parsing with regex fallback.
"""

from __future__ import annotations
import re, logging
from typing import Set, List, Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Vulnerability sinks (CWE-specific) ───────────────────────────────────────
VULN_SINKS = {
    # Buffer / memory operations
    "strcpy", "strncpy", "strcat", "strncat", "gets", "scanf",
    "sprintf", "vsprintf", "memcpy", "memmove", "memset",
    # Heap operations
    "malloc", "calloc", "realloc", "free",
    # File / format
    "printf", "fprintf", "vprintf", "fgets", "fread", "fwrite",
    # Integer / type
    "atoi", "atol", "strtol", "strtoul",
}

CONTROL_KEYWORDS = {"if", "else", "for", "while", "do", "switch", "case"}

# ── line-level node ───────────────────────────────────────────────────────────

@dataclass
class PDGNode:
    line_no:   int
    text:      str
    node_type: str   # stmt | branch | sink | decl
    cf_succs:  List[int] = field(default_factory=list)   # control-flow edges
    df_succs:  List[int] = field(default_factory=list)   # data-flow edges


@dataclass
class PDGGraph:
    nodes: Dict[int, PDGNode] = field(default_factory=dict)

    def add_node(self, n: PDGNode):
        self.nodes[n.line_no] = n

    def taint_from(self, seed_lines: Set[int],
                   hops: int = 2) -> Set[int]:
        """BFS on CF+DF edges from seed_lines, up to `hops` steps."""
        visited, frontier = set(seed_lines), set(seed_lines)
        for _ in range(hops):
            next_f = set()
            for ln in frontier:
                node = self.nodes.get(ln)
                if node is None:
                    continue
                for nb in node.cf_succs + node.df_succs:
                    if nb not in visited:
                        visited.add(nb)
                        next_f.add(nb)
            frontier = next_f
        return visited

    def safe_region(self, taint: Set[int]) -> Set[int]:
        return {ln for ln in self.nodes if ln not in taint}


# ══════════════════════════════════════════════════════════════════════════════
# Builder
# ══════════════════════════════════════════════════════════════════════════════

class PDGBuilder:
    """
    Builds a simplified PDG from C/C++ source text using:
      1. tree-sitter (if available) for proper AST edges
      2. regex-based fallback for line classification + variable def/use chains
    """

    def __init__(self):
        self._ts_available = False
        try:
            from tree_sitter import Language, Parser
            from tree_sitter_languages import get_language
            self._parser   = Parser()
            self._language = get_language("c")
            self._parser.set_language(self._language)
            self._ts_available = True
        except Exception:
            logger.debug("tree-sitter unavailable; using regex PDG builder.")

    # ── public entry point ────────────────────────────────────────────────────
    def build(self, code: str) -> PDGGraph:
        if self._ts_available:
            try:
                return self._build_ts(code)
            except Exception as e:
                logger.debug("tree-sitter PDG failed (%s); falling back.", e)
        return self._build_regex(code)

    # ── tree-sitter path ──────────────────────────────────────────────────────
    def _build_ts(self, code: str) -> PDGGraph:
        tree  = self._parser.parse(bytes(code, "utf-8"))
        lines = code.splitlines()
        graph = PDGGraph()

        def row(node): return node.start_point[0]

        def walk(node, depth=0):
            ln   = row(node)
            text = lines[ln].strip() if ln < len(lines) else ""
            ntype = self._classify_text(text)
            if ln not in graph.nodes:
                graph.add_node(PDGNode(line_no=ln, text=text,
                                       node_type=ntype))
            for child in node.children:
                child_ln = row(child)
                n = graph.nodes.get(ln)
                if n and child_ln != ln and child_ln not in n.cf_succs:
                    n.cf_succs.append(child_ln)
                walk(child, depth + 1)

        walk(tree.root_node)
        self._add_df_edges(code, graph)
        return graph

    # ── regex fallback path ───────────────────────────────────────────────────
    def _build_regex(self, code: str) -> PDGGraph:
        lines = code.splitlines()
        graph = PDGGraph()
        var_defs: Dict[str, List[int]] = {}   # var_name -> defining lines

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            ntype = self._classify_text(stripped)
            node  = PDGNode(line_no=i, text=stripped, node_type=ntype)

            # CF edge: next line
            if i + 1 < len(lines):
                node.cf_succs.append(i + 1)

            # Record variable definitions
            m = re.match(r'\b(?:int|char|size_t|unsigned|long|void\s*\*|'
                         r'uint\w*)\s+\*?\s*(\w+)\s*[=;]', stripped)
            if m:
                v = m.group(1)
                var_defs.setdefault(v, []).append(i)

            graph.add_node(node)

        # DF edges: use-def chains
        for i, line in enumerate(lines):
            if i not in graph.nodes:
                continue
            for var, def_lines in var_defs.items():
                if re.search(r'\b' + re.escape(var) + r'\b', line):
                    for dl in def_lines:
                        if dl != i and dl in graph.nodes:
                            graph.nodes[i].df_succs.append(dl)

        return graph

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _classify_text(text: str) -> str:
        t = text.lower()
        if any(s + "(" in t or s + " " in t for s in VULN_SINKS):
            return "sink"
        if any(kw in t.split() for kw in CONTROL_KEYWORDS):
            return "branch"
        if re.match(r'\b(?:int|char|size_t|unsigned|long|void)\b', t):
            return "decl"
        return "stmt"

    @staticmethod
    def _add_df_edges(code: str, graph: PDGGraph):
        """Add variable def-use data-flow edges."""
        lines = code.splitlines()
        var_defs: Dict[str, List[int]] = {}
        for i, line in enumerate(lines):
            m = re.match(r'\s*(?:int|char|size_t|unsigned|long|void\s*\*|'
                         r'uint\w*)\s+\*?\s*(\w+)\s*[=;]', line)
            if m:
                var_defs.setdefault(m.group(1), []).append(i)
        for i, line in enumerate(lines):
            if i not in graph.nodes:
                continue
            for var, def_lines in var_defs.items():
                if re.search(r'\b' + re.escape(var) + r'\b', line):
                    for dl in def_lines:
                        if dl != i and dl in graph.nodes:
                            graph.nodes[i].df_succs.append(dl)


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

_builder = PDGBuilder()


def get_taint_set(code: str, hops: int = 2) -> Tuple[Set[int], Set[int]]:
    """
    Returns (taint_set, safe_set) as sets of line numbers.

    taint_set: lines reachable from vulnerability sinks within `hops` edges.
    safe_set:  all other lines — safe to transform.
    """
    graph     = _builder.build(code)
    seed_lines = {ln for ln, n in graph.nodes.items()
                  if n.node_type == "sink"}
    if not seed_lines:
        # No sinks found: treat entire function as safe
        # (transforms will be applied everywhere, worst-case)
        all_lines = set(graph.nodes.keys())
        return set(), all_lines

    taint = graph.taint_from(seed_lines, hops=hops)
    safe  = graph.safe_region(taint)
    return taint, safe


def is_safe_line(line_no: int, safe_set: Set[int]) -> bool:
    return line_no in safe_set
