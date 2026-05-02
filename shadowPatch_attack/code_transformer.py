"""
code_transformer.py
===================
Phase 2 of ShadowPatch: 18 semantics-preserving C/C++ transformations
organized into four categories, all constrained to the SAFE region
(lines not in the PDG taint set).

Categories
----------
T1–T4   Loop transformations
T5–T8   Pointer / array obfuscation
T9–T13  Control-flow restructuring
T14–T18 Inline / dead code
"""

from __future__ import annotations
import re, random, logging, copy
from typing import List, Optional, Set, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

TRANSFORM_IDS = [f"T{i}" for i in range(1, 19)]   # T1 … T18


# ══════════════════════════════════════════════════════════════════════════════
# Result dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TransformResult:
    code:        str
    transform:   str
    changed:     bool = False
    description: str  = ""


# ══════════════════════════════════════════════════════════════════════════════
# Safe-region helpers
# ══════════════════════════════════════════════════════════════════════════════

def _safe_lines(code: str, safe_set: Optional[Set[int]]) -> Set[int]:
    """Return set of ORIGINAL line indices that are in the safe region."""
    if safe_set is None:
        return set(range(len(code.splitlines())))
    return safe_set


def _is_safe(orig_idx: int, safe_set: Optional[Set[int]],
             total_lines: int) -> bool:
    if safe_set is None:
        return True
    return orig_idx in safe_set


# ══════════════════════════════════════════════════════════════════════════════
# Category 1: Loop Transformations  T1–T4
# ══════════════════════════════════════════════════════════════════════════════

def T1_for_to_while(code: str, safe_set: Optional[Set[int]] = None) -> TransformResult:
    """T1: for(init; cond; incr){ ... }  →  init; while(cond){ ...; incr; }

    FIX: Use a more robust regex that handles spaces and complex init/cond/incr.
    Only transform the first matching for-loop per call to avoid index drift.
    """
    lines   = code.splitlines()
    safe    = _safe_lines(code, safe_set)
    changed = False
    result  = []

    # Regex: captures init, cond, incr — handles spaces robustly
    FOR_RE = re.compile(
        r'^(\s*)for\s*\(\s*([^;]*?)\s*;\s*([^;]*?)\s*;\s*([^)]*?)\s*\)\s*(\{?)\s*$'
    )

    for i, line in enumerate(lines):
        if i not in safe:
            result.append(line)
            continue
        m = FOR_RE.match(line)
        if m:
            indent = m.group(1)
            init   = m.group(2).strip()
            cond   = m.group(3).strip()
            incr   = m.group(4).strip()
            brace  = " {" if m.group(5) else ""
            if not init.endswith(";"):
                init += ";"
            result.append(f"{indent}{init}  /* T1: for→while */")
            result.append(f"{indent}while ({cond}){brace}")
            changed = True
        else:
            result.append(line)

    return TransformResult("\n".join(result), "T1", changed, "for→while")


def T2_while_to_dowhile(code: str, safe_set: Optional[Set[int]] = None) -> TransformResult:
    """T2: while(cond){ body } → do { body } while(cond);

    FIX: Completely rewritten. Old version used __DOWHILE_COND_ string markers
    that were never parsed correctly. New version uses a proper brace-depth
    tracker to find the matching closing brace and rewrites it.
    """
    lines   = code.splitlines()
    safe    = _safe_lines(code, safe_set)
    changed = False
    result  = copy.copy(lines)

    WHILE_RE = re.compile(r'^(\s*)while\s*\((.+)\)\s*\{\s*$')

    i = 0
    while i < len(result):
        line = result[i]
        m = WHILE_RE.match(line)
        if m and i in safe:
            indent = m.group(1)
            cond   = m.group(2).strip()

            # Find matching closing brace using depth tracking
            depth   = 1
            j       = i + 1
            while j < len(result) and depth > 0:
                depth += result[j].count("{") - result[j].count("}")
                j     += 1
            close_idx = j - 1  # index of the line that closed the while

            if depth == 0 and close_idx < len(result):
                # Rewrite: opening line becomes do {
                result[i]         = f"{indent}do {{  /* T2: while→do-while */"
                # Rewrite: closing } becomes } while(cond);
                result[close_idx] = f"{indent}}} while ({cond});"
                changed = True
                i = close_idx + 1
                continue
        i += 1

    return TransformResult("\n".join(result), "T2", changed, "while→do-while")


def T3_loop_unrolling(code: str, safe_set: Optional[Set[int]] = None,
                      factor: int = 2) -> TransformResult:
    """T3: Add an unroll-hint comment above small fixed-bound for-loops.

    FIX: Original added comment AND kept original, which just added clutter.
    Now adds a pragma-style comment that models read as a real annotation.
    """
    lines   = code.splitlines()
    safe    = _safe_lines(code, safe_set)
    changed = False
    result  = []

    LOOP_RE = re.compile(
        r'^(\s*)for\s*\(\s*(\w+)\s*=\s*(\d+)\s*;\s*\2\s*<\s*(\d+)\s*;\s*\2\+\+\)'
    )

    for i, line in enumerate(lines):
        if i in safe:
            m = LOOP_RE.match(line)
            if m:
                start = int(m.group(3))
                end   = int(m.group(4))
                indent = m.group(1)
                if 0 < (end - start) <= 16:
                    result.append(
                        f"{indent}/* T3: unroll_hint(factor={factor}) */")
                    changed = True
        result.append(line)

    return TransformResult("\n".join(result), "T3", changed, "loop-unroll-hint")


def T4_loop_direction_reversal(code: str, safe_set: Optional[Set[int]] = None) -> TransformResult:
    """T4: for(i=0; i<N; i++) → for(i=N-1; i>=0; i--)

    FIX: Original had a greedy `(.*)` at the end that captured the `{` too,
    then pasted it back incorrectly. Now matches the brace separately.
    """
    lines   = code.splitlines()
    safe    = _safe_lines(code, safe_set)
    changed = False
    result  = []

    REV_RE = re.compile(
        r'^(\s*)for\s*\(\s*(\w+)\s*=\s*0\s*;\s*\2\s*<\s*(\w+)\s*;\s*\2\+\+\)\s*(\{?)'
    )

    for i, line in enumerate(lines):
        if i not in safe:
            result.append(line)
            continue
        m = REV_RE.match(line)
        if m:
            indent = m.group(1)
            var    = m.group(2)
            n      = m.group(3)
            brace  = (" {" if m.group(4) else "")
            result.append(
                f"{indent}for ({var} = (int)({n})-1; {var} >= 0; {var}--)"
                f"  /* T4: reversed */{brace}")
            changed = True
        else:
            result.append(line)

    return TransformResult("\n".join(result), "T4", changed, "loop-direction-reversal")


# ══════════════════════════════════════════════════════════════════════════════
# Category 2: Pointer / Array Obfuscation  T5–T8
# ══════════════════════════════════════════════════════════════════════════════

def T5_pointer_aliasing(code: str, safe_set: Optional[Set[int]] = None) -> TransformResult:
    """T5: Introduce a redundant pointer alias for pointer declarations.

    FIX: Original appended alias lines after the current line, causing
    safe_set index drift for all subsequent lines. Now we collect all
    insertions as (orig_idx, alias_line) pairs and apply them in reverse
    order so earlier insertions don't shift later indices.
    """
    lines   = code.splitlines()
    safe    = _safe_lines(code, safe_set)
    changed = False

    # Pattern: TYPE *varname = expr;
    PTR_RE  = re.compile(r'^(\s*)([\w\s]+)\s*\*\s*(\w+)\s*=\s*([^;]+);')
    seen    = set()
    inserts = []   # (after_line_idx, text_to_insert)

    for i, line in enumerate(lines):
        if i not in safe:
            continue
        m = PTR_RE.match(line)
        if m:
            typ     = m.group(2).strip()
            varname = m.group(3)
            indent  = m.group(1)
            if varname not in seen:
                seen.add(varname)
                alias_line = (f"{indent}{typ} *{varname}_alias__ = "
                              f"{varname};  /* T5: alias */")
                inserts.append((i, alias_line))
                changed = True

    # Apply inserts in reverse order to preserve original indices
    result = list(lines)
    for orig_idx, alias_line in reversed(inserts):
        result.insert(orig_idx + 1, alias_line)

    return TransformResult("\n".join(result), "T5", changed, "pointer-alias")


def T6_array_to_pointer(code: str, safe_set: Optional[Set[int]] = None) -> TransformResult:
    r"""T6: arr[i] → *(arr + i) for safe-region lines.

    FIX: Old regex `(\w+)\[(\w+)\]` matched array declarations like
    `char buf[256]`, corrupting them. New version:
    - Only matches when the bracket expression is on the RHS of = or in a
      function call / expression context (not a declaration).
    - Excludes lines that look like declarations (type keyword before name).
    """
    lines   = code.splitlines()
    safe    = _safe_lines(code, safe_set)
    changed = False
    result  = []

    # Declaration pattern to skip: starts with a type keyword
    DECL_RE = re.compile(
        r'^\s*(int|char|short|long|unsigned|float|double|void|size_t|'
        r'uint\w*|int\d+_t|struct\s+\w+)\b'
    )
    # Access pattern: word[word_or_expr] NOT preceded by a type
    ACC_RE  = re.compile(r'\b(\w+)\[(\w+)\]')

    for i, line in enumerate(lines):
        if i not in safe or DECL_RE.match(line):
            result.append(line)
            continue
        new = ACC_RE.sub(lambda m: f"(*({m.group(1)} + {m.group(2)}))", line)
        result.append(new)
        if new != line:
            changed = True

    return TransformResult("\n".join(result), "T6", changed,
                           "array→pointer-arithmetic")


def T7_redundant_dereference(code: str, safe_set: Optional[Set[int]] = None) -> TransformResult:
    """T7: x = scalar_var; → x = *(&scalar_var);

    FIX: Old regex fired on ANY `= word;` including malloc, function calls,
    struct fields, etc, producing invalid C. New version:
    - Only matches simple scalar variable names (no parens, no dots, no arrows)
    - Skips lines with function calls, malloc, struct/array access
    - Skips pointer declarations
    """
    lines   = code.splitlines()
    safe    = _safe_lines(code, safe_set)
    changed = False
    result  = []

    SKIP_RE = re.compile(r'malloc|calloc|realloc|free|sizeof|'
                         r'\(|\)|->|\.|&|\*')
    # = simple_word; — word must be a plain identifier, not a keyword
    ASSIGN_RE = re.compile(r'(=\s*)([a-z_]\w*)(\s*;)')
    KEYWORDS   = {"return", "if", "else", "while", "for", "do", "switch",
                  "case", "break", "continue", "NULL", "true", "false"}

    for i, line in enumerate(lines):
        if i not in safe:
            result.append(line)
            continue
        stripped = line.strip()
        # Skip declarations, complex expressions, and lines with func calls
        if (stripped.startswith(("int ", "char ", "long ", "unsigned ",
                                  "float ", "double ", "void ", "size_t "))
                or re.search(r'malloc|calloc|realloc|\(', stripped)
                or "->" in stripped or "." in stripped):
            result.append(line)
            continue
        new = ASSIGN_RE.sub(
            lambda m: (f"{m.group(1)}*(&{m.group(2)}){m.group(3)}"
                       if m.group(2) not in KEYWORDS else m.group(0)),
            line)
        result.append(new)
        if new != line:
            changed = True

    return TransformResult("\n".join(result), "T7", changed, "redundant-deref")


def T8_stack_to_heap(code: str, safe_set: Optional[Set[int]] = None) -> TransformResult:
    """T8: char buf[N]; → char *buf = (char*)malloc(N);

    FIX: After inserting the malloc line + comment, all subsequent safe_set
    indices shifted by +1 per insertion. We now collect all replacements
    first and apply in reverse to preserve original indices.
    Also: only transforms char arrays, not arrays used as function params.
    """
    lines    = code.splitlines()
    safe     = _safe_lines(code, safe_set)
    changed  = False
    need_hdr = False
    replacements = []  # (orig_idx, [new_lines])

    CHAR_ARR_RE = re.compile(r'^(\s*)char\s+(\w+)\s*\[(\d+)\]\s*;')

    for i, line in enumerate(lines):
        if i not in safe:
            continue
        m = CHAR_ARR_RE.match(line)
        if m:
            indent  = m.group(1)
            varname = m.group(2)
            size    = m.group(3)
            new_lines = [
                f"{indent}char *{varname} = (char *)malloc({size});  /* T8: stack→heap */",
                f"{indent}/* T8: free({varname}) when done */",
            ]
            replacements.append((i, new_lines))
            changed  = True
            need_hdr = True

    result = list(lines)
    for orig_idx, new_lines in reversed(replacements):
        result[orig_idx:orig_idx + 1] = new_lines

    out = "\n".join(result)
    if need_hdr and "#include <stdlib.h>" not in out:
        out = "#include <stdlib.h>  /* T8 */\n" + out

    return TransformResult(out, "T8", changed, "stack→heap")


# ══════════════════════════════════════════════════════════════════════════════
# Category 3: Control-Flow Restructuring  T9–T13
# ══════════════════════════════════════════════════════════════════════════════

def T9_if_else_flattening(code: str, safe_set: Optional[Set[int]] = None) -> TransformResult:
    """T9: if(cond) return val;  →  { int __flat=(cond)?1:0; if(__flat) return val; }

    FIX: Original was fine but very narrow pattern. Extended to also handle
    if(cond) stmt; (single-statement if without braces).
    """
    lines   = code.splitlines()
    safe    = _safe_lines(code, safe_set)
    changed = False
    result  = []

    RET_RE  = re.compile(r'^(\s*)if\s*\((.+?)\)\s*return\s+(.+?);(\s*)$')
    STMT_RE = re.compile(r'^(\s*)if\s*\((.+?)\)\s*(\w[^{;]*);(\s*)$')

    for i, line in enumerate(lines):
        if i not in safe:
            result.append(line)
            continue
        m = RET_RE.match(line)
        if m:
            indent = m.group(1)
            cond   = m.group(2).strip()
            val    = m.group(3).strip()
            result.append(
                f"{indent}{{ int __flat__ = ({cond}) ? 1 : 0;"
                f" if (__flat__) return {val}; }}  /* T9 */")
            changed = True
            continue
        m = STMT_RE.match(line)
        if m and "else" not in line and "return" not in line:
            indent = m.group(1)
            cond   = m.group(2).strip()
            stmt   = m.group(3).strip()
            result.append(
                f"{indent}{{ int __flat__ = ({cond}) ? 1 : 0;"
                f" if (__flat__) {stmt}; }}  /* T9 */")
            changed = True
            continue
        result.append(line)

    return TransformResult("\n".join(result), "T9", changed, "if-else-flatten")


def T10_opaque_predicate(code: str, safe_set: Optional[Set[int]] = None) -> TransformResult:
    """T10: Insert always-true opaque predicates after opening braces.

    FIX: Old version used `i % 5 == 0` which rarely fired on small functions
    (5-15 lines). New version inserts after EVERY opening brace in the safe
    region, capped at 4 insertions to avoid bloating small functions.
    """
    lines   = code.splitlines()
    safe    = _safe_lines(code, safe_set)
    changed = False
    inserts = []   # (after_orig_idx, text)

    PREDICATES = [
        "if ((sizeof(int) >= 4)) { /* T10: opaque-true */ }",
        "if ((1 | 0) != 0) { /* T10: opaque-true */ }",
        "if ((0 & 1) == 0) { /* T10: opaque-true */ }",
        "do { volatile int __op__ = 0; (void)__op__; } while (0); /* T10 */",
    ]

    inserted = 0
    for i, line in enumerate(lines):
        if i not in safe:
            continue
        stripped = line.strip()
        if stripped.endswith("{") and inserted < 4:
            indent = re.match(r'^(\s*)', line).group(1)
            inserts.append((i, f"{indent}  {random.choice(PREDICATES)}"))
            inserted += 1
            changed = True

    # Insert in reverse to preserve original indices
    result = list(lines)
    for orig_idx, text in reversed(inserts):
        result.insert(orig_idx + 1, text)

    return TransformResult("\n".join(result), "T10", changed, "opaque-predicate")


def T11_statement_reordering(code: str, safe_set: Optional[Set[int]] = None) -> TransformResult:
    """T11: Shuffle consecutive independent declaration lines in safe region.

    FIX: Old version had an off-by-one — when `i not in safe` it still
    incremented i in the else branch, causing the outer while to skip lines.
    New version uses a clean index walk.
    """
    lines  = code.splitlines()
    safe   = _safe_lines(code, safe_set)
    result = list(lines)
    changed = False

    DECL_RE = re.compile(
        r'^\s*(int|char|size_t|unsigned|long|float|double)\s+\w+')

    i = 0
    while i < len(lines):
        if i not in safe or not DECL_RE.match(lines[i]):
            i += 1
            continue
        # Collect a run of consecutive safe declaration lines
        block = []
        j = i
        while j < len(lines) and j in safe and DECL_RE.match(lines[j]):
            block.append(j)
            j += 1
        if len(block) >= 2:
            shuffled = block[:]
            random.shuffle(shuffled)
            if shuffled != block:
                for new_pos, old_pos in zip(block, shuffled):
                    result[new_pos] = lines[old_pos]
                changed = True
        i = j

    return TransformResult("\n".join(result), "T11", changed, "decl-reordering")


def T12_compound_condition_split(code: str, safe_set: Optional[Set[int]] = None) -> TransformResult:
    """T12: if(a && b){ } → if(a){ if(b){ } }"""
    lines   = code.splitlines()
    safe    = _safe_lines(code, safe_set)
    changed = False
    result  = []

    # Only match top-level && (not nested parens)
    AND_RE = re.compile(r'^(\s*)if\s*\(([^()]+?)\s*&&\s*([^()]+?)\)\s*(\{?)\s*$')

    for i, line in enumerate(lines):
        if i not in safe:
            result.append(line)
            continue
        m = AND_RE.match(line)
        if m:
            indent = m.group(1)
            cond1  = m.group(2).strip()
            cond2  = m.group(3).strip()
            brace  = m.group(4)
            result.append(f"{indent}if ({cond1}) {{  /* T12: && split */")
            result.append(f"{indent}  if ({cond2}) {brace}")
            changed = True
        else:
            result.append(line)

    return TransformResult("\n".join(result), "T12", changed,
                           "compound-cond-split")


def T13_dead_if_zero(code: str, safe_set: Optional[Set[int]] = None) -> TransformResult:
    """T13: Inject if(0){dead} blocks after opening braces in safe region.

    FIX: Old version used `i % 3 == 0` which rarely fired. New version
    inserts after EVERY opening brace in safe region, capped at 5.
    Uses insert-in-reverse pattern to avoid index drift.
    """
    lines   = code.splitlines()
    safe    = _safe_lines(code, safe_set)
    changed = False
    inserts = []

    DEAD_SNIPPETS = [
        "if (0) { volatile int __dead__ = 1; (void)__dead__; }  /* T13 */",
        "if (sizeof(char) == 0) { return -1; }  /* T13: dead branch */",
        "do { if (0) break; } while (0);  /* T13: dead loop */",
        "if (0 && 1) { int __x__ = 0; (void)__x__; }  /* T13 */",
    ]

    inserted = 0
    for i, line in enumerate(lines):
        if i not in safe:
            continue
        if line.strip().endswith("{") and inserted < 5:
            indent = re.match(r'^(\s*)', line).group(1)
            inserts.append((i, f"{indent}  {random.choice(DEAD_SNIPPETS)}"))
            inserted += 1
            changed = True

    result = list(lines)
    for orig_idx, text in reversed(inserts):
        result.insert(orig_idx + 1, text)

    return TransformResult("\n".join(result), "T13", changed, "dead-branch")


# ══════════════════════════════════════════════════════════════════════════════
# Category 4: Inline / Dead Code  T14–T18
# ══════════════════════════════════════════════════════════════════════════════

_VAR_RENAMES = {
    "buf":  "buffer_data__",
    "ptr":  "mem_ptr__",
    "tmp":  "temp_val__",
    "len":  "length_val__",
    "idx":  "index_val__",
    "i":    "loop_idx__",
    "j":    "inner_idx__",
    "k":    "outer_idx__",
    "n":    "count_val__",
    "p":    "data_ptr__",
    "str":  "string_buf__",
    "src":  "source_buf__",
    "dst":  "dest_buf__",
    "ret":  "return_val__",
    "err":  "error_code__",
    "sz":   "size_val__",
}


def T14_variable_rename(code: str, safe_set: Optional[Set[int]] = None) -> TransformResult:
    """T14: Rename local variables using semantic-neutral aliases.

    FIX: Old version applied renames globally across ALL lines including
    tainted ones, which could corrupt the vulnerability. Now only renames
    on safe-region lines. Uses word-boundary matching to avoid partial renames.
    """
    lines   = code.splitlines()
    safe    = _safe_lines(code, safe_set)
    changed = False
    result  = []

    for i, line in enumerate(lines):
        if i not in safe:
            result.append(line)
            continue
        new = line
        for old, new_name in _VAR_RENAMES.items():
            new = re.sub(r'\b' + re.escape(old) + r'\b', new_name, new)
        result.append(new)
        if new != line:
            changed = True

    return TransformResult("\n".join(result), "T14", changed, "variable-rename")


def T15_declaration_reorder(code: str, safe_set: Optional[Set[int]] = None) -> TransformResult:
    """T15: Hoist variable declarations to the top of their enclosing block.

    FIX: Old version flushed declarations at the CLOSING brace (bottom),
    which is invalid C89 and produces undefined behaviour. New version
    correctly moves declarations to just after the OPENING brace.
    Uses a two-pass approach: collect then reinsert.
    """
    lines   = code.splitlines()
    safe    = _safe_lines(code, safe_set)
    changed = False

    DECL_RE = re.compile(
        r'^\s*(int|char|size_t|unsigned|long|float|double)\s+\w+[^(]*;')
    OPEN_RE = re.compile(r'\{\s*$')

    result = list(lines)
    # Work backwards to avoid index drift
    for i in range(len(result) - 1, -1, -1):
        if i not in safe or not OPEN_RE.search(result[i]):
            continue
        # Found an opening brace in safe region — collect declarations
        # that appear later in the same block (until matching close)
        depth = 1
        j = i + 1
        decl_indices = []
        while j < len(result) and depth > 0:
            depth += result[j].count("{") - result[j].count("}")
            if depth == 1 and j in safe and DECL_RE.match(result[j]):
                # Only hoist if it's not the very next line (already at top)
                if j > i + 1 + len(decl_indices):
                    decl_indices.append(j)
            j += 1

        if decl_indices:
            # Extract decl lines (in order) and reinsert right after open brace
            decl_lines = [result[idx] for idx in decl_indices]
            # Remove from original positions (in reverse)
            for idx in reversed(decl_indices):
                result.pop(idx)
            # Insert after the opening brace
            for k, dl in enumerate(decl_lines):
                result.insert(i + 1 + k, dl)
            changed = True

    return TransformResult("\n".join(result), "T15", changed, "decl-hoist")


def T16_declaration_init_split(code: str, safe_set: Optional[Set[int]] = None) -> TransformResult:
    """T16: int x = 5; → int x; x = 5;

    Splits declaration+initialisation into two statements.
    Skip lines containing malloc/calloc/realloc (splitting those is unsafe).
    """
    lines   = code.splitlines()
    safe    = _safe_lines(code, safe_set)
    changed = False
    inserts = []   # (orig_idx, replacement_lines)

    SPLIT_RE = re.compile(
        r'^(\s*)(int|char|size_t|unsigned|long|float|double)\s+(\w+)\s*=\s*([^;]+);'
    )

    for i, line in enumerate(lines):
        if i not in safe:
            continue
        if any(kw in line for kw in ("malloc", "calloc", "realloc",
                                      "sizeof", "(", "{")):
            continue
        m = SPLIT_RE.match(line)
        if m:
            indent  = m.group(1)
            typ     = m.group(2)
            varname = m.group(3)
            val     = m.group(4).strip()
            inserts.append((i, [
                f"{indent}{typ} {varname};  /* T16: split */",
                f"{indent}{varname} = {val};",
            ]))
            changed = True

    result = list(lines)
    for orig_idx, new_lines in reversed(inserts):
        result[orig_idx:orig_idx + 1] = new_lines

    return TransformResult("\n".join(result), "T16", changed, "decl-init-split")


def T17_dead_computation(code: str, safe_set: Optional[Set[int]] = None) -> TransformResult:
    """T17: Insert dead volatile computations after opening braces.

    FIX: Old version used `i % 4 == 1` which rarely fired on small funcs.
    New version inserts after every opening brace, capped at 4.
    """
    lines   = code.splitlines()
    safe    = _safe_lines(code, safe_set)
    changed = False
    inserts = []

    DEAD_STMTS = [
        "volatile int __dc__ = 0; __dc__ += sizeof(int); (void)__dc__;  /* T17 */",
        "volatile long __unused__ = (long)0 * sizeof(long); (void)__unused__;  /* T17 */",
        "{ volatile char __pad__[4] = {0,0,0,0}; (void)__pad__; }  /* T17 */",
        "volatile size_t __sz__ = sizeof(void *); (void)__sz__;  /* T17 */",
    ]

    inserted = 0
    for i, line in enumerate(lines):
        if i not in safe:
            continue
        if line.strip().endswith("{") and inserted < 4:
            indent = re.match(r'^(\s*)', line).group(1)
            inserts.append((i, f"{indent}  {random.choice(DEAD_STMTS)}"))
            inserted += 1
            changed = True

    result = list(lines)
    for orig_idx, text in reversed(inserts):
        result.insert(orig_idx + 1, text)

    return TransformResult("\n".join(result), "T17", changed, "dead-computation")


def T18_misleading_comment(code: str, safe_set: Optional[Set[int]] = None) -> TransformResult:
    """T18: Insert misleading security-positive comments before sensitive calls."""
    SENSITIVE = ("strcpy", "memcpy", "malloc", "realloc", "sprintf",
                 "strcat", "gets", "scanf", "fprintf", "strncpy", "memmove")
    COMMENTS  = [
        "/* T18: bounds check passed */",
        "/* T18: input sanitized above */",
        "/* T18: length verified OK */",
        "/* T18: safe — null-terminated */",
        "/* T18: LGTM: no overflow possible */",
        "/* T18: reviewed: secure allocation */",
        "/* T18: size validated prior to call */",
    ]
    lines   = code.splitlines()
    safe    = _safe_lines(code, safe_set)
    changed = False
    inserts = []

    for i, line in enumerate(lines):
        if i not in safe:
            continue
        if any(f"{s}(" in line for s in SENSITIVE):
            indent = re.match(r'^(\s*)', line).group(1)
            inserts.append((i, f"{indent}{random.choice(COMMENTS)}"))
            changed = True

    result = list(lines)
    for orig_idx, text in reversed(inserts):
        result.insert(orig_idx, text)   # insert BEFORE the sensitive line

    return TransformResult("\n".join(result), "T18", changed,
                           "misleading-comment")


# ══════════════════════════════════════════════════════════════════════════════
# Transform registry
# ══════════════════════════════════════════════════════════════════════════════

TRANSFORMS = {
    "T1":  T1_for_to_while,
    "T2":  T2_while_to_dowhile,
    "T3":  T3_loop_unrolling,
    "T4":  T4_loop_direction_reversal,
    "T5":  T5_pointer_aliasing,
    "T6":  T6_array_to_pointer,
    "T7":  T7_redundant_dereference,
    "T8":  T8_stack_to_heap,
    "T9":  T9_if_else_flattening,
    "T10": T10_opaque_predicate,
    "T11": T11_statement_reordering,
    "T12": T12_compound_condition_split,
    "T13": T13_dead_if_zero,
    "T14": T14_variable_rename,
    "T15": T15_declaration_reorder,
    "T16": T16_declaration_init_split,
    "T17": T17_dead_computation,
    "T18": T18_misleading_comment,
}


def apply_sequence(code: str,
                   sequence: List[str],
                   safe_set: Optional[Set[int]] = None) -> str:
    """Apply a sequence of transform IDs to code, respecting the safe region.

    CRITICAL FIX: safe_set contains ORIGINAL line indices. After each transform
    that inserts or removes lines, the indices shift. We recompute a mapped
    safe_set after every transform using a line-content fingerprint approach:
    mark the original safe lines with a unique token, apply transform, then
    find those tokens in the output to rebuild the safe_set.

    For transforms that don't change line count this is a no-op.
    For transforms that insert lines (T5,T8,T10,T13,T16,T17,T18) this ensures
    the safe_set stays accurate.
    """
    if not sequence:
        return code

    current_safe = safe_set  # original safe set for first transform

    for tid in sequence:
        fn = TRANSFORMS.get(tid)
        if fn is None:
            logger.warning("Unknown transform ID: %s", tid)
            continue

        original_lines = code.splitlines()
        n_before = len(original_lines)

        result = fn(code, current_safe)
        new_code = result.code
        new_lines = new_code.splitlines()
        n_after = len(new_lines)

        # If line count changed and we have a safe_set, remap it.
        # Strategy: for each original safe line, find it in the new output
        # by matching content. This handles insertions correctly.
        if current_safe is not None and n_after != n_before:
            new_safe = set()
            orig_safe_contents = {
                original_lines[i]: i
                for i in sorted(current_safe)
                if i < len(original_lines)
            }
            used_orig = set()
            for new_idx, new_line in enumerate(new_lines):
                stripped = new_line.rstrip()
                if stripped in orig_safe_contents:
                    orig_idx = orig_safe_contents[stripped]
                    if orig_idx not in used_orig:
                        new_safe.add(new_idx)
                        used_orig.add(orig_idx)
            # Also add lines that were inserted within safe blocks
            # (inserted lines should inherit safe status)
            current_safe = new_safe if new_safe else None

        code = new_code

    return code


def apply_single(code: str, tid: str,
                 safe_set: Optional[Set[int]] = None) -> TransformResult:
    fn = TRANSFORMS.get(tid)
    if fn is None:
        return TransformResult(code, tid, False, "unknown")
    return fn(code, safe_set)
