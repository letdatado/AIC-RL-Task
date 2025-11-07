# -*- coding: utf-8 -*-
"""
grader/grade.py — clean rewrite

What this grader enforces:
  - y_true and y_pred must have SAME length (else raise ValueError)
  - Accept general iterables (generators): we materialize to lists ONCE
  - NaN is a valid label and NaN == NaN (via sentinel normalization)
  - Macro-F1 computed over union of labels, equal class weights
  - No input mutation (we pass lists into the candidate and compare after)
  - Edge cases: unseen predicted class, gapped labels, strings, numpy arrays,
    perfect/all-wrong, extreme imbalance, composite labels, generator inputs.

NOTE: We allow numpy in candidate solutions again.
"""

import importlib.util
import os
import sys
from typing import Iterable, Any
import math

try:
    import numpy as np
except Exception as e:
    print("FAIL: numpy not available in environment:", e)
    sys.exit(1)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLUTION_PATH = os.path.join(ROOT, "starter", "solution.py")

if not os.path.exists(SOLUTION_PATH):
    print("FAIL: starter/solution.py not found")
    sys.exit(1)

spec = importlib.util.spec_from_file_location("candidate_solution", SOLUTION_PATH)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)  # type: ignore
except Exception as e:
    print("FAIL: importing solution raised an exception:", e)
    sys.exit(1)

if not hasattr(mod, "macro_f1"):
    print("FAIL: solution.py must define a function named macro_f1")
    sys.exit(1)

macro_f1 = getattr(mod, "macro_f1")
# --- Style check: macro_f1 must begin with a docstring ---
try:
    import inspect, textwrap, ast
    _src = inspect.getsource(mod.macro_f1)
    # Parse the function with AST and check first statement
    _fn_ast = ast.parse(_src).body[0]  # FunctionDef
    _has_docstring = ast.get_docstring(_fn_ast) is not None and len(ast.get_docstring(_fn_ast).strip()) > 0
    if not _has_docstring:
        print("FAIL: macro_f1 must start with a docstring describing inputs, outputs, and edge cases.")
        sys.exit(1)
except Exception as _e:
    print("FAIL: unable to inspect macro_f1 source for docstring check:", _e)
    sys.exit(1)
    import inspect, textwrap, re
    _src = inspect.getsource(mod.macro_f1)
    _body = textwrap.dedent("\n".join(_src.splitlines()[1:]))  # drop "def" line, dedent body
    if re.search(r'^\s*import\s+\w+|^\s*from\s+\w+\s+import', _body, flags=re.MULTILINE):
        print("FAIL: imports are not allowed inside macro_f1; place imports at module top.")
        sys.exit(1)
except Exception as _e:
    print("FAIL: unable to inspect macro_f1 source for style checks:", _e)
    sys.exit(1)

# ---------- Reference implementation (with generator + NaN handling) ----------

_SENTINEL_NAN = object()

def _is_nan(x) -> bool:
    try:
        return isinstance(x, float) and math.isnan(x)
    except Exception:
        return False

def _normalize_label(x):
    # Collapse any NaN (float('nan'), numpy.nan) to a shared sentinel so NaN == NaN
    if _is_nan(x):
        return _SENTINEL_NAN
    return x

def _materialize(a):
    # Accept general iterables: convert to list once (handles generators)
    try:
        if isinstance(a, (list, tuple, np.ndarray)):
            return list(a)
    except Exception:
        pass
    return list(a)

def _macro_f1_ref(y_true: Iterable[Any], y_pred: Iterable[Any]) -> float:
    # Materialize for safe multiple passes and to support len()
    yt = _materialize(y_true)
    yp = _materialize(y_pred)

    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have the same length")

    yt = [_normalize_label(v) for v in yt]
    yp = [_normalize_label(v) for v in yp]

    classes = sorted(set(yt) | set(yp), key=lambda x: (str(type(x)), str(x)))

    f1s = []
    for c in classes:
        tp = sum(1 for a,b in zip(yt, yp) if a == c and b == c)
        fp = sum(1 for a,b in zip(yt, yp) if a != c and b == c)
        fn = sum(1 for a,b in zip(yt, yp) if a == c and b != c)

        prec = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
        rec  = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        f1   = 0.0 if (prec + rec) == 0 else (2 * prec * rec) / (prec + rec)
        f1s.append(f1)

    return 0.0 if not f1s else float(sum(f1s) / len(f1s))

def _approx_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    if math.isfinite(a) and math.isfinite(b):
        return abs(a - b) <= tol
    return False

def _frozen_copy_list(x_list):
    # x_list is a list (we pass lists to the candidate); copy for mutation check
    return list(x_list)

# ---------- Single test runner (materializes before calling candidate) ----------

def run_case(name, y_true, y_pred, expect=None, expect_exception: type | None = None):
    # Materialize inputs ONCE (so both candidate and reference see the same data)
    yt = _materialize(y_true)
    yp = _materialize(y_pred)

    yt0 = _frozen_copy_list(yt)
    yp0 = _frozen_copy_list(yp)

    try:
        out = macro_f1(yt, yp)
        if expect_exception is not None:
            print(f"FAIL:{name}: expected exception {expect_exception.__name__}, but none was raised")
            return False
    except Exception as e:
        if expect_exception is not None and isinstance(e, expect_exception):
            print(f"PASS:{name}: raised expected {expect_exception.__name__}")
            return True
        print(f"FAIL:{name}: unexpected exception: {e}")
        return False

    if not isinstance(out, (float, int)):
        print(f"FAIL:{name}: output must be a float (or int convertible), got {type(out)}")
        return False

    out = float(out)
    if not (0.0 - 1e-12 <= out <= 1.0 + 1e-12):
        print(f"FAIL:{name}: output {out} not in [0, 1]")
        return False

    ref = _macro_f1_ref(yt, yp) if expect is None else expect
    if not _approx_equal(out, ref):
        print(f"FAIL:{name}: value mismatch. got={out:.12f}, expected(ref)={ref:.12f}")
        return False

    # Mutation checks: candidate received lists; they must remain equal to snapshots
    if yt != yt0:
        print(f"FAIL:{name}: y_true was mutated")
        return False
    if yp != yp0:
        print(f"FAIL:{name}: y_pred was mutated")
        return False

    print(f"PASS:{name}: {out:.12f}")
    return True

# ---------- Test suite ----------

def main():
    all_ok = True

    # Base battery
    all_ok &= run_case("balanced_3class", ["A","A","B","B","C","C"], ["A","B","B","B","C","A"])
    all_ok &= run_case("no_pred_for_class", [0,0,1,1,1,2,2], [0,0,1,1,1,0,0])
    all_ok &= run_case("class_absent_in_truth", [0,0,1,1,1,0,0], [0,0,1,1,1,2,2])
    all_ok &= run_case("gapped_labels", [0,0,2,2,5,5,5], [0,2,2,5,5,5,0])
    all_ok &= run_case("string_labels_unseen_pred", ["dog","dog","cat","cat","mouse"], ["dog","cat","cat","mouse","unicorn"])
    all_ok &= run_case("numpy_arrays", np.array([1,1,2,2,3,3]), np.array([1,2,2,3,3,1]))
    all_ok &= run_case("perfect", ["x","y","z","x","y","z"], ["x","y","z","x","y","z"], expect=1.0)
    all_ok &= run_case("all_wrong", [0,1,2], [1,2,0])

    # Length mismatch must raise ValueError
    all_ok &= run_case("length_mismatch", [0,1,2], [0,1], expect_exception=ValueError)

    # NaN as label (NaN == NaN); also contains unseen predicted label 'x'
    nan = float("nan")
    y_true = [nan, "a", "a", nan, "b"]
    y_pred = [nan, "a", nan, "x", "b"]
    all_ok &= run_case("nan_as_label", y_true, y_pred)

    # Extreme imbalance
    yt = [0]*98 + [1]*2
    yp = [0]*99 + [1]
    all_ok &= run_case("extreme_imbalance", yt, yp)

    # Composite labels
    y_true = [("A",1), ("A",2), "B", "B", frozenset({1,2})]
    y_pred = [("A",1), "B", "B", frozenset({2,1}), ("X",9)]
    all_ok &= run_case("composite_labels", y_true, y_pred)

    # Generator inputs (no __len__)
    def gen1():
        for x in ["a","b","a","c"]:
            yield x
    def gen2():
        for x in ["a","a","c","c"]:
            yield x
    all_ok &= run_case("generator_inputs", gen1(), gen2())

    if all_ok:
        print("GRADE:PASS")
        sys.exit(0)
    else:
        print("GRADE:FAIL")
        sys.exit(1)

if __name__ == "__main__":
    main()

