
````markdown
# Reinforcement Learning Task for LLMs: Implementing Macro-F1 Score

## Overview

This repository contains a reinforcement learning (RL) task designed to train or evaluate large language models (LLMs) on practical machine learning engineering skills — specifically, implementing a **Macro-F1 score function** from scratch with robust handling of edge cases.

The task was developed as part of the AI RL Challenge. The goal is to build a realistic coding task for an ML engineer or researcher that can be automatically graded and used in an RL training loop.

---

## Task Description

The model is given a prompt asking it to **implement a Python function**:

```python
def macro_f1(y_true, y_pred):
    """Compute the macro-averaged F1 score."""
````

The implementation must:

* Compute **macro-averaged F1** over all unique labels appearing in either `y_true` or `y_pred`.
* Support both **numerical and string labels**.
* Handle **NaN values** safely and consistently.
* Accept **NumPy arrays**, **Python lists**, or **generators**.
* Raise a `ValueError` if `y_true` and `y_pred` have different lengths.
* Return a **single built-in `float`** value in the range [0, 1].
* (Optional style requirement for tuning): include a docstring and place any imports at the top of the file.

---

## Structure

```
AIC-RL-Task/
│
├── prompt.md                # The instruction prompt shown to the model
├── grader/
│   └── grade.py             # Automated grading script
├── evaluate_with_claude.py  # Script to test pass-rate using Claude API
├── runs/                    # Saved results of model trials
└── README.md                # This file
```

---

## Why Macro-F1?

Macro-F1 is a core metric in machine learning for evaluating classification performance, especially in **imbalanced multi-class** settings. It teaches models to:

* Implement mathematical formulas correctly.
* Handle edge cases (missing classes, NaN, length mismatches).
* Generalize across data types and input forms.

These are realistic tasks faced by ML practitioners and researchers, making this a strong training signal for LLMs learning coding and reasoning.

---

## Grading System

The `grader/grade.py` file runs a battery of tests covering:

| Test Name                   | Purpose                              |
| --------------------------- | ------------------------------------ |
| `balanced_3class`           | Simple 3-class balanced data         |
| `no_pred_for_class`         | Missing prediction class             |
| `class_absent_in_truth`     | Class only in predictions            |
| `gapped_labels`             | Non-contiguous class labels          |
| `string_labels_unseen_pred` | String label generalization          |
| `numpy_arrays`              | NumPy compatibility                  |
| `perfect`                   | All correct predictions              |
| `all_wrong`                 | All predictions wrong                |
| `length_mismatch`           | Raises ValueError on unequal lengths |
| `nan_as_label`              | Handles NaN labels consistently      |
| `extreme_imbalance`         | Handles highly imbalanced data       |
| `composite_labels`          | Multi-type labels (string + int)     |
| `generator_inputs`          | Works with generator inputs          |

Each test computes the macro-F1 value and compares it against a reference implementation or expected behavior.

If all tests pass, the model receives `GRADE: PASS`.

---

## Results

The current implementation achieved:

```
Passes: 12/12  →  Pass-rate: 100.0%
```

This means Claude Haiku 4.5 reliably solved the task under the default setup.

However, as per RL task design guidelines, **the pass rate can be tuned to 10–40%** by:

* Increasing model sampling variance (`--temperature 0.9–1.0`).
* Adding style constraints (e.g., “must include docstring”, “no import inside function”).
* Enforcing stricter output-type checks (e.g., built-in `float` only).

These variations preserve the core learning goal but introduce realistic failure diversity.

---

## Setup & Usage

### 1. Install dependencies

```powershell
uv sync
```

### 2. Run the grader locally

```powershell
uv run python .\grader\grade.py
```

### 3. Evaluate with Claude API

```powershell
uv run python .\evaluate_with_claude.py -n 10 --model claude-haiku-4-5-20251001 --temperature 0.7
```

You’ll need an **Anthropic API key** set as:

```powershell
$env:ANTHROPIC_API_KEY = "<your-key-here>"
```

---

## Notes for Reviewers

* Current pass-rate: **100%**
* Tunable to: **10–40%** with added style constraints or temperature changes.
* Tested on **Claude 4.5 Haiku** model.
* Developed using **uv** (Python package manager).
* Designed for clarity, robustness, and ease of RL integration.

---

## License

MIT License — freely reusable for educational or research purposes.

````

