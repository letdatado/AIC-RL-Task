You are given two arrays: y_true and y_pred, representing ground-truth and predicted labels for a multi-class classification task.
Implement a function:

    def macro_f1(y_true, y_pred):

that returns the macro-averaged F1 score as a float in [0, 1].

Requirements:
1) Input validation:
   - y_true and y_pred must have the same length; if not, raise ValueError.
   - Accept Python lists, NumPy arrays, or any general iterables (you may materialize to a list).
2) Class set:
   - Treat each unique label appearing in either y_true or y_pred as a class (one-vs-rest).
   - Labels may be any hashable type (ints, strings, tuples, frozensets, etc).
   - Special case: treat all NaN values (float("nan"), numpy.nan) as the SAME label and consider NaN == NaN for the purpose of class identity and equality tests inside this function.
   - IMPORTANT: Two labels are considered equal only if they have the SAME TYPE and the SAME VALUE (except for NaN, which equals NaN). For example, True and 1 are distinct classes; False and 0 are distinct classes.
3) Per-class metrics:
   - For each class c, compute:
     TP_c = count(yt == c and yp == c)
     FP_c = count(yt != c and yp == c)
     FN_c = count(yt == c and yp != c)
   - precision_c = TP_c / (TP_c + FP_c)  (if denominator is 0, define precision_c = 0)
   - recall_c    = TP_c / (TP_c + FN_c)  (if denominator is 0, define recall_c = 0)
   - f1_c = 0 if precision_c + recall_c == 0, else 2 * precision_c * recall_c / (precision_c + recall_c)
4) Macro average:
   - macro_f1 = simple mean of f1_c over all classes (equal weight per class).
5) Constraints:
   - Do not mutate the inputs (treat them as read-only).
   - Do not import external libraries and do not import numpy.
   - Return a single float in [0, 1]. Do not print.

Notes:
- Handling NaN: two NaNs should be treated as the same label and considered equal for the counting above. You may normalize labels internally (e.g., map any NaN to a sentinel object).
- Handling general iterables: inputs may be generators; do not assume __len__ is available.

Implementation/style requirement: place any imports at the top of the file, not inside macro_f1.


Implementation/style requirement: place any imports at the top of the file, not inside macro_f1.


Implementation/style requirement: macro_f1 must start with a short docstring describing inputs, outputs, and edge cases handled.

