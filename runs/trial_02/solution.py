import math

def macro_f1(y_true, y_pred):
    """
    Compute the macro-averaged F1 score for multi-class classification.
    
    Args:
        y_true: Ground-truth labels (list, array, or iterable).
        y_pred: Predicted labels (list, array, or iterable).
    
    Returns:
        float: Macro-averaged F1 score in [0, 1].
    
    Raises:
        ValueError: If y_true and y_pred have different lengths.
    
    Edge cases:
        - Handles NaN values: all NaNs are treated as the same label.
        - Handles type distinction: True != 1, False != 0, etc.
        - Handles zero-division: precision and recall are 0 if denominator is 0.
        - Accepts any hashable label types.
    """
    
    # Sentinel object for NaN normalization
    _NAN_SENTINEL = object()
    
    def normalize_label(label):
        """Normalize label, treating all NaNs as the same sentinel."""
        try:
            if isinstance(label, float) and math.isnan(label):
                return _NAN_SENTINEL
        except (TypeError, ValueError):
            pass
        return label
    
    # Materialize inputs to lists and validate length
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)
    
    if len(y_true_list) != len(y_pred_list):
        raise ValueError("y_true and y_pred must have the same length")
    
    # Normalize all labels
    y_true_normalized = [normalize_label(label) for label in y_true_list]
    y_pred_normalized = [normalize_label(label) for label in y_pred_list]
    
    # Collect all unique classes
    classes = set(y_true_normalized) | set(y_pred_normalized)
    
    # Handle empty case
    if not classes:
        return 0.0
    
    # Compute F1 for each class
    f1_scores = []
    
    for c in classes:
        # Count TP, FP, FN
        tp = sum(1 for yt, yp in zip(y_true_normalized, y_pred_normalized)
                 if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true_normalized, y_pred_normalized)
                 if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true_normalized, y_pred_normalized)
                 if yt == c and yp != c)
        
        # Compute precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Compute F1
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        f1_scores.append(f1)
    
    # Compute macro average
    macro_f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    return macro_f1_score