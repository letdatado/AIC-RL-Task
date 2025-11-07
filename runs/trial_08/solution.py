import math


def macro_f1(y_true, y_pred):
    """
    Compute macro-averaged F1 score for multi-class classification.
    
    Args:
        y_true: Ground-truth labels (list, array, or iterable)
        y_pred: Predicted labels (list, array, or iterable)
    
    Returns:
        float: Macro-averaged F1 score in [0, 1]
    
    Raises:
        ValueError: If y_true and y_pred have different lengths
    
    Handles:
        - NaN values treated as equal to each other
        - Type-strict equality (True != 1, False != 0)
        - Zero-division cases (precision/recall = 0 when denominator is 0)
        - General iterables (materialized to lists)
    """
    # Materialize inputs to lists
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)
    
    # Validate lengths
    if len(y_true_list) != len(y_pred_list):
        raise ValueError("y_true and y_pred must have the same length")
    
    # Normalize NaN values to a sentinel
    # We need to handle float NaN specially since NaN != NaN
    sentinel = object()  # Unique sentinel for NaN
    
    def normalize_label(label):
        """Convert NaN to sentinel, keep everything else as-is."""
        try:
            if isinstance(label, float) and math.isnan(label):
                return sentinel
        except (TypeError, ValueError):
            pass
        return label
    
    y_true_norm = [normalize_label(label) for label in y_true_list]
    y_pred_norm = [normalize_label(label) for label in y_pred_list]
    
    # Get all unique classes from both y_true and y_pred
    classes = set(y_true_norm) | set(y_pred_norm)
    
    # Handle edge case: no samples
    if len(y_true_norm) == 0:
        return 0.0
    
    # Compute F1 for each class
    f1_scores = []
    
    for cls in classes:
        # Compute TP, FP, FN
        tp = sum(1 for yt, yp in zip(y_true_norm, y_pred_norm) if yt == cls and yp == cls)
        fp = sum(1 for yt, yp in zip(y_true_norm, y_pred_norm) if yt != cls and yp == cls)
        fn = sum(1 for yt, yp in zip(y_true_norm, y_pred_norm) if yt == cls and yp != cls)
        
        # Compute precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Compute F1
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        f1_scores.append(f1)
    
    # Macro average
    macro_f1_score = sum(f1_scores) / len(f1_scores) if len(f1_scores) > 0 else 0.0
    
    return macro_f1_score