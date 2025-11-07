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
        - NaN values: treated as the same label (NaN == NaN)
        - Type-strict equality: True != 1, False != 0, etc.
        - Zero-division: precision/recall undefined cases return 0
        - General iterables: converts to lists internally
    """
    
    # Convert to lists to handle any iterable and allow multiple passes
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)
    
    # Validate lengths
    if len(y_true_list) != len(y_pred_list):
        raise ValueError("y_true and y_pred must have the same length")
    
    # Normalize labels: convert NaN to a sentinel object
    _NAN_SENTINEL = object()
    
    def normalize_label(label):
        # Check if label is a float NaN
        try:
            if isinstance(label, float) and label != label:  # NaN check
                return _NAN_SENTINEL
        except (TypeError, ValueError):
            pass
        return label
    
    y_true_normalized = [normalize_label(label) for label in y_true_list]
    y_pred_normalized = [normalize_label(label) for label in y_pred_list]
    
    # Get all unique classes from both true and pred
    classes = set(y_true_normalized) | set(y_pred_normalized)
    
    # Handle empty case
    if len(classes) == 0:
        return 0.0
    
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
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
        
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)
        
        # Compute F1
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        f1_scores.append(f1)
    
    # Macro average
    macro_f1_score = sum(f1_scores) / len(f1_scores)
    
    return macro_f1_score