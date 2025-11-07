def macro_f1(y_true, y_pred):
    """
    Compute the macro-averaged F1 score for multi-class classification.
    
    Args:
        y_true: Ground-truth labels (list, array, or iterable)
        y_pred: Predicted labels (list, array, or iterable)
    
    Returns:
        float: Macro-averaged F1 score in [0, 1]
    
    Raises:
        ValueError: If y_true and y_pred have different lengths
    
    Handles:
        - Any hashable label types (int, str, tuple, frozenset, etc.)
        - NaN values treated as equal to each other
        - Type-strict equality (True != 1, False != 0)
        - Zero-division cases (precision/recall = 0 if denominator is 0)
    """
    
    # Materialize inputs to lists
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)
    
    # Validate lengths
    if len(y_true_list) != len(y_pred_list):
        raise ValueError("y_true and y_pred must have the same length")
    
    # Normalize NaN values: create a sentinel for NaN comparison
    class _NaNSentinel:
        def __eq__(self, other):
            return isinstance(other, _NaNSentinel)
        def __hash__(self):
            return hash(id(self.__class__))
    
    _nan_sentinel = _NaNSentinel()
    
    def normalize_label(label):
        """Convert NaN to sentinel, keep everything else as-is."""
        try:
            # Check if it's a float NaN
            if isinstance(label, float) and label != label:  # NaN != NaN is True
                return _nan_sentinel
        except (TypeError, ValueError):
            pass
        return label
    
    # Normalize all labels
    y_true_normalized = [normalize_label(label) for label in y_true_list]
    y_pred_normalized = [normalize_label(label) for label in y_pred_list]
    
    # Get all unique classes
    classes = set(y_true_normalized) | set(y_pred_normalized)
    
    # Handle edge case: empty input
    if len(y_true_normalized) == 0:
        return 0.0
    
    # Compute F1 for each class
    f1_scores = []
    
    for c in classes:
        # Count TP, FP, FN
        tp = sum(1 for yt, yp in zip(y_true_normalized, y_pred_normalized) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true_normalized, y_pred_normalized) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true_normalized, y_pred_normalized) if yt == c and yp != c)
        
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
    macro_f1_score = sum(f1_scores) / len(f1_scores) if len(f1_scores) > 0 else 0.0
    
    return macro_f1_score