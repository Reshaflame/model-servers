def majority_voting(predictions):
    """
    Simple majority voting method.
    Input: predictions (list of model predictions [-1 or 1])
    Output: True if majority predicts anomaly (-1), else False.
    """
    return sum(predictions) / len(predictions) < 0.5
