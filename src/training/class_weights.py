import numpy as np


def compute_class_weights(y):

    """
    Compute class weights for imbalanced datasets.
    """

    classes = np.unique(y)
    total = len(y)

    weights = {}

    for c in classes:
        count = np.sum(y == c)
        weights[c] = total / (len(classes) * count)

    return weights


def compute_scale_pos_weight(y):

    """
    For XGBoost imbalance handling.
    """

    pos = np.sum(y == 1)
    neg = np.sum(y == 0)

    return neg / pos