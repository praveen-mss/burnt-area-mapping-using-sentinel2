import numpy as np


def population_stability_index(expected, actual, bins=10):

    expected_hist, bin_edges = np.histogram(expected, bins=bins)
    actual_hist, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = expected_hist / np.sum(expected_hist)
    actual_pct = actual_hist / np.sum(actual_hist)

    psi = np.sum(
        (actual_pct - expected_pct) *
        np.log((actual_pct + 1e-6) / (expected_pct + 1e-6))
    )

    return psi