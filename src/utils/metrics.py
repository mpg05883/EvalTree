import numpy as np
from scipy.stats import rankdata


def kendallw(scores: np.ndarray) -> float:
    """Compute Kendall's W (coefficient of concordance) across models.

    Each instance acts as a judge that ranks all models, and W measures how
    consistently instances agree on the model ranking.

    Args:
        scores: 2D array of shape (num_instances, num_models) where each row
            is an instance and each column is a model. Entry [i, j] is the
            score of model j on instance i; higher scores are treated as
            better.

    Returns:
        W in [0, 1]. 0 indicates no agreement, 1 indicates perfect agreement.
    """
    scores = np.asarray(scores, dtype=float)
    num_instances, num_models = scores.shape

    # Rank each instance's model scores independently (average method handles ties)
    ranks = np.apply_along_axis(rankdata, axis=1, arr=scores)

    # Sum of ranks for each model across all instances
    rank_sums = ranks.sum(axis=0)

    # S: sum of squared deviations of rank sums from the grand mean
    S = np.sum((rank_sums - rank_sums.mean()) ** 2)

    # Tie correction factor: T = sum over instances of sum over tied groups of (t^3 - t)
    T = 0.0
    for i in range(num_instances):
        _, counts = np.unique(scores[i], return_counts=True)
        T += np.sum(counts**3 - counts)

    denominator = num_instances**2 * (num_models**3 - num_models) - num_instances * T
    if denominator == 0:
        return 0.0

    return 12 * S / denominator
