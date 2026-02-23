import numpy as np
import pandas as pd


def winrate(
    scores1: np.ndarray | pd.Series,
    scores2: np.ndarray | pd.Series,
    include_ties: bool = True,
) -> float:
    """
    Compute the win-rate between two sets of scores.

    Args:
        scores1: The first set of scores.
        scores2: The second set of scores.
        include_ties: Whether to include ties in the win-rate calculation.

    Returns:
        The win-rate between the two sets of scores.
    """
    scores1 = scores1.to_numpy() if isinstance(scores1, pd.Series) else scores1
    scores2 = scores2.to_numpy() if isinstance(scores2, pd.Series) else scores2

    assert scores1.shape == scores2.shape

    if np.array_equal(scores1, scores2):
        return 0.5

    wins = (scores1 > scores2).sum()
    num_instances = scores1.size

    if include_ties:
        ties = (scores1 == scores2).sum()
        return (wins + 0.5 * ties) / num_instances
    else:
        losses = (scores1 < scores2).sum()
        denom = wins + losses
        return 0.5 if denom == 0 else wins / denom
