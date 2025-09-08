import numpy as np

def rand_hullermeier(P, Q, true_class=False):
    """
    Computes the Hüllermeier Rand Index using L¹ metric for fuzzy or hard clusterings.
    
    Source for implementation can be found on https://hal.science/hal-00734389v1/document.
    
    Parameters
    ----------
    P, Q : ndarray
        Fuzzy membership matrices of the clusters to be compared.
    true_class : bool, default True
        If true_class is True, the function treats Q as a true label vector instead of a membership matrix.

    Returns
    -------
    float
        Hüllermeier Rand Index
    """
    n_data = P.shape[0]

    distances_P = np.abs(P[:, np.newaxis, :] - P[np.newaxis, :, :]).sum(axis=2)
    equivalence_P = 1 - 0.5 * distances_P

    if not true_class:
        distances_Q = np.abs(Q[:, np.newaxis, :] - Q[np.newaxis, :, :]).sum(axis=2)
        equivalence_Q = 1 - 0.5 * distances_Q
    else:
        equivalence_Q = (Q[:, np.newaxis] == Q[np.newaxis, :]).astype(float)

    # Only upper triangle indices (i < j)
    triu_indices = np.triu_indices(n_data, k=1)
    discordances = np.abs(equivalence_P[triu_indices] - equivalence_Q[triu_indices])

    return 1 - discordances.sum() / (n_data * (n_data - 1) / 2)
    
def pairwise_counts(P, Q):
    """
    Computes the (N_SS, N_SD, N_DS, N_DD) values for two different fuzzy partitions P and Q of the same data. 

    These are similar to TP, FP, TN and FN metrics and are calculated by treating the fuzzy membership matrix as a probability matrix and calculating the expected value.

    Parameters
    ----------
    P : ndarray
        Membership matrix.
    Q : ndarray
        Membership matrix or label vector.

    Returns
    -------
    N_SS : float
        Expected value of pairs that are on the same cluster on both partitions.
    N_SD : float
        Expected value of pairs that are on the same cluster in P and on different clusters in Q.
    N_DS : float
        Expected value of pairs that are on different clusters in P and on the same cluster in Q.
    N_DD : float
        Expected value of pairs that are on different clusters on both partitions. 
    """
    
    P = np.asarray(P)
    Q = np.asarray(Q)

    n_data = P.shape[0]
    probabilities_P = P @ P.T

    if Q.ndim == 1 or Q.shape[1] == 1: 
        probabilities_Q = (Q.reshape(-1, 1) == Q.reshape(1, -1)).astype(float)
    else: 
        probabilities_Q = Q @ Q.T

    triu_indices = np.triu_indices(n_data, k=1)
    p_vals = probabilities_P[triu_indices]
    q_vals = probabilities_Q[triu_indices]

    same_same = np.sum(p_vals * q_vals)
    same_different = np.sum(p_vals * (1 - q_vals))
    different_same = np.sum((1 - p_vals) * q_vals)
    different_different = np.sum((1 - p_vals) * (1 - q_vals))

    return same_same, same_different, different_same, different_different


def fuzzy_rand_index(P, Q):
    """
    Receives membership matrices P and Q and outputs fuzzy Rand index.
    
    Parameters
    ----------
    P : ndarray
        Membership matrix.
    Q : ndarray
        Membership matrix or label vector.

    Returns
    -------
    float
        Fuzzy Rand index.
    """
    same_same, same_different, different_same, different_different = pairwise_counts(P, Q)

    total = same_same + same_different + different_same + different_different
    return (same_same + different_different) / total if total != 0 else 0.0

def fuzzy_F1_measure(P, Q):
    """
    Receives membership matrices P and Q and outputs the fuzzy F1-measure. For hard clusterings, it's equivalent to the F1-measure.
    
    Parameters
    ----------
    P : ndarray
        Membership matrix.
    Q : ndarray
        Membership matrix or label vector.

    Returns
    -------
    float
        Fuzzy F1-measure.
    """
    same_same, same_different, different_same, different_different = pairwise_counts(P, Q)

    denominator = 2*same_same + different_same + same_different
    return 2*same_same / denominator if denominator != 0 else 0.0