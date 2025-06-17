import numpy as np
import itertools

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

    # Compute pairwise L1 distances for P
    P_diff = np.abs(P[:, np.newaxis, :] - P[np.newaxis, :, :]).sum(axis=2)

    if not true_class:
        # Compute pairwise L1 distances for Q
        Q_diff = np.abs(Q[:, np.newaxis, :] - Q[np.newaxis, :, :]).sum(axis=2)
    else:
        # Hard class case: Q is a label vector
        Q_diff = (Q[:, np.newaxis] != Q[np.newaxis, :]).astype(float)  # 1 if different, 0 if same

    # Compute matrix of |E_q - E_p| = ||P(x)-P(x')|-|Q(x)-Q(x')||
    diff = np.abs(Q_diff - P_diff)

    # Only keep indices such that i < j
    triu_indices = np.triu_indices(n_data, k=1)
    total = diff[triu_indices].sum()

    # Final index
    return 1 - (2 * total) / (n_data * (n_data - 1))

def rand_frigui(P, Q, true_class=False):
    '''
    Receives membership matrices P and Q to generate Frigui rand index.
    The boolean flag true_class treats Q as a true class vector in a similar manner to true/prediction metrics.
    '''
    # these variables are similar to TP, FN, TN and FP
    same_same = 0
    same_different = 0
    different_different = 0
    different_same = 0

    cluster_num = len(P[0])
    data_num = len(P)

    if true_class is not True:
        for i, j in itertools.product(range(data_num), repeat=2):
            if i >= j:
                continue # 1 <= i < j <= n condition

            psi1, psi2 = 0, 0

            for cluster in range(cluster_num):
                psi1 += P[i][cluster]*P[j][cluster]
                psi2 += Q[i][cluster]*Q[j][cluster]

            same_same += psi1*psi2
            same_different += psi1*(1-psi2)
            different_same += (1-psi1)*psi2
            different_different += (1-psi1)*(1-psi2)

        return (same_same+different_different)/(same_same+same_different+different_same+different_different)

    else:
        for i, j in itertools.product(range(data_num), repeat=2):
            if i >= j:
                continue # 1 <= i < j <= n condition

            psi1, psi2 = 0, 0

            for cluster in range(cluster_num):
                psi1 += P[i][cluster]*P[j][cluster]

            psi2 = int(Q[i]==Q[j])

            same_same += psi1*psi2
            same_different += psi1*(1-psi2)
            different_same += (1-psi1)*psi2
            different_different += (1-psi1)*(1-psi2)

        return (same_same+different_different)/(same_same+same_different+different_same+different_different)