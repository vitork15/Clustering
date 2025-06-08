import numpy as np
import itertools

def rand_hullermeier(P, Q, true_class=False):
    '''
    Receives membership matrices P and Q to generate Hullemeier rand index using L¹ metric.
    The boolean flag true_class treats Q as a true class vector in a similar manner to true/prediction metrics.
    Source for implementation can be found on https://hal.science/hal-00734389v1/document.
    '''

    distance = 0
    cluster_num = len(P[0])
    data_num = len(P)

    if true_class is not True:
        for i, j in itertools.product(range(data_num), repeat=2):
            if i >= j:
                continue # 1 <= i < j <= n condition

            e_q, e_p = 0, 0

            for cluster in range(cluster_num):
                e_q += abs(Q[i][cluster]-Q[j][cluster])
                e_p += abs(P[i][cluster]-P[j][cluster])

            distance += abs(e_q - e_p)

        return 1 - 2*distance/(data_num*(data_num-1))

    else:
        for i, j in itertools.product(range(data_num), repeat=2):
            if i >= j:
                continue # 1 <= i < j <= n condition

            e_q, e_p = 0, 0

            e_q += 1 - int(Q[i]==Q[j])
            for cluster in range(cluster_num):
                e_p += abs(P[i][cluster]-P[j][cluster])

            distance += abs(e_q - e_p)

        return 1 - 2*distance/(data_num*(data_num-1))

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