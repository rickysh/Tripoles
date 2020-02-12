from itertools import combinations, product
import numpy as np
import operator
import pandas as pd
import pickle
import time


N = 95  # No. of stocks

M = int(N/2)
shift = 2  # delay (in days)


delta = 0.2  # jump threshold
tau = 0.7  # similarity threshold
kappa = 0.7

coeff = [1, 1, delta, -delta]
sd = np.real(np.roots(coeff)[2])

filename = "D"
symbols_list = []
infile = open(filename, 'rb')
D = pickle.load(infile)
D = D.iloc[:, :(N + 1)]
infile.close()
for col in D.columns:
    symbols_list.append(col.rstrip('\n'))
symbols_list.remove('Date')
S = combinations(symbols_list, 3)


def calculate_strength(D, T0, T1, T2):
    return D[T0].corr(D[T1] + D[T2])


def calculate_jump(D, T0, T1, T2):
    sigma = calculate_strength(D, T0, T1, T2)
    return sigma ** 2 - max((D[T0].corr(D[T1])) ** 2, (D[T0].corr(D[T2])) ** 2)


def get_interesting_tripoles(D, S, delta):
    U = []
    for (T0, T1, T2) in S:
        if calculate_jump(D, T0, T1, T2) > delta:
            U.append( (T0, T1, T2) )
    return U


def is_tripole_similarity(D, tri, tri_tag, tau):
    if (D[tri[0]].corr(D[tri_tag[0]]) >= tau) \
            and (D[tri[1]].corr(D[tri_tag[1]]) >= tau) \
            and (D[tri[2]].corr(D[tri_tag[2]]) >= tau):
        return True
    else:
        return False


def remove_all_similar_tripoles(D, C0, ht, tau):
    tripoles_to_remove = [tri for tri in C0 if (is_tripole_similarity(D, ht, tri, tau))]
    for tri in tripoles_to_remove:
        del C0[tri]
    return C0


def get_non_redundant_tripoles(D, C0, tau):
    C = []
    while C0 != {}:
        ht = max(C0.items(), key=operator.itemgetter(1))[0]
        C.append(ht)
        C0 = remove_all_similar_tripoles(D, C0, ht, tau)
    return C


def are_only_negative_tripoles(D, S):
    answer = True
    for (T0, T1, T2) in S:
        if calculate_strength(D, T0, T1, T2) >= 0:
            answer = False
            break
    return answer


def find_pairs_with_sd_magnitude(D, sd, symbols_list):
    S2 = combinations(symbols_list, 2)
    pairs = [(T1, T2) for (T1, T2) in S2 if abs(D[T1].corr(D[T2])) >= abs(sd)]
    return pairs


def keep_pairs_with_negative_correlation(D, L):
    pairs_to_remove = [pair for pair in L if D[pair[0]].corr(D[pair[1]]) >= 0]
    for pair in pairs_to_remove:
        L.remove(pair)
    return L


def find_candidate_super_pairs(D, S, sd, delta, symbols_list):
    L = find_pairs_with_sd_magnitude(D, sd, symbols_list)
    if are_only_negative_tripoles(D, S) and (delta >= 0.0903):
        L = keep_pairs_with_negative_correlation(D, L)
    return L


def find_tripoles_for_super_pair(D, pair, delta, symbols_list):
    Ta, Tb = pair
    sp_corr = D[Ta].corr(D[Tb])
    Q = [Ti for Ti in symbols_list if max(abs(D[Ti].corr(D[Ta])), abs(D[Ti].corr(D[Tb]))) < abs(sp_corr)]
    P = {(Ti, Ta, Tb): calculate_jump(D, Ti, Ta, Tb) for Ti in Q if calculate_jump(D, Ti, Ta, Tb) >= delta}
    C2 = {(Ta, Ti, Tb): calculate_jump(D, Ta, Ti, Tb) for Ti in Q if calculate_jump(D, Ta, Ti, Tb) >= delta}
    C3 = {(Tb, Ti, Ta): calculate_jump(D, Tb, Ti, Ta) for Ti in Q if calculate_jump(D, Tb, Ti, Ta) >= delta}
    P.update(C2)
    P.update(C3)
    return P


def most_negative_pair(D, E):
    neg = float("inf")
    neg_pair = E[0]
    for pair in E:
        if D[pair[0]].corr(D[pair[1]]) < neg:
            neg = D[pair[0]].corr(D[pair[1]])
            neg_pair = pair
    return neg_pair


def find_non_redundant_super_pairs(D, E, kappa, symbols_list):
    L = []
    while E != []:
        Ta, Tb = most_negative_pair(D, E)
        L.append( (Ta, Tb) )
        Z_Ta = [t for t in symbols_list if D[Ta].corr(D[t]) >= kappa]
        Z_Tb = [t for t in symbols_list if D[Tb].corr(D[t]) >= kappa]
        cross_pairs = product(Z_Ta, Z_Tb)
        for pair in cross_pairs:
            if pair in E:
                E.remove(pair)
    return L


def Naive_Approach(D, S, delta, tau):
    C0_U = get_interesting_tripoles(D, S, delta)
    C0 = {}
    for (T0, T1, T2) in C0_U:
        C0[(T0, T1, T2)] = calculate_jump(D, T0, T1, T2)
    C = get_non_redundant_tripoles(D, C0, tau)
    return C


def CONTRaComplete(D, S, delta, tau, sd, symbols_list):
    C = {}
    L = find_candidate_super_pairs(D, S, sd, delta, symbols_list)
    for pair in L:
        P = find_tripoles_for_super_pair(D, pair, delta, symbols_list)
        C.update(P)
    C = get_non_redundant_tripoles(D, C, tau)
    return C


def CONTRaFast(D, S, delta, tau, kappa, sd, symbols_list):
    C = {}
    L = find_candidate_super_pairs(D, S, sd, delta, symbols_list)
    L = find_non_redundant_super_pairs(D, L, kappa, symbols_list)
    for pair in L:
        P = find_tripoles_for_super_pair(D, pair, delta, symbols_list)
        C.update(P)
    C = get_non_redundant_tripoles(D, C, tau)
    return C


def create_shifted_stocks(D, shift, M):
    D = D.iloc[:, :(M + 1)]
    E = D.iloc[:, 1:(M + 1)]
    E = E.add_suffix('_' + str(shift))
    E = E.shift(-shift)
    D = pd.concat([D, E], axis=1)
    D = D.iloc[:D.shape[0] - shift]
    return D


def print_results(start_time, C):
    elapsed_time = time.time() - start_time
    print("C = " + str(C))
    print("No. of tripoles: = " + str(len(C)))
    print("Running time took: %s [s]" % elapsed_time)
    print()


print("Naive Approach :")
print("================")
start_time = time.time()
C = Naive_Approach(D, S, delta, tau)
print_results(start_time, C)

print("CONTRaComplete :")
print("================")
start_time = time.time()
C = CONTRaComplete(D, S, delta, tau, sd, symbols_list)
print_results(start_time, C)

print("CONTRaFast :")
print("============")
start_time = time.time()
C = CONTRaFast(D, S, delta, tau, kappa, sd, symbols_list)
print_results(start_time, C)

symbols_list = []
D = create_shifted_stocks(D, shift, M)
for col in D.columns:
    symbols_list.append(col.rstrip('\n'))
symbols_list.remove('Date')
S = combinations(symbols_list, 3)
print("============================")
print("Checking Time-lagged stocks:")
print("============================")
print()
print("Naive Approach :")
print("================")
start_time = time.time()
C = Naive_Approach(D, S, delta, tau)
print_results(start_time, C)

print("CONTRaComplete :")
print("================")
start_time = time.time()
C = CONTRaComplete(D, S, delta, tau, sd, symbols_list)
print_results(start_time, C)

print("CONTRaFast :")
print("============")
start_time = time.time()
C = CONTRaFast(D, S, delta, tau, kappa, sd, symbols_list)
print_results(start_time, C)