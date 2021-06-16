# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import cython
from cython.parallel cimport prange, parallel
cimport numpy
import numpy
import math

def floyd_warshall(adjacency_matrix):

    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    cdef unsigned int n = nrows

    adj_mat_copy = adjacency_matrix.astype(long, order='C', casting='safe', copy=True)
    assert adj_mat_copy.flags['C_CONTIGUOUS']
    cdef numpy.ndarray[long, ndim=2, mode='c'] M = adj_mat_copy
    cdef numpy.ndarray[long, ndim=2, mode='c'] path = numpy.zeros([n, n], dtype=numpy.int64)

    cdef unsigned int i, j, k
    cdef long M_ij, M_ik, cost_ikkj
    cdef long* M_ptr = &M[0,0]
    cdef long* M_i_ptr
    cdef long* M_k_ptr

    # set unreachable nodes distance to 510
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = 0
            elif M[i][j] == 0:
                M[i][j] = 510

    # floyed algo
    for k in range(n):
        M_k_ptr = M_ptr + n*k
        for i in range(n):
            M_i_ptr = M_ptr + n*i
            M_ik = M_i_ptr[k]
            for j in range(n):
                cost_ikkj = M_ik + M_k_ptr[j]
                M_ij = M_i_ptr[j]
                if M_ij > cost_ikkj:
                    M_i_ptr[j] = cost_ikkj
                    path[i][j] = k

    # set unreachable path to 510
    for i in range(n):
        for j in range(n):
            if M[i][j] >= 510:
                path[i][j] = 510
                M[i][j] = 510

    return M, path


def get_all_edges(path, i, j):
    cdef unsigned int k = path[i][j]
    if k == 0:
        return []
    else:
        return get_all_edges(path, i, k) + [k] + get_all_edges(path, k, j)


def gen_edge_input(max_dist, path, edge_feat):

    (nrows, ncols) = path.shape
    assert nrows == ncols
    cdef unsigned int n = nrows
    cdef unsigned int max_dist_copy = max_dist

    path_copy = path.astype(long, order='C', casting='safe', copy=True)
    edge_feat_copy = edge_feat.astype(long, order='C', casting='safe', copy=True)
    assert path_copy.flags['C_CONTIGUOUS']
    assert edge_feat_copy.flags['C_CONTIGUOUS']

    cdef numpy.ndarray[long, ndim=4, mode='c'] edge_fea_all = -1 * numpy.ones([n, n, max_dist_copy, edge_feat.shape[-1]], dtype=numpy.int64)
    cdef unsigned int i, j, k, num_path, cur

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if path_copy[i][j] == 510:
                continue
            path = [i] + get_all_edges(path_copy, i, j) + [j]
            num_path = len(path) - 1
            for k in range(num_path):
                edge_fea_all[i, j, k, :] = edge_feat_copy[path[k], path[k+1], :]

    return edge_fea_all


def easy_bin(x, bin):
    x = float(x)
    cnt = 0
    if math.isinf(x):
        return 509
    if math.isnan(x):
        return 510

    while True:
        if cnt == len(bin):
            return cnt
        if x > bin[cnt]:
            cnt += 1
        else:
            return cnt


def bin_rel_pos_3d_1(rel_pos_3d, noise=False):
    (nrows, ncols) = rel_pos_3d.shape
    assert nrows == ncols
    cdef unsigned int n = nrows

    rel_pos_3d_copy = rel_pos_3d.astype(float, order='C', casting='safe', copy=True)
    assert rel_pos_3d_copy.flags['C_CONTIGUOUS']

    cdef numpy.ndarray[long, ndim=2, mode='c'] rel_pos_new = numpy.zeros([n, n], dtype=numpy.int64)
    cdef float t

    for i in range(n):
        for j in range(n):
            t = rel_pos_3d_copy[i][j]
            if noise:
                t += numpy.random.laplace(0.001994, 0.031939)

            rel_pos_new[i][j] = easy_bin(t,
            [ 0.        ,  1.2203719 ,  1.32591315,  1.34945971,  1.36901592,
                1.38554966,  1.39408201,  1.39821761,  1.40340829,  1.42018548,
                1.44106823,  1.46643972,  1.48281769,  1.50389185,  1.51891267,
                1.52631782,  1.53147129,  1.53860769,  1.55639537,  2.14345225,
                2.23482929,  2.29313914,  2.33590191,  2.36154539,  2.38264428,
                2.39810588,  2.40864859,  2.41532955,  2.42179826,  2.42905216,
                2.43669323,  2.44682172,  2.45996124,  2.47527397,  2.49097653,
                2.50466232,  2.51628465,  2.52561919,  2.53675456,  2.55197741,
                2.57106068,  2.60527318,  2.68841622,  2.7575051 ,  2.78908825,
                2.81353888,  2.85781152,  2.92351878,  2.98054664,  3.04443951,
                3.10098176,  3.15351515,  3.22568385,  3.31520129,  3.4091448 ,
                3.48933436,  3.5482047 ,  3.59131896,  3.62714856,  3.65466199,
                3.68500344,  3.71346017,  3.74434484,  3.77633935,  3.80365214,
                3.82832487,  3.87205751,  3.92707567,  4.02696632,  4.1215165 ,
                4.18112367,  4.23146583,  4.27758791,  4.3201998 ,  4.36266293,
                4.41073577,  4.45611755,  4.50280075,  4.54940725,  4.60003344,
                4.6575826 ,  4.71576474,  4.76870864,  4.81631325,  4.86024147,
                4.90682051,  4.95473686,  5.00370847,  5.05290008,  5.10632542,
                5.17100525,  5.2472393 ,  5.33202845,  5.41861402,  5.50807017,
                5.59746962,  5.6866421 ,  5.76981234,  5.85412093,  5.93681606,
                6.02088036,  6.10996272,  6.21190522,  6.31658428,  6.41034744,
                6.49800775,  6.58573874,  6.68761503,  6.79945406,  6.92216562,
                7.06072914,  7.20704533,  7.36422158,  7.55458991,  7.77904509,
                8.01482232,  8.31290005,  8.64146229,  9.04834077,  9.79610547,
            40.27716475])
    
    return rel_pos_new

