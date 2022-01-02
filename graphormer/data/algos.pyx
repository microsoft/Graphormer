# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# distutils: language = c++

import cython
from cython.parallel cimport prange, parallel
cimport numpy
import numpy

cdef floyd_warshall(adjacency_matrix):

    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    cdef unsigned int n = nrows

    adj_mat_copy = adjacency_matrix.astype(long, order='C', casting='safe', copy=True)
    assert adj_mat_copy.flags['C_CONTIGUOUS']
    cdef numpy.ndarray[long, ndim=2, mode='c'] M = adj_mat_copy
    cdef numpy.ndarray[long, ndim=2, mode='c'] path = - numpy.ones([n, n], dtype=numpy.int64)

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


cdef get_all_edges(path, i, j):
    cdef int k = path[i][j]
    if k == -1:
        return []
    else:
        return get_all_edges(path, i, k) + [k] + get_all_edges(path, k, j)

def fw_spatial_pos_and_edge_input(adj, edge_feat, max_dist=5):

    shortest_path_result, path = floyd_warshall(adj)

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
            num_path = min(len(path) - 1, max_dist_copy)
            for k in range(num_path):
                edge_fea_all[i, j, k, :] = edge_feat_copy[path[k], path[k+1], :]

    return shortest_path_result, edge_fea_all

from libcpp.vector cimport vector
from libcpp.queue cimport queue
import numpy as np
cimport numpy as np

cdef inline reverse_path(vector[int] path_):
    cdef:
        int i
        int n = path_.size()
    for i in range(n//2):
        path_[i], path_[n-i-1] = path_[n-i-1], path_[i]
    return path_

cdef bfs_shortest_path(vector[vector[int]] adj_list, int startVertex):

    cdef:
        int n = adj_list.size()
        unsigned size, vertex, adjVertex, idx
        np.int64_t[:] path = np.full(n, -1, dtype="int64")
        np.int64_t[:] dist = np.full(n, -1, dtype="int64")
        queue[int] q = queue[int]()
        vector[int] adjVertices

    dist[startVertex] = 0
    path[startVertex] = startVertex
    q.push(startVertex)

    while not q.empty():
        size = q.size()
        while size > 0:
            size -= 1
            vertex = q.front()
            q.pop()
            adjVertices = adj_list[vertex]
            for idx in range(adjVertices.size()):
                if dist[adjVertices[idx]] == -1:
                    dist[adjVertices[idx]] = dist[vertex] + 1
                    path[adjVertices[idx]] = vertex
                    q.push(adjVertices[idx])

    return dist, path

cdef get_full_path(
    np.int64_t[:] path,
    np.int64_t[:, :, :] edge_type,
    int max_dist,
    int cur_node
):

    cdef:
        unsigned i, j, cur
        int n = path.shape[0]
        int size = edge_type.shape[2]
        np.int64_t[:, :, :] edge_input = np.full(
            shape=(n, max_dist, size),
            fill_value=-1,
            dtype="int64"
        )
        vector[int] path_

    for i in range(n):
        if i == cur_node:
            continue
        path_ = vector[int]()
        if path[i] == -1:
            continue
        path_.push_back(i)
        cur = i
        while path[cur] != cur_node:
            path_.push_back(path[cur])
            cur = path[cur]
        path_.push_back(cur_node)
        path_ = reverse_path(path_)
        for j in range(min(max_dist, path_.size() - 1)):
            edge_input[i, j, :] = edge_type[path_[j], path_[j+1], :]

    return edge_input

def bfs_spatial_pos_and_edge_input(
    np.int64_t[:, :] adj_matrix,
    np.int64_t[:, :, :] edge_type,
    int max_dist=5
):

    cdef:
        int i, j
        int n = adj_matrix.shape[0]
        int edge_type_shape = edge_type.shape[2]
        np.ndarray[np.int64_t, ndim=4, mode='c'] edge_input = np.full(
            shape=(n, n, max_dist, edge_type_shape),
            fill_value=-1,
            dtype="int64"
        )
        np.int64_t[:, :] spatial_pos = np.full((n ,n), 510, dtype="int64")
        cdef vector[vector[int]] adj_list

    for i in range(n):
        adj_list.push_back(vector[int]())
        for j in range(n):
            if adj_matrix[i][j] == 1:
                adj_list[i].push_back(j)

    for i in range(n):
        dist, path = bfs_shortest_path(adj_list, i)
        edge_input[i] = np.asarray(
            get_full_path(
                path, edge_type, max_dist, i
            )
        )
        for j in range(n):
            if dist[j] != -1:
                spatial_pos[i, j] = dist[j]

    return np.asarray(spatial_pos), np.asarray(edge_input)

def bfs_target_spatial_pos_and_edge_input(
    np.int64_t[:, :] adj_matrix,
    np.int64_t[:, :, :] edge_type,
    int max_dist=5,
):

    cdef:
        int i, j
        int n = adj_matrix.shape[0]
        int edge_type_shape = edge_type.shape[2]
        np.ndarray[np.int64_t, ndim=4, mode='c'] edge_input = np.full(
            shape=(n, n, max_dist, edge_type_shape),
            fill_value=-1,
            dtype="int64"
        )
        np.int64_t[:, :] spatial_pos = np.full((n ,n), 510, dtype="int64")
        cdef vector[vector[int]] adj_list

    for i in range(n):
        adj_list.push_back(vector[int]())
        for j in range(i+1):
            if adj_matrix[i][j] == 1:
                adj_list[i].push_back(j)
        for j in range(i):
            if adj_matrix[j][i] == 1:
                adj_list[j].push_back(i)
        dist, path = bfs_shortest_path(adj_list, i)
        edge_input[i, :i+1, :, :] = np.asarray(get_full_path(
            path, edge_type[:i+1, :i+1, :], max_dist, i))
        for j in range(i+1):
            if dist[j] != -1:
                spatial_pos[i, j] = dist[j]

    return np.asarray(spatial_pos), np.asarray(edge_input)
