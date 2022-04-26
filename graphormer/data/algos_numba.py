from numba import njit, prange
import numpy as np

@njit
def bfs_shortest_path(adj_list, startVertex):

    n = adj_list.shape[0]
    path = np.full(n, -1)
    dist = np.full(n, -1)
    q = np.full(n, -1)
    q_start, q_end = -1, 0

    dist[startVertex] = 0
    path[startVertex] = startVertex
    q[0] = startVertex

    while q_start != q_end:
        q_start += 1
        vertex = q[q_start]
        adjVertices = adj_list[vertex]
        cur = 0
        while adjVertices[cur] != -1:
            val = adjVertices[cur]
            cur += 1
            if dist[val] == -1:
                q_end += 1
                q[q_end] = val
                dist[val] = dist[vertex] + 1
                path[val] = vertex

    return dist, path

@njit
def get_full_path(path, edge_type, max_dist, cur_node):

    n = path.shape[0]
    size = edge_type.shape[2]
    edge_input = np.full(
        shape=(n, max_dist, size),
        fill_value=-1,
        dtype="int64"
    )
    path_ = np.full(n, -1)

    for i in range(n):
        cur_path = 0
        if i == cur_node:
            continue
        if path[i] == -1:
            continue
        path_[cur_path] = i
        cur_path += 1
        cur = i
        while path[cur] != cur_node:
            path_[cur_path] = path[cur]
            cur_path += 1
            cur = path[cur]
        path_[cur_path] = cur_node
        for j in range(min(max_dist, cur_path)):
            edge_input[i, j, :] = edge_type[
                path_[cur_path-j], path_[cur_path-j-1], :]

    return edge_input

@njit(parallel=True)
def bfs_numba_spatial_pos_and_edge_input(
    adj_matrix,
    edge_type,
    max_dist=5,
):
    n = adj_matrix.shape[0]
    edge_type_shape = edge_type.shape[2]
    edge_input = np.full(
        shape=(n, n, max_dist, edge_type_shape),
        fill_value=-1,
    )
    spatial_pos = np.full((n ,n), 510)
    adj_list = np.full((n ,n), -1)

    for i in range(n):
        cur = 0
        for j in range(n):
            if adj_matrix[i, j] == 1:
                adj_list[i, cur] = j
                cur += 1

    for i in prange(n):
        dist, path = bfs_shortest_path(adj_list, i)
        edge_input[i] = get_full_path(
            path, edge_type, max_dist, i
        )
        for j in range(n):
            if dist[j] != -1:
                spatial_pos[i, j] = dist[j]

    return spatial_pos, edge_input
