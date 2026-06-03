"""TSP-based frontier ordering baseline, replicating FUEL/TARE's approach.

FUEL: Global ATSP (LKH) → local Dijkstra refinement
TARE: Global TSP (OR-Tools) → local greedy + TSP

This implementation: Nearest-Neighbor + 2-opt TSP on Euclidean distance matrix,
pick the first frontier in the tour. Re-solves every step (replanning).
"""
import numpy as np


def _build_cost_matrix(vp_positions):
    n = len(vp_positions)
    if n == 0:
        return np.zeros((0, 0))
    diff = vp_positions[:, None, :] - vp_positions[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def _nearest_neighbor_tour(dist, start=0):
    n = len(dist)
    visited = [False] * n
    tour = [start]
    visited[start] = True
    for _ in range(n - 1):
        cur = tour[-1]
        best = -1
        best_d = np.inf
        for j in range(n):
            if not visited[j] and dist[cur, j] < best_d:
                best_d = dist[cur, j]
                best = j
        tour.append(best)
        visited[best] = True
    return tour


def _2opt_improve(dist, tour, max_iter=200):
    n = len(tour)
    if n <= 3:
        return tour
    tour = list(tour)

    def tour_len(t):
        return sum(dist[t[i], t[(i + 1) % n]] for i in range(n))

    best_len = tour_len(tour)
    for _ in range(max_iter):
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue
                d_old = dist[tour[i - 1], tour[i]] + dist[tour[j], tour[(j + 1) % n]]
                d_new = dist[tour[i - 1], tour[j]] + dist[tour[i], tour[(j + 1) % n]]
                if d_new < d_old - 1e-9:
                    tour[i:j + 1] = reversed(tour[i:j + 1])
                    improved = True
        if not improved:
            break
    return tour


def solve_tsp_nn_2opt(vp_positions):
    if len(vp_positions) <= 1:
        return list(range(len(vp_positions)))
    dist = _build_cost_matrix(vp_positions)
    tour = _nearest_neighbor_tour(dist, start=0)
    if len(tour) > 3:
        tour = _2opt_improve(dist, tour)
    return tour


def tsp_policy(obs, n_valid):
    frontiers = obs["frontiers"][:n_valid]
    if n_valid <= 1:
        return 0
    vp_positions = frontiers[:, 0:3] * 10.0
    dists_to_robot = frontiers[:, 4] * 15.0
    dist = _build_cost_matrix(vp_positions)
    augmented = np.zeros((n_valid + 1, n_valid + 1))
    augmented[1:, 1:] = dist
    augmented[0, 1:] = dists_to_robot
    augmented[1:, 0] = 0.0
    start = 0
    tour = _nearest_neighbor_tour(augmented, start=start)
    if len(tour) > 3:
        tour = _2opt_improve(augmented, tour, max_iter=100)
    for node in tour:
        if node != 0:
            return node - 1
    return 0


def tsp_orienteering_policy(obs, n_valid):
    frontiers = obs["frontiers"][:n_valid]
    if n_valid <= 1:
        return 0
    vp_positions = frontiers[:, 0:3] * 10.0
    sizes = frontiers[:, 3] * 2000.0
    dists = frontiers[:, 4] * 15.0
    scores = sizes / (dists + 1.0)
    return int(np.argmax(scores))


def tsp_fuel_policy(obs, n_valid):
    frontiers = obs["frontiers"][:n_valid]
    if n_valid <= 1:
        return 0

    vp_positions = frontiers[:, 0:3] * 10.0
    visib = frontiers[:, 5] * 100.0
    dists = frontiers[:, 4] * 15.0
    dist = _build_cost_matrix(vp_positions)

    augmented = np.zeros((n_valid + 1, n_valid + 1))
    augmented[1:, 1:] = dist
    augmented[0, 1:] = dists
    augmented[1:, 0] = 0.0

    tour = _nearest_neighbor_tour(augmented, start=0)
    if len(tour) > 3:
        tour = _2opt_improve(augmented, tour, max_iter=150)

    for node in tour:
        if node != 0:
            idx = node - 1
            break
    else:
        idx = 0

    top_k = min(3, n_valid)
    candidates = []
    order_idx = 1
    for node in tour[1:]:
        fi = node - 1
        if fi >= n_valid:
            continue
        candidates.append((fi, order_idx, visib[fi]))
        order_idx += 1
        if len(candidates) >= top_k:
            break

    if candidates:
        best_local = max(candidates, key=lambda x: x[2])
        return best_local[0]
    return idx
