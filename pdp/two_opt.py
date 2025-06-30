import numpy as np
import numba as nb
from numba.typed import List


@nb.njit(nogil=True, cache=True)
def _check_full_pdp_tour_constraints_numba(
    tour_np, node_types_np, pair_ids_np, depot_idx, n_pickups, n_dels
):
    tour_len = len(tour_np)
    if tour_len < 2:
        return False
    if tour_np[0] != depot_idx or tour_np[-1] != depot_idx:
        return False

    if n_pickups == 0:
        return tour_len == 2

    customers = tour_np[1:-1]
    if len(np.unique(customers)) != len(customers):
        return False
    if len(customers) != n_pickups + n_dels:
        return False

    max_pid = (
        np.max(pair_ids_np) if len(pair_ids_np) > 0 and np.any(pair_ids_np > 0) else 0
    )
    if max_pid == 0:
        return False

    visited_pickups_flags = np.zeros(max_pid + 1, dtype=np.bool_)
    for node in tour_np:
        p_id = pair_ids_np[node]
        if p_id > 0:
            if node_types_np[node] == 1:
                visited_pickups_flags[p_id] = True
            elif node_types_np[node] == 2:
                if not visited_pickups_flags[p_id]:
                    return False
    return True


@nb.njit(nb.float32(nb.uint16[:], nb.float32[:, :]), nogil=True, cache=True)
def _calculate_cost_single_numba(tour_np, distmat_np):
    cost = np.float32(0.0)
    for i in range(len(tour_np) - 1):
        cost += distmat_np[tour_np[i], tour_np[i + 1]]
    return cost


@nb.njit(nogil=True, cache=True)
def _shaw_ruin_by_pair_numba(tour, dists, num_pairs_to_remove, p_to_d_map, depot_idx):
    customers = tour[1:-1]
    if len(customers) < 2:
        return tour, np.empty((0, 2), dtype=np.uint16)

    tour_pairs_list = List()
    for node in customers:
        if p_to_d_map[node] != -1:
            tour_pairs_list.append((np.uint16(node), np.uint16(p_to_d_map[node])))

    if len(tour_pairs_list) <= num_pairs_to_remove:
        return tour, np.empty((0, 2), dtype=np.uint16)

    tour_list = List(customers)
    removed_pairs = List()

    seed_idx = np.random.randint(0, len(tour_pairs_list))
    seed_p, seed_d = tour_pairs_list.pop(seed_idx)
    removed_pairs.append((seed_p, seed_d))
    tour_list.remove(seed_p)
    tour_list.remove(seed_d)

    while len(removed_pairs) < num_pairs_to_remove and len(tour_pairs_list) > 0:
        rand_p, _ = removed_pairs[np.random.randint(0, len(removed_pairs))]
        most_related_pair_idx, max_relatedness = -1, -1.0
        for i in range(len(tour_pairs_list)):
            p, _ = tour_pairs_list[i]
            relatedness = 1.0 / (dists[rand_p, p] + 1e-9)
            if relatedness > max_relatedness:
                max_relatedness, most_related_pair_idx = relatedness, i
        if most_related_pair_idx != -1:
            p_rem, d_rem = tour_pairs_list.pop(most_related_pair_idx)
            removed_pairs.append((p_rem, d_rem))
            tour_list.remove(p_rem)
            tour_list.remove(d_rem)

    ruined_tour_arr = np.empty(len(tour_list) + 2, dtype=np.uint16)
    ruined_tour_arr[0], ruined_tour_arr[-1] = depot_idx, depot_idx
    if len(tour_list) > 0:
        for i, node in enumerate(tour_list):
            ruined_tour_arr[i + 1] = node

    removed_arr = np.empty((len(removed_pairs), 2), dtype=np.uint16)
    for i in range(len(removed_pairs)):
        removed_arr[i] = removed_pairs[i]
    return ruined_tour_arr, removed_arr


@nb.njit(nogil=True, cache=True)
def _greedy_insertion_by_pair_numba(
    partial_tour, pairs_to_insert, cost_matrix, depot_idx
):
    tour_list = List(partial_tour)
    for p_node, d_node in pairs_to_insert:
        best_cost_delta, best_p_pos, best_d_pos = np.inf, -1, -1
        for i in range(1, len(tour_list) + 1):
            u_p = tour_list[i - 1]
            v_p = tour_list[i] if i < len(tour_list) else depot_idx
            cost_delta_p = (
                cost_matrix[u_p, p_node]
                + cost_matrix[p_node, v_p]
                - cost_matrix[u_p, v_p]
            )
            tour_list.insert(i, p_node)
            for j in range(i + 1, len(tour_list) + 1):
                u_d = tour_list[j - 1]
                v_d = tour_list[j] if j < len(tour_list) else depot_idx
                cost_delta_d = (
                    cost_matrix[u_d, d_node]
                    + cost_matrix[d_node, v_d]
                    - cost_matrix[u_d, v_d]
                )
                total_delta = cost_delta_p + cost_delta_d
                if total_delta < best_cost_delta:
                    best_cost_delta, best_p_pos, best_d_pos = total_delta, i, j
            tour_list.pop(i)
        if best_p_pos != -1:
            tour_list.insert(best_p_pos, p_node)
            tour_list.insert(best_d_pos, d_node)
    final_tour_arr = np.empty(len(tour_list), dtype=np.uint16)
    for i, node in enumerate(tour_list):
        final_tour_arr[i] = node
    return final_tour_arr


@nb.njit(nogil=True, cache=True)
def _lns_solver_single_tour(
    tour,
    distmat_np,
    node_types_np,
    pair_ids_np,
    depot_idx,
    n_pickups,
    n_dels,
    max_iter,
    n_remove_percent,
    cost_matrix,
    lns_w,
):
    if not _check_full_pdp_tour_constraints_numba(
        tour, node_types_np, pair_ids_np, depot_idx, n_pickups, n_dels
    ):
        return tour, False

    best_tour, current_tour = tour.copy(), tour.copy()
    best_cost = _calculate_cost_single_numba(best_tour, distmat_np)
    current_cost = best_cost

    max_pid = (
        np.max(pair_ids_np) if len(pair_ids_np) > 0 and np.any(pair_ids_np > 0) else 0
    )
    p_to_d_map = np.full(len(node_types_np), -1, dtype=np.int32)
    if max_pid > 0:
        temp_map_d = np.full(max_pid + 1, -1, dtype=np.int32)
        for i in range(len(node_types_np)):
            if node_types_np[i] == 2:
                temp_map_d[pair_ids_np[i]] = i
        for i in range(len(node_types_np)):
            if node_types_np[i] == 1:
                p_to_d_map[i] = temp_map_d[pair_ids_np[i]]

    temperature = np.float32(best_cost * 0.05) if best_cost > 1e-5 else np.float32(0.5)
    cooling_rate = np.float32(0.995)

    num_pairs_to_remove = int(max(1.0, n_pickups * n_remove_percent))

    for iter_num in range(max_iter):
        if num_pairs_to_remove == 0:
            break

        destroy_ratio = n_remove_percent * (1 - iter_num / max_iter)
        current_num_remove = int(max(1.0, n_pickups * destroy_ratio))

        ruined_tour, removed_pairs = _shaw_ruin_by_pair_numba(
            current_tour, distmat_np, current_num_remove, p_to_d_map, depot_idx
        )
        if len(removed_pairs) == 0:
            continue

        recreate_cost_matrix = lns_w * distmat_np + (1.0 - lns_w) * cost_matrix
        new_tour = _greedy_insertion_by_pair_numba(
            ruined_tour, removed_pairs, recreate_cost_matrix, depot_idx
        )

        if _check_full_pdp_tour_constraints_numba(
            new_tour, node_types_np, pair_ids_np, depot_idx, n_pickups, n_dels
        ):
            new_cost = _calculate_cost_single_numba(new_tour, distmat_np)
            if new_cost < best_cost:
                best_cost, best_tour = new_cost, new_tour.copy()

            cost_diff = new_cost - current_cost
            if cost_diff < 0 or (
                temperature > 1e-9
                and np.random.random() < np.exp(-cost_diff / temperature)
            ):
                current_cost, current_tour = new_cost, new_tour.copy()

        temperature *= cooling_rate
    return best_tour, True


@nb.njit(
    nb.types.Tuple((nb.int16[:, :], nb.boolean[:]))(
        nb.float32[:, :],
        nb.int16[:, :],
        nb.uint8[:],
        nb.uint16[:],
        nb.uint16,
        nb.int64,
        nb.int64,
        nb.int64,
        nb.int64,
        nb.float32[:, :],
        nb.float32,
    ),
    nogil=True,
    cache=True,
    parallel=True,
)
def batched_lns_pdp_numba(
    distmat_np,
    tours_np_padded,
    node_types_np,
    pair_ids_np,
    depot_idx,
    n_pickups,
    n_dels,
    max_iter,
    n_remove_nodes,
    cost_matrix,
    lns_w,
):
    num_tours = tours_np_padded.shape[0]
    final_tours = np.copy(tours_np_padded)
    final_mask = np.ones(num_tours, dtype=np.bool_)

    n_remove_percent = (
        (n_remove_nodes / (n_pickups + n_dels)) * 100.0
        if (n_pickups + n_dels) > 0
        else 0.0
    )

    for i in nb.prange(num_tours):
        tour = tours_np_padded[i][tours_np_padded[i] != -1].astype(np.uint16)

        if len(tour) > 2:
            best_tour, is_valid_run = _lns_solver_single_tour(
                tour,
                distmat_np,
                node_types_np,
                pair_ids_np,
                depot_idx,
                n_pickups,
                n_dels,
                max_iter,
                n_remove_percent,
                cost_matrix,
                lns_w,
            )
            if is_valid_run:
                final_tours[i, :] = -1
                final_tours[i, : len(best_tour)] = best_tour.astype(np.int16)
            else:
                final_mask[i] = False
        else:
            final_mask[i] = _check_full_pdp_tour_constraints_numba(
                tour, node_types_np, pair_ids_np, depot_idx, n_pickups, n_dels
            )
    return final_tours, final_mask


@nb.njit(
    nb.types.Tuple((nb.int16[:, :], nb.boolean[:]))(
        nb.float32[:, :], nb.int16[:, :], nb.uint8[:], nb.uint16[:], nb.int64
    ),
    nogil=True,
    cache=True,
    parallel=True,
)
def batched_two_opt_pdp_numba(dist, tours, types, pids, max_iter):
    n_tours = tours.shape[0]
    opt_tours = np.copy(tours)
    valid_mask = np.ones(n_tours, dtype=np.bool_)
    n_p, n_d = np.sum(types == 1), np.sum(types == 2)
    depot_idx = np.uint16(0)
    for i in nb.prange(n_tours):
        tour = tours[i][tours[i] != -1].astype(np.uint16)
        if len(tour) > 2:
            if _check_full_pdp_tour_constraints_numba(
                tour, types, pids, depot_idx, n_p, n_d
            ):
                for _ in range(max_iter):
                    improved = False
                    for i_ in range(1, len(tour) - 2):
                        for j_ in range(i_ + 1, len(tour) - 1):
                            new_tour = tour.copy()
                            new_tour[i_ : j_ + 1] = tour[j_ : i_ - 1 : -1]
                            if _check_full_pdp_tour_constraints_numba(
                                new_tour, types, pids, depot_idx, n_p, n_d
                            ):
                                cost_old = (
                                    dist[tour[i_ - 1], tour[i_]]
                                    + dist[tour[j_], tour[j_ + 1]]
                                )
                                cost_new = (
                                    dist[new_tour[i_ - 1], new_tour[i_]]
                                    + dist[new_tour[j_], new_tour[j_ + 1]]
                                )
                                if cost_new < cost_old:
                                    tour = new_tour
                                    improved = True
                    if not improved:
                        break
                opt_tours[i, :] = -1
                opt_tours[i, : len(tour)] = tour.astype(np.int16)

        final_tour_check = opt_tours[i][opt_tours[i] != -1].astype(np.uint16)
        if not _check_full_pdp_tour_constraints_numba(
            final_tour_check, types, pids, depot_idx, n_p, n_d
        ):
            valid_mask[i] = False
    return opt_tours, valid_mask
