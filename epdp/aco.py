import torch
import numpy as np
from torch.distributions import Categorical
from two_opt import batched_lns_pdp_numba, batched_two_opt_pdp_numba
from functools import cached_property
import time


class ACO:
    def __init__(
        self,
        distances,
        node_types,
        pair_ids,
        depot_node_idx=0,
        num_pickup_nodes=0,
        num_delivery_nodes=0,
        n_ants=20,
        decay=0.9,
        alpha=1,
        beta=1,
        elitist=False,
        min_max=False,
        pheromone=None,
        heuristic=None,
        min_val=None,
        device="cpu",
        local_search=None,
        stagnation_limit=20,
        neural_lns_weight=0.5,
        time_limit_seconds=None,
    ):

        self.device = torch.device(device)
        self.EPS = 1e-9
        self.problem_size = distances.shape[0]
        self.distances = distances.to(self.device)
        self.node_types = node_types.to(self.device)
        self.pair_ids = pair_ids.to(self.device)
        self.depot_node_idx = int(depot_node_idx)
        self.num_pickup_nodes = int(num_pickup_nodes)
        self.num_delivery_nodes = int(num_delivery_nodes)

        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.elitist = elitist
        self.min_max = min_max
        self.stagnation_limit = stagnation_limit
        self.neural_lns_weight = neural_lns_weight
        self.time_limit = time_limit_seconds

        max_pair_id = (
            self.pair_ids.max().item()
            if self.pair_ids.numel() > 0 and self.pair_ids.max() > 0
            else 0
        )
        self.node_to_pair_id = torch.full(
            (self.problem_size,), 0, dtype=torch.long, device=self.device
        )
        if self.num_pickup_nodes > 0:
            pickup_nodes = (self.node_types == 1).nonzero(as_tuple=True)[0]
            delivery_nodes = (self.node_types == 2).nonzero(as_tuple=True)[0]
            self.node_to_pair_id[pickup_nodes] = self.pair_ids[pickup_nodes]
            self.node_to_pair_id[delivery_nodes] = self.pair_ids[delivery_nodes]

        if min_max:
            self.min_pheromone_val = min_val if min_val is not None else 0.1
            self.max_pheromone_val = None
        else:
            self.min_pheromone_val, self.max_pheromone_val = 1e-9, float("inf")

        if pheromone is None:
            self.pheromone = torch.full_like(self.distances, 1.0, device=self.device)
            if self.min_max:
                self.pheromone.fill_(self.min_pheromone_val)
        else:
            self.pheromone = pheromone.clone().to(self.device)

        assert local_search in [
            None,
            "none",
            "lns_pdp",
            "neural_lns_pdp",
        ], f"Unsupported LS: {local_search}"
        self.local_search_type = local_search if local_search != "none" else None

        if heuristic is None:
            self.heuristic = 1.0 / self.distances.clamp(min=self.EPS)
        else:
            self.heuristic = heuristic.clone().to(self.device)
        self.heuristic.clamp_(min=self.EPS, max=1e9)

        self.shortest_path_coords, self.lowest_cost = None, float("inf")

    @torch.no_grad()
    def sparsify(self, k_sparse):
        temp_distances = self.distances.clone()
        diag_mask = torch.eye(self.problem_size, device=self.device, dtype=torch.bool)
        temp_distances[diag_mask] = float("inf")
        k_actual = min(k_sparse, self.problem_size - 1) if self.problem_size > 1 else 0
        if k_actual > 0:
            _, topk_indices = torch.topk(
                temp_distances, k=k_actual, dim=1, largest=False
            )
            edge_index_u = torch.arange(
                self.problem_size, device=self.device
            ).repeat_interleave(k_actual)
            edge_index_v = torch.flatten(topk_indices)
            sparse_distances_heuristic = torch.full_like(self.distances, float("inf"))
            sparse_distances_heuristic[edge_index_u, edge_index_v] = self.distances[
                edge_index_u, edge_index_v
            ]
            if 0 <= self.depot_node_idx < self.problem_size:
                sparse_distances_heuristic[self.depot_node_idx, :] = self.distances[
                    self.depot_node_idx, :
                ]
                sparse_distances_heuristic[:, self.depot_node_idx] = self.distances[
                    :, self.depot_node_idx
                ]
            sparse_distances_heuristic[diag_mask] = float("inf")
            clamped_sparse_distances = sparse_distances_heuristic.clamp(min=self.EPS)
            self.heuristic = 1.0 / clamped_sparse_distances
        else:
            self.heuristic = torch.full_like(self.distances, self.EPS)
        self.heuristic.clamp_(min=self.EPS, max=1e9)

    def sample(self, inference=False):
        paths_tensor, log_probs_tensor, path_valid_mask, entropies_tensor = (
            self.gen_paths_batch(require_prob=not inference)
        )
        costs = torch.full(
            (self.n_ants,), float("inf"), device=self.device, dtype=torch.float32
        )
        if path_valid_mask.any():
            valid_paths_tensor = paths_tensor[path_valid_mask]
            if valid_paths_tensor.numel() > 0:
                costs[path_valid_mask] = self.gen_path_costs_for_batch(
                    valid_paths_tensor
                )
        return costs, log_probs_tensor, entropies_tensor, paths_tensor, path_valid_mask

    def sample_ls(self, paths_tensor_initial, path_valid_mask_initial, inference=False):
        ls_costs = torch.full(
            (self.n_ants,), float("inf"), device=self.device, dtype=torch.float32
        )
        ls_paths_final = paths_tensor_initial.clone()
        if self.local_search_type is None or not path_valid_mask_initial.any():
            if path_valid_mask_initial.any():
                valid_initial_paths = paths_tensor_initial[path_valid_mask_initial]
                if valid_initial_paths.numel() > 0:
                    ls_costs[path_valid_mask_initial] = self.gen_path_costs_for_batch(
                        valid_initial_paths
                    )
            return ls_costs, ls_paths_final

        valid_initial_paths_for_ls = paths_tensor_initial[path_valid_mask_initial]
        if valid_initial_paths_for_ls.numel() == 0:
            return ls_costs, ls_paths_final

        improved_paths_after_ls, ls_step_valid_mask = self.local_search_pdp(
            valid_initial_paths_for_ls, inference
        )

        successful_ls_mask = torch.zeros_like(path_valid_mask_initial)
        original_indices_of_valid_initial = path_valid_mask_initial.nonzero(
            as_tuple=True
        )[0]
        successful_ls_mask[original_indices_of_valid_initial[ls_step_valid_mask]] = True

        if successful_ls_mask.any():
            ls_paths_final[successful_ls_mask] = improved_paths_after_ls[
                ls_step_valid_mask
            ]
            ls_costs[successful_ls_mask] = self.gen_path_costs_for_batch(
                ls_paths_final[successful_ls_mask]
            )

        failed_ls_mask = path_valid_mask_initial & ~successful_ls_mask
        if failed_ls_mask.any():
            ls_costs[failed_ls_mask] = self.gen_path_costs_for_batch(
                paths_tensor_initial[failed_ls_mask]
            )

        return ls_costs, ls_paths_final

    @torch.no_grad()
    def run(self, n_iterations, inference=False):
        assert inference, "run() is for inference only, sets pheromone internally."

        start_time = time.time()

        if self.min_max:
            initial_ph = (
                self.min_pheromone_val if self.min_pheromone_val is not None else 0.1
            )
            self.pheromone = torch.full_like(
                self.distances, initial_ph, device=self.device
            )
        else:
            self.pheromone = torch.ones_like(self.distances, device=self.device)

        self.lowest_cost = float("inf")
        self.shortest_path_coords = None

        stagnation_counter = 0
        last_iteration_duration = 0.0

        for iter_idx in range(n_iterations):
            if self.time_limit is not None:
                elapsed_time = time.time() - start_time
                if (elapsed_time + last_iteration_duration) > self.time_limit:
                    break

            iter_start_time = time.time()

            if stagnation_counter >= self.stagnation_limit:
                break

            paths_tensor, _, path_masks_initial, _ = self.gen_paths_batch(
                require_prob=False
            )

            if self.local_search_type is not None:
                current_iter_costs, paths_after_ls = self.sample_ls(
                    paths_tensor, path_masks_initial, inference=True
                )
            else:
                paths_after_ls = paths_tensor
                current_iter_costs = torch.full(
                    (self.n_ants,),
                    float("inf"),
                    device=self.device,
                    dtype=torch.float32,
                )
                if path_masks_initial.any():
                    valid_paths = paths_tensor[path_masks_initial]
                    if valid_paths.numel() > 0:
                        current_iter_costs[path_masks_initial] = (
                            self.gen_path_costs_for_batch(valid_paths)
                        )

            iter_best_cost_tensor, iter_best_idx = torch.min(current_iter_costs, dim=0)
            iter_best_cost_val = iter_best_cost_tensor.item()

            if iter_best_cost_val < self.lowest_cost and iter_best_cost_val != float(
                "inf"
            ):
                self.lowest_cost = iter_best_cost_val
                best_path_tensor_this_iter = paths_after_ls[iter_best_idx]
                self.shortest_path_coords = (
                    best_path_tensor_this_iter[best_path_tensor_this_iter != -1]
                    .cpu()
                    .tolist()
                )
                stagnation_counter = 0
            elif iter_best_cost_val != float("inf"):
                stagnation_counter += 1

            if (
                self.min_max
                and self.lowest_cost > 0
                and self.lowest_cost != float("inf")
            ):
                actual_path_len = (
                    len(self.shortest_path_coords)
                    if self.shortest_path_coords
                    else self.problem_size
                )
                safe_actual_path_len = max(actual_path_len, 1.0)
                denominator_max_val = (1.0 - self.decay) * self.lowest_cost
                new_max_val = (
                    1.0 / denominator_max_val
                    if denominator_max_val > self.EPS
                    else 1.0 / self.EPS
                )
                if (
                    self.max_pheromone_val is None
                    or new_max_val < self.max_pheromone_val
                ):
                    self.max_pheromone_val = new_max_val
                    if self.max_pheromone_val > self.EPS:
                        denominator_min_val = 2.0 * safe_actual_path_len
                        potential_min_val = (
                            self.max_pheromone_val / denominator_min_val
                            if denominator_min_val > self.EPS
                            else self.max_pheromone_val * 0.01
                        )
                        self.min_pheromone_val = min(
                            potential_min_val, self.max_pheromone_val * 0.1
                        )
                        self.min_pheromone_val = max(self.min_pheromone_val, self.EPS)
                        self.pheromone.clamp_min_(self.min_pheromone_val)
                    else:
                        self.min_pheromone_val = self.EPS

            final_valid_mask_for_pheromone = current_iter_costs != float("inf")
            if final_valid_mask_for_pheromone.any():
                paths_for_update = paths_after_ls[final_valid_mask_for_pheromone]
                costs_for_update = current_iter_costs[final_valid_mask_for_pheromone]
                if paths_for_update.numel() > 0:
                    self.update_pheromone(paths_for_update, costs_for_update)
            else:
                self.pheromone *= self.decay
                if self.min_max:
                    self.pheromone.clamp_min_(self.min_pheromone_val)

            if self.time_limit is not None:
                last_iteration_duration = time.time() - iter_start_time

        return self.lowest_cost, self.shortest_path_coords

    @torch.no_grad()
    def update_pheromone(self, valid_paths_tensor, valid_costs_tensor):
        self.pheromone *= self.decay
        safe_costs = valid_costs_tensor.clamp(min=self.EPS)

        for i in range(valid_paths_tensor.shape[0]):
            path = valid_paths_tensor[i]
            cost = safe_costs[i]
            actual_nodes = path[path != -1]
            if len(actual_nodes) < 2:
                continue
            delta_pheromone_val = 1.0 / cost
            u, v = actual_nodes[:-1], actual_nodes[1:]
            self.pheromone[u, v] += delta_pheromone_val
            self.pheromone[v, u] += delta_pheromone_val

        if (
            self.elitist
            and self.shortest_path_coords is not None
            and self.lowest_cost > self.EPS
        ):
            elitist_path_t = torch.tensor(
                self.shortest_path_coords, dtype=torch.long, device=self.device
            )
            if len(elitist_path_t) >= 2:
                elitist_delta_ph = 1.0 / self.lowest_cost
                el_u, el_v = elitist_path_t[:-1], elitist_path_t[1:]
                self.pheromone[el_u, el_v] += elitist_delta_ph
                self.pheromone[el_v, el_u] += elitist_delta_ph

        if self.min_max:
            self.pheromone.clamp_min_(self.min_pheromone_val)
            if self.max_pheromone_val is not None:
                self.pheromone.clamp_max_(self.max_pheromone_val)

    @torch.no_grad()
    def gen_path_costs_for_batch(self, paths_batch_tensor):
        costs = torch.zeros(
            paths_batch_tensor.shape[0], device=self.device, dtype=torch.float32
        )
        for i in range(paths_batch_tensor.shape[0]):
            path = paths_batch_tensor[i]
            actual_nodes = path[path != -1]
            if len(actual_nodes) < 2:
                costs[i] = float("inf")
                continue
            u, v = actual_nodes[:-1], actual_nodes[1:]
            costs[i] = torch.sum(self.distances[u, v])
        return costs

    def gen_paths_batch(self, require_prob=False):
        if self.num_pickup_nodes == 0:
            max_steps = self.problem_size
        else:
            max_steps = self.num_pickup_nodes + self.num_delivery_nodes + 1

        paths = torch.full(
            (self.n_ants, max_steps + 1), -1, dtype=torch.long, device=self.device
        )
        log_probs = entropies = None
        if require_prob:
            log_probs = torch.zeros((self.n_ants, max_steps), device=self.device)
            entropies = torch.zeros((self.n_ants, max_steps), device=self.device)

        current_nodes = torch.full(
            (self.n_ants,), self.depot_node_idx, dtype=torch.long, device=self.device
        )
        paths[:, 0] = current_nodes

        visitable_mask = torch.ones(
            self.n_ants, self.problem_size, dtype=torch.bool, device=self.device
        )
        visitable_mask[:, self.depot_node_idx] = False

        max_pair_id = (
            self.pair_ids.max().item()
            if self.pair_ids.numel() > 0 and self.pair_ids.max() > 0
            else 0
        )
        pickups_visited_by_ant = torch.zeros(
            self.n_ants, max_pair_id + 1, dtype=torch.bool, device=self.device
        )

        num_pickups_done = torch.zeros(
            self.n_ants, dtype=torch.long, device=self.device
        )
        num_deliveries_done = torch.zeros(
            self.n_ants, dtype=torch.long, device=self.device
        )

        active_ants_mask = torch.ones(self.n_ants, dtype=torch.bool, device=self.device)
        ant_indices = torch.arange(self.n_ants, device=self.device)
        prob_mat_base = (self.pheromone.pow(self.alpha)) * (
            self.heuristic.pow(self.beta)
        )

        delivery_nodes = (self.node_types == 2).nonzero(as_tuple=True)[0]
        delivery_pair_ids = self.node_to_pair_id[delivery_nodes]

        for step in range(max_steps):
            if not active_ants_mask.any():
                break

            current_active_indices = ant_indices[active_ants_mask]
            n_active_ants = current_active_indices.size(0)

            current_probs = prob_mat_base[current_nodes[active_ants_mask]]
            dynamic_mask = visitable_mask[active_ants_mask].clone()

            all_serviced = (
                num_pickups_done[active_ants_mask] == self.num_pickup_nodes
            ) & (num_deliveries_done[active_ants_mask] == self.num_delivery_nodes)

            dynamic_mask[~all_serviced, self.depot_node_idx] = False
            dynamic_mask[all_serviced, :] = False
            dynamic_mask[all_serviced, self.depot_node_idx] = True

            if delivery_nodes.numel() > 0:
                p_visited_status = pickups_visited_by_ant[active_ants_mask].gather(
                    1, delivery_pair_ids.expand(n_active_ants, -1)
                )
                dynamic_mask[:, delivery_nodes] &= p_visited_status

            final_probs = current_probs * dynamic_mask
            row_sums = final_probs.sum(dim=1)

            is_stuck = row_sums < self.EPS
            if is_stuck.any():
                active_ants_mask[current_active_indices[is_stuck]] = False
                if not active_ants_mask.any():
                    break
                non_stuck_mask = ~is_stuck
                current_active_indices, final_probs, row_sums, all_serviced = (
                    current_active_indices[non_stuck_mask],
                    final_probs[non_stuck_mask],
                    row_sums[non_stuck_mask],
                    all_serviced[non_stuck_mask],
                )
                if final_probs.numel() == 0:
                    break

            final_probs = final_probs / row_sums.unsqueeze(-1).clamp(min=self.EPS)
            dist = Categorical(probs=final_probs)
            next_nodes = dist.sample()

            if require_prob:
                log_probs[current_active_indices, step] = dist.log_prob(next_nodes)
                entropies[current_active_indices, step] = dist.entropy()

            current_nodes[current_active_indices] = next_nodes
            paths[current_active_indices, step + 1] = next_nodes

            is_not_final_depot_move = ~(
                (next_nodes == self.depot_node_idx) & all_serviced
            )
            ants_to_update, nodes_to_update = (
                current_active_indices[is_not_final_depot_move],
                next_nodes[is_not_final_depot_move],
            )
            if ants_to_update.numel() > 0:
                visitable_mask[ants_to_update, nodes_to_update] = False

            chosen_node_types, chosen_pair_ids = (
                self.node_types[next_nodes],
                self.node_to_pair_id[next_nodes],
            )
            is_pickup = (chosen_node_types == 1) & (chosen_pair_ids > 0)
            if is_pickup.any():
                ants_p, p_ids_chosen = (
                    current_active_indices[is_pickup],
                    chosen_pair_ids[is_pickup],
                )
                pickups_visited_by_ant[ants_p, p_ids_chosen] = True
                num_pickups_done[ants_p] += 1
            is_delivery = (chosen_node_types == 2) & (chosen_pair_ids > 0)
            if is_delivery.any():
                num_deliveries_done[current_active_indices[is_delivery]] += 1

        is_finished = (num_pickups_done == self.num_pickup_nodes) & (
            num_deliveries_done == self.num_delivery_nodes
        )
        path_lengths = (paths != -1).sum(dim=1)
        last_nodes = paths[ant_indices, (path_lengths - 1).clamp(min=0)]
        is_ended = (last_nodes == self.depot_node_idx) | (path_lengths <= 1)
        final_valid_mask = (is_finished & is_ended) | (
            self.num_pickup_nodes == 0 & is_ended
        )
        final_valid_mask &= active_ants_mask

        final_entropies = entropies.sum(dim=1) if require_prob else None
        return paths, log_probs, final_valid_mask, final_entropies

    @cached_property
    def distances_numpy(self):
        return self.distances.cpu().numpy().astype(np.float32)

    @cached_property
    def heuristic_numpy(self):
        return self.heuristic.detach().cpu().numpy().astype(np.float32)

    @cached_property
    def node_types_numpy(self):
        return self.node_types.cpu().numpy().astype(np.uint8)

    @cached_property
    def pair_ids_numpy(self):
        return self.pair_ids.cpu().numpy().astype(np.uint16)

    @cached_property
    def heuristic_dist_numpy(self):
        h_np = self.heuristic_numpy
        safe_h = np.where(h_np > self.EPS, h_np, self.EPS)
        dist = 1.0 / safe_h
        dist[h_np <= self.EPS] = 1e8
        if dist.shape[0] == dist.shape[1]:
            np.fill_diagonal(dist, np.inf)
        return dist.astype(np.float32)

    def local_search_pdp(self, valid_paths_input_tensor, inference=False):
        if self.local_search_type is None or valid_paths_input_tensor.numel() == 0:
            return valid_paths_input_tensor, torch.ones(
                valid_paths_input_tensor.shape[0], dtype=torch.bool
            )

        tours_np_padded = valid_paths_input_tensor.cpu().numpy().astype(np.int16)
        dist_mat_np = self.distances_numpy
        node_types_np_ls = self.node_types_numpy
        pair_ids_np_ls = self.pair_ids_numpy
        depot_idx_np_ls = np.uint16(self.depot_node_idx)
        num_p_ls, num_d_ls = np.int64(self.num_pickup_nodes), np.int64(
            self.num_delivery_nodes
        )

        optimized_tours_np_padded = np.array([])
        ls_success_mask_np = np.array([])

        if self.local_search_type in ["lns_pdp", "neural_lns_pdp"]:
            n_remove = int(max(4, self.problem_size * 0.2))
            max_iters_lns = 100 if inference else 40

            max_iters_2opt_init = max(10, min(self.problem_size, 500))
            tours_after_2opt, valid_mask_2opt = batched_two_opt_pdp_numba(
                dist_mat_np,
                tours_np_padded,
                node_types_np_ls,
                pair_ids_np_ls,
                np.int64(max_iters_2opt_init),
            )

            optimized_tours_np_padded, ls_success_mask_np = (
                tours_after_2opt.copy(),
                valid_mask_2opt.copy(),
            )

            if np.any(valid_mask_2opt):
                tours_to_improve = tours_after_2opt[valid_mask_2opt]

                cost_matrix = (
                    self.heuristic_dist_numpy
                    if self.local_search_type == "neural_lns_pdp"
                    else dist_mat_np
                )

                improved, lns_valid = batched_lns_pdp_numba(
                    dist_mat_np,
                    tours_to_improve,
                    node_types_np_ls,
                    pair_ids_np_ls,
                    depot_idx_np_ls,
                    num_p_ls,
                    num_d_ls,
                    np.int64(max_iters_lns),
                    np.int64(n_remove),
                    cost_matrix,
                    np.float32(self.neural_lns_weight),
                )

                optimized_tours_np_padded[valid_mask_2opt] = improved
                ls_success_mask_np[valid_mask_2opt] = lns_valid

        if optimized_tours_np_padded.size > 0:
            new_paths = torch.from_numpy(optimized_tours_np_padded).to(
                self.device, dtype=torch.long
            )
            ls_valid = torch.from_numpy(ls_success_mask_np).to(
                self.device, dtype=torch.bool
            )
            return new_paths, ls_valid
        return valid_paths_input_tensor, torch.zeros_like(
            valid_paths_input_tensor, dtype=torch.bool
        )
