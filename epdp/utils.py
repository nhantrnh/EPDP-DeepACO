import os
import torch
from torch_geometric.data import Data


def normalize_coordinates_instance_locally(original_coordinates: torch.Tensor):
    if original_coordinates.ndim != 2 or original_coordinates.shape[1] != 2:
        raise ValueError("Invalid coordinates shape")
    if original_coordinates.shape[0] == 0:
        return original_coordinates
    min_vals, max_vals = (
        torch.min(original_coordinates, dim=0)[0],
        torch.max(original_coordinates, dim=0)[0],
    )
    scale = max_vals - min_vals
    scale[scale < 1e-9] = 1.0
    return (original_coordinates - min_vals) / scale


def calculate_distance_matrix(coordinates_tensor: torch.Tensor):
    return torch.cdist(coordinates_tensor, coordinates_tensor, p=2)


def _create_pyg_from_pdp_data(
    original_coordinates: torch.Tensor,
    pd_pairs_list: list,
    demands_tensor_original: torch.Tensor,
    time_windows_original: torch.Tensor,
    depot_node_idx: int,
    k_sparse: int,
    device: torch.device,
    node_feat_dim: int,
):
    n_nodes = original_coordinates.shape[0]
    demands = demands_tensor_original.to(device)
    time_windows = time_windows_original.to(device)
    normalized_coords = normalize_coordinates_instance_locally(original_coordinates).to(
        device
    )

    node_input_features = torch.zeros(
        (n_nodes, node_feat_dim), device=device, dtype=torch.float32
    )

    if node_feat_dim >= 8:
        node_input_features[:, :2] = normalized_coords
        node_input_features[depot_node_idx, 2] = 1.0

        is_pickup_mask = demands > 0
        is_delivery_mask = demands < 0
        node_input_features[is_pickup_mask, 3] = 1.0
        node_input_features[is_delivery_mask, 4] = 1.0

        capacity = demands[is_pickup_mask].sum().item()
        if capacity <= 0:
            capacity = 200.0
        node_input_features[:, 5] = demands / capacity

        max_tw_val = time_windows[:, 1].max()
        if max_tw_val > 0:
            node_input_features[:, 6] = time_windows[:, 0] / max_tw_val
            node_input_features[:, 7] = time_windows[:, 1] / max_tw_val

    node_types_tensor = torch.zeros(n_nodes, dtype=torch.long, device=device)
    node_types_tensor[demands > 0] = 1
    node_types_tensor[demands < 0] = 2

    pair_ids_tensor = torch.zeros(n_nodes, dtype=torch.long, device=device)
    num_actual_pickups = 0
    current_pair_id_counter = 1

    for node1, node2 in pd_pairs_list:
        p_id, d_id = -1, -1
        if demands_tensor_original[node1] > 0 and demands_tensor_original[node2] < 0:
            p_id, d_id = node1, node2
        elif demands_tensor_original[node2] > 0 and demands_tensor_original[node1] < 0:
            p_id, d_id = node2, node1
        else:
            continue
        if p_id != -1 and pair_ids_tensor[p_id] == 0 and pair_ids_tensor[d_id] == 0:
            pair_ids_tensor[p_id] = current_pair_id_counter
            pair_ids_tensor[d_id] = current_pair_id_counter
            current_pair_id_counter += 1
            num_actual_pickups += 1

    distances_from_normalized = calculate_distance_matrix(normalized_coords)
    temp_distances_for_knn = distances_from_normalized.clone().fill_diagonal_(
        float("inf")
    )
    actual_k_sparse = min(k_sparse, n_nodes - 1) if n_nodes > 1 else 0

    if actual_k_sparse > 0:
        _, topk_indices = torch.topk(
            temp_distances_for_knn, k=actual_k_sparse, dim=1, largest=False
        )
        edge_index_u = torch.arange(n_nodes, device=device).repeat_interleave(
            actual_k_sparse
        )
        edge_index_v = torch.flatten(topk_indices)
        edge_index_for_gnn = torch.stack([edge_index_u, edge_index_v])
        edge_attr_for_gnn = distances_from_normalized[
            edge_index_u, edge_index_v
        ].unsqueeze(1)
    else:
        edge_index_for_gnn, edge_attr_for_gnn = torch.empty(
            (2, 0), dtype=torch.long, device=device
        ), torch.empty((0, 1), dtype=torch.float32, device=device)

    pyg_data_object = Data(
        x=node_input_features,
        edge_index=edge_index_for_gnn,
        edge_attr=edge_attr_for_gnn,
    )
    actual_full_distance_matrix = calculate_distance_matrix(
        original_coordinates.to(device)
    )

    pdp_info = {
        "node_types": node_types_tensor,
        "pair_ids": pair_ids_tensor,
        "depot_node_idx": depot_node_idx,
        "num_pickup_nodes": num_actual_pickups,
        "num_delivery_nodes": num_actual_pickups,
        "problem_size": n_nodes,
        "pd_pairs": pd_pairs_list,
        "demands": demands_tensor_original,
        "time_windows": time_windows_original,
    }
    return pyg_data_object, actual_full_distance_matrix, pdp_info


def load_pdp_dataset(
    filepath: str, k_sparse: int, device: torch.device, node_feat_dim_fixed: int
):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Datafile not found: {filepath}")
    all_loaded_instances_cpu = torch.load(filepath, map_location="cpu")
    processed_dataset = []
    for i, data_dict in enumerate(all_loaded_instances_cpu):
        pyg_data, dists, pdp_info = _create_pyg_from_pdp_data(
            data_dict["coordinates"],
            data_dict["pd_pairs"],
            data_dict["demands"],
            data_dict.get(
                "time_windows",
                torch.zeros_like(data_dict["demands"]).unsqueeze(-1).repeat(1, 2),
            ),
            data_dict["depot_node_idx"],
            k_sparse,
            device,
            node_feat_dim_fixed,
        )
        processed_dataset.append(
            {
                "pyg_data": pyg_data,
                "actual_distances": dists,
                "pdp_info": pdp_info,
                "instance_name": data_dict.get("instance_name", f"instance_{i}"),
            }
        )
    return processed_dataset


def validate_pdp_solution(
    solution_path: list[int], instance_info: dict
) -> tuple[bool, str]:
    if not solution_path or len(solution_path) < 2:
        return False, "Error: Tour is too short or empty."
    depot = instance_info["depot_node_idx"]
    demands = instance_info["demands"]
    pd_pairs = instance_info.get("pd_pairs", [])
    if not pd_pairs:
        is_valid = solution_path == [depot, depot] or solution_path == [depot]
        return is_valid, (
            "Valid: Only depot exists."
            if is_valid
            else "Error: Incorrect tour for no-customer case."
        )
    if solution_path[0] != depot or solution_path[-1] != depot:
        return False, f"Error: Does not start/end at depot {depot}."
    customer_nodes_in_path = solution_path[1:-1]
    visited_customers = set(customer_nodes_in_path)
    if len(customer_nodes_in_path) != len(visited_customers):
        from collections import Counter

        counts = Counter(customer_nodes_in_path)
        duplicates = [item for item, count in counts.items() if count > 1]
        return False, f"Error: Customer node is repeated: {duplicates}."
    all_required_customers = set()
    true_pd_pairs = []
    for n1, n2 in pd_pairs:
        p, d = (-1, -1)
        if demands[n1] > 0 and demands[n2] < 0:
            p, d = n1, n2
        elif demands[n2] > 0 and demands[n1] < 0:
            p, d = n2, n1
        if p != -1:
            all_required_customers.add(p)
            all_required_customers.add(d)
            true_pd_pairs.append((p, d))
    if visited_customers != all_required_customers:
        missing = sorted(list(all_required_customers - visited_customers))
        extra = sorted(list(visited_customers - all_required_customers))
        return (
            False,
            f"Error: Customer set mismatch. Missing: {missing}, Extra: {extra}.",
        )
    node_positions = {node: i for i, node in enumerate(solution_path)}
    for p_node, d_node in true_pd_pairs:
        if node_positions[p_node] >= node_positions[d_node]:
            return (
                False,
                f"Error: Precedence constraint violated. Pickup {p_node} (pos {node_positions[p_node]}) is not before Delivery {d_node} (pos {node_positions[d_node]}).",
            )
    return True, "Valid tour (P-D constraints only)."
