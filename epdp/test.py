import time
import torch
import numpy as np
import os
import argparse
import csv
import json
from tqdm import tqdm
from functools import partial
from multiprocessing import cpu_count, Pool
from net import Net
from aco import ACO
from utils import load_pdp_dataset


def evaluate_single_instance_worker(data_item, model, args, device):
    model.eval()

    start_time = time.time()

    pyg_data = data_item["pyg_data"].to(device)
    pdp_info = data_item["pdp_info"]

    with torch.no_grad():
        heuristic_vector, _ = model(pyg_data)
        heuristic_matrix = Net.reshape_to_matrix(pyg_data, heuristic_vector)

    aco_solver = ACO(
        distances=data_item["actual_distances"].to(device),
        node_types=pdp_info["node_types"].to(device),
        pair_ids=pdp_info["pair_ids"].to(device),
        depot_node_idx=pdp_info["depot_node_idx"],
        num_pickup_nodes=pdp_info["num_pickup_nodes"],
        num_delivery_nodes=pdp_info["num_delivery_nodes"],
        n_ants=args.n_ants,
        heuristic=heuristic_matrix,
        device=str(device),
        local_search=args.ls_mode,
        time_limit_seconds=args.time_limit,
        alpha=args.aco_alpha,
        beta=args.aco_beta,
        decay=args.aco_decay,
        elitist=args.elitist,
        min_max=args.aco_min_max,
        min_val=args.aco_min_val,
        stagnation_limit=args.aco_stagnation,
        neural_lns_weight=args.neural_lns_weight,
    )

    best_cost, best_path = aco_solver.run(args.fixed_aco_iters, inference=True)
    duration = time.time() - start_time

    return {
        "instance_name": data_item.get("instance_name", "N/A"),
        "best_cost": (
            best_cost
            if (best_cost != float("inf") and not np.isnan(best_cost))
            else "inf"
        ),
        "time_seconds": duration,
        "best_path_str": "->".join(map(str, best_path)) if best_path else "",
    }


def test_main(args):
    start_wall_clock = time.time()

    main_device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    print(f"\n--- Starting Evaluation on Test Dataset ---")
    print(f"Using main device: {main_device.type}")
    print(f"Parameters used: {vars(args)}")

    if not os.path.isfile(args.config_file):
        raise FileNotFoundError(f"Config file not found: {args.config_file}")
    with open(args.config_file, "r") as f:
        config = json.load(f)
    model_params = config["model_constructor_args"]
    print(f"Loaded model architecture from config.")

    model = Net(**model_params).to(main_device)
    if not os.path.isfile(args.model_file):
        raise FileNotFoundError(f"Model weights file not found: {args.model_file}")
    model.load_state_dict(torch.load(args.model_file, map_location=main_device))
    print(f"Successfully loaded model weights from: {args.model_file}")
    model.eval()

    if main_device.type == "cuda":
        model.share_memory()

    dataset = load_pdp_dataset(
        args.dataset_file,
        args.k_sparse,
        torch.device("cpu"),
        model_params["node_feat_dim"],
    )
    instances_to_run = (
        dataset[: args.num_test_instances] if args.num_test_instances else dataset
    )
    print(f"Number of instances to evaluate: {len(instances_to_run)}")

    num_workers = min(args.num_workers, cpu_count())
    print(f"Using {num_workers} parallel processes for evaluation...")

    worker_func = partial(
        evaluate_single_instance_worker, model=model, args=args, device=main_device
    )

    all_results = []
    if num_workers > 0 and len(instances_to_run) > 0:
        with Pool(processes=num_workers) as pool:
            results_iterator = pool.imap(worker_func, instances_to_run)
            all_results = list(
                tqdm(
                    results_iterator,
                    total=len(instances_to_run),
                    desc="Evaluating in parallel",
                )
            )
    elif len(instances_to_run) > 0:
        all_results = [
            worker_func(item)
            for item in tqdm(instances_to_run, desc="Evaluating sequentially")
        ]

    wall_clock_duration = time.time() - start_wall_clock

    valid_costs = [r["best_cost"] for r in all_results if r["best_cost"] != "inf"]
    print(f"\n--- Evaluation Summary on {len(instances_to_run)} Instances ---")

    print(f"Total wall-clock execution time: {wall_clock_duration:.2f}s")

    if all_results:
        total_solver_time = sum(r["time_seconds"] for r in all_results)
        avg_time_per_instance = total_solver_time / len(all_results)
        print(f"Average solving time per instance: {avg_time_per_instance:.3f}s")

    if valid_costs:
        print(f"Solved instances: {len(valid_costs)}/{len(instances_to_run)}")
        print(f"   - Average cost: {np.mean(valid_costs):.4f}")
    else:
        print("No instances were solved successfully.")

    if args.results_csv:
        base, ext = os.path.splitext(args.results_csv)
        csv_filename = f"{base}_{args.ls_mode or 'none'}.csv"
        with open(csv_filename, "w", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=[
                    "instance_name",
                    "best_cost",
                    "time_seconds",
                    "best_path_str",
                ],
            )
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Detailed results saved to: {csv_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DeepACO model for PDP."
    )
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("-m", "--model_file", type=str, required=True)
    parser.add_argument("-c", "--config_file", type=str, required=True)
    parser.add_argument("--k_sparse", type=int, default=70)
    parser.add_argument("--num_test_instances", type=int, default=None)
    parser.add_argument("--n_ants", type=int, default=40)
    parser.add_argument("--fixed_aco_iters", type=int, default=50)
    parser.add_argument("--aco_stagnation", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--results_csv", type=str, default="pdp_test_results.csv")
    parser.add_argument(
        "--time_limit",
        type=float,
        default=None,
        help="Time limit for solving each instance (in seconds). Default is no limit.",
    )
    parser.add_argument(
        "--ls_mode",
        type=str,
        default="neural_lns_pdp",
        choices=["none", "lns_pdp", "neural_lns_pdp"],
    )
    parser.add_argument("--aco_beta", type=float, default=2.0)
    parser.add_argument("--aco_alpha", type=float, default=1.0)
    parser.add_argument("--aco_decay", type=float, default=0.9)
    parser.add_argument(
        "--elitist", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--aco_min_max", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--aco_min_val", type=float, default=None)
    parser.add_argument(
        "--neural_lns_weight",
        type=float,
        default=1.0,
        help="Weight for the real cost in Neural LNS (w). 1.0 = classical LNS.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, cpu_count() // 2),
        help="Number of parallel processes for evaluation.",
    )

    args = parser.parse_args()
    test_main(args)
