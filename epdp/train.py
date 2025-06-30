import torch
import torch.nn.functional as F
import numpy as np
import os
import random
import argparse
import json
import csv
from tqdm import tqdm
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch
from multiprocessing import cpu_count, Pool
from functools import partial
from net import Net
from aco import ACO
from utils import load_pdp_dataset, _create_pyg_from_pdp_data


def get_aco_builder_args(pdp_info_dict):
    """
    Filters out valid arguments for ACO.__init__ from a large pdp_info dictionary.
    """
    valid_keys = [
        "node_types",
        "pair_ids",
        "depot_node_idx",
        "num_pickup_nodes",
        "num_delivery_nodes",
    ]
    return {key: pdp_info_dict[key] for key in valid_keys if key in pdp_info_dict}


def generate_on_the_fly_pdp_instance(
    num_cust: int, k_sparse: int, device: torch.device, node_feat_dim: int
):
    """Helper function to generate on-the-fly data."""
    if num_cust % 2 != 0 and num_cust > 1:
        num_cust -= 1
    num_cust = max(2, num_cust)
    n_nodes = 1 + num_cust
    coords = torch.rand(n_nodes, 2)
    depot_idx = 0
    available_nodes = list(range(1, n_nodes))
    random.shuffle(available_nodes)
    pd_pairs, demands, time_windows = [], torch.zeros(n_nodes), torch.zeros(n_nodes, 2)
    time_windows[0, 1] = 480.0
    for _ in range(num_cust // 2):
        p_node, d_node = available_nodes.pop(0), available_nodes.pop(0)
        demand_val = float(random.randint(5, 20))
        demands[p_node], demands[d_node] = demand_val, -demand_val
        pd_pairs.append((p_node, d_node))

    pyg_data, dists, pdp_info = _create_pyg_from_pdp_data(
        coords,
        pd_pairs,
        demands,
        time_windows,
        depot_idx,
        k_sparse,
        device,
        node_feat_dim,
    )
    return {"pyg_data": pyg_data, "actual_distances": dists, "pdp_info": pdp_info}


def train_actor_critic_step(
    model: Net,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
):
    """Performs a 'super-step' with a loss function that considers Local Search."""
    model.train()
    optimizer.zero_grad()

    batch_loss_components_list = []

    for _ in range(args.accumulation_steps):
        instance_batch_list = [
            generate_on_the_fly_pdp_instance(
                random.randint(args.min_cust_nodes, args.max_cust_nodes),
                args.k_sparse,
                device,
                args.node_feat_dim,
            )
            for _ in range(args.batch_size)
        ]
        instance_batch_list = [
            inst for inst in instance_batch_list if inst and inst.get("pyg_data")
        ]
        if not instance_batch_list:
            continue

        pyg_batch = Batch.from_data_list(
            [d["pyg_data"] for d in instance_batch_list]
        ).to(device)

        heuristic_vector_batch, value_predictions_batch = model(pyg_batch)

        edge_batch_vector = pyg_batch.batch[pyg_batch.edge_index[0]]
        heuristic_vectors_per_instance = unbatch(
            heuristic_vector_batch, edge_batch_vector
        )

        actor_losses, critic_losses, entropy_losses, mean_costs_ls = [], [], [], []

        for i, instance_data in enumerate(instance_batch_list):
            pdp_info_for_aco = get_aco_builder_args(instance_data["pdp_info"])
            pyg_instance_data = instance_batch_list[i]["pyg_data"]
            heuristic_matrix = Net.reshape_to_matrix(
                pyg_instance_data, heuristic_vectors_per_instance[i]
            )

            aco = ACO(
                distances=instance_data["actual_distances"],
                **pdp_info_for_aco,
                n_ants=args.train_ants,
                heuristic=heuristic_matrix,
                device=str(device),
                alpha=args.train_aco_alpha,
                beta=args.train_aco_beta,
                decay=args.train_aco_decay,
                elitist=args.train_aco_elitist,
                min_max=args.train_aco_min_max,
                local_search=args.train_ls_mode,
                neural_lns_weight=args.neural_lns_weight,
            )

            initial_costs, log_probs, entropies, initial_paths, valid_mask = aco.sample(
                inference=False
            )

            if not valid_mask.any():
                continue

            ls_costs, _ = aco.sample_ls(initial_paths, valid_mask, inference=False)

            valid_ls_costs = ls_costs[valid_mask]
            valid_initial_costs = initial_costs[valid_mask]
            valid_log_probs = log_probs[valid_mask]
            valid_entropies = entropies[valid_mask]

            value_prediction_i = value_predictions_batch[i]

            reward = -(valid_initial_costs + args.W_ls * valid_ls_costs)
            advantage = reward.detach() - value_prediction_i.detach()

            if advantage.numel() > 1:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-9)

            actor_loss = (advantage * valid_log_probs.sum(dim=1)).mean()

            critic_loss = F.mse_loss(value_prediction_i, reward.mean())
            entropy_loss = -valid_entropies.mean()

            if not torch.isnan(actor_loss):
                actor_losses.append(actor_loss)
            if not torch.isnan(critic_loss):
                critic_losses.append(critic_loss)
            if not torch.isnan(entropy_loss):
                entropy_losses.append(entropy_loss)
            mean_costs_ls.append(valid_ls_costs.mean().item())  # Track post-LS cost

        if not actor_losses:
            continue

        avg_actor_loss = torch.stack(actor_losses).mean()
        avg_critic_loss = torch.stack(critic_losses).mean()
        avg_entropy_loss = torch.stack(entropy_losses).mean()

        total_loss = (
            avg_actor_loss
            + args.critic_loss_coef * avg_critic_loss
            + args.entropy_coef * avg_entropy_loss
        )

        if not torch.isnan(total_loss):
            (total_loss / args.accumulation_steps).backward()
            batch_loss_components_list.append(
                {
                    "total": total_loss.item(),
                    "actor": avg_actor_loss.item(),
                    "critic": avg_critic_loss.item(),
                    "entropy": avg_entropy_loss.item(),
                    "cost": np.mean(mean_costs_ls) if mean_costs_ls else float("nan"),
                }
            )

    if args.grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
    optimizer.step()

    return batch_loss_components_list


def validation_worker(instance_data, model, args, device):
    """
    Worker function to run validation on a single instance.
    Designed for use with multiprocessing.Pool.
    """
    model.eval()

    pyg_data = instance_data["pyg_data"].to(device)
    pdp_info_for_aco = instance_data["pdp_info"]

    heuristic_vector, _ = model(pyg_data)
    heuristic_matrix = Net.reshape_to_matrix(pyg_data, heuristic_vector)

    aco = ACO(
        distances=instance_data["actual_distances"].to(device),
        **get_aco_builder_args(pdp_info_for_aco),
        n_ants=args.val_ants,
        heuristic=heuristic_matrix,
        device=str(device),
        local_search=args.val_ls_mode,
        beta=args.val_aco_beta,
        elitist=True,
        min_max=True,
        stagnation_limit=args.val_aco_stagnation,
        neural_lns_weight=args.neural_lns_weight,
    )
    cost, _ = aco.run(args.T_val_aco, inference=True)
    return cost


@torch.no_grad()
def validation_pdp(
    model: Net, val_dataset: list, args: argparse.Namespace, device: torch.device
):
    """
    Runs validation in parallel on the entire validation dataset.
    """
    model.eval()

    worker_func = partial(validation_worker, model=model, args=args, device=device)

    all_costs = []
    num_workers = min(args.val_num_workers, cpu_count(), len(val_dataset))

    print(f"\nStarting validation with {num_workers} workers...")

    if num_workers > 0:
        with Pool(processes=num_workers) as pool:
            results_iterator = pool.imap(worker_func, val_dataset)
            all_costs = list(
                tqdm(
                    results_iterator,
                    total=len(val_dataset),
                    desc="Validating",
                    leave=False,
                    ncols=100,
                )
            )
    else:
        for data in tqdm(
            val_dataset, desc="Validating (sequential)", leave=False, ncols=100
        ):
            all_costs.append(worker_func(data))

    valid_costs = [c for c in all_costs if c != float("inf") and not np.isnan(c)]
    return np.mean(valid_costs) if valid_costs else float("inf")


def train_main(args):
    train_device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {train_device}")

    os.makedirs(args.save_dir, exist_ok=True)
    config_filepath = os.path.join(args.save_dir, f"{args.problem_id}_config.json")
    best_model_filepath = os.path.join(args.save_dir, f"{args.problem_id}_best.pt")
    log_filepath = os.path.join(args.save_dir, f"{args.problem_id}_training_log.csv")

    with open(log_filepath, "w", newline="") as f:
        log_writer = csv.writer(f)
        log_writer.writerow(
            [
                "epoch",
                "val_cost",
                "avg_total_loss",
                "avg_actor_loss",
                "avg_critic_loss",
                "avg_entropy_loss",
                "avg_train_cost",
                "learning_rate",
            ]
        )

    mlp_hidden_heuristic = [args.decoder_hidden_dim] * args.decoder_layers
    model_params = {
        "node_feat_dim": args.node_feat_dim,
        "edge_feat_dim": args.edge_feat_dim,
        "gnn_embedding_dim": args.gnn_embedding_dim,
        "gnn_depth": args.gnn_depth,
        "gnn_n_heads": args.gnn_n_heads,
        "gnn_dropout_p": args.gnn_dropout_p,
        "mlp_hidden_layers_heuristic": mlp_hidden_heuristic,
        "use_checkpoint_gnn": args.use_checkpoint_gnn,
        "gnn_act_fn": args.gnn_act_fn,
        "mlp_act_fn_heuristic": args.mlp_act_fn_heuristic,
    }
    model = Net(**model_params).to(train_device)

    if train_device.type == "cuda":
        model.share_memory()

    with open(config_filepath, "w") as f:
        train_config_to_save = {
            "model_constructor_args": model_params,
            "train_args": vars(args),
        }
        json.dump(train_config_to_save, f, indent=4)
        print(f"Configuration saved to {config_filepath}")

    val_dataset = load_pdp_dataset(
        args.val_file, args.k_sparse, torch.device("cpu"), args.node_feat_dim
    )

    print("\n" + "=" * 50 + "\n STARTING REINFORCEMENT LEARNING TRAINING\n" + "=" * 50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.rl_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=5, min_lr=1e-7
    )

    best_val_cost = validation_pdp(model, val_dataset, args, train_device)
    print(f"Initial Validation Cost (random model): {best_val_cost:.4f}")
    if not np.isinf(best_val_cost):
        torch.save(model.state_dict(), best_model_filepath)

    with open(log_filepath, "a", newline="") as log_file:
        log_writer = csv.writer(log_file)
        for epoch in range(1, args.rl_epochs + 1):
            prog_bar = tqdm(
                range(args.rl_steps_per_epoch),
                desc=f"RL Epoch {epoch}/{args.rl_epochs}",
                leave=False,
                ncols=100,
            )
            epoch_losses = {
                k: [] for k in ["total", "actor", "critic", "entropy", "cost"]
            }

            for _ in prog_bar:
                batch_loss_components = train_actor_critic_step(
                    model, optimizer, args, train_device
                )
                if batch_loss_components:
                    for comp in batch_loss_components:
                        for key, val in comp.items():
                            if not np.isnan(val):
                                epoch_losses[key].append(val)
                if epoch_losses["total"]:
                    prog_bar.set_postfix(
                        loss=f"{np.mean(epoch_losses['total']):.4f}",
                        cost=f"{np.mean(epoch_losses['cost']):.2f}",
                    )

            val_cost = validation_pdp(model, val_dataset, args, train_device)
            scheduler.step(val_cost)

            avg_losses = {
                k: np.mean(v) if v else float("nan") for k, v in epoch_losses.items()
            }
            lr = optimizer.param_groups[0]["lr"]
            log_writer.writerow(
                [
                    f"RL_{epoch}",
                    f"{val_cost:.4f}",
                    f"{avg_losses['total']:.4f}",
                    f"{avg_losses['actor']:.4f}",
                    f"{avg_losses['critic']:.4f}",
                    f"{avg_losses['entropy']:.4f}",
                    f"{avg_losses['cost']:.2f}",
                    f"{lr:.2e}",
                ]
            )

            print(
                f"\nRL Epoch {epoch} | Val Cost: {val_cost:.4f} | Avg Loss: {avg_losses['total']:.4f} | LR: {lr:.2e}"
            )
            if val_cost < best_val_cost:
                best_val_cost = val_cost
                torch.save(model.state_dict(), best_model_filepath)
                print(f"New best RL model saved with Val Cost: {best_val_cost:.4f}")

    print(
        f"\n--- TRAINING COMPLETE ---\nBest model (Val Cost: {best_val_cost:.4f}) saved at: {best_model_filepath}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DeepACO for PDP with Reinforcement Learning."
    )

    # === Main Parameters ===
    parser.add_argument(
        "--val_file",
        type=str,
        required=True,
        help="Path to the .pt file containing validation data.",
    )
    parser.add_argument("--problem_id", type=str, default="pdp_rl_only")
    parser.add_argument("--save_dir", type=str, default="pretrained/pdp_rl_only")
    parser.add_argument("--device", type=str, default="auto")

    # === Model Architecture ===
    parser.add_argument(
        "--node_feat_dim",
        type=int,
        default=8,
        help="Must match the feature creation logic in utils.py",
    )
    parser.add_argument(
        "--edge_feat_dim",
        type=int,
        default=1,
        help="Dimension of edge features, usually 1 (distance).",
    )
    parser.add_argument("--gnn_embedding_dim", type=int, default=128)
    parser.add_argument("--gnn_depth", type=int, default=15)
    parser.add_argument("--gnn_n_heads", type=int, default=4)
    parser.add_argument("--gnn_dropout_p", type=float, default=0.1)
    parser.add_argument(
        "--use_checkpoint_gnn",
        action="store_true",
        help="Use torch.checkpoint to save GNN memory.",
    )
    parser.add_argument(
        "--decoder_layers",
        type=int,
        default=2,
        help="Number of hidden layers in the heuristic MLP.",
    )
    parser.add_argument(
        "--decoder_hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension in the heuristic MLP.",
    )
    parser.add_argument(
        "--gnn_act_fn",
        type=str,
        default="relu",
        help="Activation function for GNN (relu, silu).",
    )
    parser.add_argument(
        "--mlp_act_fn_heuristic",
        type=str,
        default="relu",
        help="Activation function for Heuristic MLP (relu, silu).",
    )

    # === RL Training ===
    parser.add_argument(
        "--rl_epochs", type=int, default=100, help="Number of epochs for RL training."
    )
    parser.add_argument("--rl_steps_per_epoch", type=int, default=200)
    parser.add_argument(
        "--rl_lr", type=float, default=1e-5, help="Learning rate for RL."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of instances in a small accumulation batch.",
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=8,
        help="Simulate a larger batch size (effective_batch = batch_size * accumulation_steps).",
    )
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--critic_loss_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)

    # === Data Generation Parameters ===
    parser.add_argument("--min_cust_nodes", type=int, default=40)
    parser.add_argument("--max_cust_nodes", type=int, default=50)
    parser.add_argument("--k_sparse", type=int, default=20)

    # === ACO Parameters for RL Training ===
    parser.add_argument("--train_ants", type=int, default=20)
    parser.add_argument("--train_aco_alpha", type=float, default=1.0)
    parser.add_argument(
        "--train_aco_beta",
        type=float,
        default=2.0,
        help="Low beta during training encourages exploration.",
    )
    parser.add_argument("--train_aco_decay", type=float, default=0.98)
    parser.add_argument(
        "--train_aco_elitist", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--train_aco_min_max", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--train_ls_mode",
        type=str,
        default="neural_lns_pdp",
        help="LS mode during training.",
    )
    parser.add_argument(
        "--W_ls", type=float, default=1.0, help="Weight for the LS loss term."
    )

    # === ACO/LS Parameters for Validation ===
    parser.add_argument("--val_ants", type=int, default=20)
    parser.add_argument("--T_val_aco", type=int, default=50)
    parser.add_argument(
        "--val_aco_beta",
        type=float,
        default=5.0,
        help="High beta during validation for exploitation.",
    )
    parser.add_argument(
        "--val_ls_mode",
        type=str,
        default="lns_pdp",
        choices=["none", "lns_pdp", "neural_lns_pdp"],
    )
    parser.add_argument("--val_aco_stagnation", type=int, default=20)
    parser.add_argument("--neural_lns_weight", type=float, default=0.5)
    parser.add_argument(
        "--val_num_workers",
        type=int,
        default=max(1, cpu_count() // 2),
        help="Number of parallel processes for validation.",
    )
    args = parser.parse_args()
    train_main(args)
