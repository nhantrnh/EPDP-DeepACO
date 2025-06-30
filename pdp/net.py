import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import scatter
from torch.utils.checkpoint import checkpoint


class GATv2EmbNet(nn.Module):
    def __init__(
        self,
        node_input_dim,
        edge_input_dim,
        embedding_dim,
        gnn_depth,
        n_heads=4,
        dropout_p=0.1,
        act_fn_str="relu",
    ):
        super().__init__()
        self.gnn_depth = gnn_depth

        if act_fn_str.lower() == "relu":
            self.act_module = nn.ReLU()
        elif act_fn_str.lower() in ["silu", "swish"]:
            self.act_module = nn.SiLU()
        else:
            raise ValueError(f"Activation module '{act_fn_str}' not supported.")

        self.act_fn_functional = getattr(F, act_fn_str)

        self.node_emb_initial = nn.Linear(node_input_dim, embedding_dim)
        self.edge_emb_initial = nn.Linear(edge_input_dim, embedding_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.edge_updates = nn.ModuleList()

        for _ in range(gnn_depth):
            self.convs.append(
                pyg_nn.GATv2Conv(
                    embedding_dim,
                    embedding_dim,
                    heads=n_heads,
                    concat=False,
                    edge_dim=embedding_dim,
                )
            )
            self.norms.append(nn.LayerNorm(embedding_dim))
            self.dropouts.append(nn.Dropout(p=dropout_p))
            self.edge_updates.append(
                nn.Sequential(
                    nn.Linear(embedding_dim * 3, embedding_dim),
                    self.act_module,
                    nn.Linear(embedding_dim, embedding_dim),
                )
            )
        self.reset_parameters()

    def reset_parameters(self):
        self.node_emb_initial.reset_parameters()
        self.edge_emb_initial.reset_parameters()
        for i in range(self.gnn_depth):
            self.convs[i].reset_parameters()
            self.norms[i].reset_parameters()
            for layer in self.edge_updates[i]:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

    def forward(self, node_feat_in, edge_index_in, edge_attr_in, use_checkpoint=False):
        node_emb = self.act_fn_functional(self.node_emb_initial(node_feat_in))
        edge_emb = self.act_fn_functional(self.edge_emb_initial(edge_attr_in))
        row, col = edge_index_in
        for i in range(self.gnn_depth):
            node_emb_prev, edge_emb_prev = node_emb, edge_emb

            if use_checkpoint and self.training:

                def conv_wrapper(x, edge_index, edge_attr):
                    return self.convs[i](x, edge_index, edge_attr=edge_attr)

                node_emb_update = checkpoint(
                    conv_wrapper, node_emb_prev, edge_index_in, edge_emb_prev
                )
            else:
                node_emb_update = self.convs[i](
                    node_emb_prev, edge_index_in, edge_attr=edge_emb_prev
                )

            node_emb_update = self.dropouts[i](node_emb_update)
            node_emb = self.norms[i](node_emb_prev + node_emb_update)
            node_emb = self.act_fn_functional(node_emb)

            source_node_emb, dest_node_emb = node_emb[row], node_emb[col]
            edge_update_input = torch.cat(
                [source_node_emb, dest_node_emb, edge_emb_prev], dim=-1
            )
            edge_emb_update = self.edge_updates[i](edge_update_input)
            edge_emb = self.act_fn_functional(edge_emb_prev + edge_emb_update)

        return node_emb, edge_emb


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims_list,
        act_fn_str="relu",
        output_act_fn_str="sigmoid",
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        current_dim = input_dim
        for h_dim in hidden_dims_list:
            self.layers.append(nn.Linear(current_dim, h_dim))
            self.layers.append(self._get_activation_fn(act_fn_str))
            current_dim = h_dim
        self.layers.append(nn.Linear(current_dim, output_dim))
        if output_act_fn_str is not None:
            self.layers.append(self._get_activation_fn(output_act_fn_str))
        self.reset_parameters()

    def _get_activation_fn(self, act_fn_str):
        if act_fn_str.lower() == "sigmoid":
            return nn.Sigmoid()
        if act_fn_str.lower() == "relu":
            return nn.ReLU()
        if act_fn_str.lower() in ["silu", "swish"]:
            return nn.SiLU()
        raise ValueError(f"Activation function '{act_fn_str}' not supported.")

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class HeuristicNet(MLP):
    def __init__(self, edge_embedding_dim, mlp_hidden_dims, act_fn_str="relu"):
        super().__init__(
            input_dim=edge_embedding_dim,
            output_dim=1,
            hidden_dims_list=mlp_hidden_dims,
            act_fn_str=act_fn_str,
        )

    def forward(self, edge_embeddings):
        return super().forward(edge_embeddings).squeeze(dim=-1)


class Net(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.use_checkpoint_gnn = kwargs.get("use_checkpoint_gnn", False)

        # Actor
        self.emb_net = GATv2EmbNet(
            node_input_dim=kwargs.get("node_feat_dim"),
            edge_input_dim=kwargs.get("edge_feat_dim", 1),
            embedding_dim=kwargs.get("gnn_embedding_dim", 64),
            gnn_depth=kwargs.get("gnn_depth", 15),
            n_heads=kwargs.get("gnn_n_heads", 4),
            dropout_p=kwargs.get("gnn_dropout_p", 0.1),
            act_fn_str=kwargs.get("gnn_act_fn", "relu"),
        )
        self.heuristic_predictor_net = HeuristicNet(
            edge_embedding_dim=kwargs.get("gnn_embedding_dim", 64),
            mlp_hidden_dims=kwargs.get("mlp_hidden_layers_heuristic", [64, 32]),
            act_fn_str=kwargs.get("mlp_act_fn_heuristic", "relu"),
        )

        # Critic
        self.critic_net = nn.Sequential(
            nn.Linear(kwargs.get("gnn_embedding_dim", 64), 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, pyg_data):
        node_embeddings, edge_embeddings = self.emb_net(
            pyg_data.x,
            pyg_data.edge_index,
            pyg_data.edge_attr,
            use_checkpoint=self.use_checkpoint_gnn and self.training,
        )

        # Actor head
        heuristic_vector = self.heuristic_predictor_net(edge_embeddings)

        # Critic head
        batch_vector = getattr(
            pyg_data,
            "batch",
            torch.zeros(pyg_data.num_nodes, dtype=torch.long, device=pyg_data.x.device),
        )
        graph_embedding = pyg_nn.global_mean_pool(node_embeddings, batch_vector)
        value_prediction = self.critic_net(graph_embedding).squeeze(-1)

        return heuristic_vector, value_prediction

    @staticmethod
    def reshape_to_matrix(pyg_data_instance, edge_vector_values, fill_value=1e-9):
        num_nodes = pyg_data_instance.num_nodes
        device = edge_vector_values.device
        adj_matrix = torch.full(
            (num_nodes, num_nodes),
            fill_value,
            device=device,
            dtype=edge_vector_values.dtype,
        )
        if pyg_data_instance.edge_index.numel() > 0:
            adj_matrix[
                pyg_data_instance.edge_index[0], pyg_data_instance.edge_index[1]
            ] = edge_vector_values
        return adj_matrix
