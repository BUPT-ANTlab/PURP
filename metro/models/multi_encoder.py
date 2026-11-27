"""Equivalent sparse aggregation for the author's GNN3 message-passing equations."""
import numpy as np
import torch
from metro.models.state_encoder import GNN3StateEncoder


class MultiObjectiveEncoder(GNN3StateEncoder):
    def __init__(self, cfg, agent):
        super().__init__(cfg, agent)
        count = agent.env.core.node_num
        for name, edges in (("spatial_adjacency", agent.env.core.edge_index_dis),
                            ("od_adjacency", agent.env.core.edge_index_od)):
            rows = np.concatenate([edges[:, 0], edges[:, 1]])
            cols = np.concatenate([edges[:, 1], edges[:, 0]])
            degree = np.bincount(rows, minlength=count).astype(np.float32)
            weights = 1. / (degree[rows] + self.EPSILON)
            matrix = torch.sparse_coo_tensor(torch.tensor(np.stack([rows, cols])),
                        torch.tensor(weights), (count, count)).coalesce()
            self.register_buffer(name, matrix, persistent=False)

    @staticmethod
    def aggregate(matrix, nodes):
        batch, count, width = nodes.shape
        flattened = nodes.transpose(0, 1).reshape(count, batch * width)
        return torch.sparse.mm(matrix, flattened).reshape(count, batch, width).transpose(0, 1)

    def forward(self, x):
        numerical, node_feature, _, _, mask, stage = self.batch_data(x)
        global_features = self.numerical_feature_encoder(numerical.float())
        nodes = self.node_encoder(node_feature.float())
        # Preserve the author's layer semantics, including independent residual branches.
        # Transform each node once, then average its neighbors using fixed adjacency.
        for layer in self.edge_fc_layers1:
            spatial = nodes + self.aggregate(self.spatial_adjacency, layer(nodes))
        for layer in self.edge_fc_layers2:
            od = nodes + self.aggregate(self.od_adjacency, layer(nodes))
        nodes = torch.cat([spatial, od], dim=-1)
        built = node_feature[:, :, 30] > 0
        counts = built.sum(dim=1)
        if torch.any(counts == 0):
            raise ValueError("Attention requires an initial metro station")
        width = int(counts.max().item())
        indices = torch.arange(nodes.shape[1], device=nodes.device).expand(nodes.shape[:2])
        indices = indices.masked_fill(~built, nodes.shape[1]).sort(dim=1).values[:, :width]
        padding = indices == nodes.shape[1]
        keys = nodes.gather(1, indices.clamp_max(nodes.shape[1]-1).unsqueeze(-1)
                            .expand(-1, -1, nodes.shape[-1]))
        attended = self.attention1(nodes, keys, key_padding_mask=padding)
        pooled = self.mean_features(attended)
        value = torch.cat([global_features, pooled, stage], dim=-1)
        policy = torch.cat([attended, global_features.unsqueeze(1).expand(-1, nodes.shape[1], -1)], dim=-1)
        return policy, value, mask, stage
