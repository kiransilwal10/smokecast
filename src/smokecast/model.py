"""SmokeCast: encode-process-decode GNN over the PM2.5 monitoring network.

    encode   per-site GRU over the last 48 h of PM2.5, weather and satellite smoke,
             tokenised into 6 h patches
    process  L rounds of attention message passing between neighbouring sites;
             every edge carries the sender's current wind projected onto the edge,
             so a site hears most from the neighbours whose air is blowing toward it
    decode   per-site MLP -> PM2.5 now (lead 0) and for each of the next H hours

Lead 0 matters only for sites whose own readings were hidden: it is the model's
estimate of what that sensor should be reading, given its neighbours. Comparing
it with what the sensor actually reports is how faults are detected.

As in SurgeCast, the decoder output is added to a linear forecast from the raw
inputs and to the current reading, so the network learns a correction on top
of persistence and a linear autoregressive model.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .dataset import SMOKE, U, V
from .graph import WIND_EDGE_DIM, edge_geometry, knn_edges, node_positions, static_edge_features, wind_edge_features
from .layers import MLP, GraphAttentionLayer

N_TIME = 6


class SmokeCast(nn.Module):
    def __init__(self, lat: np.ndarray, lon: np.ndarray, in_channels: int, horizon: int,
                 dim: int = 128, layers: int = 4, heads: int = 4, k: int = 8,
                 use_graph: bool = True, wind_edges: bool = True, use_smoke: bool = True,
                 emb_dim: int = 16, patch: int = 6, history: int = 48):
        super().__init__()
        n = len(lat)
        src, dst = knn_edges(lat, lon, k)
        geometry = edge_geometry(lat, lon, src, dst)
        self.register_buffer("src", torch.tensor(src))
        self.register_buffer("dst", torch.tensor(dst))
        self.register_buffer("geometry", torch.tensor(geometry))
        self.register_buffer("edge_static", torch.tensor(static_edge_features(geometry)))
        self.register_buffer("pos", torch.tensor(node_positions(lat, lon)))

        self.use_graph, self.wind_edges, self.use_smoke, self.patch = use_graph, wind_edges, use_smoke, patch
        self.patch_embed = nn.Linear(patch * in_channels, dim)
        self.temporal = nn.GRU(dim, dim, batch_first=True)
        self.node_emb = nn.Embedding(n, emb_dim)
        self.node_encoder = MLP(dim + emb_dim + 3 + N_TIME, dim, dim)
        self.edge_encoder = MLP(3 + WIND_EDGE_DIM, dim, dim)
        self.processor = nn.ModuleList(GraphAttentionLayer(dim, heads) for _ in range(layers))
        self.decoder = MLP(dim + N_TIME, dim, horizon + 1)
        self.linear = nn.Linear(history * in_channels + N_TIME, horizon + 1)

    def forward(self, batch: dict, return_attention: bool = False):
        x = batch["x"]
        if not self.use_smoke:
            x = x.clone()
            x[..., list(SMOKE)] = 0.0
        B, T, N, C = x.shape
        tokens = x.permute(0, 2, 1, 3).reshape(B * N, T // self.patch, self.patch * C)
        _, h_last = self.temporal(self.patch_embed(tokens))
        h = h_last[-1].view(B, N, -1)

        time = batch["time"][:, None].expand(-1, N, -1)
        static = torch.cat([self.node_emb.weight, self.pos], -1).expand(B, -1, -1)
        h = self.node_encoder(torch.cat([h, static, time], -1))

        edge = self.edge_static.expand(B, -1, -1)
        wind = x[:, -6:, :, [U, V]].mean(1) * 10.0  # m/s, mean of the last 6 h
        dynamic = wind_edge_features(wind, self.geometry, self.src) if self.wind_edges else \
            torch.zeros(B, edge.shape[1], WIND_EDGE_DIM, device=x.device)
        e = self.edge_encoder(torch.cat([edge, dynamic], -1))

        attention = []
        for layer in self.processor:
            h, e, alpha = layer(h, e, self.src, self.dst, self.use_graph)
            attention.append(alpha)

        flat = torch.cat([x.permute(0, 2, 1, 3).reshape(B, N, T * C), time], -1)
        out = self.decoder(torch.cat([h, time], -1)) + self.linear(flat) + batch["base"][..., None]
        out = out.transpose(1, 2)  # (B, H + 1, N)
        return (out, attention) if return_attention else out
