"""Message passing written directly in PyTorch (no PyG / DGL dependency).

Node tensors are (B, N, D) and the graph is shared across the batch, so a
message-passing step is a gather over ``src``, a scatter over ``dst``, and a
softmax over each node's incoming edges.
"""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Sequential):
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__(nn.Linear(d_in, d_hidden), nn.SiLU(), nn.Linear(d_hidden, d_out))


def segment_softmax(logits: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Softmax of ``logits`` (B, E, H) over the edges that share the same ``index`` (E,)."""
    B, E, H = logits.shape
    idx = index.view(1, E, 1).expand(B, E, H)
    peak = torch.full((B, num_segments, H), float("-inf"), device=logits.device, dtype=logits.dtype)
    peak = peak.scatter_reduce(1, idx, logits, reduce="amax", include_self=False)
    w = torch.exp(logits - peak.gather(1, idx))
    total = torch.zeros((B, num_segments, H), device=logits.device, dtype=logits.dtype)
    total = total.index_add(1, index, w)
    return w / total.gather(1, idx).clamp_min(1e-12)


class GraphAttentionLayer(nn.Module):
    """One processor step.

    Edge update as in an interaction network (GraphCast / MeshGraphNets):
    each edge latent is refreshed from its two endpoints. That latent then
    supplies both the message and multi-head attention scores, so a site can
    learn which neighbours (and which directions) matter in the current state.

    With ``use_graph=False`` the aggregation is zeroed out, giving an otherwise
    identical per-station model for ablations.
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.norm_node = nn.LayerNorm(dim)
        self.norm_edge = nn.LayerNorm(dim)
        # edge MLP on [h_src, h_dst, e]. The first linear layer is split so the
        # node terms are computed once per node and gathered, instead of once
        # per edge: same function, roughly a third of the FLOPs.
        self.edge_src = nn.Linear(dim, dim, bias=False)
        self.edge_dst = nn.Linear(dim, dim, bias=False)
        self.edge_self = nn.Linear(dim, dim)
        self.edge_out = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim))
        self.score = nn.Linear(dim, heads)
        self.value = nn.Linear(dim, dim)
        self.node_mlp = MLP(2 * dim, dim, dim)

    def forward(self, h, e, src, dst, use_graph: bool = True):
        B, N, D = h.shape
        hn = self.norm_node(h)
        if not use_graph:
            return h + self.node_mlp(torch.cat([hn, torch.zeros_like(hn)], -1)), e, None

        pre = self.edge_src(hn)[:, src] + self.edge_dst(hn)[:, dst] + self.edge_self(self.norm_edge(e))
        z = self.edge_out(pre)
        alpha = segment_softmax(self.score(z), dst, N)  # (B, E, heads)
        v = self.value(z).view(B, -1, self.heads, D // self.heads)
        msg = (alpha.unsqueeze(-1) * v).reshape(B, -1, D)
        agg = torch.zeros_like(h).index_add(1, dst, msg)
        return h + self.node_mlp(torch.cat([hn, agg], -1)), e + z, alpha
