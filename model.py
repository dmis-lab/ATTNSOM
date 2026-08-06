"""ATTNSOM model.

Architecture (see Materials and Methods in the paper):

1. a **shared graph encoder** producing atom representations ``n_i`` and a
   graph representation ``g`` (GraphCliff by default),
2. **FiLM** conditioning of the atom representations on ``g`` (Eq. 1-2),
3. **cross-attention** from atoms to learnable CYP isoform embeddings
   (Eq. 3-4),
4. a **prediction head** over ``[n'_i || n_i^attn || c_t]`` (Eq. 5).

The ablated variants of Table 3 are obtained by toggling ``use_attention`` /
``use_film`` and by swapping ``encoder``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGPooling, GINConv, GINEConv, GCNConv, GATConv
from torch_geometric.nn import global_mean_pool as gap


def normalize_edges(num_nodes, edge_index, edge_weight):
    if edge_index.numel() == 0:
        return edge_weight

    src, dst = edge_index
    deg = torch.zeros(num_nodes, device=edge_index.device, dtype=edge_weight.dtype)
    deg = deg.scatter_add_(0, src, edge_weight)
    deg = deg.clamp(min=1e-12)
    d_inv_sqrt = deg.pow(-0.5)
    norm = edge_weight * d_inv_sqrt[src] * d_inv_sqrt[dst]
    return norm


def propagate(x, edge_index, edge_weight_norm):
    if edge_index.numel() == 0:
        return torch.zeros_like(x)
    src, dst = edge_index
    ew = edge_weight_norm.to(x.dtype)
    msg = x[src] * ew.unsqueeze(-1)
    out = torch.zeros_like(x)
    out.index_add_(0, dst, msg)
    return out


class AtomEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.RMSNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class ShortGINE(nn.Module):
    def __init__(self, in_dim, edge_dim, dropout=0.0, deg_power=0.5):
        super().__init__()
        # Node MLP (GIN)
        node_mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim)
        )
        edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim)
        ) if edge_dim and edge_dim > 0 else None

        self.conv = GINConv(node_mlp, train_eps=True)
        self.edge_mlp = edge_mlp
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.RMSNorm(in_dim)
        self.deg_power = deg_power

    def forward(self, x, edge_index, edge_attr):
        residual = x
        src, dst = edge_index

        num_nodes = x.size(0)
        deg = torch.bincount(dst, minlength=num_nodes).float().clamp(min=1.0)
        deg_src = deg[src]
        deg_dst = deg[dst]
        edge_weight = 1.0 / ((deg_src * deg_dst) ** self.deg_power)

        if self.edge_mlp is not None and edge_attr is not None:
            edge_msg = self.edge_mlp(edge_attr)
            messages = x[src] + edge_msg
        else:
            messages = x[src]

        messages = messages * edge_weight.unsqueeze(-1)

        out = torch.zeros_like(x)
        out.scatter_add_(0, dst.unsqueeze(-1).expand(-1, x.size(-1)), messages)

        eps = self.conv.eps if hasattr(self.conv, 'eps') else 0.0
        combined = (1 + eps) * x + out

        out = self.conv.nn(combined)
        out = self.dropout(out) + residual
        out = self.norm(out)
        return out

class LongPoly(nn.Module):
    def __init__(self, hidden_size, K=5, groups=4, dropout=0.1):
        super().__init__()
        assert hidden_size % groups == 0, "hidden_size must be divisible by groups"
        self.K = K
        self.groups = groups
        self.group_channels = hidden_size // groups

        self.cheb_coeffs = nn.Parameter(torch.empty(groups, K + 1))
        nn.init.xavier_uniform_(self.cheb_coeffs, gain=0.1)
        self.group_scale = nn.Parameter(torch.ones(groups))
        self.group_bias  = nn.Parameter(torch.zeros(groups))

        self.norm = nn.RMSNorm(hidden_size)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.register_buffer('_cached_edge_index', None)
        self.register_buffer('_cached_polynomials', None)

    def forward(self, x, edge_index, edge_weight_norm):
        N, H = x.shape
        if edge_index.numel() == 0:
            x_grouped = x.view(N, self.groups, self.group_channels)
            result = self.cheb_coeffs[:, 0].view(1, -1, 1) * x_grouped
            result = result * self.group_scale.view(1, -1, 1) + self.group_bias.view(1, -1, 1)
            return self.dropout(self.activation(self.norm(result.reshape(N, H))))

        x_grouped = x.view(N, self.groups, self.group_channels)
        result = self.cheb_coeffs[:, 0].view(1, -1, 1) * x_grouped

        if self.K >= 1:
            T_prev2 = x
            T_prev1 = propagate(x, edge_index, edge_weight_norm)
            T1_grouped = T_prev1.view(N, self.groups, self.group_channels)
            result += self.cheb_coeffs[:, 1].view(1, -1, 1) * T1_grouped
            for k in range(2, self.K + 1):
                T_curr = 2 * propagate(T_prev1, edge_index, edge_weight_norm) - T_prev2
                T_curr_grouped = T_curr.view(N, self.groups, self.group_channels)
                result += self.cheb_coeffs[:, k].view(1, -1, 1) * T_curr_grouped
                T_prev2, T_prev1 = T_prev1, T_curr

        result = result * self.group_scale.view(1, -1, 1) + self.group_bias.view(1, -1, 1)
        output = result.reshape(N, H)
        return self.dropout(self.activation(self.norm(output)))

class GraphCliffFilter(nn.Module):
    def __init__(self, hidden_size, edge_dim, groups=4, short_dropout=0.1, mid_K=3):
        super().__init__()
        self.pre_norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        nn.init.xavier_normal_(self.proj.weight, gain=1)
        nn.init.zeros_(self.proj.bias)
        self.short = ShortGINE(3 * hidden_size, edge_dim, short_dropout)
        self.long  = LongPoly(hidden_size, K=mid_K, groups=groups)
    def forward(self, u, edge_index, edge_attr):
        h = self.pre_norm(u)
        z = self.proj(h)
        z = self.short(z, edge_index, edge_attr)
        x2, x1, v = torch.chunk(z, 3, dim=-1)
        if edge_index.numel() > 0:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device, dtype=x2.dtype)
            edge_norm = normalize_edges(u.size(0), edge_index, edge_weight)
        else:
            edge_norm = torch.tensor([], device=u.device, dtype=u.dtype)
        mid_out = self.long(x2, edge_index, edge_norm)
        gate = torch.sigmoid(x1)
        y = mid_out * gate + v
        z_in = y + u
        return z_in

class GraphCliffEncoder(nn.Module):
    """Short/long-range gated encoder (GraphCliff), the default backbone."""

    def __init__(self, hidden_size, edge_dim, num_layers=3, groups=4, mid_K=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphCliffFilter(hidden_size, edge_dim, groups, short_dropout=dropout*0.5, mid_K=mid_K)
            for _ in range(num_layers)
        ])
    def forward(self, x, edge_index, edge_attr, rev_edge_index=None):
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x


# ---------------------------------------------------------------------------
# Alternative graph backbones (Table 3: "ATTNSOM w/ Chemprop / GIN / GCN / GAT")
# ---------------------------------------------------------------------------

class MPNNEncoder(nn.Module):
    """GIN / GCN / GAT encoder with residual connections and layer norm."""

    def __init__(self, conv_type, hidden_size, edge_dim, num_layers=3, dropout=0.1,
                 heads=4):
        super().__init__()
        self.conv_type = conv_type
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        use_edges = bool(edge_dim)

        for _ in range(num_layers):
            if conv_type == 'gin':
                mlp = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                )
                conv = GINEConv(mlp, train_eps=True, edge_dim=edge_dim) if use_edges \
                    else GINConv(mlp, train_eps=True)
            elif conv_type == 'gcn':
                conv = GCNConv(hidden_size, hidden_size)
            elif conv_type == 'gat':
                assert hidden_size % heads == 0, "hidden_size must be divisible by heads"
                conv = GATConv(hidden_size, hidden_size // heads, heads=heads,
                               edge_dim=edge_dim if use_edges else None)
            else:
                raise ValueError(f"Unknown conv type: {conv_type}")
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_size))

        self.dropout = nn.Dropout(dropout)
        self.use_edges = use_edges

    def forward(self, x, edge_index, edge_attr, rev_edge_index=None):
        for conv, norm in zip(self.convs, self.norms):
            if self.conv_type == 'gcn' or not self.use_edges:
                h = conv(x, edge_index)
            else:
                h = conv(x, edge_index, edge_attr)
            x = norm(x + self.dropout(F.relu(h)))
        return x


class DMPNNEncoder(nn.Module):
    """Directed bond-message encoder (Chemprop-style D-MPNN).

    Messages live on directed bonds; the message arriving at bond ``e`` sums the
    incoming bonds of its source atom minus the reverse bond, which prevents
    messages from immediately bouncing back.
    """

    def __init__(self, hidden_size, edge_dim, num_layers=3, dropout=0.1):
        super().__init__()
        self.depth = max(num_layers, 1)
        self.W_i = nn.Linear(hidden_size + edge_dim, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.W_o = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, rev_edge_index=None):
        if edge_index.numel() == 0:
            zeros = torch.zeros_like(x)
            return F.relu(self.W_o(torch.cat([x, zeros], dim=-1)))
        if rev_edge_index is None:
            raise ValueError("The D-MPNN encoder requires `rev_edge_index`.")

        src, dst = edge_index
        h_0 = F.relu(self.W_i(torch.cat([x[src], edge_attr], dim=-1)))
        h = h_0

        for _ in range(self.depth - 1):
            # Sum incoming bond messages per atom, then gather at each bond's source.
            node_msg = torch.zeros_like(x)
            node_msg.index_add_(0, dst, h)
            msg = node_msg[src] - h[rev_edge_index]
            h = self.dropout(F.relu(h_0 + self.W_h(msg)))

        node_msg = torch.zeros_like(x)
        node_msg.index_add_(0, dst, h)
        return F.relu(self.W_o(torch.cat([x, node_msg], dim=-1)))


ENCODERS = ('graphcliff', 'chemprop', 'gin', 'gcn', 'gat')


def build_encoder(name, hidden_size, edge_dim, num_layers, groups, mid_K, dropout):
    if name == 'graphcliff':
        return GraphCliffEncoder(hidden_size, edge_dim, num_layers, groups, mid_K, dropout)
    if name == 'chemprop':
        return DMPNNEncoder(hidden_size, edge_dim, num_layers, dropout)
    if name in ('gin', 'gcn', 'gat'):
        return MPNNEncoder(name, hidden_size, edge_dim, num_layers, dropout)
    raise ValueError(f"Unknown encoder '{name}', expected one of {ENCODERS}")


class GraphFiLM(nn.Module):
    """Feature-wise linear modulation of atoms by the graph context (Eq. 1-2).

    ``n'_i = (1 + tanh(gamma)) * n_i + beta``, with ``(gamma, beta) = MLP(g)``.
    """

    def __init__(self, node_dim: int, graph_dim: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(graph_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * node_dim)
        )

    def forward(self, h: torch.Tensor, g: torch.Tensor, batch: torch.Tensor):
        g_node = g[batch]
        gb = self.mlp(g_node)
        gamma, beta = gb.chunk(2, dim=-1)

        gamma = torch.tanh(gamma)
        h_film = h * (1 + gamma) + beta
        return h_film


class MultiHeadCYPAttention(nn.Module):
    """Atom-centric cross-attention over CYP isoform embeddings (Eq. 3-4).

    Atoms are queries; isoform embeddings are keys and values. An extra learned
    null key lets an atom abstain from attending to any isoform; it is excluded
    from the attention map that is returned (and hence from the auxiliary loss).
    """

    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

        self.null_token = nn.Parameter(torch.randn(1, hidden_size))


    def forward(self, h_node, cyp_kv, tau=0.1):
        N, H = h_node.shape
        num_agents = cyp_kv.size(0)

        full_kv = torch.cat([cyp_kv, self.null_token], dim=0)

        Q = self.q_proj(h_node).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        K = self.k_proj(full_kv).view(num_agents+1, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.v_proj(full_kv).view(num_agents+1, self.num_heads, self.head_dim).transpose(0, 1)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # tau sharpens the isoform distribution before normalisation.
        attn_weights = F.softmax(attn_scores / tau, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Multi-head Merge
        attn_output = torch.matmul(attn_weights, V) # (num_heads, N, head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(N, H)

        output = self.out_proj(attn_output)

        avg_attn = attn_weights.mean(dim=0)

        return output, avg_attn[:, :num_agents]

class ATTNSOM(nn.Module):
    """Isoform-aware atom-level site-of-metabolism predictor.

    Args:
        use_attention: cross-attention over isoform embeddings. Disable for the
            ``w/o attn.`` ablation.
        use_film: molecule-conditioned FiLM modulation. Disable for the
            ``w/o FiLM`` ablation.
        encoder: graph backbone, one of ``ENCODERS``.
    """

    def __init__(self, atom_in_dim, edge_dim=0, hidden_size=256,
                 num_layers=3, groups=4, mid_K=3, dropout=0.1,
                 cyp_names=None, num_attn_heads=4, use_attention=True,
                 use_film=True, encoder='graphcliff', attn_tau=0.1):
        super().__init__()
        self.use_attention = use_attention
        self.use_film = use_film
        self.encoder_type = encoder
        self.attn_tau = attn_tau
        self.cyp_names = list(cyp_names) if cyp_names else None
        self.num_agents = len(cyp_names) if cyp_names else 9

        # 1. Encoding Layers
        self.atom_encoder = AtomEncoder(atom_in_dim, hidden_size, dropout)
        self.encoder = build_encoder(encoder, hidden_size, edge_dim, num_layers,
                                     groups, mid_K, dropout)

        # 2. Context & Modulation
        if self.use_film:
            self.sagpool = SAGPooling(hidden_size, ratio=0.8)
            self.graph_film = GraphFiLM(hidden_size, hidden_size)

        # 3. Attention & Prediction
        if self.use_attention:
            self.attn_head = MultiHeadCYPAttention(hidden_size, num_attn_heads, dropout)
            input_dim = 3 * hidden_size
        else:
            input_dim = 2 * hidden_size

        self.pred_head = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        self.cyp_embs = nn.Parameter(torch.randn(self.num_agents, hidden_size))
        nn.init.xavier_normal_(self.cyp_embs)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        rev_edge_index = getattr(data, 'rev_edge_index', None)
        cyp_idx = data.cyp_idx
        batch = data.batch

        # 1) Atom/Graph Encoding
        h = self.atom_encoder(x)
        h = self.encoder(h, edge_index, edge_attr, rev_edge_index)

        # 2) Graph Context (FiLM)
        if self.use_film:
            # Single-graph batches skip the pooling layer and use a plain mean.
            if batch.max() == 0 and batch.size(0) > 1:
                g = gap(h, batch)
            else:
                x_p, _, _, b_p, _, _ = self.sagpool(h, edge_index, edge_attr, batch)
                g = gap(x_p, b_p)

            h_modulated = self.graph_film(h, g, batch)
        else:
            h_modulated = h

        # 3) Attention & Prediction Logic
        cyp_emb_node = self.cyp_embs[cyp_idx]

        if self.use_attention:
            h_attended, attn_weights = self.attn_head(h_modulated, self.cyp_embs,
                                                      tau=self.attn_tau)
            pred_input = torch.cat([h_modulated, h_attended, cyp_emb_node], dim=-1)
        else:
            pred_input = torch.cat([h_modulated, cyp_emb_node], dim=-1)
            attn_weights = None

        logits = self.pred_head(pred_input).squeeze(-1)

        return logits, h_modulated, attn_weights


# Backwards-compatible alias: earlier checkpoints were trained under this name.
GraphCliffMultiRegressor = ATTNSOM
