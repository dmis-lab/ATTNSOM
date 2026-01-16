import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGPooling, GINConv
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
    def __init__(self, hidden_size, edge_dim, num_layers=3, groups=4, mid_K=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphCliffFilter(hidden_size, edge_dim, groups, short_dropout=dropout*0.5, mid_K=mid_K)
            for _ in range(num_layers)
        ])
    def forward(self, x, edge_index, edge_attr):
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x



class GraphFiLM(nn.Module):
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
        
        attn_weights = F.softmax(attn_scores / tau, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Multi-head Merge
        attn_output = torch.matmul(attn_weights, V) # (num_heads, N, head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(N, H)
        
        output = self.out_proj(attn_output)
        
        avg_attn = attn_weights.mean(dim=0)
        
        return output, avg_attn[:, :num_agents]

class GraphCliffMultiRegressor(nn.Module):
    def __init__(self, atom_in_dim, edge_dim=0, hidden_size=256,
                 num_layers=3, groups=4, mid_K=3, dropout=0.1,
                 cyp_names=None, num_attn_heads=4, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.num_agents = len(cyp_names) if cyp_names else 9

        # 1. Encoding Layers
        self.atom_encoder = AtomEncoder(atom_in_dim, hidden_size, dropout)
        self.encoder = GraphCliffEncoder(hidden_size, edge_dim, num_layers, groups, mid_K, dropout)
        
        # 2. Context & Modulation
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
        cyp_idx = data.cyp_idx
        batch = data.batch

        # 1) Atom/Graph Encoding
        h = self.atom_encoder(x)
        h = self.encoder(h, edge_index, edge_attr)

        # 2) Graph Context (FiLM)
        if batch.max() == 0 and batch.size(0) > 1:
             g = gap(h, batch)
        else:
            x_p, _, _, b_p, _, _ = self.sagpool(h, edge_index, edge_attr, batch)
            g = gap(x_p, b_p)
        
        h_modulated = self.graph_film(h, g, batch)

        # 3) Attention & Prediction Logic
        cyp_emb_node = self.cyp_embs[cyp_idx]
        
        if self.use_attention:
            h_attended, attn_weights = self.attn_head(h_modulated, self.cyp_embs, tau=0.1)
            pred_input = torch.cat([h_modulated, h_attended, cyp_emb_node], dim=-1)
        else:
            pred_input = torch.cat([h_modulated, cyp_emb_node], dim=-1)
            attn_weights = None

        logits = self.pred_head(pred_input).squeeze(-1)

        return logits, h_modulated, attn_weights
    
