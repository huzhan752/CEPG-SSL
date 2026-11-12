
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ISTF_attention import SelfAttentionLayer

class GLU(nn.Module):

    def __init__(self, features, dropout=0.1):
        super(GLU, self).__init__()
        self.conv1 = nn.Conv2d(features, features, (1, 1))
        self.conv2 = nn.Conv2d(features, features, (1, 1))
        self.conv3 = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)
        return out


class TemporalPositionalEncoding(nn.Module):

    def __init__(self, features, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, features)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, features, 2, dtype=torch.float)
            * -(math.log(10000.0) / features)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if features > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, seq_len, num_nodes):
        return self.pe[:seq_len].unsqueeze(0).unsqueeze(0).expand(1, num_nodes, seq_len, -1)


class TemporalEmbedding(nn.Module):

    def __init__(self, time, features, max_len=1024):
        super(TemporalEmbedding, self).__init__()
        self.time = time
        self.features = features
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)
        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)
        self.time_position = TemporalPositionalEncoding(features, max_len=max_len)

    def forward(self, x):
        B, T, N, _ = x.shape
        day_emb = x[..., 1]
        day_index = torch.clamp((day_emb * self.time).long(), min=0, max=self.time - 1).to(x.device)
        time_day = self.time_day[day_index]
        time_day = time_day.transpose(1, 2).contiguous()  # [B, N, T, features]

        week_emb = x[..., 2]
        week_index = torch.clamp(week_emb.long(), min=0, max=6).to(x.device)
        time_week = self.time_week[week_index]
        time_week = time_week.transpose(1, 2).contiguous()  # [B, N, T, features]

        time_pos = self.time_position(T, N).to(device=x.device, dtype=time_day.dtype)
        tem_emb = time_day + time_week + time_pos
        tem_emb = tem_emb.permute(0, 3, 1, 2)  # [B, features, N, T]
        return tem_emb


class Diffusion_GCN(nn.Module):

    def __init__(self, channels=128, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.diffusion_step = diffusion_step
        self.conv = nn.Conv2d(diffusion_step * channels, channels, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        out = []
        for _ in range(0, self.diffusion_step):
            if adj.dim() == 3:
                x = torch.einsum("bcnt,bnm->bcmt", x, adj).contiguous()
                out.append(x)
            elif adj.dim() == 2:
                x = torch.einsum("bcnt,nm->bcmt", x, adj).contiguous()
                out.append(x)
            else:
                raise ValueError("adj must have shape [N, N] or [B, N, N].")
        x = torch.cat(out, dim=1)
        x = self.conv(x)
        output = self.dropout(x)
        return output


class MultiGraph_Fusion(nn.Module):

    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(channels, num_nodes))
        nn.init.xavier_uniform_(self.memory)
        self.fc = nn.Linear(2, 1)
        self.E_adaptive = nn.Parameter(torch.randn(num_nodes, 10))

        # DPDG dynamic graph: Delayed Propagation Dynamic Graph.
        # Only the original dynamic graph generation is replaced; the subsequent
        # adaptive graph, physical graph Hadamard product, fusion and top-k logic
        # are kept unchanged.
        self.max_lag = 3
        self.dpdg_q = nn.Conv2d(channels, channels, (1, 1), bias=False)
        self.dpdg_k = nn.Conv2d(channels, channels, (1, 1), bias=False)
        self.dpdg_lag_weight = nn.Parameter(torch.zeros(self.max_lag))

    def _delayed_propagation_dynamic_graph(self, x):
        B, C, N, T = x.shape
        if T <= 1:
            raise ValueError("Delayed Propagation Dynamic Graph requires temporal length T > 1.")

        # Delayed propagation uses current node states as queries and
        # historical node states as keys to model asynchronous propagation.
        q = self.dpdg_q(x)
        k = self.dpdg_k(x)

        max_lag = min(self.max_lag, T - 1)
        lag_weight = torch.softmax(self.dpdg_lag_weight[:max_lag], dim=0)

        # score_tgt_src[:, i, j] measures how well node j at historical lag
        # explains node i at the current time. The diffusion layer uses
        # adjacency as source->target, i.e., x'[:, target] = sum_source
        # x[:, source] * A[source, target]. Therefore we transpose the score
        # before row-wise softmax so that A[j, i] denotes propagation j -> i.
        score_tgt_src = x.new_zeros(B, N, N)
        for idx, lag in enumerate(range(1, max_lag + 1)):
            q_now = q[..., lag:]
            k_past = k[..., :-lag]
            score_lag = torch.einsum("bcnt,bcmt->bnm", q_now, k_past).contiguous()
            score_lag = score_lag / (math.sqrt(C) * max(1, q_now.shape[-1]))
            score_tgt_src = score_tgt_src + lag_weight[idx] * score_lag

        score_src_tgt = score_tgt_src.transpose(-1, -2).contiguous()
        return torch.softmax(F.relu(score_src_tgt), -1)

    def forward(self, x, AG):
        adj_dyn_2 = self._delayed_propagation_dynamic_graph(x)

        adj_adp = F.softmax(F.relu(torch.mm(self.E_adaptive, self.E_adaptive.transpose(0, 1))), dim=1)

        B = x.shape[0]
        if AG.dim() == 2:
            AG_batch = AG.unsqueeze(0).expand(B, -1, -1)
        elif AG.dim() == 3:
            AG_batch = AG
        else:
            raise ValueError("AG must have shape [N, N] or [B, N, N].")

        adj_adp = adj_adp.unsqueeze(0) * AG_batch

        adj_f = torch.cat([(adj_dyn_2).unsqueeze(-1)] + [(adj_adp).unsqueeze(-1)], dim=-1)
        adj_f = torch.softmax(self.fc(adj_f).squeeze(-1), -1)

        k = max(1, int(adj_f.shape[1] * 0.8))
        _, topk_indices = torch.topk(adj_f, k=k, dim=-1)
        mask = torch.zeros_like(adj_f)
        mask.scatter_(-1, topk_indices, 1)
        adj_f = adj_f * mask
        return adj_f


class MGFDGCN(nn.Module):

    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1, emb=None):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, (1, 1))
        self.generator = MultiGraph_Fusion(channels, num_nodes, diffusion_step, dropout)
        self.gcn = Diffusion_GCN(channels, diffusion_step, dropout)
        self.emb = emb

    def forward(self, x, AG):
        skip = x
        x = self.conv(x)
        adj_dyn = self.generator(x, AG)
        x = self.gcn(x, adj_dyn)
        x = x * self.emb + skip
        return x


@torch.no_grad()
def sinkhorn(out, epsilon=0.05, sinkhorn_iterations=3):
    Q = torch.exp(out / epsilon).t()  # [K, M]
    B = Q.shape[1]
    K = Q.shape[0]

    Q /= torch.sum(Q).clamp_min(1e-12)

    for _ in range(sinkhorn_iterations):
        Q /= torch.sum(Q, dim=1, keepdim=True).clamp_min(1e-12)
        Q /= K

        Q /= torch.sum(Q, dim=0, keepdim=True).clamp_min(1e-12)
        Q /= B

    Q *= B
    return Q.t()


class PivotalGraphGuidedRegionRoleLearning(nn.Module):

    def __init__(
        self,
        c_in,
        nmb_prototype=6,
        cluster_tau=0.5,
        proj_dim=None,
        sinkhorn_epsilon=0.05,
        sinkhorn_iterations=3,
    ):
        super(PivotalGraphGuidedRegionRoleLearning, self).__init__()
        self.d_model = c_in
        self.proj_dim = proj_dim or c_in
        self.nmb_prototype = nmb_prototype
        self.cluster_tau = cluster_tau
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iterations = sinkhorn_iterations

        self.proj = nn.Sequential(
            nn.Linear(c_in, self.proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.proj_dim, self.proj_dim),
        )
        self.prototypes = nn.Linear(self.proj_dim, nmb_prototype, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.prototypes.weight)

    @staticmethod
    def l2norm(x, dim=-1):
        return F.normalize(x, p=2, dim=dim)

    def forward(self, z_ori, z_piv, node_weight=None):
        B, T, N, C = z_ori.shape
        if N <= 1:
            return z_ori.new_tensor(0.0)

        with torch.no_grad():
            w = F.normalize(self.prototypes.weight.data, p=2, dim=1)
            self.prototypes.weight.copy_(w)

        q = self.proj(z_ori)              # [B,T,N,D], learnable path
        q = self.l2norm(q, dim=-1)

        with torch.no_grad():
            k = self.proj(z_piv.detach())
            k = self.l2norm(k, dim=-1)
            flat_k = k.reshape(B * T * N, self.proj_dim)
            teacher_scores = self.prototypes(flat_k).float()
            q_target = sinkhorn(
                teacher_scores,
                epsilon=self.sinkhorn_epsilon,
                sinkhorn_iterations=self.sinkhorn_iterations,
            )
            q_target = q_target.clamp_min(1e-8)
            q_target = q_target / q_target.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        flat_q = q.reshape(B * T * N, self.proj_dim)
        query_scores = self.prototypes(flat_q).float() / self.cluster_tau
        log_prob = F.log_softmax(query_scores, dim=-1)
        per_sample_loss = -torch.sum(q_target * log_prob, dim=-1).reshape(B, T, N)

        if node_weight is None:
            return per_sample_loss.mean()

        node_weight = node_weight.to(device=z_ori.device, dtype=per_sample_loss.dtype)
        if node_weight.dim() == 1:
            weight = node_weight.view(1, 1, N).expand(B, T, N)
        elif node_weight.dim() == 2:
            weight = node_weight.view(B, 1, N).expand(B, T, N)
        else:
            raise ValueError("node_weight must be None, [N], or [B,N].")
        return (per_sample_loss * weight).sum() / weight.sum().clamp_min(1e-8)


class CitywideTrafficStateDiscrimination(nn.Module):

    def __init__(self, c_in, num_nodes, filter_delta=1):
        super(CitywideTrafficStateDiscrimination, self).__init__()
        self.num_nodes = num_nodes
        self.filter_delta = int(filter_delta)
        self.w1 = nn.Parameter(torch.ones(c_in))
        self.w2 = nn.Parameter(torch.ones(c_in))
        self.context_weight = nn.Parameter(torch.empty(c_in, c_in))
        nn.init.xavier_uniform_(self.context_weight)

    def _sample_negative_time(self, T, device):
        if T <= 1:
            return torch.zeros(T, dtype=torch.long, device=device)

        idx = torch.arange(T, device=device)
        dist = (idx.view(T, 1) - idx.view(1, T)).abs()
        valid = dist > max(0, int(self.filter_delta))

        fallback = dist > 0
        row_has_valid = valid.any(dim=1, keepdim=True)
        valid = torch.where(row_has_valid, valid, fallback)

        probs = valid.float()
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        neg_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return neg_idx

    def forward(self, z1, z2):
        B, T, N, C = z1.shape
        if T <= 1:
            return z1.new_tensor(0.0)

        z2 = z2.detach()
        w1 = self.w1.view(1, 1, 1, C)
        w2 = self.w2.view(1, 1, 1, C)

        v = w1 * z1 + w2 * z2  # [B,T,N,C]

        s = torch.sigmoid(v.mean(dim=2))  # [B,T,C]

        neg_idx = self._sample_negative_time(T, z1.device)  # [T]
        s_neg = s[:, neg_idx, :]  # [B,T,C]

        pos_logits = torch.einsum("btnc,cd,btd->btn", v, self.context_weight, s)
        neg_logits = torch.einsum("btnc,cd,btd->btn", v, self.context_weight, s_neg)

        pos_loss = F.binary_cross_entropy_with_logits(pos_logits.float(), torch.ones_like(pos_logits).float())
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits.float(), torch.zeros_like(neg_logits).float())
        return pos_loss + neg_loss


class TemporalEvolutionReconstruction(nn.Module):

    def __init__(
        self,
        input_dim,
        c_in,
        max_delta=3,
        kernel_size=3,
        dropout=0.1,
    ):
        super(TemporalEvolutionReconstruction, self).__init__()
        self.max_delta = int(max_delta)
        self.kernel_size = max(1, int(kernel_size))

        self.raw_proj = nn.Conv2d(input_dim, c_in, kernel_size=(1, 1))
        self.causal_conv = nn.Conv2d(c_in, c_in, kernel_size=(1, self.kernel_size))
        self.norm = nn.LayerNorm(c_in)
        self.lineage_embedding = nn.Embedding(self.max_delta + 1, c_in)
        self.predictor = nn.Sequential(
            nn.Conv2d(c_in, c_in, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(c_in, c_in, kernel_size=(1, 1)),
        )
        self.online_proj = nn.Conv2d(c_in, c_in, kernel_size=(1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.raw_proj.weight)
        if self.raw_proj.bias is not None:
            nn.init.zeros_(self.raw_proj.bias)
        nn.init.xavier_uniform_(self.causal_conv.weight)
        if self.causal_conv.bias is not None:
            nn.init.zeros_(self.causal_conv.bias)
        nn.init.xavier_uniform_(self.lineage_embedding.weight)
        for module in self.predictor.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.online_proj.weight)
        if self.online_proj.bias is not None:
            nn.init.zeros_(self.online_proj.bias)

    def causal_encode(self, inputx):
        h = self.raw_proj(inputx)
        if self.kernel_size > 1:
            h = F.pad(h, (self.kernel_size - 1, 0, 0, 0))
        h = self.causal_conv(h)
        h = self.norm(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        h = F.relu(h, inplace=True)
        return h

    @staticmethod
    def cosine_distance(x, y):
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        return 1.0 - (x * y).sum(dim=1)

    def forward(self, H, H_T, inputx):
        B, C, N, T = H.shape
        D = min(self.max_delta, T - 1)
        if D <= 0:
            return H.new_tensor(0.0)

        delta = int(torch.randint(1, D + 1, (1,), device=torch.device("cpu")).item())

        C_causal = self.causal_encode(inputx)  # [B,C,N,T]
        past_state = C_causal[:, :, :, :-delta]        # C(t-delta), [B,C,N,T-d]
        teacher_now = H_T.detach()[:, :, :, delta:]    # H_T(t), [B,C,N,T-d]
        online_now = H[:, :, :, delta:]                # H(t), gradients enabled

        lineage = self.lineage_embedding.weight[delta].view(1, C, 1, 1)
        pred_now = self.predictor(past_state + lineage)

        rec_loss = self.cosine_distance(pred_now, teacher_now).mean()

        online_pred = self.online_proj(online_now)
        align_loss = self.cosine_distance(online_pred, pred_now.detach()).mean()

        return 0.5 * (rec_loss + align_loss)


class STTBCL(nn.Module):

    def __init__(
        self,
        device,
        input_dim,
        num_nodes,
        channels,
        granularity,
        supports,
        dropout=0.1,
        dilation_channels=32,
        temporal_aug_ratio=0.1,
        spatial_aug_ratio=0.0,
        nmb_prototype=6,
        shm_tau=0.5,
        temporal_point_mask_min=0.03,
        temporal_point_mask_max=0.10,
        temporal_segment_prob=0.20,
        temporal_segment_max_len=3,
        temporal_jitter_std=0.01,
        spatial_balance_weight=0.05,
        spatial_confidence_weight=0.01,
        spatial_edge_drop_ratio=0.0,
        spatial_edge_add_ratio=0.0,
        pivotal_topk_ratio=0.2,
        pivotal_delay=1,
        pivotal_node_mask_ratio=0.0,
        pivotal_weight_beta=0.5,
        temporal_filter_delta=1,
        lineage_max_delta=3,
        lineage_kernel_size=3,
        lineage_loss_weight=1.0,
    ):
        super().__init__()

        self.supports = supports
        self.device = device
        self.num_nodes = num_nodes
        self.output_len = 12
        self.input_len = 12

        self.temporal_aug_ratio = temporal_aug_ratio
        self.spatial_aug_ratio = spatial_aug_ratio

        self.temporal_point_mask_min = temporal_point_mask_min
        self.temporal_point_mask_max = temporal_point_mask_max
        self.temporal_segment_prob = temporal_segment_prob
        self.temporal_segment_max_len = temporal_segment_max_len
        self.temporal_jitter_std = temporal_jitter_std

        self.spatial_edge_drop_ratio = spatial_edge_drop_ratio
        self.spatial_edge_add_ratio = spatial_edge_add_ratio

        self.pivotal_topk_ratio = pivotal_topk_ratio
        self.pivotal_delay = max(1, int(pivotal_delay))
        self.pivotal_node_mask_ratio = pivotal_node_mask_ratio
        self.pivotal_weight_beta = float(pivotal_weight_beta)
        self.temporal_filter_delta = int(temporal_filter_delta)
        self.lineage_loss_weight = float(lineage_loss_weight)

        self.spatial_balance_weight = spatial_balance_weight
        self.spatial_confidence_weight = spatial_confidence_weight

        diffusion_step = 1
        hidden_channels = channels * 2

        self.Temb = TemporalEmbedding(granularity, channels)
        self.start_conv = nn.Conv2d(in_channels=input_dim, out_channels=channels, kernel_size=(1, 1))
        self.glu = GLU(hidden_channels, dropout)
        self.regression_layer = nn.Conv2d(hidden_channels, self.output_len, kernel_size=(1, self.output_len))

        num_layers = 3
        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(hidden_channels, feed_forward_dim=256, num_heads=4, dropout=0.1)
                for _ in range(num_layers)
            ]
        )

        self.mgf_emb = nn.Parameter(torch.randn(hidden_channels, num_nodes, self.input_len))
        self.MGF_DGCN = MGFDGCN(
            hidden_channels,
            num_nodes,
            diffusion_step=diffusion_step,
            dropout=dropout,
            emb=self.mgf_emb,
        )

        self.pim_weight = nn.Parameter(torch.ones(num_nodes, num_nodes), requires_grad=False)

        self.temporal_ssl = CitywideTrafficStateDiscrimination(
            hidden_channels,
            num_nodes,
            filter_delta=self.temporal_filter_delta,
        )
        self.lineage_ssl = TemporalEvolutionReconstruction(
            input_dim=input_dim,
            c_in=hidden_channels,
            max_delta=lineage_max_delta,
            kernel_size=lineage_kernel_size,
            dropout=dropout,
        )
        self.spatial_ssl = PivotalGraphGuidedRegionRoleLearning(
            hidden_channels,
            nmb_prototype=nmb_prototype,
            cluster_tau=shm_tau,
            sinkhorn_epsilon=0.05,
            sinkhorn_iterations=3,
        )

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def encode_view(self, inputx, adj):
        time_emb = self.Temb(inputx.permute(0, 3, 2, 1))

        x = torch.cat([self.start_conv(inputx), time_emb], dim=1)  # [B, 2C, N, T]

        x = x.permute(0, 3, 2, 1)  # [B, T, N, 2C]
        for attn in self.attn_layers_t:
            x = attn(x, dim=1)
        x = x.permute(0, 3, 2, 1)  # [B, 2C, N, T]

        x = self.MGF_DGCN(x, adj)
        x = self.glu(x) + x
        return x

    def temporal_augmentation(self, inputx, H):
        with torch.no_grad():
            x_aug = inputx.clone()
            flow = inputx[:, 0, :, :]  # [B,N,T]
            B, N, T = flow.shape

            z = H.detach().permute(0, 3, 2, 1).contiguous()
            z = F.normalize(z, p=2, dim=-1)
            mean_t = F.normalize(z.mean(dim=1, keepdim=True), p=2, dim=-1)
            sim = (z * mean_t).sum(dim=-1)  # [B,T,N]

            sim_prob = F.softmax(sim, dim=1)  # normalize along time for each node
            drop_raw = 1.0 - sim_prob
            drop_prob = drop_raw / drop_raw.mean(dim=(1, 2), keepdim=True).clamp_min(1e-8)
            drop_prob = (drop_prob * float(self.temporal_aug_ratio)).clamp(min=0.0, max=0.95)

            drop_prob = drop_prob.permute(0, 2, 1).contiguous()
            keep_mask = (torch.rand_like(flow) > drop_prob).to(flow.dtype)

            x_aug[:, 0, :, :] = flow * keep_mask
            return x_aug

    def build_pivotal_graph(self, inputx, adj, return_weight=False):
        with torch.no_grad():
            flow = inputx[:, 0, :, :].float()  # [B, N, T]
            B, N, T = flow.shape
            device = flow.device
            dtype = adj.dtype
            d = max(1, min(int(self.pivotal_delay), T - 1)) if T > 1 else 1

            if T <= d:
                A = adj.unsqueeze(0).expand(B, -1, -1).clone().to(device=device, dtype=dtype)
                eye = torch.eye(N, device=device, dtype=torch.bool)
                A[:, eye] = 1.0
                A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                if return_weight:
                    weight = torch.ones(B, N, device=device, dtype=dtype)
                    return A, weight
                return A

            x_now = flow[:, :, d:]      # [B, N, T-d]
            x_prev = flow[:, :, :-d]    # [B, N, T-d]

            numerator = torch.einsum("bit,bjt->bij", x_now, x_prev)
            norm_now = torch.sqrt((x_now ** 2).sum(dim=-1).clamp_min(1e-12))   # [B,N]
            norm_prev = torch.sqrt((x_prev ** 2).sum(dim=-1).clamp_min(1e-12)) # [B,N]
            denom = norm_now.unsqueeze(-1) * norm_prev.unsqueeze(1)

            W = self.pim_weight.to(device=device, dtype=flow.dtype)
            E = numerator / denom.clamp_min(1e-12)
            E = E * W.unsqueeze(0)
            E = torch.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)

            score = E.sum(dim=-1) + E.sum(dim=-2)  # [B,N]
            k = max(1, int(N * float(self.pivotal_topk_ratio)))
            k = min(k, N)
            topk_idx = torch.topk(score, k=k, dim=-1).indices  # [B,K]

            piv_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
            piv_mask.scatter_(1, topk_idx, True)
            related = piv_mask.unsqueeze(-1) | piv_mask.unsqueeze(1)  # [B,N,N]

            node_weight = torch.ones(B, N, device=device, dtype=flow.dtype)
            node_weight = node_weight + self.pivotal_weight_beta * piv_mask.to(flow.dtype)

            base = adj.to(device=device, dtype=flow.dtype).unsqueeze(0).expand(B, -1, -1)
            A_piv = torch.where(related, torch.sigmoid(E), base)

            A_piv = torch.clamp(A_piv, min=0.0)
            eye = torch.eye(N, device=device, dtype=torch.bool)
            A_piv[:, eye] = 1.0
            A_piv = A_piv / A_piv.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        A_piv = A_piv.to(dtype=dtype)
        if return_weight:
            return A_piv, node_weight.to(dtype=dtype)
        return A_piv

    def pivotal_graph_augmentation(self, A_piv):
        with torch.no_grad():
            input_was_2d = (A_piv.dim() == 2)
            if input_was_2d:
                A_aug = A_piv.unsqueeze(0).clone()
            elif A_piv.dim() == 3:
                A_aug = A_piv.clone()
            else:
                raise ValueError("A_piv must have shape [N, N] or [B, N, N].")

            B, N, _ = A_aug.shape
            device = A_aug.device
            dtype = A_aug.dtype
            eye = torch.eye(N, dtype=torch.bool, device=device)

            for b in range(B):
                Ab = A_aug[b]

                edge_mask = (Ab > 0) & (~eye)
                edge_pos = edge_mask.nonzero(as_tuple=False)
                num_edges = edge_pos.shape[0]

                drop_num = int(num_edges * float(self.spatial_edge_drop_ratio))
                if drop_num > 0 and num_edges > 0:
                    drop_num = min(drop_num, num_edges)
                    perm = torch.randperm(num_edges, device=device)[:drop_num]
                    drop_edges = edge_pos[perm]
                    Ab[drop_edges[:, 0], drop_edges[:, 1]] = torch.zeros((), dtype=dtype, device=device)

                candidate_mask = (Ab <= 0) & (~eye)
                cand_pos = candidate_mask.nonzero(as_tuple=False)
                add_num = int(num_edges * float(self.spatial_edge_add_ratio))

                if add_num > 0 and cand_pos.shape[0] > 0:
                    add_num = min(add_num, cand_pos.shape[0])
                    perm = torch.randperm(cand_pos.shape[0], device=device)[:add_num]
                    add_edges = cand_pos[perm]

                    positive_values = Ab[(Ab > 0) & (~eye)]
                    if positive_values.numel() > 0:
                        add_weight = positive_values.mean().to(dtype=dtype)
                    else:
                        add_weight = torch.tensor(1.0 / max(N, 1), dtype=dtype, device=device)
                    Ab[add_edges[:, 0], add_edges[:, 1]] = add_weight

                node_ratio = float(self.pivotal_node_mask_ratio)
                if node_ratio > 0:
                    mask_num = int(N * node_ratio)
                    if mask_num > 0:
                        mask_num = min(mask_num, N)
                        nodes = torch.randperm(N, device=device)[:mask_num]
                        Ab[nodes, :] = 0.0
                        Ab[:, nodes] = 0.0

                Ab[eye] = torch.ones((), dtype=dtype, device=device)
                Ab = Ab / Ab.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                A_aug[b] = Ab

            if input_was_2d:
                A_aug = A_aug[0]

        return A_aug

    def forward(self, inputx):
        A = self.supports[0].float()

        H = self.encode_view(inputx, A)
        prediction = self.regression_layer(F.relu(H))

        if not self.training:
            zero = torch.tensor(0.0, dtype=H.dtype, device=H.device)
            return prediction, zero, zero

        X_T = self.temporal_augmentation(inputx, H)
        with torch.no_grad():
            H_T = self.encode_view(X_T, A)

        A_piv, piv_weight = self.build_pivotal_graph(inputx, A, return_weight=True)
        with torch.no_grad():
            H_piv = self.encode_view(inputx, A_piv)

        Z = H.permute(0, 3, 2, 1).contiguous()
        Z_T = H_T.detach().permute(0, 3, 2, 1).contiguous()
        Z_P = H_piv.detach().permute(0, 3, 2, 1).contiguous()

        loss_ctsd = self.temporal_ssl(Z, Z_T)
        loss_ter = self.lineage_ssl(H, H_T.detach(), inputx)
        loss_t = 0.1 * loss_ctsd + self.lineage_loss_weight * loss_ter
        loss_s = self.spatial_ssl(Z, Z_P, node_weight=piv_weight)

        return prediction, loss_t, loss_s
