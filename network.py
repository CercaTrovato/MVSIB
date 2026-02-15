import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_clustering

# 1.1 编码器（Encoder）与解码器（Decoder）对应论文 Section 2.1
class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)

# 1.2 forward：分别对每个视图进行编码／解码，对应论文中“View-Specific Autoencoder”
class Network(nn.Module):
    def __init__(self, num_views, num_samples, num_clusters, device,
                 input_size, feature_dim,
                 tau=5, eps=1e-3,
                 fn_hn_k=5,       # kNN 邻居数,
                 fn_hn_hidden=64,  # MLP2 隐藏维度
                 membership_mode='softmax_distance',
                 uncertainty_mode='entropy'):

        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        for i in range(num_views):
            self.encoders.append(Encoder(input_size[i], feature_dim))
            self.decoders.append(Decoder(input_size[i], feature_dim))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.num_views = num_views
        self.num_samples = num_samples
        self.num_clusters = num_clusters
        self.device = device
        self.tau = tau
        self.eps = eps
        # SCE 改进配置：支持 legacy 与无参 top2-gap 两条可切换路径，便于做消融对照
        self.membership_mode = membership_mode
        self.uncertainty_mode = uncertainty_mode
        self.step = 0
        self.psedo_labels = torch.zeros(num_samples, dtype=torch.long)
        self.weights = nn.Parameter(torch.full((self.num_views,), 1 / self.num_views), requires_grad=True)
        # Prototype-margin uncertainty threshold kappa: 仅在 epoch=1 初始化一次后固定。
        self.register_buffer('uncertain_kappa', torch.tensor(float('nan')))

        # —— 模块1：簇中心与带宽 σ存储 ——
        self.centers = [None] * (self.num_views + 1)
        self.sigmas = [None] * (self.num_views + 1)

        # —— 模块2：不确定度预测 MLP ——
        hidden_dim = feature_dim // 2
        self.mlp_uncert = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # —— 模块 3：FN/HN 子模块 ——
        self.fn_hn_k = fn_hn_k
        # —— 模块3：FN vs HN 分类器 MLP（输入维度 = num_views + 2） ——
        #   (degree_diff, center_sim, + one affinity per view)
        in_dim = 3
        self.mlp_fn_hn = nn.Sequential(
            nn.Linear(in_dim, fn_hn_hidden),
            nn.ReLU(),
            nn.Linear(fn_hn_hidden, 2)  # 输出 2 类 logits
        )

    def forward(self, xs):
        xrs = []
        zs = []
        for i in range(self.num_views):
            z = self.encoders[i](xs[i])
            xr = self.decoders[i](z)
            xrs.append(xr)
            zs.append(z)

        return xrs, zs

    def get_weights(self):
        softmax_weights = torch.softmax(self.weights, dim=0)
        weights = softmax_weights / torch.sum(softmax_weights)

        return weights

# 1.3 特征融合（fusion）：对应论文 Eq.(4)
    def fusion(self, zs):
        weights = self.get_weights()
        weighted_zs = [z * weight for z, weight in zip(zs, weights)]
        stacked_zs = torch.stack(weighted_zs)
        common_z = torch.sum(stacked_zs, dim=0)

        return common_z

# 1.4 簇中心计算（compute_centers）对应论文 Eq.(6)-(7)
    def compute_centers(self, x, psedo_labels):
        """
        严格复刻原版权重矩阵 @ x 的写法，100% 等价于 F.normalize(one_hot,1)@x
        """
        device = x.device
        ps = psedo_labels.to(device)
        N, d = x.shape
        L = self.num_clusters

        # —— 1) 构造 one-hot 矩阵 (L, N) ——
        weight = torch.zeros(L, N, device=device)
        weight[ps, torch.arange(N, device=device)] = 1.0

        # —— 2) L1 归一化每行 ——
        weight = F.normalize(weight, p=1, dim=1)  # 保证每行加和=1

        # —— 3) centers = weight @ x ——
        centers = weight @ x  # (L, d)

        # —— 4) L2 归一化每个中心 ——
        centers = F.normalize(centers, p=2, dim=1)

        return centers

    def clustering(self, features):
        kwargs = {
            'metric': 'cosine',
            'distributed': False,
            'random_state': 0,
            'n_clusters': self.num_clusters,
            'verbose': False
        }
        if not torch.isfinite(features).all():
            print("[WARN] clustering skipped due to NaN/Inf features; fallback pseudo labels used")
            return (torch.arange(features.size(0), device=features.device) % self.num_clusters).long()
        clustering_model = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
        psedo_labels = clustering_model.fit_predict(features.to(dtype=torch.float64))
        if psedo_labels is None:
            print("[WARN] clustering returned None; fallback pseudo labels used")
            return (torch.arange(features.size(0), device=features.device) % self.num_clusters).long()

        return psedo_labels

    def update_centers(self, zs_list, common_z):
        """
        每隔 tau 调用，用 PyTorchKMeans 计算簇中心 & σ。
        首次调用或任一中心为 None 时强制更新一次。整个过程不跟踪梯度。
        """
        self.step += 1
        need_init = any(c is None for c in self.centers)
        if not need_init and (self.step % self.tau != 0):
            return

        features = zs_list + [common_z]
        # 禁用梯度
        with torch.no_grad():
            for v, z in enumerate(features):
                km = torch_clustering.PyTorchKMeans(
                    init='k-means++',
                    n_clusters=self.num_clusters,
                    max_iter=300,
                    tol=1e-4,
                    metric='euclidean',
                    random_state=0,
                    verbose=False
                )
                # fit_predict 不会记录计算图；若输入异常则跳过本次中心更新以保证训练可继续。
                z64 = z.to(dtype=torch.float64)
                if not torch.isfinite(z64).all():
                    continue
                labels = km.fit_predict(z64)
                if labels is None:
                    continue
                labels = labels.to(self.device).long()
                centers = km.cluster_centers_.to(dtype=z.dtype).to(self.device)  # (L, d)

                # 计算每个样本到其簇中心的距离
                assigned = centers[labels]  # (N, d)
                dists = torch.norm(z - assigned, dim=1)  # (N,)

                # 每簇 σ_k = max(median(cluster_dists), eps)
                sigmas_v = []
                for k in range(self.num_clusters):
                    d_k = dists[labels == k]
                    if d_k.numel() > 0:
                        sigma_k = d_k.median()
                    else:
                        sigma_k = torch.tensor(0.0, device=self.device)
                    sigmas_v.append(torch.clamp(sigma_k, min=self.eps))
                sigmas_v = torch.stack(sigmas_v)  # (L,)

                self.centers[v] = centers
                self.sigmas[v] = sigmas_v

    def compute_membership(self, z, v_index):
        """
        计算视图 v 或共识空间的隶属度 (N×L)。
        - legacy: Gaussian-kernel membership（原始 Eq.(6)-(7) 风格）
        - softmax_distance: softmax(-d/scale_i)（改进版，样本自归一化降低温度超参敏感性）
        """
        centers = self.centers[v_index]  # (L, d)

        if self.membership_mode == 'softmax_distance':
            dists_sq = torch.cdist(z, centers, p=2) ** 2
            # gap-scaled 无参 membership：用最近两中心距离差自归一化，避免 median 尺度造成过平分配。
            top2 = torch.topk(dists_sq, k=min(2, dists_sq.size(1)), largest=False, dim=1).values
            d1 = top2[:, 0:1]
            if top2.size(1) > 1:
                d2 = top2[:, 1:2]
            else:
                d2 = d1 + 1.0
            gap = (d2 - d1).clamp(min=self.eps).detach()
            logits = -(dists_sq - d1) / gap
            membership = torch.softmax(logits, dim=1)
            return membership

        # legacy gaussian path
        sigmas = self.sigmas[v_index]   # (L,)
        dists_sq = torch.cdist(z, centers, p=2) ** 2
        denom = (2 * (sigmas ** 2).clamp(min=self.eps)).view(1, -1)
        unnorm = torch.exp(-dists_sq / denom)
        membership = unnorm / (unnorm.sum(dim=1, keepdim=True) + 1e-12)
        return membership

    def compute_prototype_margin(self, common_z):
        centers = self.centers[self.num_views]
        sim_proto = F.cosine_similarity(common_z.unsqueeze(1), centers.unsqueeze(0), dim=2)
        top2 = torch.topk(sim_proto, k=min(2, sim_proto.size(1)), dim=1).values
        if top2.size(1) == 1:
            return torch.zeros(common_z.size(0), device=common_z.device)
        return top2[:, 0] - top2[:, 1]

    def init_uncertainty_kappa(self, delta_epoch, q, epoch):
        if epoch == 1 and (not torch.isfinite(self.uncertain_kappa)):
            self.uncertain_kappa = torch.quantile(delta_epoch.detach(), q).to(self.uncertain_kappa.device)

    def estimate_uncertainty(self, common_z, sigma_u=0.1):
        # 对应新理论：u=sigmoid((kappa-delta)/sigma_u)，uncertain由 delta<kappa 决定。
        delta = self.compute_prototype_margin(common_z)
        kappa = self.uncertain_kappa
        if not torch.isfinite(kappa):
            kappa = torch.quantile(delta.detach(), 0.8)
        u = torch.sigmoid((kappa - delta) / max(float(sigma_u), self.eps))
        u_hat = self.mlp_uncert(common_z).squeeze(1)
        return u, u_hat, delta

    def classify_fn_hn(self,
                       zs_list, common_z, memberships,
                       batch_psedo_label, certain_mask, uncertain_mask,
                       epoch=100, fn_hn_warmup=10):

        # 1) 如果还没到 warm-up，直接跳过
        if epoch < fn_hn_warmup or not uncertain_mask.any():
            return None, None, None
        device = self.device

        # 0) 统一到同一个 device
        batch_psedo_label = batch_psedo_label.to(device)
        certain_mask = certain_mask.to(device)
        uncertain_mask = uncertain_mask.to(device)

        V = self.num_views
        N = batch_psedo_label.size(0)
        k = self.fn_hn_k
        eps = 1e-12

        # 1) 预计算 kNN distances & indices → dists: (V, N, N)
        dists = torch.stack([
            torch.cdist(z.to(device), z.to(device), p=2)
            for z in zs_list
        ], dim=0)

        # 2) top-(k+1) 排除自身后取前 k，knn_idx: (V, N, k)
        knn_idx = torch.topk(-dists, k + 1, dim=2).indices[:, :, 1:]

        # 3) 取出对应的标签 & certain mask，全部在 GPU 上
        knn_labels = batch_psedo_label[knn_idx]  # (V, N, k)
        certain_mask_knn = certain_mask.unsqueeze(0).unsqueeze(2).expand(V, N, k)  # (V, N, k)

        # 4) 计算 Δd_i 的向量化版
        same = (knn_labels == batch_psedo_label.view(1, N, 1)) & certain_mask_knn
        diff = (~same) & certain_mask_knn
        pos_cnt = same.sum(dim=2).float()  # (V, N)
        neg_cnt = diff.sum(dim=2).float()
        delta = (pos_cnt - neg_cnt) / (pos_cnt + neg_cnt + eps)  # (V, N)
        delta_d = delta.mean(dim=0)  # (N,)

        # 5) 计算 s_i：共识中心相似度
        mu_c = self.centers[V][batch_psedo_label]  # (N, d)
        s = F.cosine_similarity(common_z, mu_c, dim=1)  # (N,)

        # 6) 计算 a_i：跨视图距离均值差
        v_idx = torch.arange(V, device=device)[:, None, None]
        n_idx = torch.arange(N, device=device)[None, :, None]
        neigh_d = dists[v_idx, n_idx, knn_idx]  # (V, N, k)
        d_pos = torch.where(same, neigh_d, torch.nan).nanmean(dim=2)
        d_neg = torch.where(diff, neigh_d, torch.nan).nanmean(dim=2)
        d_pos = torch.nan_to_num(d_pos, nan=0.0)
        d_neg = torch.nan_to_num(d_neg, nan=0.0)
        a = ((d_neg - d_pos) / (d_neg + d_pos + eps)).mean(dim=0)  # (N,)

        # 7) 筛出不确定样本，拼接特征，走 MLP
        idx_uncertain = uncertain_mask.nonzero(as_tuple=True)[0]  # (M,)
        feats_all = torch.stack([delta_d, s, a], dim=1)  # (N,3)
        feats = feats_all[idx_uncertain]  # (M,3)
        logits_raw = self.mlp_fn_hn(feats)  # (M,2)

        # —— 改动开始：按置信度分位数丢弃最低 10% —— #
        probs = F.softmax(logits_raw, dim=1)  # (M,2)
        conf, preds = probs.max(dim=1)  # (M,)

        # 丢弃最低 10% 的 conf
        thresh = torch.quantile(conf, 0.1)
        keep = conf > thresh

        if keep.sum() == 0:
            return None, None, None

        idx_kept = idx_uncertain[keep]  # 选出剩余的 90%
        logits = logits_raw[keep]
        feats = feats[keep]
        # —— 改动结束 —— #

        return logits, idx_kept, feats

    def compute_consistency_scores(self, memberships, batch_psedo_label):
        """
        计算跨视图一致性分数 S_ij ∈ [0,1]，形状 (N, N)。
        Args:
            memberships (list[Tensor[N×L]]): 包含 V 个视图和 1 个共识空间
            batch_psedo_label (Tensor[N]): 每个样本在共识空间的伪标签
        Returns:
            S (Tensor[N×N]): 对称矩阵，S[i,j] 表示对 (i,j) 的一致性分数
        """
        N = batch_psedo_label.size(0)
        V = self.num_views
        device = self.device

        # 1) 提取各视图和共识的 “conf” 向量
        same = (batch_psedo_label.view(-1, 1) == batch_psedo_label.view(1, -1)).float().to(device)
        conf_v = torch.stack([
            memberships[v][torch.arange(N, device=device), batch_psedo_label]
            for v in range(V)
        ], dim=0).to(device)  # (V, N)

        conf_c = memberships[V][torch.arange(N, device=device), batch_psedo_label].to(device)  # (N,)

        # 2) 计算每个样本可靠性 r_i（由归一化熵不确定度导出，方向与置信度一致）
        u_vs = []
        for v in range(V):
            m = memberships[v]  # (N, L)
            entropy_v = -torch.sum(m * torch.log(m + self.eps), dim=1)
            u_vs.append((entropy_v / torch.log(torch.tensor(float(self.num_clusters), device=m.device))).clamp(0.0, 1.0))

        u_i = torch.stack(u_vs, dim=1).max(dim=1).values
        r = torch.clamp(1.0 - u_i, 0.2, 1.0)
        rij = r.view(-1, 1) * r.view(1, -1)

        # 3) 计算对称 pairwise 一致性：conf(i) * conf(j) * 1[y_i=y_j]
        s_view = (conf_v.unsqueeze(2) * conf_v.unsqueeze(1) * same.unsqueeze(0)).mean(dim=0)
        s_cons = conf_c.view(-1, 1) * conf_c.view(1, -1) * same

        # 4) 最终一致性 S：视图/共识对称融合 + 可靠性对称加权
        S = 0.5 * (s_view + s_cons) * rij * same
        S = S.masked_fill(torch.eye(N, device=device, dtype=torch.bool), 0.0)
        return S
