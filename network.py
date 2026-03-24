import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_clustering

# ============================================================================
# network.py
# ----------------------------------------------------------------------------
# 本文件定义 MVSIB 的核心网络结构：
# 1) 每个视图的自编码器（Encoder/Decoder）
# 2) 跨视图加权融合模块
# 3) 聚类中心/带宽的周期更新
# 4) 软隶属度计算（legacy + 改进 softmax-distance）
# 5) 不确定度估计（legacy + log-odds）
# 6) FN/HN 分类特征提取与一致性评分
#
# 设计原则：
# - 保留旧路径（legacy）用于可复现实验对照
# - 新路径通过超参数开关启用（便于做 ablation）
# ============================================================================


# 1.1 编码器（Encoder）与解码器（Decoder）对应论文 Section 2.1
class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        # input_dim: 当前视图输入特征维度
        # feature_dim: 编码后的潜在表示维度（共享语义空间）
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
        # x: [batch_size, input_dim] 当前视图输入
        # 返回: [batch_size, feature_dim] 视图潜表示 z
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        # input_dim: 重建目标维度（应与原视图输入一致）
        # feature_dim: 编码器输出维度
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
        # x: [batch_size, feature_dim]
        # 返回: [batch_size, input_dim] 视图重建 xr
        return self.decoder(x)


# 1.2 forward：分别对每个视图进行编码／解码，对应论文中“View-Specific Autoencoder”
class Network(nn.Module):
    def __init__(self, num_views, num_samples, num_clusters, device,
                 input_size, feature_dim,
                 tau=5, eps=1e-3,
                 fn_hn_k=5,
                 fn_hn_hidden=64,
                 membership_mode='softmax_distance',
                 membership_temperature=1.0,
                 uncertainty_mode='log_odds',
                 uncertainty_kappa=1.0,
                 uncertainty_temperature=0.5,
                 reliability_temperature=0.5):
        """
        参数说明（含超参数调节影响）：
        - num_views: 视图数 V，决定编码器/解码器个数。
        - num_samples: 全数据样本数 N，用于初始化伪标签缓冲区。
        - num_clusters: 聚类簇数 L。
        - device: 运行设备（cpu/cuda）。
        - input_size: list[int]，每个视图的输入维度。
        - feature_dim: 共享潜在空间维度 D；增大可提高表达能力但更易过拟合。
        - tau: 每 tau 步更新一次中心和 sigma；
               较小=更新更频繁（更敏感，计算更重），较大=更平滑（可能滞后）。
        - eps: 数值稳定常数；过小可能数值不稳，过大可能引入偏置。
        - fn_hn_k: FN/HN 特征提取的 kNN 邻居数；
                   大则统计更稳健，小则更局部。
        - fn_hn_hidden: FN/HN 分类 MLP 隐层维度；
                        大则容量更强，但过大可能过拟合。
        - membership_mode: 隶属度模式：
            * 'softmax_distance'：改进路径（默认，抗 sigma 噪声）
            * 其他值走 legacy gaussian
        - membership_temperature (T_m): 距离 softmax 温度；
            小 -> 分配更尖锐，大 -> 更平滑。
        - uncertainty_mode: 不确定度模式：'legacy' 或 'log_odds'。
        - uncertainty_kappa: log-odds 模式中“边界阈值”；
            大 -> 更容易判为不确定（u 更大）。
        - uncertainty_temperature (T_u): 不确定度 sigmoid 温度；
            小 -> 更陡峭，大 -> 更平滑。
        - reliability_temperature (T_w): 视图可靠性 softmax 温度；
            小 -> 更偏向最佳视图，大 -> 更平均融合。
        """
        super(Network, self).__init__()

        # ---- 视图私有编码器/解码器 ----
        self.encoders = []
        self.decoders = []
        for i in range(num_views):
            self.encoders.append(Encoder(input_size[i], feature_dim))
            self.decoders.append(Decoder(input_size[i], feature_dim))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        # ---- 数据与结构元信息 ----
        self.num_views = num_views
        self.num_samples = num_samples
        self.num_clusters = num_clusters
        self.device = device

        # ---- 训练控制超参数 ----
        self.tau = tau
        self.eps = eps

        # SCE 改进配置：保留 legacy 与 log-odds 两条路径，便于消融
        self.membership_mode = membership_mode
        self.membership_temperature = membership_temperature
        self.uncertainty_mode = uncertainty_mode
        self.uncertainty_kappa = uncertainty_kappa
        self.uncertainty_temperature = uncertainty_temperature
        self.reliability_temperature = reliability_temperature

        # step: 当前调用 update_centers 的计数器（用于 tau 周期更新）
        self.step = 0

        # psedo_labels: 全量伪标签缓冲区（注意代码历史拼写为 psedo）
        self.psedo_labels = torch.zeros(num_samples, dtype=torch.long)

        # views 权重参数，softmax 后用于 fusion
        # 初始为均匀权重 1/V；训练后可自适应各视图贡献
        self.weights = nn.Parameter(torch.full((self.num_views,), 1 / self.num_views), requires_grad=True)

        # 模块1：簇中心与带宽 σ存储
        # 长度 V+1：前 V 个是各视图中心，最后一个是共识空间中心
        self.centers = [None] * (self.num_views + 1)
        self.sigmas = [None] * (self.num_views + 1)

        # 模块2：不确定度预测 MLP（从 common_z 回归 u_hat）
        hidden_dim = feature_dim // 2
        self.mlp_uncert = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 模块3：FN/HN 分类器 MLP
        self.fn_hn_k = fn_hn_k
        # in_dim=3 对应三个手工统计特征：(degree_diff, center_sim, affinity_gap)
        in_dim = 3
        self.mlp_fn_hn = nn.Sequential(
            nn.Linear(in_dim, fn_hn_hidden),
            nn.ReLU(),
            nn.Linear(fn_hn_hidden, 2)  # 2 类 logits: FN vs HN
        )

    def forward(self, xs):
        # xs: list[Tensor]，长度为 num_views，每个元素形状 [B, input_size[v]]
        xrs = []
        zs = []
        for i in range(self.num_views):
            # z: 第 i 个视图编码后的潜表示
            z = self.encoders[i](xs[i])
            # xr: 对 z 做解码得到重建
            xr = self.decoders[i](z)
            xrs.append(xr)
            zs.append(z)

        # xrs: 每个视图重建；zs: 每个视图潜变量
        return xrs, zs

    def get_weights(self):
        # 使用 softmax 保证权重非负且可微
        softmax_weights = torch.softmax(self.weights, dim=0)
        # 再次归一化以避免潜在累计误差
        weights = softmax_weights / torch.sum(softmax_weights)
        return weights

    # 1.3 特征融合（fusion）：对应论文 Eq.(4)
    def fusion(self, zs):
        # zs: list[Tensor[B,D]]
        weights = self.get_weights()
        # 按视图权重做逐视图缩放
        weighted_zs = [z * weight for z, weight in zip(zs, weights)]
        # 堆叠后对视图维求和，得到共识表示 common_z
        stacked_zs = torch.stack(weighted_zs)
        common_z = torch.sum(stacked_zs, dim=0)
        return common_z

    # 1.4 簇中心计算（compute_centers）对应论文 Eq.(6)-(7)
    def compute_centers(self, x, psedo_labels):
        """
        输入:
        - x: [N,D] 需要聚类的特征
        - psedo_labels: [N] 当前伪标签
        输出:
        - centers: [L,D] 每簇中心（L2 normalize）
        """
        device = x.device
        ps = psedo_labels.to(device)
        N, d = x.shape
        L = self.num_clusters

        # 1) 构造 one-hot 指派矩阵 weight[L,N]
        weight = torch.zeros(L, N, device=device)
        weight[ps, torch.arange(N, device=device)] = 1.0

        # 2) 对每个簇行做 L1 归一，得到簇内均值系数
        weight = F.normalize(weight, p=1, dim=1)

        # 3) centers = weight @ x
        centers = weight @ x

        # 4) 对每个中心做 L2 归一，便于余弦相似度稳定
        centers = F.normalize(centers, p=2, dim=1)
        return centers

    def clustering(self, features):
        # 使用 PyTorchKMeans 得到伪标签（E-step）
        kwargs = {
            'metric': 'cosine',
            'distributed': False,
            'random_state': 0,
            'n_clusters': self.num_clusters,
            'verbose': False
        }
        clustering_model = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
        psedo_labels = clustering_model.fit_predict(features.to(dtype=torch.float64))
        return psedo_labels

    def update_centers(self, zs_list, common_z):
        """
        每隔 tau 更新一次各视图 + 共识空间的中心与 sigma。
        - zs_list: list[Tensor[B,D]]，各视图特征
        - common_z: Tensor[B,D]，融合特征
        说明：
        - 首次（center 为 None）会强制更新，确保后续 membership 可用。
        - 该过程不参与反向传播（torch.no_grad）。
        """
        self.step += 1
        need_init = any(c is None for c in self.centers)
        if not need_init and (self.step % self.tau != 0):
            return

        features = zs_list + [common_z]
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
                labels = km.fit_predict(z.to(dtype=torch.float64)).to(self.device).long()
                centers = km.cluster_centers_.to(dtype=z.dtype).to(self.device)

                # assigned: 每个样本被指派到的簇中心
                assigned = centers[labels]
                # dists: 每个样本到其簇中心的欧氏距离
                dists = torch.norm(z - assigned, dim=1)

                # sigmas_v[k] = 第 k 簇距离中位数（下限 eps）
                # 影响 legacy membership 的高斯核宽度：
                # sigma 大 -> 分配更平滑；sigma 小 -> 分配更尖锐
                sigmas_v = []
                for k in range(self.num_clusters):
                    d_k = dists[labels == k]
                    if d_k.numel() > 0:
                        sigma_k = d_k.median()
                    else:
                        sigma_k = torch.tensor(0.0, device=self.device)
                    sigmas_v.append(torch.clamp(sigma_k, min=self.eps))
                sigmas_v = torch.stack(sigmas_v)

                self.centers[v] = centers
                self.sigmas[v] = sigmas_v

    def compute_membership(self, z, v_index):
        """
        计算样本到簇的软隶属度 m ∈ R^{N×L}。
        - z: [N,D] 当前空间特征（视图或共识）
        - v_index: 第几个空间（0~V-1 对应视图，V 对应共识）

        两种路径：
        1) softmax_distance（推荐）
           m = softmax(-||z-c||^2 / T_m)
        2) legacy gaussian
           m ∝ exp(-||z-c||^2 / (2σ^2))
        """
        centers = self.centers[v_index]

        if self.membership_mode == 'softmax_distance':
            dists_sq = torch.cdist(z, centers, p=2) ** 2
            logits = -dists_sq / max(self.membership_temperature, self.eps)
            membership = torch.softmax(logits, dim=1)
            return membership

        # legacy gaussian path（基线保留）
        sigmas = self.sigmas[v_index]
        dists_sq = torch.cdist(z, centers, p=2) ** 2
        denom = (2 * (sigmas ** 2).clamp(min=self.eps)).view(1, -1)
        unnorm = torch.exp(-dists_sq / denom)
        membership = unnorm / (unnorm.sum(dim=1, keepdim=True) + 1e-12)
        return membership

    def estimate_uncertainty(self, memberships, common_z):
        """
        估计样本不确定度 u（teacher-like）与 u_hat（MLP 预测）
        输入:
        - memberships: 长度 V+1 的 list，元素为 [N,L]
        - common_z: [N,D]
        输出:
        - u: 由隶属度计算得到的目标不确定度
        - u_hat: 由网络回归得到的预测不确定度
        """
        V = self.num_views

        if self.uncertainty_mode == 'legacy':
            # legacy = 归一化熵 + 1-Top2Gap 的平均
            u_vs = []
            for v in range(V):
                m = memberships[v]
                top2 = torch.topk(m, 2, dim=1).values
                delta = top2[:, 0] - top2[:, 1]
                delta_norm = (delta - delta.min()) / (delta.max() - delta.min() + 1e-12)
                ent = -torch.sum(m * torch.log(m + 1e-12), dim=1)
                ent_norm = (ent - ent.min()) / (ent.max() - ent.min() + 1e-12)
                u_v = 0.5 * ent_norm + 0.5 * (1.0 - delta_norm)
                u_vs.append(u_v)

            u_stack = torch.stack(u_vs, dim=1)
            u = u_stack.max(dim=1).values
            u_hat = self.mlp_uncert(common_z).squeeze(1)
            return u, u_hat

        # log-odds 路径：先算每视图 margin gamma，再转不确定度，并按可靠性加权融合
        u_vs = []
        gamma_vs = []
        for v in range(V):
            m = memberships[v]
            top2 = torch.topk(m, 2, dim=1).values
            gamma_v = torch.log((top2[:, 0] + self.eps) / (top2[:, 1] + self.eps))
            u_v = torch.sigmoid((self.uncertainty_kappa - gamma_v) / max(self.uncertainty_temperature, self.eps))
            gamma_vs.append(gamma_v)
            u_vs.append(u_v)

        gamma_stack = torch.stack(gamma_vs, dim=1)
        u_stack = torch.stack(u_vs, dim=1)

        # gamma 越大（margin 越强）说明视图更可靠
        view_weights = torch.softmax(gamma_stack / max(self.reliability_temperature, self.eps), dim=1)
        u = (view_weights * u_stack).sum(dim=1)

        u_hat = self.mlp_uncert(common_z).squeeze(1)
        return u, u_hat

    def classify_fn_hn(self,
                       zs_list, common_z, memberships,
                       batch_psedo_label, certain_mask, uncertain_mask,
                       epoch=100, fn_hn_warmup=10):
        """
        对不确定样本进行 FN/HN 分类（当前主流程已弱化该路径，但接口保留便于消融）。

        参数:
        - zs_list: 各视图特征 list，[V][N,D]
        - common_z: 共识特征 [N,D]
        - memberships: 软隶属度 list
        - batch_psedo_label: 批伪标签 [N]
        - certain_mask/uncertain_mask: 课程划分掩码 [N]
        - epoch: 当前轮数
        - fn_hn_warmup: 何时开始启用 FN/HN 分类
                         小 -> 更早介入；大 -> 更晚介入
        """
        if epoch < fn_hn_warmup or not uncertain_mask.any():
            return None, None, None
        device = self.device

        batch_psedo_label = batch_psedo_label.to(device)
        certain_mask = certain_mask.to(device)
        uncertain_mask = uncertain_mask.to(device)

        V = self.num_views
        N = batch_psedo_label.size(0)
        k = self.fn_hn_k
        eps = 1e-12

        dists = torch.stack([
            torch.cdist(z.to(device), z.to(device), p=2)
            for z in zs_list
        ], dim=0)

        knn_idx = torch.topk(-dists, k + 1, dim=2).indices[:, :, 1:]

        knn_labels = batch_psedo_label[knn_idx]
        certain_mask_knn = certain_mask.unsqueeze(0).unsqueeze(2).expand(V, N, k)

        same = (knn_labels == batch_psedo_label.view(1, N, 1)) & certain_mask_knn
        diff = (~same) & certain_mask_knn
        pos_cnt = same.sum(dim=2).float()
        neg_cnt = diff.sum(dim=2).float()
        delta = (pos_cnt - neg_cnt) / (pos_cnt + neg_cnt + eps)
        delta_d = delta.mean(dim=0)

        # mu_c: 每个样本对应伪标签簇中心（在共识空间）
        mu_c = self.centers[V][batch_psedo_label]
        s = F.cosine_similarity(common_z, mu_c, dim=1)

        v_idx = torch.arange(V, device=device)[:, None, None]
        n_idx = torch.arange(N, device=device)[None, :, None]
        neigh_d = dists[v_idx, n_idx, knn_idx]
        d_pos = torch.where(same, neigh_d, torch.nan).nanmean(dim=2)
        d_neg = torch.where(diff, neigh_d, torch.nan).nanmean(dim=2)
        d_pos = torch.nan_to_num(d_pos, nan=0.0)
        d_neg = torch.nan_to_num(d_neg, nan=0.0)
        a = ((d_neg - d_pos) / (d_neg + d_pos + eps)).mean(dim=0)

        idx_uncertain = uncertain_mask.nonzero(as_tuple=True)[0]
        feats_all = torch.stack([delta_d, s, a], dim=1)
        feats = feats_all[idx_uncertain]
        logits_raw = self.mlp_fn_hn(feats)

        # 置信度过滤：丢弃最不可靠的 10%
        probs = F.softmax(logits_raw, dim=1)
        conf, _ = probs.max(dim=1)
        thresh = torch.quantile(conf, 0.1)
        keep = conf > thresh

        if keep.sum() == 0:
            return None, None, None

        idx_kept = idx_uncertain[keep]
        logits = logits_raw[keep]
        feats = feats[keep]

        return logits, idx_kept, feats

    def compute_consistency_scores(self, memberships, batch_psedo_label):
        """
        构建跨视图一致性矩阵 S (N,N)，用于 cross-view weighted InfoNCE。

        核心思想:
        - 先估计“同簇且可信”的 pair 置信
        - 再结合视图不确定度做重加权
        """
        N = batch_psedo_label.size(0)
        V = self.num_views
        device = self.device

        conf_v = torch.stack([
            memberships[v][torch.arange(N, device=device), batch_psedo_label]
            for v in range(V)
        ], dim=0).to(device)

        conf_c = memberships[V][torch.arange(N, device=device), batch_psedo_label].to(device)

        u_vs = []
        for v in range(V):
            m = memberships[v]
            top2 = torch.topk(m, 2, dim=1).values
            delta = top2[:, 0] - top2[:, 1]
            delta_norm = (delta - delta.min()) / (delta.max() - delta.min() + 1e-12)
            u_vs.append(delta_norm)

        u_i = torch.stack(u_vs, dim=1)
        u_weights = 1.0 - u_i

        same_cluster = (batch_psedo_label.view(-1, 1) == batch_psedo_label.view(1, -1)).to(device)
        p_v = conf_v.unsqueeze(2) * same_cluster
        p_view_max = p_v.max(dim=0).values

        p_c = conf_c.view(-1, 1) * same_cluster

        # 每个样本先在视图维取平均权重，再对其所在行 pair 统一缩放
        u_weights_expanded = u_weights.mean(dim=1)
        weighted_p_view_max = p_view_max * u_weights_expanded

        S = torch.min(weighted_p_view_max, p_c)
        return S
