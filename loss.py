import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# loss.py
# ----------------------------------------------------------------------------
# 本文件集中定义 MVSIB 的损失函数族：
# 1) feature-level 对比损失（支持负样本软降权）
# 2) cluster-level InfoNCE（支持少数簇伪样本增强）
# 3) 不确定度回归损失
# 4) 跨视图加权 InfoNCE
#
# 注：所有超参数均保留并在注释中说明调节影响，保证实验可对照。
# ============================================================================


class Loss(nn.Module):
    def __init__(self, batch_size, num_clusters, temperature_l, temperature_f,
                 R_max=1.0, margin=0.5, bound_weight=0.1,
                 global_minority_ratio=0.5, num_gmm_components=2, num_pseudo_samples=5,
                 mmd_weight=1.0, excl_weight=1.0, excl_sigma=2.0):
        """
        参数说明（含超参数影响）:
        - batch_size: 训练批大小（主要用于外部流程，对损失逻辑本身无硬依赖）。
        - num_clusters: 簇数 L。
        - temperature_l: 簇级对比温度 T_l；小更尖锐，大更平滑。
        - temperature_f: 特征级对比温度 T_f；小更强分离，大更稳定。
        - R_max: 少数簇伪样本边界半径上限；小更严格约束伪样本不远离簇中心。
        - margin: 边界间隔；大则更强调与其他簇保持距离。
        - bound_weight: 边界约束权重；0 则关闭边界项。
        - global_minority_ratio: 全局少数簇判定阈值比例（当前主路径保留接口）。
        - num_gmm_components: GMM 组件数（当前实现保留超参位）。
        - num_pseudo_samples: 每簇伪样本规模相关系数；大则伪样本方差更强。
        - mmd_weight: MMD 对齐项权重。
        - excl_weight: 伪样本排斥项权重。
        - excl_sigma: RBF 核带宽；小更局部敏感，大更全局平滑。
        """
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.num_clusters = num_clusters
        self.temperature_l = temperature_l
        self.temperature_f = temperature_f
        self.similarity = nn.CosineSimilarity(dim=2)

        # 边界约束超参
        self.R_max = R_max
        self.margin = margin
        self.bound_weight = bound_weight

        # 全局少数判定超参（当前主要作为可扩展参数位）
        self.global_minority_ratio = global_minority_ratio

        # GMM 伪样本生成超参
        self.num_gmm_components = num_gmm_components
        self.num_pseudo_samples = num_pseudo_samples

        # MMD 与排斥核超参
        self.mmd_weight = mmd_weight
        self.excl_weight = excl_weight
        self.excl_sigma = excl_sigma

        # 自适应权重相关：三项 [mmd, excl, bound] 的 EMA
        self.register_buffer('loss_ema', torch.ones(3))
        self.ema_momentum = 0.9
        self.weight_alpha = 1.0

        # 常用基础损失
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def feature_loss(self, zi, z, w, y_pse, neg_weights=None):
        """
        特征级对比损失（单视图 zi 对共识 z）。

        参数:
        - zi: Tensor[N,D]，某视图的表示。
        - z: Tensor[N,D]，共识表示 common_z。
        - w: Tensor[N,N]，kNN 邻接掩码（0/1）。
        - y_pse: Tensor[N,N]，由伪标签得到的同簇掩码（soft/float）。
        - neg_weights: Tensor[N,N]，负样本软权重（FN 风险路由输出）。
          若为 None，则退化为普通等权负样本。
        """
        N = z.size(0)
        device = zi.device

        # 1) 计算视图-共识 与 视图-视图 的余弦相似度矩阵
        cross_view_distance = self.similarity(zi.unsqueeze(1), z.unsqueeze(0)) / self.temperature_f
        inter_view_distance = self.similarity(zi.unsqueeze(1), zi.unsqueeze(0)) / self.temperature_f

        # 2) 构造正样本 mask（图邻接 + 对角自对）
        w_bool = w.to(device).bool()
        eye = torch.eye(N, dtype=torch.bool, device=device)
        w_mask = w_bool | eye

        # 3) y_pse 是软标签权重（同簇程度）
        y_pse = y_pse.to(device).float()

        # 4) 正样本项
        pos_mask = w_mask & (y_pse > 0)
        positive_term = (pos_mask.float() * y_pse * cross_view_distance).sum()
        positive_loss = -positive_term

        # 5) 负样本项
        neg_mask = (~w_mask) & (y_pse == 0)
        SMALL_NUM = torch.log(torch.tensor(1e-45, device=device))

        neg_weight_mat = torch.ones_like(y_pse, device=device)
        if neg_weights is not None:
            neg_weight_mat = neg_weights.to(device).float().clamp(min=0.0)

        neg_cross = (neg_mask.float() * neg_weight_mat * cross_view_distance)
        neg_cross = neg_cross.masked_fill(neg_cross == 0, SMALL_NUM)

        neg_inter = (neg_mask.float() * neg_weight_mat * inter_view_distance)
        neg_inter = neg_inter.masked_fill(neg_inter == 0, SMALL_NUM)

        # 历史实现保留：拼接后再次按 temperature_f 缩放，保证与旧实验一致
        neg_sim = torch.cat([neg_inter, neg_cross], dim=1) / self.temperature_f
        neg_loss = torch.logsumexp(neg_sim, dim=1).sum()

        # 6) 按样本数归一化
        return (positive_loss + neg_loss) / N

    def compute_mmd_loss(self, real, pseudo):
        # MMD：约束伪样本分布贴近真实少数簇分布
        n, m = real.size(0), pseudo.size(0)
        if n < 2 or m < 2:
            return torch.tensor(0., device=real.device)
        K_xx = torch.exp(-torch.cdist(real, real, p=2) ** 2 / (2 * self.excl_sigma ** 2))
        K_yy = torch.exp(-torch.cdist(pseudo, pseudo, p=2) ** 2 / (2 * self.excl_sigma ** 2))
        K_xy = torch.exp(-torch.cdist(real, pseudo, p=2) ** 2 / (2 * self.excl_sigma ** 2))
        sum_xx = (K_xx.sum() - K_xx.diag().sum()) / (n * (n - 1))
        sum_yy = (K_yy.sum() - K_yy.diag().sum()) / (m * (m - 1))
        sum_xy = 2 * K_xy.sum() / (n * m)
        return sum_xx + sum_yy - sum_xy

    def compute_exclusion_loss(self, pseudos):
        # 排斥项：避免不同簇伪样本塌缩到一起
        k = pseudos.size(0)
        if k < 2:
            return torch.tensor(0., device=pseudos.device)
        d = torch.cdist(pseudos, pseudos, p=2)
        mask = ~torch.eye(k, dtype=torch.bool, device=d.device)
        kr = torch.exp(-d[mask] ** 2 / (2 * self.excl_sigma ** 2))
        return kr.sum()

    def compute_cluster_loss(self,
                             q_centers,
                             k_centers,
                             psedo_labels_batch,
                             features_batch=None,
                             global_minority_mask=None,
                             return_mmd_excl=False):
        """
        簇级损失主入口：
        - baseline 分支：仅 cluster InfoNCE；
        - minority-enhancement 分支：加入伪样本 + 边界 + MMD + exclusion。
        """
        total_mmd = 0.0
        total_excl = 0.0
        device = q_centers.device
        L, D = q_centers.shape

        # 基础簇间相似度矩阵 d_q
        d_q_raw = q_centers.mm(q_centers.T)
        with torch.no_grad():
            d_kdiag = (q_centers * k_centers).sum(dim=1) / self.temperature_l
        I_L = torch.eye(L, device=device, dtype=d_q_raw.dtype)
        d_q = (d_q_raw / self.temperature_l) * (1 - I_L) + torch.diag(d_kdiag)

        # baseline 分支
        if features_batch is None or global_minority_mask is None:
            counts = torch.bincount(psedo_labels_batch, minlength=self.num_clusters).float()
            unique_labels = torch.unique(psedo_labels_batch)
            mask_unique = torch.zeros(self.num_clusters, device=device, dtype=torch.bool)
            mask_unique[unique_labels] = True
            zero_classes = torch.arange(self.num_clusters, device=device)[~mask_unique]

            losses = []
            eye = torch.eye(L, device=device, dtype=torch.bool)
            for k in range(L):
                if k in zero_classes:
                    continue
                pos = torch.exp(d_q[k, k])
                neg = (torch.exp(d_q[k, :]) * (~eye[k]).float()).sum()
                losses.append(-torch.log(pos / (pos + neg + 1e-8)))
            num_nonzero = L - zero_classes.numel()
            return torch.stack(losses).sum() / num_nonzero if num_nonzero > 0 else torch.tensor(0., device=device)

        # minority-enhancement 分支
        counts = torch.bincount(psedo_labels_batch, minlength=self.num_clusters).float().to(device)
        unique_labels = torch.unique(psedo_labels_batch).to(device)
        mask_unique = torch.zeros(self.num_clusters, device=device, dtype=torch.bool)
        mask_unique[unique_labels] = True
        zero_classes = torch.arange(self.num_clusters, device=device)[~mask_unique]
        batch_mask = counts > 0
        global_mask = global_minority_mask.to(device=device)
        batch_minority_mask = batch_mask & global_mask
        minority_indices = batch_minority_mask.nonzero(as_tuple=False).view(-1)

        # 伪样本生成
        pseudos = torch.zeros((L, D), device=device, dtype=q_centers.dtype)
        mu_minority = {}
        for k_tensor in minority_indices:
            k = k_tensor.item()
            feats_k = features_batch[psedo_labels_batch == k]
            n_k = feats_k.size(0)
            if n_k <= 1:
                pseudos[k] = k_centers[k]
                mu_minority[k] = k_centers[k]
            else:
                mu_k = feats_k.mean(dim=0)
                mu_minority[k] = mu_k
                xc = feats_k - mu_k.unsqueeze(0)
                cov_k = xc.T @ xc / (n_k - 1 + 1e-6)
                cov_p = (self.num_pseudo_samples / n_k) * cov_k + 1e-6 * torch.eye(D, device=device)
                try:
                    mvn = torch.distributions.MultivariateNormal(mu_k, covariance_matrix=cov_p)
                    pseudos[k] = mvn.rsample()
                except Exception:
                    pseudos[k] = k_centers[k]

        # 边界约束
        bound_loss = torch.tensor(0., device=device)
        if self.bound_weight > 0 and minority_indices.numel() > 0:
            for k in minority_indices.tolist():
                psi_k = pseudos[k]
                mu_k = mu_minority[k]
                d_self = torch.norm(psi_k - mu_k, p=2)
                dists = torch.norm(q_centers - psi_k.unsqueeze(0), dim=1)
                mask_self = torch.arange(L, device=device) == k
                d_other = dists.masked_fill(mask_self, float('inf')).min()
                bound_loss += F.relu(d_self - self.R_max) + F.relu(self.margin - d_other)

        # 簇级 InfoNCE + 伪样本
        if minority_indices.numel() > 0:
            q_exp = q_centers.unsqueeze(1).expand(L, L, D)
            psi_exp = pseudos.unsqueeze(0).expand(L, L, D)
            sim_pseudo_all = F.cosine_similarity(q_exp, psi_exp, dim=2) / self.temperature_l
        else:
            sim_pseudo_all = torch.zeros((L, L), device=device)

        eye_mask = torch.eye(L, device=device, dtype=torch.bool)
        losses = []
        for k in range(L):
            if k in zero_classes:
                continue
            pos_main = torch.exp(d_q[k, k])
            if batch_minority_mask[k]:
                pos_main = pos_main + torch.exp(sim_pseudo_all[k, k])
            dq_row = d_q[k]
            mask_neg = (~eye_mask[k]).float()
            neg_real = (torch.exp(dq_row) * mask_neg).sum()
            neg_pseudo = 0
            if minority_indices.numel() > 0:
                mask_minor = batch_minority_mask.float().to(device) * mask_neg
                neg_pseudo = (torch.exp(sim_pseudo_all[k]) * mask_minor).sum()
            denom = pos_main + neg_real + neg_pseudo + 1e-8
            losses.append(-torch.log(pos_main / denom))

        cluster_loss = torch.stack(losses).sum() / (
            L - zero_classes.numel()) if L - zero_classes.numel() > 0 else torch.tensor(0., device=device)

        if self.bound_weight > 0:
            cluster_loss = cluster_loss + self.bound_weight * bound_loss

        if features_batch is not None and minority_indices.numel() > 1:
            # 批内少数簇真实样本掩码
            device = psedo_labels_batch.device
            mask_real = torch.zeros_like(psedo_labels_batch, dtype=torch.bool).to(device)
            for k in minority_indices:
                mask_real |= (psedo_labels_batch == k.item())

            real_feats = features_batch[mask_real]
            pseudo_feats = pseudos[minority_indices]
            total_mmd = self.compute_mmd_loss(real_feats, pseudo_feats)
            total_excl = self.compute_exclusion_loss(pseudo_feats)
            cluster_loss = cluster_loss + self.mmd_weight * total_mmd + self.excl_weight * total_excl

        if return_mmd_excl:
            return cluster_loss, total_mmd, total_excl
        return cluster_loss

    # 模块：不确定度回归损失
    def uncertainty_regression_loss(self, u_hat, u_true):
        # u_hat: MLP 预测不确定度；u_true: 由 membership 计算得到的目标不确定度
        # detach u_true 以防 teacher 目标反向影响其生成路径
        return self.mse(u_hat, u_true.detach())

    # 模块：加权 InfoNCE
    def weighted_info_nce(self, reps, S, temperature):
        """
        参数:
        - reps: Tensor[N,D]，待对比表征
        - S: Tensor[N,N]，pair 权重矩阵（来自一致性估计）
        - temperature: 温度系数（小更尖锐，大更平滑）
        """
        sim = F.cosine_similarity(reps.unsqueeze(1), reps.unsqueeze(0), dim=2) / temperature

        # 对 S 做按行归一化（先去除对角）
        S = S.clone()
        eye = torch.eye(S.size(0), device=S.device, dtype=torch.bool)
        S = S.masked_fill(eye, 0.0)
        S = S / (S.sum(dim=1, keepdim=True) + 1e-8)

        exp_sim = torch.exp(sim)
        weighted_sim = exp_sim * S

        num = torch.sum(weighted_sim, dim=1)
        den = torch.sum(exp_sim, dim=1)

        eps = 1e-8
        num = torch.clamp(num, min=eps)
        den = torch.clamp(den, min=eps)

        loss = -torch.mean(torch.log(num / den))
        return loss

    def cross_view_weighted_loss(self,
                                 um,
                                 zs_list,
                                 common_z,
                                 memberships,
                                 batch_psedo_label,
                                 temperature=None):
        """
        跨视图 + 共识空间加权 InfoNCE 总损失。

        - um: 提供 compute_consistency_scores() 的模块（这里传 Network 实例）
        - zs_list: 各视图表示列表
        - common_z: 共识表示
        - memberships: 各视图+共识隶属度
        - batch_psedo_label: 批伪标签
        """
        if temperature is None:
            temperature = self.temperature_f

        # 1) 一致性权重矩阵
        S = um.compute_consistency_scores(memberships, batch_psedo_label)

        # 2) 所有视图 + 共识统一加权对比
        loss = 0.0
        for z in zs_list:
            loss += self.weighted_info_nce(z, S, temperature)
        loss += self.weighted_info_nce(common_z, S, temperature)
        return loss
