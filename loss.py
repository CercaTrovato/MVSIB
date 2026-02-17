import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, batch_size, num_clusters, temperature_l, temperature_f,
                 R_max=1.0, margin=0.5, bound_weight=0.1,
                 global_minority_ratio=0.5, num_gmm_components=2, num_pseudo_samples=5,
                 mmd_weight=1.0, excl_weight=1.0, excl_sigma=2.0,
                 module_c_minority_ratio=0.5,
                 module_c_radius_quantile=0.7,
                 module_c_gamma_threshold=0.5,
                 module_c_cov_shrink=0.2,
                 module_c_trunc_scale=1.0,
                 module_c_cov_eps=1e-5,
                 module_c_sigma_min_samples=8,
                 module_c_sample_retry=8,
                 module_c_boundary_rmax=1.0,
                 module_c_stat_ema_rho=0.9,
                 module_c_proto_align_weight=0.2,
                 module_c_conf_repulse_weight=0.2,
                 module_c_conf_margin=0.2):
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

        # 全局少数判定超参
        self.global_minority_ratio = global_minority_ratio

        # GMM 伪样本生成超参
        self.num_gmm_components = num_gmm_components
        self.num_pseudo_samples = num_pseudo_samples

        # MMD 与排斥核超参
        self.mmd_weight = mmd_weight
        self.excl_weight = excl_weight
        self.excl_sigma = excl_sigma

        # 模块C(ISM+)：全局统计与几何增强超参
        self.module_c_minority_ratio = module_c_minority_ratio
        self.module_c_radius_quantile = module_c_radius_quantile
        self.module_c_gamma_threshold = module_c_gamma_threshold
        self.module_c_cov_shrink = module_c_cov_shrink
        self.module_c_trunc_scale = module_c_trunc_scale
        self.module_c_cov_eps = module_c_cov_eps
        self.module_c_sigma_min_samples = module_c_sigma_min_samples
        self.module_c_sample_retry = module_c_sample_retry
        self.module_c_boundary_rmax = module_c_boundary_rmax
        self.module_c_stat_ema_rho = module_c_stat_ema_rho
        self.module_c_proto_align_weight = module_c_proto_align_weight
        self.module_c_conf_repulse_weight = module_c_conf_repulse_weight
        self.module_c_conf_margin = module_c_conf_margin

        # 自适应权重相关：三项 [mmd, excl, bound] 的 EMA
        self.register_buffer('loss_ema', torch.ones(3))
        self.ema_momentum = 0.9
        self.weight_alpha = 1.0
        # 损失函数实例
        self.mse = nn.MSELoss()
        self.ce  = nn.CrossEntropyLoss()

    def feature_loss(self, zi, z, w, y_pse, neg_weights=None):
        """
        zi: [N, D], z: [N, D]
        w:  [N, N] integer mask (0/1)
        y_pse: [N, N] float pseudo-label mask for positive pairs
        neg_weights: [N, N] float weights for negative-pair denominator (Design 1').
        """
        N = z.size(0)
        device = zi.device

        # —— 1) 计算距离矩阵 ——
        cross_view_distance = self.similarity(
            zi.unsqueeze(1),  # [N,1,D]
            z.unsqueeze(0)  # [1,N,D]
        ) / self.temperature_f  # → [N,N]
        inter_view_distance = self.similarity(
            zi.unsqueeze(1),
            zi.unsqueeze(0)
        ) / self.temperature_f  # → [N,N]

        # —— 2) 构造正样本 mask ——
        # 包含 w==1 的对，以及对角(自对)
        w_bool = w.to(device).bool()  # [N,N]
        eye = torch.eye(N, dtype=torch.bool, device=device)
        w_mask = w_bool | eye  # [N,N] 布尔

        # —— 3) 真实的“软”权重 y_pse ——
        y_pse = y_pse.to(device).float()  # [N,N]

        # —— 4) positive loss ——
        pos_mask = w_mask & (y_pse > 0)  # 只有 y_pse>0 才算正样本
        # 按 y_pse 乘以距离
        positive_term = (pos_mask.float() * y_pse * cross_view_distance).sum()
        positive_loss = -positive_term  # 保持原来 “-sum(...)”

        # —— 5) negative loss ——
        neg_mask = (~w_mask) & (y_pse == 0)
        SMALL_NUM = torch.log(torch.tensor(1e-45, device=device))

        neg_weight_mat = torch.ones_like(y_pse, device=device)
        if neg_weights is not None:
            neg_weight_mat = neg_weights.to(device).float().clamp(min=0.0)

        neg_cross = (neg_mask.float() * neg_weight_mat * cross_view_distance)
        neg_cross = neg_cross.masked_fill(neg_cross == 0, SMALL_NUM)

        neg_inter = (neg_mask.float() * neg_weight_mat * inter_view_distance)
        neg_inter = neg_inter.masked_fill(neg_inter == 0, SMALL_NUM)

        # 拼到一起，再除一次 temperature（可根据实际 remove）
        neg_sim = torch.cat([neg_inter, neg_cross], dim=1) / self.temperature_f
        neg_loss = torch.logsumexp(neg_sim, dim=1).sum()

        # —— 6) 最终归一 ——
        return (positive_loss + neg_loss) / N

    def compute_mmd_loss(self, real, pseudo):
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
        k = pseudos.size(0)
        if k < 2:
            return torch.tensor(0., device=pseudos.device)
        d = torch.cdist(pseudos, pseudos, p=2)
        mask = ~torch.eye(k, dtype=torch.bool, device=d.device)
        kr = torch.exp(-d[mask] ** 2 / (2 * self.excl_sigma ** 2))
        return kr.sum()

    def _sample_truncated_pseudo(self, mu_k, sigma_k, radius_k, device):
        """模块C(ISM+)：截断高斯采样，失败后回退到随机边界点，避免退化为 mu_k 常数点。"""
        D = mu_k.numel()
        radius_eff = float(max(radius_k, 1e-6))
        radius_eff = min(radius_eff, float(self.module_c_boundary_rmax))
        pseudos = []
        fail_count = 0
        fallback_count = 0

        eye_d = torch.eye(D, device=device, dtype=mu_k.dtype)
        sigma_safe = sigma_k + float(self.module_c_cov_eps) * eye_d
        mvn = None
        try:
            mvn = torch.distributions.MultivariateNormal(mu_k, covariance_matrix=sigma_safe)
        except Exception:
            mvn = None

        for _ in range(int(self.num_pseudo_samples)):
            accepted = False
            if mvn is not None:
                for _retry in range(int(self.module_c_sample_retry)):
                    psi = mvn.rsample()
                    if torch.norm(psi - mu_k, p=2) <= radius_eff * float(self.module_c_trunc_scale):
                        pseudos.append(psi)
                        accepted = True
                        break
                if not accepted:
                    fail_count += 1

            if not accepted:
                fallback_count += 1
                # 模块C回退：随机单位方向边界点，避免回退到中心点造成伪样本坍塌
                u = torch.randn_like(mu_k)
                u = u / (u.norm(p=2) + 1e-8)
                pseudos.append(mu_k + radius_eff * u)

        return torch.stack(pseudos, dim=0), fail_count, fallback_count

    def compute_cluster_loss(self,
                             q_centers,  # [L, D]
                             k_centers,  # [L, D]
                             psedo_labels_batch,  # [B]
                             features_batch=None,  # [B, D] or None
                             module_c_stats=None,
                             return_mmd_excl=False,
                             return_details=False):
        """
        如果 features_batch or module_c_stats is None: 只做原始 InfoNCE（不带伪样本）。
        否则：使用模块C(ISM+)的全局统计确定少数簇，并执行截断采样 + proto-align/conf-repulse。
        """
        total_mmd = torch.tensor(0.0, device=q_centers.device)
        total_excl = torch.tensor(0.0, device=q_centers.device)
        device = q_centers.device
        L, D = q_centers.shape

        details = {
            'minority_set_size': 0.0,
            'minority_count_mean': 0.0,
            'minority_radius_mean': 0.0,
            'minority_gamma_mean': 0.0,
            'module_c_sample_fail_rate': 0.0,
            'module_c_fallback_rate': 0.0,
            'L_proto_align': 0.0,
            'L_conf_repulse': 0.0,
        }

        # 1) 计算基础相似度矩阵 d_q
        d_q_raw = q_centers.mm(q_centers.T)
        with torch.no_grad():
            d_kdiag = (q_centers * k_centers).sum(dim=1) / self.temperature_l
        I_L = torch.eye(L, device=device, dtype=d_q_raw.dtype)
        d_q = (d_q_raw / self.temperature_l) * (1 - I_L) + torch.diag(d_kdiag)

        # 原始 InfoNCE 分支
        if features_batch is None or module_c_stats is None:
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
            cluster_loss = torch.stack(losses).sum() / num_nonzero if num_nonzero > 0 else torch.tensor(0., device=device)
            if return_mmd_excl and return_details:
                return cluster_loss, total_mmd, total_excl, details
            if return_mmd_excl:
                return cluster_loss, total_mmd, total_excl
            if return_details:
                return cluster_loss, details
            return cluster_loss

        # ——————— 模块C(ISM+) 分支 ———————
        counts = torch.bincount(psedo_labels_batch, minlength=self.num_clusters).float().to(device)
        unique_labels = torch.unique(psedo_labels_batch).to(device)
        mask_unique = torch.zeros(self.num_clusters, device=device, dtype=torch.bool)
        mask_unique[unique_labels] = True
        zero_classes = torch.arange(self.num_clusters, device=device)[~mask_unique]
        batch_mask = counts > 0

        n_ema = module_c_stats['n_k_ema'].to(device)
        r_ema = module_c_stats['R_k_ema'].to(device)
        gamma_ema = module_c_stats['gamma_bar_k_ema'].to(device)
        sigma_ema = module_c_stats['Sigma_k_ema'].to(device)

        # 模块C少数簇判定：n小 且 (R大 或 gamma小)
        n_thr = max(1.0, float(n_ema.max().item()) * float(self.module_c_minority_ratio))
        r_thr = float(torch.quantile(r_ema.detach(), float(self.module_c_radius_quantile)).item())
        gamma_thr = float(self.module_c_gamma_threshold)
        minority_global = (n_ema <= n_thr) & ((r_ema >= r_thr) | (gamma_ema <= gamma_thr))
        batch_minority_mask = batch_mask & minority_global
        minority_indices = batch_minority_mask.nonzero(as_tuple=False).view(-1)

        if minority_indices.numel() > 0:
            details['minority_set_size'] = float(minority_indices.numel())
            details['minority_count_mean'] = float(n_ema[minority_indices].mean().item())
            details['minority_radius_mean'] = float(r_ema[minority_indices].mean().item())
            details['minority_gamma_mean'] = float(gamma_ema[minority_indices].mean().item())

        pseudos = torch.zeros((L, D), device=device, dtype=q_centers.dtype)
        mu_minority = {}
        proto_align_loss = torch.tensor(0.0, device=device)
        conf_repulse_loss = torch.tensor(0.0, device=device)
        sample_total = 0
        sample_fail = 0
        sample_fallback = 0

        # k^- 仅按 prototype-level 最近邻选择，且 stop-grad
        centers_det = F.normalize(q_centers.detach(), dim=1)
        sim_cent = torch.mm(centers_det, centers_det.t())
        sim_cent.fill_diagonal_(-1e9)
        k_minus = torch.argmax(sim_cent, dim=1)

        for k_tensor in minority_indices:
            k = k_tensor.item()
            mu_k = q_centers[k].detach()
            mu_minority[k] = mu_k

            sigma_k = sigma_ema[k].detach()
            n_k_ema = float(n_ema[k].item())
            if n_k_ema < float(self.module_c_sigma_min_samples):
                sigma_k = torch.eye(D, device=device, dtype=q_centers.dtype)
            sigma_shrink = (1.0 - float(self.module_c_cov_shrink)) * sigma_k + float(self.module_c_cov_shrink) * torch.eye(D, device=device, dtype=q_centers.dtype)

            radius_k = float(r_ema[k].item())
            psi_samples, fail_k, fallback_k = self._sample_truncated_pseudo(mu_k, sigma_shrink, radius_k, device)
            sample_total += int(psi_samples.size(0))
            sample_fail += int(fail_k)
            sample_fallback += int(fallback_k)

            psi_mean = psi_samples.mean(dim=0)
            pseudos[k] = psi_mean

            # 模块C几何项：原型对齐 + 易混淆邻簇排斥
            psi_n = F.normalize(psi_samples, dim=1)
            mu_k_n = F.normalize(mu_k.unsqueeze(0), dim=1)
            mu_minus = q_centers[k_minus[k]].detach()
            mu_minus_n = F.normalize(mu_minus.unsqueeze(0), dim=1)
            sim_pos = (psi_n * mu_k_n).sum(dim=1)
            sim_neg = (psi_n * mu_minus_n).sum(dim=1)
            proto_align_loss += (1.0 - sim_pos).mean()
            conf_repulse_loss += F.relu(float(self.module_c_conf_margin) + sim_neg - sim_pos).mean()

        if sample_total > 0:
            details['module_c_sample_fail_rate'] = float(sample_fail) / float(sample_total)
            details['module_c_fallback_rate'] = float(sample_fallback) / float(sample_total)

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
            neg_pseudo = torch.tensor(0.0, device=device)
            if minority_indices.numel() > 0:
                mask_minor = batch_minority_mask.float().to(device) * mask_neg
                neg_pseudo = (torch.exp(sim_pseudo_all[k]) * mask_minor).sum()
            denom = pos_main + neg_real + neg_pseudo + 1e-8
            losses.append(-torch.log(pos_main / denom))
        cluster_loss = torch.stack(losses).sum() / (
                    L - zero_classes.numel()) if L - zero_classes.numel() > 0 else torch.tensor(0., device=device)

        # 合并边界约束
        if self.bound_weight > 0:
            cluster_loss = cluster_loss + self.bound_weight * bound_loss

        if minority_indices.numel() > 0:
            cluster_loss = cluster_loss + float(self.module_c_proto_align_weight) * proto_align_loss
            cluster_loss = cluster_loss + float(self.module_c_conf_repulse_weight) * conf_repulse_loss
            details['L_proto_align'] = float(proto_align_loss.item())
            details['L_conf_repulse'] = float(conf_repulse_loss.item())

        if features_batch is not None and minority_indices.numel() > 1:
            mask_real = torch.zeros_like(psedo_labels_batch, dtype=torch.bool).to(device)
            for k in minority_indices:
                k = k.item()
                mask_real |= (psedo_labels_batch == k)

            real_feats = features_batch[mask_real]
            pseudo_feats = pseudos[minority_indices]
            total_mmd = self.compute_mmd_loss(real_feats, pseudo_feats)
            total_excl = self.compute_exclusion_loss(pseudo_feats)
            cluster_loss = cluster_loss + self.mmd_weight * total_mmd + self.excl_weight * total_excl

        if return_mmd_excl and return_details:
            return cluster_loss, total_mmd, total_excl, details
        if return_mmd_excl:
            return cluster_loss, total_mmd, total_excl
        if return_details:
            return cluster_loss, details
        return cluster_loss




    # —— 模块2：不确定度 MLP 回归损失 —— #
    def uncertainty_regression_loss(self, u_hat, u_true):

        return self.mse(u_hat, u_true.detach())


    # —— 模块4：加权 InfoNCE —— #
    def weighted_info_nce(self, reps, S, temperature):
        sim = F.cosine_similarity(reps.unsqueeze(1), reps.unsqueeze(0), dim=2) / temperature  # (N, N)

        # 按 Eq.(48) 对 S 做按行归一化（排除对角项）
        S = S.clone()
        eye = torch.eye(S.size(0), device=S.device, dtype=torch.bool)
        S = S.masked_fill(eye, 0.0)
        S = S / (S.sum(dim=1, keepdim=True) + 1e-8)

        # 使用归一化后的 S 对所有样本对进行加权
        exp_sim = torch.exp(sim)  # (N, N)
        weighted_sim = exp_sim * S  # 每个样本对的加权相似度 (N, N)

        # 计算分子和分母
        num = torch.sum(weighted_sim, dim=1)  # 对每个样本的加权相似度求和
        den = torch.sum(exp_sim, dim=1)  # 对每个样本的相似度求和

        # 避免除零错误
        eps = 1e-8
        num = torch.clamp(num, min=eps)
        den = torch.clamp(den, min=eps)

        # 计算最终损失
        loss = -torch.mean(torch.log(num / den))  # 计算 InfoNCE 损失

        return loss

    def cross_view_weighted_loss(self,
                                 um,          # UncertaintyModule 实例
                                 zs_list,     # list of view representations
                                 common_z,    # consensus representation
                                 memberships, # list of Tensors[N×L]
                                 batch_psedo_label,
                                 temperature=None):
        """
        一体化接口，计算跨视图 & 共识空间的加权 InfoNCE 总和。
        """
        if temperature is None:
            temperature = self.temperature_f

        # 1) 计算一致性分数矩阵 S
        S = um.compute_consistency_scores(memberships, batch_psedo_label)

        # 2) 对每个视图与共识分别做加权 InfoNCE
        loss = 0.0
        for z in zs_list:
            loss += self.weighted_info_nce(z, S, temperature)
        loss += self.weighted_info_nce(common_z, S, temperature)
        return loss
