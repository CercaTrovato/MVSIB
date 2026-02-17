from dataloader import *
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np



# 2.1 get_knn_graph：对应论文第 4.1 节“构建最近邻图 G”（Eq.(8)）
def get_knn_graph(data, k):
    num_samples = data.size(0)
    graph = torch.zeros(num_samples, num_samples, dtype=torch.int32, device=data.device)

    for i in range(num_samples):
        distance = torch.sum((data - data[i]) ** 2, dim=1)
        _, small_indices = torch.topk(distance, k, largest=False)  # +1 to exclude self from neighbors
        # Fill 1 in the graph for the k nearest neighbors
        graph[i, small_indices[1:]] = 1

    # Ensure the graph is symmetric
    result_graph = torch.max(graph, graph.t())

    return result_graph


# 2.2 get_W：为每个视图、每个 batch 预先计算邻接矩阵，对应 Fine-tuning 阶段特征对比里所需的G(v)
def get_W(mv_data, k):
    W = []
    mv_data_loader, num_views, num_samples, _ = get_all_multiview_data(mv_data)
    for _, (sub_data_views, _, _) in enumerate(mv_data_loader):
        for i in range(num_views):
            result_graph = get_knn_graph(sub_data_views[i], k)
            W.append(result_graph)
    return W


# 2.3 psedo_labeling：对应论文 Fine-tuning 阶段“E 步（Expectation）”
# model.py

def psedo_labeling(model, dataset, batch_size):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    commonZ_list = []
    for xs, _, _ in loader:
        with torch.no_grad():
            xrs, zs = model(xs)
            commonz = model.fusion(zs)
            commonZ_list.append(commonz)
    commonZ = torch.cat(commonZ_list, dim=0)
    # clustering 返回 numpy array 或 Tensor
    labels = model.clustering(commonZ)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    # 确保放到同 device 且存为 Tensor buffer
    model.psedo_labels = labels.to(model.psedo_labels.device).long()


# 2.4 pre_train：对应论文 “Warm-up 阶段”（Eq.(12)）
def pre_train(model, mv_data, batch_size, epochs, optimizer):
    mv_data_loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)

    pre_train_loss_values = np.zeros(epochs + 1, dtype=np.float64)

    criterion = torch.nn.MSELoss()
    for epoch in range(1, epochs + 1):
        total_loss = 0.
        for batch_idx, (sub_data_views, _, _) in enumerate(mv_data_loader):
            xrs, _ = model(sub_data_views)
            loss_list = list()
            for idx in range(num_views):
                loss_list.append(criterion(sub_data_views[idx], xrs[idx]))
            loss = sum(loss_list)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        pre_train_loss_values[epoch] = total_loss
        if epoch % 10 == 0 or epoch == epochs:
            print('Pre-training, epoch {}, Loss:{:.7f}'.format(epoch, total_loss))

    return pre_train_loss_values


def _adaptive_u_threshold(u, method='otsu', bins=64, eps=1e-12):
    """模块A：由当下 u 分布自适应估计 tau_u（当前实现：Otsu）。"""
    u = u.detach().reshape(-1)
    if u.numel() <= 1:
        return float(u.mean().item()) if u.numel() > 0 else 0.5

    if method != 'otsu':
        return float(torch.median(u).item())

    u_min = float(u.min().item())
    u_max = float(u.max().item())
    if (u_max - u_min) < 1e-8:
        return u_min

    hist = torch.histc(u, bins=bins, min=u_min, max=u_max)
    prob = hist / (hist.sum() + eps)
    centers = torch.linspace(u_min, u_max, steps=bins, device=u.device)

    omega = torch.cumsum(prob, dim=0)
    mu = torch.cumsum(prob * centers, dim=0)
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + eps)
    sigma_b[~torch.isfinite(sigma_b)] = -1.0
    idx = int(torch.argmax(sigma_b).item())
    return float(centers[idx].item())


def _compute_theta_certificate(common_z, q_cons, temperature=0.5, eps=1e-12):
    """模块A安全证书：theta_i = sum p_ij exp(s_ij/tau) / sum exp(s_ij/tau), j!=i。"""
    sim = F.cosine_similarity(common_z.unsqueeze(1), common_z.unsqueeze(0), dim=2)
    p = torch.mm(q_cons, q_cons.t()).clamp(0.0, 1.0)
    eye = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    exp_sim = torch.exp(sim / max(temperature, eps))
    exp_sim = exp_sim.masked_fill(eye, 0.0)
    numer = (p * exp_sim).sum(dim=1)
    denom = exp_sim.sum(dim=1) + eps
    return numer / denom


def _bounded_uncertain_mask(u, raw_mask, min_ratio=0.02, max_ratio=0.6):
    """防空/防全满：将不确定样本数量约束在 [min_uncertain, max_uncertain]。"""
    B = int(u.numel())
    if B == 0:
        return raw_mask, 0
    min_uncertain = min(B, max(8, int(min_ratio * B)))
    max_uncertain = min(B, max(min_uncertain, int(max_ratio * B)))
    k_unc = int(raw_mask.sum().item())

    if k_unc < min_uncertain or k_unc > max_uncertain:
        k_unc = min(max(k_unc, min_uncertain), max_uncertain)
        _, idx = torch.topk(u, k_unc, largest=True)
        bounded = torch.zeros_like(raw_mask)
        bounded[idx] = True
        return bounded, k_unc
    return raw_mask, k_unc


def _compute_neg_mask(common_z, neg_mode='batch', knn_k=20):
    """统一构造真实负对掩码（模块B pair 层）：排除对角，仅保留候选 negative。"""
    device = common_z.device
    N = common_z.size(0)
    eye = torch.eye(N, dtype=torch.bool, device=device)
    neg_mask = ~eye
    if neg_mode == 'knn':
        k_eff = min(knn_k + 1, N)
        dist = torch.cdist(common_z, common_z, p=2)
        knn_idx = torch.topk(-dist, k_eff, dim=1).indices
        knn_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
        row_idx = torch.arange(N, device=device).unsqueeze(1).expand_as(knn_idx)
        knn_mask[row_idx, knn_idx] = True
        neg_mask = knn_mask & (~eye)
    return neg_mask, eye


def _compute_gate_by_stability(d_time, gate_mode='linear', linear_t=0.0,
                               prev_ema=None, ema_rho=0.9, s0=0.5, tg=0.1, eps=1e-12):
    """模块B门控：线性 gate（baseline）或稳定性触发 gate（Sigmoid(EMA(stab_t))）。"""
    stab_t = torch.exp(-d_time.detach()).mean().item() if d_time is not None else 0.0
    if prev_ema is None:
        ema_stab = stab_t
    else:
        ema_stab = float(ema_rho) * float(prev_ema) + (1.0 - float(ema_rho)) * float(stab_t)

    if gate_mode == 'stability':
        gate = torch.sigmoid(torch.tensor((ema_stab - s0) / max(float(tg), eps))).item()
    else:
        gate = float(linear_t)

    return float(gate), float(stab_t), float(ema_stab)


def _ensure_module_c_ema_stats(model, num_clusters, feat_dim, device):
    """模块C：维护全局EMA统计（n_k, R_k, gamma_bar_k, Sigma_k），用于稳定长尾判定。"""
    if hasattr(model, 'module_c_ema_stats') and model.module_c_ema_stats is not None:
        return
    eye = torch.eye(feat_dim, device=device)
    model.module_c_ema_stats = {
        'n_k_ema': torch.ones(num_clusters, device=device),
        'R_k_ema': torch.ones(num_clusters, device=device),
        'gamma_bar_k_ema': torch.zeros(num_clusters, device=device),
        'Sigma_k_ema': eye.unsqueeze(0).repeat(num_clusters, 1, 1),
    }


def _init_module_c_epoch_stats(num_clusters, feat_dim, device):
    return {
        'count': torch.zeros(num_clusters, device=device),
        'sum_x': torch.zeros(num_clusters, feat_dim, device=device),
        'sum_xx': torch.zeros(num_clusters, feat_dim, feat_dim, device=device),
        'gamma_sum': torch.zeros(num_clusters, device=device),
        'dist_buf': [[] for _ in range(num_clusters)],
    }


def _accumulate_module_c_epoch_stats(epoch_stats, labels, q_cons, common_z, q_centers):
    """模块C：按 batch 累积到 epoch 级统计，避免 batch-only 的 minority 抖动。"""
    top2 = torch.topk(q_cons.detach(), 2, dim=1).values
    gamma = torch.log((top2[:, 0] + 1e-12) / (top2[:, 1] + 1e-12))
    uniq = torch.unique(labels)
    for k_t in uniq:
        k = int(k_t.item())
        mask = (labels == k)
        feats = common_z[mask].detach()
        if feats.numel() == 0:
            continue
        n_k = float(feats.size(0))
        epoch_stats['count'][k] += n_k
        epoch_stats['sum_x'][k] += feats.sum(dim=0)
        epoch_stats['sum_xx'][k] += feats.t().mm(feats)
        epoch_stats['gamma_sum'][k] += gamma[mask].sum()
        d = torch.norm(feats - q_centers[k].detach().unsqueeze(0), dim=1)
        epoch_stats['dist_buf'][k].append(d.detach().cpu())


def _update_module_c_ema_stats(model, epoch_stats, ema_rho=0.9):
    """模块C：epoch 结束后更新 EMA 统计，供下一 epoch 的 ISM+ 使用。"""
    stats = model.module_c_ema_stats
    K = stats['n_k_ema'].numel()
    D = stats['Sigma_k_ema'].size(1)
    eye = torch.eye(D, device=stats['Sigma_k_ema'].device)

    for k in range(K):
        n_k = float(epoch_stats['count'][k].item())
        if n_k <= 0:
            continue
        sum_x = epoch_stats['sum_x'][k]
        sum_xx = epoch_stats['sum_xx'][k]
        mu = sum_x / n_k
        if n_k > 1:
            cov = (sum_xx - n_k * torch.ger(mu, mu)) / max(n_k - 1.0, 1.0)
        else:
            cov = eye
        cov = cov + 1e-6 * eye

        if len(epoch_stats['dist_buf'][k]) > 0:
            d_cat = torch.cat(epoch_stats['dist_buf'][k], dim=0)
            r_k = float(torch.quantile(d_cat.to(mu.device), 0.7).item())
        else:
            r_k = float(stats['R_k_ema'][k].item())

        gamma_k = float(epoch_stats['gamma_sum'][k].item() / max(n_k, 1.0))
        rho = float(ema_rho)
        stats['n_k_ema'][k] = rho * stats['n_k_ema'][k] + (1.0 - rho) * n_k
        stats['R_k_ema'][k] = rho * stats['R_k_ema'][k] + (1.0 - rho) * r_k
        stats['gamma_bar_k_ema'][k] = rho * stats['gamma_bar_k_ema'][k] + (1.0 - rho) * gamma_k
        stats['Sigma_k_ema'][k] = rho * stats['Sigma_k_ema'][k] + (1.0 - rho) * cov


def _build_pairwise_prob_weights(common_z, memberships_cons, uncertain_mask=None,
                                 pos_mask=None,
                                 neg_mode='batch', knn_k=20,
                                 w_min=0.05, w_max=1.0,
                                 lambda_h=0.5, hn_s0=0.3, hn_th=0.1,
                                 wneg_stopgrad_q=True, eps=1e-12):
    """
    模块B（prob）：连续概率权重替代分位数配额。
    - p_ij=<q_i,q_j> 作为 FN 风险证据；w 对 p 单调递减。
    - h_ij 仅在异簇证据 (1-p_ij) 上增强，且不允许抵消 FN 降权主趋势。
    """
    device = common_z.device
    N = common_z.size(0)
    neg_mask, eye = _compute_neg_mask(common_z, neg_mode=neg_mode, knn_k=knn_k)

    q_w = memberships_cons.detach() if wneg_stopgrad_q else memberships_cons
    p = torch.mm(q_w, q_w.t()).clamp(0.0, 1.0)
    sim = F.cosine_similarity(common_z.unsqueeze(1), common_z.unsqueeze(0), dim=2)

    one_minus_p = (1.0 - p).clamp(0.0, 1.0)
    h = torch.sigmoid((sim - float(hn_s0)) / max(float(hn_th), eps)) * one_minus_p

    # 基础 FN 去污染 + 可选 HN 增强（模块B pair层）
    w_base = one_minus_p
    w = w_base * (1.0 + float(lambda_h) * h)
    w_base_cap = torch.clamp(w_base, min=float(w_min), max=float(w_max))
    w = torch.clamp(w, min=float(w_min), max=float(w_max))
    w = torch.minimum(w, w_base_cap)  # 防止 HN 增强抵消 FN 降权主趋势

    # 仅在真实 negative 上生效：diag 与正对永不参与负分母
    if pos_mask is not None:
        neg_mask = neg_mask & (~pos_mask.bool())
    if uncertain_mask is not None:
        anchor_mask = uncertain_mask.unsqueeze(1).bool()
        neg_mask = neg_mask & anchor_mask
    w_neg = torch.zeros_like(w)
    w_neg[neg_mask] = w[neg_mask]
    w_neg = w_neg.masked_fill(eye, 0.0)

    neg_vals_w = w_neg[neg_mask]
    neg_vals_p = p[neg_mask]
    neg_vals_h = h[neg_mask]

    def _safe_quantile(x, q):
        if x.numel() == 0:
            return 0.0
        return float(torch.quantile(x, q).item())

    exp_sim = torch.exp(sim)
    denom_all = (w_neg * exp_sim * neg_mask.float()).sum() + eps
    fn_proxy_mask = (p > 0.5) & neg_mask
    denom_fn = (w_neg * exp_sim * fn_proxy_mask.float()).sum()  # 模块A联动监控：高同簇证据在分母占比
    denom_fn_share = (denom_fn / denom_all).item()

    stats = {
        'fn_ratio': 0.0,
        'safe_ratio': 0.0,
        'hn_ratio': 0.0,
        'FN_count': float((p[neg_mask] > 0.5).float().sum().item()),
        'HN_count': float((h[neg_mask] > 0.5).float().sum().item()),
        'neg_count': float(neg_mask.float().sum().item()),
        'safe_neg_count': float(neg_mask.float().sum().item()),
        'w_mean_on_FN': float(w_neg[(neg_mask & (p > 0.5))].mean().item()) if (neg_mask & (p > 0.5)).any() else 0.0,
        'w_mean_on_safe': float(neg_vals_w.mean().item()) if neg_vals_w.numel() > 0 else 0.0,
        'mean_s_post_fn': float(p[(neg_mask & (p > 0.5))].mean().item()) if (neg_mask & (p > 0.5)).any() else 0.0,
        'mean_s_post_non_fn': float(p[(neg_mask & (p <= 0.5))].mean().item()) if (neg_mask & (p <= 0.5)).any() else 0.0,
        'delta_post': 0.0,
        'mean_sim_hn': float(sim[(neg_mask & (h > 0.5))].mean().item()) if (neg_mask & (h > 0.5)).any() else 0.0,
        'mean_sim_safe_non_hn': float(sim[(neg_mask & (h <= 0.5))].mean().item()) if (neg_mask & (h <= 0.5)).any() else 0.0,
        'delta_sim': 0.0,
        'label_flip': 0.0,
        'stab_rate': 0.0,
        'denom_fn_share': denom_fn_share,
        'denom_safe_share': 1.0 - denom_fn_share,
        'w_hit_min_ratio': float((neg_vals_w <= (float(w_min) + eps)).float().mean().item()) if neg_vals_w.numel() > 0 else 0.0,
        'corr_u_fn_ratio': 0.0,
        'N_size': float(neg_mask.float().sum(dim=1).mean().item()) if N > 0 else 0.0,
        'neg_per_anchor': float(neg_mask.float().sum(dim=1).mean().item()) if N > 0 else 0.0,
        'U_size': int(uncertain_mask.sum().item()) if uncertain_mask is not None else int(N),
        'fn_pair_share': float(((p[neg_mask] > 0.5).float().mean().item())) if neg_vals_p.numel() > 0 else 0.0,
        'hn_pair_share': float(((h[neg_mask] > 0.5).float().mean().item())) if neg_vals_h.numel() > 0 else 0.0,
        'w_neg_mean': float(neg_vals_w.mean().item()) if neg_vals_w.numel() > 0 else 0.0,
        'w_neg_p50': _safe_quantile(neg_vals_w, 0.5),
        'w_neg_p90': _safe_quantile(neg_vals_w, 0.9),
        'pij_mean_on_neg': float(neg_vals_p.mean().item()) if neg_vals_p.numel() > 0 else 0.0,
        'hij_mean': float(neg_vals_h.mean().item()) if neg_vals_h.numel() > 0 else 0.0,
        'wneg_active_pairs': float(neg_mask.float().sum().item()),
    }
    stats['delta_post'] = stats['mean_s_post_fn'] - stats['mean_s_post_non_fn']
    stats['delta_sim'] = stats['mean_sim_hn'] - stats['mean_sim_safe_non_hn']

    aux = {
        'S': torch.zeros_like(sim),
        's_post': p,
        'sim': sim,
        'rho': torch.zeros(N, N, dtype=torch.bool, device=device),
        'eta': (h > 0.5) & neg_mask,
        'w_neg': w_neg,
        'r': torch.ones_like(sim),
        's_stab': torch.zeros_like(sim),
        'neg_mask': neg_mask,
        'tau_fn_per_anchor': torch.full((N,), float('nan'), device=device),
        'tau_hn_per_anchor': torch.full((N,), float('nan'), device=device),
        'FN_count_per_anchor': torch.zeros(N, device=device),
        'HN_count_per_anchor': ((h > 0.5) & neg_mask).float().sum(dim=1),
        'h': h,
        'p': p,
        'neg_mask_effective': neg_mask,
    }
    return w_neg, aux['eta'], aux['rho'], stats, aux


def _build_pairwise_fn_risk(common_z, memberships_cons, u_hat, batch_labels, prev_labels_batch,
                            gate_val, alpha_fn=0.1, pi_fn=0.1, w_min=0.05,
                            hn_beta=0.1, neg_mode='batch', knn_k=20,
                            uncertain_mask=None, eps=1e-12):
    """
    Design 1': pair-wise FN risk routing.
    - 对 negative pair (i,j) 估计 FN 风险并在 InfoNCE 分母软降权
    - 在可信 negatives 中按分位数选择 hard negatives（eta 矩阵）
    """
    device = common_z.device
    N = common_z.size(0)

    eye = torch.eye(N, dtype=torch.bool, device=device)
    neg_mask = ~eye
    if neg_mode == 'knn':
        k_eff = min(knn_k + 1, N)
        dist = torch.cdist(common_z, common_z, p=2)
        knn_idx = torch.topk(-dist, k_eff, dim=1).indices
        knn_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
        row_idx = torch.arange(N, device=device).unsqueeze(1).expand_as(knn_idx)
        knn_mask[row_idx, knn_idx] = True
        neg_mask = knn_mask & (~eye)

    # (E1) posterior same-cluster evidence
    s_post = torch.mm(memberships_cons, memberships_cons.t()).clamp(0.0, 1.0)

    # (E2) stability evidence
    if prev_labels_batch is None:
        s_stab = torch.zeros(N, N, device=device)
    else:
        stab_vec = (batch_labels == prev_labels_batch).float()
        s_stab = torch.ger(stab_vec, stab_vec)

    # (E3) reliability from uncertainty
    r = (1.0 - 0.5 * (u_hat.unsqueeze(1) + u_hat.unsqueeze(0))).clamp(0.0, 1.0)

    # (E4) neighborhood overlap evidence，按 gate 渐进启用
    if N > 1:
        k_nb = min(knn_k + 1, N)
        dist = torch.cdist(common_z, common_z, p=2)
        nb_idx = torch.topk(-dist, k_nb, dim=1).indices[:, 1:]
        nb_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
        rr = torch.arange(N, device=device).unsqueeze(1).expand_as(nb_idx)
        nb_mask[rr, nb_idx] = True
        inter = (nb_mask.unsqueeze(1) & nb_mask.unsqueeze(0)).sum(dim=2).float()
        union = (nb_mask.unsqueeze(1) | nb_mask.unsqueeze(0)).sum(dim=2).float()
        s_nbr = (inter / (union + eps)).clamp(0.0, 1.0)
    else:
        s_nbr = torch.zeros(N, N, device=device)

    def _logit(x):
        x = x.clamp(min=eps, max=1.0 - eps)
        return torch.log(x / (1.0 - x))

    S = r * s_stab * _logit(s_post) + gate_val * r * _logit(s_nbr + eps)
    S = S.masked_fill(~neg_mask, float('-inf'))

    # per-anchor quantile threshold for FN-risk pairs
    rho = torch.zeros(N, N, dtype=torch.bool, device=device)
    w_neg = torch.ones(N, N, device=device)
    fn_ratio_list = []
    fn_ratio_per_anchor = torch.zeros(N, device=device)
    tau_fn_per_anchor = torch.full((N,), float('nan'), device=device)
    fn_count_per_anchor = torch.zeros(N, device=device)
    for i in range(N):
        idx = neg_mask[i].nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        s_i = S[i, idx]
        tau_i = torch.quantile(s_i, max(0.0, min(1.0, 1.0 - alpha_fn)))
        tau_fn_per_anchor[i] = tau_i
        rho_i = s_i >= tau_i
        if uncertain_mask is not None and (not bool(uncertain_mask[i])):
            rho_i = torch.zeros_like(rho_i)
        rho[i, idx] = rho_i
        w_neg[i, idx[rho_i]] = max(gate_val * pi_fn, w_min)
        fn_count_i = rho_i.float().sum()
        fn_count_per_anchor[i] = fn_count_i
        fn_ratio_i = rho_i.float().mean()
        fn_ratio_list.append(fn_ratio_i)
        fn_ratio_per_anchor[i] = fn_ratio_i

    # HN from safe negatives by similarity quantile
    sim = F.cosine_similarity(common_z.unsqueeze(1), common_z.unsqueeze(0), dim=2)
    eta = torch.zeros(N, N, dtype=torch.bool, device=device)
    hn_ratio_list = []
    tau_hn_per_anchor = torch.full((N,), float('nan'), device=device)
    hn_count_per_anchor = torch.zeros(N, device=device)
    for i in range(N):
        safe_idx = (neg_mask[i] & (~rho[i])).nonzero(as_tuple=True)[0]
        if safe_idx.numel() == 0:
            continue
        sim_i = sim[i, safe_idx]
        tau_hn_i = torch.quantile(sim_i, max(0.0, min(1.0, 1.0 - hn_beta)))
        tau_hn_per_anchor[i] = tau_hn_i
        eta_i = sim_i >= tau_hn_i
        if uncertain_mask is not None and (not bool(uncertain_mask[i])):
            eta_i = torch.zeros_like(eta_i)
        eta[i, safe_idx] = eta_i
        hn_count_per_anchor[i] = eta_i.float().sum()
        hn_ratio_list.append(eta_i.float().mean())

    u_center = u_hat - u_hat.mean()
    fn_center = fn_ratio_per_anchor - fn_ratio_per_anchor.mean()
    denom = (u_center.norm() * fn_center.norm() + eps)
    corr_u_fn = (u_center * fn_center).sum() / denom

    safe_mask = neg_mask & (~rho)
    non_hn_safe_mask = safe_mask & (~eta)
    mean_s_post_fn = s_post[rho].mean().item() if rho.any() else 0.0
    mean_s_post_non_fn = s_post[safe_mask].mean().item() if safe_mask.any() else 0.0
    mean_sim_hn = sim[eta].mean().item() if eta.any() else 0.0
    mean_sim_safe_non_hn = sim[non_hn_safe_mask].mean().item() if non_hn_safe_mask.any() else 0.0

    exp_sim = torch.exp(sim)
    denom_all = (w_neg * exp_sim * neg_mask.float()).sum() + eps
    denom_fn = (w_neg * exp_sim * rho.float()).sum()
    denom_fn_share = (denom_fn / denom_all).item()

    fn_mask = neg_mask & rho
    hn_mask = safe_mask & eta
    fn_count = fn_mask.float().sum().item()
    hn_count = hn_mask.float().sum().item()
    neg_count = neg_mask.float().sum().item()
    safe_neg_count = safe_mask.float().sum().item()

    stats = {
        'fn_ratio': torch.stack(fn_ratio_list).mean().item() if fn_ratio_list else 0.0,
        'safe_ratio': ((safe_mask.float().sum() / (neg_mask.float().sum() + eps))).item(),
        'hn_ratio': torch.stack(hn_ratio_list).mean().item() if hn_ratio_list else 0.0,
        'FN_count': fn_count,
        'HN_count': hn_count,
        'neg_count': neg_count,
        'safe_neg_count': safe_neg_count,
        'w_mean_on_FN': w_neg[rho].mean().item() if rho.any() else 0.0,
        'w_mean_on_safe': w_neg[safe_mask].mean().item() if safe_mask.any() else 0.0,
        'mean_s_post_fn': mean_s_post_fn,
        'mean_s_post_non_fn': mean_s_post_non_fn,
        'delta_post': mean_s_post_fn - mean_s_post_non_fn,
        'mean_sim_hn': mean_sim_hn,
        'mean_sim_safe_non_hn': mean_sim_safe_non_hn,
        'delta_sim': mean_sim_hn - mean_sim_safe_non_hn,
        'label_flip': (1.0 - s_stab.diag().mean().item()) if prev_labels_batch is not None else 0.0,
        'stab_rate': s_stab.diag().mean().item() if prev_labels_batch is not None else 0.0,
        'denom_fn_share': denom_fn_share,
        'denom_safe_share': 1.0 - denom_fn_share,
        'w_hit_min_ratio': ((w_neg <= (w_min + eps)) & rho).float().mean().item() if rho.any() else 0.0,
        'corr_u_fn_ratio': corr_u_fn.item(),
        'N_size': (neg_mask.float().sum(dim=1).mean().item()),
        'neg_per_anchor': (neg_mask.float().sum(dim=1).mean().item()),
        'U_size': int(uncertain_mask.sum().item()) if uncertain_mask is not None else int(N),
        'fn_pair_share': (fn_count / max(neg_count, 1.0)),
        'hn_pair_share': (hn_count / max(safe_neg_count, 1.0)),
    }
    aux = {
        'S': S, 's_post': s_post, 'sim': sim, 'rho': rho, 'eta': eta, 'w_neg': w_neg,
        'r': r, 's_stab': s_stab, 'neg_mask': neg_mask,
        'tau_fn_per_anchor': tau_fn_per_anchor, 'tau_hn_per_anchor': tau_hn_per_anchor,
        'FN_count_per_anchor': fn_count_per_anchor, 'HN_count_per_anchor': hn_count_per_anchor,
        'neg_mask_effective': neg_mask,
    }
    return w_neg, eta, rho, stats, aux


def contrastive_train(model, mv_data, mvc_loss,
                      batch_size, epoch, W,
                      alpha, beta,
                      optimizer,
                      warmup_epochs,
                      lambda_u,  lambda_hn_penalty,
                      temperature_f, max_epoch=100,
                      initial_top_p=0.3,
                      cross_warmup_epochs=50,
                      alpha_fn=0.1,
                      pi_fn=0.1,
                      w_min=0.05,
                      hn_beta=0.1,
                      neg_mode='batch',
                      knn_neg_k=20,
                      route_uncertain_only=True,
                      y_prev_labels=None,
                      p_min=0.05,
                      u_min=32,
                      u_threshold_method='otsu',
                      u_tau_ema_rho=0.9,
                      min_uncertain_ratio=0.02,
                      max_uncertain_ratio=0.6,
                      theta_temperature=0.5,
                      theta_threshold=0.5,
                      enable_theta_certificate=True,
                      module_b_mode='legacy',
                      module_b_gate_mode='linear',
                      w_max=1.0,
                      lambda_h=0.5,
                      hn_s0=0.3,
                      hn_th=0.1,
                      wneg_stopgrad_q=True,
                      gate_stab_s0=0.5,
                      gate_stab_tg=0.1,
                      gate_stab_ema_rho=0.9,
                      module_c_stat_ema_rho=0.9):
    model.train()
    mv_data_loader, num_views, num_samples, num_clusters = get_multiview_data(mv_data, batch_size)

    # 将 all_features 和 all_labels 初始化为 Python 列表
    all_features = []  # 用于收集每个批次的特征
    all_labels = []  # 用于收集每个批次的标签

    if not hasattr(model, 'tau_u_ema'):
        model.tau_u_ema = None
    if not hasattr(model, 'gate_stab_ema'):
        model.gate_stab_ema = None

    # E 步：更新全量伪标签
    psedo_labeling(model, mv_data, batch_size)

    # Push/Pull Lpen 超参
    lambda_push = lambda_hn_penalty
    lambda_pull = lambda_hn_penalty
    margin = 0.2

    criterion = torch.nn.MSELoss()  # 添加重建损失的损失函数

    epoch_meter = {'L_total':0.0,'L_recon':0.0,'L_feat':0.0,'L_cross':0.0,'L_cluster':0.0,'L_uncert':0.0,'L_hn':0.0,'L_reg':0.0}
    route_meter = {'fn_ratio':0.0,'safe_ratio':0.0,'hn_ratio':0.0,'mean_s_post_fn':0.0,'mean_s_post_non_fn':0.0,'delta_post':0.0,'mean_sim_hn':0.0,'mean_sim_safe_non_hn':0.0,'delta_sim':0.0,'label_flip':0.0,'stab_rate':0.0,'denom_fn_share':0.0,'denom_safe_share':0.0,'w_hit_min_ratio':0.0,'corr_u_fn_ratio':0.0,'N_size':0.0,'U_size':0.0,'tau_u':0.0,'unsafe_ratio':0.0,'theta_p50_batch':0.0,'theta_p90_batch':0.0,'k_unc':0.0,'w_neg_mean':0.0,'w_neg_p50':0.0,'w_neg_p90':0.0,'pij_mean_on_neg':0.0,'hij_mean':0.0,'wneg_active_pairs':0.0,'gate_stab':0.0,'stab_t':0.0,'EMA_stab':0.0,
                   'minority_set_size':0.0,'minority_count_mean':0.0,'minority_radius_mean':0.0,'minority_gamma_mean':0.0,
                   'module_c_sample_fail_rate':0.0,'module_c_fallback_rate':0.0,'L_proto_align':0.0,'L_conf_repulse':0.0}
    batch_count = 0
    last_dump = {}
    module_c_epoch_stats = None

    for batch_idx, (sub_data_views, _, sample_idx) in enumerate(mv_data_loader):
        # ——— 1) 伪标签 & 同/异样本矩阵 ———
        batch_psedo_label = model.psedo_labels[sample_idx]                # [N]
        y_matrix = (batch_psedo_label.unsqueeze(1) == batch_psedo_label.unsqueeze(0)).int()

        # ——— 2) 编码 + 融合 ———
        xrs, zs = model(sub_data_views)
        common_z = model.fusion(zs)

        # 现在有了 common_z，确定 device
        device = common_z.device

        if module_c_epoch_stats is None:
            _ensure_module_c_ema_stats(model, num_clusters, common_z.size(1), device)
            module_c_epoch_stats = _init_module_c_epoch_stats(num_clusters, common_z.size(1), device)

        # 把索引张量都搬到 device
        batch_psedo_label = batch_psedo_label.to(device)

        # ——— 3) 更新中心 + 隶属度 + 不确定度 ———
        model.update_centers(zs, common_z)
        features = zs + [common_z]
        memberships = [model.compute_membership(features[v], v) for v in range(num_views + 1)]
        sample_idx_dev = sample_idx.to(device)
        u, u_hat, u_aux = model.estimate_uncertainty(
            memberships, common_z,
            sample_idx=sample_idx_dev,
            update_ema=True,
            return_parts=True,
        )
        batch_N  = u_hat.size(0)

        # ——— 4) 模块A自适应不确定划分：tau_u + 证书扩展 + 保底约束 ———
        tau_u_batch = _adaptive_u_threshold(u, method=u_threshold_method)
        if model.tau_u_ema is None:
            model.tau_u_ema = tau_u_batch
        else:
            model.tau_u_ema = float(u_tau_ema_rho) * float(model.tau_u_ema) + (1.0 - float(u_tau_ema_rho)) * float(tau_u_batch)
        tau_u = float(model.tau_u_ema)

        uncertain_mask = (u > tau_u)
        theta = _compute_theta_certificate(common_z, memberships[num_views], temperature=theta_temperature)
        unsafe_mask = (theta > theta_threshold) if enable_theta_certificate else torch.zeros_like(uncertain_mask)
        uncertain_mask = uncertain_mask | unsafe_mask
        uncertain_mask, k_unc = _bounded_uncertain_mask(
            u, uncertain_mask,
            min_ratio=min_uncertain_ratio,
            max_ratio=max_uncertain_ratio,
        )
        idx_topk = uncertain_mask.nonzero(as_tuple=True)[0]
        certain_mask = ~uncertain_mask
        u_thr = u[idx_topk].min().item() if idx_topk.numel() > 0 else 0.0

        print(f"Batch {batch_idx}: uncertain {uncertain_mask.sum().item()}/{batch_N} = {uncertain_mask.sum().item()/batch_N:.2%}")

        # ——— 5) 动态门控 Gate（模块B：线性/稳定性触发可切换）———
        u_mean = u_hat.mean().item()
        mu_start, mu_end = 0.3, 0.7
        raw_gate = (u_mean - mu_start) / (mu_end - mu_start)
        gate_u = float(max(0.0, min(1.0, raw_gate)))
        t = float((epoch - 1) / max(1, max_epoch - 1))
        gate_val, stab_t, ema_stab = _compute_gate_by_stability(
            d_time=u_aux['d_time'],
            gate_mode=module_b_gate_mode,
            linear_t=t,
            prev_ema=model.gate_stab_ema,
            ema_rho=gate_stab_ema_rho,
            s0=gate_stab_s0,
            tg=gate_stab_tg,
        )
        model.gate_stab_ema = ema_stab
        gate_fn = gate_val
        gate_hn = gate_val
        gate = torch.tensor(gate_val, device=device)

        # ——— 6) 计算共识中心 q_centers ———
        q_centers = model.compute_centers(common_z, batch_psedo_label)
        _accumulate_module_c_epoch_stats(module_c_epoch_stats, batch_psedo_label, memberships[num_views], common_z, q_centers)

        # ——— 7) 模块B路由：legacy / prob 可切换（保证baseline可复现）———
        prev_batch = None if y_prev_labels is None else y_prev_labels[sample_idx].to(device)
        route_mask = uncertain_mask if route_uncertain_only else None
        if module_b_mode == 'prob':
            w_neg, eta_mat, rho_mat, route_stats, route_aux = _build_pairwise_prob_weights(
                common_z=common_z,
                memberships_cons=memberships[num_views],
                uncertain_mask=route_mask,
                pos_mask=y_matrix.bool().to(device),
                neg_mode=neg_mode,
                knn_k=knn_neg_k,
                w_min=w_min,
                w_max=w_max,
                lambda_h=lambda_h,
                hn_s0=hn_s0,
                hn_th=hn_th,
                wneg_stopgrad_q=wneg_stopgrad_q,
            )
        else:
            w_neg, eta_mat, rho_mat, route_stats, route_aux = _build_pairwise_fn_risk(
                common_z=common_z,
                memberships_cons=memberships[num_views],
                u_hat=u_hat,
                batch_labels=batch_psedo_label,
                prev_labels_batch=prev_batch,
                gate_val=gate_val,
                alpha_fn=alpha_fn,
                pi_fn=pi_fn,
                w_min=w_min,
                hn_beta=hn_beta,
                neg_mode=neg_mode,
                knn_k=knn_neg_k,
                uncertain_mask=route_mask,
            )

        # ——— 8) 累加各项损失 ———
        loss_list = []
        Lcl = Lfeat = Lu = Lpen = Lcross = Lrecon = 0.0  # 添加重建损失

        for v in range(num_views):
            # 准备 Wv 和 y_pse
            Wv = W[v][sample_idx][:, sample_idx].to(device)
            y_pse = y_matrix.float().to(device)

            # a) 簇级 InfoNCE
            k_centers = model.compute_centers(zs[v], batch_psedo_label)
            if epoch <= 50:
                cl, module_c_detail = mvc_loss.compute_cluster_loss(
                    q_centers, k_centers, batch_psedo_label,
                    return_details=True
                )
            else:
                cl, _, _, module_c_detail = mvc_loss.compute_cluster_loss(
                    q_centers, k_centers, batch_psedo_label,
                    features_batch=common_z,
                    module_c_stats=model.module_c_ema_stats,
                    return_mmd_excl=True,
                    return_details=True,
                )
            for mk in ['minority_set_size','minority_count_mean','minority_radius_mean','minority_gamma_mean',
                       'module_c_sample_fail_rate','module_c_fallback_rate','L_proto_align','L_conf_repulse']:
                route_stats[mk] = module_c_detail.get(mk, 0.0)
            Lcl_i = alpha * cl
            Lcl += Lcl_i.item()
            loss_list.append(Lcl_i)

            # b) Feature loss + 软屏蔽 FN
            feat_loss = mvc_loss.feature_loss(zs[v], common_z, Wv, y_pse, neg_weights=w_neg)
            Lfeat_i = beta * feat_loss
            Lfeat += Lfeat_i.item()
            loss_list.append(Lfeat_i)

            # c) 不确定度回归
            u_loss = mvc_loss.uncertainty_regression_loss(u_hat, u)
            Lu_i = (1 - gate_u) * lambda_u * u_loss
            Lu += Lu_i.item()
            loss_list.append(Lu_i)

            # d) Hard-Negative penalty from safe negatives (Design 1')
            sim_mat = F.cosine_similarity(common_z.unsqueeze(1), common_z.unsqueeze(0), dim=2)
            pos_sim = torch.diag(sim_mat)
            eta_cnt = eta_mat.sum().float()
            if eta_cnt > 0:
                push_loss = torch.relu((sim_mat - pos_sim.unsqueeze(1) + margin) * eta_mat.float()).sum() / (eta_cnt + 1e-12)
            else:
                push_loss = torch.tensor(0.0, device=device)
            pull_loss = (1.0 - pos_sim).mean()
            Lpen_i = gate_hn * (lambda_push * push_loss + lambda_pull * pull_loss)

            Lpen += Lpen_i.item()
            loss_list.append(Lpen_i)

            # e) 跨视图加权 InfoNCE
            if epoch > cross_warmup_epochs:
                cross_l = mvc_loss.cross_view_weighted_loss(
                    model, zs, common_z, memberships,
                    batch_psedo_label, temperature=temperature_f
                )
                Lcross_i = gate_fn * beta  * cross_l
                Lcross += Lcross_i.item()
                loss_list.append(Lcross_i)

            # f) 每个视图的重建损失
            recon_loss = criterion(sub_data_views[v], xrs[v])  # 计算每个视图的重建损失
            Lrecon += recon_loss.item()
            loss_list.append(recon_loss)  # 加入总损失

        # ——— 9) 梯度更新 & 打印 ———
        total_loss = sum(loss_list)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_meter['L_total'] += total_loss.item()
        epoch_meter['L_recon'] += Lrecon
        epoch_meter['L_feat'] += Lfeat
        epoch_meter['L_cross'] += Lcross
        epoch_meter['L_cluster'] += Lcl
        epoch_meter['L_uncert'] += Lu
        epoch_meter['L_hn'] += Lpen
        for k in route_meter:
            route_meter[k] += route_stats.get(k, 0.0)
        batch_count += 1

        m_cons = memberships[num_views]
        top2_m = torch.topk(m_cons, 2, dim=1).values
        gamma = torch.log((top2_m[:, 0] + 1e-12) / (top2_m[:, 1] + 1e-12))
        sim_mat = route_aux['sim']
        pos_sim = F.cosine_similarity(zs[0], common_z, dim=1)
        neg_sim = sim_mat[route_aux.get('neg_mask_effective', route_aux['neg_mask'])]
        route_stats['U_ratio'] = float(k_unc) / max(batch_N, 1)
        route_stats['u_thr'] = u_thr
        route_stats['tau_u'] = tau_u
        route_stats['unsafe_ratio'] = float(unsafe_mask.float().mean().item())
        route_stats['theta_p50_batch'] = float(torch.quantile(theta.detach(), 0.5).item())
        route_stats['theta_p90_batch'] = float(torch.quantile(theta.detach(), 0.9).item())
        route_stats['k_unc'] = k_unc
        route_stats['gate_stab'] = gate_val
        route_stats['stab_t'] = stab_t
        route_stats['EMA_stab'] = ema_stab

        last_dump = {
            'u_sample': u.detach().cpu(),
            'gamma_sample': u_aux['gamma'].detach().cpu(),
            'd_view_sample': u_aux['d_view'].detach().cpu(),
            'd_time_sample': u_aux['d_time'].detach().cpu(),
            'm_top1_sample': top2_m[:, 0].detach().cpu(),
            'm_gap_sample': (top2_m[:, 0] - top2_m[:, 1]).detach().cpu(),
            'y_curr_sample': batch_psedo_label.detach().cpu(),
            'y_prev_sample': (prev_batch.detach().cpu() if prev_batch is not None else torch.full_like(batch_psedo_label.detach().cpu(), -1)),
            'flip_mask_sample': ((batch_psedo_label != prev_batch).float().detach().cpu() if prev_batch is not None else torch.zeros_like(batch_psedo_label, dtype=torch.float32).detach().cpu()),
            'S_pair_sample': route_aux['S'][route_aux['neg_mask']].detach().cpu(),
            'w_pair_sample': route_aux['w_neg'][route_aux['neg_mask']].detach().cpu(),
            's_post_pair_sample': route_aux['s_post'][route_aux['neg_mask']].detach().cpu(),
            'sim_pair_sample': neg_sim.detach().cpu(),
            'rho_fn_pair_sample': route_aux['rho'][route_aux['neg_mask']].float().detach().cpu(),
            'eta_hn_pair_sample': route_aux['eta'][route_aux['neg_mask']].float().detach().cpu(),
            'r_pair_sample': route_aux['r'][route_aux['neg_mask']].detach().cpu(),
            's_stab_pair_sample': route_aux['s_stab'][route_aux['neg_mask']].detach().cpu(),
            'sim_pos_sample': pos_sim.detach().cpu(),
            'sim_neg_sample': neg_sim.detach().cpu(),
            'pairs_sampled': torch.tensor(float(neg_sim.numel())),
            'neg_pairs_available': torch.tensor(float(route_aux['neg_mask'].float().sum().item())),
            'safe_pairs_available': torch.tensor(float((route_aux['neg_mask'] & (~route_aux['rho'])).float().sum().item())),
            'pos_pairs_count': torch.tensor(float(pos_sim.numel())),
            'uncertain_mask_sample': uncertain_mask.detach().cpu(),
            'unsafe_mask_sample': unsafe_mask.detach().cpu(),
            'theta_sample': theta.detach().cpu(),
            'neg_mask_sample': route_aux['neg_mask'].detach().cpu(),
            'tau_u': torch.tensor(tau_u),
            'k_unc': torch.tensor(k_unc),
            'tau_fn_per_anchor': route_aux['tau_fn_per_anchor'].detach().cpu(),
            'tau_hn_per_anchor': route_aux['tau_hn_per_anchor'].detach().cpu(),
            'FN_count_per_anchor': route_aux['FN_count_per_anchor'].detach().cpu(),
            'HN_count_per_anchor': route_aux['HN_count_per_anchor'].detach().cpu(),
            'gate_val': torch.tensor(gate_val),
            'p_pair_sample': route_aux['p'][route_aux['neg_mask_effective']].detach().cpu() if 'p' in route_aux else torch.zeros(0),
            'h_pair_sample': route_aux['h'][route_aux['neg_mask_effective']].detach().cpu() if 'h' in route_aux else torch.zeros(0),
            'minority_set_size': torch.tensor(route_stats.get('minority_set_size', 0.0)),
            'minority_count_mean': torch.tensor(route_stats.get('minority_count_mean', 0.0)),
            'minority_radius_mean': torch.tensor(route_stats.get('minority_radius_mean', 0.0)),
            'minority_gamma_mean': torch.tensor(route_stats.get('minority_gamma_mean', 0.0)),
            'module_c_sample_fail_rate': torch.tensor(route_stats.get('module_c_sample_fail_rate', 0.0)),
            'module_c_fallback_rate': torch.tensor(route_stats.get('module_c_fallback_rate', 0.0)),
            'L_proto_align': torch.tensor(route_stats.get('L_proto_align', 0.0)),
            'L_conf_repulse': torch.tensor(route_stats.get('L_conf_repulse', 0.0)),
        }

        print(f"[Epoch {epoch} Batch {batch_idx}] "
              f"Total={total_loss.item():.4f}  "
              f"FN_ratio={route_stats['fn_ratio']:.3f} HN_ratio={route_stats['hn_ratio']:.3f} "
              f"U_ratio={route_stats['U_ratio']:.3f} stab={route_stats['stab_rate']:.3f}")

    # ===== 训练循环结束 =====
    if module_c_epoch_stats is not None:
        _update_module_c_ema_stats(model, module_c_epoch_stats, ema_rho=module_c_stat_ema_rho)

    if batch_count > 0:
        for k in epoch_meter:
            epoch_meter[k] /= batch_count
        for k in route_meter:
            route_meter[k] /= batch_count

    return {'loss': epoch_meter, 'route': route_meter, 'dump': last_dump, 'gate': gate_val if batch_count > 0 else 0.0, 'gate_u': gate_u if batch_count > 0 else 0.0, 'gate_fn': gate_fn if batch_count > 0 else 0.0, 'gate_hn': gate_hn if batch_count > 0 else 0.0, 't': t if batch_count > 0 else 0.0, 'warmup_epochs': warmup_epochs, 'cross_warmup_epochs': cross_warmup_epochs}




def contrastive_largedatasetstrain(model, mv_data, mvc_loss,
                                   batch_size, epoch, k,
                                   alpha, beta,
                                   optimizer,
                                   warmup_epochs=10,    # 默认热身 10 个 epoch
                                   prog=1.0,            # 默认进度权重 1.0
                                   lambda_u=0.1,        # 默认不确定度回归权重
                                   lambda_hn_penalty=0.1,
                                   temperature_f=0.5,    # 默认温度系数
                                   max_epoch=100,
                                   initial_top_p=0.3,
                                   cross_warmup_epochs=50,
                                   alpha_fn=0.1,
                                   pi_fn=0.1,
                                   w_min=0.05,
                                   hn_beta=0.1,
                                   neg_mode='batch',
                                   knn_neg_k=20,
                                   route_uncertain_only=True,
                                   y_prev_labels=None,
                                   p_min=0.05,
                                   u_min=32,
                                   u_threshold_method='otsu',
                                   u_tau_ema_rho=0.9,
                                   min_uncertain_ratio=0.02,
                                   max_uncertain_ratio=0.6,
                                   theta_temperature=0.5,
                                   theta_threshold=0.5,
                                   enable_theta_certificate=True,
                                   module_b_mode='legacy',
                                   module_b_gate_mode='linear',
                                   w_max=1.0,
                                   lambda_h=0.5,
                                   hn_s0=0.3,
                                   hn_th=0.1,
                                   wneg_stopgrad_q=True,
                                   gate_stab_s0=0.5,
                                   gate_stab_tg=0.1,
                                   gate_stab_ema_rho=0.9,
                                   module_c_stat_ema_rho=0.9):
    """
    大数据集版 Contrastive Training：
    - k: 用于构建每个视图下的 k-NN 图
    - 其它参数含义同原版 contrastive_train
    """
    model.train()
    mv_loader, num_views, _, num_clusters = get_multiview_data(mv_data, batch_size)
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    epoch_meter = {'L_total':0.0,'L_recon':0.0,'L_feat':0.0,'L_cross':0.0,'L_cluster':0.0,'L_uncert':0.0,'L_hn':0.0,'L_reg':0.0}
    route_meter = {'fn_ratio':0.0,'safe_ratio':0.0,'hn_ratio':0.0,'mean_s_post_fn':0.0,'mean_s_post_non_fn':0.0,'delta_post':0.0,'mean_sim_hn':0.0,'mean_sim_safe_non_hn':0.0,'delta_sim':0.0,'label_flip':0.0,'stab_rate':0.0,'denom_fn_share':0.0,'denom_safe_share':0.0,'w_hit_min_ratio':0.0,'corr_u_fn_ratio':0.0,'N_size':0.0,'U_size':0.0,'tau_u':0.0,'unsafe_ratio':0.0,'theta_p50_batch':0.0,'theta_p90_batch':0.0,'k_unc':0.0,'w_neg_mean':0.0,'w_neg_p50':0.0,'w_neg_p90':0.0,'pij_mean_on_neg':0.0,'hij_mean':0.0,'wneg_active_pairs':0.0,'gate_stab':0.0,'stab_t':0.0,'EMA_stab':0.0,
                   'minority_set_size':0.0,'minority_count_mean':0.0,'minority_radius_mean':0.0,'minority_gamma_mean':0.0,
                   'module_c_sample_fail_rate':0.0,'module_c_fallback_rate':0.0,'L_proto_align':0.0,'L_conf_repulse':0.0}
    batch_count = 0
    last_dump = {}
    module_c_epoch_stats = None

    if not hasattr(model, 'tau_u_ema'):
        model.tau_u_ema = None
    if not hasattr(model, 'gate_stab_ema'):
        model.gate_stab_ema = None

    # 2) E 步：更新全量伪标签
    psedo_labeling(model, mv_data, batch_size)

    for batch_idx, (sub_data_views, _, sample_idx) in enumerate(mv_loader):
        # ▶ 设备
        device = next(model.parameters()).device
        # ▶ 准备输入
        sub_views = [v.to(device) for v in sub_data_views]
        batch_label = model.psedo_labels[sample_idx].to(device)

        # ——— 伪标签相似矩阵 ———
        y_matrix = (batch_label.unsqueeze(1) == batch_label.unsqueeze(0)).int()

        # ——— 编码 + 融合 ———
        _, zs = model(sub_views)
        zs = [z_i.to(device) for z_i in zs]
        common_z = model.fusion(zs).to(device)
        if module_c_epoch_stats is None:
            _ensure_module_c_ema_stats(model, num_clusters, common_z.size(1), device)
            module_c_epoch_stats = _init_module_c_epoch_stats(num_clusters, common_z.size(1), device)

        # ——— 更新中心、隶属度、不确定度 ———
        model.update_centers(zs, common_z)
        feats = zs + [common_z]
        memberships = [model.compute_membership(feats[v], v) for v in range(num_views + 1)]
        sample_idx_dev = sample_idx.to(device)
        u, u_hat, u_aux = model.estimate_uncertainty(
            memberships, common_z,
            sample_idx=sample_idx_dev,
            update_ema=True,
            return_parts=True,
        )
        B = u_hat.size(0)

        # ——— 模块A自适应不确定划分：tau_u + 证书扩展 + 保底约束 ———
        tau_u_batch = _adaptive_u_threshold(u, method=u_threshold_method)
        if model.tau_u_ema is None:
            model.tau_u_ema = tau_u_batch
        else:
            model.tau_u_ema = float(u_tau_ema_rho) * float(model.tau_u_ema) + (1.0 - float(u_tau_ema_rho)) * float(tau_u_batch)
        tau_u = float(model.tau_u_ema)

        uncertain = (u > tau_u)
        theta = _compute_theta_certificate(common_z, memberships[num_views], temperature=theta_temperature)
        unsafe_mask = (theta > theta_threshold) if enable_theta_certificate else torch.zeros_like(uncertain)
        uncertain = uncertain | unsafe_mask
        uncertain, k_unc = _bounded_uncertain_mask(
            u, uncertain,
            min_ratio=min_uncertain_ratio,
            max_ratio=max_uncertain_ratio,
        )
        topk_idx = uncertain.nonzero(as_tuple=True)[0]
        certain = ~uncertain
        u_thr = u[topk_idx].min().item() if topk_idx.numel() > 0 else 0.0

        # ——— 动态门控 Gate（模块B：线性/稳定性触发可切换）———
        u_mean = u_hat.mean().item()
        mu_lo, mu_hi = 0.3, 0.7
        gate_u = float((u_mean - mu_lo) / (mu_hi - mu_lo))
        gate_u = max(0.0, min(1.0, gate_u))
        t = float((epoch - 1) / max(1, max_epoch - 1))
        gate, stab_t, ema_stab = _compute_gate_by_stability(
            d_time=u_aux['d_time'],
            gate_mode=module_b_gate_mode,
            linear_t=t,
            prev_ema=model.gate_stab_ema,
            ema_rho=gate_stab_ema_rho,
            s0=gate_stab_s0,
            tg=gate_stab_tg,
        )
        model.gate_stab_ema = ema_stab
        gate_fn = gate
        gate_hn = gate
        gate_t = torch.tensor(gate, device=device)

        # ——— 共识中心 ———
        q_centers = model.compute_centers(common_z, batch_label)
        _accumulate_module_c_epoch_stats(module_c_epoch_stats, batch_label, memberships[num_views], common_z, q_centers)

        # ——— 模块B路由：legacy / prob 可切换（保证baseline可复现）———
        prev_batch = None if y_prev_labels is None else y_prev_labels[sample_idx].to(device)
        route_mask = uncertain if route_uncertain_only else None
        if module_b_mode == 'prob':
            w_neg, eta_mat, rho_mat, route_stats, route_aux = _build_pairwise_prob_weights(
                common_z=common_z,
                memberships_cons=memberships[num_views],
                uncertain_mask=route_mask,
                pos_mask=y_matrix.bool().to(device),
                neg_mode=neg_mode,
                knn_k=knn_neg_k,
                w_min=w_min,
                w_max=w_max,
                lambda_h=lambda_h,
                hn_s0=hn_s0,
                hn_th=hn_th,
                wneg_stopgrad_q=wneg_stopgrad_q,
            )
        else:
            w_neg, eta_mat, rho_mat, route_stats, route_aux = _build_pairwise_fn_risk(
                common_z=common_z,
                memberships_cons=memberships[num_views],
                u_hat=u_hat,
                batch_labels=batch_label,
                prev_labels_batch=prev_batch,
                gate_val=gate,
                alpha_fn=alpha_fn,
                pi_fn=pi_fn,
                w_min=w_min,
                hn_beta=hn_beta,
                neg_mode=neg_mode,
                knn_k=knn_neg_k,
                uncertain_mask=route_mask,
            )

        # ——— 构造并累加各视图的损失 ———
        batch_loss = 0.0
        for v in range(num_views):
            # 动态 k-NN Graph
            Wv = get_knn_graph(sub_views[v], k).to(device)
            y_pse = y_matrix.float()

            # a) 簇级 InfoNCE
            kv_centers = model.compute_centers(zs[v], batch_label)
            if epoch <= 50:
                cl, module_c_detail = mvc_loss.compute_cluster_loss(
                    q_centers, kv_centers, batch_label,
                    return_details=True,
                )
            else:
                cl, _, _, module_c_detail = mvc_loss.compute_cluster_loss(
                    q_centers, kv_centers, batch_label,
                    features_batch=common_z,
                    module_c_stats=model.module_c_ema_stats,
                    return_mmd_excl=True,
                    return_details=True,
                )
            for mk in ['minority_set_size','minority_count_mean','minority_radius_mean','minority_gamma_mean',
                       'module_c_sample_fail_rate','module_c_fallback_rate','L_proto_align','L_conf_repulse']:
                route_stats[mk] = module_c_detail.get(mk, 0.0)
            Lcl = alpha * cl
            batch_loss += Lcl

            # b) Feature loss（软屏蔽 FN）
            feat_l = mvc_loss.feature_loss(zs[v], common_z, Wv, y_pse, neg_weights=w_neg)
            Lfeat = beta * feat_l
            batch_loss += Lfeat

            # c) 不确定度回归
            u_l = mvc_loss.uncertainty_regression_loss(u_hat, u)
            Lu = (1 - gate_u) * lambda_u * u_l
            batch_loss += Lu

            # d) Hard-Negative penalty from safe negatives (Design 1')
            sim_mat = F.cosine_similarity(common_z.unsqueeze(1), common_z.unsqueeze(0), dim=2)
            pos_sim = torch.diag(sim_mat)
            eta_cnt = eta_mat.sum().float()
            if eta_cnt > 0:
                push = torch.relu((sim_mat - pos_sim.unsqueeze(1) + 0.2) * eta_mat.float()).sum() / (eta_cnt + 1e-12)
            else:
                push = torch.tensor(0.0, device=device)
            pull = (1.0 - pos_sim).mean()
            Lpen = gate_hn * (lambda_hn_penalty * push + lambda_hn_penalty * pull)
            batch_loss += Lpen

            # e) 跨视图加权 InfoNCE
            if epoch > cross_warmup_epochs:
                cross_l = mvc_loss.cross_view_weighted_loss(
                    model, zs, common_z, memberships,
                    batch_label, temperature=temperature_f
                )
                Lcross = gate_fn * beta * prog * cross_l
                batch_loss += Lcross

        # ——— 梯度更新 ———
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()

        epoch_meter['L_total'] += batch_loss.item()
        epoch_meter['L_cluster'] += Lcl.item() if hasattr(Lcl, 'item') else float(Lcl)
        epoch_meter['L_feat'] += Lfeat.item() if hasattr(Lfeat, 'item') else float(Lfeat)
        epoch_meter['L_uncert'] += Lu.item() if hasattr(Lu, 'item') else float(Lu)
        epoch_meter['L_hn'] += Lpen.item() if hasattr(Lpen, 'item') else float(Lpen)
        epoch_meter['L_cross'] += Lcross.item() if ('Lcross' in locals() and hasattr(Lcross, 'item')) else 0.0
        for k in route_meter:
            route_meter[k] += route_stats.get(k, 0.0)
        batch_count += 1

        m_cons = memberships[num_views]
        top2_m = torch.topk(m_cons, 2, dim=1).values
        gamma = torch.log((top2_m[:, 0] + 1e-12) / (top2_m[:, 1] + 1e-12))
        sim_mat = route_aux['sim']
        pos_sim = F.cosine_similarity(zs[0], common_z, dim=1)
        neg_sim = sim_mat[route_aux.get('neg_mask_effective', route_aux['neg_mask'])]
        route_stats['U_ratio'] = float(k_unc) / max(B, 1)
        route_stats['u_thr'] = u_thr
        route_stats['tau_u'] = tau_u
        route_stats['unsafe_ratio'] = float(unsafe_mask.float().mean().item())
        route_stats['theta_p50_batch'] = float(torch.quantile(theta.detach(), 0.5).item())
        route_stats['theta_p90_batch'] = float(torch.quantile(theta.detach(), 0.9).item())
        route_stats['k_unc'] = k_unc
        route_stats['gate_stab'] = gate
        route_stats['stab_t'] = stab_t
        route_stats['EMA_stab'] = ema_stab

        last_dump = {
            'u_sample': u.detach().cpu(),
            'gamma_sample': u_aux['gamma'].detach().cpu(),
            'd_view_sample': u_aux['d_view'].detach().cpu(),
            'd_time_sample': u_aux['d_time'].detach().cpu(),
            'm_top1_sample': top2_m[:, 0].detach().cpu(),
            'm_gap_sample': (top2_m[:, 0] - top2_m[:, 1]).detach().cpu(),
            'y_curr_sample': batch_label.detach().cpu(),
            'y_prev_sample': (prev_batch.detach().cpu() if prev_batch is not None else torch.full_like(batch_label.detach().cpu(), -1)),
            'flip_mask_sample': ((batch_label != prev_batch).float().detach().cpu() if prev_batch is not None else torch.zeros_like(batch_label, dtype=torch.float32).detach().cpu()),
            'S_pair_sample': route_aux['S'][route_aux['neg_mask']].detach().cpu(),
            'w_pair_sample': route_aux['w_neg'][route_aux['neg_mask']].detach().cpu(),
            's_post_pair_sample': route_aux['s_post'][route_aux['neg_mask']].detach().cpu(),
            'sim_pair_sample': neg_sim.detach().cpu(),
            'rho_fn_pair_sample': route_aux['rho'][route_aux['neg_mask']].float().detach().cpu(),
            'eta_hn_pair_sample': route_aux['eta'][route_aux['neg_mask']].float().detach().cpu(),
            'r_pair_sample': route_aux['r'][route_aux['neg_mask']].detach().cpu(),
            's_stab_pair_sample': route_aux['s_stab'][route_aux['neg_mask']].detach().cpu(),
            'sim_pos_sample': pos_sim.detach().cpu(),
            'sim_neg_sample': neg_sim.detach().cpu(),
            'pairs_sampled': torch.tensor(float(neg_sim.numel())),
            'neg_pairs_available': torch.tensor(float(route_aux['neg_mask'].float().sum().item())),
            'safe_pairs_available': torch.tensor(float((route_aux['neg_mask'] & (~route_aux['rho'])).float().sum().item())),
            'pos_pairs_count': torch.tensor(float(pos_sim.numel())),
            'uncertain_mask_sample': uncertain.detach().cpu(),
            'unsafe_mask_sample': unsafe_mask.detach().cpu(),
            'theta_sample': theta.detach().cpu(),
            'neg_mask_sample': route_aux['neg_mask'].detach().cpu(),
            'tau_u': torch.tensor(tau_u),
            'k_unc': torch.tensor(k_unc),
            'tau_fn_per_anchor': route_aux['tau_fn_per_anchor'].detach().cpu(),
            'tau_hn_per_anchor': route_aux['tau_hn_per_anchor'].detach().cpu(),
            'FN_count_per_anchor': route_aux['FN_count_per_anchor'].detach().cpu(),
            'HN_count_per_anchor': route_aux['HN_count_per_anchor'].detach().cpu(),
            'gate_val': torch.tensor(gate),
            'p_pair_sample': route_aux['p'][route_aux['neg_mask_effective']].detach().cpu() if 'p' in route_aux else torch.zeros(0),
            'h_pair_sample': route_aux['h'][route_aux['neg_mask_effective']].detach().cpu() if 'h' in route_aux else torch.zeros(0),
            'minority_set_size': torch.tensor(route_stats.get('minority_set_size', 0.0)),
            'minority_count_mean': torch.tensor(route_stats.get('minority_count_mean', 0.0)),
            'minority_radius_mean': torch.tensor(route_stats.get('minority_radius_mean', 0.0)),
            'minority_gamma_mean': torch.tensor(route_stats.get('minority_gamma_mean', 0.0)),
            'module_c_sample_fail_rate': torch.tensor(route_stats.get('module_c_sample_fail_rate', 0.0)),
            'module_c_fallback_rate': torch.tensor(route_stats.get('module_c_fallback_rate', 0.0)),
            'L_proto_align': torch.tensor(route_stats.get('L_proto_align', 0.0)),
            'L_conf_repulse': torch.tensor(route_stats.get('L_conf_repulse', 0.0)),
        }

    if module_c_epoch_stats is not None:
        _update_module_c_ema_stats(model, module_c_epoch_stats, ema_rho=module_c_stat_ema_rho)

    if batch_count > 0:
        for k in epoch_meter:
            epoch_meter[k] /= batch_count
        for k in route_meter:
            route_meter[k] /= batch_count

    return {'loss': epoch_meter, 'route': route_meter, 'dump': last_dump, 'gate': gate if batch_count > 0 else 0.0, 'gate_u': gate_u if batch_count > 0 else 0.0, 'gate_fn': gate_fn if batch_count > 0 else 0.0, 'gate_hn': gate_hn if batch_count > 0 else 0.0, 't': t if batch_count > 0 else 0.0, 'warmup_epochs': warmup_epochs, 'cross_warmup_epochs': cross_warmup_epochs}
