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
    # 数值安全：过滤 NaN/Inf，避免 torch.histc 在非有限区间报错
    finite_mask = torch.isfinite(u)
    if finite_mask.any():
        u = u[finite_mask]
    else:
        return 0.5

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


def _sanitize_uncertainty_outputs(u, u_hat, u_aux):
    """数值防护：将不确定度相关张量中的 NaN/Inf 回填到有限值域。"""
    u = torch.nan_to_num(u, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    u_hat = torch.nan_to_num(u_hat, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    u_aux = dict(u_aux)
    u_aux['gamma'] = torch.nan_to_num(u_aux['gamma'], nan=0.0, posinf=20.0, neginf=-20.0)
    u_aux['d_view'] = torch.nan_to_num(u_aux['d_view'], nan=0.0, posinf=10.0, neginf=0.0).clamp(min=0.0)
    u_aux['d_time'] = torch.nan_to_num(u_aux['d_time'], nan=0.0, posinf=10.0, neginf=0.0).clamp(min=0.0)
    u_aux['q_cons'] = torch.nan_to_num(u_aux['q_cons'], nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return u, u_hat, u_aux


def _sanitize_latent_outputs(zs, common_z):
    """数值防护：清理编码器输出，避免 NaN/Inf 进入后续损失与中心更新。"""
    zs_safe = [torch.nan_to_num(z, nan=0.0, posinf=1.0, neginf=-1.0) for z in zs]
    common_z_safe = torch.nan_to_num(common_z, nan=0.0, posinf=1.0, neginf=-1.0)
    return zs_safe, common_z_safe


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


def _build_pairwise_fn_risk(common_z, memberships_cons, u_hat, gamma, d_time,
                            gate_val, route_s0=0.3, route_t_fn=0.5,
                            route_th=0.2, route_hn_temp=0.2, w_min=0.05,
                            neg_mode='batch', knn_k=20,
                            uncertain_mask=None, eps=1e-12):
    """
    模块B（pair层）: 用连续 FN-risk 概率去污染分母，并独立构造 HN 难度质量。
    - r_fn(i,j) 越大, 表示“负对被 FN 污染”的概率越高，w_neg 越小
    - hn_score(i,j) 只用于 HN margin，不用于放大 InfoNCE 分母权重
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

    # (E1) 同簇后验证据 p_ij = <q_i, q_j>
    s_post = torch.mm(memberships_cons, memberships_cons.t()).clamp(0.0, 1.0)

    # (E2) 无记忆稳定证据 + 时间稳定证据（对应模块B冷启动安全门控）
    gamma_stab = torch.sigmoid(gamma)
    time_stab = torch.exp(-d_time.clamp(min=0.0))
    stab_vec = (gamma_stab * time_stab).clamp(0.0, 1.0)
    s_stab = torch.ger(stab_vec, stab_vec)

    # (E3) uncertainty 可靠性门，防止 u_hat 高时过度依赖 pair 证据
    r = (1.0 - 0.5 * (u_hat.unsqueeze(1) + u_hat.unsqueeze(0))).clamp(0.0, 1.0)

    # (E4) neighborhood overlap evidence
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

    # (E5) 连续 FN-risk: logit(p_ij) + 稳定证据 + 邻域证据
    S = r * (_logit(s_post) + _logit(s_stab + eps) + gate_val * _logit(s_nbr + eps))
    r_fn = torch.sigmoid(S / max(route_t_fn, eps)).masked_fill(~neg_mask, 0.0)

    # uncertain-only 路由: 只在 U_t 内启用强去污染
    if uncertain_mask is not None:
        anc = uncertain_mask.float().unsqueeze(1)
        r_fn = r_fn * anc

    # 模块B去污染权重: 仅随 FN-risk 降权（不做 HN 上权）
    w_neg = (1.0 - gate_val * r_fn).clamp(min=w_min, max=1.0)
    w_neg = w_neg.masked_fill(~neg_mask, 0.0)

    # HN 难度质量: 高相似且低 FN-risk，用于后续 margin 惩罚
    sim = F.cosine_similarity(common_z.unsqueeze(1), common_z.unsqueeze(0), dim=2)
    hn_score = torch.sigmoid((sim - route_th) / max(route_hn_temp, eps)) * (1.0 - r_fn)
    hn_score = hn_score.masked_fill(~neg_mask, 0.0)

    fn_mass_i = (r_fn * neg_mask.float()).sum(dim=1) / (neg_mask.float().sum(dim=1) + eps)
    hn_mass_i = (hn_score * neg_mask.float()).sum(dim=1) / (neg_mask.float().sum(dim=1) + eps)
    rho = r_fn > 0.5
    eta = hn_score > 0.5
    fn_count_per_anchor = rho.float().sum(dim=1)
    hn_count_per_anchor = eta.float().sum(dim=1)

    u_center = u_hat - u_hat.mean()
    fn_center = fn_mass_i - fn_mass_i.mean()
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
    denom_fn = (w_neg * exp_sim * r_fn).sum()
    denom_fn_share = (denom_fn / denom_all).item()

    fn_mask = neg_mask & rho
    hn_mask = safe_mask & eta
    fn_count = fn_mask.float().sum().item()
    hn_count = hn_mask.float().sum().item()
    neg_count = neg_mask.float().sum().item()
    safe_neg_count = safe_mask.float().sum().item()

    stats = {
        'fn_ratio': fn_mass_i.mean().item(),
        'safe_ratio': ((safe_mask.float().sum() / (neg_mask.float().sum() + eps))).item(),
        'hn_ratio': hn_mass_i.mean().item(),
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
        'label_flip': (1.0 - stab_vec.mean().item()),
        'stab_rate': stab_vec.mean().item(),
        'denom_fn_share': denom_fn_share,
        'denom_safe_share': 1.0 - denom_fn_share,
        'w_hit_min_ratio': ((w_neg <= (w_min + eps)) & rho).float().mean().item() if rho.any() else 0.0,
        'corr_u_fn_ratio': corr_u_fn.item(),
        'N_size': (neg_mask.float().sum(dim=1).mean().item()),
        'neg_per_anchor': (neg_mask.float().sum(dim=1).mean().item()),
        'U_size': int(uncertain_mask.sum().item()) if uncertain_mask is not None else int(N),
        'fn_pair_share': (fn_count / max(neg_count, 1.0)),
        'hn_pair_share': (hn_count / max(safe_neg_count, 1.0)),
        'r_fn_mean': r_fn[neg_mask].mean().item() if neg_mask.any() else 0.0,
        'w_neg_mean': w_neg[neg_mask].mean().item() if neg_mask.any() else 0.0,
        'w_neg_p90': torch.quantile(w_neg[neg_mask], 0.9).item() if neg_mask.any() else 0.0,
        'fn_mass_i_mean': fn_mass_i.mean().item(),
        'hn_mass_i_mean': hn_mass_i.mean().item(),
        'fn_mass_i_p90': torch.quantile(fn_mass_i, 0.9).item(),
        'hn_mass_i_p90': torch.quantile(hn_mass_i, 0.9).item(),
    }
    aux = {
        'S': S, 's_post': s_post, 'sim': sim, 'rho': rho, 'eta': eta, 'w_neg': w_neg,
        'r_fn': r_fn, 'hn_score': hn_score, 'fn_mass_i': fn_mass_i, 'hn_mass_i': hn_mass_i,
        'r': r, 's_stab': s_stab, 'neg_mask': neg_mask,
        'tau_fn_per_anchor': torch.full((N,), float('nan'), device=device),
        'tau_hn_per_anchor': torch.full((N,), float('nan'), device=device),
        'FN_count_per_anchor': fn_count_per_anchor, 'HN_count_per_anchor': hn_count_per_anchor,
    }
    return w_neg, eta, rho, stats, aux


def _compute_stability_gate(u_aux, gate_s0=0.5, gate_tg=0.2, gate_ema_prev=None, gate_ema_rho=0.9):
    """模块B门控：用 margin 与时间稳定性的并联证据触发，避免仅靠 epoch 线性门控。"""
    gamma = torch.nan_to_num(u_aux['gamma'].detach(), nan=0.0, posinf=20.0, neginf=-20.0)
    d_time = torch.nan_to_num(u_aux['d_time'].detach(), nan=0.0, posinf=10.0, neginf=0.0).clamp(min=0.0)
    stab_i = (torch.sigmoid(gamma) * torch.exp(-d_time)).clamp(0.0, 1.0)
    stab_batch = float(stab_i.mean().item())
    if gate_ema_prev is None:
        gate_ema = stab_batch
    else:
        gate_ema = float(gate_ema_rho) * float(gate_ema_prev) + (1.0 - float(gate_ema_rho)) * stab_batch
    gate_val = 1.0 / (1.0 + np.exp(-(gate_ema - gate_s0) / max(gate_tg, 1e-12)))
    gate_val = float(max(0.0, min(1.0, gate_val)))
    return gate_val, stab_i, gate_ema


def _compute_anchor_route_losses(common_z, q_cons, ema_q_batch, uncertain_mask,
                                 q_centers, route_aux,
                                 bayes_lambda_p=0.7, bayes_lambda_l=1.0,
                                 bayes_delta=0.2, mass_delta=0.05,
                                 hn_margin=0.2):
    """模块B anchor层：FN/HN/neutral 路由 + 定向损失（FN pull / HN margin）。"""
    device = common_z.device
    eps = 1e-12
    N = common_z.size(0)
    fn_loss = torch.tensor(0.0, device=device)
    hn_loss = torch.tensor(0.0, device=device)
    fn_type = torch.zeros(N, dtype=torch.bool, device=device)
    hn_type = torch.zeros(N, dtype=torch.bool, device=device)
    neutral_type = torch.ones(N, dtype=torch.bool, device=device)

    if uncertain_mask.sum() == 0:
        return fn_loss, hn_loss, fn_type, hn_type, neutral_type

    q_tilde = (ema_q_batch.clamp(min=eps) ** bayes_lambda_p) * (q_cons.clamp(min=eps) ** bayes_lambda_l)
    q_tilde = q_tilde / (q_tilde.sum(dim=1, keepdim=True) + eps)
    top2_val, top2_idx = torch.topk(q_tilde, 2, dim=1)
    B = torch.log((top2_val[:, 0] + eps) / (top2_val[:, 1] + eps))

    fn_mass_i = route_aux['fn_mass_i']
    hn_mass_i = route_aux['hn_mass_i']
    low_conf = uncertain_mask & (B < bayes_delta)
    fn_type = low_conf & (fn_mass_i > hn_mass_i + mass_delta)
    hn_type = low_conf & (hn_mass_i > fn_mass_i + mass_delta)
    neutral_type = ~(fn_type | hn_type)

    if fn_type.any():
        k1 = top2_idx[:, 0]
        mu_k1 = q_centers[k1]
        w_fn = (torch.sigmoid(B) * torch.minimum(torch.ones_like(B), B.abs()))
        fn_loss = (w_fn[fn_type] * (1.0 - F.cosine_similarity(common_z[fn_type], mu_k1[fn_type], dim=1))).mean()

    if hn_type.any():
        k1 = top2_idx[:, 0]
        k2 = top2_idx[:, 1]
        mu_k1 = q_centers[k1]
        mu_k2 = q_centers[k2]
        w_hn = (torch.sigmoid(-B) * torch.minimum(torch.ones_like(B), B.abs()))
        sim1 = F.cosine_similarity(common_z[hn_type], mu_k1[hn_type], dim=1)
        sim2 = F.cosine_similarity(common_z[hn_type], mu_k2[hn_type], dim=1)
        hn_loss = (w_hn[hn_type] * torch.relu(hn_margin + sim1 - sim2)).mean()

    return fn_loss, hn_loss, fn_type, hn_type, neutral_type


def contrastive_train(model, mv_data, mvc_loss,
                      batch_size, epoch, W,
                      alpha, beta,
                      optimizer,
                      warmup_epochs,
                      lambda_u,  lambda_hn_penalty,
                      temperature_f, max_epoch=100,
                      initial_top_p=0.3,
                      cross_warmup_epochs=50,
                      w_min=0.05,
                      route_s0=0.3,
                      route_t_fn=0.5,
                      route_hn_temp=0.2,
                      gate_s0=0.5,
                      gate_tg=0.2,
                      gate_ema_rho=0.9,
                      bayes_lambda_p=0.7,
                      bayes_lambda_l=1.0,
                      bayes_delta=0.2,
                      mass_delta=0.05,
                      lambda_fn_pull=0.1,
                      lambda_hn_margin=0.1,
                      hn_margin=0.2,
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
                      enable_theta_certificate=True):
    model.train()
    mv_data_loader, num_views, num_samples, num_clusters = get_multiview_data(mv_data, batch_size)

    # 将 all_features 和 all_labels 初始化为 Python 列表
    all_features = []  # 用于收集每个批次的特征
    all_labels = []  # 用于收集每个批次的标签

    if not hasattr(model, 'tau_u_ema'):
        model.tau_u_ema = None
    if not hasattr(model, 'gate_ema'):
        model.gate_ema = None
    if not hasattr(model, 'gate_ema'):
        model.gate_ema = None

    # E 步：更新全量伪标签
    psedo_labeling(model, mv_data, batch_size)

    criterion = torch.nn.MSELoss()  # 添加重建损失的损失函数

    epoch_meter = {'L_total':0.0,'L_recon':0.0,'L_feat':0.0,'L_cross':0.0,'L_cluster':0.0,'L_uncert':0.0,'L_hn':0.0,'L_reg':0.0}
    route_meter = {'fn_ratio':0.0,'safe_ratio':0.0,'hn_ratio':0.0,'mean_s_post_fn':0.0,'mean_s_post_non_fn':0.0,'delta_post':0.0,'mean_sim_hn':0.0,'mean_sim_safe_non_hn':0.0,'delta_sim':0.0,'label_flip':0.0,'stab_rate':0.0,'denom_fn_share':0.0,'denom_safe_share':0.0,'w_hit_min_ratio':0.0,'corr_u_fn_ratio':0.0,'N_size':0.0,'U_size':0.0,'tau_u':0.0,'unsafe_ratio':0.0,'theta_p50_batch':0.0,'theta_p90_batch':0.0,'k_unc':0.0,
                   'r_fn_mean':0.0,'w_neg_mean':0.0,'w_neg_p90':0.0,'fn_mass_i_mean':0.0,'hn_mass_i_mean':0.0,'fn_mass_i_p90':0.0,'hn_mass_i_p90':0.0,
                   'fn_type_ratio':0.0,'hn_type_ratio':0.0,'neutral_ratio':0.0,'gate_stab':0.0,'L_fn_pull':0.0,'L_hn_margin':0.0}
    batch_count = 0
    last_dump = {}

    for batch_idx, (sub_data_views, _, sample_idx) in enumerate(mv_data_loader):
        # ——— 1) 伪标签 & 同/异样本矩阵 ———
        batch_psedo_label = model.psedo_labels[sample_idx]                # [N]
        y_matrix = (batch_psedo_label.unsqueeze(1) == batch_psedo_label.unsqueeze(0)).int()

        # ——— 2) 编码 + 融合 ———
        xrs, zs = model(sub_data_views)
        common_z = model.fusion(zs)
        zs, common_z = _sanitize_latent_outputs(zs, common_z)

        # 现在有了 common_z，确定 device
        device = common_z.device

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
        u, u_hat, u_aux = _sanitize_uncertainty_outputs(u, u_hat, u_aux)
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
        theta = torch.nan_to_num(theta, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
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

        # ——— 5) 模块B稳定性门控：由稳定证据触发，而非 epoch 线性 ——
        gate_val, stab_i, model.gate_ema = _compute_stability_gate(
            u_aux,
            gate_s0=gate_s0,
            gate_tg=gate_tg,
            gate_ema_prev=model.gate_ema,
            gate_ema_rho=gate_ema_rho,
        )
        u_mean = u_hat.mean().item()
        mu_start, mu_end = 0.3, 0.7
        raw_gate = (u_mean - mu_start) / (mu_end - mu_start)
        gate_u = float(max(0.0, min(1.0, raw_gate)))
        t = float((epoch - 1) / max(1, max_epoch - 1))
        gate_fn = gate_val
        gate_hn = gate_val
        gate = torch.tensor(gate_val, device=device)

        # ——— 6) 计算共识中心 q_centers ———
        q_centers = model.compute_centers(common_z, batch_psedo_label)

        # ——— 7) Design 1': pair-wise FN 风险路由（停用原 FN/HN MLP 路径）———
        route_mask = uncertain_mask if route_uncertain_only else None
        w_neg, eta_mat, rho_mat, route_stats, route_aux = _build_pairwise_fn_risk(
            common_z=common_z,
            memberships_cons=memberships[num_views],
            u_hat=u_hat,
            gamma=u_aux['gamma'],
            d_time=u_aux['d_time'],
            gate_val=gate_val,
            route_s0=route_s0,
            route_t_fn=route_t_fn,
            route_th=route_s0,
            route_hn_temp=route_hn_temp,
            w_min=w_min,
            neg_mode=neg_mode,
            knn_k=knn_neg_k,
            uncertain_mask=route_mask,
        )

        # ——— 8) 累加各项损失 ———
        loss_list = []
        Lcl = Lfeat = Lu = Lpen = Lcross = Lrecon = Lfnpull = Lhnmargin = 0.0

        idx = sample_idx.to(device).long()
        ema_q_batch = model.ema_q[idx]
        fn_pull_loss, hn_margin_loss, fn_type, hn_type, neutral_type = _compute_anchor_route_losses(
            common_z=common_z,
            q_cons=memberships[num_views],
            ema_q_batch=ema_q_batch,
            uncertain_mask=uncertain_mask,
            q_centers=q_centers,
            route_aux=route_aux,
            bayes_lambda_p=bayes_lambda_p,
            bayes_lambda_l=bayes_lambda_l,
            bayes_delta=bayes_delta,
            mass_delta=mass_delta,
            hn_margin=hn_margin,
        )
        Lfn = gate_fn * lambda_fn_pull * fn_pull_loss
        Lhnm = gate_hn * lambda_hn_margin * hn_margin_loss

        for v in range(num_views):
            # 准备 Wv 和 y_pse
            Wv = W[v][sample_idx][:, sample_idx].to(device)
            y_pse = y_matrix.float().to(device)

            # a) 簇级 InfoNCE
            k_centers = model.compute_centers(zs[v], batch_psedo_label)
            if epoch <= 50:
                cl = mvc_loss.compute_cluster_loss(q_centers, k_centers, batch_psedo_label)
            else:
                mask = torch.ones(mvc_loss.num_clusters, dtype=torch.bool, device=device)
                cl, _, _ = mvc_loss.compute_cluster_loss(
                    q_centers, k_centers, batch_psedo_label,
                    features_batch=common_z,
                    global_minority_mask=mask,
                    return_mmd_excl=True
                )
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

            # d) 模块B anchor 定向损失（FN pull + HN margin）
            Lpen_i = Lfn + Lhnm

            Lpen += Lpen_i.item()
            Lfnpull += Lfn.item()
            Lhnmargin += Lhnm.item()
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
        if not torch.isfinite(total_loss):
            print(f"[WARN] Skip batch {batch_idx} at epoch {epoch} due to non-finite total_loss={float(total_loss)}")
            continue
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
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
        neg_sim = sim_mat[route_aux['neg_mask']]
        route_stats['U_ratio'] = float(k_unc) / max(batch_N, 1)
        route_stats['u_thr'] = u_thr
        route_stats['tau_u'] = tau_u
        route_stats['unsafe_ratio'] = float(unsafe_mask.float().mean().item())
        route_stats['theta_p50_batch'] = float(torch.quantile(theta.detach(), 0.5).item())
        route_stats['theta_p90_batch'] = float(torch.quantile(theta.detach(), 0.9).item())
        route_stats['k_unc'] = k_unc
        route_stats['fn_type_ratio'] = float(fn_type.float().mean().item())
        route_stats['hn_type_ratio'] = float(hn_type.float().mean().item())
        route_stats['neutral_ratio'] = float(neutral_type.float().mean().item())
        route_stats['gate_stab'] = float(stab_i.mean().item())
        route_stats['L_fn_pull'] = Lfn.item()
        route_stats['L_hn_margin'] = Lhnm.item()
        route_stats['fn_type_ratio'] = float(fn_type.float().mean().item())
        route_stats['hn_type_ratio'] = float(hn_type.float().mean().item())
        route_stats['neutral_ratio'] = float(neutral_type.float().mean().item())
        route_stats['gate_stab'] = float(stab_i.mean().item())
        route_stats['L_fn_pull'] = Lfnpull
        route_stats['L_hn_margin'] = Lhnmargin

        last_dump = {
            'u_sample': u.detach().cpu(),
            'gamma_sample': u_aux['gamma'].detach().cpu(),
            'd_view_sample': u_aux['d_view'].detach().cpu(),
            'd_time_sample': u_aux['d_time'].detach().cpu(),
            'm_top1_sample': top2_m[:, 0].detach().cpu(),
            'm_gap_sample': (top2_m[:, 0] - top2_m[:, 1]).detach().cpu(),
            'y_curr_sample': batch_psedo_label.detach().cpu(),
            'y_prev_sample': torch.full_like(batch_psedo_label.detach().cpu(), -1),
            'flip_mask_sample': torch.zeros_like(batch_psedo_label, dtype=torch.float32).detach().cpu(),
            'S_pair_sample': route_aux['S'][route_aux['neg_mask']].detach().cpu(),
            'w_pair_sample': route_aux['w_neg'][route_aux['neg_mask']].detach().cpu(),
            's_post_pair_sample': route_aux['s_post'][route_aux['neg_mask']].detach().cpu(),
            'sim_pair_sample': neg_sim.detach().cpu(),
            'rho_fn_pair_sample': route_aux['rho'][route_aux['neg_mask']].float().detach().cpu(),
            'eta_hn_pair_sample': route_aux['eta'][route_aux['neg_mask']].float().detach().cpu(),
            'r_pair_sample': route_aux['r'][route_aux['neg_mask']].detach().cpu(),
            's_stab_pair_sample': route_aux['s_stab'][route_aux['neg_mask']].detach().cpu(),
            'r_fn_pair_sample': route_aux['r_fn'][route_aux['neg_mask']].detach().cpu(),
            'hn_score_pair_sample': route_aux['hn_score'][route_aux['neg_mask']].detach().cpu(),
            'fn_mass_i_sample': route_aux['fn_mass_i'].detach().cpu(),
            'hn_mass_i_sample': route_aux['hn_mass_i'].detach().cpu(),
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
            'gate_stab': torch.tensor(float(stab_i.mean().item())),
        }

        print(f"[Epoch {epoch} Batch {batch_idx}] "
              f"Total={total_loss.item():.4f}  "
              f"FN_ratio={route_stats['fn_ratio']:.3f} HN_ratio={route_stats['hn_ratio']:.3f} "
              f"U_ratio={route_stats['U_ratio']:.3f} stab={route_stats['stab_rate']:.3f}")

    # ===== 训练循环结束 =====
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
                                   w_min=0.05,
                                   route_s0=0.3,
                                   route_t_fn=0.5,
                                   route_hn_temp=0.2,
                                   gate_s0=0.5,
                                   gate_tg=0.2,
                                   gate_ema_rho=0.9,
                                   bayes_lambda_p=0.7,
                                   bayes_lambda_l=1.0,
                                   bayes_delta=0.2,
                                   mass_delta=0.05,
                                   lambda_fn_pull=0.1,
                                   lambda_hn_margin=0.1,
                                   hn_margin=0.2,
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
                                   enable_theta_certificate=True):
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
    route_meter = {'fn_ratio':0.0,'safe_ratio':0.0,'hn_ratio':0.0,'mean_s_post_fn':0.0,'mean_s_post_non_fn':0.0,'delta_post':0.0,'mean_sim_hn':0.0,'mean_sim_safe_non_hn':0.0,'delta_sim':0.0,'label_flip':0.0,'stab_rate':0.0,'denom_fn_share':0.0,'denom_safe_share':0.0,'w_hit_min_ratio':0.0,'corr_u_fn_ratio':0.0,'N_size':0.0,'U_size':0.0,'tau_u':0.0,'unsafe_ratio':0.0,'theta_p50_batch':0.0,'theta_p90_batch':0.0,'k_unc':0.0,'r_fn_mean':0.0,'w_neg_mean':0.0,'w_neg_p90':0.0,'fn_mass_i_mean':0.0,'hn_mass_i_mean':0.0,'fn_mass_i_p90':0.0,'hn_mass_i_p90':0.0,'fn_type_ratio':0.0,'hn_type_ratio':0.0,'neutral_ratio':0.0,'gate_stab':0.0,'L_fn_pull':0.0,'L_hn_margin':0.0}
    batch_count = 0
    last_dump = {}

    if not hasattr(model, 'tau_u_ema'):
        model.tau_u_ema = None

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
        zs, common_z = _sanitize_latent_outputs(zs, common_z)

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
        u, u_hat, u_aux = _sanitize_uncertainty_outputs(u, u_hat, u_aux)
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
        theta = torch.nan_to_num(theta, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
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

        # ——— 模块B稳定性门控：由稳定证据触发 ——
        gate, stab_i, model.gate_ema = _compute_stability_gate(
            u_aux,
            gate_s0=gate_s0,
            gate_tg=gate_tg,
            gate_ema_prev=model.gate_ema,
            gate_ema_rho=gate_ema_rho,
        )
        u_mean = u_hat.mean().item()
        mu_lo, mu_hi = 0.3, 0.7
        gate_u = float((u_mean - mu_lo) / (mu_hi - mu_lo))
        gate_u = max(0.0, min(1.0, gate_u))
        t = float((epoch - 1) / max(1, max_epoch - 1))
        gate_fn = gate
        gate_hn = gate
        gate_t = torch.tensor(gate, device=device)

        # ——— 共识中心 ———
        q_centers = model.compute_centers(common_z, batch_label)

        # ——— Design 1': pair-wise FN 风险路由（停用原 FN/HN MLP 路径）———
        route_mask = uncertain if route_uncertain_only else None
        w_neg, eta_mat, rho_mat, route_stats, route_aux = _build_pairwise_fn_risk(
            common_z=common_z,
            memberships_cons=memberships[num_views],
            u_hat=u_hat,
            gamma=u_aux['gamma'],
            d_time=u_aux['d_time'],
            gate_val=gate,
            route_s0=route_s0,
            route_t_fn=route_t_fn,
            route_th=route_s0,
            route_hn_temp=route_hn_temp,
            w_min=w_min,
            neg_mode=neg_mode,
            knn_k=knn_neg_k,
            uncertain_mask=route_mask,
        )

        idx = sample_idx.to(device).long()
        ema_q_batch = model.ema_q[idx]
        fn_pull_loss, hn_margin_loss, fn_type, hn_type, neutral_type = _compute_anchor_route_losses(
            common_z=common_z,
            q_cons=memberships[num_views],
            ema_q_batch=ema_q_batch,
            uncertain_mask=uncertain,
            q_centers=q_centers,
            route_aux=route_aux,
            bayes_lambda_p=bayes_lambda_p,
            bayes_lambda_l=bayes_lambda_l,
            bayes_delta=bayes_delta,
            mass_delta=mass_delta,
            hn_margin=hn_margin,
        )
        Lfn = gate_fn * lambda_fn_pull * fn_pull_loss
        Lhnm = gate_hn * lambda_hn_margin * hn_margin_loss

        # ——— 构造并累加各视图的损失 ———
        batch_loss = 0.0
        for v in range(num_views):
            # 动态 k-NN Graph
            Wv = get_knn_graph(sub_views[v], k).to(device)
            y_pse = y_matrix.float()

            # a) 簇级 InfoNCE
            kv_centers = model.compute_centers(zs[v], batch_label)
            if epoch <= 50:
                cl = mvc_loss.compute_cluster_loss(q_centers, kv_centers, batch_label)
            else:
                mask = torch.ones(mvc_loss.num_clusters, dtype=torch.bool, device=device)
                cl, _, _ = mvc_loss.compute_cluster_loss(
                    q_centers, kv_centers, batch_label,
                    features_batch=common_z,
                    global_minority_mask=mask,
                    return_mmd_excl=True
                )
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

            # d) 模块B anchor 定向损失（FN pull + HN margin）
            Lpen = Lfn + Lhnm
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
        if not torch.isfinite(batch_loss):
            print(f"[WARN] Skip batch {batch_idx} at epoch {epoch} due to non-finite batch_loss={float(batch_loss)}")
            continue
        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
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
        neg_sim = sim_mat[route_aux['neg_mask']]
        route_stats['U_ratio'] = float(k_unc) / max(B, 1)
        route_stats['u_thr'] = u_thr
        route_stats['tau_u'] = tau_u
        route_stats['unsafe_ratio'] = float(unsafe_mask.float().mean().item())
        route_stats['theta_p50_batch'] = float(torch.quantile(theta.detach(), 0.5).item())
        route_stats['theta_p90_batch'] = float(torch.quantile(theta.detach(), 0.9).item())
        route_stats['k_unc'] = k_unc

        last_dump = {
            'u_sample': u.detach().cpu(),
            'gamma_sample': u_aux['gamma'].detach().cpu(),
            'd_view_sample': u_aux['d_view'].detach().cpu(),
            'd_time_sample': u_aux['d_time'].detach().cpu(),
            'm_top1_sample': top2_m[:, 0].detach().cpu(),
            'm_gap_sample': (top2_m[:, 0] - top2_m[:, 1]).detach().cpu(),
            'y_curr_sample': batch_label.detach().cpu(),
            'y_prev_sample': torch.full_like(batch_label.detach().cpu(), -1),
            'flip_mask_sample': torch.zeros_like(batch_label, dtype=torch.float32).detach().cpu(),
            'S_pair_sample': route_aux['S'][route_aux['neg_mask']].detach().cpu(),
            'w_pair_sample': route_aux['w_neg'][route_aux['neg_mask']].detach().cpu(),
            's_post_pair_sample': route_aux['s_post'][route_aux['neg_mask']].detach().cpu(),
            'sim_pair_sample': neg_sim.detach().cpu(),
            'rho_fn_pair_sample': route_aux['rho'][route_aux['neg_mask']].float().detach().cpu(),
            'eta_hn_pair_sample': route_aux['eta'][route_aux['neg_mask']].float().detach().cpu(),
            'r_pair_sample': route_aux['r'][route_aux['neg_mask']].detach().cpu(),
            's_stab_pair_sample': route_aux['s_stab'][route_aux['neg_mask']].detach().cpu(),
            'r_fn_pair_sample': route_aux['r_fn'][route_aux['neg_mask']].detach().cpu(),
            'hn_score_pair_sample': route_aux['hn_score'][route_aux['neg_mask']].detach().cpu(),
            'fn_mass_i_sample': route_aux['fn_mass_i'].detach().cpu(),
            'hn_mass_i_sample': route_aux['hn_mass_i'].detach().cpu(),
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
            'gate_stab': torch.tensor(float(stab_i.mean().item())),
        }

    if batch_count > 0:
        for k in epoch_meter:
            epoch_meter[k] /= batch_count
        for k in route_meter:
            route_meter[k] /= batch_count

    return {'loss': epoch_meter, 'route': route_meter, 'dump': last_dump, 'gate': gate if batch_count > 0 else 0.0, 'gate_u': gate_u if batch_count > 0 else 0.0, 'gate_fn': gate_fn if batch_count > 0 else 0.0, 'gate_hn': gate_hn if batch_count > 0 else 0.0, 't': t if batch_count > 0 else 0.0, 'warmup_epochs': warmup_epochs, 'cross_warmup_epochs': cross_warmup_epochs}
