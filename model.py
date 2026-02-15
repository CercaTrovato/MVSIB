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
    }
    return w_neg, eta, rho, stats, aux


def _safe_sigmoid(x):
    return torch.sigmoid(x.clamp(min=-30.0, max=30.0))


def _compute_denom_fn_share(neg_terms, r_fn, neg_mask, eps=1e-12):
    # 与 loss 分母完全同源的 FN 污染占比统计，避免日志与优化目标脱节。
    numer = (neg_terms * r_fn * neg_mask.float()).sum(dim=1)
    denom = (neg_terms * neg_mask.float()).sum(dim=1) + eps
    return numer / denom


def _compute_sce_outputs(model, zs, common_z, q_ema_prev, lambda_vote=0.5, tau_p=0.2,
                         knn_k=20, kappa=10.0, delta=0.6,
                         use_view_divergence=True, lambda_vote_eff=0.0,
                         vote_eps=0.05, alpha_u=12.0, beta_u=10.0,
                         m0=0.35, a0=0.55, eps=1e-12):
    # UCM：不确定度由边界间隔与跨视图一致性决定，避免伪后验自证循环。
    device = common_z.device
    B = common_z.size(0)
    V = model.num_views
    K = model.num_clusters
    z_norm = F.normalize(common_z, dim=1)
    alpha = model.get_weights().to(device)

    k_eff = min(knn_k + 1, B)
    dist = torch.cdist(z_norm, z_norm, p=2)
    knn_idx = torch.topk(-dist, k_eff, dim=1).indices[:, 1:] if k_eff > 1 else torch.zeros(B, 0, dtype=torch.long, device=device)
    vote = q_ema_prev[knn_idx].mean(dim=1) if knn_idx.numel() > 0 else q_ema_prev
    vote = (1.0 - vote_eps) * vote + vote_eps * (1.0 / K)
    vote = vote.clamp(min=eps, max=1.0)

    if lambda_vote_eff <= 0.0:
        lambda_vote_eff = lambda_vote

    logits_proto = torch.zeros(B, K, device=device)
    q_views = []
    for v in range(V):
        zv = F.normalize(zs[v], dim=1)
        centers_v = F.normalize(model.centers[v], dim=1)
        logits_v = (zv @ centers_v.t()) / tau_p
        logits_proto = logits_proto + alpha[v] * logits_v
        q_views.append(torch.softmax(logits_v, dim=1))

    q_proto = torch.softmax(logits_proto, dim=1)
    logits = logits_proto + lambda_vote_eff * torch.log(vote)
    q = torch.softmax(logits, dim=1)

    top2 = torch.topk(q_proto, 2, dim=1).values
    margin_gap = (top2[:, 0] - top2[:, 1]).clamp(0.0, 1.0)

    if use_view_divergence and V > 1:
        kl = []
        for qv in q_views:
            kl.append((qv * (torch.log(qv + eps) - torch.log(q_proto + eps))).sum(dim=1))
        kl_mean = torch.stack(kl, dim=1).mean(dim=1)
        agreement = torch.exp(-kl_mean).clamp(0.0, 1.0)
    else:
        agreement = torch.ones(B, device=device)

    u_margin = torch.sigmoid(alpha_u * (m0 - margin_gap))
    u_agree = torch.sigmoid(beta_u * (a0 - agreement))
    u = (u_margin * u_agree).clamp(0.0, 1.0)

    t = torch.sigmoid(kappa * (agreement - delta))
    L_cal = ((1.0 - u - t) ** 2).mean()
    return q, q_proto, u, vote, z_norm, L_cal, margin_gap, agreement


def _compute_csd_losses(model, z, q_proto, u, margin_gap, theta_u,
                        neg_mode='batch', knn_neg_k=20, tau=0.2,
                        sh=0.7, hn_margin=0.2, gamma_h=2.0, w_max=1.0,
                        gamma_gate=1.0, share_target=0.08, share_lambda=0.0,
                        share_lambda_lr=0.05, fn_ratio_cut=0.5, hn_ratio_cut=0.5,
                        top_m=5, uncertain_mask=None,
                        pi_high_sim=0.6, pi_high_same=0.5, pi_low_same=0.2,
                        eps=1e-12):
    # LNM+HPM：FN 做分母去污染概率建模，HN 用 bounded focal+margin push。
    device = z.device
    B = z.size(0)
    sim = z @ z.t()
    p_same = (q_proto @ q_proto.t()).clamp(0.0, 1.0)
    eye = torch.eye(B, dtype=torch.bool, device=device)
    neg_mask = ~eye
    if neg_mode == 'knn':
        k_eff = min(knn_neg_k + 1, B)
        knn_idx = torch.topk(sim, k_eff, dim=1).indices
        knn_mask = torch.zeros_like(neg_mask)
        row = torch.arange(B, device=device).unsqueeze(1).expand_as(knn_idx)
        knn_mask[row, knn_idx] = True
        neg_mask = knn_mask & (~eye)

    u_i = u.unsqueeze(1).expand(B, B)
    u_j = u.unsqueeze(0).expand(B, B)
    gap_diff = torch.abs(margin_gap.unsqueeze(1) - margin_gap.unsqueeze(0))
    pair_feat = torch.stack([sim, p_same, u_i, u_j, gap_diff], dim=-1)
    pi_logits = model.pi_net(pair_feat).squeeze(-1)
    pi_prob = torch.sigmoid(pi_logits) * neg_mask.float()

    gate = (u.clamp(0.0, 1.0) ** gamma_gate).detach().unsqueeze(1)
    if uncertain_mask is not None:
        gate = gate * uncertain_mask.float().unsqueeze(1)
    pi_eff = (pi_prob * gate).clamp(0.0, 1.0)

    w_neg = torch.clamp(1.0 - pi_eff, min=0.05, max=1.0)
    neg_terms = w_neg * torch.exp(sim / tau) * neg_mask.float()

    pos_mask = eye.clone()
    topm = min(max(top_m + 1, 2), B)
    top_idx = torch.topk(p_same.masked_fill(eye, -1.0), topm, dim=1).indices[:, 1:]
    row = torch.arange(B, device=device).unsqueeze(1).expand_as(top_idx)
    pos_mask[row, top_idx] = True
    num = (torch.exp(sim / tau) * pos_mask.float()).sum(dim=1)
    den = num + neg_terms.sum(dim=1) + eps
    L_nce = (-torch.log((num + eps) / den)).mean()

    share_fn = _compute_denom_fn_share(neg_terms, pi_eff, neg_mask, eps=eps)
    share_fn_batch = share_fn.mean()
    L_share = share_lambda * (share_fn_batch - share_target)
    share_lambda_new = max(0.0, share_lambda + share_lambda_lr * (share_fn_batch.detach().item() - share_target))

    # HPM：异簇高相似且不确定 anchor 的难度强度权重。
    h_ij = _safe_sigmoid(12.0 * (sim - sh)) * _safe_sigmoid(10.0 * (u_i - theta_u))
    w_hn = torch.clamp((h_ij ** gamma_h) * neg_mask.float(), min=0.0, max=w_max)
    s_i_plus = (sim * pos_mask.float()).sum(dim=1) / (pos_mask.float().sum(dim=1) + eps)
    hn_margin_term = F.relu(hn_margin + sim - s_i_plus.unsqueeze(1)) * neg_mask.float()
    L_hn = (w_hn * hn_margin_term).sum(dim=1).mean()

    # L_pi：高风险 pair 的 FN 概率应高于低风险 pair。
    high_mask = neg_mask & (sim > pi_high_sim) & (p_same > pi_high_same)
    low_mask = neg_mask & (p_same < pi_low_same)
    high_vals = pi_prob[high_mask]
    low_vals = pi_prob[low_mask]
    if high_vals.numel() > 0 and low_vals.numel() > 0:
        n = min(high_vals.numel(), low_vals.numel(), 256)
        high_vals = high_vals[:n]
        low_vals = low_vals[:n]
        L_pi = F.relu(1.0 - (high_vals - low_vals)).mean()
    else:
        L_pi = torch.tensor(0.0, device=device)

    safe_mask = neg_mask & (pi_eff <= fn_ratio_cut)
    fn_mask = neg_mask & (pi_eff > fn_ratio_cut)
    hn_mask = neg_mask & (w_hn > hn_ratio_cut)
    non_hn_safe = safe_mask & (~hn_mask)
    stats = {
        'fn_ratio': fn_mask.float().mean().item(),
        'hn_ratio': hn_mask.float().mean().item(),
        'FN_count': fn_mask.float().sum().item(),
        'HN_count': hn_mask.float().sum().item(),
        'neg_count': neg_mask.float().sum().item(),
        'safe_neg_count': safe_mask.float().sum().item(),
        'safe_ratio': safe_mask.float().mean().item(),
        'mean_s_post_fn': p_same[fn_mask].mean().item() if fn_mask.any() else 0.0,
        'mean_s_post_non_fn': p_same[safe_mask].mean().item() if safe_mask.any() else 0.0,
        'mean_sim_hn': sim[hn_mask].mean().item() if hn_mask.any() else 0.0,
        'mean_sim_safe_non_hn': sim[non_hn_safe].mean().item() if non_hn_safe.any() else 0.0,
        'denom_fn_share': share_fn_batch.item(),
        'denom_safe_share': 1.0 - share_fn_batch.item(),
        'w_mean_on_FN': w_neg[fn_mask].mean().item() if fn_mask.any() else 0.0,
        'w_mean_on_safe': w_neg[safe_mask].mean().item() if safe_mask.any() else 0.0,
        'w_hit_min_ratio': ((w_neg <= (0.05 + eps)) & fn_mask).float().mean().item() if fn_mask.any() else 0.0,
        'delta_post': (p_same[fn_mask].mean() - p_same[safe_mask].mean()).item() if fn_mask.any() and safe_mask.any() else 0.0,
        'delta_sim': (sim[hn_mask].mean() - sim[non_hn_safe].mean()).item() if hn_mask.any() and non_hn_safe.any() else 0.0,
        'N_size': neg_mask.float().sum(dim=1).mean().item(),
        'neg_per_anchor': neg_mask.float().sum(dim=1).mean().item(),
        'fn_pair_share': fn_mask.float().sum().item() / max(neg_mask.float().sum().item(), 1.0),
        'hn_pair_share': hn_mask.float().sum().item() / max(safe_mask.float().sum().item(), 1.0),
        'neg_used_in_loss_size': int(neg_mask.float().sum().item()),
        'pi_mean': pi_prob[neg_mask].mean().item() if neg_mask.any() else 0.0,
        'pi_p90': torch.quantile(pi_prob[neg_mask], 0.9).item() if neg_mask.any() else 0.0,
        'w_hn_mean': w_hn[neg_mask].mean().item() if neg_mask.any() else 0.0,
    }
    aux = {'sim': sim, 'r_fn': pi_eff, 'r_hn': w_hn, 'w_neg': w_neg, 'neg_mask': neg_mask,
           'pi_prob': pi_prob, 'w_hn': w_hn, 'p_same': p_same}
    return L_nce, L_hn, L_share, L_pi, share_lambda_new, stats, aux


def _ism_star_losses(z, q, centers, cluster_size_ema, rho_n=0.5, p_per_cluster=8,
                     p_radius=0.9, sigma=0.01, t_proj=3, margin_c=0.05,
                     rep_xi=0.9, tau_excl=0.2, min_points=2, eps=1e-12):
    # ISM*：只在少数簇做严格边界约束的伪样本增强。
    device = z.device
    K = q.size(1)
    hard = torch.argmax(q, dim=1)
    minority = cluster_size_ema < (rho_n * cluster_size_ema.mean())
    L_mmd = torch.tensor(0.0, device=device)
    L_rep = torch.tensor(0.0, device=device)
    L_excl = torch.tensor(0.0, device=device)

    for c in range(K):
        if not bool(minority[c]):
            continue
        xc = z[hard == c]
        if xc.size(0) < min_points:
            continue
        mu = xc.mean(dim=0)
        dist = torch.norm(xc - mu.unsqueeze(0), dim=1)
        Rc = torch.quantile(dist, p_radius)
        pseudos = []
        for _ in range(p_per_cluster):
            idx = torch.randperm(xc.size(0), device=device)[:2]
            xa, xb = xc[idx[0]], xc[idx[1]]
            alpha = torch.rand(1, device=device)
            psi = mu + alpha * (xa - mu) + (1.0 - alpha) * (xb - mu) + sigma * torch.randn_like(mu)
            rad = torch.norm(psi - mu)
            if rad > Rc:
                psi = mu + Rc * (psi - mu) / (rad + eps)
            psi = F.normalize(psi.unsqueeze(0), dim=1).squeeze(0)

            for _ in range(t_proj):
                logits = centers @ psi
                logits[c] = -1e9
                kstar = int(torch.argmax(logits).item())
                gap = torch.dot(psi, centers[c]) - torch.dot(psi, centers[kstar]) - margin_c
                if gap >= 0:
                    break
                psi = psi + 0.1 * (centers[c] - centers[kstar])
                psi = F.normalize(psi.unsqueeze(0), dim=1).squeeze(0)
            pseudos.append(psi)

        psi_c = torch.stack(pseudos, dim=0)
        xx = torch.exp(-torch.cdist(xc, xc, p=2) ** 2 / (2 * (sigma + eps) ** 2)).mean()
        yy = torch.exp(-torch.cdist(psi_c, psi_c, p=2) ** 2 / (2 * (sigma + eps) ** 2)).mean()
        xy = torch.exp(-torch.cdist(xc, psi_c, p=2) ** 2 / (2 * (sigma + eps) ** 2)).mean()
        L_mmd = L_mmd + xx + yy - 2.0 * xy

        sim_pp = psi_c @ psi_c.t()
        mask = ~torch.eye(sim_pp.size(0), dtype=torch.bool, device=device)
        if mask.any():
            L_rep = L_rep + F.relu(sim_pp[mask] - rep_xi).mean()

        excl_logits = (psi_c @ centers.t()) / tau_excl
        excl_logits[:, c] = -1e9
        L_excl = L_excl + torch.logsumexp(excl_logits, dim=1).mean()

    return L_mmd, L_rep, L_excl


def contrastive_train(model, mv_data, mvc_loss,
                      batch_size, epoch, W,
                      alpha, beta,
                      optimizer,
                      warmup_epochs,
                      lambda_u, lambda_hn_penalty,
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
                      enable_star_modules=True,
                      tau_p=0.2,
                      lambda_vote=0.5,
                      beta_d=0.2,
                      beta_c=0.3,
                      cal_kappa=10.0,
                      cal_delta=0.6,
                      use_view_divergence=True,
                      share_target=0.08,
                      share_lambda_lr=0.05,
                      fn_s0=0.6,
                      fn_p0=0.5,
                      fn_ts=0.05,
                      fn_tp=0.1,
                      hn_sh=0.7,
                      hn_ph=0.6,
                      hn_eta=0.2,
                      hn_th=0.05,
                      hn_tb=0.05,
                      hn_margin=0.2,
                      gamma_gate=1.0,
                      fn_ratio_cut=0.5,
                      hn_ratio_cut=0.5,
                      u_threshold=0.5,
                      ism_rho_n=0.5,
                      ism_p_per_cluster=8,
                      ism_p_radius=0.9,
                      ism_sigma=0.01,
                      ism_t_proj=3,
                      ism_margin=0.05,
                      lambda_cal=0.1,
                      lambda_share=1.0,
                      lambda_ism=0.2,
                      lambda_rep=0.1,
                      lambda_excl=0.1,
                      vote_warmup_epochs=8,
                      sce_vote_eps=0.05,
                      debug_star_epochs=5,
                      ucm_alpha_u=12.0,
                      ucm_beta_u=10.0,
                      ucm_m0=0.35,
                      ucm_a0=0.55,
                      theta_u_momentum=0.1,
                      theta_u_quantile=0.7,
                      lambda_pi=0.1):
    model.train()
    mv_data_loader, num_views, num_samples, num_clusters = get_multiview_data(mv_data, batch_size)
    psedo_labeling(model, mv_data, batch_size)

    criterion = torch.nn.MSELoss()
    if not hasattr(model, 'q_ema_global') or model.q_ema_global.size(0) != num_samples:
        # SCE* 冷启动使用均匀后验，避免 one-hot 票据导致伪确定。
        model.q_ema_global = torch.full((num_samples, model.num_clusters), 1.0 / model.num_clusters, device=model.device)
    if not hasattr(model, 'cluster_size_ema') or model.cluster_size_ema.size(0) != model.num_clusters:
        model.cluster_size_ema = torch.ones(model.num_clusters, device=model.device)
    if not hasattr(model, 'share_lambda_state'):
        model.share_lambda_state = 0.0
    if not hasattr(model, 'theta_u_state'):
        model.theta_u_state = float(u_threshold)

    epoch_meter = {'L_total':0.0,'L_recon':0.0,'L_feat':0.0,'L_cross':0.0,'L_cluster':0.0,'L_uncert':0.0,'L_hn':0.0,'L_reg':0.0}
    route_meter = {
        'fn_ratio':0.0,'safe_ratio':0.0,'hn_ratio':0.0,
        'mean_s_post_fn':0.0,'mean_s_post_non_fn':0.0,'delta_post':0.0,
        'mean_sim_hn':0.0,'mean_sim_safe_non_hn':0.0,'delta_sim':0.0,
        'label_flip':0.0,'stab_rate':0.0,
        'denom_fn_share':0.0,'denom_safe_share':0.0,
        'w_hit_min_ratio':0.0,'corr_u_fn_ratio':0.0,
        'N_size':0.0,'U_size':0.0,
        'neg_used_in_loss_size':0.0,
        # ROUTE/DISTR 里会读取的计数字段，必须显式累计。
        'neg_count':0.0,'safe_neg_count':0.0,'FN_count':0.0,'HN_count':0.0,
        'w_mean_on_FN':0.0,'w_mean_on_safe':0.0,
        'neg_per_anchor':0.0,'fn_pair_share':0.0,'hn_pair_share':0.0,
        'pi_mean':0.0,'pi_p90':0.0,'w_hn_mean':0.0,
        'theta_u':0.0,'u_p50':0.0,'u_p90':0.0,
    }
    batch_count = 0
    last_dump = {}
    gate_val = gate_u = gate_fn = gate_hn = t = 0.0

    for batch_idx, (sub_data_views, _, sample_idx) in enumerate(mv_data_loader):
        batch_psedo_label = model.psedo_labels[sample_idx]
        y_matrix = (batch_psedo_label.unsqueeze(1) == batch_psedo_label.unsqueeze(0)).int()
        xrs, zs = model(sub_data_views)
        common_z = model.fusion(zs)
        device = common_z.device
        batch_psedo_label = batch_psedo_label.to(device)

        model.update_centers(zs, common_z)
        features = zs + [common_z]
        memberships = [model.compute_membership(features[v], v) for v in range(num_views + 1)]
        u, u_hat_legacy = model.estimate_uncertainty(memberships, common_z)
        batch_N = common_z.size(0)

        q_ema_prev = model.q_ema_global[sample_idx].to(device)
        top_p_e = max(p_min, initial_top_p * max(0.0, 1.0 - (epoch - 1) / float(max_epoch - 1)))
        if enable_star_modules:
            lambda_vote_eff = lambda_vote * min(1.0, float(epoch) / max(1.0, float(vote_warmup_epochs)))
            q, q_proto, u_hat, vote, common_z, L_cal, margin_gap, agreement = _compute_sce_outputs(
                model, zs, common_z, q_ema_prev,
                lambda_vote=lambda_vote, tau_p=tau_p, knn_k=knn_neg_k,
                kappa=cal_kappa, delta=cal_delta, use_view_divergence=use_view_divergence,
                lambda_vote_eff=lambda_vote_eff, vote_eps=sce_vote_eps,
                alpha_u=ucm_alpha_u, beta_u=ucm_beta_u, m0=ucm_m0, a0=ucm_a0,
            )
            theta_target = torch.quantile(u_hat.detach(), theta_u_quantile).item()
            model.theta_u_state = (1.0 - theta_u_momentum) * model.theta_u_state + theta_u_momentum * theta_target
            theta_u = float(model.theta_u_state)
            uncertain_mask = u_hat > theta_u
            k_unc = int(uncertain_mask.sum().item())
            u_thr = theta_u
            q_ema_target = q_proto.detach() if epoch == 1 else q.detach()
            model.q_ema_global[sample_idx] = 0.9 * q_ema_prev + 0.1 * q_ema_target
        else:
            q = memberships[num_views]
            u_hat = u_hat_legacy
            k_unc = max(min(u_min, batch_N), int(batch_N * top_p_e))
            _, idx_topk = torch.topk(u_hat, k_unc, largest=True)
            uncertain_mask = torch.zeros(batch_N, dtype=torch.bool, device=device)
            uncertain_mask[idx_topk] = True
            u_thr = u_hat[idx_topk].min().item() if idx_topk.numel() > 0 else 0.0
            L_cal = torch.tensor(0.0, device=device)
            margin_gap = torch.zeros_like(u_hat)
            agreement = torch.ones_like(u_hat)
            theta_u = float(u_thr)

        u_mean = u_hat.mean().item()
        mu_start, mu_end = 0.3, 0.7
        raw_gate = (u_mean - mu_start) / (mu_end - mu_start)
        gate_u = float(max(0.0, min(1.0, raw_gate)))
        t = float((epoch - 1) / max(1, max_epoch - 1))
        gate_fn = t
        gate_hn = t
        gate_val = t

        prev_batch = None if y_prev_labels is None else y_prev_labels[sample_idx].to(device)
        route_stats = {}
        if enable_star_modules:
            route_mask = uncertain_mask if route_uncertain_only else None
            L_nce, L_hn, L_share, L_pi, model.share_lambda_state, route_stats, route_aux = _compute_csd_losses(
                model=model, z=common_z, q_proto=q_proto, u=u_hat, margin_gap=margin_gap, theta_u=theta_u,
                neg_mode=neg_mode, knn_neg_k=knn_neg_k, tau=temperature_f,
                sh=hn_sh, hn_margin=hn_margin, gamma_gate=gamma_gate,
                share_target=share_target, share_lambda=model.share_lambda_state,
                share_lambda_lr=share_lambda_lr, fn_ratio_cut=fn_ratio_cut, hn_ratio_cut=hn_ratio_cut,
                uncertain_mask=route_mask,
            )
            w_neg = route_aux['w_neg']
        else:
            route_mask = uncertain_mask if route_uncertain_only else None
            w_neg, eta_mat, _, route_stats, route_aux = _build_pairwise_fn_risk(
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
            L_nce = torch.tensor(0.0, device=device)
            L_hn = torch.tensor(0.0, device=device)
            L_share = torch.tensor(0.0, device=device)
            L_pi = torch.tensor(0.0, device=device)

        loss_list = []
        Lcl = Lfeat = Lu = Lpen = Lcross = Lrecon = 0.0
        q_centers = model.compute_centers(common_z, batch_psedo_label)
        for v in range(num_views):
            Wv = W[v][sample_idx][:, sample_idx].to(device)
            y_pse = y_matrix.float().to(device)
            k_centers = model.compute_centers(zs[v], batch_psedo_label)
            cl = mvc_loss.compute_cluster_loss(q_centers, k_centers, batch_psedo_label)
            Lcl_i = alpha * cl
            Lcl += Lcl_i.item()
            loss_list.append(Lcl_i)

            if not enable_star_modules:
                feat_l = mvc_loss.feature_loss(zs[v], common_z, Wv, y_pse, neg_weights=w_neg)
                Lfeat_i = alpha * feat_l
                Lfeat += Lfeat_i.item()
                loss_list.append(Lfeat_i)

                Lu_i = (1 - gate_u) * lambda_u * mvc_loss.uncertainty_regression_loss(u_hat, u)
                Lu += Lu_i.item()
                loss_list.append(Lu_i)

                sim_mat = F.cosine_similarity(common_z.unsqueeze(1), common_z.unsqueeze(0), dim=2)
                pos_sim = torch.diag(sim_mat)
                eta_cnt = eta_mat.sum().float()
                push_loss = torch.relu((sim_mat - pos_sim.unsqueeze(1) + 0.2) * eta_mat.float()).sum() / (eta_cnt + 1e-12) if eta_cnt > 0 else torch.tensor(0.0, device=device)
                pull_loss = (1.0 - pos_sim).mean()
                Lpen_i = gate_hn * lambda_hn_penalty * (push_loss + pull_loss)
                Lpen += Lpen_i.item()
                loss_list.append(Lpen_i)

            recon_loss = criterion(sub_data_views[v], xrs[v])
            Lrecon += recon_loss.item()
            loss_list.append(recon_loss)

        if epoch > cross_warmup_epochs:
            # Cross-view consistency should be computed once per batch to avoid num_views-times over-counting.
            cross_l = mvc_loss.cross_view_weighted_loss(model, zs, common_z, memberships, batch_psedo_label, temperature=temperature_f)
            Lcross_i = gate_fn * beta * cross_l
            Lcross += Lcross_i.item()
            loss_list.append(Lcross_i)

        if enable_star_modules:
            L_mmd, L_rep, L_excl = _ism_star_losses(
                common_z, q, F.normalize(model.centers[num_views], dim=1), model.cluster_size_ema,
                rho_n=ism_rho_n, p_per_cluster=ism_p_per_cluster, p_radius=ism_p_radius,
                sigma=ism_sigma, t_proj=ism_t_proj, margin_c=ism_margin,
            )
            L_ism = L_mmd + lambda_rep * L_rep + lambda_excl * L_excl
            loss_list.extend([L_nce, lambda_hn_penalty * L_hn, lambda_cal * L_cal, lambda_share * L_share, lambda_pi * L_pi, lambda_ism * L_ism])
            Lfeat += L_nce.item()
            Lu += L_cal.item()
            Lpen += L_hn.item()

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

        m_cons = q if enable_star_modules else memberships[num_views]
        top2_m = torch.topk(m_cons, 2, dim=1).values
        gamma = torch.log((top2_m[:, 0] + 1e-12) / (top2_m[:, 1] + 1e-12))
        sim_mat = route_aux['sim']
        neg_sim = sim_mat[route_aux['neg_mask']]
        route_stats['U_ratio'] = uncertain_mask.float().mean().item()
        route_stats['u_thr'] = u_thr
        route_stats['top_p_e'] = 0.0 if enable_star_modules else top_p_e
        route_stats['k_unc'] = k_unc
        route_stats['U_size'] = int(uncertain_mask.sum().item())
        route_stats['label_flip'] = ((batch_psedo_label != prev_batch).float().mean().item() if prev_batch is not None else 0.0)
        route_stats['stab_rate'] = 1.0 - route_stats['label_flip']
        route_stats['theta_u'] = theta_u
        route_stats['u_p50'] = torch.quantile(u_hat.detach(), 0.5).item()
        route_stats['u_p90'] = torch.quantile(u_hat.detach(), 0.9).item()
        if enable_star_modules and epoch <= debug_star_epochs:
            # 调试监控：验证 SCE/CSD 闭环没有被 one-hot vote 压死。
            vote_entropy = -(vote * torch.log(vote + 1e-12)).sum(dim=1).mean().item()
            g_dbg = (u_hat.clamp(0.0, 1.0) ** gamma_gate)
            print(
                f"[STAR-CHECK E{epoch} B{batch_idx}] u_mean={u_hat.mean().item():.4f} U_size={int((u_hat > u_threshold).sum().item())} "
                f"vote_max_mean={vote.max(dim=1).values.mean().item():.4f} vote_entropy_mean={vote_entropy:.4f} "
                f"g_mean={g_dbg.mean().item():.4f} g_p10={torch.quantile(g_dbg, 0.1).item():.4f} g_p90={torch.quantile(g_dbg, 0.9).item():.4f} "
                f"w_mean={route_aux['w_neg'].mean().item():.4f} w_p10={torch.quantile(route_aux['w_neg'], 0.1).item():.4f} w_min={route_aux['w_neg'].min().item():.4f}"
            )

        for k in route_meter:
            route_meter[k] += route_stats.get(k, 0.0)
        batch_count += 1

        last_dump = {
            'u_sample': u_hat.detach().cpu(),
            'gamma_sample': gamma.detach().cpu(),
            'm_top1_sample': top2_m[:, 0].detach().cpu(),
            'm_gap_sample': (top2_m[:, 0] - top2_m[:, 1]).detach().cpu(),
            'y_curr_sample': batch_psedo_label.detach().cpu(),
            'y_prev_sample': (prev_batch.detach().cpu() if prev_batch is not None else torch.full_like(batch_psedo_label.detach().cpu(), -1)),
            'flip_mask_sample': ((batch_psedo_label != prev_batch).float().detach().cpu() if prev_batch is not None else torch.zeros_like(batch_psedo_label, dtype=torch.float32).detach().cpu()),
            'S_pair_sample': route_aux['r_fn'][route_aux['neg_mask']].detach().cpu() if 'r_fn' in route_aux else route_aux['S'][route_aux['neg_mask']].detach().cpu(),
            'w_pair_sample': route_aux['w_neg'][route_aux['neg_mask']].detach().cpu(),
            'w_neg_pair_sample': route_aux['w_neg'][route_aux['neg_mask']].detach().cpu(),
            's_post_pair_sample': (q @ q.t())[route_aux['neg_mask']].detach().cpu(),
            'p_same_pair_sample': (q @ q.t())[route_aux['neg_mask']].detach().cpu(),
            'sim_pair_sample': neg_sim.detach().cpu(),
            'rho_fn_pair_sample': (route_aux['r_fn'][route_aux['neg_mask']] > fn_ratio_cut).float().detach().cpu() if 'r_fn' in route_aux else route_aux['rho'][route_aux['neg_mask']].float().detach().cpu(),
            'eta_hn_pair_sample': (route_aux['r_hn'][route_aux['neg_mask']] > hn_ratio_cut).float().detach().cpu() if 'r_hn' in route_aux else route_aux['eta'][route_aux['neg_mask']].float().detach().cpu(),
            'r_fn_pair_sample': route_aux['r_fn'][route_aux['neg_mask']].detach().cpu() if 'r_fn' in route_aux else route_aux['S'][route_aux['neg_mask']].detach().cpu(),
            'r_hn_pair_sample': route_aux['r_hn'][route_aux['neg_mask']].detach().cpu() if 'r_hn' in route_aux else route_aux['sim'][route_aux['neg_mask']].detach().cpu(),
            'r_pair_sample': route_aux['r_fn'][route_aux['neg_mask']].detach().cpu() if 'r_fn' in route_aux else route_aux['r'][route_aux['neg_mask']].detach().cpu(),
            's_stab_pair_sample': torch.ones_like(neg_sim).detach().cpu(),
            'sim_pos_sample': F.cosine_similarity(zs[0], common_z, dim=1).detach().cpu(),
            'sim_neg_sample': neg_sim.detach().cpu(),
            'pairs_sampled': torch.tensor(float(neg_sim.numel())),
            'neg_pairs_available': torch.tensor(float(route_aux['neg_mask'].float().sum().item())),
            'safe_pairs_available': torch.tensor(float((route_aux['neg_mask'] & (~(route_aux['r_fn'] > fn_ratio_cut))).float().sum().item())) if 'r_fn' in route_aux else torch.tensor(float((route_aux['neg_mask'] & (~route_aux['rho'])).float().sum().item())),
            'pos_pairs_count': torch.tensor(float(batch_N)),
            'uncertain_mask_sample': uncertain_mask.detach().cpu(),
            'neg_mask_sample': route_aux['neg_mask'].detach().cpu(),
            'top_p_e': torch.tensor(route_stats['top_p_e']),
            'k_unc': torch.tensor(k_unc),
            'tau_fn_per_anchor': torch.zeros(batch_N),
            'tau_hn_per_anchor': torch.zeros(batch_N),
            'FN_count_per_anchor': torch.zeros(batch_N),
            'HN_count_per_anchor': torch.zeros(batch_N),
            'gate_val': torch.tensor(gate_val),
        }

        counts = torch.bincount(torch.argmax(q.detach(), dim=1), minlength=model.num_clusters).float().to(device)
        model.cluster_size_ema = 0.9 * model.cluster_size_ema + 0.1 * counts
        print(f"[Epoch {epoch} Batch {batch_idx}] Total={total_loss.item():.4f} FN_ratio={route_stats['fn_ratio']:.3f} HN_ratio={route_stats['hn_ratio']:.3f} U_ratio={route_stats['U_ratio']:.3f}")

    if batch_count > 0:
        for k in epoch_meter:
            epoch_meter[k] /= batch_count
        for k in route_meter:
            route_meter[k] /= batch_count

    return {'loss': epoch_meter, 'route': route_meter, 'dump': last_dump,
            'gate': gate_val if batch_count > 0 else 0.0, 'gate_u': gate_u if batch_count > 0 else 0.0,
            'gate_fn': gate_fn if batch_count > 0 else 0.0, 'gate_hn': gate_hn if batch_count > 0 else 0.0,
            't': t if batch_count > 0 else 0.0, 'warmup_epochs': warmup_epochs,
            'cross_warmup_epochs': cross_warmup_epochs}



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
                                   u_min=32):
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
    route_meter = {
        'fn_ratio':0.0,'safe_ratio':0.0,'hn_ratio':0.0,
        'mean_s_post_fn':0.0,'mean_s_post_non_fn':0.0,'delta_post':0.0,
        'mean_sim_hn':0.0,'mean_sim_safe_non_hn':0.0,'delta_sim':0.0,
        'label_flip':0.0,'stab_rate':0.0,
        'denom_fn_share':0.0,'denom_safe_share':0.0,
        'w_hit_min_ratio':0.0,'corr_u_fn_ratio':0.0,
        'N_size':0.0,'U_size':0.0,
        'neg_count':0.0,'safe_neg_count':0.0,'FN_count':0.0,'HN_count':0.0,
        'w_mean_on_FN':0.0,'w_mean_on_safe':0.0,
        'neg_per_anchor':0.0,'fn_pair_share':0.0,'hn_pair_share':0.0,
        'neg_used_in_loss_size':0.0,
        'pi_mean':0.0,'pi_p90':0.0,'w_hn_mean':0.0,
        'theta_u':0.0,'u_p50':0.0,'u_p90':0.0,
    }
    batch_count = 0
    last_dump = {}

    # 1) 课程学习式动态不确定比例
    top_p = max(p_min, initial_top_p * max(0.0, 1.0 - (epoch - 1) / float(max_epoch - 1)))

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

        # ——— 更新中心、隶属度、不确定度 ———
        model.update_centers(zs, common_z)
        feats = zs + [common_z]
        memberships = [model.compute_membership(feats[v], v) for v in range(num_views + 1)]
        u, u_hat = model.estimate_uncertainty(memberships, common_z)
        B = u_hat.size(0)

        # ——— 课程学习式不确定划分 ———
        k_unc = max(min(u_min, B), int(B * top_p))
        _, topk_idx = torch.topk(u_hat, k_unc, largest=True)
        uncertain = torch.zeros(B, dtype=torch.bool, device=device)
        uncertain[topk_idx] = True
        certain = ~uncertain

        # ——— 动态门控 Gate ———
        u_mean = u_hat.mean().item()
        mu_lo, mu_hi = 0.3, 0.7
        gate_u = float((u_mean - mu_lo) / (mu_hi - mu_lo))
        gate_u = max(0.0, min(1.0, gate_u))
        t = float((epoch - 1) / max(1, max_epoch - 1))
        gate_fn = t
        gate_hn = t
        gate = t
        gate_t = torch.tensor(gate, device=device)

        # ——— 共识中心 ———
        q_centers = model.compute_centers(common_z, batch_label)

        # ——— Design 1': pair-wise FN 风险路由（停用原 FN/HN MLP 路径）———
        prev_batch = None if y_prev_labels is None else y_prev_labels[sample_idx].to(device)
        route_mask = uncertain if route_uncertain_only else None
        u_thr = u_hat[topk_idx].min().item() if topk_idx.numel() > 0 else 0.0
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

        # e) 跨视图加权 InfoNCE（每个 batch 只计算一次）
        if epoch > cross_warmup_epochs:
            cross_l = mvc_loss.cross_view_weighted_loss(
                model, zs, common_z, memberships,
                batch_label, temperature=temperature_f
            )
            Lcross = gate_fn * beta * prog * cross_l
            batch_loss += Lcross
        else:
            Lcross = torch.tensor(0.0, device=device)

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
        epoch_meter['L_cross'] += Lcross.item() if hasattr(Lcross, 'item') else float(Lcross)

        m_cons = memberships[num_views]
        top2_m = torch.topk(m_cons, 2, dim=1).values
        gamma = torch.log((top2_m[:, 0] + 1e-12) / (top2_m[:, 1] + 1e-12))
        sim_mat = route_aux['sim']
        pos_sim = F.cosine_similarity(zs[0], common_z, dim=1)
        neg_sim = sim_mat[route_aux['neg_mask']]
        route_stats['U_ratio'] = float(k_unc) / max(B, 1)
        route_stats['u_thr'] = u_thr
        route_stats['top_p_e'] = top_p
        route_stats['k_unc'] = k_unc
        route_stats['U_size'] = int(uncertain.sum().item())

        for k in route_meter:
            route_meter[k] += route_stats.get(k, 0.0)
        batch_count += 1

        last_dump = {
            'u_sample': u_hat.detach().cpu(),
            'gamma_sample': gamma.detach().cpu(),
            'm_top1_sample': top2_m[:, 0].detach().cpu(),
            'm_gap_sample': (top2_m[:, 0] - top2_m[:, 1]).detach().cpu(),
            'y_curr_sample': batch_label.detach().cpu(),
            'y_prev_sample': (prev_batch.detach().cpu() if prev_batch is not None else torch.full_like(batch_label.detach().cpu(), -1)),
            'flip_mask_sample': ((batch_label != prev_batch).float().detach().cpu() if prev_batch is not None else torch.zeros_like(batch_label, dtype=torch.float32).detach().cpu()),
            'S_pair_sample': route_aux['S'][route_aux['neg_mask']].detach().cpu(),
            'w_pair_sample': route_aux['w_neg'][route_aux['neg_mask']].detach().cpu(),
            'w_neg_pair_sample': route_aux['w_neg'][route_aux['neg_mask']].detach().cpu(),
            's_post_pair_sample': route_aux['s_post'][route_aux['neg_mask']].detach().cpu(),
            'p_same_pair_sample': route_aux['s_post'][route_aux['neg_mask']].detach().cpu(),
            'sim_pair_sample': neg_sim.detach().cpu(),
            'rho_fn_pair_sample': route_aux['rho'][route_aux['neg_mask']].float().detach().cpu(),
            'eta_hn_pair_sample': route_aux['eta'][route_aux['neg_mask']].float().detach().cpu(),
            'r_fn_pair_sample': route_aux['S'][route_aux['neg_mask']].detach().cpu(),
            'r_hn_pair_sample': route_aux['sim'][route_aux['neg_mask']].detach().cpu(),
            'r_pair_sample': route_aux['r'][route_aux['neg_mask']].detach().cpu(),
            's_stab_pair_sample': route_aux['s_stab'][route_aux['neg_mask']].detach().cpu(),
            'sim_pos_sample': pos_sim.detach().cpu(),
            'sim_neg_sample': neg_sim.detach().cpu(),
            'pairs_sampled': torch.tensor(float(neg_sim.numel())),
            'neg_pairs_available': torch.tensor(float(route_aux['neg_mask'].float().sum().item())),
            'safe_pairs_available': torch.tensor(float((route_aux['neg_mask'] & (~route_aux['rho'])).float().sum().item())),
            'pos_pairs_count': torch.tensor(float(pos_sim.numel())),
            'uncertain_mask_sample': uncertain.detach().cpu(),
            'neg_mask_sample': route_aux['neg_mask'].detach().cpu(),
            'top_p_e': torch.tensor(top_p),
            'k_unc': torch.tensor(k_unc),
            'tau_fn_per_anchor': route_aux['tau_fn_per_anchor'].detach().cpu(),
            'tau_hn_per_anchor': route_aux['tau_hn_per_anchor'].detach().cpu(),
            'FN_count_per_anchor': route_aux['FN_count_per_anchor'].detach().cpu(),
            'HN_count_per_anchor': route_aux['HN_count_per_anchor'].detach().cpu(),
            'gate_val': torch.tensor(gate),
        }

    if batch_count > 0:
        for k in epoch_meter:
            epoch_meter[k] /= batch_count
        for k in route_meter:
            route_meter[k] /= batch_count

    return {'loss': epoch_meter, 'route': route_meter, 'dump': last_dump, 'gate': gate if batch_count > 0 else 0.0, 'gate_u': gate_u if batch_count > 0 else 0.0, 'gate_fn': gate_fn if batch_count > 0 else 0.0, 'gate_hn': gate_hn if batch_count > 0 else 0.0, 't': t if batch_count > 0 else 0.0, 'warmup_epochs': warmup_epochs, 'cross_warmup_epochs': cross_warmup_epochs}
