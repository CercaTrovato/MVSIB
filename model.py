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
        stab_vec = torch.zeros(N, device=device)
        assign_stability = 0.0
        s_stab = torch.zeros(N, N, device=device)
    else:
        stab_vec = _aligned_assignment_stability(batch_labels, prev_labels_batch).to(device)
        assign_stability = stab_vec.mean().item()
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

    routed_anchor = uncertain_mask if uncertain_mask is not None else torch.ones(N, dtype=torch.bool, device=device)
    routed_anchor_mask = routed_anchor.unsqueeze(1).expand_as(neg_mask)
    routed_neg_mask = neg_mask & routed_anchor_mask
    routed_safe_mask = safe_mask & routed_anchor_mask
    fn_mask = routed_neg_mask & rho
    hn_mask = routed_safe_mask & eta
    fn_count = fn_mask.float().sum().item()
    hn_count = hn_mask.float().sum().item()
    neg_count = routed_neg_mask.float().sum().item()
    safe_neg_count = routed_safe_mask.float().sum().item()
    candidate_neg_size = (routed_anchor.float().sum().item() * max(N - 1, 0))

    stats = {
        'fn_ratio': (fn_count / max(neg_count, 1.0)),
        'safe_ratio': (safe_neg_count / max(neg_count, 1.0)),
        'hn_ratio': (hn_count / max(safe_neg_count, 1.0)),
        'FN_count': fn_count,
        'HN_count': hn_count,
        'neg_count': neg_count,
        'safe_neg_count': safe_neg_count,
        'candidate_neg_size': candidate_neg_size,
        'neg_after_filter_size': neg_count,
        'neg_used_in_loss_size': neg_count,
        'w_mean_on_FN': w_neg[rho].mean().item() if rho.any() else 0.0,
        'w_mean_on_safe': w_neg[safe_mask].mean().item() if safe_mask.any() else 0.0,
        'mean_s_post_fn': mean_s_post_fn,
        'mean_s_post_non_fn': mean_s_post_non_fn,
        'delta_post': mean_s_post_fn - mean_s_post_non_fn,
        'mean_sim_hn': mean_sim_hn,
        'mean_sim_safe_non_hn': mean_sim_safe_non_hn,
        'delta_sim': mean_sim_hn - mean_sim_safe_non_hn,
        'label_flip': (1.0 - assign_stability) if prev_labels_batch is not None else 0.0,
        'stab_rate': assign_stability if prev_labels_batch is not None else 0.0,
        'assignment_stability': assign_stability if prev_labels_batch is not None else 0.0,
        'denom_fn_share': denom_fn_share,
        'denom_safe_share': 1.0 - denom_fn_share,
        'w_hit_min_ratio': ((w_neg <= (w_min + eps)) & rho).float().mean().item() if rho.any() else 0.0,
        'corr_u_fn_ratio': corr_u_fn.item(),
        'N_size': (neg_mask.float().sum(dim=1).mean().item()),
        'neg_per_anchor': (neg_mask.float().sum(dim=1).mean().item()),
        'U_size': int(uncertain_mask.sum().item()) if uncertain_mask is not None else int(N),
        'fn_pair_share': (fn_count / max(neg_count, 1.0)),
        'hn_pair_share': (hn_count / max(safe_neg_count, 1.0)),
        'tau_fn_p10': torch.nanquantile(tau_fn_per_anchor, 0.1).item() if torch.isfinite(tau_fn_per_anchor).any() else 0.0,
        'tau_fn_p50': torch.nanquantile(tau_fn_per_anchor, 0.5).item() if torch.isfinite(tau_fn_per_anchor).any() else 0.0,
        'tau_fn_p90': torch.nanquantile(tau_fn_per_anchor, 0.9).item() if torch.isfinite(tau_fn_per_anchor).any() else 0.0,
        'tau_hn_p10': torch.nanquantile(tau_hn_per_anchor, 0.1).item() if torch.isfinite(tau_hn_per_anchor).any() else 0.0,
        'tau_hn_p50': torch.nanquantile(tau_hn_per_anchor, 0.5).item() if torch.isfinite(tau_hn_per_anchor).any() else 0.0,
        'tau_hn_p90': torch.nanquantile(tau_hn_per_anchor, 0.9).item() if torch.isfinite(tau_hn_per_anchor).any() else 0.0,
        'FN_count_anchor_p50': torch.quantile(fn_count_per_anchor, 0.5).item() if fn_count_per_anchor.numel() > 0 else 0.0,
        'HN_count_anchor_p50': torch.quantile(hn_count_per_anchor, 0.5).item() if hn_count_per_anchor.numel() > 0 else 0.0,
    }
    aux = {
        'S': S, 's_post': s_post, 'sim': sim, 'rho': rho, 'eta': eta, 'w_neg': w_neg,
        'r': r, 's_stab': s_stab, 'neg_mask': neg_mask,
        'tau_fn_per_anchor': tau_fn_per_anchor, 'tau_hn_per_anchor': tau_hn_per_anchor,
        'FN_count_per_anchor': fn_count_per_anchor, 'HN_count_per_anchor': hn_count_per_anchor,
    }
    return w_neg, eta, rho, stats, aux




def _aligned_assignment_stability(batch_labels, prev_labels_batch):
    """对齐簇ID后计算伪标签稳定性，缓解簇编号置换导致的虚假翻转。"""
    if prev_labels_batch is None:
        return None
    curr = batch_labels.detach().clone()
    prev = prev_labels_batch.detach().clone()
    mapped = curr.clone()
    for c in curr.unique():
        m = (curr == c)
        prev_on_c = prev[m]
        if prev_on_c.numel() == 0:
            continue
        vals, cnts = prev_on_c.unique(return_counts=True)
        mapped[m] = vals[cnts.argmax()]
    return (mapped == prev).float()

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
                      fn_route_warmup_epochs=15,
                      feature_base_weight=1.0,
                      feature_route_weight=1.0,
                      y_prev_labels=None,
                      p_min=0.05,
                      u_min=32,
                      lambda_cross=1.0,
                      cross_ramp_epochs=10):
    model.train()
    mv_data_loader, num_views, num_samples, num_clusters = get_multiview_data(mv_data, batch_size)

    # 将 all_features 和 all_labels 初始化为 Python 列表
    all_features = []  # 用于收集每个批次的特征
    all_labels = []  # 用于收集每个批次的标签

    # 课程学习式动态不确定比例
    top_p_e = max(p_min, initial_top_p * max(0.0, 1.0 - (epoch - 1) / float(max_epoch - 1)))

    # E 步：更新全量伪标签
    psedo_labeling(model, mv_data, batch_size)

    # Push/Pull Lpen 超参
    lambda_push = lambda_hn_penalty
    lambda_pull = lambda_hn_penalty
    margin = 0.2

    criterion = torch.nn.MSELoss()  # 添加重建损失的损失函数

    epoch_meter = {'L_total':0.0,'L_recon':0.0,'L_feat':0.0,'L_cross':0.0,'L_cluster':0.0,'L_uncert':0.0,'L_hn':0.0,'L_reg':0.0}
    route_meter = {'fn_ratio':0.0,'safe_ratio':0.0,'hn_ratio':0.0,'mean_s_post_fn':0.0,'mean_s_post_non_fn':0.0,'delta_post':0.0,'mean_sim_hn':0.0,'mean_sim_safe_non_hn':0.0,'delta_sim':0.0,'label_flip':0.0,'stab_rate':0.0,'assignment_stability':0.0,'denom_fn_share':0.0,'denom_safe_share':0.0,'w_hit_min_ratio':0.0,'corr_u_fn_ratio':0.0,'N_size':0.0,'U_size':0.0,'neg_per_anchor':0.0,'FN_count':0.0,'HN_count':0.0,'neg_count':0.0,'safe_neg_count':0.0,'candidate_neg_size':0.0,'neg_after_filter_size':0.0,'neg_used_in_loss_size':0.0,'fn_pair_share':0.0,'hn_pair_share':0.0,'w_mean_on_FN':0.0,'w_mean_on_safe':0.0,'tau_fn_p10':0.0,'tau_fn_p50':0.0,'tau_fn_p90':0.0,'tau_hn_p10':0.0,'tau_hn_p50':0.0,'tau_hn_p90':0.0,'FN_count_anchor_p50':0.0,'HN_count_anchor_p50':0.0,'U_ratio':0.0,'u_thr':0.0,'top_p_e':0.0,'k_unc':0.0}
    batch_count = 0
    last_dump = {}

    for batch_idx, (sub_data_views, _, sample_idx) in enumerate(mv_data_loader):
        # ——— 1) 伪标签 & 同/异样本矩阵 ———
        batch_psedo_label = model.psedo_labels[sample_idx]                # [N]
        y_matrix = (batch_psedo_label.unsqueeze(1) == batch_psedo_label.unsqueeze(0)).int()

        # ——— 2) 编码 + 融合 ———
        xrs, zs = model(sub_data_views)
        common_z = model.fusion(zs)

        # 现在有了 common_z，确定 device
        device = common_z.device

        # 把索引张量都搬到 device
        batch_psedo_label = batch_psedo_label.to(device)

        # ——— 3) 更新中心 + 隶属度 + 不确定度 ———
        model.update_centers(zs, common_z)
        features = zs + [common_z]
        memberships = [model.compute_membership(features[v], v) for v in range(num_views + 1)]
        u, u_hat = model.estimate_uncertainty(memberships, common_z)
        batch_N  = u_hat.size(0)

        # ——— 4) 课程学习式不确定划分 ———
        k_unc = max(min(u_min, batch_N), int(batch_N * top_p_e))
        _, idx_topk = torch.topk(u_hat, k_unc, largest=True)
        uncertain_mask = torch.zeros(batch_N, dtype=torch.bool, device=device)
        uncertain_mask[idx_topk] = True
        certain_mask = ~uncertain_mask

        print(f"Batch {batch_idx}: uncertain {uncertain_mask.sum().item()}/{batch_N} = {uncertain_mask.sum().item()/batch_N:.2%}")

        # ——— 5) 动态门控 Gate ———
        u_mean = u_hat.mean().item()
        mu_start, mu_end = 0.3, 0.7
        raw_gate = (u_mean - mu_start) / (mu_end - mu_start)
        gate_u = float(max(0.0, min(1.0, raw_gate)))
        t = float((epoch - 1) / max(1, max_epoch - 1))
        gate_fn = t
        gate_hn = t
        gate_val = t
        gate = torch.tensor(gate_val, device=device)

        # ——— 6) 计算共识中心 q_centers ———
        q_centers = model.compute_centers(common_z, batch_psedo_label)

        # ——— 7) Design 1': pair-wise FN 风险路由（停用原 FN/HN MLP 路径）———
        prev_batch = None if y_prev_labels is None else y_prev_labels[sample_idx].to(device)
        route_active = (epoch > fn_route_warmup_epochs)
        route_mask = uncertain_mask if (route_uncertain_only and route_active) else None
        u_thr = u_hat[idx_topk].min().item() if idx_topk.numel() > 0 else 0.0
        if route_active:
            # 仅在 warmup 后启用 FN 风险路由；warmup 阶段保持全量对比信号
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
        else:
            eye = torch.eye(batch_N, dtype=torch.bool, device=device)
            neg_mask = (~eye) & (y_matrix.to(device) == 0)
            w_neg = torch.ones(batch_N, batch_N, device=device)
            eta_mat = torch.zeros(batch_N, batch_N, dtype=torch.bool, device=device)
            rho_mat = torch.zeros(batch_N, batch_N, dtype=torch.bool, device=device)
            route_stats = {
                'fn_ratio':0.0,'safe_ratio':1.0,'hn_ratio':0.0,'FN_count':0.0,'HN_count':0.0,
                'neg_count':neg_mask.float().sum().item(),'safe_neg_count':neg_mask.float().sum().item(),
                'candidate_neg_size':neg_mask.float().sum().item(),'neg_after_filter_size':neg_mask.float().sum().item(),
                'neg_used_in_loss_size':neg_mask.float().sum().item(),'mean_s_post_fn':0.0,'mean_s_post_non_fn':0.0,
                'delta_post':0.0,'mean_sim_hn':0.0,'mean_sim_safe_non_hn':0.0,'delta_sim':0.0,
                'label_flip':0.0,'stab_rate':0.0,'assignment_stability':0.0,'denom_fn_share':0.0,'denom_safe_share':1.0,
                'w_hit_min_ratio':0.0,'corr_u_fn_ratio':0.0,'N_size':neg_mask.float().sum(dim=1).mean().item(),
                'neg_per_anchor':neg_mask.float().sum(dim=1).mean().item(),'U_size':int(uncertain_mask.sum().item()),
                'fn_pair_share':0.0,'hn_pair_share':0.0,'w_mean_on_FN':0.0,'w_mean_on_safe':1.0,
                'tau_fn_p10':0.0,'tau_fn_p50':0.0,'tau_fn_p90':0.0,'tau_hn_p10':0.0,'tau_hn_p50':0.0,'tau_hn_p90':0.0,
                'FN_count_anchor_p50':0.0,'HN_count_anchor_p50':0.0,
            }
            route_aux = {
                'w_neg':w_neg,'neg_mask':neg_mask,'rho':rho_mat,'eta':eta_mat,
                'tau_fn_per_anchor':torch.zeros(batch_N, device=device),
                'tau_hn_per_anchor':torch.zeros(batch_N, device=device),
                'S':torch.zeros(batch_N, batch_N, device=device),
                'r':torch.zeros(batch_N, batch_N, device=device),
                's_post':torch.zeros(batch_N, batch_N, device=device),
            }

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

            # b) Feature loss：全体 anchor 的基础 InfoNCE + uncertain 路由修正 InfoNCE
            feat_base = mvc_loss.feature_loss(zs[v], common_z, Wv, y_pse, neg_weights=None)
            feat_route = mvc_loss.feature_loss(zs[v], common_z, Wv, y_pse, neg_weights=w_neg)
            feat_loss = feature_base_weight * feat_base + feature_route_weight * feat_route
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
                cross_l = cross_l / max(float(num_views), 1.0)
                cross_ramp = min(1.0, max(0.0, (epoch - cross_warmup_epochs) / float(max(1, cross_ramp_epochs))))
                Lcross_i = gate_fn * beta * lambda_cross * cross_ramp * cross_l
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
        neg_sim = sim_mat[route_aux['neg_mask']]
        route_stats['U_ratio'] = float(k_unc) / max(batch_N, 1)
        route_stats['u_thr'] = u_thr
        route_stats['top_p_e'] = top_p_e
        route_stats['k_unc'] = k_unc

        last_dump = {
            'u_sample': u_hat.detach().cpu(),
            'gamma_sample': gamma.detach().cpu(),
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
            'neg_mask_sample': route_aux['neg_mask'].detach().cpu(),
            'top_p_e': torch.tensor(top_p_e),
            'k_unc': torch.tensor(k_unc),
            'tau_fn_per_anchor': route_aux['tau_fn_per_anchor'].detach().cpu(),
            'tau_hn_per_anchor': route_aux['tau_hn_per_anchor'].detach().cpu(),
            'FN_count_per_anchor': route_aux['FN_count_per_anchor'].detach().cpu(),
            'HN_count_per_anchor': route_aux['HN_count_per_anchor'].detach().cpu(),
            'gate_val': torch.tensor(gate_val),
        }

        route_stats['U_ratio'] = float(k_unc) / max(batch_N, 1)
        route_stats['u_thr'] = u_thr
        route_stats['top_p_e'] = top_p_e
        route_stats['k_unc'] = k_unc
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
                                   alpha_fn=0.1,
                                   pi_fn=0.1,
                                   w_min=0.05,
                                   hn_beta=0.1,
                                   neg_mode='batch',
                                   knn_neg_k=20,
                                   route_uncertain_only=True,
                                   fn_route_warmup_epochs=15,
                                   feature_base_weight=1.0,
                                   feature_route_weight=1.0,
                                   y_prev_labels=None,
                                   p_min=0.05,
                                   u_min=32,
                                   lambda_cross=1.0,
                                   cross_ramp_epochs=10):
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
    route_meter = {'fn_ratio':0.0,'safe_ratio':0.0,'hn_ratio':0.0,'mean_s_post_fn':0.0,'mean_s_post_non_fn':0.0,'delta_post':0.0,'mean_sim_hn':0.0,'mean_sim_safe_non_hn':0.0,'delta_sim':0.0,'label_flip':0.0,'stab_rate':0.0,'assignment_stability':0.0,'denom_fn_share':0.0,'denom_safe_share':0.0,'w_hit_min_ratio':0.0,'corr_u_fn_ratio':0.0,'N_size':0.0,'U_size':0.0,'neg_per_anchor':0.0,'FN_count':0.0,'HN_count':0.0,'neg_count':0.0,'safe_neg_count':0.0,'candidate_neg_size':0.0,'neg_after_filter_size':0.0,'neg_used_in_loss_size':0.0,'fn_pair_share':0.0,'hn_pair_share':0.0,'w_mean_on_FN':0.0,'w_mean_on_safe':0.0,'tau_fn_p10':0.0,'tau_fn_p50':0.0,'tau_fn_p90':0.0,'tau_hn_p10':0.0,'tau_hn_p50':0.0,'tau_hn_p90':0.0,'FN_count_anchor_p50':0.0,'HN_count_anchor_p50':0.0,'U_ratio':0.0,'u_thr':0.0,'top_p_e':0.0,'k_unc':0.0}
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
        route_active = (epoch > fn_route_warmup_epochs)
        route_mask = uncertain if (route_uncertain_only and route_active) else None
        u_thr = u_hat[topk_idx].min().item() if topk_idx.numel() > 0 else 0.0
        if route_active:
            # 仅在 warmup 后启用 FN 风险路由；warmup 阶段保持全量对比信号
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
        else:
            eye = torch.eye(B, dtype=torch.bool, device=device)
            neg_mask = (~eye) & (y_matrix.to(device) == 0)
            w_neg = torch.ones(B, B, device=device)
            eta_mat = torch.zeros(B, B, dtype=torch.bool, device=device)
            rho_mat = torch.zeros(B, B, dtype=torch.bool, device=device)
            route_stats = {
                'fn_ratio':0.0,'safe_ratio':1.0,'hn_ratio':0.0,'FN_count':0.0,'HN_count':0.0,
                'neg_count':neg_mask.float().sum().item(),'safe_neg_count':neg_mask.float().sum().item(),
                'candidate_neg_size':neg_mask.float().sum().item(),'neg_after_filter_size':neg_mask.float().sum().item(),
                'neg_used_in_loss_size':neg_mask.float().sum().item(),'mean_s_post_fn':0.0,'mean_s_post_non_fn':0.0,
                'delta_post':0.0,'mean_sim_hn':0.0,'mean_sim_safe_non_hn':0.0,'delta_sim':0.0,
                'label_flip':0.0,'stab_rate':0.0,'assignment_stability':0.0,'denom_fn_share':0.0,'denom_safe_share':1.0,
                'w_hit_min_ratio':0.0,'corr_u_fn_ratio':0.0,'N_size':neg_mask.float().sum(dim=1).mean().item(),
                'neg_per_anchor':neg_mask.float().sum(dim=1).mean().item(),'U_size':int(uncertain.sum().item()),
                'fn_pair_share':0.0,'hn_pair_share':0.0,'w_mean_on_FN':0.0,'w_mean_on_safe':1.0,
                'tau_fn_p10':0.0,'tau_fn_p50':0.0,'tau_fn_p90':0.0,'tau_hn_p10':0.0,'tau_hn_p50':0.0,'tau_hn_p90':0.0,
                'FN_count_anchor_p50':0.0,'HN_count_anchor_p50':0.0,
            }
            route_aux = {
                'w_neg':w_neg,'neg_mask':neg_mask,'rho':rho_mat,'eta':eta_mat,
                'tau_fn_per_anchor':torch.zeros(B, device=device),
                'tau_hn_per_anchor':torch.zeros(B, device=device),
                'S':torch.zeros(B, B, device=device),
                'r':torch.zeros(B, B, device=device),
                's_post':torch.zeros(B, B, device=device),
            }

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

            # b) Feature loss：全体 anchor 的基础 InfoNCE + uncertain 路由修正 InfoNCE
            feat_base = mvc_loss.feature_loss(zs[v], common_z, Wv, y_pse, neg_weights=None)
            feat_route = mvc_loss.feature_loss(zs[v], common_z, Wv, y_pse, neg_weights=w_neg)
            feat_l = feature_base_weight * feat_base + feature_route_weight * feat_route
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
                cross_l = cross_l / max(float(num_views), 1.0)
                cross_ramp = min(1.0, max(0.0, (epoch - cross_warmup_epochs) / float(max(1, cross_ramp_epochs))))
                Lcross = gate_fn * beta * lambda_cross * prog * cross_ramp * cross_l
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
        neg_sim = sim_mat[route_aux['neg_mask']]
        route_stats['U_ratio'] = float(k_unc) / max(B, 1)
        route_stats['u_thr'] = u_thr
        route_stats['top_p_e'] = top_p
        route_stats['k_unc'] = k_unc

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
