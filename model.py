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
                            gate_val, alpha_fn=0.1,
                            hn_beta=0.1, neg_mode='batch', knn_k=20,
                            uncertain_mask=None, eps=1e-6):
    """
    Design 1': calibrated soft FN-risk routing.
    - 保持全体 negatives 参与 InfoNCE，仅对疑似 FN 负对做软降权
    - 使用 sim-tail calibrated 概率：p_fn=σ((sim-τ_99)/σ_s) * g_unc
    - uncertain 子集只用于调试统计，不再作为 loss 的硬筛选集合
    """
    # Routing 是启发式重加权模块，不参与反传，避免数值链路干扰主干梯度。
    common_z = common_z.detach()
    memberships_cons = memberships_cons.detach()
    u_hat = u_hat.detach()
    batch_labels = batch_labels.detach()
    if prev_labels_batch is not None:
        prev_labels_batch = prev_labels_batch.detach()
    if uncertain_mask is not None:
        uncertain_mask = uncertain_mask.detach()

    device = common_z.device
    N = common_z.size(0)

    with torch.no_grad():
        eye = torch.eye(N, dtype=torch.bool, device=device)
        neg_mask_full = ~eye
        if neg_mode == 'knn':
            k_eff = min(knn_k + 1, N)
            dist = torch.cdist(common_z, common_z, p=2)
            knn_idx = torch.topk(-dist, k_eff, dim=1).indices
            knn_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
            row_idx = torch.arange(N, device=device).unsqueeze(1).expand_as(knn_idx)
            knn_mask[row_idx, knn_idx] = True
            neg_mask_full = knn_mask & (~eye)

        if uncertain_mask is None:
            uncertain_mask = torch.ones(N, dtype=torch.bool, device=device)
        neg_mask_routed = neg_mask_full & uncertain_mask.unsqueeze(1) & uncertain_mask.unsqueeze(0)

        s_post = torch.mm(memberships_cons, memberships_cons.t()).clamp(0.0, 1.0)
        if prev_labels_batch is None:
            stab_vec = torch.zeros(N, device=device)
            assign_stability = 0.0
            s_stab = torch.zeros(N, N, device=device)
        else:
            stab_vec = _aligned_assignment_stability(batch_labels, prev_labels_batch).to(device)
            assign_stability = stab_vec.mean().item()
            s_stab = torch.ger(stab_vec, stab_vec)

        r = (1.0 - 0.5 * (u_hat.unsqueeze(1) + u_hat.unsqueeze(0))).clamp(0.0, 1.0)
        S = r * ((s_post + s_stab) / 2.0)
        S = S * neg_mask_full.float()

        sim = F.cosine_similarity(common_z.unsqueeze(1), common_z.unsqueeze(0), dim=2)
        sim_neg = sim[neg_mask_full]
        if sim_neg.numel() > 0:
            tau = torch.quantile(sim_neg, 0.99).detach()
        else:
            tau = torch.tensor(1.0, device=device)

        # sim-tail calibrated 风险：直接对齐 sim_neg_p99 的尾部污染。
        sigma_s = 0.05
        p_fn_sim = torch.sigmoid((sim - tau) / (sigma_s + eps))
        g_unc = (u_hat.unsqueeze(1) + u_hat.unsqueeze(0)).clamp(0.0, 1.0)
        p_fn = (p_fn_sim * g_unc) * neg_mask_full.float()

        w_min = 0.1
        lambda_r = float(gate_val)
        w_neg = (1.0 - lambda_r * p_fn).clamp(min=w_min, max=1.0)

        rho = (p_fn > 0.5) & neg_mask_full
        eta = torch.zeros(N, N, dtype=torch.bool, device=device)

        fn_risk_anchor = (p_fn.sum(dim=1) / neg_mask_full.float().sum(dim=1).clamp(min=1.0))
        u_center = u_hat - u_hat.mean()
        fn_center = fn_risk_anchor - fn_risk_anchor.mean()
        denom = (u_center.norm() * fn_center.norm() + eps)
        corr_u_fn = (u_center * fn_center).sum() / denom

        safe_mask = neg_mask_full & (~rho)
        non_hn_safe_mask = safe_mask & (~eta)
        mean_s_post_fn = s_post[rho].mean().item() if rho.any() else 0.0
        mean_s_post_non_fn = s_post[safe_mask].mean().item() if safe_mask.any() else 0.0
        mean_sim_hn = sim[eta].mean().item() if eta.any() else 0.0
        mean_sim_safe_non_hn = sim[non_hn_safe_mask].mean().item() if non_hn_safe_mask.any() else 0.0

        exp_sim = torch.exp(sim)
        denom_all = (w_neg * exp_sim * neg_mask_full.float()).sum() + eps
        denom_fn = (w_neg * exp_sim * rho.float()).sum()
        denom_fn_share = (denom_fn / denom_all).item()

        fn_count = rho.float().sum().item()
        hn_count = 0.0
        neg_count = neg_mask_full.float().sum().item()
        safe_neg_count = safe_mask.float().sum().item()
        routed_candidate_neg_size = neg_mask_routed.float().sum().item()

        fn_count_per_anchor = rho.float().sum(dim=1)
        hn_count_per_anchor = torch.zeros(N, device=device)

        stats = {
            'fn_ratio': (fn_count / max(neg_count, 1.0)),
            'safe_ratio': (safe_neg_count / max(neg_count, 1.0)),
            'hn_ratio': 0.0,
            'FN_count': fn_count,
            'HN_count': hn_count,
            'neg_count': neg_count,
            'safe_neg_count': safe_neg_count,
            'candidate_neg_size': neg_count,
            'routed_candidate_neg_size': routed_candidate_neg_size,
            'routed_stat_neg_size': routed_candidate_neg_size,
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
            'w_hit_min_ratio': (((w_neg <= (w_min + 1e-5)) & neg_mask_full).float().sum() / max(neg_count, 1.0)).item(),
            'corr_u_fn': corr_u_fn.item(),
            'corr_u_fn_ratio': corr_u_fn.item(),
            'N_size': (neg_mask_full.float().sum(dim=1).mean().item()),
            'neg_per_anchor': (neg_mask_full.float().sum(dim=1).mean().item()),
            'U_size': int(uncertain_mask.sum().item()),
            'fn_pair_share': (fn_count / max(neg_count, 1.0)),
            'hn_pair_share': 0.0,
            'tau_fn_p10': torch.quantile(p_fn[neg_mask_full], 0.1).item() if neg_mask_full.any() else 0.0,
            'tau_fn_p50': torch.quantile(p_fn[neg_mask_full], 0.5).item() if neg_mask_full.any() else 0.0,
            'tau_fn_p90': torch.quantile(p_fn[neg_mask_full], 0.9).item() if neg_mask_full.any() else 0.0,
            'tau_hn_p10': 0.0,
            'tau_hn_p50': 0.0,
            'tau_hn_p90': 0.0,
            'FN_count_anchor_p50': torch.quantile(fn_count_per_anchor, 0.5).item() if fn_count_per_anchor.numel() > 0 else 0.0,
            'HN_count_anchor_p50': 0.0,
            'sigma_s': sigma_s,
            'w_min': w_min,
            'p_fn_thr': 0.5,
        }
        aux = {
            'S': S, 's_post': s_post, 'sim': sim, 'rho': rho, 'eta': eta, 'w_neg': w_neg,
            'r': r, 's_stab': s_stab, 'neg_mask': neg_mask_full, 'neg_mask_routed': neg_mask_routed,
            'tau_fn_per_anchor': fn_risk_anchor, 'tau_hn_per_anchor': torch.zeros(N, device=device),
            'FN_count_per_anchor': fn_count_per_anchor, 'HN_count_per_anchor': hn_count_per_anchor,
            'p_fn': p_fn, 'tau_sim_p99': tau,
        }

    return w_neg.detach(), eta.detach(), rho.detach(), stats, aux



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
                      hn_beta=0.1,
                      neg_mode='batch',
                      knn_neg_k=20,
                      route_uncertain_only=True,
                      fn_route_warmup_epochs=15,
                      feature_base_weight=1.0,
                      feature_route_weight=1.0,
                      y_prev_labels=None,
                      p_min=0.05,
                      lambda_cross=0.1,
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
    route_meter = {'fn_ratio':0.0,'safe_ratio':0.0,'hn_ratio':0.0,'mean_s_post_fn':0.0,'mean_s_post_non_fn':0.0,'delta_post':0.0,'mean_sim_hn':0.0,'mean_sim_safe_non_hn':0.0,'delta_sim':0.0,'label_flip':0.0,'stab_rate':0.0,'assignment_stability':0.0,'denom_fn_share':0.0,'denom_safe_share':0.0,'w_hit_min_ratio':0.0,'corr_u_fn':0.0,'corr_u_fn_ratio':0.0,'N_size':0.0,'U_size':0.0,'neg_per_anchor':0.0,'FN_count':0.0,'HN_count':0.0,'neg_count':0.0,'safe_neg_count':0.0,'candidate_neg_size':0.0,'routed_candidate_neg_size':0.0,'routed_stat_neg_size':0.0,'neg_after_filter_size':0.0,'neg_used_in_loss_size':0.0,'fn_pair_share':0.0,'hn_pair_share':0.0,'w_mean_on_FN':0.0,'w_mean_on_safe':0.0,'tau_fn_p10':0.0,'tau_fn_p50':0.0,'tau_fn_p90':0.0,'tau_hn_p10':0.0,'tau_hn_p50':0.0,'tau_hn_p90':0.0,'FN_count_anchor_p50':0.0,'HN_count_anchor_p50':0.0,'sigma_s':0.05,'w_min':0.1,'p_fn_thr':0.5,'U_ratio':0.0,'u_thr':0.0,'top_p_e':0.0,'k_unc':0.0}
    batch_count = 0
    last_dump = {}

    for batch_idx, (sub_data_views, _, sample_idx) in enumerate(mv_data_loader):
        # ——— 1) 伪标签 & 同/异样本矩阵 ———
        batch_psedo_label = model.psedo_labels[sample_idx]                # [N]
        y_matrix = (batch_psedo_label.unsqueeze(1) == batch_psedo_label.unsqueeze(0)).int()

        # ——— 2) 编码 + 融合 ———
        xrs, zs = model(sub_data_views)
        common_z = model.fusion(zs)

        # 数值稳定防护：若表征已出现 NaN/Inf，跳过该 batch 防止污染后续聚类与优化
        finite_ok = torch.isfinite(common_z).all()
        if finite_ok:
            for z_v in zs:
                if not torch.isfinite(z_v).all():
                    finite_ok = False
                    break
        if not finite_ok:
            optimizer.zero_grad(set_to_none=True)
            print(f"[Epoch {epoch} Batch {batch_idx}] non-finite embedding detected, skip batch")
            continue

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
        u_det = u.detach()
        u_thr_tensor = torch.quantile(u_det, 0.8)
        uncertain_mask = u_det >= u_thr_tensor
        if uncertain_mask.sum() == 0:
            uncertain_mask[torch.argmax(u_det)] = True
        k_unc = int(uncertain_mask.sum().item())
        idx_topk = uncertain_mask.nonzero(as_tuple=True)[0]
        certain_mask = ~uncertain_mask

        print(f"Batch {batch_idx}: uncertain {uncertain_mask.sum().item()}/{batch_N} = {uncertain_mask.sum().item()/batch_N:.2%}")

        # ——— 5) 动态门控 Gate ———
        t = float((epoch - 1) / max(1, max_epoch - 1))
        gate_u = t
        gate_fn = t
        gate_hn = t
        gate_val = t
        gate = torch.tensor(gate_val, device=device)

        # ——— 6) 计算共识中心 q_centers ———
        q_centers = model.compute_centers(common_z, batch_psedo_label)

        # ——— 7) Design 1': pair-wise FN 风险路由（停用原 FN/HN MLP 路径）———
        prev_batch = None if y_prev_labels is None else y_prev_labels[sample_idx].to(device)
        route_progress = max(0.0, float(epoch - fn_route_warmup_epochs) / 10.0)
        route_gate = min(1.0, route_progress)
        route_active = (route_gate > 0.0)
        route_mask = uncertain_mask if (route_uncertain_only and route_active) else None
        u_thr = u_det[idx_topk].min().item() if idx_topk.numel() > 0 else 0.0
        if route_active:
            # warmup 后渐进启用 FN 风险路由，避免启用时分母结构断崖变化
            w_neg, eta_mat, rho_mat, route_stats, route_aux = _build_pairwise_fn_risk(
                common_z=common_z,
                memberships_cons=memberships[num_views],
                u_hat=u_hat,
                batch_labels=batch_psedo_label,
                prev_labels_batch=prev_batch,
                gate_val=route_gate,
                alpha_fn=alpha_fn,
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
                'candidate_neg_size':neg_mask.float().sum().item(),'routed_candidate_neg_size':0.0,'routed_stat_neg_size':0.0,'neg_after_filter_size':neg_mask.float().sum().item(),
                'neg_used_in_loss_size':neg_mask.float().sum().item(),'mean_s_post_fn':0.0,'mean_s_post_non_fn':0.0,
                'delta_post':0.0,'mean_sim_hn':0.0,'mean_sim_safe_non_hn':0.0,'delta_sim':0.0,
                'label_flip':0.0,'stab_rate':0.0,'assignment_stability':0.0,'denom_fn_share':0.0,'denom_safe_share':1.0,
                'w_hit_min_ratio':0.0,'corr_u_fn':0.0,'corr_u_fn_ratio':0.0,'N_size':neg_mask.float().sum(dim=1).mean().item(),
                'neg_per_anchor':neg_mask.float().sum(dim=1).mean().item(),'U_size':int(uncertain_mask.sum().item()),
                'fn_pair_share':0.0,'hn_pair_share':0.0,'w_mean_on_FN':0.0,'w_mean_on_safe':1.0,
                'tau_fn_p10':0.0,'tau_fn_p50':0.0,'tau_fn_p90':0.0,'tau_hn_p10':0.0,'tau_hn_p50':0.0,'tau_hn_p90':0.0,
                'FN_count_anchor_p50':0.0,'HN_count_anchor_p50':0.0,'sigma_s':0.05,'w_min':0.1,'p_fn_thr':0.5,
            }
            route_aux = {
                'w_neg':w_neg,'neg_mask':neg_mask,'rho':rho_mat,'eta':eta_mat,
                'tau_fn_per_anchor':torch.zeros(batch_N, device=device),
                'tau_hn_per_anchor':torch.zeros(batch_N, device=device),
                'FN_count_per_anchor':torch.zeros(batch_N, device=device),
                'HN_count_per_anchor':torch.zeros(batch_N, device=device),
                'S':torch.zeros(batch_N, batch_N, device=device),
                'r':torch.zeros(batch_N, batch_N, device=device),
                's_post':torch.zeros(batch_N, batch_N, device=device),
                's_stab':torch.zeros(batch_N, batch_N, device=device),
                'sim':F.cosine_similarity(common_z.unsqueeze(1), common_z.unsqueeze(0), dim=2),
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
                if mvc_loss.ism_mode == 'legacy':
                    cl, _, _ = mvc_loss.compute_cluster_loss(
                        q_centers, k_centers, batch_psedo_label,
                        features_batch=common_z,
                        global_minority_mask=mask,
                        return_mmd_excl=True
                    )
                else:
                    cl = mvc_loss.compute_cluster_loss(
                        q_centers, k_centers, batch_psedo_label,
                        features_batch=common_z,
                        global_minority_mask=mask,
                        return_mmd_excl=False
                    )
            Lcl_i = alpha * cl
            Lcl += Lcl_i.item()
            loss_list.append(Lcl_i)

            # b) Feature loss：warmup 前仅基础 InfoNCE；routing 启用后切换到 routed InfoNCE。
            if route_active:
                feat_loss = mvc_loss.feature_loss(zs[v], common_z, Wv, y_pse, neg_weights=w_neg)
            else:
                feat_loss = mvc_loss.feature_loss(zs[v], common_z, Wv, y_pse, neg_weights=None)
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
            if epoch >= cross_warmup_epochs:
                cross_l = mvc_loss.cross_view_weighted_loss(
                    model, zs, common_z, memberships,
                    batch_psedo_label, temperature=temperature_f
                )
                cross_ramp = min(1.0, max(0.0, (epoch - cross_warmup_epochs) / float(max(1, cross_ramp_epochs))))
                cross_gate = cross_ramp
                Lcross_i = beta * lambda_cross * cross_gate * cross_l
                Lcross += Lcross_i.item()
                loss_list.append(Lcross_i)

            # f) 每个视图的重建损失
            recon_loss = criterion(sub_data_views[v], xrs[v])  # 计算每个视图的重建损失
            Lrecon += recon_loss.item()
            loss_list.append(recon_loss)  # 加入总损失

        # ——— 9) 梯度更新 & 打印 ———
        total_loss = sum(loss_list)
        if not torch.isfinite(total_loss):
            optimizer.zero_grad(set_to_none=True)
            print(f"[Epoch {epoch} Batch {batch_idx}] non-finite total_loss detected, skip backward")
            continue
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

    return {'loss': epoch_meter, 'route': route_meter, 'dump': last_dump, 'gate': gate_val if batch_count > 0 else 0.0, 'route_gate': route_gate if batch_count > 0 else 0.0, 'gate_u': gate_u if batch_count > 0 else 0.0, 'gate_fn': gate_fn if batch_count > 0 else 0.0, 'gate_hn': gate_hn if batch_count > 0 else 0.0, 't': t if batch_count > 0 else 0.0, 'warmup_epochs': warmup_epochs, 'cross_warmup_epochs': cross_warmup_epochs}




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
                                   hn_beta=0.1,
                                   neg_mode='batch',
                                   knn_neg_k=20,
                                   route_uncertain_only=True,
                                   fn_route_warmup_epochs=15,
                                   feature_base_weight=1.0,
                                   feature_route_weight=1.0,
                                   y_prev_labels=None,
                                   p_min=0.05,
                                   lambda_cross=0.1,
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
    route_meter = {'fn_ratio':0.0,'safe_ratio':0.0,'hn_ratio':0.0,'mean_s_post_fn':0.0,'mean_s_post_non_fn':0.0,'delta_post':0.0,'mean_sim_hn':0.0,'mean_sim_safe_non_hn':0.0,'delta_sim':0.0,'label_flip':0.0,'stab_rate':0.0,'assignment_stability':0.0,'denom_fn_share':0.0,'denom_safe_share':0.0,'w_hit_min_ratio':0.0,'corr_u_fn':0.0,'corr_u_fn_ratio':0.0,'N_size':0.0,'U_size':0.0,'neg_per_anchor':0.0,'FN_count':0.0,'HN_count':0.0,'neg_count':0.0,'safe_neg_count':0.0,'candidate_neg_size':0.0,'routed_candidate_neg_size':0.0,'routed_stat_neg_size':0.0,'neg_after_filter_size':0.0,'neg_used_in_loss_size':0.0,'fn_pair_share':0.0,'hn_pair_share':0.0,'w_mean_on_FN':0.0,'w_mean_on_safe':0.0,'tau_fn_p10':0.0,'tau_fn_p50':0.0,'tau_fn_p90':0.0,'tau_hn_p10':0.0,'tau_hn_p50':0.0,'tau_hn_p90':0.0,'FN_count_anchor_p50':0.0,'HN_count_anchor_p50':0.0,'sigma_s':0.05,'w_min':0.1,'p_fn_thr':0.5,'U_ratio':0.0,'u_thr':0.0,'top_p_e':0.0,'k_unc':0.0}
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

        finite_ok = torch.isfinite(common_z).all()
        if finite_ok:
            for z_v in zs:
                if not torch.isfinite(z_v).all():
                    finite_ok = False
                    break
        if not finite_ok:
            optimizer.zero_grad(set_to_none=True)
            print(f"[Epoch {epoch} Batch {batch_idx}] non-finite embedding detected, skip batch")
            continue

        # ——— 更新中心、隶属度、不确定度 ———
        model.update_centers(zs, common_z)
        feats = zs + [common_z]
        memberships = [model.compute_membership(feats[v], v) for v in range(num_views + 1)]
        u, u_hat = model.estimate_uncertainty(memberships, common_z)
        B = u_hat.size(0)

        # ——— 课程学习式不确定划分 ———
        u_det = u.detach()
        u_thr_tensor = torch.quantile(u_det, 0.8)
        uncertain = u_det >= u_thr_tensor
        if uncertain.sum() == 0:
            uncertain[torch.argmax(u_det)] = True
        k_unc = int(uncertain.sum().item())
        topk_idx = uncertain.nonzero(as_tuple=True)[0]
        certain = ~uncertain

        # ——— 动态门控 Gate ———
        t = float((epoch - 1) / max(1, max_epoch - 1))
        gate_u = t
        gate_fn = t
        gate_hn = t
        gate = t
        gate_t = torch.tensor(gate, device=device)

        # ——— 共识中心 ———
        q_centers = model.compute_centers(common_z, batch_label)

        # ——— Design 1': pair-wise FN 风险路由（停用原 FN/HN MLP 路径）———
        prev_batch = None if y_prev_labels is None else y_prev_labels[sample_idx].to(device)
        route_progress = max(0.0, float(epoch - fn_route_warmup_epochs) / 10.0)
        route_gate = min(1.0, route_progress)
        route_active = (route_gate > 0.0)
        route_mask = uncertain if (route_uncertain_only and route_active) else None
        u_thr = u_det[topk_idx].min().item() if topk_idx.numel() > 0 else 0.0
        if route_active:
            # warmup 后渐进启用 FN 风险路由，避免启用时分母结构断崖变化
            w_neg, eta_mat, rho_mat, route_stats, route_aux = _build_pairwise_fn_risk(
                common_z=common_z,
                memberships_cons=memberships[num_views],
                u_hat=u_hat,
                batch_labels=batch_label,
                prev_labels_batch=prev_batch,
                gate_val=route_gate,
                alpha_fn=alpha_fn,
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
                'candidate_neg_size':neg_mask.float().sum().item(),'routed_candidate_neg_size':0.0,'routed_stat_neg_size':0.0,'neg_after_filter_size':neg_mask.float().sum().item(),
                'neg_used_in_loss_size':neg_mask.float().sum().item(),'mean_s_post_fn':0.0,'mean_s_post_non_fn':0.0,
                'delta_post':0.0,'mean_sim_hn':0.0,'mean_sim_safe_non_hn':0.0,'delta_sim':0.0,
                'label_flip':0.0,'stab_rate':0.0,'assignment_stability':0.0,'denom_fn_share':0.0,'denom_safe_share':1.0,
                'w_hit_min_ratio':0.0,'corr_u_fn':0.0,'corr_u_fn_ratio':0.0,'N_size':neg_mask.float().sum(dim=1).mean().item(),
                'neg_per_anchor':neg_mask.float().sum(dim=1).mean().item(),'U_size':int(uncertain.sum().item()),
                'fn_pair_share':0.0,'hn_pair_share':0.0,'w_mean_on_FN':0.0,'w_mean_on_safe':1.0,
                'tau_fn_p10':0.0,'tau_fn_p50':0.0,'tau_fn_p90':0.0,'tau_hn_p10':0.0,'tau_hn_p50':0.0,'tau_hn_p90':0.0,
                'FN_count_anchor_p50':0.0,'HN_count_anchor_p50':0.0,'sigma_s':0.05,'w_min':0.1,'p_fn_thr':0.5,
            }
            route_aux = {
                'w_neg':w_neg,'neg_mask':neg_mask,'rho':rho_mat,'eta':eta_mat,
                'tau_fn_per_anchor':torch.zeros(B, device=device),
                'tau_hn_per_anchor':torch.zeros(B, device=device),
                'FN_count_per_anchor':torch.zeros(B, device=device),
                'HN_count_per_anchor':torch.zeros(B, device=device),
                'S':torch.zeros(B, B, device=device),
                'r':torch.zeros(B, B, device=device),
                's_post':torch.zeros(B, B, device=device),
                's_stab':torch.zeros(B, B, device=device),
                'sim':F.cosine_similarity(common_z.unsqueeze(1), common_z.unsqueeze(0), dim=2),
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
                if mvc_loss.ism_mode == 'legacy':
                    cl, _, _ = mvc_loss.compute_cluster_loss(
                        q_centers, kv_centers, batch_label,
                        features_batch=common_z,
                        global_minority_mask=mask,
                        return_mmd_excl=True
                    )
                else:
                    cl = mvc_loss.compute_cluster_loss(
                        q_centers, kv_centers, batch_label,
                        features_batch=common_z,
                        global_minority_mask=mask,
                        return_mmd_excl=False
                    )
            Lcl = alpha * cl
            batch_loss += Lcl

            # b) Feature loss：warmup 前仅基础 InfoNCE；routing 启用后切换到 routed InfoNCE。
            if route_active:
                feat_l = mvc_loss.feature_loss(zs[v], common_z, Wv, y_pse, neg_weights=w_neg)
            else:
                feat_l = mvc_loss.feature_loss(zs[v], common_z, Wv, y_pse, neg_weights=None)
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
            if epoch >= cross_warmup_epochs:
                cross_l = mvc_loss.cross_view_weighted_loss(
                    model, zs, common_z, memberships,
                    batch_label, temperature=temperature_f
                )
                cross_ramp = min(1.0, max(0.0, (epoch - cross_warmup_epochs) / float(max(1, cross_ramp_epochs))))
                cross_gate = prog * cross_ramp
                Lcross = beta * lambda_cross * cross_gate * cross_l
                batch_loss += Lcross

        # ——— 梯度更新 ———
        if not torch.isfinite(batch_loss):
            optimizer.zero_grad(set_to_none=True)
            print(f"[Epoch {epoch} Batch {batch_idx}] non-finite total_loss detected, skip backward")
            continue
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

    return {'loss': epoch_meter, 'route': route_meter, 'dump': last_dump, 'gate': gate if batch_count > 0 else 0.0, 'route_gate': route_gate if batch_count > 0 else 0.0, 'gate_u': gate_u if batch_count > 0 else 0.0, 'gate_fn': gate_fn if batch_count > 0 else 0.0, 'gate_hn': gate_hn if batch_count > 0 else 0.0, 't': t if batch_count > 0 else 0.0, 'warmup_epochs': warmup_epochs, 'cross_warmup_epochs': cross_warmup_epochs}
