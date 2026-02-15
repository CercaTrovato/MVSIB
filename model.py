from dataloader import *
import torch
import torch.nn.functional as F
import numpy as np


def get_knn_graph(data, k):
    num_samples = data.size(0)
    graph = torch.zeros(num_samples, num_samples, dtype=torch.int32, device=data.device)
    for i in range(num_samples):
        distance = torch.sum((data - data[i]) ** 2, dim=1)
        _, small_indices = torch.topk(distance, k, largest=False)
        graph[i, small_indices[1:]] = 1
    return torch.max(graph, graph.t())


def get_W(mv_data, k):
    W = []
    mv_data_loader, num_views, _, _ = get_all_multiview_data(mv_data)
    for _, (sub_data_views, _, _) in enumerate(mv_data_loader):
        for i in range(num_views):
            W.append(get_knn_graph(sub_data_views[i], k))
    return W


def _collect_common_z(model, dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    common_list = []
    for xs, _, _ in loader:
        with torch.no_grad():
            _, zs = model(xs)
            common_list.append(model.fusion(zs))
    return torch.cat(common_list, dim=0)


def _cache_consensus_centers(model, common_z, labels):
    # E-step后使用全量common_z和labels缓存共识中心，避免batch级中心抖动。
    with torch.no_grad():
        centers = model.compute_centers(common_z, labels)
        model.centers[model.num_views] = F.normalize(centers, p=2, dim=1)


def psedo_labeling(model, dataset, batch_size):
    common_z = _collect_common_z(model, dataset, batch_size)
    labels = model.clustering(common_z)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    labels = labels.to(model.psedo_labels.device).long()
    model.psedo_labels = labels
    _cache_consensus_centers(model, common_z.to(model.psedo_labels.device), labels)
    return common_z, labels


def pre_train(model, mv_data, batch_size, epochs, optimizer):
    mv_data_loader, num_views, _, _ = get_multiview_data(mv_data, batch_size)
    criterion = torch.nn.MSELoss()
    hist = np.zeros(epochs + 1, dtype=np.float64)
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for sub_data_views, _, _ in mv_data_loader:
            xrs, _ = model(sub_data_views)
            loss = sum(criterion(sub_data_views[idx], xrs[idx]) for idx in range(num_views))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        hist[epoch] = total_loss
        if epoch % 10 == 0 or epoch == epochs:
            print(f'Pre-training, epoch {epoch}, Loss:{total_loss:.7f}')
    return hist


def _fit_two_gaussian(s_vals, iters=12, eps=1e-6):
    if s_vals.numel() < 8:
        mu = s_vals.mean() if s_vals.numel() > 0 else torch.tensor(0.0, device=s_vals.device)
        std = s_vals.std(unbiased=False).clamp(min=0.1) if s_vals.numel() > 1 else torch.tensor(0.1, device=s_vals.device)
        return mu - 0.2 * std, std, mu + 0.2 * std, std
    q30 = torch.quantile(s_vals, 0.3)
    q70 = torch.quantile(s_vals, 0.7)
    mu1, mu2 = q30.clone(), q70.clone()
    std1 = s_vals.std(unbiased=False).clamp(min=0.05)
    std2 = std1.clone()
    pi1 = torch.tensor(0.5, device=s_vals.device)
    for _ in range(iters):
        n1 = torch.exp(-0.5 * ((s_vals - mu1) / (std1 + eps)) ** 2) / (std1 + eps)
        n2 = torch.exp(-0.5 * ((s_vals - mu2) / (std2 + eps)) ** 2) / (std2 + eps)
        r1 = pi1 * n1
        r2 = (1.0 - pi1) * n2
        den = (r1 + r2).clamp(min=eps)
        g1 = r1 / den
        g2 = r2 / den
        w1 = g1.sum().clamp(min=eps)
        w2 = g2.sum().clamp(min=eps)
        mu1 = (g1 * s_vals).sum() / w1
        mu2 = (g2 * s_vals).sum() / w2
        std1 = torch.sqrt(((g1 * (s_vals - mu1) ** 2).sum() / w1).clamp(min=0.0025))
        std2 = torch.sqrt(((g2 * (s_vals - mu2) ** 2).sum() / w2).clamp(min=0.0025))
        pi1 = (w1 / (w1 + w2)).clamp(0.05, 0.95)
    if mu1 <= mu2:
        return mu1, std1, mu2, std2
    return mu2, std2, mu1, std1


def _route_fn_hn(common_z, batch_labels, centers, uncertain_mask, u, tau_fn, sigma_t, gamma, tau_hn, sigma_hn, z0, zs, eps=1e-6):
    n = common_z.size(0)
    device = common_z.device
    sim = F.cosine_similarity(common_z.unsqueeze(1), common_z.unsqueeze(0), dim=2)
    eye = torch.eye(n, dtype=torch.bool, device=device)
    neg_mask = (~eye) & (batch_labels.unsqueeze(1) != batch_labels.unsqueeze(0))
    route_mask = neg_mask & uncertain_mask.unsqueeze(1)

    y_i, y_j = batch_labels.unsqueeze(1), batch_labels.unsqueeze(0)
    ci_yj = centers[y_j]
    cj_yi = centers[y_i]
    t_ij = torch.maximum(F.cosine_similarity(common_z.unsqueeze(1), ci_yj, dim=2),
                         F.cosine_similarity(common_z.unsqueeze(0), cj_yi, dim=2))

    p_fn = torch.zeros_like(sim)
    p_hn = torch.zeros_like(sim)
    post_high = torch.zeros_like(sim)

    if route_mask.any():
        s_vals = sim[route_mask]
        mu_low, std_low, mu_high, std_high = _fit_two_gaussian(s_vals)
        n_low = torch.exp(-0.5 * ((s_vals - mu_low) / (std_low + eps)) ** 2) / (std_low + eps)
        n_high = torch.exp(-0.5 * ((s_vals - mu_high) / (std_high + eps)) ** 2) / (std_high + eps)
        post = n_high / (n_low + n_high + eps)
        post_high[route_mask] = post

        u_anchor = (u.unsqueeze(1).expand_as(sim))[route_mask]
        t_vals = t_ij[route_mask]
        gate_t_fn = torch.sigmoid((t_vals - tau_fn) / sigma_t)
        p_fn_vals = post * gate_t_fn * (u_anchor ** gamma)
        p_fn[route_mask] = p_fn_vals

        z_tn = (s_vals - mu_low) / (std_low + eps)
        gate_s = torch.sigmoid((z_tn - z0) / zs)
        gate_t_hn = torch.sigmoid((tau_hn - t_vals) / sigma_hn)
        p_hn_vals = (1.0 - p_fn_vals) * gate_s * gate_t_hn * (u_anchor ** gamma)
        p_hn[route_mask] = p_hn_vals

    return {
        'sim': sim,
        'neg_mask': neg_mask,
        'route_mask': route_mask,
        't_ij': t_ij,
        'post_high': post_high,
        'p_fn': p_fn * route_mask.float(),
        'p_hn': p_hn * route_mask.float(),
    }


def contrastive_train(model, mv_data, mvc_loss,
                      batch_size, epoch, W,
                      alpha, beta,
                      optimizer,
                      warmup_epochs,
                      lambda_u, lambda_hn_penalty,
                      temperature_f, max_epoch=100,
                      initial_top_p=0.3,
                      p_min=0.05,
                      uncert_decay_epochs=20,
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
                      lambda_cross=0.1,
                      cross_ramp_epochs=10,
                      fn_prob_tau=0.5,
                      tail_s_cap=0.5,
                      tail_beta=4.0,
                      route_uncertain_only_train_applied=True,
                      uncert_kappa_init_q=0.8,
                      sigma_u=0.1,
                      sigma_t=0.1,
                      gamma_u=2.0,
                      tau_hn=0.2,
                      sigma_hn=0.1,
                      hn_z0=0.0,
                      hn_zs=1.0,
                      tau_pos=0.5,
                      hn_margin=0.2,
                      lambda_fn_attr=0.1,
                      lambda_hn_margin=0.1):
    model.train()
    mv_data_loader, num_views, _, _ = get_multiview_data(mv_data, batch_size)

    common_all, labels_all = psedo_labeling(model, mv_data, batch_size)
    common_all = common_all.to(model.psedo_labels.device)
    delta_all = model.compute_prototype_margin(common_all)
    model.init_uncertainty_kappa(delta_all, uncert_kappa_init_q, epoch)

    criterion = torch.nn.MSELoss()
    epoch_meter = {'L_total': 0.0, 'L_recon': 0.0, 'L_feat': 0.0, 'L_cross': 0.0, 'L_cluster': 0.0,
                   'L_uncert': 0.0, 'L_hn': 0.0, 'L_reg': 0.0, 'L_fn_attr': 0.0, 'L_hn_margin': 0.0}
    route_meter = {'fn_ratio': 0.0, 'safe_ratio': 0.0, 'hn_ratio': 0.0, 'FN_count': 0.0, 'HN_count': 0.0,
                   'neg_count': 0.0, 'safe_neg_count': 0.0, 'U_size': 0.0, 'neg_used_in_loss_size': 0.0,
                   'delta_p50': 0.0, 'kappa': 0.0}
    batch_count = 0
    last_dump = {}

    for sub_data_views, _, sample_idx in mv_data_loader:
        batch_labels = model.psedo_labels[sample_idx].to(model.psedo_labels.device)
        xrs, zs = model(sub_data_views)
        common_z = model.fusion(zs)

        # 共识中心来自E-step缓存，禁止batch内中心更新。
        centers = model.centers[num_views].to(common_z.device)
        u, u_hat, delta = model.estimate_uncertainty(common_z, sigma_u=sigma_u)
        uncertain_mask = delta < model.uncertain_kappa

        route = _route_fn_hn(
            common_z=common_z,
            batch_labels=batch_labels,
            centers=centers,
            uncertain_mask=uncertain_mask,
            u=u,
            tau_fn=fn_prob_tau,
            sigma_t=sigma_t,
            gamma=gamma_u,
            tau_hn=tau_hn,
            sigma_hn=sigma_hn,
            z0=hn_z0,
            zs=hn_zs,
        )

        bias = (-alpha_fn * route['p_fn'] + hn_beta * route['p_hn']) * route['route_mask'].float()

        y_mat = (batch_labels.unsqueeze(1) == batch_labels.unsqueeze(0)).int()
        q_centers = model.compute_centers(common_z, batch_labels)
        batch_loss = 0.0
        L_recon, L_cluster, L_feat, L_uncert, L_hn, L_cross = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for v in range(num_views):
            Wv = get_knn_graph(sub_data_views[v], k=5).to(common_z.device)
            kv_centers = model.compute_centers(zs[v], batch_labels)
            cl = mvc_loss.compute_cluster_loss(q_centers, kv_centers, batch_labels)
            feat_l = mvc_loss.feature_loss(zs[v], common_z, Wv, y_mat.float(), neg_bias=bias)
            rec_l = criterion(sub_data_views[v], xrs[v])
            u_l = mvc_loss.uncertainty_regression_loss(u_hat, u)
            L_cluster += alpha * cl
            L_feat += beta * feat_l
            L_recon += rec_l
            L_uncert += lambda_u * u_l

        L_fn_attr = mvc_loss.fn_attraction_loss(route['sim'], route['p_fn'], route['route_mask'], tau_pos=tau_pos)
        L_hn_margin = mvc_loss.hn_prototype_margin_loss(common_z, batch_labels, centers, u, route['p_hn'], margin=hn_margin)
        L_hn = lambda_fn_attr * L_fn_attr + lambda_hn_margin * L_hn_margin
        batch_loss = L_cluster + L_feat + L_recon + L_uncert + L_hn + L_cross

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        neg_cnt = route['neg_mask'].float().sum().item()
        routed_cnt = route['route_mask'].float().sum().item()
        fn_cnt = route['p_fn'].sum().item()
        hn_cnt = route['p_hn'].sum().item()
        epoch_meter['L_total'] += batch_loss.item()
        epoch_meter['L_recon'] += float(L_recon)
        epoch_meter['L_feat'] += float(L_feat)
        epoch_meter['L_cluster'] += float(L_cluster)
        epoch_meter['L_uncert'] += float(L_uncert)
        epoch_meter['L_hn'] += float(L_hn)
        epoch_meter['L_cross'] += float(L_cross)
        epoch_meter['L_fn_attr'] += float(L_fn_attr)
        epoch_meter['L_hn_margin'] += float(L_hn_margin)

        route_meter['fn_ratio'] += fn_cnt / max(routed_cnt, 1.0)
        route_meter['hn_ratio'] += hn_cnt / max(routed_cnt, 1.0)
        route_meter['safe_ratio'] += max(routed_cnt - fn_cnt, 0.0) / max(routed_cnt, 1.0)
        route_meter['FN_count'] += fn_cnt
        route_meter['HN_count'] += hn_cnt
        route_meter['neg_count'] += neg_cnt
        route_meter['safe_neg_count'] += max(neg_cnt - fn_cnt, 0.0)
        route_meter['U_size'] += float(uncertain_mask.sum().item())
        route_meter['neg_used_in_loss_size'] += neg_cnt
        route_meter['delta_p50'] += float(torch.quantile(delta.detach(), 0.5).item())
        route_meter['kappa'] += float(model.uncertain_kappa.item())
        batch_count += 1

        last_dump = {
            'u_sample': u.detach().cpu(),
            'delta_sample': delta.detach().cpu(),
            'sim_neg_sample': route['sim'][route['neg_mask']].detach().cpu(),
            'rho_fn_pair_sample': route['p_fn'][route['neg_mask']].detach().cpu(),
            'eta_hn_pair_sample': route['p_hn'][route['neg_mask']].detach().cpu(),
            'uncertain_mask_sample': uncertain_mask.detach().cpu(),
            'route_mask_sample': route['route_mask'].detach().cpu(),
        }

    if batch_count > 0:
        for k in epoch_meter:
            epoch_meter[k] /= batch_count
        for k in route_meter:
            route_meter[k] /= batch_count

    return {
        'loss': epoch_meter,
        'route': route_meter,
        'dump': last_dump,
        'gate': 1.0,
        'route_gate': 1.0,
        'gate_u': 1.0,
        'gate_fn': 1.0,
        'gate_hn': 1.0,
        't': 1.0,
        'warmup_epochs': warmup_epochs,
        'cross_warmup_epochs': cross_warmup_epochs,
    }
