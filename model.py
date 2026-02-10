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



def contrastive_train(model, mv_data, mvc_loss,
                      batch_size, epoch, W,
                      alpha, beta,
                      optimizer,
                      warmup_epochs,

                      lambda_u,  lambda_hn_penalty,
                      temperature_f, max_epoch=100,
                      initial_top_p=0.3):
    model.train()
    mv_data_loader, num_views, num_samples, num_clusters = get_multiview_data(mv_data, batch_size)

    # 将 all_features 和 all_labels 初始化为 Python 列表
    all_features = []  # 用于收集每个批次的特征
    all_labels = []  # 用于收集每个批次的标签

    # 课程学习式动态不确定比例
    top_p_e = initial_top_p * max(0.0, 1.0 - (epoch - 1) / float(max_epoch - 1))

    # E 步：更新全量伪标签
    psedo_labeling(model, mv_data, batch_size)

    # Push/Pull Lpen 超参
    lambda_push = lambda_hn_penalty
    lambda_pull = lambda_hn_penalty
    margin = 0.2

    criterion = torch.nn.MSELoss()  # 添加重建损失的损失函数

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
        k_unc = max(1, int(batch_N * top_p_e))
        _, idx_topk = torch.topk(u_hat, k_unc, largest=True)
        uncertain_mask = torch.zeros(batch_N, dtype=torch.bool, device=device)
        uncertain_mask[idx_topk] = True
        certain_mask = ~uncertain_mask

        print(f"Batch {batch_idx}: uncertain {uncertain_mask.sum().item()}/{batch_N} = {uncertain_mask.sum().item()/batch_N:.2%}")

        # ——— 5) 动态门控 Gate ———
        u_mean = u_hat.mean().item()
        mu_start, mu_end = 0.3, 0.7
        raw_gate = (u_mean - mu_start) / (mu_end - mu_start)
        gate_val = float(max(0.0, min(1.0, raw_gate)))
        gate = torch.tensor(gate_val, device=device)

        # ——— 6) 计算共识中心 q_centers ———
        q_centers = model.compute_centers(common_z, batch_psedo_label)

        # ——— 7) FN/HN 子网划分 ———
        res = model.classify_fn_hn(
            zs, common_z, memberships,
            batch_psedo_label, certain_mask, uncertain_mask,
            epoch, fn_hn_warmup=10
        )
        if res[0] is not None:
            logits, idx_kept, feats = res
            _, y_fn = logits.max(dim=1)
            fn_idx = idx_kept[y_fn == 1].to(device)
            hn_idx = idx_kept[y_fn == 0].to(device)
        else:
            logits = fn_idx = hn_idx = None

        # ——— 8) 累加各项损失 ———
        loss_list = []
        Lcl = Lfeat = Lu = Lpen = Lcross = Lrecon = 0.0  # 添加重建损失

        for v in range(num_views):
            # 准备 Wv 和 y_pse
            Wv = W[v][sample_idx][:, sample_idx].to(device)
            y_pse = y_matrix.float().to(device)
            if fn_idx is not None and fn_idx.numel() > 0:
                y_pse[fn_idx, :] = 0.1
                y_pse[:, fn_idx] = 0.1

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
            feat_loss = mvc_loss.feature_loss(zs[v], common_z, Wv, y_pse)
            Lfeat_i = beta * feat_loss
            Lfeat += Lfeat_i.item()
            loss_list.append(Lfeat_i)

            # c) 不确定度回归
            u_loss = mvc_loss.uncertainty_regression_loss(u_hat, u)
            Lu_i = (1 - gate_val) * lambda_u * u_loss
            Lu += Lu_i.item()
            loss_list.append(Lu_i)

            # d) Hard‐Negative Push & Pull Loss
            if hn_idx is not None and hn_idx.numel() > 0:
                # Hard Negative 现表示
                z_hn = common_z[hn_idx]  # (K, d)
                # 它们的正负簇中心（这里用自身簇中心作为负中心）
                pos_ctr = model.centers[num_views].to(device)[batch_psedo_label[hn_idx]]
                neg_ctr = pos_ctr  # 或者用全局负中心

                sim_pos = F.cosine_similarity(z_hn, pos_ctr, dim=1)  # (K,)
                sim_neg = F.cosine_similarity(z_hn, neg_ctr, dim=1)  # (K,)

                # push: sim_pos - sim_neg + margin <= 0
                push_loss = torch.relu(sim_pos - sim_neg + margin).mean()
                # pull: 1 - sim_neg <= 0
                pull_loss = (1.0 - sim_neg).mean()

                Lpen_i = gate_val * (lambda_push * push_loss + lambda_pull * pull_loss)
            else:
                Lpen_i = torch.tensor(0.0, device=device)

            Lpen += Lpen_i.item()
            loss_list.append(Lpen_i)

            # e) 跨视图加权 InfoNCE
            if epoch > warmup_epochs:
                cross_l = mvc_loss.cross_view_weighted_loss(
                    model, zs, common_z, memberships,
                    batch_psedo_label, temperature=temperature_f
                )
                Lcross_i = gate_val * beta  * cross_l
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

        print(f"[Epoch {epoch} Batch {batch_idx}] "
              f"Total={total_loss.item():.4f}  "
              )

    # ===== 训练循环结束 =====
    return total_loss




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
                                   initial_top_p=0.3):
    """
    大数据集版 Contrastive Training：
    - k: 用于构建每个视图下的 k-NN 图
    - 其它参数含义同原版 contrastive_train
    """
    model.train()
    mv_loader, num_views, _, num_clusters = get_multiview_data(mv_data, batch_size)
    criterion = torch.nn.MSELoss()
    total_loss = 0.0

    # 1) 课程学习式动态不确定比例
    top_p = initial_top_p * max(0.0, 1.0 - (epoch - 1) / float(max_epoch - 1))

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
        k_unc = max(1, int(B * top_p))
        _, topk_idx = torch.topk(u_hat, k_unc, largest=True)
        uncertain = torch.zeros(B, dtype=torch.bool, device=device)
        uncertain[topk_idx] = True
        certain = ~uncertain

        # ——— 动态门控 Gate ———
        u_mean = u_hat.mean().item()
        mu_lo, mu_hi = 0.3, 0.7
        gate = float((u_mean - mu_lo) / (mu_hi - mu_lo))
        gate = max(0.0, min(1.0, gate))
        gate_t = torch.tensor(gate, device=device)

        # ——— 共识中心 ———
        q_centers = model.compute_centers(common_z, batch_label)

        # ——— FN/HN 子网划分 ———
        res = model.classify_fn_hn(
            zs, common_z, memberships,
            batch_label, certain, uncertain,
            epoch, fn_hn_warmup=10
        )
        if res[0] is not None:
            logits, kept_idx, _ = res
            _, y_fn = logits.max(dim=1)
            fn_idx = kept_idx[y_fn == 1].to(device)
            hn_idx = kept_idx[y_fn == 0].to(device)
        else:
            fn_idx = hn_idx = None

        # ——— 构造并累加各视图的损失 ———
        batch_loss = 0.0
        for v in range(num_views):
            # 动态 k-NN Graph
            Wv = get_knn_graph(sub_views[v], k).to(device)
            y_pse = y_matrix.float()
            if fn_idx is not None and fn_idx.numel() > 0:
                y_pse[fn_idx, :] = 0.1
                y_pse[:, fn_idx] = 0.1

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
            feat_l = mvc_loss.feature_loss(zs[v], common_z, Wv, y_pse)
            Lfeat = beta * feat_l
            batch_loss += Lfeat

            # c) 不确定度回归
            u_l = mvc_loss.uncertainty_regression_loss(u_hat, u)
            Lu = (1 - gate) * lambda_u * u_l
            batch_loss += Lu

            # d) Hard‐Negative Push/Pull
            if hn_idx is not None and hn_idx.numel() > 0:
                z_hn = common_z[hn_idx]
                pos_ctr = model.centers[num_views][batch_label[hn_idx]].to(device)
                neg_ctr = pos_ctr
                sim_pos = F.cosine_similarity(z_hn, pos_ctr, dim=1)
                sim_neg = F.cosine_similarity(z_hn, neg_ctr, dim=1)
                push = torch.relu(sim_pos - sim_neg + 0.2).mean()
                pull = (1.0 - sim_neg).mean()
                Lpen = gate * (lambda_hn_penalty * push + lambda_hn_penalty * pull)
            else:
                Lpen = torch.tensor(0.0, device=device)
            batch_loss += Lpen

            # e) 跨视图加权 InfoNCE
            if epoch > warmup_epochs:
                cross_l = mvc_loss.cross_view_weighted_loss(
                    model, zs, common_z, memberships,
                    batch_label, temperature=temperature_f
                )
                Lcross = gate * beta * prog * cross_l
                batch_loss += Lcross

        # ——— 梯度更新 ———
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()

    # 每 5 个 epoch 打印一次
    if epoch % 5 == 0:
        print(f"[Epoch {epoch}] total loss: {total_loss:.7f}")

    return total_loss
