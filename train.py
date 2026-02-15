import argparse
import random
from network import Network
from metric import *
from model import *
from loss import *
from logger import Logger
import datetime
import os
import torch
import numpy as np


Dataname = 'RGB-D'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname,
                    help='[CCV, RGB-D, Cora, Hdigit, prokaryotic]')
parser.add_argument('--save_model', default=False, help='Saving the model after training.')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5, type=float)
parser.add_argument("--temperature_l", default=0.5, type=float)
parser.add_argument("--learning_rate", default=0.0001, type=float)
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--mse_epochs", default=200, type=int)
parser.add_argument("--con_epochs", default=100, type=int)
parser.add_argument("--feature_dim", default =256, type=int)
parser.add_argument("--large_datasets", default=False, type=lambda x: x.lower()=='true')
parser.add_argument("--k", default=5, type=int)
parser.add_argument('--gpu', default='0', type=str, help='GPU device idx.')
# 以下是我们新增的动态策略超参
parser.add_argument('--warmup_epochs', default=20, type=int)
parser.add_argument('--lambda_u', default=0.1, type=float)
parser.add_argument('--lambda_hn_penalty',type=float,default=0.1)
parser.add_argument('--cross_warmup_epochs', default=50, type=int,
                    help='Epoch to start cross-view weighted consistency loss (Stage-3).')
parser.add_argument('--cross_ramp_epochs', default=10, type=int,
                    help='Ramp epochs for cross-view loss after cross warmup.')
parser.add_argument('--lambda_cross', default=0.1, type=float,
                    help='Extra global weight for cross-view loss.')
parser.add_argument('--membership_mode', default='softmax_distance', type=str,
                    choices=['gaussian', 'softmax_distance'],
                    help='Membership kernel mode: paper-improved softmax_distance or legacy gaussian.')
parser.add_argument('--uncertainty_mode', default='entropy', type=str,
                    choices=['legacy', 'entropy'],
                    help='Uncertainty mode: legacy entropy/top2 or improved normalized-entropy uncertainty.')
parser.add_argument('--neg_mode', default='batch', type=str, choices=['batch', 'knn'],
                    help='Negative candidate mode for pair-wise FN risk routing.')
parser.add_argument('--knn_neg_k', default=20, type=int,
                    help='k in kNN negatives when neg_mode=knn.')
parser.add_argument('--alpha_fn', default=0.1, type=float,
                    help='Top-risk quantile ratio for FN-risk negatives.')
parser.add_argument('--hn_beta', default=0.1, type=float,
                    help='Hard-negative quantile in safe negatives.')
parser.add_argument('--route_uncertain_only', default=True, type=lambda x: x.lower()=='true',
                    help='Apply pair-wise routing only for uncertain anchors.')
parser.add_argument('--fn_route_warmup_epochs', default=15, type=int,
                    help='Warmup epochs before enabling FN-risk routing; use full contrastive negatives before this stage.')
parser.add_argument('--feature_base_weight', default=1.0, type=float,
                    help='Weight for baseline feature InfoNCE on all anchors.')
parser.add_argument('--feature_route_weight', default=1.0, type=float,
                    help='Weight for routed feature InfoNCE with FN-risk negative weights.')
parser.add_argument('--log_dist_interval', default=5, type=int,
                    help='Epoch interval for DISTR summary and debug dump.')
parser.add_argument('--save_debug_npz', default=True, type=lambda x: x.lower()=='true',
                    help='Save debug npz dump periodically.')
parser.add_argument('--debug_dir', default='debug', type=str,
                    help='Directory to save debug npz files.')
parser.add_argument('--ism_mode', default='improved', type=str, choices=['legacy', 'improved'],
                    help='ISM implementation: legacy(GMM+MMD+exclusion) or improved(prior+manifold+moment+coverage).')
parser.add_argument('--ism_prior_weight', default=0.1, type=float,
                    help='Weight of Dirichlet prior barrier in improved ISM.')
parser.add_argument('--ism_dirichlet_alpha_eps', default=0.1, type=float,
                    help='Epsilon in alpha=1+eps for Dirichlet barrier.')
parser.add_argument('--ism_align_weight', default=0.5, type=float,
                    help='Weight of moment alignment (mean + variance) in improved ISM.')
parser.add_argument('--ism_cov_weight', default=0.1, type=float,
                    help='Weight of coverage regularizer in improved ISM.')
parser.add_argument('--ism_align_var_weight', default=0.1, type=float,
                    help='Variance term weight inside moment alignment.')
parser.add_argument('--ism_small_cluster_ratio', default=0.5, type=float,
                    help='Quantile ratio to identify small clusters in improved ISM.')
parser.add_argument('--ism_pseudo_conf_q', default=0.8, type=float,
                    help='Confidence quantile for selecting pseudo anchors in improved ISM.')
parser.add_argument('--ism_local_knn_k', default=5, type=int,
                    help='Reserved local scale factor for lightweight manifold pseudo sampling.')
parser.add_argument('--ism_cov_shrink', default=0.3, type=float,
                    help='Diagonal variance shrinkage ratio for pseudo sampling.')
parser.add_argument('--ism_pseudo_noise_beta', default=0.5, type=float,
                    help='Noise scale beta for pseudo sampling.')
parser.add_argument('--ism_manifold_radius_q', default=0.8, type=float,
                    help='Radius quantile for manifold-consistency filtering.')
parser.add_argument('--ism_cov_radius_min', default=0.5, type=float,
                    help='Lower radius factor for coverage regularizer.')
parser.add_argument('--ism_cov_radius_max', default=1.5, type=float,
                    help='Upper radius factor for coverage regularizer.')
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def _qstats(x):
    x = np.asarray(x).reshape(-1)
    if x.size == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    return (float(np.quantile(x, 0.1)), float(np.quantile(x, 0.5)), float(np.quantile(x, 0.9)), float(np.mean(x)), float(np.std(x)))

def _to_np(v):
    if hasattr(v, 'detach'):
        return v.detach().cpu().numpy()
    return np.asarray(v)

def _rget(route, key, default=0.0):
    return route[key] if key in route else default


def _route_with_dump_fallback(route, dump_dict):
    # Keep ROUTE counters self-consistent when epoch aggregation keys are missing.
    r = dict(route) if route is not None else {}
    fn_ratio = float(_rget(r, 'fn_ratio', 0.0))
    hn_ratio = float(_rget(r, 'hn_ratio', 0.0))
    fn_count = float(_rget(r, 'FN_count', 0.0))
    hn_count = float(_rget(r, 'HN_count', 0.0))
    neg_count = float(_rget(r, 'neg_count', 0.0))
    safe_neg_count = float(_rget(r, 'safe_neg_count', 0.0))
    inconsistent = ((fn_ratio > 1e-12 and (fn_count <= 0.0 or neg_count <= 0.0)) or
                    (hn_ratio > 1e-12 and (hn_count <= 0.0 or safe_neg_count <= 0.0)))
    if inconsistent and dump_dict:
        rho = _to_np(dump_dict.get('rho_fn_pair_sample', np.array([]))).reshape(-1)
        eta = _to_np(dump_dict.get('eta_hn_pair_sample', np.array([]))).reshape(-1)
        if rho.size > 0:
            neg_count = float(rho.size)
            fn_count = float(np.nansum(rho.astype(np.float32)))
            safe_neg_count = max(neg_count - fn_count, 0.0)
            hn_count = float(np.nansum(eta.astype(np.float32))) if eta.size == rho.size else float(hn_count)
            r['neg_count'] = neg_count
            r['FN_count'] = fn_count
            r['safe_neg_count'] = safe_neg_count
            r['HN_count'] = min(hn_count, safe_neg_count)
            r['fn_pair_share'] = fn_count / max(neg_count, 1.0)
            r['hn_pair_share'] = r['HN_count'] / max(safe_neg_count, 1.0)
    r['route_count_inconsistent'] = int(((fn_ratio > 1e-12 and (float(_rget(r, 'FN_count', 0.0)) <= 0.0 or float(_rget(r, 'neg_count', 0.0)) <= 0.0)) or
                                         (hn_ratio > 1e-12 and (float(_rget(r, 'HN_count', 0.0)) <= 0.0 or float(_rget(r, 'safe_neg_count', 0.0)) <= 0.0))))
    return r

def _save_debug_npz(debug_path, dump_dict, cluster_sizes, empty_cluster_count, min_cluster_size, gate_value, loss_dict):
    os.makedirs(os.path.dirname(debug_path), exist_ok=True)
    arrays = {}
    for k, v in dump_dict.items():
        arr = _to_np(v)
        arrays[k] = arr
    arrays.update({
        'cluster_sizes': np.asarray(cluster_sizes),
        'empty_cluster_count': np.asarray(empty_cluster_count),
        'min_cluster_size': np.asarray(min_cluster_size),
        'gate_value': np.asarray(gate_value),
    })
    for lk, lv in loss_dict.items():
        arrays[lk] = np.asarray(lv)
    np.savez_compressed(debug_path, **arrays)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # — 数据集特定超参 —
    if args.dataset == "CCV":
        args.seed, args.k, alpha, beta = 10, 10, 0.0001, 0.001
        args.seed, args.k, alpha, beta = 10, 4, 0.01, 0.1
    elif args.dataset == "RGB-D":
        args.seed, args.k, alpha, beta = 5, 10, 0.01, 1
    elif args.dataset == "Cora":
        args.seed, args.k, alpha, beta = 10, 10, 0.01, 0.1
        args.con_epochs = 100
    elif args.dataset == "Hdigit":
        args.large_datasets = True
        args.seed, args.k, alpha, beta = 10, 5, 1, 0.1
    elif args.dataset == "prokaryotic":
        args.seed, args.k, alpha, beta = 10, 5, 0.01, 0.1

    print("==================================\nArgs:{}\n==================================".format(args))
    set_seed(args.seed)
    # — 准备数据和模型 —
    mv_data = MultiviewData(args.dataset, device)
    num_views = len(mv_data.data_views)
    num_samples = mv_data.labels.size
    num_clusters = int(np.unique(mv_data.labels).size)
    input_sizes = [mv_data.data_views[i].shape[1] for i in range(num_views)]

    network = Network(
        num_views, num_samples, num_clusters, device,
        input_sizes, args.feature_dim,
        membership_mode=args.membership_mode,
        uncertainty_mode=args.uncertainty_mode,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(network.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    mvc_loss = Loss(
        args.batch_size, num_clusters,
        args.temperature_l, args.temperature_f,
        ism_mode=args.ism_mode,
        prior_weight=args.ism_prior_weight,
        dirichlet_alpha_eps=args.ism_dirichlet_alpha_eps,
        align_weight=args.ism_align_weight,
        cov_weight=args.ism_cov_weight,
        align_var_weight=args.ism_align_var_weight,
        small_cluster_ratio=args.ism_small_cluster_ratio,
        pseudo_conf_quantile=args.ism_pseudo_conf_q,
        local_knn_k=args.ism_local_knn_k,
        cov_shrink=args.ism_cov_shrink,
        pseudo_noise_beta=args.ism_pseudo_noise_beta,
        manifold_radius_quantile=args.ism_manifold_radius_q,
        cov_radius_min=args.ism_cov_radius_min,
        cov_radius_max=args.ism_cov_radius_max,
    ).to(device)

    nowtime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logger = Logger(f"{args.dataset}=={nowtime}")
    logger.info("Args: " + str(args))

    # — Warm-up 预训练 —
    epoch_list = []
    totalloss_list = []
    pre_train(network, mv_data, args.batch_size,
              args.mse_epochs, optimizer)

    best_acc = 0
    best_epoch = -1
    best_metrics = None
    acc_list, nmi_list, pur_list, ari_list, f1_list = [], [], [], [], []

    if not args.large_datasets:
        W = get_W(mv_data, k=args.k)
        mv_loader, _, _, _ = get_multiview_data(mv_data, args.batch_size)
        y_prev = None

        for epoch in range(1, args.con_epochs + 1):
            y_prev = network.psedo_labels.clone() if epoch > 1 else None
            train_out = contrastive_train(
                network, mv_data, mvc_loss,
                args.batch_size, epoch, W,
                alpha, beta,
                optimizer,
                args.warmup_epochs,
                args.lambda_u,
                args.lambda_hn_penalty,
                args.temperature_f,
                cross_warmup_epochs=args.cross_warmup_epochs,
                cross_ramp_epochs=args.cross_ramp_epochs,
                lambda_cross=args.lambda_cross,
                alpha_fn=args.alpha_fn,
                hn_beta=args.hn_beta,
                neg_mode=args.neg_mode,
                knn_neg_k=args.knn_neg_k,
                route_uncertain_only=args.route_uncertain_only,
                fn_route_warmup_epochs=args.fn_route_warmup_epochs,
                feature_base_weight=args.feature_base_weight,
                feature_route_weight=args.feature_route_weight,
                y_prev_labels=y_prev,
            )

            epoch_list.append(epoch)
            totalloss_list.append(train_out['loss']['L_total'])

            # 每轮评估
            acc, nmi, pur, ari, f_score = valid(network, mv_data, num_samples, num_clusters)
            logger.info(f"ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f} F1={f_score:.4f}")
            print(f"[Epoch {epoch}] ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f} F1={f_score:.4f}")

            lr = optimizer.param_groups[0]['lr']
            L = train_out['loss']
            R = _route_with_dump_fallback(train_out['route'], train_out.get('dump', {}))
            metric_line = (
                f"METRIC: epoch={epoch} step={epoch} ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f} F1={f_score:.4f} "
                f"gate={train_out['gate']:.4f} lr={lr:.6g} temp_f={args.temperature_f:.4f} temp_l={args.temperature_l:.4f} "
                f"L_total={L['L_total']:.6f} L_recon={L['L_recon']:.6f} L_feat={L['L_feat']:.6f} L_cross={L['L_cross']:.6f} "
                f"L_cluster={L['L_cluster']:.6f} L_uncert={L['L_uncert']:.6f} L_hn={L['L_hn']:.6f} L_reg={L['L_reg']:.6f}"
            )
            counts = np.bincount(network.psedo_labels.detach().cpu().numpy(), minlength=num_clusters)
            empty_cluster = int((counts == 0).sum())
            min_cluster = int(counts.min()) if counts.size > 0 else 0
            D = train_out.get('dump', {})
            sim_neg_vals = _to_np(D.get('sim_neg_sample', np.array([0.0]))).reshape(-1)
            sim_neg_vals = sim_neg_vals[np.isfinite(sim_neg_vals)] if sim_neg_vals.size > 0 else np.array([0.0])
            sim_neg_p99_route = float(np.quantile(sim_neg_vals, 0.99)) if sim_neg_vals.size > 0 else 0.0
            corr_u_fn_route = _rget(R, 'corr_u_fn', _rget(R, 'corr_u_fn_ratio', 0.0))
            route_line = (
                f"ROUTE: epoch={epoch} neg_mode={args.neg_mode} knn_neg_k={args.knn_neg_k} route_uncertain_only={int(args.route_uncertain_only)} "
                f"U_size={int(_rget(R, 'U_size', 0))} neg_per_anchor={_rget(R, 'neg_per_anchor', _rget(R, 'N_size', 0.0)):.2f} alpha_fn={args.alpha_fn:.4f} "
                f"hn_beta={args.hn_beta:.4f} FN_ratio={_rget(R, 'fn_ratio', 0.0):.4f} safe_ratio={_rget(R, 'safe_ratio', 0.0):.4f} "
                f"HN_ratio={_rget(R, 'hn_ratio', 0.0):.4f} FN_count={_rget(R, 'FN_count', 0.0):.0f} HN_count={_rget(R, 'HN_count', 0.0):.0f} neg_count={_rget(R, 'neg_count', 0.0):.0f} safe_neg_count={_rget(R, 'safe_neg_count', 0.0):.0f} "
                f"mean_s_post_FN={_rget(R, 'mean_s_post_fn', 0.0):.4f} mean_s_post_nonFN={_rget(R, 'mean_s_post_non_fn', 0.0):.4f} "
                f"delta_post={_rget(R, 'delta_post', 0.0):.4f} mean_sim_HN={_rget(R, 'mean_sim_hn', 0.0):.4f} mean_sim_safe_nonHN={_rget(R, 'mean_sim_safe_non_hn', 0.0):.4f} "
                f"delta_sim={_rget(R, 'delta_sim', 0.0):.4f} label_flip={_rget(R, 'label_flip', 0.0):.4f} stab_rate={_rget(R, 'stab_rate', 0.0):.4f} "
                f"empty_cluster={empty_cluster} min_cluster={min_cluster} denom_fn_share={_rget(R, 'denom_fn_share', 0.0):.4f} denom_safe_share={_rget(R, 'denom_safe_share', 0.0):.4f} "
                f"w_hit_min_ratio={_rget(R, 'w_hit_min_ratio', 0.0):.4f} w_mean_on_FN={_rget(R, 'w_mean_on_FN', 0.0):.4f} w_mean_on_safe={_rget(R, 'w_mean_on_safe', 0.0):.4f} assignment_stability={_rget(R, 'assignment_stability', 0.0):.4f} tau_fn_p50={_rget(R, 'tau_fn_p50', 0.0):.4f} tau_hn_p50={_rget(R, 'tau_hn_p50', 0.0):.4f} corr_u_fn={corr_u_fn_route:.4f} sim_neg_p99={sim_neg_p99_route:.4f} candidate_neg_size={_rget(R, 'candidate_neg_size', _rget(R, 'neg_count', 0.0)):.0f} routed_candidate_neg_size={_rget(R, 'routed_candidate_neg_size', 0.0):.0f} neg_after_filter_size={_rget(R, 'neg_after_filter_size', _rget(R, 'neg_count', 0.0)):.0f} neg_used_in_loss_size={_rget(R, 'neg_used_in_loss_size', _rget(R, 'neg_count', 0.0)):.0f} route_count_inconsistent={int(_rget(R, 'route_count_inconsistent', 0))}"
            )
            logger.info(metric_line)
            logger.info(route_line)

            if epoch % args.log_dist_interval == 0:
                D = train_out.get('dump', {})
                u_p10, u_p50, u_p90, u_mean, u_std = _qstats(_to_np(D.get('u_sample', np.array([]))))
                g_p10, g_p50, g_p90, g_mean, g_std = _qstats(_to_np(D.get('gamma_sample', np.array([]))))
                S_p10, S_p50, S_p90, S_mean, S_std = _qstats(_to_np(D.get('S_pair_sample', np.array([]))))
                w_p10, w_p50, w_p90, w_mean, w_std = _qstats(_to_np(D.get('w_pair_sample', np.array([]))))
                sp50, _, sp90, _, _ = _qstats(_to_np(D.get('sim_pos_sample', np.array([]))))
                _, _, sn90, _, _ = _qstats(_to_np(D.get('sim_neg_sample', np.array([]))))
                sn99 = float(np.quantile(_to_np(D.get('sim_neg_sample', np.array([0.0]))).reshape(-1), 0.99))
                mt10, mt50, mt90, _, _ = _qstats(_to_np(D.get('m_top1_sample', np.array([]))))
                mg10, mg50, mg90, _, _ = _qstats(_to_np(D.get('m_gap_sample', np.array([]))))
                distr_line = (
                    f"DISTR: epoch={epoch} u_p10={u_p10:.4f} u_p50={u_p50:.4f} u_p90={u_p90:.4f} u_mean={u_mean:.4f} u_std={u_std:.4f} "
                    f"gamma_p10={g_p10:.4f} gamma_p50={g_p50:.4f} gamma_p90={g_p90:.4f} gamma_mean={g_mean:.4f} gamma_std={g_std:.4f} "
                    f"S_p10={S_p10:.4f} S_p50={S_p50:.4f} S_p90={S_p90:.4f} S_mean={S_mean:.4f} S_std={S_std:.4f} "
                    f"w_p10={w_p10:.4f} w_p50={w_p50:.4f} w_p90={w_p90:.4f} w_mean={w_mean:.4f} w_std={w_std:.4f} "
                    f"sim_pos_p50={sp50:.4f} sim_pos_p90={sp90:.4f} sim_neg_p90={sn90:.4f} sim_neg_p99={sn99:.4f} "
                    f"m_top1_p10={mt10:.4f} m_top1_p50={mt50:.4f} m_top1_p90={mt90:.4f} "
                    f"m_gap_p10={mg10:.4f} m_gap_p50={mg50:.4f} m_gap_p90={mg90:.4f} fn_pair_share={_rget(R, 'fn_pair_share', 0.0):.4f} hn_pair_share={_rget(R, 'hn_pair_share', 0.0):.4f}"
                )
                logger.info(distr_line)

                if args.save_debug_npz:
                    counts = np.bincount(network.psedo_labels.detach().cpu().numpy(), minlength=num_clusters)
                    debug_path = os.path.join(args.debug_dir, f"debug_epoch_{epoch:03d}.npz")
                    _save_debug_npz(
                        debug_path,
                        D,
                        cluster_sizes=counts,
                        empty_cluster_count=int((counts == 0).sum()),
                        min_cluster_size=int(counts.min()) if counts.size > 0 else 0,
                        gate_value=train_out['gate'],
                        loss_dict=L,
                    )

            acc_list.append(acc)
            nmi_list.append(nmi)
            pur_list.append(pur)
            ari_list.append(ari)
            f1_list.append(f_score)

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                best_metrics = (acc, nmi, pur, ari, f_score)

        avg_acc = sum(acc_list) / len(acc_list)
        avg_nmi = sum(nmi_list) / len(nmi_list)
        avg_pur = sum(pur_list) / len(pur_list)
        avg_ari = sum(ari_list) / len(ari_list)
        avg_f1 = sum(f1_list) / len(f1_list)
        logger.info(" Average over all epochs::")
        logger.info(f"ACC={avg_acc:.4f} NMI={avg_nmi:.4f} PUR={avg_pur:.4f} ARI={avg_ari:.4f} F1={avg_f1:.4f}")
        print("\n Average over all epochs:")
        print(f"AVG ACC = {avg_acc:.4f}  AVG NMI = {avg_nmi:.4f}  AVG PUR = {avg_pur:.4f}  "
              f"AVG ARI = {avg_ari:.4f}  AVG F1 = {avg_f1:.4f}")

        # 最后一轮评估
        print("\n Final Evaluation (Last Epoch):")
        acc, nmi, pur, ari, f_score = valid(network, mv_data, num_samples, num_clusters)
        logger.info("Final Evaluation (Last Epoch):")
        logger.info(f"ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f} F1={f_score:.4f}")
        print('ACC = {:.4f} NMI = {:.4f} PUR = {:.4f} ARI = {:.4f} F1 = {:.4f}'.format(
            acc, nmi, pur, ari, f_score))

        # 最优一轮
        if best_metrics:
            acc, nmi, pur, ari, f_score = best_metrics
            logger.info(f"Best Evaluation (Epoch {best_epoch}):")
            logger.info(f"ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f} F1={f_score:.4f}")
            print(f"\n Best Evaluation (Epoch {best_epoch}):")
            print('ACC = {:.4f} NMI = {:.4f} PUR = {:.4f} ARI = {:.4f} F1 = {:.4f}'.format(
                acc, nmi, pur, ari, f_score))


        # — 保存模型 —
        if args.save_model:
            torch.save({
                'network_state_dict': network.state_dict(),
            }, f'./models/{args.dataset}_complete_model.pth')
    else:
        best_acc = 0
        best_epoch = -1
        best_metrics = None

        y_prev = None
        for epoch in range(1, args.con_epochs + 1):
            y_prev = network.psedo_labels.clone() if epoch > 1 else None
            train_out = contrastive_largedatasetstrain(
                network, mv_data, mvc_loss,
                args.batch_size, epoch,
                args.k, alpha, beta, optimizer,
                warmup_epochs=args.warmup_epochs,
                lambda_u=args.lambda_u,
                lambda_hn_penalty=args.lambda_hn_penalty,
                temperature_f=args.temperature_f,
                cross_warmup_epochs=args.cross_warmup_epochs,
                cross_ramp_epochs=args.cross_ramp_epochs,
                lambda_cross=args.lambda_cross,
                alpha_fn=args.alpha_fn,
                hn_beta=args.hn_beta,
                neg_mode=args.neg_mode,
                knn_neg_k=args.knn_neg_k,
                route_uncertain_only=args.route_uncertain_only,
                fn_route_warmup_epochs=args.fn_route_warmup_epochs,
                feature_base_weight=args.feature_base_weight,
                feature_route_weight=args.feature_route_weight,
                y_prev_labels=y_prev,
            )


            acc, nmi, pur, ari, f_score = valid(network, mv_data, num_samples, num_clusters)
            logger.info(f"ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f} F1={f_score:.4f}")
            print(f"[Epoch {epoch}] ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f} F1={f_score:.4f}")

            lr = optimizer.param_groups[0]['lr']
            L = train_out['loss']
            R = _route_with_dump_fallback(train_out['route'], train_out.get('dump', {}))
            metric_line = (
                f"METRIC: epoch={epoch} step={epoch} ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f} F1={f_score:.4f} "
                f"gate={train_out['gate']:.4f} lr={lr:.6g} temp_f={args.temperature_f:.4f} temp_l={args.temperature_l:.4f} "
                f"L_total={L['L_total']:.6f} L_recon={L['L_recon']:.6f} L_feat={L['L_feat']:.6f} L_cross={L['L_cross']:.6f} "
                f"L_cluster={L['L_cluster']:.6f} L_uncert={L['L_uncert']:.6f} L_hn={L['L_hn']:.6f} L_reg={L['L_reg']:.6f}"
            )
            counts = np.bincount(network.psedo_labels.detach().cpu().numpy(), minlength=num_clusters)
            empty_cluster = int((counts == 0).sum())
            min_cluster = int(counts.min()) if counts.size > 0 else 0
            D = train_out.get('dump', {})
            sim_neg_vals = _to_np(D.get('sim_neg_sample', np.array([0.0]))).reshape(-1)
            sim_neg_vals = sim_neg_vals[np.isfinite(sim_neg_vals)] if sim_neg_vals.size > 0 else np.array([0.0])
            sim_neg_p99_route = float(np.quantile(sim_neg_vals, 0.99)) if sim_neg_vals.size > 0 else 0.0
            corr_u_fn_route = _rget(R, 'corr_u_fn', _rget(R, 'corr_u_fn_ratio', 0.0))
            route_line = (
                f"ROUTE: epoch={epoch} neg_mode={args.neg_mode} knn_neg_k={args.knn_neg_k} route_uncertain_only={int(args.route_uncertain_only)} "
                f"U_size={int(_rget(R, 'U_size', 0))} neg_per_anchor={_rget(R, 'neg_per_anchor', _rget(R, 'N_size', 0.0)):.2f} alpha_fn={args.alpha_fn:.4f} "
                f"hn_beta={args.hn_beta:.4f} FN_ratio={_rget(R, 'fn_ratio', 0.0):.4f} safe_ratio={_rget(R, 'safe_ratio', 0.0):.4f} "
                f"HN_ratio={_rget(R, 'hn_ratio', 0.0):.4f} FN_count={_rget(R, 'FN_count', 0.0):.0f} HN_count={_rget(R, 'HN_count', 0.0):.0f} neg_count={_rget(R, 'neg_count', 0.0):.0f} safe_neg_count={_rget(R, 'safe_neg_count', 0.0):.0f} "
                f"mean_s_post_FN={_rget(R, 'mean_s_post_fn', 0.0):.4f} mean_s_post_nonFN={_rget(R, 'mean_s_post_non_fn', 0.0):.4f} "
                f"delta_post={_rget(R, 'delta_post', 0.0):.4f} mean_sim_HN={_rget(R, 'mean_sim_hn', 0.0):.4f} mean_sim_safe_nonHN={_rget(R, 'mean_sim_safe_non_hn', 0.0):.4f} "
                f"delta_sim={_rget(R, 'delta_sim', 0.0):.4f} label_flip={_rget(R, 'label_flip', 0.0):.4f} stab_rate={_rget(R, 'stab_rate', 0.0):.4f} "
                f"empty_cluster={empty_cluster} min_cluster={min_cluster} denom_fn_share={_rget(R, 'denom_fn_share', 0.0):.4f} denom_safe_share={_rget(R, 'denom_safe_share', 0.0):.4f} "
                f"w_hit_min_ratio={_rget(R, 'w_hit_min_ratio', 0.0):.4f} w_mean_on_FN={_rget(R, 'w_mean_on_FN', 0.0):.4f} w_mean_on_safe={_rget(R, 'w_mean_on_safe', 0.0):.4f} assignment_stability={_rget(R, 'assignment_stability', 0.0):.4f} tau_fn_p50={_rget(R, 'tau_fn_p50', 0.0):.4f} tau_hn_p50={_rget(R, 'tau_hn_p50', 0.0):.4f} corr_u_fn={corr_u_fn_route:.4f} sim_neg_p99={sim_neg_p99_route:.4f} candidate_neg_size={_rget(R, 'candidate_neg_size', _rget(R, 'neg_count', 0.0)):.0f} routed_candidate_neg_size={_rget(R, 'routed_candidate_neg_size', 0.0):.0f} neg_after_filter_size={_rget(R, 'neg_after_filter_size', _rget(R, 'neg_count', 0.0)):.0f} neg_used_in_loss_size={_rget(R, 'neg_used_in_loss_size', _rget(R, 'neg_count', 0.0)):.0f} route_count_inconsistent={int(_rget(R, 'route_count_inconsistent', 0))}"
            )
            logger.info(metric_line)
            logger.info(route_line)

            if epoch % args.log_dist_interval == 0:
                D = train_out.get('dump', {})
                u_p10, u_p50, u_p90, u_mean, u_std = _qstats(_to_np(D.get('u_sample', np.array([]))))
                g_p10, g_p50, g_p90, g_mean, g_std = _qstats(_to_np(D.get('gamma_sample', np.array([]))))
                S_p10, S_p50, S_p90, S_mean, S_std = _qstats(_to_np(D.get('S_pair_sample', np.array([]))))
                w_p10, w_p50, w_p90, w_mean, w_std = _qstats(_to_np(D.get('w_pair_sample', np.array([]))))
                sp50, _, sp90, _, _ = _qstats(_to_np(D.get('sim_pos_sample', np.array([]))))
                _, _, sn90, _, _ = _qstats(_to_np(D.get('sim_neg_sample', np.array([]))))
                sn99 = float(np.quantile(_to_np(D.get('sim_neg_sample', np.array([0.0]))).reshape(-1), 0.99))
                mt10, mt50, mt90, _, _ = _qstats(_to_np(D.get('m_top1_sample', np.array([]))))
                mg10, mg50, mg90, _, _ = _qstats(_to_np(D.get('m_gap_sample', np.array([]))))
                distr_line = (
                    f"DISTR: epoch={epoch} u_p10={u_p10:.4f} u_p50={u_p50:.4f} u_p90={u_p90:.4f} u_mean={u_mean:.4f} u_std={u_std:.4f} "
                    f"gamma_p10={g_p10:.4f} gamma_p50={g_p50:.4f} gamma_p90={g_p90:.4f} gamma_mean={g_mean:.4f} gamma_std={g_std:.4f} "
                    f"S_p10={S_p10:.4f} S_p50={S_p50:.4f} S_p90={S_p90:.4f} S_mean={S_mean:.4f} S_std={S_std:.4f} "
                    f"w_p10={w_p10:.4f} w_p50={w_p50:.4f} w_p90={w_p90:.4f} w_mean={w_mean:.4f} w_std={w_std:.4f} "
                    f"sim_pos_p50={sp50:.4f} sim_pos_p90={sp90:.4f} sim_neg_p90={sn90:.4f} sim_neg_p99={sn99:.4f} "
                    f"m_top1_p10={mt10:.4f} m_top1_p50={mt50:.4f} m_top1_p90={mt90:.4f} "
                    f"m_gap_p10={mg10:.4f} m_gap_p50={mg50:.4f} m_gap_p90={mg90:.4f} fn_pair_share={_rget(R, 'fn_pair_share', 0.0):.4f} hn_pair_share={_rget(R, 'hn_pair_share', 0.0):.4f}"
                )
                logger.info(distr_line)

                if args.save_debug_npz:
                    counts = np.bincount(network.psedo_labels.detach().cpu().numpy(), minlength=num_clusters)
                    debug_path = os.path.join(args.debug_dir, f"debug_epoch_{epoch:03d}.npz")
                    _save_debug_npz(
                        debug_path,
                        D,
                        cluster_sizes=counts,
                        empty_cluster_count=int((counts == 0).sum()),
                        min_cluster_size=int(counts.min()) if counts.size > 0 else 0,
                        gate_value=train_out['gate'],
                        loss_dict=L,
                    )

            acc_list.append(acc)
            nmi_list.append(nmi)
            pur_list.append(pur)
            ari_list.append(ari)
            f1_list.append(f_score)

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                best_metrics = (acc, nmi, pur, ari, f_score)

        avg_acc = sum(acc_list) / len(acc_list)
        avg_nmi = sum(nmi_list) / len(nmi_list)
        avg_pur = sum(pur_list) / len(pur_list)
        avg_ari = sum(ari_list) / len(ari_list)
        avg_f1 = sum(f1_list) / len(f1_list)
        logger.info(" Average over all epochs::")
        logger.info(f"ACC={avg_acc:.4f} NMI={avg_nmi:.4f} PUR={avg_pur:.4f} ARI={avg_ari:.4f} F1={avg_f1:.4f}")
        print("\n Average over all epochs:")
        print(f"AVG ACC = {avg_acc:.4f}  AVG NMI = {avg_nmi:.4f}  AVG PUR = {avg_pur:.4f}  "
              f"AVG ARI = {avg_ari:.4f}  AVG F1 = {avg_f1:.4f}")

        print("\n Final Evaluation (Last Epoch):")
        acc, nmi, pur, ari, f_score = valid(network, mv_data, num_samples, num_clusters)
        logger.info("Final Evaluation (Last Epoch):")
        logger.info(f"ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f} F1={f_score:.4f}")
        print('ACC = {:.4f} NMI = {:.4f} PUR = {:.4f} ARI = {:.4f} F1 = {:.4f}'.format(
            acc, nmi, pur, ari, f_score))

        if best_metrics:
            acc, nmi, pur, ari, f_score = best_metrics
            print(f"\n Best Evaluation (Epoch {best_epoch}):")
            print('ACC = {:.4f} NMI = {:.4f} PUR = {:.4f} ARI = {:.4f} F1 = {:.4f}'.format(
                acc, nmi, pur,    ari, f_score))
