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
parser.add_argument('--membership_mode', default='softmax_distance', type=str,
                    choices=['gaussian', 'softmax_distance'],
                    help='Membership kernel mode: paper-improved softmax_distance or legacy gaussian.')
parser.add_argument('--membership_temperature', default=1.0, type=float,
                    help='Temperature T_m for softmax-distance membership.')
parser.add_argument('--uncertainty_mode', default='log_odds', type=str,
                    choices=['legacy', 'log_odds', 'posterior_evidence'],
                    help='Uncertainty mode: legacy / log_odds / posterior_evidence (Module-A).')
parser.add_argument('--uncertainty_kappa', default=1.0, type=float,
                    help='Margin threshold kappa in u=Sigmoid((kappa-gamma)/T_u).')
parser.add_argument('--uncertainty_temperature', default=0.5, type=float,
                    help='Temperature T_u for uncertainty sigmoid mapping.')
parser.add_argument('--reliability_temperature', default=0.5, type=float,
                    help='Temperature T_w for reliability-weighted view fusion.')
parser.add_argument('--uncertainty_alpha', default=1.0, type=float,
                    help='Module-A weight alpha for margin evidence.')
parser.add_argument('--uncertainty_beta', default=1.0, type=float,
                    help='Module-A weight beta for cross-view KL evidence.')
parser.add_argument('--uncertainty_eta', default=1.0, type=float,
                    help='Module-A weight eta for EMA-time KL evidence.')
parser.add_argument('--uncertainty_time_momentum', default=0.9, type=float,
                    help='EMA momentum m for per-sample posterior buffer in Module-A.')
parser.add_argument('--neg_mode', default='batch', type=str, choices=['batch', 'knn'],
                    help='Negative candidate mode for pair-wise FN risk routing.')
parser.add_argument('--knn_neg_k', default=20, type=int,
                    help='k in kNN negatives when neg_mode=knn.')
parser.add_argument('--w_min', default=0.05, type=float,
                    help='Minimum negative weight for high FN-risk pairs.')
parser.add_argument('--route_s0', default=0.3, type=float,
                    help='Module-B FN-risk prior center in pair evidence logits.')
parser.add_argument('--route_t_fn', default=0.5, type=float,
                    help='Module-B temperature for FN-risk probability mapping.')
parser.add_argument('--route_hn_temp', default=0.2, type=float,
                    help='Module-B temperature for HN hardness score.')
parser.add_argument('--gate_s0', default=0.5, type=float,
                    help='Module-B stability gate center.')
parser.add_argument('--gate_tg', default=0.2, type=float,
                    help='Module-B stability gate temperature.')
parser.add_argument('--gate_ema_rho', default=0.9, type=float,
                    help='EMA smoothing for Module-B stability gate.')
parser.add_argument('--bayes_lambda_p', default=0.7, type=float,
                    help='Exponent on EMA posterior in Module-B anchor Bayes fusion.')
parser.add_argument('--bayes_lambda_l', default=1.0, type=float,
                    help='Exponent on current posterior in Module-B anchor Bayes fusion.')
parser.add_argument('--bayes_delta', default=0.2, type=float,
                    help='Low-confidence gate for Module-B anchor routing.')
parser.add_argument('--mass_delta', default=0.05, type=float,
                    help='Minimum gap between fn_mass and hn_mass for anchor typing.')
parser.add_argument('--lambda_fn_pull', default=0.1, type=float,
                    help='Weight of Module-B FN pull loss.')
parser.add_argument('--lambda_hn_margin', default=0.1, type=float,
                    help='Weight of Module-B HN margin loss.')
parser.add_argument('--hn_margin', default=0.2, type=float,
                    help='Margin in Module-B HN directional loss.')
parser.add_argument('--route_uncertain_only', default=True, type=lambda x: x.lower()=='true',
                    help='Apply pair-wise routing only for uncertain anchors.')
parser.add_argument('--log_dist_interval', default=5, type=int,
                    help='Epoch interval for DISTR summary and debug dump.')
parser.add_argument('--save_debug_npz', default=True, type=lambda x: x.lower()=='true',
                    help='Save debug npz dump periodically.')
parser.add_argument('--debug_dir', default='debug', type=str,
                    help='Directory to save debug npz files.')
parser.add_argument('--u_threshold_method', default='otsu', type=str, choices=['otsu'],
                    help='Adaptive threshold method for uncertain set in Module-A.')
parser.add_argument('--u_tau_ema_rho', default=0.9, type=float,
                    help='EMA smoothing rho for batch-level adaptive tau_u.')
parser.add_argument('--min_uncertain_ratio', default=0.02, type=float,
                    help='Lower bound ratio for uncertain set size safeguard.')
parser.add_argument('--max_uncertain_ratio', default=0.6, type=float,
                    help='Upper bound ratio for uncertain set size safeguard.')
parser.add_argument('--theta_temperature', default=0.5, type=float,
                    help='Temperature in theta_i certificate exp(sim/tau).')
parser.add_argument('--theta_threshold', default=0.5, type=float,
                    help='Unsafe certificate threshold Theta for forcing into U_t.')
parser.add_argument('--enable_theta_certificate', default=True, type=lambda x: x.lower()=='true',
                    help='Enable theta_i safety certificate to expand uncertain set.')
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def _qstats(x):
    x = np.asarray(x).reshape(-1)
    if x.size == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    return (float(np.quantile(x, 0.1)), float(np.quantile(x, 0.5)), float(np.quantile(x, 0.9)), float(np.mean(x)), float(np.std(x)))

def _to_np(v):
    if hasattr(v, 'detach'):
        return v.detach().cpu().numpy()
    return np.asarray(v)

def _rget(route, key, default=0.0):
    return route[key] if key in route else default

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
        membership_temperature=args.membership_temperature,
        uncertainty_mode=args.uncertainty_mode,
        uncertainty_kappa=args.uncertainty_kappa,
        uncertainty_temperature=args.uncertainty_temperature,
        reliability_temperature=args.reliability_temperature,
        uncertainty_alpha=args.uncertainty_alpha,
        uncertainty_beta=args.uncertainty_beta,
        uncertainty_eta=args.uncertainty_eta,
        uncertainty_time_momentum=args.uncertainty_time_momentum,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(network.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    mvc_loss = Loss(args.batch_size, num_clusters,
                    args.temperature_l, args.temperature_f).to(device)

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
                w_min=args.w_min,
                route_s0=args.route_s0,
                route_t_fn=args.route_t_fn,
                route_hn_temp=args.route_hn_temp,
                gate_s0=args.gate_s0,
                gate_tg=args.gate_tg,
                gate_ema_rho=args.gate_ema_rho,
                bayes_lambda_p=args.bayes_lambda_p,
                bayes_lambda_l=args.bayes_lambda_l,
                bayes_delta=args.bayes_delta,
                mass_delta=args.mass_delta,
                lambda_fn_pull=args.lambda_fn_pull,
                lambda_hn_margin=args.lambda_hn_margin,
                hn_margin=args.hn_margin,
                neg_mode=args.neg_mode,
                knn_neg_k=args.knn_neg_k,
                route_uncertain_only=args.route_uncertain_only,
                y_prev_labels=y_prev,
                u_threshold_method=args.u_threshold_method,
                u_tau_ema_rho=args.u_tau_ema_rho,
                min_uncertain_ratio=args.min_uncertain_ratio,
                max_uncertain_ratio=args.max_uncertain_ratio,
                theta_temperature=args.theta_temperature,
                theta_threshold=args.theta_threshold,
                enable_theta_certificate=args.enable_theta_certificate,
            )

            epoch_list.append(epoch)
            totalloss_list.append(train_out['loss']['L_total'])

            # 每轮评估
            acc, nmi, pur, ari, f_score = valid(network, mv_data, num_samples, num_clusters)
            logger.info(f"ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f} F1={f_score:.4f}")
            print(f"[Epoch {epoch}] ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f} F1={f_score:.4f}")

            lr = optimizer.param_groups[0]['lr']
            L = train_out['loss']
            R = train_out['route']
            metric_line = (
                f"METRIC: epoch={epoch} step={epoch} ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f} F1={f_score:.4f} "
                f"gate={train_out['gate']:.4f} lr={lr:.6g} temp_f={args.temperature_f:.4f} temp_l={args.temperature_l:.4f} "
                f"L_total={L['L_total']:.6f} L_recon={L['L_recon']:.6f} L_feat={L['L_feat']:.6f} L_cross={L['L_cross']:.6f} "
                f"L_cluster={L['L_cluster']:.6f} L_uncert={L['L_uncert']:.6f} L_hn={L['L_hn']:.6f} L_reg={L['L_reg']:.6f}"
            )
            counts = np.bincount(network.psedo_labels.detach().cpu().numpy(), minlength=num_clusters)
            empty_cluster = int((counts == 0).sum())
            min_cluster = int(counts.min()) if counts.size > 0 else 0
            route_line = (
                f"ROUTE: epoch={epoch} neg_mode={args.neg_mode} knn_neg_k={args.knn_neg_k} route_uncertain_only={int(args.route_uncertain_only)} "
                f"U_size={int(_rget(R, 'U_size', 0))} neg_per_anchor={_rget(R, 'neg_per_anchor', _rget(R, 'N_size', 0.0)):.2f} route_s0={args.route_s0:.4f} route_t_fn={args.route_t_fn:.4f} "
                f"w_min={args.w_min:.4f} route_hn_temp={args.route_hn_temp:.4f} FN_ratio={_rget(R, 'fn_ratio', 0.0):.4f} safe_ratio={_rget(R, 'safe_ratio', 0.0):.4f} "
                f"HN_ratio={_rget(R, 'hn_ratio', 0.0):.4f} FN_count={_rget(R, 'FN_count', 0.0):.0f} HN_count={_rget(R, 'HN_count', 0.0):.0f} neg_count={_rget(R, 'neg_count', 0.0):.0f} safe_neg_count={_rget(R, 'safe_neg_count', 0.0):.0f} "
                f"mean_s_post_FN={_rget(R, 'mean_s_post_fn', 0.0):.4f} mean_s_post_nonFN={_rget(R, 'mean_s_post_non_fn', 0.0):.4f} "
                f"delta_post={_rget(R, 'delta_post', 0.0):.4f} mean_sim_HN={_rget(R, 'mean_sim_hn', 0.0):.4f} mean_sim_safe_nonHN={_rget(R, 'mean_sim_safe_non_hn', 0.0):.4f} "
                f"delta_sim={_rget(R, 'delta_sim', 0.0):.4f} label_flip={_rget(R, 'label_flip', 0.0):.4f} stab_rate={_rget(R, 'stab_rate', 0.0):.4f} "
                f"empty_cluster={empty_cluster} min_cluster={min_cluster} denom_fn_share={_rget(R, 'denom_fn_share', 0.0):.4f} denom_safe_share={_rget(R, 'denom_safe_share', 0.0):.4f} "
                f"w_hit_min_ratio={_rget(R, 'w_hit_min_ratio', 0.0):.4f} w_mean_on_FN={_rget(R, 'w_mean_on_FN', 0.0):.4f} w_mean_on_safe={_rget(R, 'w_mean_on_safe', 0.0):.4f} "
                f"r_fn_mean={_rget(R, 'r_fn_mean', 0.0):.4f} w_neg_mean={_rget(R, 'w_neg_mean', 0.0):.4f} w_neg_p90={_rget(R, 'w_neg_p90', 0.0):.4f} "
                f"fn_mass_i_mean={_rget(R, 'fn_mass_i_mean', 0.0):.4f} hn_mass_i_mean={_rget(R, 'hn_mass_i_mean', 0.0):.4f} "
                f"fn_type_ratio={_rget(R, 'fn_type_ratio', 0.0):.4f} hn_type_ratio={_rget(R, 'hn_type_ratio', 0.0):.4f} neutral_ratio={_rget(R, 'neutral_ratio', 0.0):.4f} "
                f"L_fn_pull={_rget(R, 'L_fn_pull', 0.0):.6f} L_hn_margin={_rget(R, 'L_hn_margin', 0.0):.6f} gate_stab={_rget(R, 'gate_stab', 0.0):.4f} "
                f"tau_u={_rget(R, 'tau_u', 0.0):.4f} unsafe_ratio={_rget(R, 'unsafe_ratio', 0.0):.4f} theta_p50_batch={_rget(R, 'theta_p50_batch', 0.0):.4f}"
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
                w_min=args.w_min,
                route_s0=args.route_s0,
                route_t_fn=args.route_t_fn,
                route_hn_temp=args.route_hn_temp,
                gate_s0=args.gate_s0,
                gate_tg=args.gate_tg,
                gate_ema_rho=args.gate_ema_rho,
                bayes_lambda_p=args.bayes_lambda_p,
                bayes_lambda_l=args.bayes_lambda_l,
                bayes_delta=args.bayes_delta,
                mass_delta=args.mass_delta,
                lambda_fn_pull=args.lambda_fn_pull,
                lambda_hn_margin=args.lambda_hn_margin,
                hn_margin=args.hn_margin,
                neg_mode=args.neg_mode,
                knn_neg_k=args.knn_neg_k,
                route_uncertain_only=args.route_uncertain_only,
                y_prev_labels=y_prev,
                u_threshold_method=args.u_threshold_method,
                u_tau_ema_rho=args.u_tau_ema_rho,
                min_uncertain_ratio=args.min_uncertain_ratio,
                max_uncertain_ratio=args.max_uncertain_ratio,
                theta_temperature=args.theta_temperature,
                theta_threshold=args.theta_threshold,
                enable_theta_certificate=args.enable_theta_certificate,
            )


            acc, nmi, pur, ari, f_score = valid(network, mv_data, num_samples, num_clusters)
            logger.info(f"ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f} F1={f_score:.4f}")
            print(f"[Epoch {epoch}] ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f} F1={f_score:.4f}")

            lr = optimizer.param_groups[0]['lr']
            L = train_out['loss']
            R = train_out['route']
            metric_line = (
                f"METRIC: epoch={epoch} step={epoch} ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f} F1={f_score:.4f} "
                f"gate={train_out['gate']:.4f} lr={lr:.6g} temp_f={args.temperature_f:.4f} temp_l={args.temperature_l:.4f} "
                f"L_total={L['L_total']:.6f} L_recon={L['L_recon']:.6f} L_feat={L['L_feat']:.6f} L_cross={L['L_cross']:.6f} "
                f"L_cluster={L['L_cluster']:.6f} L_uncert={L['L_uncert']:.6f} L_hn={L['L_hn']:.6f} L_reg={L['L_reg']:.6f}"
            )
            counts = np.bincount(network.psedo_labels.detach().cpu().numpy(), minlength=num_clusters)
            empty_cluster = int((counts == 0).sum())
            min_cluster = int(counts.min()) if counts.size > 0 else 0
            route_line = (
                f"ROUTE: epoch={epoch} neg_mode={args.neg_mode} knn_neg_k={args.knn_neg_k} route_uncertain_only={int(args.route_uncertain_only)} "
                f"U_size={int(_rget(R, 'U_size', 0))} neg_per_anchor={_rget(R, 'neg_per_anchor', _rget(R, 'N_size', 0.0)):.2f} route_s0={args.route_s0:.4f} route_t_fn={args.route_t_fn:.4f} "
                f"w_min={args.w_min:.4f} route_hn_temp={args.route_hn_temp:.4f} FN_ratio={_rget(R, 'fn_ratio', 0.0):.4f} safe_ratio={_rget(R, 'safe_ratio', 0.0):.4f} "
                f"HN_ratio={_rget(R, 'hn_ratio', 0.0):.4f} FN_count={_rget(R, 'FN_count', 0.0):.0f} HN_count={_rget(R, 'HN_count', 0.0):.0f} neg_count={_rget(R, 'neg_count', 0.0):.0f} safe_neg_count={_rget(R, 'safe_neg_count', 0.0):.0f} "
                f"mean_s_post_FN={_rget(R, 'mean_s_post_fn', 0.0):.4f} mean_s_post_nonFN={_rget(R, 'mean_s_post_non_fn', 0.0):.4f} "
                f"delta_post={_rget(R, 'delta_post', 0.0):.4f} mean_sim_HN={_rget(R, 'mean_sim_hn', 0.0):.4f} mean_sim_safe_nonHN={_rget(R, 'mean_sim_safe_non_hn', 0.0):.4f} "
                f"delta_sim={_rget(R, 'delta_sim', 0.0):.4f} label_flip={_rget(R, 'label_flip', 0.0):.4f} stab_rate={_rget(R, 'stab_rate', 0.0):.4f} "
                f"empty_cluster={empty_cluster} min_cluster={min_cluster} denom_fn_share={_rget(R, 'denom_fn_share', 0.0):.4f} denom_safe_share={_rget(R, 'denom_safe_share', 0.0):.4f} "
                f"w_hit_min_ratio={_rget(R, 'w_hit_min_ratio', 0.0):.4f} w_mean_on_FN={_rget(R, 'w_mean_on_FN', 0.0):.4f} w_mean_on_safe={_rget(R, 'w_mean_on_safe', 0.0):.4f} "
                f"r_fn_mean={_rget(R, 'r_fn_mean', 0.0):.4f} w_neg_mean={_rget(R, 'w_neg_mean', 0.0):.4f} w_neg_p90={_rget(R, 'w_neg_p90', 0.0):.4f} "
                f"fn_mass_i_mean={_rget(R, 'fn_mass_i_mean', 0.0):.4f} hn_mass_i_mean={_rget(R, 'hn_mass_i_mean', 0.0):.4f} "
                f"fn_type_ratio={_rget(R, 'fn_type_ratio', 0.0):.4f} hn_type_ratio={_rget(R, 'hn_type_ratio', 0.0):.4f} neutral_ratio={_rget(R, 'neutral_ratio', 0.0):.4f} "
                f"L_fn_pull={_rget(R, 'L_fn_pull', 0.0):.6f} L_hn_margin={_rget(R, 'L_hn_margin', 0.0):.6f} gate_stab={_rget(R, 'gate_stab', 0.0):.4f} "
                f"tau_u={_rget(R, 'tau_u', 0.0):.4f} unsafe_ratio={_rget(R, 'unsafe_ratio', 0.0):.4f} theta_p50_batch={_rget(R, 'theta_p50_batch', 0.0):.4f}"
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
