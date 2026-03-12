from sklearn.metrics import v_measure_score, adjusted_rand_score, accuracy_score, fowlkes_mallows_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch_clustering import PyTorchKMeans

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def evaluate(label, pred):
    nmi = v_measure_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    f_score = fowlkes_mallows_score(label, pred)
    return nmi, ari, acc, pur, f_score

def inference(loader, model, data_size):
    model.eval()
    commonZ_list = []
    labels_vector = []
    for batch_idx, (xs, y, _) in enumerate(loader):
        with torch.no_grad():
            xrs, zs = model(xs)
            commonz = model.fusion(zs)
            commonZ_list.append(commonz)
        labels_vector.extend(y)
    commonZ = torch.cat(commonZ_list, dim=0)
    labels_vector = np.array(labels_vector).reshape(data_size)
    return labels_vector, commonZ

def valid(model, dataset, data_size, class_num):
    model.eval()
    with torch.no_grad():
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
            drop_last=False,
        )
        labels_vector, commonZ = inference(test_loader, model, data_size)

        # 用 PyTorchKMeans 替代 model.clustering
        clustering_model = PyTorchKMeans(
            init='k-means++',
            n_clusters=class_num,
            metric='cosine',
            random_state=0,
            verbose=False
        )
        y_pred = clustering_model.fit_predict(commonZ.to(dtype=torch.float64))
        y_pred = y_pred.cpu().numpy()

        # 聚类评估
        print('Clustering results:')
        nmi, ari, acc, pur, f_score = evaluate(labels_vector, y_pred)
        print('ACC = {:.4f} NMI = {:.4f} PUR = {:.4f} ARI = {:.4f} f_score = {:.4f}'.format(
            acc, nmi, pur, ari, f_score
        ))

    model.train()
    return acc, nmi, pur, ari, f_score


def get_cluster_class_mapping(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    d = max(y_true.max(), y_pred.max()) + 1
    align = np.zeros((d, d), dtype=np.int64)
    for i in range(y_pred.size):
        align[y_pred[i], y_true[i]] += 1
    row, col = linear_sum_assignment(align.max() - align)
    return {int(r): int(c) for r, c in zip(row, col)}


def remap_pred_by_matching(y_pred, mapping):
    return np.asarray([mapping.get(int(c), int(c)) for c in y_pred], dtype=np.int64)


def extract_commonz_and_labels(model, dataset, data_size):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False)
        labels_vector, commonz = inference(loader, model, data_size)
    if was_training:
        model.train()
    return labels_vector.astype(np.int64), commonz.detach().cpu().numpy()


def cluster_from_commonz(commonZ, class_num):
    commonz_tensor = torch.as_tensor(commonZ, dtype=torch.float64)
    clustering_model = PyTorchKMeans(
        init='k-means++',
        n_clusters=class_num,
        metric='cosine',
        random_state=0,
        verbose=False,
    )
    y_pred = clustering_model.fit_predict(commonz_tensor)
    return y_pred.detach().cpu().numpy().astype(np.int64)


def per_class_clustering_report(y_true, y_pred):
    labels = np.arange(max(y_true.max(), y_pred.max()) + 1)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        'labels': labels,
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
    }


def analyze_minority_absorption(cm, class_support, minority_threshold=50):
    class_support = np.asarray(class_support)
    minority_mask = class_support <= minority_threshold
    majority_mask = class_support > minority_threshold
    absorbed = np.zeros_like(class_support, dtype=np.float64)
    for idx in np.where(minority_mask)[0]:
        denom = float(class_support[idx])
        if denom <= 0:
            absorbed[idx] = 0.0
            continue
        row = cm[idx].copy()
        if idx < row.shape[0]:
            row[idx] = 0
        absorbed[idx] = float(row[majority_mask].sum()) / denom if majority_mask.any() else 0.0
    minority_mean = float(absorbed[minority_mask].mean()) if minority_mask.any() else 0.0
    return absorbed, minority_mean


def compute_class_centroids(embeddings, y_true):
    labels = np.arange(int(y_true.max()) + 1)
    d = embeddings.shape[1]
    centroids = np.zeros((labels.shape[0], d), dtype=np.float64)
    support = np.zeros(labels.shape[0], dtype=np.int64)
    for c in labels:
        mask = y_true == c
        support[c] = int(mask.sum())
        if support[c] > 0:
            centroids[c] = embeddings[mask].mean(axis=0)
    return centroids, support


def compute_centroid_drift(pretrain_embeddings, final_embeddings, y_true):
    pre_centroids, _ = compute_class_centroids(pretrain_embeddings, y_true)
    fin_centroids, support = compute_class_centroids(final_embeddings, y_true)
    drift = np.linalg.norm(fin_centroids - pre_centroids, axis=1)
    drift[support == 0] = 0.0
    return drift


def compute_classwise_fn_hn_ratios(embeddings, y_true, y_remap, k=10):
    n = embeddings.shape[0]
    k_eff = int(min(max(1, k), max(1, n - 1)))
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric='cosine')
    nbrs.fit(embeddings)
    neigh = nbrs.kneighbors(return_distance=False)[:, 1:]

    classes = int(y_true.max()) + 1
    fn_sum = np.zeros(classes, dtype=np.float64)
    hn_sum = np.zeros(classes, dtype=np.float64)
    counts = np.zeros(classes, dtype=np.int64)
    for i in range(n):
        yi = int(y_true[i])
        neigh_idx = neigh[i]
        neigh_true = y_true[neigh_idx]
        neigh_pred = y_remap[neigh_idx]
        same_true = neigh_true == yi
        diff_true = neigh_true != yi
        fn_like = np.logical_and(same_true, neigh_pred != y_remap[i])
        hn_like = np.logical_and(diff_true, neigh_pred == y_remap[i])
        fn_sum[yi] += float(fn_like.mean())
        hn_sum[yi] += float(hn_like.mean())
        counts[yi] += 1

    fn_ratio = np.divide(fn_sum, np.maximum(counts, 1), dtype=np.float64)
    hn_ratio = np.divide(hn_sum, np.maximum(counts, 1), dtype=np.float64)
    return fn_ratio, hn_ratio


def run_epoch_class_analysis(model, dataset, data_size, class_num, pretrain_embeddings,
                             minority_threshold=50, analysis_knn_k=10):
    y_true, commonz = extract_commonz_and_labels(model, dataset, data_size)
    y_pred = cluster_from_commonz(commonz, class_num)
    mapping = get_cluster_class_mapping(y_true, y_pred)
    y_remap = remap_pred_by_matching(y_pred, mapping)

    per_class = per_class_clustering_report(y_true, y_remap)
    cm = per_class['confusion_matrix']
    support = per_class['support']
    minority_mask = support <= minority_threshold

    absorbed_ratio, minority_absorption_mean = analyze_minority_absorption(
        cm, support, minority_threshold=minority_threshold
    )
    centroid_drift = compute_centroid_drift(pretrain_embeddings, commonz, y_true)
    fn_ratio, hn_ratio = compute_classwise_fn_hn_ratios(commonz, y_true, y_remap, k=analysis_knn_k)

    f1 = per_class['f1']
    recall = per_class['recall']

    def _safe_mean(values, mask=None):
        if mask is None:
            mask = np.ones(values.shape[0], dtype=bool)
        vals = values[mask]
        return float(vals.mean()) if vals.size > 0 else 0.0

    def _topk_mean(values, k=3, largest=False):
        if values.size == 0:
            return 0.0
        kk = min(k, values.size)
        order = np.argsort(values)
        pick = order[-kk:] if largest else order[:kk]
        return float(values[pick].mean())

    align_labels = np.arange(max(y_true.max(), y_pred.max()) + 1)
    align_matrix = confusion_matrix(y_pred, y_true, labels=align_labels)

    summary = {
        'macro_F1': _safe_mean(f1),
        'minority_recall_mean': _safe_mean(recall, minority_mask),
        'minority_f1_mean': _safe_mean(f1, minority_mask),
        'minority_absorption_mean': float(minority_absorption_mean),
        'minority_centroid_drift_mean': _safe_mean(centroid_drift, minority_mask),
        'minority_FN_ratio_mean': _safe_mean(fn_ratio, minority_mask),
        'minority_HN_ratio_mean': _safe_mean(hn_ratio, minority_mask),
        'worst3_f1_mean': _topk_mean(f1, k=3, largest=False),
        'top3_FN_mean': _topk_mean(fn_ratio, k=3, largest=True),
        'top3_HN_mean': _topk_mean(hn_ratio, k=3, largest=True),
    }

    return {
        'summary': summary,
        'per_class_metrics': {
            'class_id': per_class['labels'],
            'support': support,
            'precision': per_class['precision'],
            'recall': recall,
            'f1': f1,
            'fn_ratio': fn_ratio,
            'hn_ratio': hn_ratio,
            'absorbed_majority_ratio': absorbed_ratio,
            'centroid_drift': centroid_drift,
        },
        'confusion_matrix': cm,
        'alignment_matrix': align_matrix,
        'commonZ': commonz,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_remap': y_remap,
    }
