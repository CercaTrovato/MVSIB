import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_clustering

# 1.1 Encoder/Decoder modules corresponding to paper Section 2.1.
class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)

# 1.2 forward: per-view encode/decode pass for view-specific autoencoders.
class Network(nn.Module):
    def __init__(self, num_views, num_samples, num_clusters, device,
                 input_size, feature_dim,
                 tau=5, eps=1e-3,
                 fn_hn_k=5,       # kNN neighbor count,
                 fn_hn_hidden=64,  # MLP2 hidden dimension
                 membership_mode='softmax_distance',
                 membership_temperature=1.0,
                 uncertainty_mode='log_odds',
                 uncertainty_kappa=1.0,
                 uncertainty_temperature=0.5,
                 reliability_temperature=0.5):

        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        for i in range(num_views):
            self.encoders.append(Encoder(input_size[i], feature_dim))
            self.decoders.append(Decoder(input_size[i], feature_dim))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.num_views = num_views
        self.num_samples = num_samples
        self.num_clusters = num_clusters
        self.device = device
        self.tau = tau
        self.eps = eps
        #SCE configuration: switch between legacy and log-odds paths for ablations.
        self.membership_mode = membership_mode
        self.membership_temperature = membership_temperature
        self.uncertainty_mode = uncertainty_mode
        self.uncertainty_kappa = uncertainty_kappa
        self.uncertainty_temperature = uncertainty_temperature
        self.reliability_temperature = reliability_temperature
        self.step = 0
        self.psedo_labels = torch.zeros(num_samples, dtype=torch.long)
        self.weights = nn.Parameter(torch.full((self.num_views,), 1 / self.num_views), requires_grad=True)

        # -- Module 1: cluster centers and bandwidth σ buffers --
        self.centers = [None] * (self.num_views + 1)
        self.sigmas = [None] * (self.num_views + 1)

        # -- Module 2: uncertainty-prediction MLP --
        hidden_dim = feature_dim // 2
        self.mlp_uncert = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # -- Module 3: FN/HN sub-module --
        self.fn_hn_k = fn_hn_k
        # -- Module 3: FN-vs-HN classifier MLP (input dim = num_views + 2) --
        #   (degree_diff, center_sim, + one affinity per view)
        in_dim = 3
        self.mlp_fn_hn = nn.Sequential(
            nn.Linear(in_dim, fn_hn_hidden),
            nn.ReLU(),
            nn.Linear(fn_hn_hidden, 2)  # outputs 2-class logits
        )

    def forward(self, xs):
        xrs = []
        zs = []
        for i in range(self.num_views):
            z = self.encoders[i](xs[i])
            xr = self.decoders[i](z)
            xrs.append(xr)
            zs.append(z)

        return xrs, zs

    def get_weights(self):
        softmax_weights = torch.softmax(self.weights, dim=0)
        weights = softmax_weights / torch.sum(softmax_weights)

        return weights

# 1.3 Feature fusion module corresponding to Eq.(4).
    def fusion(self, zs):
        weights = self.get_weights()
        weighted_zs = [z * weight for z, weight in zip(zs, weights)]
        stacked_zs = torch.stack(weighted_zs)
        common_z = torch.sum(stacked_zs, dim=0)

        return common_z

# 1.4 Cluster center computation for Eq.(6)-(7).
    def compute_centers(self, x, psedo_labels):
        """
        Exactly reproduces the original weight-matrix @ x form, equivalent to F.normalize(one_hot,1)@x.
        """
        device = x.device
        ps = psedo_labels.to(device)
        N, d = x.shape
        L = self.num_clusters

        # -- 1) Build one-hot matrix (L, N) --
        weight = torch.zeros(L, N, device=device)
        weight[ps, torch.arange(N, device=device)] = 1.0

        # -- 2) L1-normalize each row --
        weight = F.normalize(weight, p=1, dim=1)  # ensures each row sums to 1

        # —— 3) centers = weight @ x ——
        centers = weight @ x  # (L, d)

        # -- 4) L2-normalize each center --
        centers = F.normalize(centers, p=2, dim=1)

        return centers

    def clustering(self, features):
        kwargs = {
            'metric': 'cosine',
            'distributed': False,
            'random_state': 0,
            'n_clusters': self.num_clusters,
            'verbose': False
        }
        clustering_model = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
        psedo_labels = clustering_model.fit_predict(features.to(dtype=torch.float64))

        return psedo_labels

    def update_centers(self, zs_list, common_z):
        """
        Called every tau steps: compute centers and σ via PyTorchKMeans.
        Force an update on first call or when any center is None; no gradient tracking.
        """
        self.step += 1
        need_init = any(c is None for c in self.centers)
        if not need_init and (self.step % self.tau != 0):
            return

        features = zs_list + [common_z]
        # disable gradients
        with torch.no_grad():
            for v, z in enumerate(features):
                km = torch_clustering.PyTorchKMeans(
                    init='k-means++',
                    n_clusters=self.num_clusters,
                    max_iter=300,
                    tol=1e-4,
                    metric='euclidean',
                    random_state=0,
                    verbose=False
                )
                # fit_predict is outside autograd graph
                labels = km.fit_predict(z.to(dtype=torch.float64)).to(self.device).long()
                centers = km.cluster_centers_.to(dtype=z.dtype).to(self.device)  # (L, d)

                # compute distance from each sample to its assigned center
                assigned = centers[labels]  # (N, d)
                dists = torch.norm(z - assigned, dim=1)  # (N,)

                # per-cluster σ_k = max(median(cluster_dists), eps)
                sigmas_v = []
                for k in range(self.num_clusters):
                    d_k = dists[labels == k]
                    if d_k.numel() > 0:
                        sigma_k = d_k.median()
                    else:
                        sigma_k = torch.tensor(0.0, device=self.device)
                    sigmas_v.append(torch.clamp(sigma_k, min=self.eps))
                sigmas_v = torch.stack(sigmas_v)  # (L,)

                self.centers[v] = centers
                self.sigmas[v] = sigmas_v

    def compute_membership(self, z, v_index):
        """
        Compute memberships (N×L) for view v or consensus space.
        - legacy: Gaussian-kernel membership (original Eq.6-7 style)
        - softmax_distance: softmax(-d/T_m), less sensitive to noisy σ estimates
        """
        centers = self.centers[v_index]  # (L, d)

        if self.membership_mode == 'softmax_distance':
            dists_sq = torch.cdist(z, centers, p=2) ** 2
            logits = -dists_sq / max(self.membership_temperature, self.eps)
            membership = torch.softmax(logits, dim=1)
            return membership

        # legacy gaussian path
        sigmas = self.sigmas[v_index]   # (L,)
        dists_sq = torch.cdist(z, centers, p=2) ** 2
        denom = (2 * (sigmas ** 2).clamp(min=self.eps)).view(1, -1)
        unnorm = torch.exp(-dists_sq / denom)
        membership = unnorm / (unnorm.sum(dim=1, keepdim=True) + 1e-12)
        return membership

    def estimate_uncertainty(self, memberships, common_z):
        """
        SCE uncertainty estimation:
        - legacy: entropy + top-2 gap + max-view fusion (original path)
        - log_odds: option A using log-odds margin and reliability-weighted view fusion (recommended)
        """
        V = self.num_views

        if self.uncertainty_mode == 'legacy':
            u_vs = []
            for v in range(V):
                m = memberships[v]  # (N, L)
                top2 = torch.topk(m, 2, dim=1).values
                delta = top2[:, 0] - top2[:, 1]
                delta_norm = (delta - delta.min()) / (delta.max() - delta.min() + 1e-12)
                ent = -torch.sum(m * torch.log(m + 1e-12), dim=1)
                ent_norm = (ent - ent.min()) / (ent.max() - ent.min() + 1e-12)
                u_v = 0.5 * ent_norm + 0.5 * (1.0 - delta_norm)
                u_vs.append(u_v)

            u_stack = torch.stack(u_vs, dim=1)
            u = u_stack.max(dim=1).values
            u_hat = self.mlp_uncert(common_z).squeeze(1)
            return u, u_hat

        # Option A: log-odds margin -> sigmoid, then reliability-softmax fusion across views.
        u_vs = []
        gamma_vs = []
        for v in range(V):
            m = memberships[v]
            top2 = torch.topk(m, 2, dim=1).values
            gamma_v = torch.log((top2[:, 0] + self.eps) / (top2[:, 1] + self.eps))
            u_v = torch.sigmoid((self.uncertainty_kappa - gamma_v) / max(self.uncertainty_temperature, self.eps))
            gamma_vs.append(gamma_v)
            u_vs.append(u_v)

        gamma_stack = torch.stack(gamma_vs, dim=1)  # (N, V)
        u_stack = torch.stack(u_vs, dim=1)          # (N, V)
        view_weights = torch.softmax(gamma_stack / max(self.reliability_temperature, self.eps), dim=1)
        u = (view_weights * u_stack).sum(dim=1)

        u_hat = self.mlp_uncert(common_z).squeeze(1)
        return u, u_hat

    def classify_fn_hn(self,
                       zs_list, common_z, memberships,
                       batch_psedo_label, certain_mask, uncertain_mask,
                       epoch=100, fn_hn_warmup=10):

        # 1) Skip early epochs before warm-up threshold.
        if epoch < fn_hn_warmup or not uncertain_mask.any():
            return None, None, None
        device = self.device

        # 0) Move tensors onto the same device.
        batch_psedo_label = batch_psedo_label.to(device)
        certain_mask = certain_mask.to(device)
        uncertain_mask = uncertain_mask.to(device)

        V = self.num_views
        N = batch_psedo_label.size(0)
        k = self.fn_hn_k
        eps = 1e-12

        # 1) Precompute kNN distances & indices; dists shape is (V, N, N).
        dists = torch.stack([
            torch.cdist(z.to(device), z.to(device), p=2)
            for z in zs_list
        ], dim=0)

        # 2) Take top-(k+1), remove self index, keep top-k: knn_idx (V, N, k).
        knn_idx = torch.topk(-dists, k + 1, dim=2).indices[:, :, 1:]

        # 3) Gather neighbor labels and certain-mask values on GPU.
        knn_labels = batch_psedo_label[knn_idx]  # (V, N, k)
        certain_mask_knn = certain_mask.unsqueeze(0).unsqueeze(2).expand(V, N, k)  # (V, N, k)

        # 4) Vectorized computation of Δd_i.
        same = (knn_labels == batch_psedo_label.view(1, N, 1)) & certain_mask_knn
        diff = (~same) & certain_mask_knn
        pos_cnt = same.sum(dim=2).float()  # (V, N)
        neg_cnt = diff.sum(dim=2).float()
        delta = (pos_cnt - neg_cnt) / (pos_cnt + neg_cnt + eps)  # (V, N)
        delta_d = delta.mean(dim=0)  # (N,)

        # 5) Compute s_i: consensus-center similarity.
        mu_c = self.centers[V][batch_psedo_label]  # (N, d)
        s = F.cosine_similarity(common_z, mu_c, dim=1)  # (N,)

        # 6) Compute a_i: mean distance discrepancy across views.
        v_idx = torch.arange(V, device=device)[:, None, None]
        n_idx = torch.arange(N, device=device)[None, :, None]
        neigh_d = dists[v_idx, n_idx, knn_idx]  # (V, N, k)
        d_pos = torch.where(same, neigh_d, torch.nan).nanmean(dim=2)
        d_neg = torch.where(diff, neigh_d, torch.nan).nanmean(dim=2)
        d_pos = torch.nan_to_num(d_pos, nan=0.0)
        d_neg = torch.nan_to_num(d_neg, nan=0.0)
        a = ((d_neg - d_pos) / (d_neg + d_pos + eps)).mean(dim=0)  # (N,)

        # 7) Select uncertain samples, concatenate features, and infer with MLP.
        idx_uncertain = uncertain_mask.nonzero(as_tuple=True)[0]  # (M,)
        feats_all = torch.stack([delta_d, s, a], dim=1)  # (N,3)
        feats = feats_all[idx_uncertain]  # (M,3)
        logits_raw = self.mlp_fn_hn(feats)  # (M,2)

        # -- Change start: drop bottom 10% by confidence quantile -- #
        probs = F.softmax(logits_raw, dim=1)  # (M,2)
        conf, preds = probs.max(dim=1)  # (M,)

        # drop bottom 10% confidence samples
        thresh = torch.quantile(conf, 0.1)
        keep = conf > thresh

        if keep.sum() == 0:
            return None, None, None

        idx_kept = idx_uncertain[keep]  # keep the remaining 90%
        logits = logits_raw[keep]
        feats = feats[keep]
        # -- Change end -- #

        return logits, idx_kept, feats

    def compute_consistency_scores(self, memberships, batch_psedo_label):
        """
        Compute cross-view consistency S_ij in [0,1], shape (N, N).
        Args:
            English documentation details.
            English documentation details.
        Returns:
            English documentation details.
        """
        N = batch_psedo_label.size(0)
        V = self.num_views
        device = self.device

        # 1) Extract confidence vectors from each view and consensus.
        conf_v = torch.stack([
            memberships[v][torch.arange(N, device=device), batch_psedo_label]
            for v in range(V)
        ], dim=0).to(device)  # (V, N)

        conf_c = memberships[V][torch.arange(N, device=device), batch_psedo_label].to(device)  # (N,)

        # 2) Compute uncertainty for each view.
        u_vs = []
        for v in range(V):
            m = memberships[v]  # (N, L)
            # top-2 difference computation
            top2 = torch.topk(m, 2, dim=1).values  # (N, 2)
            delta = top2[:, 0] - top2[:, 1]  # (N,)
            delta_norm = (delta - delta.min()) / (delta.max() - delta.min() + 1e-12)
            u_vs.append(delta_norm)

        u_i = torch.stack(u_vs, dim=1)  # (N, V)
        u_weights = 1.0 - u_i  # lower uncertainty gives larger weight

        # 3) Compute per-view consistency matrix p_v (V, N, N).
        p_v = conf_v.unsqueeze(2) * (batch_psedo_label.view(-1, 1) == batch_psedo_label.view(1, -1)).to(
            device)  # (V, N, N)
        p_view_max = p_v.max(dim=0).values  # (N, N)

        # 4) Compute consensus-space consistency score.
        p_c = conf_c.view(-1, 1) * (batch_psedo_label.view(-1, 1) == batch_psedo_label.view(1, -1)).to(device)  # (N, N)

        # 5) Weight view consistency by uncertainty-based reliability.
        # expand u_weights to match p_view_max dimensions
        u_weights_expanded = u_weights.mean(dim=1)  # (N, V) -> (N,) average over the view dimension

        # weighted view consistency score
        weighted_p_view_max = p_view_max * u_weights_expanded  # (N, N)

        # 6) Compute final consistency score S_ij
        S = torch.min(weighted_p_view_max, p_c)  # (N, N)
        return S

