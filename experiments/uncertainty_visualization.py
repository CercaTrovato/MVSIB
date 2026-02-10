import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

plt.rcParams['font.family'] = 'Times New Roman'

np.random.seed(42)

n_samples = 300
n_clusters = 3
dim = 2
samples_per_cluster = n_samples // n_clusters

means = np.array([[0, 0], [4.5, 0], [1.5, 2]])
cov = np.array([[1.5, 0.3], [0.3, 1.5]])

X, y = [], []
for i, mean in enumerate(means):
    pts = np.random.multivariate_normal(mean, cov, samples_per_cluster)
    X.append(pts)
    y.extend([i]*samples_per_cluster)

X = np.vstack(X)
y = np.array(y)

sig2 = 2.0
memberships = np.zeros((n_samples, n_clusters))
for k in range(n_clusters):
    dists = np.linalg.norm(X - means[k], axis=1)
    memberships[:,k] = np.exp(-dists**2 / (2*sig2))
memberships /= memberships.sum(axis=1, keepdims=True)

sorted_memberships = np.sort(memberships, axis=1)[:,::-1]
top2_diff = sorted_memberships[:,0] - sorted_memberships[:,1]
H = entropy(memberships.T)
u_i = 0.5*(1-H) + 0.5*(1-top2_diff)

fig, axs = plt.subplots(1,3, figsize=(18,5))
lims = [X[:,0].min()-1, X[:,0].max()+1, X[:,1].min()-1, X[:,1].max()+1]

for k in range(n_clusters):
    axs[0].scatter(X[y==k,0], X[y==k,1], label=f'Cluster {k}', alpha=0.6)
axs[0].set_title("Original Samples")
axs[0].set_xlim(lims[:2])
axs[0].set_ylim(lims[2:])
axs[0].legend()

colors = memberships.max(axis=1)
sizes = 20 + 60*(1-top2_diff)
sc = axs[1].scatter(X[:,0], X[:,1], c=colors, cmap='viridis', s=sizes, alpha=0.7)
plt.colorbar(sc, ax=axs[1], label='Max Membership')
axs[1].set_title("Membership & Top-2 Diff")
axs[1].set_xlim(lims[:2])
axs[1].set_ylim(lims[2:])

candidate_idxs = np.argsort(top2_diff)[:20]
best_idx = candidate_idxs[np.argmin(np.linalg.norm(X[candidate_idxs] - X.mean(axis=0), axis=1))]

axs[1].scatter(X[best_idx,0], X[best_idx,1], facecolors='none', edgecolors='r', s=120, linewidths=2)
top2_clusters = memberships[best_idx].argsort()[-2:][::-1]
for c in top2_clusters:
    axs[1].plot([X[best_idx,0], means[c,0]], [X[best_idx,1], means[c,1]], 'k--', lw=1)

# 改为取最不确定的10%
threshold_idx = int(0.10*n_samples)
sorted_idx = np.argsort(u_i)[-threshold_idx:]

for k in range(n_clusters):
    axs[2].scatter(X[y==k,0], X[y==k,1], label=f'Cluster {k}', alpha=0.4)
axs[2].scatter(X[sorted_idx,0], X[sorted_idx,1], facecolors='none', edgecolors='red', s=80, linewidths=2, label='Most Uncertain 10%')
axs[2].set_title("Uncertain Samples")
axs[2].set_xlim(lims[:2])
axs[2].set_ylim(lims[2:])
axs[2].legend()

for i, ax in enumerate(axs):
    ax.text(0.5, -0.12, f"({chr(97+i)})", transform=ax.transAxes,
            fontsize=14, fontname='Times New Roman', ha='center')

plt.tight_layout()
plt.show()
