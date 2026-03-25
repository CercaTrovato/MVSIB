# import numpy as np
# import matplotlib.pyplot as plt
#
# English explanation comment.
# np.random.seed(42)
#
# English explanation comment.
# means = [(0, 0), (5, 5), (-5, 5)]
# cov = np.array([[3.0, 0], [0, 1.5]])
# n_samples = 40
# outlier_ratio = 0.10
#
# English explanation comment.
# clusters, labels = [], []
# for i, mu in enumerate(means):
#     pts = np.random.multivariate_normal(mu, cov, size=n_samples)
# English explanation comment.
#     n_out = int(n_samples * outlier_ratio)
#     eigvals, _ = np.linalg.eig(cov)
#     r_min, r_max = np.sqrt(eigvals.max())*2.0, np.sqrt(eigvals.max())*2.5
#     outliers = []
#     while len(outliers) < n_out:
#         other = np.random.choice([j for j in range(3) if j != i])
#         p = np.random.multivariate_normal(means[other], cov)
#         if r_min <= np.linalg.norm(p - mu) <= r_max:
#             outliers.append(p)
#     clusters.append(np.vstack([pts, outliers]))
#     labels += [i] * (n_samples + n_out)
#
# all_pts = np.vstack(clusters)
# labels = np.array(labels)
# colors = ['C0', 'C1', 'C2']
#
# English explanation comment.
# fig, ax = plt.subplots(figsize=(1.5,1.5))
#
# for i in range(3):
#     pts = all_pts[labels == i]
#     ax.scatter(pts[:, 0], pts[:, 1], s=1, color=colors[i], alpha=1)
#
# English explanation comment.
# center = np.mean(all_pts, axis=0)
# r = np.max(np.linalg.norm(all_pts - center, axis=1))
# circle = plt.Circle(center, r, edgecolor='gray', facecolor='none', linewidth=0.5)
# ax.add_patch(circle)
#
# angles = [0, 2*np.pi/3, 4*np.pi/3]
# for angle, col in zip(angles, colors):
#     x1 = center[0] + r * np.cos(angle)
#     y1 = center[1] + r * np.sin(angle)
#     ax.plot([center[0], x1], [center[1], y1],
#             linestyle='--',  linewidth=0.5)
#
# English explanation comment.
# ax.set_xticks([])
# ax.set_yticks([])
# for spine in ax.spines.values():
#     spine.set_visible(False)
#
# plt.tight_layout()
# plt.show()




import numpy as np
import matplotlib.pyplot as plt

# English explanation comment.
np.random.seed(42)

# English explanation comment.
means = [(0, 0), (5, 5), (-5, 5)]
cov = np.array([[3.0, 0], [0, 1.5]]) * 0.6  # Shrink covariance to make clusters more compact
n_samples = 40
outlier_ratio = 0.0  # Disable boundary outliers so every sample belongs to its intended cluster
n_out = int(n_samples * outlier_ratio)
cluster_size = n_samples + n_out

# English explanation comment.
clusters = []
labels = []
for i, mu in enumerate(means):
    # English explanation comment.
    pts_norm = np.random.multivariate_normal(mu, cov, size=n_samples)
    # English explanation comment.
    pts_all = pts_norm
    clusters.append(pts_all)
    labels += [i] * pts_all.shape[0]

# English explanation comment.
all_pts = np.vstack(clusters)
labels = np.array(labels)

# English explanation comment.
center = all_pts.mean(axis=0)
phis = np.arctan2(all_pts[:,1] - center[1],
                  all_pts[:,0] - center[0]) % (2*np.pi)

# English explanation comment.
angles = np.array([0, 2*np.pi/3, 4*np.pi/3])

# English explanation comment.
fig, ax = plt.subplots(figsize=(1.5, 1.5))
colors = ['C0','C1','C2']

# English explanation comment.
for i in range(3):
    pts = all_pts[labels == i]
    ax.scatter(pts[:,0], pts[:,1], s=1, color=colors[i], alpha=0.9)

# English explanation comment.
r = np.linalg.norm(all_pts - center, axis=1).max()
circle = plt.Circle(center, r, edgecolor='gray', facecolor='none', linewidth=0.5)
ax.add_patch(circle)
for angle, col in zip(angles, colors):
    x1 = center[0] + r * np.cos(angle)
    y1 = center[1] + r * np.sin(angle)
    ax.plot([center[0], x1], [center[1], y1],
            linestyle='--', linewidth=0.5, color=col)

# English explanation comment.
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values(): spine.set_visible(False)

plt.tight_layout()
plt.show()
