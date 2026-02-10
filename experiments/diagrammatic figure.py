# import numpy as np
# import matplotlib.pyplot as plt
#
# # 设置随机种子以复现
# np.random.seed(42)
#
# # 1. 定义三个簇的参数（簇心和协方差）
# means = [(0, 0), (5, 5), (-5, 5)]
# cov = np.array([[3.0, 0], [0, 1.5]])
# n_samples = 40
# outlier_ratio = 0.10
#
# # 2. 生成数据
# clusters, labels = [], []
# for i, mu in enumerate(means):
#     pts = np.random.multivariate_normal(mu, cov, size=n_samples)
#     # 边界外来点
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
# # 3. 绘图
# fig, ax = plt.subplots(figsize=(1.5,1.5))
#
# for i in range(3):
#     pts = all_pts[labels == i]
#     ax.scatter(pts[:, 0], pts[:, 1], s=1, color=colors[i], alpha=1)
#
# # 4. 外框圆 + 三条虚线分割线
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
# # 5. 去除坐标轴和边框
# ax.set_xticks([])
# ax.set_yticks([])
# for spine in ax.spines.values():
#     spine.set_visible(False)
#
# plt.tight_layout()
# plt.show()




import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以复现
np.random.seed(42)

# 参数
means = [(0, 0), (5, 5), (-5, 5)]
cov = np.array([[3.0, 0], [0, 1.5]]) * 0.6  # 缩小协方差，使簇更紧凑
n_samples = 40
outlier_ratio = 0.0  # 取消边界外来点，确保所有样本都属于正确簇
n_out = int(n_samples * outlier_ratio)
cluster_size = n_samples + n_out

# 生成数据并记录 global 索引与真实标签
clusters = []
labels = []
for i, mu in enumerate(means):
    # 正常高斯样本
    pts_norm = np.random.multivariate_normal(mu, cov, size=n_samples)
    # 无 outliers
    pts_all = pts_norm
    clusters.append(pts_all)
    labels += [i] * pts_all.shape[0]

# 合并
all_pts = np.vstack(clusters)
labels = np.array(labels)

# 计算全局中心与各点极角
center = all_pts.mean(axis=0)
phis = np.arctan2(all_pts[:,1] - center[1],
                  all_pts[:,0] - center[0]) % (2*np.pi)

# 边界线角度（0, 120, 240 度）
angles = np.array([0, 2*np.pi/3, 4*np.pi/3])

# 可视化
fig, ax = plt.subplots(figsize=(1.5, 1.5))
colors = ['C0','C1','C2']

# 绘制散点
for i in range(3):
    pts = all_pts[labels == i]
    ax.scatter(pts[:,0], pts[:,1], s=1, color=colors[i], alpha=0.9)

# 外框圆 + 三条虚线分割线
r = np.linalg.norm(all_pts - center, axis=1).max()
circle = plt.Circle(center, r, edgecolor='gray', facecolor='none', linewidth=0.5)
ax.add_patch(circle)
for angle, col in zip(angles, colors):
    x1 = center[0] + r * np.cos(angle)
    y1 = center[1] + r * np.sin(angle)
    ax.plot([center[0], x1], [center[1], y1],
            linestyle='--', linewidth=0.5, color=col)

# 隐藏坐标轴与边框
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values(): spine.set_visible(False)

plt.tight_layout()
plt.show()
