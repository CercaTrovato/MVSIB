import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置随机种子以复现结果
np.random.seed(42)

# 生成数据：与原代码一致
n_samples = 100
n_clusters = 4
samples_per_cluster = n_samples // n_clusters
means = np.array([
    [0, 0, 0],
    [5, 0, 0],
    [2.5, 4, 0],
    [2.5, 2, 5]
])
cov = np.eye(3) * 1.0

# 样本采样
X = np.vstack([
    np.random.multivariate_normal(means[i], cov, samples_per_cluster)
    for i in range(n_clusters)
])
y = np.repeat(np.arange(n_clusters), samples_per_cluster)

# 生成紧凑版样本 X_compact
X_compact = X.copy()
for k in range(n_clusters):
    X_compact[y == k] = means[k] + 0.5 * (X[y == k] - means[k])

# 仅绘制第四幅图 (5x5 英寸)
fig = plt.figure(figsize=(3.5,3.5 ))
ax = fig.add_subplot(111, projection='3d')

colors = ['blue', 'orange', 'green', 'purple']
for k in range(n_clusters):
    pts = X_compact[y == k]
    # 散点
    ax.scatter(*pts.T, color=colors[k], alpha=0.6)
    # 球体等高线
    max_dist = np.max(np.linalg.norm(pts - means[k], axis=1))
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = means[k][0] + max_dist * np.cos(u) * np.sin(v)
    y_s = means[k][1] + max_dist * np.sin(u) * np.sin(v)
    z = means[k][2] + max_dist * np.cos(v)
    ax.plot_wireframe(x, y_s, z, color=colors[k], alpha=0.3, linewidth=0.5)
    # 簇心标记
    ax.scatter(*means[k], color=colors[k], marker='*', s=100, zorder=3, label=f'Center {k}')



# 移除坐标轴标签与刻度
plt.tight_layout()
plt.show()
