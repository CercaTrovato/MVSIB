import numpy as np
import matplotlib.pyplot as plt

# English explanation comment.
original_counts = np.array([300, 200, 50, 20, 10])
balanced_counts = np.full_like(original_counts, original_counts.max())  # Pad each cluster to the size of the largest cluster

K = len(original_counts)
clusters = np.arange(1, K + 1)

# English explanation comment.
fig = plt.figure(figsize=(8, 4))

# English explanation comment.
fig_width_cm = 8 * 2.54  # Canvas width (cm)
gap_cm = 2               # Desired gap (cm)
gap_frac = gap_cm / fig_width_cm  # Fraction in [0,1]

# English explanation comment.
margin_frac = 0.05
# English explanation comment.
ax_width = (1 - 2 * margin_frac - gap_frac) / 2
ax_height = 0.8
bottom_frac = 0.1

# English explanation comment.
ax1 = fig.add_axes([margin_frac, bottom_frac, ax_width, ax_height])
ax1.bar(clusters, original_counts, color='C0', alpha=0.7)
ax1.set_title('原始样本数分布')
ax1.set_xlabel('簇标签 k')
ax1.set_ylabel('样本数 nₖ')
ax1.set_xticks(clusters)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# English explanation comment.
x0 = margin_frac + ax_width + gap_frac
ax2 = fig.add_axes([x0, bottom_frac, ax_width, ax_height])
ax2.bar(clusters, balanced_counts, color='C1', alpha=0.7)
ax2.set_title('补齐后样本数分布')
ax2.set_xlabel('簇标签 k')
ax2.set_xticks(clusters)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.show()
