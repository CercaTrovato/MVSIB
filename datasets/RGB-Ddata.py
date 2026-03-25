import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib import rcParams

# English explanation comment.
rcParams['font.family'] = 'Times New Roman'
rcParams['axes.unicode_minus'] = False
# ====================================================

# ================================
# English explanation comment.
# ================================
mapping = sio.loadmat("class13Mapping.mat")
data = mapping["classMapping13"][0][0]
label_ids = data[0].squeeze()
label_names = [n[0] for n in data[1][0]]

df_map = pd.DataFrame({
    "LabelID": np.arange(1, len(label_names) + 1),
    "ClassName": label_names
})
print("✅ 解析出的 13 类映射关系如下：")
print(df_map)

# ================================
# English explanation comment.
# ================================
rgbd = sio.loadmat("RGB-D.mat")
X_raw = rgbd["X"].squeeze()
Y = rgbd["Y"].squeeze().astype(int)
print("\n📘 X 结构信息：")
print("X type:", type(X_raw))
print("X dtype:", X_raw.dtype)
print("X shape:", X_raw.shape)

# ================================
# English explanation comment.
# ================================
if len(X_raw.shape) == 1 and len(X_raw) == 2:
    print("\n检测到 X_raw 含有 2 个模态 (image/text)，逐模态展开中…")
    image_features = np.array(X_raw[0])
    text_features = np.array(X_raw[1])
else:
    print("\n检测到 X_raw 是样本级二元组，逐样本解析中…")
    image_features = np.array([sample[0].flatten() for sample in X_raw])
    text_features = np.array([sample[1].flatten() for sample in X_raw])

print(f"Image features shape: {image_features.shape}")
print(f"Text features shape: {text_features.shape}")
print(f"Y shape: {Y.shape}")

# ================================
# English explanation comment.
# ================================
unique, counts = np.unique(Y, return_counts=True)
proportions = counts / counts.sum()
df_stats = pd.DataFrame({
    "LabelID": unique,
    "ClassName": [label_names[i-1] for i in unique],
    "Count": counts,
    "Proportion (%)": np.round(proportions * 100, 2)
})
print("\n📊 数据集类别统计：")
print(df_stats)

# English explanation comment.
plt.figure(figsize=(10,5))
plt.bar(df_stats["ClassName"], df_stats["Count"], color="skyblue")
plt.xticks(rotation=45)
plt.title("Class Sample Count")
plt.xlabel("Class Name")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# English explanation comment.
plt.figure(figsize=(8,8))
plt.pie(df_stats["Count"], labels=df_stats["ClassName"], autopct='%1.1f%%', startangle=140)
plt.title("Class Distribution (%)")
plt.tight_layout()
plt.show()

# ================================
# English explanation comment.
# ================================
pca_image = PCA(n_components=2)
image_pca = pca_image.fit_transform(image_features)
plt.figure(figsize=(8,6))
plt.scatter(image_pca[:,0], image_pca[:,1], c=Y, cmap="tab10", s=20)
plt.title("PCA of Image Features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Class")
plt.show()

# ================================
# English explanation comment.
# ================================
pca_text = PCA(n_components=2)
text_pca = pca_text.fit_transform(text_features)
plt.figure(figsize=(8,6))
plt.scatter(text_pca[:,0], text_pca[:,1], c=Y, cmap="tab10", s=20)
plt.title("PCA of Text Features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Class")
plt.show()

# ================================
# English explanation comment.
# ================================
print("\n📈 图像特征统计：")
print("Mean:", np.round(np.mean(image_features, axis=0)[:5], 3))
print("Std:", np.round(np.std(image_features, axis=0)[:5], 3))

print("\n📈 文本特征统计：")
print("Mean:", np.round(np.mean(text_features, axis=0)[:5], 3))
print("Std:", np.round(np.std(text_features, axis=0)[:5], 3))

# ================================
# English explanation comment.
# ================================
# English explanation comment.
corr_img = np.corrcoef(image_features.T)
plt.figure(figsize=(10,7))
sns.heatmap(corr_img, cmap="coolwarm", cbar=False)
plt.title("Correlation Heatmap - Image Features")
plt.tight_layout()
plt.show()

# English explanation comment.
corr_text = np.corrcoef(text_features.T)
plt.figure(figsize=(10,7))
sns.heatmap(corr_text, cmap="coolwarm", cbar=False)
plt.title("Correlation Heatmap - Text Features")
plt.tight_layout()
plt.show()

# ================================
# English explanation comment.
# ================================
combined_features = np.concatenate([image_features, text_features], axis=1)
print(f"\n🔗 Combined feature shape: {combined_features.shape}")

pca_combined = PCA(n_components=2)
combined_pca = pca_combined.fit_transform(combined_features)
plt.figure(figsize=(8,6))
plt.scatter(combined_pca[:,0], combined_pca[:,1], c=Y, cmap="tab10", s=20)
plt.title("PCA of Combined Features (Image + Text)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Class")
plt.show()

# ================================
# English explanation comment.
# ================================
corr_combined = np.corrcoef(combined_features.T)
plt.figure(figsize=(10,7))
sns.heatmap(corr_combined, cmap="coolwarm", cbar=False)
plt.title("Correlation Heatmap - Combined Features")
plt.tight_layout()
plt.show()

print("\n✅ 完成所有 EDA 分析！")
