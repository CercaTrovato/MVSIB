from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from torch.utils.data import Dataset
import torch
import hdf5storage
import os

# ============================================================================
# dataloader.py
# ----------------------------------------------------------------------------
# 本文件负责多视图数据读取、归一化、以及 DataLoader 打包。
# 研究代码注意：
# - 不在此文件内“偷偷”改变预处理策略（会影响可复现性）；
# - 仅做确定性数据组织，避免引入额外随机性。
# ============================================================================

class MultiviewData(Dataset):
    def __init__(self, db, device, path="datasets/"):
        """
        参数:
        - db: 数据集名称（RGB-D / CCV / Cora / Hdigit / prokaryotic）。
        - device: 目标设备，数据会在加载后转到该设备。
        - path: 数据目录。

        输出对象关键属性:
        - self.data_views: list[Tensor]，每个元素形状 [N, D_v]。
        - self.labels: ndarray[N]，类别标签。
        - self.num_views: 视图数量 V。
        """
        self.data_views = []

        if db == "RGB-D":
            mat = hdf5storage.loadmat(os.path.join(path, 'RGB-D.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            # Min-Max 归一化：将每维压到 [0,1]，减少视图间尺度偏置
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.squeeze(mat['Y']).astype(np.int32)

        elif db == 'CCV':
            mat = hdf5storage.loadmat(os.path.join(path, 'CCV.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.squeeze(mat['Y']).astype(np.int32)

        elif db == 'Cora':
            mat = hdf5storage.loadmat(os.path.join(path, 'Cora.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.squeeze(mat['Y']).astype(np.int32)

        elif db == 'Hdigit':
            mat = hdf5storage.loadmat(os.path.join(path, 'Hdigit.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.squeeze(mat['Y']).astype(np.int32)

        elif db == 'prokaryotic':
            # 1) 读取 mat 文件
            mat = hdf5storage.loadmat(os.path.join(path, 'prokaryotic.mat'))

            # 2) 动态识别所有视图字段（排除元数据 __*__ 和标签字段 truth）
            view_keys = [k for k in mat.keys()
                         if not k.startswith('__') and k != 'truth']
            if not view_keys:
                raise KeyError("prokaryotic.mat 中未找到任何视图字段，请检查字段名。")

            # 3) 按字段加载每个视图
            for key in view_keys:
                arr = mat[key]
                # 如果为稀疏格式，先转换 dense
                if hasattr(arr, 'toarray'):
                    arr = arr.toarray()
                arr = arr.astype(np.float32)
                # 保证样本在行上：若行数 < 列数，则转置
                if arr.shape[0] < arr.shape[1]:
                    arr = arr.T
                self.data_views.append(arr)

            self.num_views = len(self.data_views)

            # 4) 加载标签 truth
            if 'truth' not in mat:
                raise KeyError("prokaryotic.mat 中缺少 'truth' 标签字段。")
            self.labels = np.squeeze(mat['truth']).astype(np.int32)

            # 5) 对每个视图做 Min–Max 归一化
            scaler = MinMaxScaler()
            for i in range(self.num_views):
                self.data_views[i] = scaler.fit_transform(self.data_views[i])

        else:
            raise NotImplementedError(f"Dataset '{db}' not supported.")

        # 最后：转为 Tensor 并移动到指定 device
        for i in range(self.num_views):
            self.data_views[i] = torch.from_numpy(self.data_views[i]).to(device)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # index: 样本下标 i
        # sub_data_views: list[Tensor[D_v]]，同一条样本在各视图下的特征
        sub_data_views = list()
        for view_idx in range(self.num_views):
            data_view = self.data_views[view_idx]
            sub_data_views.append(data_view[index])

        return sub_data_views, self.labels[index], index


def get_multiview_data(mv_data, batch_size):
    """
    训练模式 DataLoader（shuffle=True）。
    - batch_size 大：吞吐更高，但显存占用更高。
    - 返回 num_clusters 仅用于外部日志与模型配置。
    """
    num_views = len(mv_data.data_views)
    num_samples = len(mv_data.labels)
    num_clusters = len(np.unique(mv_data.labels))

    mv_data_loader = torch.utils.data.DataLoader(
        mv_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    return mv_data_loader, num_views, num_samples, num_clusters


def get_all_multiview_data(mv_data):
    """
    全量模式 DataLoader（单 batch，shuffle=False）。
    主要用于需要全样本聚类/统计的步骤（如 E-step）。
    """
    num_views = len(mv_data.data_views)
    num_samples = len(mv_data.labels)
    num_clusters = len(np.unique(mv_data.labels))

    mv_data_loader = torch.utils.data.DataLoader(
        mv_data,
        batch_size=num_samples,
        shuffle=False,
        drop_last=False,
    )

    return mv_data_loader, num_views, num_samples, num_clusters
