from network import Network
from metric import valid
from model import *
import numpy as np
import argparse

# ============================================================================
# test.py
# ----------------------------------------------------------------------------
# 评估脚本：加载训练好的模型参数，在指定数据集上执行 valid() 指标评估。
# 注意：该脚本不做训练，只负责“构建同构网络 + 加载 checkpoint + 测评”。
# ============================================================================

Dataname = 'Cora'
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--dataset', default=Dataname, help = '[CCV, RGB-D, Cora, ALOI-100, Hdigit, Digit-Product]')
parser.add_argument('--load_model', default=False, help='Testing if True or training.')
# feature_dim 必须与训练阶段保持一致，否则权重形状不匹配
parser.add_argument("--feature_dim", default=256)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 读取多视图数据（每个视图将被组织为一个 Tensor）
mv_data = MultiviewData(args.dataset, device)
num_views = len(mv_data.data_views)
num_samples = mv_data.labels.size
num_clusters = np.unique(mv_data.labels).size
# input_sizes[v] = 第 v 个视图输入维度 D_v
input_sizes = np.zeros(num_views, dtype=int)
for idx in range(num_views):
    input_sizes[idx] = mv_data.data_views[idx].shape[1]

# 构建与训练期同结构的网络实例
network = Network(num_views, num_samples, num_clusters, device, input_sizes, args.feature_dim)
network = network.to(device)
# 加载模型参数：支持两种格式（纯 state_dict / 包含 network_state_dict 的字典）
checkpoint = torch.load('./models/%s.pth' % args.dataset, map_location=device)

if isinstance(checkpoint, dict) and 'network_state_dict' in checkpoint:
    network.load_state_dict(checkpoint['network_state_dict'])
else:
    network.load_state_dict(checkpoint)
print("Dataset:{}".format(args.dataset))
print("Loading models...")
# valid() 会打印/返回 ACC/NMI/ARI/PUR/F1 等聚类指标
valid(network, mv_data, num_samples, num_clusters)
