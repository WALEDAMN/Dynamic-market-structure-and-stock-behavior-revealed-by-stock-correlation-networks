# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import os

os.environ["OMP_NUM_THREADS"] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse
import time
import networkx as nx
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

# 模型和自定义函数
import model
from model import VGAERModel
from cluster import community
from Qvalue import Q

# 进度条库
from tqdm import tqdm

# 定义命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=8, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=2, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.05, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
# (修改) cluster的default值不再重要，因为我们会在代码中遍历它
parser.add_argument('--cluster', type=int, default=4, help='Number of community (will be iterated from 3 to 10).')
parser.add_argument('--gml_folder', type=str,
                    default='D:\\管科-机器学习\\基于动态网络的股票市场内在关联特征分析及其应用\\Test\\动态演化分析\\gml',
                    help='Path to the folder containing GML files.')
parser.add_argument('--patience', type=int, default=100, help='Patience for early stopping.')

args = parser.parse_args()

# 检查设备
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# 全局变量列表只用于在单次运行中临时存储Q值
q_values = []


def get_time_window_label(gml_file, index):
    date_str = gml_file.split('_')[5]
    return f"{date_str}_{index:02d}"


def match_communities(prev_communities, current_communities):
    from scipy.optimize import linear_sum_assignment
    num_communities = max(max(prev_communities), max(current_communities)) + 1
    cost_matrix = np.zeros((num_communities, num_communities))
    for i in range(len(prev_communities)):
        cost_matrix[prev_communities[i], current_communities[i]] -= 1
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    community_map = {col: row for row, col in zip(row_ind, col_ind)}
    matched_communities = [community_map.get(comm, comm) for comm in current_communities]
    return matched_communities


# (修改) vgaer函数直接接收 num_clusters 参数
def vgaer(gml_file, num_clusters, prev_communities=None):
    G = nx.read_gml(gml_file, label='label')
    A = torch.Tensor(nx.adjacency_matrix(G).todense())
    A_orig = A.detach().numpy()
    A_orig_ten = A.to(device)

    K = 1 / (A.sum().item()) * (A.sum(dim=1).reshape(A.shape[0], 1) @ A.sum(dim=1).reshape(1, A.shape[0]))
    B = A - K
    B = B.to(device)

    A = A + torch.eye(A.shape[0])
    D = torch.diag(torch.pow(A.sum(dim=1), -0.5))
    A_hat = D @ A @ D
    A_hat = A_hat.to(device)
    A = A.to(device)

    feats = B
    in_dim = feats.shape[-1]
    vgaer_model = model.VGAERModel(in_dim, args.hidden1, args.hidden2, device)
    vgaer_model = vgaer_model.to(device)
    optimizer = torch.optim.Adam(vgaer_model.parameters(), lr=args.lr)

    def compute_loss_para(adj):
        pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        weight_mask = adj.view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0)).to(device)
        weight_tensor[weight_mask] = pos_weight
        return weight_tensor, norm

    weight_tensor, norm = compute_loss_para(A)

    epochs_no_improve = 0
    min_loss = float('inf')
    best_hidemb = None

    for epoch in range(args.epochs):
        vgaer_model.train()
        recovered = vgaer_model.forward(A_hat, feats)
        logits = torch.sigmoid(recovered[0])
        hidemb = recovered[1]

        loss = norm * F.binary_cross_entropy(logits.view(-1), A_orig_ten.view(-1), weight=weight_tensor)
        kl_divergence = 0.5 / logits.size(0) * (
                1 + 2 * vgaer_model.log_std - vgaer_model.mean ** 2 - torch.exp(vgaer_model.log_std) ** 2).sum(1).mean()
        loss -= kl_divergence

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < min_loss:
            min_loss = loss.item()
            epochs_no_improve = 0
            best_hidemb = hidemb.detach().cpu()
        else:
            epochs_no_improve += 1

        if epoch > 200 and epochs_no_improve >= args.patience:
            break

    if best_hidemb is not None:
        # (修改) community 和 eye 函数使用传入的 num_clusters 参数
        commu_pred = community(best_hidemb, num_clusters, None, None)
        if prev_communities is not None:
            commu_pred = match_communities(prev_communities, commu_pred)
        Q_value = Q(A_orig, np.eye(num_clusters)[commu_pred])
        q_values.append(Q_value)
    else:
        print(f"警告: 文件 {gml_file} 的训练未能产生有效结果。")
        commu_pred = None

    return commu_pred, G


if __name__ == '__main__':
    # 预先加载所有gml文件名
    gml_files = sorted([f for f in os.listdir(args.gml_folder) if f.endswith('.gml')])
    print(f"总共找到 {len(gml_files)} 个GML文件进行分析。")

    # ===================================================================================
    # (新增) 创建一个列表来存储最终结果
    final_results = []

    # (新增) 外层循环，遍历聚类数 k 从 3 到 10
    for k in range(3, 11):
        print("\n" + "=" * 60)
        print(f"开始计算: 聚类数 (k) = {k}")
        print("=" * 60)

        # (修改) 在每次k值循环开始时，重置q_values和prev_communities
        q_values = []
        prev_communities = None

        # 内部循环处理所有gml文件
        # 使用tqdm创建进度条
        for index, gml_file in enumerate(tqdm(gml_files, desc=f"处理文件 (k={k})"), start=1):
            # 将当前的k值传递给vgaer函数
            current_communities, G = vgaer(os.path.join(args.gml_folder, gml_file),
                                           num_clusters=k,
                                           prev_communities=prev_communities)

            if current_communities is not None:
                prev_communities = current_communities

        # 计算当前k值下的平均模块度
        if q_values:
            average_q = sum(q_values) / len(q_values)
            # 立即打印当前k值的结果
            print(f"计算完成 (k={k}): 平均模块度 (Q值) = {average_q:.6f}")
            # 将结果存入最终列表
            final_results.append((k, average_q))
        else:
            print(f"计算失败 (k={k}): 未能计算任何模块度值。")
            final_results.append((k, 'N/A'))

    # (新增) 所有循环结束后，打印一个漂亮的总结表格
    print("\n\n" + "#" * 60)
    print("所有聚类数遍历完成，结果汇总如下：")
    print("#" * 60)
    print(f"{'聚类数 (k)':<15} | {'平均模块度 (Q值)':<20}")
    print("-" * 40)
    for k, avg_q in final_results:
        if isinstance(avg_q, float):
            print(f"{k:<15} | {avg_q:<20.6f}")
        else:
            print(f"{k:<15} | {avg_q:<20}")
    print("#" * 60)
    # ===================================================================================