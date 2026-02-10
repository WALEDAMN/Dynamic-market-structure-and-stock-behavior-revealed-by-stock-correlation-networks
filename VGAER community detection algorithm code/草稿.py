from __future__ import division
from __future__ import print_function

import os
os.environ["OMP_NUM_THREADS"] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse
import time
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from matplotlib import pyplot as plt
from torch import optim
import pandas as pd
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset , PubmedGraphDataset
import csv
# import scanpy
import model
from model import VGAERModel
import torch.nn.functional as F
from cluster import community
from NMI import load_label, NMI, label_change
from Qvalue import Q

# 定义命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=8, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=2, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.05, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='stock', help='type of dataset.')
parser.add_argument('--cluster', type=int, default=4, help='Number of community')
parser.add_argument('--gml_folder', type=str, default='D:\\管科-机器学习\\基于动态网络的股票市场内在关联特征分析及其应用\\Test\\新建文件夹', help='Path to the folder containing GML files.')
parser.add_argument('--output_folder', type=str, default='D:\\管科-机器学习\\基于动态网络的股票市场内在关联特征分析及其应用\\Test\\社区划分', help='Output folder for results.')
args = parser.parse_args()

# 检查设备
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# 定义函数来计算和存储 Q 值
q_values = []
community_changes = []
community_assignments = []

# 用于存储所有时间窗口的社区划分和变化
community_assignments_dict = {}
community_changes_dict = {}
node_labels = []

def get_time_window_label(gml_file, index):
    date_str = gml_file.split('_')[5]
    return f"{date_str}_{index:02d}"

def match_communities(prev_communities, current_communities):
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    # Create a cost matrix
    num_communities = max(max(prev_communities), max(current_communities)) + 1
    cost_matrix = np.zeros((num_communities, num_communities))

    for i in range(len(prev_communities)):
        cost_matrix[prev_communities[i], current_communities[i]] -= 1

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    community_map = {col: row for row, col in zip(row_ind, col_ind)}

    matched_communities = [community_map[comm] for comm in current_communities]

    return matched_communities

def vgaer(gml_file, time_window_label, output_folder, prev_communities=None):
    # 读取GML文件
    G = nx.read_gml(gml_file, label='label')
    A = torch.Tensor(nx.adjacency_matrix(G).todense())
    print(f"Processing {gml_file}")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    global node_labels
    if not node_labels:
        node_labels = list(G.nodes())
        node_labels = [label[:6] for label in node_labels]  # 仅需要标签的前6位

    A_orig = A.detach().numpy()
    A_orig_ten = A.to(device)

    # 计算B矩阵
    K = 1 / (A.sum().item()) * (A.sum(dim=1).reshape(A.shape[0], 1) @ A.sum(dim=1).reshape(1, A.shape[0]))
    B = A - K
    B = B.to(device)

    # 计算A_hat矩阵
    A = A + torch.eye(A.shape[0])
    D = torch.diag(torch.pow(A.sum(dim=1), -0.5))  # D = D^-1/2
    A_hat = D @ A @ D
    A_hat = A_hat.to(device)
    A = A.to(device)

    feats = B
    in_dim = feats.shape[-1]

    # 创建模型
    vgaer_model = model.VGAERModel(in_dim, args.hidden1, args.hidden2, device)
    vgaer_model = vgaer_model.to(device)

    # 创建训练组件
    optimizer = torch.optim.Adam(vgaer_model.parameters(), lr=args.lr)
    print('Total Parameters:', sum([p.nelement() for p in vgaer_model.parameters()]))

    def compute_loss_para(adj):
        pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        weight_mask = adj.view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0)).to(device)
        weight_tensor[weight_mask] = pos_weight
        return weight_tensor, norm

    weight_tensor, norm = compute_loss_para(A)
    print(f"Weight Tensor:\n{weight_tensor}")
    print(f"Norm: {norm}")

    # 训练循环
    for epoch in range(args.epochs):
        vgaer_model.train()
        recovered = vgaer_model.forward(A_hat, feats)
        logits = torch.sigmoid(recovered[0])  # 应用Sigmoid激活函数
        hidemb = recovered[1]

        loss = norm * F.binary_cross_entropy(logits.view(-1), A_orig_ten.view(-1), weight=weight_tensor)
        kl_divergence = 0.5 / logits.size(0) * (
                1 + 2 * vgaer_model.log_std - vgaer_model.mean ** 2 - torch.exp(vgaer_model.log_std) ** 2).sum(1).mean()
        loss -= kl_divergence
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == args.epochs - 1:
            hidemb = hidemb.cpu()
            save_path = os.path.join(output_folder, f"{time_window_label}.png")
            commu_pred = community(hidemb, args.cluster, save_path, time_window_label)
            if prev_communities is not None:
                commu_pred = match_communities(prev_communities, commu_pred)
            Q_value = Q(A_orig, np.eye(args.cluster)[commu_pred])
            q_values.append((time_window_label, Q_value))
            community_assignments_dict[time_window_label] = {label[:6]: commu_pred[node] for node, label in enumerate(G.nodes())}
            print(f"Q value for {time_window_label}: {Q_value}")
    return commu_pred, G

if __name__ == '__main__':
    prev_communities = None
    prev_graph = None
    for index, gml_file in enumerate(sorted(os.listdir(args.gml_folder)), start=1):
        if gml_file.endswith('.gml'):
            time_window_label = get_time_window_label(gml_file, index)
            print(time_window_label)
            current_communities, G = vgaer(os.path.join(args.gml_folder, gml_file), time_window_label, args.output_folder)
            if prev_communities is not None:
                changes = {}
                for node, label in enumerate(G.nodes()):
                    short_label = label[:6]
                    if current_communities[node] != prev_communities[node]:
                        changes[short_label] = f"{prev_communities[node]}->{current_communities[node]}"
                    else:
                        changes[short_label] = str(current_communities[node])
                community_changes_dict[time_window_label] = changes
            prev_communities = current_communities
            prev_graph = G


    # 创建输出文件夹
    os.makedirs(args.output_folder, exist_ok=True)

    # 确保输出文件名具有正确的扩展名
    output_excel_q = os.path.join(args.output_folder, 'q_values.xlsx')
    output_excel_changes = os.path.join(args.output_folder, 'community_changes.xlsx')

    # 将Q值保存到Excel文件中
    df_q = pd.DataFrame(q_values, columns=['Time Window', 'Q Value'])
    df_q.to_excel(output_excel_q, index=False)

    # 将社区变化保存到另一个Excel文件中
    df_changes = pd.DataFrame.from_dict(community_changes_dict, orient='index').reindex(columns=node_labels)
    df_changes.index.name = 'Time Window'
    df_changes.reset_index(inplace=True)
    df_changes.to_excel(output_excel_changes, index=False)

    # 将每个时间窗口的社区分配分别保存到单独的Excel文件中
    for time_window, assignments in community_assignments_dict.items():
        community_dict = {}
        for node, community in assignments.items():
            if community not in community_dict:
                community_dict[community] = []
            community_dict[community].append(node)
            df_assignments = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in community_dict.items()]))
            df_assignments = df_assignments.reindex(sorted(df_assignments.columns), axis=1)  # 社区编号从小到大排列
            output_excel_assignments = os.path.join(args.output_folder, f'community_assignments_{time_window}.xlsx')
            df_assignments.to_excel(output_excel_assignments, index=False)

        print(f"Q values have been saved to {output_excel_q}")
        print(f"Community changes have been saved to {output_excel_changes}")
        print(f"Community assignments have been saved to individual files in {args.output_folder}")