# -*- coding: utf-8 -*-

import os
import networkx as nx
import numpy as np
import pandas as pd

from sklearn.cluster import SpectralClustering, KMeans, MeanShift
from sklearn.mixture import GaussianMixture

from networkx.algorithms.community.quality import modularity
from networkx.algorithms.community import asyn_lpa_communities

from sklearn.preprocessing import StandardScaler

# === 读取所有GML文件（自然排序） ===
def sorted_gml_files(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith(".gml")]
    files.sort()  # 可改为自然排序
    return [os.path.join(folder_path, f) for f in files]

# === 将标签转换为社区格式（用于计算Q值） ===
def labels_to_communities(labels, nodes):
    communities = {}
    for node, label in zip(nodes, labels):
        communities.setdefault(label, []).append(node)
    return list(communities.values())

# === 计算模块度Q值 ===
def compute_modularity(G, labels):
    nodes = list(G.nodes())
    communities = labels_to_communities(labels, nodes)
    return modularity(G, communities)

# === 主函数 ===
def clustering_q_for_gmls(folder_path, output_excel):
    files = sorted_gml_files(folder_path)
    result = {
        'time_window': [],
        'spectral': [],
        'kmeans': [],
        'gmm': [],
        'meanshift': []
    }

    for file in files:
        G = nx.read_gml(file)
        A = nx.to_numpy_array(G)
        scaler = StandardScaler()
        A_scaled = scaler.fit_transform(A)

        n_clusters = 4  # 默认聚类数

        # 节点数记录
        n_nodes = A.shape[0]
        time_label = os.path.splitext(os.path.basename(file))[0]

        # 聚类方法
        try:
            spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0).fit_predict(A)
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(A_scaled)
            gmm = GaussianMixture(n_components=n_clusters, random_state=0).fit(A_scaled).predict(A_scaled)
            meanshift = MeanShift().fit(A_scaled).labels_
        except Exception as e:
            print(f"聚类失败：{file}, 错误：{e}")
            continue

        # 计算Q值
        q_spectral = compute_modularity(G, spectral)
        q_kmeans = compute_modularity(G, kmeans)
        q_gmm = compute_modularity(G, gmm)
        q_meanshift = compute_modularity(G, meanshift)

        # 保存结果
        result['time_window'].append(time_label)
        result['spectral'].append(q_spectral)
        result['kmeans'].append(q_kmeans)
        result['gmm'].append(q_gmm)
        result['meanshift'].append(q_meanshift)

        print(f"{time_label} 完成")

    # 导出Excel
    df = pd.DataFrame(result)
    df.to_excel(output_excel, index=False)
    print(f"已保存至：{output_excel}")

# === 示例调用 ===
if __name__ == '__main__':
    folder_path = 'D:\\管科-机器学习\\基于动态网络的股票市场内在关联特征分析及其应用\\Test\\动态演化分析\\gml'  # 修改为你的GML文件夹路径
    output_excel = '聚类Q值结果.xlsx'
    clustering_q_for_gmls(folder_path, output_excel)
