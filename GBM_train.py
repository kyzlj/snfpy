import pandas as pd
from snf import  compute, metrics
import snf
import numpy as np
from sklearn import cluster
import numpy as np

# 读取每个数据文件
gene_expression = pd.read_csv('snf/tests/data/GBM/GLIO_Gene_Expression.csv', index_col=0)
methy_expression = pd.read_csv('snf/tests/data/GBM/GLIO_Methy_Expression.csv', index_col=0)
mirna_expression = pd.read_csv('snf/tests/data/GBM/GLIO_Mirna_Expression.csv', index_col=0)
survival_data = pd.read_csv('snf/tests/data/GBM/GLIO_Survival.csv', index_col=0)

# 检查每个数据的形状
print("Gene Expression shape:", gene_expression.shape)
print("Methylation shape:", methy_expression.shape)
print("miRNA Expression shape:", mirna_expression.shape)
print("Survival Data shape:", survival_data.shape)

# 假设每个数据都是不同的视角，我们将其转换为相似性矩阵
data_views = [gene_expression.values, methy_expression.values, mirna_expression.values]

# 为每个视角创建相似性网络
affinities = [snf.make_affinity(data, metric='euclidean', K=20, mu=0.5) for data in data_views]

# 融合相似性网络
fused = compute.snf(affinities, K=20)

# 估计聚类数
first, second = compute.get_n_clusters(fused)
print("Best number of clusters:", first)
print("Second best number of clusters:", second)

# 应用聚类算法（这里使用谱聚类）
fused_labels = cluster.spectral_clustering(fused, n_clusters=first)

# 将融合矩阵的聚类结果与原始矩阵单独聚类的结果进行比较
labels = [data_views.labels, fused_labels]
for arr in affinities:
    labels += [cluster.spectral_clustering(arr, n_clusters=first)]

# 计算聚类的标准化互信息（NMI）
nmi = metrics.nmi(labels)
print("NMI matrix:\n", nmi)

# 使用 silhouette score 评估聚类的好坏
np.fill_diagonal(fused, 0)  # 将融合矩阵的对角线设为 0
sil = metrics.silhouette_score(fused, fused_labels)
print('Silhouette score for the fused matrix is: {:.2f}'.format(sil))



