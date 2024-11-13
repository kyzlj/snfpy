import pandas as pd
from snf import datasets
import snf
import numpy as np
from sklearn.cluster import spectral_clustering
from sklearn.metrics import v_measure_score

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
affinity_networks = [snf.make_affinity(data, metric='euclidean', K=20, mu=0.5) for data in data_views]

# 融合相似性网络
fused_network = snf(affinity_networks, K=20)

# 获取最佳聚类数
best, second = snf.get_n_clusters(fused_network)
print("Best number of clusters:", best)
print("Second best number of clusters:", second)

# 对融合网络进行聚类
labels = spectral_clustering(fused_network, n_clusters=best)

# 如果有标签数据，可以计算V-measure（这里假设生存数据中的某一列是标签）
# 注意：请根据实际情况调整标签列
true_labels = survival_data['label_column'].values  # 请将'label_column'替换为实际标签列名称
score = v_measure_score(labels, true_labels)
print("V-measure score:", score)
