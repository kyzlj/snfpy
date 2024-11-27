import pandas as pd
from snf import compute, metrics
import snf
import numpy as np
from sklearn import cluster
from lifelines import CoxPHFitter

# 读取每个数据文件
gene_expression = pd.read_csv('snf/tests/data/GBM/GLIO_Gene_Expression.csv', index_col=0)
methy_expression = pd.read_csv('snf/tests/data/GBM/GLIO_Methy_Expression.csv', index_col=0)
mirna_expression = pd.read_csv('snf/tests/data/GBM/GLIO_Mirna_Expression.csv', index_col=0)
survival_data = pd.read_csv('snf/tests/data/GBM/GLIO_Survival.csv', index_col=0)

# # 检查每个数据的形状
# print("Gene Expression shape:", gene_expression.shape)
# print("Methylation shape:", methy_expression.shape)
# print("miRNA Expression shape:", mirna_expression.shape)
# print("Survival Data shape:", survival_data.shape)
#
# # 打印生存数据的列名，确认列名
# print("Survival Data columns:", survival_data.columns)

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

# 为每个视角的数据分别进行聚类
labels_dict = {
    'mRNA': cluster.spectral_clustering(affinities[0], n_clusters=first),
    'DNA_methylation': cluster.spectral_clustering(affinities[1], n_clusters=first),
    'miRNA': cluster.spectral_clustering(affinities[2], n_clusters=first),
    'SNF_fusion': fused_labels
}

# 使用 CoxPHFitter 进行生存分析
coxph = CoxPHFitter()

# 使用实际的生存数据列名
survival_status_column = 'Death'  # 生存状态列名
survival_time_column = 'Survival'  # 生存时间列名

if survival_status_column not in survival_data.columns or survival_time_column not in survival_data.columns:
    raise KeyError(f"Cannot find columns '{survival_status_column}' or '{survival_time_column}' in survival data.")

# 确保生存状态列是整数类型
survival_data[survival_status_column] = survival_data[survival_status_column].astype(int)

p_values = {}

for key, labels in labels_dict.items():
    # 创建生存数据框，将聚类标签添加到生存数据中
    survival_data_with_labels = survival_data.copy()
    survival_data_with_labels['cluster'] = labels

    # 拟合 Cox 模型
    coxph.fit(survival_data_with_labels, duration_col=survival_time_column, event_col=survival_status_column,
              formula="cluster")

    # 获取 P 值
    summary = coxph.summary
    p_value = summary.loc['cluster', 'p']
    p_values[key] = p_value
    print(f"Cox log-rank test P-value for {key}: {p_value:.4g}")

# 输出所有 P 值
print("\nP-values for each data type:")
for key, p_value in p_values.items():
    print(f"{key}: {p_value:.4g}")
