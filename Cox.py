import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sklearn.cluster import KMeans
from snf import compute

# 加载数据
gene_data = pd.read_csv('snf/tests/data/GBM/GLIO_Gene_Expression.csv', index_col=0, header=0)
methy_data = pd.read_csv('snf/tests/data/GBM/GLIO_Methy_Expression.csv', index_col=0, header=0)
mirna_data = pd.read_csv('snf/tests/data/GBM/GLIO_Mirna_Expression.csv', index_col=0, header=0)
survival_data = pd.read_csv('snf/tests/data/GBM/GLIO_Survival.csv', header=0)

# 输出数据形状
print("Gene Expression Data Shape:", gene_data.shape)
print("Methylation Data Shape:", methy_data.shape)
print("miRNA Data Shape:", mirna_data.shape)
print("Survival Data Shape:", survival_data.shape)

# 确保数据样本一致
samples = survival_data['PatientID'].values
gene_data = gene_data.loc[samples]
methy_data = methy_data.loc[samples]
mirna_data = mirna_data.loc[samples]


# 定义聚类和计算Cox P值的函数
def cluster_and_cox_p_value(data, n_clusters, survival_data):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(data)
    survival_data['Cluster'] = clusters

    # 计算Cox对数秩检验的P值
    cph = CoxPHFitter()
    cph.fit(survival_data, duration_col='Duration', event_col='Event', formula="Cluster")
    return cph.summary.loc['Cluster', 'p']


# 对每种数据类型进行聚类并计算Cox P值
p_values = {}
n_clusters_dict = {'GBM': 3, 'BIC': 5, 'KRCCC': 3, 'LSCC': 4, 'COAD': 3}

# 基因表达数据
p_values['mRNA'] = cluster_and_cox_p_value(gene_data, n_clusters_dict['GBM'], survival_data)

# DNA甲基化数据
p_values['DNA_methylation'] = cluster_and_cox_p_value(methy_data, n_clusters_dict['GBM'], survival_data)

# miRNA数据
p_values['miRNA'] = cluster_and_cox_p_value(mirna_data, n_clusters_dict['GBM'], survival_data)

# 相似性网络融合（SNF）
gene_similarity = compute.construct_similarity_matrix(gene_data)
methy_similarity = compute.construct_similarity_matrix(methy_data)
mirna_similarity = compute.construct_similarity_matrix(mirna_data)

# 执行SNF
snf_matrix = compute.snf([gene_similarity, methy_similarity, mirna_similarity], K=20)

# 对SNF后的数据进行聚类和Cox P值计算
snf_clusters = KMeans(n_clusters=n_clusters_dict['GBM'], random_state=0).fit_predict(snf_matrix)
survival_data['SNF_Cluster'] = snf_clusters
p_values['SNF'] = cluster_and_cox_p_value(snf_matrix, n_clusters_dict['GBM'], survival_data)

# 输出P值结果
print("P values from Cox log-rank test:")
print(p_values)
