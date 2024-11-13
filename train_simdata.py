from snf import datasets, compute, metrics
from sklearn import cluster
import numpy as np

simdata = datasets.load_simdata()
print(sorted(simdata.keys()))  # 输出: ['data', 'labels']

# 查看数据集的结构
n_dtypes = len(simdata.data)
n_samp = len(simdata.labels)
print('Simdata has {} datatypes with {} samples each.'.format(n_dtypes, n_samp))

# 将原始数据数组转换为相似性矩阵（亲和力矩阵）
affinities = compute.make_affinity(simdata.data, metric='euclidean', K=20, mu=0.5)

# 融合相似性矩阵
fused = compute.snf(affinities, K=20)

# 估计聚类数
first, second = compute.get_n_clusters(fused)
print("Best number of clusters:", first)
print("Second best number of clusters:", second)

# 应用聚类算法（这里使用谱聚类）
fused_labels = cluster.spectral_clustering(fused, n_clusters=first)

# 将融合矩阵的聚类结果与原始矩阵单独聚类的结果进行比较
labels = [simdata.labels, fused_labels]
for arr in affinities:
    labels += [cluster.spectral_clustering(arr, n_clusters=first)]

# 计算聚类的标准化互信息（NMI）
nmi = metrics.nmi(labels)
print("NMI matrix:\n", nmi)

# 使用 silhouette score 评估聚类的好坏
np.fill_diagonal(fused, 0)  # 将融合矩阵的对角线设为 0
sil = metrics.silhouette_score(fused, fused_labels)
print('Silhouette score for the fused matrix is: {:.2f}'.format(sil))
