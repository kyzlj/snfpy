from snf import datasets
import snf
import numpy as np
from sklearn.cluster import spectral_clustering
from sklearn.metrics import v_measure_score

digits = datasets.load_digits()

# 查看数据集的键
print(digits.keys())  # 输出: dict_keys(['data', 'labels'])

# 检查每个数据数组的形状
for arr in digits.data:
    print(arr.shape)

# 检查标签分布
groups, samples = np.unique(digits.labels, return_counts=True)
for grp, count in zip(groups, samples):
    print('Group {:.0f}: {} samples'.format(grp, count))

# 构建相似性网络
affinity_networks = snf.make_affinity(digits.data, metric='euclidean', K=20, mu=0.5)

# 融合相似性网络
fused_network = snf.snf(affinity_networks, K=20)

# 获取最佳聚类数
best, second = snf.get_n_clusters(fused_network)
print("Best number of clusters:", best)
print("Second best number of clusters:", second)

# 对融合网络进行聚类
labels = spectral_clustering(fused_network, n_clusters=best)

# 评估聚类结果
score = v_measure_score(labels, digits.labels)
print("V-measure score:", score)
