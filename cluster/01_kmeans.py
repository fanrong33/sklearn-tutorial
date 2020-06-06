# encoding: utf-8
""" k-means 聚类
http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html

cd /Users/fanrong33/kuaipan/github/sklearn-tutorial/cluster/
python 01_kmeans.py

"""

# Author: Phil Roth <mr.phil.roth@gmail.com>
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


plt.figure(figsize=(6, 6))


# 默认随机创建标准方差的3种类型斑点
n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)
print(X)  # shape (1500, 2)
'''
[[ -5.19811282e+00   6.41869316e-01]
 [ -5.75229538e+00   4.18627111e-01]
 [ -1.08448984e+01  -7.55352273e+00]
 ...,
 [  1.36105255e+00  -9.07491863e-01]
 [ -3.54141108e-01   7.12241630e-01]
 [  1.88577252e+00   1.41185693e-03]]
'''
print(y)
''' [1 1 0 ..., 2 2 2] '''


# 1、错误聚簇分类示例，将 3 聚簇分为 2 聚簇
y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)
print(y_pred)
''' [0 0 1 ..., 0 0 0] '''
plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)  # 对c的理解还不够！！
plt.title("Incorrect Number of Blobs")


# 2、Anisotropicly distributed data  各向异性分布式数据
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X, transformation)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)

plt.subplot(222)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
plt.title("Anisotropicly Distributed Blobs")


# 3、Different variance  不同方差的斑点数据
X_varied, y_varied = make_blobs(n_samples=n_samples,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=random_state)
print(X_varied)
'''
[[ -6.11119721   1.47153062]
 [ -7.49665361   0.9134251 ]
 [-10.84489837  -7.55352273]
 ...,
 [  1.64990343  -0.20117787]
 [  0.79230661   0.60868888]
 [  1.91226342   0.25327399]]
'''
print(y_varied)
''' [1 1 0 ..., 2 2 2] '''
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)
print(y_pred)
''' [2 2 1 ..., 0 0 0] '''
plt.subplot(223)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
plt.title("Unequal Variance")  # 不等方差


# 4、Unevenly sized blobs 过滤版本的示例
X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_filtered)

plt.subplot(224)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
plt.title("Unevenly Sized Blobs")

plt.show()


