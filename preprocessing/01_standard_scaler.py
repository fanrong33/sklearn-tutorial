# encoding: utf-8
""" 标准化
Standardization标准化:将特征数据的分布调整成标准正太分布，也叫高斯分布，也就是使得数据的均值为0，方差为1.
标准化的原因在于如果有些特征的方差过大，则会主导目标函数从而使参数估计器无法正确地去学习其他特征。
标准化的过程为两步：去均值的中心化（均值变为0）；方差的规模化（方差变为1）。

http://sklearn.apachecn.org/cn/0.19.0/modules/preprocessing.html

"""

import numpy as np
from sklearn.preprocessing import StandardScaler


# 创建一组特征数据，每一行表示一个样本，每一列表示一个特征
x = np.array([[1., -1., 2.],
              [2., 0., 0.],
              [0., 1., -1.]])



standard_scaler = StandardScaler()

standard_scaler.fit(x)
x_scaled = standard_scaler.transform(x)
print(x_scaled)
'''
[[ 0.         -1.22474487  1.33630621]
 [ 1.22474487  0.         -0.26726124]
 [-1.22474487  1.22474487 -1.06904497]]
'''

x2 = np.array([[2., 1., -1.]])
x_scaled2 = standard_scaler.transform(x2)
print(x_scaled2)
'''
[[ 1.22474487  1.22474487 -1.06904497]]
'''


# 注意顶部的说明，均值为0，方差为1
x3 = np.array([[1., -1., 2., 2., 0., 0., 0., 1., -1.]])

standard_scaler2 = StandardScaler()
standard_scaler2.fit(x3)
x3_scaled = standard_scaler2.transform(x3)
print(x3_scaled)
'''
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
'''


