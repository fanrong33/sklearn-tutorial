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
train_set_scaled = standard_scaler.transform(x)
print(train_set_scaled)
'''
[[ 0.         -1.22474487  1.33630621]
 [ 1.22474487  0.         -0.26726124]
 [-1.22474487  1.22474487 -1.06904497]]
'''

test = np.array([[2., 3., 1.],
                 [0., 1., 5.]])
test_set_scaled = standard_scaler.transform(test)
print(test_set_scaled)
'''
[[ 1.22474487  3.67423461  0.53452248]
 [-1.22474487  1.22474487  3.74165739]]
'''


