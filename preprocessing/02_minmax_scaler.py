# encoding: utf-8
"""  归一化 将特征缩放到一个范围内（0，1）
缩放特征到给定的最小值到最大值之间，通常在0到1之间。或则使得每个特征的最大绝对值被缩放到单位大小。
这可以分别使用MinMaxScaler或MaxAbsScaler函数实现。

MinMaxScaler 参数feature_range=(0, 1)数据集的分布范围, copy=True 副本
计算公式如下：
X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X_scaled = X_std * (max - min) + min


https://blog.csdn.net/cheng9981/article/details/61650879

"""

import numpy as np

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler


x = np.array([[10, 11, 12], 
              [13, 10, 12], 
              [8,  9,  11], 
              [12, 13, 10],
              [10, 12, 9]])


scaler = MinMaxScaler(feature_range=(0, 1))
x_scaled = scaler.fit_transform(x)
print(x_scaled)
''' A   B  C
[[ 0.4         0.5         1.        ]
 [ 1.          0.25        1.        ]
 [ 0.          0.          0.66666667]
 [ 0.8         1.          0.33333333]
 [ 0.4         0.75        0.        ]]
'''

prediction_scaled = np.array([1., 1., 0.66666667, 0.])


# fit C scaler
x2 = np.array([[12], [12], [11], [10], [9]])
scaler2 = MinMaxScaler(feature_range=(0, 1))
scaler2.fit(x2)
raw_data = scaler2.inverse_transform([[1.], [1.], [0.66666667], [0.]])
print(raw_data)
'''
[[ 12.        ]
 [ 12.        ]
 [ 11.00000001]
 [  9.        ]]
'''


