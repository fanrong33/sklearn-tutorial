# encoding: utf-8

import numpy as np

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0, 1))

training_set = np.array([10, 11, 12, 13, 10])
training_set = training_set.reshape(-1, 1)

training_set_scaled = scaler.fit_transform(training_set)
print(training_set_scaled)
'''
[[ 0.        ]     10
 [ 0.33333333]     11
 [ 0.66666667]     12
 [ 1.        ]     13
 [ 0.        ]]    10
'''
test_set = np.array([15, 16, 10])  # 超过训练值的最高值13
test_set = test_set.reshape(-1, 1)
test_set_scaled = scaler.fit_transform(test_set)
print(test_set_scaled)
'''
[[ 0.83333333]     15
 [ 1.        ]     16  （因为训练数据最高为13，当测试数据高于则不合理）
 [ 0.        ]]    10
'''

