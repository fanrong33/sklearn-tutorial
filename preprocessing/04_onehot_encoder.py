# encoding: utf-8
"""

http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder


onehot_encoder = OneHotEncoder()


# sklearn 必须先转化为数值才能处理
onehot_encoder.fit([['female'], ['male'], ['femal']])
exit()

sex = np.array([1, 0, 1])
sex = sex.reshape(-1, 1)
onehot_encoder.fit(sex)
print(onehot_encoder)
''' OneHotEncoder(categorical_features='all', dtype=<... 'numpy.float64'>,
       handle_unknown='error', n_values='auto', sparse=True) '''

print(onehot_encoder.feature_indices_)
''' [0 2] '''
d = onehot_encoder.transform([[1], [0], [1]])
print(d.toarray())
'''
[[ 0.  1.]
 [ 1.  0.]
 [ 0.  1.]]
'''
# print(enc.n_values_)
# # array([2, 3, 4])
# print(enc.feature_indices_)
# # array([0, 2, 5, 9])
# arr = enc.transform([[0, 1, 0]]).toarray()
# print(arr)
# array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.]])


