# encoding: utf-8
""" 逻辑回归 分类任务

cd /Users/fanrong33/kuaipan/github/sklearn-tutorial/linear_model/LogisticRegression/
python 01.py
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


iris = datasets.load_iris()
X, y = iris.data, iris.target
print(X)
'''
[[ 6.2  3.4  5.4  2.3]
 [ 5.9  3.   5.1  1.8]]
'''
print(y)
'''
[0 1]
'''


# 预处理
from sklearn.preprocessing import MinMaxScaler 
#区间缩放，返回值为缩放到[0, 1]区间的数据 
X = MinMaxScaler().fit_transform(X)

train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.33, random_state=999)
# print(train_data)
# print(test_data)

model = LogisticRegression()
model.fit(train_data, train_labels)
print(model)

pred = model.predict(test_data)
print(pred)
prob = model.predict_proba(test_data)
print(prob)

accuracy = model.score(test_data, test_labels)
print('Accuracy: %s'%accuracy)


