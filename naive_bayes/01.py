# encoding: utf-8
""" 朴素贝叶斯分类 iris
包含3个分类
"""

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss


# 1、加载数据集
# 假定 sepal length, sepal width, petal length, petal width 4个量独立且服从高斯分布，用贝叶斯分类器建模
# 高斯分布（又名正态分布）
iris = datasets.load_iris()
print(iris.data)
'''
array([[ 5.1,  3.5,  1.4,  0.2],
       [ 4.9,  3. ,  1.4,  0.2],
       [ 4.7,  3.2,  1.3,  0.2],
       [ 4.6,  3.1,  1.5,  0.2],
       [ 5. ,  3.6,  1.4,  0.2]])
'''
print(iris.data.shape)
''' (150, 4) # 150行4列 '''
print(iris.target[:5])
''' [0 0 0 0 0] '''


# 2、训练模型
model = GaussianNB()
model.fit(iris.data, iris.target)

# 3、预测测试数据
y_pred = model.predict(iris.data)


# 4、评价模型的准确率
accuracy = accuracy_score(iris.target, y_pred)  # labels与实际预测的值
print(accuracy)
''' 0.96 '''


# 5、log损失
# 在这个多分类问题中，Kaggle的评定标准并不是precision，而是multi-class log_loss，这个值越小，表示最后的效果越好
predicted = model.predict_proba(iris.data)  # 置信度
loss = log_loss(iris.target, predicted)
print(loss)
''' 0.111248820748 '''


# 6、使用模型进行预测
pred = model.predict([[5., 3.2, 1.3, 0.3]])
print(pred)
''' [0] '''


