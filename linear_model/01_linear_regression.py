# encoding: utf-8
""" 线性回归  预测 糖尿病

 * [Diabetes数据集线性回归：最小平方回归](https://blog.csdn.net/kt513226724/article/details/79801317)
"""
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1、加载diabetes（糖尿病）数据集
diabetes = datasets.load_diabetes()


# 数据前十列数值为生理数据，分别表示
# 年龄，性别，体质指数，血压，S1,S2,S3,S4,S5,S6(六种血清的化验数据)
# 数据已均值中心化处理
print(diabetes.data[0])
'''
[ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
 -0.04340085 -0.00259226  0.01990842 -0.01764613]
'''

# 表明疾病进展的数据，用target属性获得
diabetes.target
# 介乎于25到346之间的整数



# 2、建立Use only one feature
# diabetes_X = diabetes.data[:, np.newaxis]
diabetes_X = diabetes.data

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]
print(diabetes_X_train.shape)
print(diabetes_y_train.shape)


# Create linear regression object
model = linear_model.LinearRegression()

# Train the model using the training sets
model.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = model.predict(diabetes_X_test)
print(diabetes_y_test)
print(diabetes_y_pred)



# The coefficients 调用预测模型的coef_属性，就可以得到每种生理数据的回归系数b
print('Coefficients: \n', model.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# 可以用方差来评价预测结果好坏，方差越接近1，说明预测越准确
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.plot(diabetes_y_test,  color='blue', label='test')
plt.plot(diabetes_y_pred, color='red', label='pred')

# plt.xticks(())
# plt.yticks(())

plt.legend(loc='best')
plt.show()


