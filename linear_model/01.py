# encoding: utf-8

from sklearn import datasets
# from sklearn import svm

iris = datasets.load_iris()
X, y = iris.data, iris.target
print(X)
print(y)

# model = svm.SVC()
# model.fit(X, y)

# print(model)

from sklearn.externals import joblib
# joblib.dump(model, 'iris_svm.pkl')


model = joblib.load('iris_svm.pkl')

test_data = X[40:62]
test_labels = y[40:62]
pred = model.predict(test_data)
accuracy = model.score(test_data, test_labels)

print(pred)
print(accuracy)

