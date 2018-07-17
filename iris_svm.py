import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 导入数据集
data_temp = []
fr = open('iris.data')
lines = fr.readlines()
for line in lines:
    items = line.strip().split(',')
    data_temp.append(items)
data_temp = np.array(data_temp)
# 物种名字
species = [0, 1, 2]
name = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
labels = []
# 生成对应标签集
for i in range(0, len(data_temp)):
    if data_temp[i][4] == name[0]:
        labels.append(0)
    elif data_temp[i][4] == name[1]:
        labels.append(1)
    else:
        labels.append(2)
labels = np.array(labels)
# 获得训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(data_temp, labels, test_size=0.2, random_state=1)
# 获得训练集的各项数据
x = np.array(list(map(float, x_train[:, 0])))
for i in range(1, 4):
    items = np.array(list(map(float, x_train[:, i])))
    x = np.c_[x, items]
x_train = x
# 获得测试集各项
x_ = np.array(list(map(float, x_test[:, 0])))
for i in range(1, 4):
    items = np.array(list(map(float, x_test[:, i])))
    x_ = np.c_[x_, items]
x_test = x_
# 标准化
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

C = 1 #正则化参数
svc = svm.SVC(kernel='linear', C=C).fit(x_train_std, y_train)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(x_train_std, y_train)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(x_train_std, y_train)
# lin_svc = svm.LinearSVC(C=C).fit(x_train_std, y_train)
svc.fit(x_train_std, y_train)
y_pred1 = svc.predict(x_test_std)
print('Predictions(linear):', y_pred1)
print(svc.score(x_test_std,y_test))
y_pred2 = rbf_svc.predict(x_test_std)
print('Predictions(rbf):', y_pred2)
print(rbf_svc.score(x_test_std,y_test))
y_pred3 = poly_svc.predict(x_test_std)
print('Predictions(poly):', y_pred3)
print(poly_svc.score(x_test_std,y_test))
print('y_test:', y_test)
# enumerate((svc, lin_svc))
