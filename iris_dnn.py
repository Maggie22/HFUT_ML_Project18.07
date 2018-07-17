import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow.contrib.learn as tflearn

# 导入数据集
data_temp = []
fr = open('iris.data')
lines = fr.readlines()
for line in lines:
    items = line.strip().split(',')
    data_temp.append(items)
data_temp = np.array(data_temp)
# 物种名
species = [0, 1, 2]
name = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
# 生成对应标签集
labels = []
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
# 获得训练集的数据
x = np.array(list(map(float, x_train[:, 0])))
for i in range(1, 4):
    items = np.array(list(map(float, x_train[:, i])))
    x = np.c_[x, items]
x_train = np.c_[np.ones((x.shape[0])), x]
# 获得测试集的数据
x_ = np.array(list(map(float, x_test[:, 0])))
for i in range(1, 4):
    items = np.array(list(map(float, x_test[:, i])))
    x_ = np.c_[x_, items]
x_test = np.c_[np.ones((x_.shape[0])), x_]
# 标准化数据
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)
# 四个特征值
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
# 建立3层，分别有10,20,10个单元的神经网络
classifier = tflearn.DNNClassifier(feature_columns=feature_columns,
                                   hidden_units=[10, 20, 10],
                                   n_classes=3
                                   )
# 数据集的训练，训练步数为2000
classifier.fit(x=x_train_std,
               y=y_train,
               steps=2000)
# 输出精确度，预测结果和实际结果
y_pred = list(classifier.predict(x_test_std))
accuracy_score = classifier.evaluate(x=x_test_std, y=y_test)
print(accuracy_score)
print('Predictions:{}'.format(y_pred))
print('y_test:', y_test)
