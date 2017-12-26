from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
import numpy as np


def my_euclidean(a, b):
    return distance.euclidean(a, b)


class MyKNN:

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.__closest(row)
            predictions.append(label)
        return predictions

    def __closest(self, row):
        all_labels = []
        for i in range(0, len(self.X_train)):
            dist = my_euclidean(row, self.X_train[i])
            # 获取k个最近距离的邻居，格式为(distance, index)的tuple集合
            all_labels = self.__append_neighbors(all_labels, (dist, i))

        # 将k个距离最近的邻居，映射为label的集合
        nearest_ones = np.array([self.y_train[idx] for val, idx in all_labels])
        # 使用numpy的unique方法，分组计算label的唯一值及其对应的值第一次出现的index和值的计数
        # 例： elements = [1, 2],  elements_index = [3,0], elements_count = [1, 4] 这个结合表示：
        #   elements = [1, 2] ： 出现了1和2两种类型的数据
        #   elements_index = [3,0] ： 1第一次出现的index是3， 2第一次出现的index是0
        #   elements_count = [1, 4] ： 1共出现了1次， 2共出现了4次
        elements, elements_index, elements_count = np.unique(nearest_ones, return_counts=True, return_index=True)
        # 返回最大可能性的那种类型的label值
        return elements[list(elements_count).index(max(elements_count))]

    def __append_neighbors(self, arr, item):
        if len(arr) <= self.n_neighbors:
            arr.append(item)
        return sorted(arr, key=lambda tup: tup[0])[:self.n_neighbors]


iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.6)

# N neighbors classifier
my_classification = MyKNN()
my_classification.fit(X_train, y_train)
predictions = my_classification.predict(X_test)
print(accuracy_score(y_test, predictions))
