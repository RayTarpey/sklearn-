import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta:float
        Learning rate (between 0.0 and 1.0)
    n_iter:int
        Passes over the training dataset.

    Attributes
    -------------
    w_: 1d-array
        Weights after fitting.
    errors_: list
        Numebr of misclassifications in every epoch.

    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ------------
        X: {array-like}, shape=[n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_featuers is the number of features.
        y: array-like, shape=[n_smaples]
            Target values.

        Returns
        ----------
        self: object
        """

        self.w_ = np.zeros(1 + X.shape[1]) # Add w_0
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
# 輸出最後20行的資料，並觀察資料結構 萼片長度（sepal length），萼片寬度()，
# 花瓣長度（petal length），花瓣寬度，種類
print(df.tail(n=20))
print(df.shape)

# 0到100行，第5列
y = df.iloc[0:100, 4].values
# 將target值轉數字化 Iris-setosa為-1，否則值為1
y = np.where(y == "Iris-setosa", -1, 1)
# 取出0到100行，第1，第三列的值
x = df.iloc[0:100, [0, 2]].values

# scatter繪製點圖
plt.scatter(x[0:50, 0], x[0:50, 1], color="red", marker="o", label="setosa")
plt.scatter(x[50:100, 0], x[50:100, 1], color="blue", marker="x", label="versicolor")
plt.title("鳶尾花散點圖")
plt.xlabel(u"花瓣長度")
plt.ylabel(u"萼片長度")
plt.legend(loc="upper left")
plt.show()

"""
    訓練模型並且記錄錯誤次數，觀察錯誤次數的變化
"""
# 資料真實值
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", -1, 1)
x = df.iloc[0:100, [0, 2]].values

"""
    誤差數折線圖 
    @:param eta: 0.1 學習速率
    @:param n_iter：0.1 迭代次數
"""
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(x, y)
# plot繪製折線圖
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker="o")
plt.xlabel("迭代次數（n_iter）")
plt.ylabel("錯誤分類次數（error_number）")
plt.show()

from matplotlib.colors import ListedColormap
def plot_decision_regions(x, y, classifier, resolution=0.2):
    """
    二維資料集決策邊界視覺化
    :parameter
    -----------------------------
    :param self: 將鳶尾花花萼長度、花瓣長度進行視覺化及分類
    :param x: list 被分類的樣本
    :param y: list 樣本對應的真實分類
    :param classifier: method  分類器：感知器
    :param resolution:
    :return:
    -----------------------------
    """
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    # y去重之後的種類
    listedColormap = ListedColormap(colors[:len(np.unique(y))])

    # 花萼長度最小值-1，最大值+1
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    # 花瓣長度最小值-1，最大值+1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    # 將最大值，最小值向量生成二維陣列xx1,xx2
    # np.arange(x1_min, x1_max, resolution)  最小值最大值中間，步長為resolution
    new_x1 = np.arange(x1_min, x1_max, resolution)
    new_x2 = np.arange(x2_min, x2_max, resolution)
    xx1, xx2 = np.meshgrid(new_x1, new_x2)

    # 預測值
    # z = classifier.predict([xx1, xx2])
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.4, camp=listedColormap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=x[y == c1, 0], y=x[y == c1, 1], alpha=0.8, c=listedColormap(idx), marker=markers[idx], label=c1)

plot_decision_regions(x, y, classifier=ppn)
plt.xlabel("花瓣長度 [cm]")
plt.ylabel("花萼長度 [cm]")
plt.legend(loc="upper left")
plt.show()