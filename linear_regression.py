import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import sys

class LinearRegression:
    def __init__(self, lr=0.001, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.weights=np.zeros(n_features)
        self.bias=0

        for _ in range(self.n_iter):
            y_pred = np.dot(x, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(x.T, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weights -= self.lr*dw
            self.bias -= self.lr*db




    def predict(self, x):
        y_pred = np.dot(x, self.weights) + self.bias
        return y_pred

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)

def r2_score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2

if __name__=='__main__':
    x, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

    model = LinearRegression(lr=0.01, n_iter=1000)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    mse = mean_squared_error(y_test, predictions)
    print("MSE:", mse)

    accu = r2_score(y_test, predictions)
    print("Accuracy:", accu)

    y_pred_line = model.predict(x)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(x_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(x_test, y_test, color=cmap(0.5), s=10)
    plt.plot(x, y_pred_line, color="black", linewidth=2, label="Prediction")
    plt.show()

# fig = plt.figure(figsize=(8,6))
# plt.scatter(x[:,0], y, color = 'b', marker='o' , s=30)
# plt.show()
