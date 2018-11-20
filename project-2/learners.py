import numpy as np
from abc import ABC, abstractmethod

class Regressor(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class NotYetTrainedException(Exception):
    """Learner must be trained (fit) before it can predict points."""
    pass


def simple_kernel(x1, x2):
    return (np.dot(x1, x2) + 1)**2


class ToyRegressor(Regressor):
    def __init__(self):
        self.mean = None

    def fit(self, X, y):
        self.mean = np.average(y)


    def predict(self, X):
        if self.mean is not None:
            return np.array([self.mean for _ in X])
        else:
            raise NotYetTrainedException
        pass


class OLS(Regressor):
    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        self.theta = np.linalg.solve(np.dot(np.transpose(X), X), np.dot(np.transpose(X), y))

    def predict(self, X):
        if self.theta is not None:
            return np.dot(X, self.theta)
        else:
            raise NotYetTrainedException
        pass


class RidgeRegression(Regressor):
    def __init__(self, lamb):
        self.theta = None
        self.lamb = lamb

    def fit(self, X, y):
        n,d = X.shape
        self.theta = np.linalg.solve(((np.dot(np.transpose(X), X) / n) + (self.lamb * np.identity(d))), np.dot(np.transpose(X), y) / n)

    def predict(self, X):
        if self.theta is not None:
            return np.dot(X, self.theta)
        else:
            raise NotYetTrainedException
        pass


class GeneralizedRidgeRegression(Regressor):
    def __init__(self, reg_weights):
        self.theta = None
        self.reg_weights = reg_weights

    def fit(self, X, y):
        n,d = X.shape
        self.theta = np.linalg.solve(((np.dot(np.transpose(X), X) / n) + np.diag(self.reg_weights)), np.dot(np.transpose(X), y) / n)

    def predict(self, X):
        if self.theta is not None:
            return np.dot(X, self.theta)
        else:
            raise NotYetTrainedException
        pass


class DualRidgeRegression(Regressor):
    def __init__(self, lamb, kernel):
        self.a = None
        self.lamb = lamb
        self.kernel = kernel

    def fit(self, X, y):
        n,d = X.shape
        K = np.zeros((n, d))
        for row in range(n):
            for col in range(d):
                K[row][col] = self.kernel(X[col], X[row])
        self.a = np.linalg.solve(K + (self.lamb * np.identity(d)), y)

    def predict(self, X):
        if self.a is not None:
            n,d = X.shape
            out = np.array([])
            for point in range(n):
                k = np.array([])
                for el in range(n):
                    k = np.append(k, self.kernel(X[el], X[point]))
                func = np.dot(np.transpose(self.a), k)
                out = np.append(out, func)
            return out
        else:
            raise NotYetTrainedException
        pass


if __name__ == "__main__":
    reg = RidgeRegression(np.array(2.))
    reg.fit(np.array([[10., 11.], [1., 15.], [5., 0.], [3., 4.]]), np.array([5., 4., 3., 10.]))
    print(reg.predict(np.array([[6., 2.], [4., 3.], [.5, 6.5], [0., 1.]])))


    reg = GeneralizedRidgeRegression(np.array([2., 2.]))
    reg.fit(np.array([[10., 11.], [1., 15.], [5., 0.], [3., 4.]]), np.array([5., 4., 3., 10.]))
    print(reg.predict(np.array([[6., 2.], [4., 3.], [.5, 6.5], [0., 1.]])))
