import numpy as np

def prepend_1s(X):
    Y = np.array([])
    n,d = X.shape
    for m in X:
        m = np.insert(m, 0, 1., axis=0)
        Y = np.append(Y, m)
    Y = Y.reshape(n, d + 1)
    return Y


def poly_lift(X, degree):
    Y = np.array([])
    n, = X.shape
    for point in X:
        m = np.array([])
        for d in range(degree):
            m = np.append(m, point**d)
        Y = np.append(Y, m)
    Y = Y.reshape(n, degree)
    return Y


def standardize(X):
    Y = X
    n,d = X.shape
    max = np.max(X, axis=0)
    min = np.min(X, axis=0)
    for row in range(n):
        for col in range(d):
            Y[row][col] = (Y[row][col] - min[col]) / (max[col] - min[col])
    return Y
