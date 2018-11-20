import numpy as np

# A decorator -- don't worry if you don't understand this.
# It just makes it so that each loss function you implement automatically checks that arguments have the same number of elements
def loss_fun(fun):
    def toR(y_true, y_preds):
        n,  = y_true.shape
        npreds, = y_preds.shape
        assert n == npreds, "There must be as many predictions as there are true values"
        return fun(y_true, y_preds)
    return toR

@loss_fun
def zero_one(y_true, y_preds):
    n, = y_true.shape
    return np.sum([1 for yt, yp in zip(y_true, y_preds) if yt == yp ])/n

@loss_fun
def MSE(y_true, y_preds):
    n, = y_true.shape
    return np.sum((y_true - y_preds)**2) / n

@loss_fun
def MAD(y_true, y_preds):
    n, = y_true.shape
    return np.sum(np.absolute(y_true - y_preds)) / n

def cross_validation(X, y, reg, evaler, num_folds = 10):
    sum_scores = 0
    for fold in range(num_folds):
        # train on k-1 folds - fit
        n,d = X.shape
        points = np.arange(n)

        X1 = np.take(X, [point for point in points if point%num_folds != fold], axis=0)
        X2 = np.take(X, [point for point in points if point%num_folds == fold], axis=0)
        y1 = np.take(y, [point for point in points if point%num_folds != fold])
        y2 = np.take(y, [point for point in points if point%num_folds == fold])



        reg.fit(X1, y1)
        sum_scores = sum_scores + evaler(y2, reg.predict(X2))
        # test on last - predict
        # get score - evaluate
    return sum_scores / num_folds
