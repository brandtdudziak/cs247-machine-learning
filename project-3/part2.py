import numpy as np
import dill as pickle
import sklearn
import matplotlib.pyplot as plt
from helpers import plot_num

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Uncomment the following 3 lines if you're getting annoyed with warnings from sklearn
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


if __name__ == "__main__":
    assert 1/2 == 0.5, "Are you sure you're using python 3?"
    #print(f"Version of sklearn: {sklearn.__version__}")
    #print("(It should be 0.20.0)")


    fin = open("digits.pkl", "rb")
    train, test = pickle.load(fin)
    X_tr, y_tr = train
    X_te, y_te = test

    # Q1
    # A: Count number of points in each class
    digits = [0] * 10
    for p in y_tr:
        digits[p] += 1

    # shows how many points of each class (digit) there are
    print("Number of points in each class: " + str(digits))

    # B: Pixel-wise average of each class
    avg = np.zeros((10, 64))
    n,d = X_tr.shape
    for point in range(n):
        for pixel in range(d):
            avg[y_tr[point]][pixel] += X_tr[point][pixel]

    for point in range(len(digits)):
        for pixel in range(d):
            avg[point][pixel] /= digits[point]


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("pixel-wise average")
    for d in range(len(digits)):
        plot_num(ax, avg[d, :])
        plt.savefig('p2q1b_{0}.pdf'.format(d))


    # C: Any features useless? Any pixel with average 0 across all classes is useless
    useless = [True] * 64
    for point in range(len(avg)):
        for pixel in range(len(avg[point])):
            if avg[point][pixel] > 0:
                useless[pixel] = False

    for pixel in range(len(useless)):
        if useless[pixel] == True:
            print("Useless pixel: " + str(pixel))


    # Q2
    # Train logistic regression classifier on training set
    clf = LogisticRegression()
    clf.fit(X_tr, y_tr)

    # Predict test set
    y_pred = clf.predict(X_te)

    # Report precision and recall
    print("Precision: " + str(precision_score(y_te, y_pred, average=None)))
    print("Recall: " + str(recall_score(y_te, y_pred, average=None)))
    print(confusion_matrix(y_te, y_pred))

    # Plot misclassified digits
    fig = plt.figure()
    ax = fig.add_subplot(111)
    incorrect = [-1] * 10
    for point in range(len(X_te)):
        if incorrect[y_te[point]] == -1 and not y_pred[point] == y_te[point]:
            incorrect[y_te[point]] = point
    for d in range(len(digits)):
        ax.set_title("incorrect prediction: " + str(y_pred[incorrect[d]]) + ", actually " + str(y_te[incorrect[d]]))
        plot_num(ax, X_te[incorrect[d], :])
        plt.savefig('p2q2_{0}.pdf'.format(d))


    # Q3
    fin = open("cancer.pkl", "rb")
    train, test = pickle.load(fin)
    X_tr, y_tr = train
    X_te, y_te = test

    # Standardize data
    scaler = MinMaxScaler()
    X_tr_transformed = scaler.fit_transform(X_tr, y_tr)

    # Train, predict, and state confusion matrix
    clf = KNeighborsClassifier(n_neighbors = 3)
    clf.fit(X_tr_transformed, y_tr)

    X_te_transformed = scaler.fit_transform(X_te, y_te)
    y_pred = clf.predict(X_te_transformed)
    print(confusion_matrix(y_te, y_pred))


    #Q4
    # Setup grid search
    param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [1.0, 0.1, 0.01, 0.001]},
        {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3, 4, 5]},
        {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.1, 1, 10, 100]},
    ]

    clf = SVC()
    clf.fit(X_tr_transformed, y_tr)
    cv = GridSearchCV(clf, param_grid, cv=5, scoring='f1')
    cv.fit(X_tr_transformed, y_tr)
    # Print best learner
    print(cv.best_estimator_.get_params())
    # Predict using this learner
    y_pred = cv.predict(X_te_transformed)
    print(confusion_matrix(y_te, y_pred))
    print("Precision: " + str(precision_score(y_te, y_pred)))
    print("Recall: " + str(recall_score(y_te, y_pred)))
    print("F1: " + str(f1_score(y_te, y_pred)))
