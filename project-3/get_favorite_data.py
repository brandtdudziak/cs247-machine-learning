import numpy as np
import sklearn
import matplotlib.pyplot as plt
import math

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from helpers import plot_decision_boundary
from helpers import get_bounds

def get_favorite_data():
    # Split into n sections that alternate sign left to right. Points are sampled from Gaussian dist. within each section
    d = 2
    n = 121
    scale = 3
    mu = []

    for row in range(11):
        for col in range(11):
            mu.append(np.array([col*scale, row*scale]))

    section = np.random.randint(0, n)

    x = np.random.multivariate_normal(mean = mu[section], cov = np.eye(d))
    y = section % 2

    return x, y

def example_get_favorite_data():
    # Two, far apart, spherical Gaussian blobs
    d = 5

    mu0 = np.array([-5 for i in range(d)])
    mu1 = np.array([ 5 for i in range(d)])

    y = np.random.binomial(1, 0.5) #flip a coin for y

    if y == 0:
        x = np.random.multivariate_normal(mean = mu0, cov = np.eye(d))
    else:
        x = np.random.multivariate_normal(mean = mu1, cov = np.eye(d))

    return x, y

def get_lots_of_favorite_data(n = 100, data_fun = get_favorite_data):
    pts = [data_fun() for _ in range(n)]
    Xs, ys = zip(*pts)
    X = np.array(Xs)
    y = np.array(ys)
    return (X, y)

if __name__ == "__main__":
    print("Here are some points from example_get_favorite_data:")
    for i in range(4):
        x, y = example_get_favorite_data()
        print(f"\tx: {x}")
        print(f"\ty: {y}")

    print("And here we use get_lots_of_favorite_data to obtain X and y:")
    X, y = get_lots_of_favorite_data(10, example_get_favorite_data)

    print("X:")
    print(X)
    print("y:")
    print(y)

    # Testing using my dist.
    print("Here are some points from get_favorite_data:")
    for i in range(4):
        x, y = get_favorite_data()
        print(f"\tx: {x}")
        print(f"\ty: {y}")

    print("And here we use get_lots_of_favorite_data to obtain X and y:")
    X, y = get_lots_of_favorite_data(10)

    print("X:")
    print(X)
    print("y:")
    print(y)


    # Getting actual data
    X, y = get_lots_of_favorite_data(1000)
    X_tr = np.copy(X)
    y_tr = np.copy(y)
    X_te = []
    y_te = []
    for p in range(250):
        index = np.random.randint(0,len(X_tr))
        X_te.append(X[index].tolist())
        y_te.append(y[index].tolist())

        X_tr = np.delete(X_tr, index, axis=0)
        y_tr = np.delete(y_tr, index, axis=0)

    X_te = np.array(X_te)
    y_te = np.array(y_te)


    # Laura is kNN with k = 1
    laura = KNeighborsClassifier(n_neighbors = 1)

    # Create plots
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Laura: kNN with k = 1")

    # Decision boundary
    laura.fit(X_tr, y_tr)
    f1_min, f1_max, f2_min, f2_max = get_bounds(X_tr)
    plot_decision_boundary(ax, laura, x_min = f1_min, x_max = f1_max, y_min = f2_min, y_max = f2_max)

    # Scatter plot of training set
    ax.scatter(X_tr[:,0], X_tr[:,1], c=['red' if y == 0 else 'blue' for y in y_tr])

    # Scatter plot of test set
    ax.scatter(X_te[:,0], X_te[:,1], marker="x", c=['red' if y == 0 else 'blue' for y in y_te])

    # Report loss on training and test sets
    y_tr_pred = laura.predict(X_tr)
    tr_loss = zero_one_loss(y_tr, y_tr_pred)
    y_te_pred = laura.predict(X_te)
    te_loss = zero_one_loss(y_te, y_te_pred)
    loss_text = "Tr Loss: " + str(tr_loss) + "\nTe Loss: " + str(te_loss)
    plt.figtext(0.5, 0.01, loss_text, wrap=True, horizontalalignment='center', fontsize=12)

    # Produce plots
    fig.set_figheight(7)
    fig.set_figwidth(7)
    # plt.tight_layout()
    plt.savefig('laura.pdf')


    # Other learners
    learners = [
    KNeighborsClassifier(n_neighbors = 2), KNeighborsClassifier(n_neighbors = 3),
    KNeighborsClassifier(n_neighbors = 4), KNeighborsClassifier(n_neighbors = 5),
    DecisionTreeClassifier(criterion = 'entropy', max_depth = 1), DecisionTreeClassifier(criterion = 'entropy', max_depth = 2),
    DecisionTreeClassifier(criterion = 'entropy', max_depth = 3), DecisionTreeClassifier(criterion = 'entropy', max_depth = 4),
    DecisionTreeClassifier(criterion = 'entropy', max_depth = None),
    SVC(kernel='linear'), SVC(kernel = 'rbf'),
    SVC(kernel='poly')
    ]
    i=0
    for learner in learners:

        # Create plots
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("learner: " + str(learner))

        # Decision boundary
        clf = learner
        clf.fit(X_tr, y_tr)
        f1_min, f1_max, f2_min, f2_max = get_bounds(X_tr)
        plot_decision_boundary(ax, clf, x_min = f1_min, x_max = f1_max, y_min = f2_min, y_max = f2_max)

        # Scatter plot of training set
        ax.scatter(X_tr[:,0], X_tr[:,1], c=['red' if y == 0 else 'blue' for y in y_tr])

        # Scatter plot of test set
        ax.scatter(X_te[:,0], X_te[:,1], marker="x", c=['red' if y == 0 else 'blue' for y in y_te])

        # Report loss on training and test sets
        y_tr_pred = clf.predict(X_tr)
        tr_loss = zero_one_loss(y_tr, y_tr_pred)
        y_te_pred = clf.predict(X_te)
        te_loss = zero_one_loss(y_te, y_te_pred)
        loss_text = "Tr Loss: " + str(tr_loss) + "\nTe Loss: " + str(te_loss)
        plt.figtext(0.5, 0.01, loss_text, wrap=True, horizontalalignment='center', fontsize=12)

        # Produce plots
        fig.set_figheight(7)
        fig.set_figwidth(7)
        # plt.tight_layout()
        plt.savefig('p1q4_{0}.pdf'.format(i))
        i += 1
