import numpy as np
import dill as pickle
import sklearn
import matplotlib.pyplot as plt
from helpers import plot_decision_boundary
from helpers import get_bounds

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss

# Uncomment the following 3 lines if you're getting annoyed with warnings from sklearn
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


if __name__ == "__main__":
    assert 1/2 == 0.5, "Are you sure you're using python 3?"
    # print(f"Version of sklearn: {sklearn.__version__}")
    # print("(It should be 0.20.0)")


    # Q1
    # Iterate over data sets and n_neighbors
    data_sets = ['simple_task.pkl', 'moons.pkl']
    for data_set in data_sets:

        # Load data
        fin = open(data_set, "rb")
        train, test = pickle.load(fin)
        X_tr, y_tr = train
        X_te, y_te = test

        for k in range(1, 6):
    
            # Create plots
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title("n_neighbors: " + str(k))

            # Decision boundary
            clf = KNeighborsClassifier(n_neighbors = k)
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
            plt.savefig('p1q1_{0}_{1}.pdf'.format('simple' if data_set=='simple_task.pkl' else 'moons', k))


    # Q2
    # Iterate over data sets and n_neighbors
    for data_set in data_sets:

        # Load data
        fin = open(data_set, "rb")
        train, test = pickle.load(fin)
        X_tr, y_tr = train
        X_te, y_te = test

        for d in range(1, 6):

            # Create plots
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title("max_depth: " + str(d if d < 5 else None))

            # Decision boundary
            clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = d if d < 5 else None)
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
            plt.savefig('p1q2_{0}_{1}.pdf'.format('simple' if data_set=='simple_task.pkl' else 'moons', d))

    # Q3
    # Iterate over data sets and n_neighbors
    kernels = ['linear', 'rbf', 'poly']
    for data_set in data_sets:

        # Load data
        fin = open(data_set, "rb")
        train, test = pickle.load(fin)
        X_tr, y_tr = train
        X_te, y_te = test

        for k in kernels:

            # Create plots
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title("kernel: " + str(k))

            # Decision boundary
            clf = SVC(kernel=k)
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
            plt.savefig('p1q3_{0}_{1}.pdf'.format('simple' if data_set=='simple_task.pkl' else 'moons', k))
