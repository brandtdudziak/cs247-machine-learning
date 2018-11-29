import numpy as np
import matplotlib.pyplot as plt
def plot_decision_boundary(ax, clf, x_min = 0, x_max = 1, y_min = 0, y_max = 1, res=0.01, cm = plt.cm.RdBu):
    xx, yy = np.meshgrid(np.arange(x_min, x_max, res),
                         np.arange(y_min, y_max, res))
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=.5, cmap=cm)

def plot_num(ax, x):
    ax.imshow(x.reshape(8, 8), cmap = plt.cm.bone_r)

def get_bounds(X):
    f1_min, f1_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    f2_min, f2_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    return f1_min, f1_max, f2_min, f2_max
