"""scikit basics with  Sebastian Raschka - code modified by andzu schaefer"""
''''''

'''initialization and data prep'''
from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target

#if we executed np.unique(y)...
np.unique(y) #should see that the iris flow class names are already stored as integers.

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler() #initialize new standardscaler object
sc.fit(X_train) #estimate mean, std of X_train
X_train_std = sc.transform(X_train) #standardize training data
X_test_std = sc.transform(X_test) #standardize test data


from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    #setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1,
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1,
    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max, resolution),
                           np.arange(x2_min,x2_max, resolution)) #arange creates arrays with numbers from first param to second param at intervals of third param
    #meshgrid functions as explained here: http://louistiao.me/posts/numpy-mgrid-vs-meshgrid/
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T) #np.ravel flattens arrays,
    #.T : Same as self.transpose(), except that self is returned if self.ndim < 2.
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2, Z, alpha=0.04,cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    #plot all samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0], y=X[y==cl,1], alpha=0.8, c=cmap(idx), marker=markers[idx],label=cl)

    #highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx,:], y[test_idx]
        plt.scatter(X_test[:,0], X_test[:,1], c='', alpha=1.0, linewidths=1,marker='o',s=55, label='test set')


#scikit learn implements a highly optimized version of logistic regression that also supports multiclass settings off the shelf

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std,
                      y = y_combined,
                      classifier = lr,
                      test_idx = range(105,150))
#https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.scatter documentation
plt.xlabel('petal length[standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
