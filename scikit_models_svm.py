"""scikit basics with  Sebastian Raschka - code modified by andzu schaefer"""
#todo: create standalone files, depreciation prep
'''here we will classify some flowers with help from the sklearn.linear_model's Perceptron'''

'''initialization and data prep'''
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# if we executed np.unique(y)...
np.unique(y)  # should see that the iris flow class names are already stored as integers.

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  # initialize new standardscaler object
sc.fit(X_train)  # estimate mean, std of X_train
X_train_std = sc.transform(X_train)  # standardize training data
X_test_std = sc.transform(X_test)  # standardize test data

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test)) # Take a sequence of arrays and stack them horizontally to make a single array





'''plot some decision regions'''
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max,
                                     resolution))  # arange creates arrays with numbers from first param to second param at intervals of third param
    # meshgrid functions as explained here: http://louistiao.me/posts/numpy-mgrid-vs-meshgrid/

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)  # np.ravel flattens arrays,
    # .T : Same as self.transpose(), except that self is returned if self.ndim < 2.
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.04, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    #plt.ylim(xx2.min(), xx2.max()) #throwing an error, script runs (relatively) fine without... something to come back to

    # plot all samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidths=1, marker='o', s=55, label='test set')



''''Support Vector Machine '''
from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std,
                      y_combined,
                      classifier=svm,
                      test_idx=range(105,150))
'''
plt.xlabel('petal length[standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
'''
y_pred = svm.predict(X_test_std)
#print('accuracy : %.2f' % accuracy_score(y_test, y_pred))  #honestly dont remember how this ended up here, clean up later

''''''
'''solving nonlinear problems using a kernel SVM'''
np.random.seed(0)
X_xor = np.random.randn(200,2) #returns a 200 x 2 array with samples from the std normal distribution
y_xor = np.logical_xor(X_xor[:,0] > 0, X_xor[:,1]>0)
y_xor = np.where(y_xor,1,-1) #(condition[, x, y])¶ Return elements, either from x or y,
#  depending on condition.

#plot the xor data
plt.scatter(X_xor[y_xor==1,0], X_xor[y_xor==1,1],
            c='b', marker='x', label='1')

plt.scatter(X_xor[y_xor==-1,0], X_xor[y_xor==-1,1],
            c='r', marker='s', label='-1')
plt.ylim=(-3,0)
plt.legend()
plt.show()

#train the SVM
svm = SVC(kernel='rbf', C=10.0, gamma = 0.10, random_state=0) #notice kernel = rbf
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor,
                      y_xor,
                      classifier=svm)
plt.legend(loc='upper left')
plt.show()

#gamma is the parameter that controls the 'cut-off' of the gaussian sphere
svm = SVC(kernel='rbf', C=1.0, gamma = 0.20, random_state=0)
svm.fit(X_train_std,y_train)
plot_decision_regions(X_combined_std,
                      y_combined,
                      classifier=svm,
                      test_idx=range(105,150))

plt.xlabel('petal length[standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

#gamma is the parameter that controls the 'cut-off' of the gaussian sphere
svm = SVC(kernel='rbf', C=1.0, gamma = 100, random_state=0)
svm.fit(X_train_std,y_train)
plot_decision_regions(X_combined_std,
                      y_combined,
                      classifier=svm,
                      test_idx=range(105,150))

plt.xlabel('petal length[standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()