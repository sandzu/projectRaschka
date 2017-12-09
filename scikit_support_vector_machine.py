
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

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std,y_combined)