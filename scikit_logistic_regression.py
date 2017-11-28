"""scikit basics with  Sebastian Raschka - code modified by andzu schaefer"""
'''here we will classify some flowers with help from the sklearn.linear_model's Perceptron'''

#scikit learn implements a highly optimized version of logiztic regression that also supports multiclass settings off the shelf

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)

plot_decision_regions(X=X_combined_std,
                      y = y_combined,
                      classifier = lr,
                      test_idx = range(105,150))
#https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.scatter documentation
plt.xlabel('petal length[standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
