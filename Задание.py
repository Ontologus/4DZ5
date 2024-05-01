import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

param_grid = {'criterion':['gini', 'entropy'],
              'max_depth':[2, 4, 6, 8],
              'min_samples_split':[3, 5, 7]}
tree_clf = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator=tree_clf, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
best = grid_search.best_params_
print(best)

b_criterion, b_max_depth, b_min_samples_split = best['criterion'], best['max_depth'], best['min_samples_split']
my_tree = DecisionTreeClassifier(criterion=b_criterion, max_depth=b_max_depth, min_samples_split=b_min_samples_split)
my_tree.fit(X_train, y_train)

pred_test = my_tree.predict(X_test)
pred_train = my_tree.predict(X_train)
a_test = accuracy_score(y_test, pred_test)
a_train = accuracy_score(y_train, pred_train)
print(a_test)
print(a_train)