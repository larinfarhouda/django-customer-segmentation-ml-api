# import nltk
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import GridSearchCV, learning_curve
# from sklearn import neighbors, linear_model, svm, tree, ensemble
# from sklearn import preprocessing, model_selection, metrics, feature_selection
# import plotly.graph_objs as go
# from plotly.graph_objs import *

# from core.code import customerscat
import plotly.graph_objs as go
from plotly.graph_objs import *

# # ____________________
# # define a class that allows to interface several of the functionalities common to these different classifiers in order to simplify their use:
# # ____________________

# class Class_Fit(object):
#     def __init__(self, clf, params=None):
#         if params:
#             self.clf = clf(**params)
#         else:
#             self.clf = clf()

#     def train(self, x_train, y_train):
#         self.clf.fit(x_train, y_train)

#     def predict(self, x):
#         return self.clf.predict(x)

#     def grid_search(self, parameters, Kfold):
#         self.grid = GridSearchCV(
#             estimator=self.clf, param_grid=parameters, cv=Kfold)

#     def grid_fit(self, X, Y):
#         self.grid.fit(X, Y)

#     def grid_predict(self, X, Y):
#         self.predictions = self.grid.predict(X)
#         return "Precision: {:.2f} % ".format(
#             100*metrics.accuracy_score(Y, self.predictions))


# # ____________________
# # defining x y and split the dataset in train and test sets
# # ____________________
# columns = ['mean', 'categ_0', 'categ_1', 'categ_2', 'categ_3', 'categ_4']
# X = customerscat.selected_customers[columns]
# Y = customerscat.selected_customers['cluster']

# X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
#     X, Y, train_size=0.8)


# # =========================================================================================
# # crating classifiers
# # =========================================================================================

# # # ____________________
# # # random forest
# # # ____________________

# # rf = Class_Fit(clf=ensemble.RandomForestClassifier)
# # param_grid = {'criterion': ['entropy', 'gini'], 'n_estimators': [20, 40, 60, 80, 100],
# #               'max_features': ['sqrt', 'log2']}
# # rf.grid_search(parameters=param_grid, Kfold=5)
# # rf.grid_fit(X=X_train, Y=Y_train)
# # rf.grid_predict(X_test, Y_test)
# # print('rf')
# # print(rf.grid_predict(X_test, Y_test))
# # # 90.30 %

# # # ____________________
# # # Gradient Boosting
# # # ____________________

# # gb = Class_Fit(clf=ensemble.GradientBoostingClassifier)
# # param_grid = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
# # gb.grid_search(parameters=param_grid, Kfold=5)
# # gb.grid_fit(X=X_train, Y=Y_train)
# # gb.grid_predict(X_test, Y_test)
# # print('gb')
# # print(gb.grid_predict(X_test, Y_test))


# # # ____________________
# # # Logistic Regression
# # # ____________________

# # lr = Class_Fit(clf=linear_model.LogisticRegression)
# # lr.grid_search(parameters=[{'C': np.logspace(-2, 2, 20)}], Kfold=5)
# # lr.grid_fit(X=X_train, Y=Y_train)
# # lr.grid_predict(X_test, Y_test)
# # print('lr')
# # print(lr.grid_predict(X_test, Y_test))

# # # ____________________
# # #  Decision Tree
# # # ____________________
# tr = Class_Fit(clf=tree.DecisionTreeClassifier)
# tr.grid_search(parameters=[
#                {'criterion': ['entropy', 'gini'], 'max_features':['sqrt', 'log2']}], Kfold=5)
# tr.grid_fit(X=X_train, Y=Y_train)
# tr.grid_predict(X_test, Y_test)
# print('tr')
# print(tr.grid_predict(X_test, Y_test))
# # 88.37%    82.41%


classifiers = {'classifier': ['Logistic Regression', 'Random Forest',
                              'Gradient Boosting'], 'Precision': [89.06, 90.03, 89.06]}

bardata = dict(type="bar", x=classifiers['classifier'],
                    y=classifiers['Precision'])

figbarr = go.Figure(data=[bardata])
figbarr.update_layout(
    title_text="Classifiers Precision percentage",  xaxis_title="classifier",
    yaxis_title="Precision percentage")
# figbarr.show(renderer='colab')
figbarr = figbarr.to_json()
