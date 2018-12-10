# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 13:14:07 2018

@author: umang
"""

# Support vector machine
from urllib import request
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import json
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

# Importing the dataset
dataset = pd.read_csv('{}/dataset/social_network_ads.csv'.format(dir_path))
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# SVM fitting
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(x_train, y_train)

from sklearn.metrics import accuracy_score
# predicting the data

y_pred = classifier.predict(x_test)

#save the model
filename = 'dataset/svm.pkl'
_ = joblib.dump(classifier, filename, compress=9)


data = request.urlopen("http://5bff0f87362b930013f652d1.mockapi.io/api/svm")

json_ = json.loads(data.read())
query = pd.get_dummies(pd.DataFrame(json_))
query = query.reindex(columns=model_columsn, fill_value=0)
query = sc.fit_transform(query)

print(classifier.predict(query))


#
# # Confusion maetric
# from sklearn.metrics import confusion_matrix
#
# mat = confusion_matrix(y_test, y_pred)
#
# # Visualizing the Data
#
# from matplotlib.colors import ListedColormap
#
# x_set, y_set = x_train, y_train
# X1, X2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 2, step=0.01),
#                      np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 2, step=0.01))
# Z_train = np.array([X1.ravel(), X2.ravel()]).T
# plt.contourf(X1, X2, classifier.predict(Z_train).reshape(X1.shape), alpha=0.5,
#              cmap=ListedColormap(
#                  ('red', 'green')))
# # outliers
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_train)):
#     plt.scatter(x_train[y_train == j, 0], x_train[y_train == j,
#                                                   1],
#                 c=ListedColormap(('red', 'green'))(i), label=j)
#
# plt.title('SVM (Trainning set results)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()
#
# #  Test set results
# from matplotlib.colors import ListedColormap
#
# X_set, y_set = x_test, y_test
# X1, X2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
#                      np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha=0.75, cmap=ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c=ListedColormap(('red', 'green'))(i), label=j)
# plt.title('Classifier (Test set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()