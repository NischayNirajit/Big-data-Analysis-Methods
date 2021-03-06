# -*- coding: utf-8 -*-
#logistic regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')
x=dataset.iloc[:, 2:4].values
y=dataset.iloc[:, -1].values



#splitting dataset into trainning set andtest set
from sklearn.cross_validation import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.25 , random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(random_state=0)
LR.fit(x_train,y_train)

#predicting test set results
predict_y = LR.predict(x_test)
#making confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predict_y) #produces the matrix which stores number of success and unsuccess

#visualizing the dataset

from matplotlib.colors import ListedColormap
x_set , y_set = x_train , y_train
x1 , x2 = np.meshgrid(np.arange(start = x_set[:, 0].min()-1, stop = x_set[:, 0].max()+1, step = 0.05),
                      np.arange(start = x_set[:, 1].min()-1, stop = x_set[:, 1].max()+1, step = 0.05))
plt.contourf(x1, x2, LR.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j, 0], x_set[y_set == j, 1],c = ListedColormap(('red', 'green'))(i), label = j)
   
plt.title('Logistic regression')
plt.xlabel('age')
plt.ylabel('salary')
plt.show()

#using test set
from matplotlib.colors import ListedColormap
x_set , y_set = x_test , y_test
x1 , x2 = np.meshgrid(np.arange(start = x_set[:, 0].min()-1, stop = x_set[:, 0].max()+1, step = 0.05),
                      np.arange(start = x_set[:, 1].min()-1, stop = x_set[:, 1].max()+1, step = 0.05))
plt.contourf(x1, x2, LR.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j, 0], x_set[y_set == j, 1],c = ListedColormap(('red', 'green'))(i), label = j)
   
plt.title('Logistic regression')
plt.xlabel('age')
plt.ylabel('salary')
plt.show()
