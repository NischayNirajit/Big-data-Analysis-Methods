# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:, 1:2].values
y=dataset.iloc[:, 2].values



#splitting dataset into trainning set andtest set
'''from sklearn.cross_validation import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=0)'''

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = np.array(y)
y = y.reshape(len(y) , 1)
y = sc_y.fit_transform(y)

#importing support vector regression library
from sklearn.svm import SVR
regressor =SVR(kernel = 'rbf')
regressor.fit(x , y)


predict_y = regressor.predict(sc_x.transform(np.reshape(len(np.array([6.5])),1)))
predict_y = sc_y.inverse_transform(predict_y)


#visualizing the polynomial model
x_grid = np.arange(min(x) , max(x) , 0.01)
x_grid = x_grid.reshape(len(x_grid) , 1)
plt.scatter(x, y, color='black')
plt.plot(x_grid, regressor.predict(x_grid), color='red')
plt.xlabel('Position')
plt.ylabel('salary')
plt.title('truth or bluff using SVR')
plt.show()


