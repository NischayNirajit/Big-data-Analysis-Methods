# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:, 1:2].values
y=dataset.iloc[:, 2].values


'''#splitting dataset into trainning set andtest set
from sklearn.cross_validation import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)'''

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000 , random_state = 0)
regressor.fit(x , y)

regressor.predict(6.5)

x_grid = np.arange(min(x) , max(x), 0.0001)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x, y, color = 'red')
plt.plot(x_grid , regressor.predict(x_grid), color='blue')
plt.xlabel('position')
plt.ylabel('salaries')
plt.title('truth or bluff using random forest model')
