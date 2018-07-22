# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencode_x= LabelEncoder()
x[:, 3]=labelencode_x.fit_transform(x[:, 3])
onehotencode=OneHotEncoder(categorical_features=[3])
x=onehotencode.fit_transform(x).toarray()

#In order to avoid dummy variable trap remove the first column
x = x[:, 1:]



#splitting dataset into trainning set and test set
from sklearn.cross_validation import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=0)

'''#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)'''

#fitting into multiple linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predicting test sets results
predict_y = regressor.predict(x_test)

#implementing backward elemination
#the model is fitted with all predictors
import statsmodels.formula.api as sfa
x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1)
optimized_x = x[:, [0,1,2,3,4,5]]
regressor_OLS = sfa.OLS(endog = y, exog = optimized_x ).fit()
print(regressor_OLS.summary())
optimized_x = x[:, [0,1,3,4,5]]
regressor_OLS = sfa.OLS(endog = y, exog = optimized_x ).fit()
print(regressor_OLS.summary())
optimized_x = x[:, [0,3,4,5]]
regressor_OLS = sfa.OLS(endog = y, exog = optimized_x ).fit()
print(regressor_OLS.summary())
optimized_x = x[:, [0,3,5]]
regressor_OLS = sfa.OLS(endog = y, exog = optimized_x ).fit()
print(regressor_OLS.summary())
optimized_x = x[:, [0,3]]
regressor_OLS = sfa.OLS(endog = y, exog = optimized_x ).fit()
print(regressor_OLS.summary())


x_train1 , x_test1 , y_train1 , y_test1 = train_test_split(optimized_x, y , test_size=0.2 , random_state=0)

regressor1 = LinearRegression()
regressor1.fit(x_train1, y_train1)

predict_new_y = regressor1.predict(x_test1)
plt.scatter(x_train1 , y_train1)
plt.plot(x_test1 , predict_new_y , color = 'blue')

