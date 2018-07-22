# -*- coding: utf-8 -*-
#k means clustering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, 3:5].values

#using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11), wcss)
plt.title('Elbow method')
plt.xlabel('number of clusters')
plt.ylabel('wcss')

#Applying k means algorithm
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
predict = kmeans.fit_predict(x)

plt.scatter(x[predict ==0, 0],x[predict==0, 1], s=50, c='red',label='careful')
plt.scatter(x[predict ==1, 0],x[predict==1, 1], s=50, c='green',label='standard')
plt.scatter(x[predict ==2, 0],x[predict==2, 1], s=50, c='blue',label='target')
plt.scatter(x[predict ==3, 0],x[predict==3, 1], s=50, c='cyan',label='careless')
plt.scatter(x[predict ==4, 0],x[predict==4, 1], s=50, c='magenta',label='sensible') 
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300,c= 'yellow', label='centroid')   
plt.xlabel('Salary')
plt.ylabel('spending rate')
plt.legend()
plt.show()
