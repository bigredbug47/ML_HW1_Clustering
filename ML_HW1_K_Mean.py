#Name: Nguyen Cao Minh ID: 14520529
#Class: Machine Learning in Computer Vision
#HOMEWORK 1: Clustering in Machine Learning
    #Exercise 1: K-Mean Clustering with 2 Gaussians
        #Updated date: 04-Oct-2017


import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

n_samples = 1500
random_state = 340
X, y = make_blobs(n_samples=n_samples, random_state=random_state)


plt.figure(1)

y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)


plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("KMean with 2 Gaussian")

plt.show()
