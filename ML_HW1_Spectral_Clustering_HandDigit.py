#Name: Nguyen Cao Minh ID: 14520529
#Class: Machine Learning in Computer Vision
#HOMEWORK 1: Clustering in Machine Learning
    #Exercise 2.2: Spectral Clustering with HandDigit Written dataset
        #Updated date: 04-Oct-2017

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import cluster
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


from sklearn.cluster import SpectralClustering

np.random.seed(42)

digits = load_digits()
data = scale(digits.data)

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))

reduced_data = PCA(n_components=2).fit_transform(data)
scluster = cluster.SpectralClustering(n_clusters=10, eigen_solver='arpack', affinity="nearest_neighbors")
Z = scluster.fit_predict(reduced_data)

plt.figure(1)

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=Z)
plt.title("Spectral Clustering with PCA reduced")
plt.show()