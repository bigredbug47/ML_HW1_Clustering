#Name: Nguyen Cao Minh ID: 14520529
#Class: Machine Learning in Computer Vision
#HOMEWORK 1: Clustering in Machine Learning
    #Exercise 2.2: Spectral Clustering with HandDigit Written dataset
        #Updated date: 05-Oct-2017
        #Description: Change the source code and find the other way to compile the spectral clustering

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

data=digits.data
trained_data = metrics.pairwise.cosine_similarity(data)

scluster = cluster.SpectralClustering(n_clusters=10, eigen_solver='arpack', affinity="precomputed").fit_predict(trained_data)
reduced_data = PCA(n_components=2).fit_transform(data)

plt.figure(1)
plt.subplot(221)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=scluster)
plt.title("Spectral Clustering with PCA reduced")


digits = load_digits()

data=digits.data
similiar_data=np.corrcoef(data)

Y = cluster.SpectralClustering(n_clusters=10, eigen_solver='arpack', affinity="precomputed").fit_predict(similiar_data)
reduced_data = PCA(n_components=2).fit_transform(data)

plt.subplot(222)
plt.scatter(reduced_data[:,0],reduced_data[:,1],c=Y)
plt.title("Case 2 of Spectral Clustering")
plt.show()