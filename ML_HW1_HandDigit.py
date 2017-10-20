#Name: Nguyen Cao Minh ID: 14520529
#Class: Machine Learning in Computer Vision
#HOMEWORK 1: Clustering in Machine Learning
    #Exercise 2: Clustering with HandDigit dataset 
        #Updated date: 19-Oct-2017

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn import cluster
from sklearn.cluster import DBSCAN, SpectralClustering, KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn import metrics

data = load_digits().data
label = load_digits().target
n_clusters = 10


# #############################################################################
# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=2).fit_transform(data)

plt.figure(1)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=label)
plt.title("STANDARD")

# =================================================================
# KMEAN CLUSTERING

kmean = KMeans(init='k-means++', n_clusters=n_clusters).fit_predict(data)
kmean_accuracy = metrics.adjusted_mutual_info_score(label, kmean)

plt.figure(2)
plt.subplot(221)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmean)
plt.title("HAND DIGITS CLUSTERING USING KMEAN %0.2f"%(kmean_accuracy*100) + "%")

# =================================================================
# SPECTRAL CLUSTERING

spectral = SpectralClustering(n_clusters=n_clusters,affinity="nearest_neighbors").fit_predict(data)
spectral_accuracy = metrics.adjusted_mutual_info_score(label, spectral)
plt.subplot(222)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=spectral)
plt.title("HAND DIGITS CLUSTERING USING SPECTRAL %0.2f"%(spectral_accuracy*100) + "%")

# =================================================================
# AGGLOMERATIVE CLUSTERING

agglomerative = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(data)
agglomerative_accuracy = metrics.adjusted_mutual_info_score(label, agglomerative)
plt.subplot(223)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=agglomerative)
plt.title("HAND DIGITS CLUSTERING USING AGGOLOMERATIVE %0.2f"%(agglomerative_accuracy*100) + "%")

# =================================================================
# DBSCAN CLUSTERING

db = DBSCAN(eps=21.5, min_samples=9).fit_predict(data)
db_accuracy = metrics.adjusted_mutual_info_score(label, db)
plt.subplot(224)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=db)
plt.title("HAND DIGITS CLUSTERING USING DBSCAN %0.2f"%(db_accuracy*100) + "%")
plt.show()