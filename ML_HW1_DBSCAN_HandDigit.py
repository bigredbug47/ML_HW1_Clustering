#Name: Nguyen Cao Minh ID: 14520529
#Class: Machine Learning in Computer Vision
#HOMEWORK 1: Clustering in Machine Learning
    #Exercise 2.3: DBSCAN Clustering with HandDigit Written dataset
        #Updated date: 05-Oct-2017


import numpy as np


from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


# #############################################################################
# Generate sample data
digits = load_digits()
data=scale(digits.data)

X = PCA(n_components=2).fit_transform(data)


# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.355, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# #############################################################################
# Plot result
import matplotlib.pyplot as plt
#reduced_data = PCA(n_components=2).fit_transform(X)
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()