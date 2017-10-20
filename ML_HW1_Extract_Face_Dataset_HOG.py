#Name: Nguyen Cao Minh ID: 14520529
#Class: Machine Learning in Computer Vision
#HOMEWORK 1: Clustering in Machine Learning
    #Exercise 3.2: Extract data and label from face dataset using HOG Feature
        #Updated date: 19-Oct-2017

from sklearn.datasets import fetch_lfw_people
from skimage.feature import local_binary_pattern
import numpy as np


lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    
X = []
for image in lfw_people.images:
    hog_image = hog(image, block_norm='L2')
    X.append(hog_image)
np.save(file='face_data_hog.npy',arr=X)


