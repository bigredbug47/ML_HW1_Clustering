#Name: Nguyen Cao Minh ID: 14520529
#Class: Machine Learning in Computer Vision
#HOMEWORK 1: Clustering in Machine Learning
    #Exercise 3.1: Extract data and label from face dataset using LBT feature 
        #Updated date: 19-Oct-2017

from sklearn.datasets import fetch_lfw_people
from skimage.feature import local_binary_pattern
import numpy as np


lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    
X = np.array([]).reshape(0,1850)
for image in lfw_people.images:
    lbt_image = local_binary_pattern(image,P=8,R=0.5).flatten()
    X = np.append(X,[lbt_image],axis=0)
np.save(file='face_data.npy',arr=X)
np.save(file='face_label.npy', arr=lfw_people.target)

