# import tensorflow, keras, numpy, and matplotlib
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import matplotlib.image as mpimg
import cv2
from PIL import Image
from keras.callbacks import EarlyStopping
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score

#load the data
train_folder = '/datasources/tomatodiseases/train'
test_folder = '/datasources/tomatodiseases/valid'

#declare labels and image size
labels = os.listdir(train_folder)
img_size = 255
num_labels = len(labels)

#create function that gets the data from the folders
def get_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
                img_arr = cv2.imread(os.path.join(path, img)) 
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
    return np.array(data)

#retrieve the data, store as train and test
train = get_data(train_folder)
test = get_data(test_folder)

#organizing the data into train test splits, with x being data, y being label, etc
train_x = []
train_y = []
for features, label in train:
    train_x.append(features)
    train_y.append(label)

#when we reshape the data here, we have to set it to 2d, and thats what the next line does
train_x = np.array(train_x).reshape(len(train_x),-1)
train_y = np.array(train_y)

test_x = []
test_y = []
for features, label in test:
    test_x.append(features)
    test_y.append(label)

#when we reshape the data here, we have to set it to 2d, and thats what the next line does
test_x = np.array(train_x).reshape(len(train_y),-1)
test_y = np.array(train_y)

train_x = train_x / 255
test_x = test_x / 255

#double check the shape
print(train_x.shape)
print(test_x.shape)

#trying numerous different cluster values
total_clusters = [11,22,44,66,88,99,110,121]

#pre allocating empty results
results = []

#for each number of clusters to test in total_clusters,
for clusters in total_clusters:

    print('Training with cluster size ',clusters)
    
    #create the mini batch of kmeans
    kmeans = MiniBatchKMeans(n_clusters = clusters)

    #fit the training data to kmeans
    kmeans.fit(train_x)

    #get the kmeans labels
    kmeans.labels_

    #get the associated cluster label from the training data
    def retrieve_info(cluster_labels,train_y):

     #pre allocating empty reference_labels set
     reference_labels = {}
     
     # For loop to run through each label of cluster label
     for i in range(len(np.unique(kmeans.labels_))):
         index = np.where(cluster_labels == i,1,0)
         num = np.bincount(train_y[index==1]).argmax()
         reference_labels[i] = num
     return reference_labels

    #get all the ref labels
    reference_labels = retrieve_info(kmeans.labels_,train_y)

    number_labels = np.random.rand(len(kmeans.labels_))
    for i in range(len(kmeans.labels_)):
        number_labels[i] = reference_labels[kmeans.labels_[i]]
    print(number_labels[:100].astype('int'))
    print(train_y[:100])

    #print and store results at the end of each k-means cluster made
    print(accuracy_score(number_labels,train_y))
    results.append([clusters,accuracy_score(number_labels,train_y)])
    print(results)

print(results)
