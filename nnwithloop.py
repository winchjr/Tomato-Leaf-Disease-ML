# import tensorflow, keras, numpy, and others
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import matplotlib.image as mpimg
import cv2
from keras.callbacks import EarlyStopping

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
train_x = np.array(train_x).reshape(-1, img_size, img_size, 3)
train_y = np.array(train_y)

test_x = []
test_y = []
for features, label in test:
    test_x.append(features)
    test_y.append(label)
test_x = np.array(train_x).reshape(-1, img_size, img_size, 3)
test_y = np.array(train_y)

#verify the data is of correct shape, type, etc
print('In the training data, we are working with',len(train_x),'number of images.')
print('The type of data we are working with is', type(train_x[0]))
print('The shape of data we are working with is', train_x.shape)
print('The min and max image value range for training images, respectively:',(np.min(train_x), np.max(train_x)))
print('The # of labels we are working with:', num_labels)
print('The min and max image value range for testing images, respectively:',(np.min(test_x), np.max(test_x)))
print('The full list of labels: ', labels)

#scaling images appropriately, to be between 0 and 1
train_x = train_x / 255
test_x = test_x / 255

#implement early stopping, so if we stop improving, we stop training the net. this means if the accuracy doesnt improve by .01 in 2 attempts, to stop.
es = EarlyStopping(monitor='accuracy', min_delta=.01, patience=2)

#accuracy data, pre allocate
acc_data = []

#optimizer list to test
optimizers = ['SGD','RMSprop','Adam','Adadelta','Adagrad','Adamax','Nadam','Ftrl']

#was having issues where IDLE would stall after trying a few optimizers- would start again from here by uncommenting
#optimizers = ['Adamax','Nadam','Ftrl']

###begin loop

#for each optimizer, create a neural net model, run it, and store the accuracy/loss metrics associated with that optimizer.
for o in optimizers:
    
    print('Now fitting model with optimizer',o)
    
    #building the architecture of the model
    firstnet = tf.keras.Sequential([
        layers.Flatten(input_shape=(255,255, 3), name='input_layer'), #reshaping image to vector (flattening)
        layers.Dense(units=32, activation='relu', name='hidden1'),
        #layers.Dense(units=128, activation='relu', name='hidden1'), #uncomment these to do 1, 2, or 3 hidden layers with 128 neurons
        #layers.Dense(units=128, activation='relu', name='hidden2'),
        #layers.Dense(units=128, activation='relu', name='hidden3'),
        layers.Dense(units=num_labels, activation='softmax', name='output_layer')
    ], name='firstnet') 

    
    firstnet.compile(optimizer=o, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    firstnet.summary()

    #train up to 50 epochs. With the callback, it will stop if it doesnt improve after a few tries.
    firstnet.fit(x=train_x, y=train_y, epochs=50, batch_size=128, callbacks=[es])

    #get the accuracy/loss metrics
    test_loss, test_acc = firstnet.evaluate(test_x,test_y, verbose=1)

    print("\nTest accuracy:", test_acc)

    #save the results
    acc_data.append([o,test_acc,test_loss])
    print(acc_data)
    
    ###end loop

#print all the results 
print(acc_data)
