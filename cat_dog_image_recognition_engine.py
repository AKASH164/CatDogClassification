# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:02:18 2020

Project:- cat and dog image recognition

@author: Author
"""
# import the required libraries
import cv2 # for image resizing
import numpy as np
import os # for setting directories
from random import shuffle # to shuffle the data
from tqdm import tqdm # for looping with progress bar

TRAIN_DIR = 'C:/Users/User/Desktop/Hand Raise Recognition/practice_cats_dogs/train'
# TEST_DIR = 'C:/Users/User/Desktop/Hand Raise Recognition/practice_cats_dogs/test'
path = 'C:/Users/User/Desktop/Hand Raise Recognition/practice_cats_dogs/'

# define parameters to be tuned
IMG_SIZE = 256 # size of all image files should be 50*50
LR = 1e-3 # learning rate for optimisation algorithm 

MODEL_NAME = 'dogsvscats-{}-{}-{}-'.format(LR, IMG_SIZE, '6conv-3hidden')

# function to define a word label for our target
def label_img(img):
    # name of the image file are cat.1.jpg, cat.2.jpg ... , dog.1.jpg, dog.2.jpg .... on the train data
    word_label = img.split('.')[-3] # we need to extract only cat and dog label from each file
    
    if word_label == 'cat':
        return [1, 0] # for cat return [1, 0] to target array
    elif word_label == 'dog':
        return [0, 1] # for dog return [0, 1] to target array

# function to create train data (features)
def create_train_data():
    training_data = [] # create a empty list to store training data
    
    for img in tqdm(os.listdir(TRAIN_DIR)): # iterate file names through train directory
        label = label_img(img) # call the label_img functions to create labels for each image file
        
        img_path = os.path.join(TRAIN_DIR, img) # create the full path for each image file
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # read the image file in gray scale
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) # resize the image 
        
        # the image data will be nothing but a 2d numerical array
        # training dataset is appended for each image file
        training_data.append([np.array(img), np.array(label)])
        
    # then we will shuffle the training data with correspondence
    shuffle(training_data)
        
    np.save(path + 'training_data.npy', training_data) # save the numpy array
        
    return training_data
'''    
# function to process test data (features)
def process_test_data():
    testing_data = [] # create a empty list to store testing data
    
    for img in tqdm(os.listdir(TEST_DIR)): # iterate file names through test directory
        # name of the image file are 1.jpg, 2.jpg ... on the test data
        img_num =  img.split('.')[0] # this will be image idntifier
        
        path = os.path.join(TEST_DIR, img) # create the full path for each image file
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # read the image file in gray scale
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) # resize the image 
        
        # the image data will be nothing but a 2d numerical array
        # testing dataset is appended for each image file
        testing_data.append([np.array(img), np.array(img_num)])
        
    # then we will shuffle the testing data with correspondence
    shuffle(testing_data)
        
    np.save(path + 'testing_data.npy', testing_data) # save the numpy array
        
    return testing_data
 '''   
# call the training_data function to create training data
# training_data = create_train_data() # for 1st time or if there is any change on the dataset
# if train data is already created, then load it
training_data = np.load(path + "training_data.npy", allow_pickle = True) 
# data size reduced from 550 MB to 62.4 MB

'''
## architecture of convolutional neural network
# import the required libraries
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# input layer will consists of 50*50 numpy aray for each image
convnet = input_data(shape = [None, IMG_SIZE, IMG_SIZE, 1], name = "Input")
# 1st element uld be 'None' (representing batch size), 
# if 1st element is not provided, it will be added automatically
# size of input is 50*50*1 for black & white and 50*50*3 for RGB

# 1st hidden convolutional layers
convnet = conv_2d(convnet, 32, 2, activation = 'relu') # filter = 32 and stride = 2 
convnet = max_pool_2d(convnet, 2) # stride = 2 
# number of filters and strides should be tuned

# 2nd hidden convolutional layers
convnet = conv_2d(convnet, 64, 2, activation = 'relu') # filter = 64 and stride = 2 
convnet = max_pool_2d(convnet, 2) # stride = 2 
# number of filters and strides should be tuned

# 3rd hidden layer is a fully connected layer
convnet = fully_connected(convnet, 1024, activation = 'relu') # total number of neurons = 1024
# number of neurons should be tuned
convnet = dropout (convnet, 0.8) # for regularisation and avoid overfitting problems

# output layer
convnet = fully_connected(convnet, len(training_data[0][1]), activation = 'softmax') 
# number of neurons = number of labels = 2

# define the optimizer for the loss function
convnet = regression(convnet, 
                     optimizer = 'adam', 
                     loss = 'categorical_crossentropy',
                     learning_rate = LR, # to be tuned
                     name = 'targets')

model = tflearn.DNN(convnet, tensorboard_dir = 'log')

# we can also train the model in installments such as 10 epochs at time for quick tuning of parameters.
# therefore we can save the model after 10 epochs and start from there in the next sesssion
if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load()
    print('{} model loaded'.format(MODEL_NAME))
'''
  
## separate train and validation set from training data
# as the training data is randomly shuffled already no need to randomly pick for train and validation
# we can pick first 1000 images out of 25000 for validation 
val = training_data[0:1000]
train = training_data[1000:]

## separate features and target in both train and validation set
# train data is a list of 24000 element where each element is also a list of two numpy array
# the first numpy array is for features and 2nd numpy array is for labels/target
# so we iterate over the list
train_x = [i[0] for i in train] # list of features for each image in training set (24000 elem)
# convert into numpy array for reshaping & reshape according the input layer of CNN
train_x = np.array(train_x).reshape(-1, IMG_SIZE, IMG_SIZE, 1) 
# -1 represents all 24000 images. numpy will figure that out while reshaping

train_y = [i[1] for i in train] # labels for each image in train data
train_y = np.array(train_y) # convert to numpy array from list

val_x = [i[0] for i in val] # list of features for each image in validation set (1000 elem)
# convert into numpy array for reshaping & reshape according the input layer of CNN
val_x = np.array(val_x).reshape(-1, IMG_SIZE, IMG_SIZE, 1) 

val_y = [i[1] for i in val] 
val_y = np.array(val_y)       

## Architecture of convolutional neural network        

# Import required libraries for CNN
# Importing the Keras libraries and packages
from keras.models import Sequential, load_model  
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Flatten 
from keras.optimizers import SGD # for building own optimizers instead of default
# We can evaluate the performance of our model based on train dataset using k-fold cross validation
# The CNN model requires the keras and the k-fold cross validation requires the scikit-learn 
# Hence, we need to somehow combine these library. A wrapper function is available in the keras library 
# called KerasClassifier, that wraps the k-fold cross validation function from sci-kit learn.
from keras.wrappers.scikit_learn import KerasClassifier
# Now we need the k-fold cross validation function from sci-kit learn
from sklearn.model_selection import cross_val_score

def build_model():
    # CNN Architecture
    model = Sequential() # creating a instance of keras sequential class
    
    # 1st hidden convolutional layers
    # adjust the weights of each neuron based not only on each data point in the input vector, 
    # but also on the surrounding data points
    model.add(Conv2D(filters = 32, # numbers of filters that convolutional layers will learn from. 
                     # use powers of 2 for filters, e.g. 4, 8, 16, 32, 64, etc.
                     # each filter is responsible for extracting some  feature from a raw image. 
                     kernel_size = (3, 3), # dimensions of each kernel/ filter. 
                     # kernel_size parameter must be an odd integer
                     strides = (1, 1), # specify “step” of the convolution along with 
                     # the height and width of the input data.
                     activation ='relu', # activation function
                     input_shape = (IMG_SIZE, IMG_SIZE, 1))) # input data size
    # Pooling layer is responsible for reducing the spatial size of the Convolved Feature.This is 
    # done to decrease the computational power required to process the data through 
    
    # dimensionality reduction. There are two types of Pooling: Max Pooling and Average Pooling
    model.add(MaxPooling2D(pool_size = (2,2))) # window size for pooling 
    # Batch Normalization is a technique that can dramatically reduce the time required to train 
    # a deep neural network.
    # Normalize the batch which allows higher learning rate and regularisation
    model.add(BatchNormalization())

    # 2nd hidden convolutional layers
    model.add(Conv2D(filters = 64, kernel_size = (3,3), activation ='relu')) 
    # number of filters increased from 32 to 64
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(BatchNormalization())
    
    # 3rd hidden convolutional layers
    model.add(Conv2D(filters = 128, kernel_size = (3,3), activation ='relu'))
    # number of filters increased from 64 to 128
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(BatchNormalization())

    # 4th hidden convolutional layers
    model.add(Conv2D(filters = 256, kernel_size = (3,3), activation ='relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(BatchNormalization())
    
    # 5th hidden convolutional layers (use this layer when image size is more than 256)
    model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(BatchNormalization())
    
    # 6th hidden convolutional layers (use this layer when image size is more than 256)
    model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(BatchNormalization())
    
    # # 7th hidden convolutional layers (use this layer when image size is more than 256)
    # model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
    # model.add(MaxPooling2D(pool_size = (2,2)))
    # model.add(BatchNormalization())
    
    # dropout layers prevent overfitting, which occurs when a neural network learns its 
    # input data a little too well.  
    model.add(Dropout(rate = 0.1)) # dropout rate is 0.1
    
    # Flatten layer
    # to flattens our previous multi-dimensional layers into a single-dimensional vector for
    # building a traditional neural network (Non- CNN)
    model.add(Flatten())
    
    # 6th hidden traditional neural network
    model.add(Dense(units = 128, kernel_initializer = 'uniform', activation='relu'))
    model.add(Dropout(rate = 0.1)) # to reduce overfitting

    # 7th hidden traditiona neural network
    model.add(Dense(units = 64, kernel_initializer = 'uniform', activation='relu'))
    model.add(Dropout(rate = 0.1)) # to reduce overfitting (could be removed)
    
    # 7th hidden traditiona neural network
    model.add(Dense(units = 32, kernel_initializer = 'uniform', activation='relu'))
    model.add(Dropout(rate = 0.1)) # to reduce overfitting (could be removed)
    
    # output layer
    model.add(Dense(units = len(training_data[0][1]), activation = 'softmax'))
              
    # Compile model to find the best weights using the adam optimizer  
    model.compile(loss = 'binary_crossentropy', optimizer ='adam', metrics = ['accuracy'])
    
    return model

# Now we have build a object from class KerasClassifier which will take a function as input argument
model = KerasClassifier(build_fn = build_model, batch_size = 32, epochs = 17 )
# Above classifier variable is a global one

# Now the cross_val_score will be used which will return 10 accuries for 10-fold cross validation
accuracies = cross_val_score(estimator = model, X = train_x, y = train_y, cv = 5, n_jobs = 1)
print(accuracies)

mean = accuracies.mean() # mean = 84.14 %
standard_deviation = accuracies.std() # # standard deviation = 0.69 % (no overfitting problem as sd is low)

model = build_model() # call the build_model to get the model architecture
# fit the training data into the model 
model.fit(train_x, train_y, batch_size = 50, epochs = 20, verbose = 1)

# save the model
model.save( path + MODEL_NAME + "model.h5")
 
## Load the model
model = load_model(path + MODEL_NAME + 'model.h5')

## Validation
# predict probabilities
model_out = model.predict(val_x) # predict the probability score

res_index = np.argmax(model_out, axis = 1) # find the index of highest probability for each row
# cat:[1, 0] 
# dog:[0, 1]
# hence index of cat is 0 and dog is 1, that is cat is 0 and dog is 1
dog_pred = res_index # res_index is nothing but whether image is dog or not, 1=dog, 0=cat

# now we have to find the second element of val_y to find whether the image is dog or not
dog_actual = val_y[:,1]

from sklearn.metrics import accuracy_score # bring the metrics accuracy_score from sklearn
accuracy_score(dog_actual, dog_pred)








