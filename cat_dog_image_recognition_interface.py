# -*- coding: utf-8 -*-
"""
Created on Sat May  2 15:51:46 2020

@author: user
"""
# import the required libraries
import cv2 # for image resizing
import numpy as np
import os # for setting directories
from random import shuffle # to shuffle the data
from tqdm import tqdm # for looping with progress bar

TEST_DIR = 'C:/Users/User/Desktop/Hand Raise Recognition/practice_cats_dogs/test'
path = 'C:/Users/User/Desktop/Hand Raise Recognition/practice_cats_dogs/'

IMG_SIZE = 256 # size of all image files should be 50*50
LR = 1e-3 # learning rate for optimisation algorithm 

MODEL_NAME = 'dogsvscats-{}-{}-{}-'.format(LR, IMG_SIZE, '6conv-3hidden')
# function to process test data (features)
def process_test_data():
    testing_data = [] # create a empty list to store testing data
    
    for img in tqdm(os.listdir(TEST_DIR)): # iterate file names through test directory
        # name of the image file are 1.jpg, 2.jpg ... on the test data
        img_num =  img.split('.')[0] # this will be image idntifier
        
        img_path = os.path.join(TEST_DIR, img) # create the full path for each image file
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # read the image file in gray scale
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) # resize the image 
        
        # the image data will be nothing but a 2d numerical array
        # testing dataset is appended for each image file
        testing_data.append([np.array(img), np.array(img_num)])
        
    # then we will shuffle the testing data with correspondence
    shuffle(testing_data)
        
    np.save(path + 'testing_data.npy', testing_data) # save the numpy array
        
    return testing_data

# call the testing_data function to create testing data
testing_data = process_test_data() # for 1st time or if there is any change on the dataset
# if test data is already created, then load it
# testing_data = np.load(path + "testing_data.npy", allow_pickle = True) 
# data size reduced from 550 MB to 62.4 MB

## import the model
from keras.models import load_model 
## Load the model
model = load_model(path + MODEL_NAME + 'model.h5')

# plot the figure vs prediction
import matplotlib.pyplot as plt

fig = plt.figure()

for num,data in enumerate(testing_data[20:30]): # iterate over first 10 images   
# cat:[1, 0] 
# dog:[0, 1]

    img_num = data[1] # image number
    img_data = data[0] # image data in 50*50 numpy array
        
    y = fig.add_subplot(2, 5, num+1) # add a subplot for each image with 2*5 subplots.
    # as the num increases iteratively each subplots gets added
    
    original_img = img_data
    
    data = np.array(img_data).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # reshape the data to be fed for prediction
    
    model_out = model.predict(data) # find the probabilty score of both cat and dog  
    idx = np.argmax(model_out) # find the index with highest probability score
    # If the highest score is for index 0 then it is a cat else is a dog
    
    if idx == 0:
        str_label = 'Çat'
    else:
        str_label = 'Dog'
    
    y.imshow(original_img, cmap = 'gray')
    plt.title(str_label)
    # do not shows the ticks for both x and y axis
    
plt.show()
    
    
    
    
    
    





















