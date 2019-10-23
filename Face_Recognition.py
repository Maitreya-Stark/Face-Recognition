#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 22:45:05 2018

@author: maitreyasatavalekar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 12:38:08 2018

@author: maitreyasatavalekar
"""
#%%
import cv2
import numpy as np
from random import shuffle
from tqdm import tqdm
import os
import time
from sklearn.metrics import confusion_matrix

#%%
Train_dir = '/Users/maitreyasatavalekar/Desktop/Batch Converted Face extracted'
Test_dir = '/Users/maitreyasatavalekar/Desktop/TestData'
Img_size = 64
LR = 0.0001
Model_name_tf = 'FaceRecognition-{}-{}-TensorFlow.model'.format(LR,'6-conv-basic')
Model_name_keras = 'FaceRecognition-{}-{}-Keras.model'.format(LR,'6-conv-basic')


#%% Process data
def label_img(img):
    word_label = img.split('.')[0]
    if word_label =='Ariel_Sharon':
        return [1,0,0,0,0,0,0,0,0,0,0]
    elif word_label =='Colin_Powell':
        return [0,1,0,0,0,0,0,0,0,0,0]
    elif word_label =='George_Bush':
        return [0,0,1,0,0,0,0,0,0,0,0]
    elif word_label =='Gerhard_Schroed':
        return [0,0,0,1,0,0,0,0,0,0,0]
    elif word_label =='Hugo_Chavez':
        return [0,0,0,0,1,0,0,0,0,0,0]
    elif word_label =='Jacques_Chirac':
        return [0,0,0,0,0,1,0,0,0,0,0]
    elif word_label =='Jean_Chretien':
        return [0,0,0,0,0,0,1,0,0,0,0]
    elif word_label =='John_Aschcroft':
        return [0,0,0,0,0,0,0,1,0,0,0]
    elif word_label =='Junichiro_Koizumi':
        return [0,0,0,0,0,0,0,0,1,0,0]
    elif word_label =='Serena_Williams':
        return [0,0,0,0,0,0,0,0,0,1,0]
    elif word_label =='Tony_Blair':
        return [0,0,0,0,0,0,0,0,0,0,1]

#%%
def create_training_data():
    training_data = []
    time.sleep(1.0)
    print('[INFO] Creating Training Data')
    for img in tqdm(os.listdir(Train_dir)):
        label = label_img(img)
        if label==None:
            continue
        path = os.path.join(Train_dir,img)
        #print(path,label)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(Img_size,Img_size))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('sc_train_data.npy',training_data)
    return training_data

#%%
def process_test_data():
    testing_data = []
    print('[INFO] Creating Testing Data')
    time.sleep(1.0)
    for img in tqdm(os.listdir(Test_dir)):
        path = os.path.join(Test_dir,img)
        img_num = img.split('.')[0]
        if img == '.DS_Store':
            continue
        #print(path,img_num)
        img =cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(Img_size,Img_size))
        testing_data.append([np.array(img),np.array(img_num)])
    np.save('sc_test_data.npy',testing_data)
    shuffle(testing_data)
    return testing_data

#%%
def predict_class_label(img_data):
    img_data = img_data.reshape(Img_size,Img_size,1)
    model_output = model.predict([img_data])
    if np.argmax(model_output) == 0: str_label = 'Ariel_Sharon'
    elif np.argmax(model_output) == 1: str_label = 'Colin_Powell'
    elif np.argmax(model_output) == 2: str_label = 'George_Bush'
    elif np.argmax(model_output) == 3: str_label = 'Gerhard_Schroed'
    elif np.argmax(model_output) == 4: str_label = 'Hugo_Chavez'
    elif np.argmax(model_output) == 5: str_label = 'Jacques_Chirac'
    elif np.argmax(model_output) == 6: str_label = 'Jean_Chretien'
    elif np.argmax(model_output) == 7: str_label = 'John_Aschcroft'
    elif np.argmax(model_output) == 8: str_label = 'Junichiro_Koizumi'
    elif np.argmax(model_output) == 9: str_label = 'Serena_Williams'
    elif np.argmax(model_output) == 10: str_label = 'Tony_Blair'
    return str_label
#%%Load Traing and Testing Data:
train_data = create_training_data()
#if already created the npy file use:
#train_data = np.load('sc_train_data.npy')
#print('[INFO] Training Data Loaded')

test_data = process_test_data()
#if already created use:
#test_data = np.load('sc_test_data.npy')
#print('[INFO] Testing Data Loaded')

#%%
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
tf.reset_default_graph()
convnet = input_data(shape=[None, Img_size, Img_size, 1], name='input')

convnet = conv_2d(convnet, 64, 11, activation='linear')
convnet = conv_2d(convnet, 96, 5, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 256, 3, activation='sigmoid')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 384, 3, activation='relu')
convnet = conv_2d(convnet, 384, 3, activation='linear')
convnet = conv_2d(convnet, 256, 5, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = fully_connected(convnet, 1024, activation='sigmoid')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 11, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

#%%
if os.path.exists('{}.meta'.format(Model_name_tf)):
    model.load(Model_name_tf)
    print("[INFO] Model Loaded")
    time.sleep(3.0)

#%%
#else:
train = train_data[:-50] 
test = train_data[-50:]

#%%
X = np.array([i[0] for i in train]).reshape(-1, Img_size, Img_size, 1)
Y = np.array([i[1] for i in train])

test_x = np.array([i[0] for i in test]).reshape(-1, Img_size, Img_size, 1)
test_y = np.array([i[1] for i in test])
#%%

y_pred = model.predict([train])
cm_train = confusion_matrix(Y,y_pred)
print(cm_train)



ytest_pred = model.predict([test_x])
cm_test = confusion_matrix(test_y,ytest_pred)
print(cm_test)
#%%
print('[INFO] Fitting Model: ',Model_name_tf)
model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=Model_name_tf)
model.save(Model_name_tf)
print('[INFO] Saved Model')

#%%
#import matplotlib.pyplot as plt
#fig =plt.figure(figsize=(13,13))
#correct = 0
#for num,data in enumerate(test_data[:40]):
#    img_num = data[1]
#    img_data = data[0]
#    y = fig.add_subplot(10,4,num+1)
#    og = img_data
#    str_label = predict_class_label(img_data)
#    if (str_label == img_num): 
#        print('{:2} | {:25} | {:10}'.format(num+1,img_num,"Correct")) 
#        correct +=1
#    else: 
#        print('{:2} | {:25} | {:10}'.format(num+1,img_num,"Wrong"))    
#    y = plt.imshow(og,cmap='gray')
#    plt.title(str_label)
#    y.axes.get_xaxis().set_visible(False)
#    y.axes.get_yaxis().set_visible(False)
#print("[INFO] Correct Clasifications :",correct)
#plt.show()

#%%
#def Create_Keras_Model():
#    from keras.models import Sequential
#    from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
#    input_shape = (64,64,1)
#    nClasses=11
#    
#    
#    model = Sequential()
#    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
#    model.add(Conv2D(32, (3, 3), activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))
#     
#    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#    model.add(Conv2D(64, (3, 3), activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))
#     
#    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#    model.add(Conv2D(64, (3, 3), activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))
#     
#    model.add(Flatten())
#    model.add(Dense(512, activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(nClasses, activation='softmax'))
#     
#    return model
#
##Train Keras Model:
#model_keras= Create_Keras_Model()
#batch_size = 50
#epochs = 10
#
#if os.path.exists('{}.meta'.format(Model_name_keras)):
#    model_keras.load(Model_name_tf)
#    print("[INFO] Model Loaded")
#    time.sleep(3.0)
#
#train = train_data[:-50] 
#test = train_data[-50:]
#
#X = np.array([i[0] for i in train]).reshape(-1, Img_size, Img_size, 1)
#Y = np.array([i[1] for i in train])
#
#test_x = np.array([i[0] for i in test]).reshape(-1, Img_size, Img_size, 1)
#test_y = np.array([i[1] for i in test])
#
#
#
#model_keras.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# 
#history = model_keras.fit(X, Y, batch_size = batch_size, epochs=epochs, verbose=1, 
#                   validation_data=(test_x, test_y))
#model_keras.save(Model_name_keras) 
#model_keras.evaluate(test_data[0], test_data[1])
#
#import matplotlib.pyplot as plt
#
## Loss Curves
#plt.figure(figsize=[8,6])
#plt.plot(history.history['loss'],'r',linewidth=3.0)
#plt.plot(history.history['val_loss'],'b',linewidth=3.0)
#plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
#plt.xlabel('Epochs ',fontsize=16)
#plt.ylabel('Loss',fontsize=16)
#plt.title('Loss Curves',fontsize=16)
# 
## Accuracy Curves
#plt.figure(figsize=[8,6])
#plt.plot(history.history['acc'],'r',linewidth=3.0)
#plt.plot(history.history['val_acc'],'b',linewidth=3.0)
#plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
#plt.xlabel('Epochs ',fontsize=16)
#plt.ylabel('Accuracy',fontsize=16)
#plt.title('Accuracy Curves',fontsize=16)
#
#
#
