import cv2
import numpy as np
from sklearn.model_selection import train_test_split

#my modules
import commonmethods.image_modification as im

#modules for creating the CNN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

def get_training_data():
    """
    Method for getting the training data for the CNN model.
    :return: the train and test datas with the labels
    """
    datas = []
    labels = []
    number_of_images = 11
    
    px = 15
    middle_index = int(px/2)
    window_quantity = 5000
    
    for i in range(number_of_images):
        image = cv2.imread('images/' + format(i) + '.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        resized_image = im.resize_image(image, 1024)
        
        height = resized_image.shape[0]
        width = resized_image.shape[1]
        
        x = np.random.randint(0, height-px, window_quantity)
        y = np.random.randint(0, width-px, window_quantity)
        
        for j in range(window_quantity):
            window = resized_image[x[j]:x[j]+px, y[j]:y[j]+px]
            
            h = window[middle_index][middle_index][0]*2
            s = window[middle_index][middle_index][1]/255
            v = window[middle_index][middle_index][2]/255
                        
            rgb_window = cv2.cvtColor(window, cv2.COLOR_HSV2RGB)
            gray_window = cv2.cvtColor(rgb_window, cv2.COLOR_RGB2GRAY)
            
            label = -1
            
            if(v < 0.1):
                label = 11
            elif(v > 0.9 and s < 0.1):
                label = 12
            elif(h < 20):
                label = 0
            elif(h >= 20 and h < 60):
                label = 1 
            elif(h >= 60 and h < 90):
                label = 2 
            elif(h >= 90 and h < 120):
                label = 3
            elif(h >= 120 and h < 150):
                label = 4
            elif(h >= 150 and h < 180):
                label = 5
            elif(h >= 180 and h < 210):
                label = 6
            elif(h >= 210 and h < 240):
                label = 7
            elif(h >= 240 and h < 270):
                label = 8
            elif(h >= 270 and h < 300):
                label = 9
            elif(h >= 300):
                label = 10
                
            if(label != -1):
                labels.append(label)
                
                if len(datas) == 0:
                    datas = np.array([gray_window])
                else:
                    datas = np.vstack([datas, [gray_window]])
      
    labels = to_categorical(labels)
    
    x_train, x_test, y_train, y_test = train_test_split(datas, labels, stratify = labels, train_size = 0.8)
    
    return x_train, x_test, y_train, y_test

def get_model():
    """
    Method for creating the CNN model.
    :return: the created CNN model
    """
    x_train, x_test, y_train, y_test = get_training_data()

    #normalizing the pixel values
    x_train=x_train/255
    x_test=x_test/255
    
    model=Sequential()
    
    model.add(Conv2D(64,(3,3),activation='relu',input_shape=(15, 15, 1)))
    model.add(MaxPool2D(2,2))
    
    model.add(Conv2D(32,(3,3),activation='relu'))
    model.add(MaxPool2D(2,2))
    
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    
    model.add(Dense(13, activation='softmax'))
    
    opt = Adam(learning_rate=0.001)
    
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=30, validation_data=(x_test, y_test))
    
    return model