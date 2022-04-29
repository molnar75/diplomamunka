import cv2
import math
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def get_window(image, row_number, column_number, px):
    """
    Method for getting the px x px sized window from the image.
    :param image: the image that I want to get the window from
    :param row_number: the row where the window left corner pixel should start
    :param column_number: the column where the window left corner pixel should start
    :param px: the width and height of the window
    :return: the px x px sized window from the image
    """
    window = np.zeros((px, px))

    for i in range(0, px):
        column = np.zeros((px))
        for j in range(0, px):
            column[j] = image[row_number+i][column_number+j]
        window[i] = column
    
    return window

def get_window_values(image, window_quantity, px):
    """
    Method for getting random windows from the image, stacked in one vector. 
    The resulted vector size will be window_quantity x px*px.
    :param image: the image that i want to get the random windows from
    :param window_quantity: the amount of windows that i want to get
    :param px: the width and height of the windows
    :return: the result, which is the vector created from the 1 x px*px sized windows, and the pixels, an array that contains the start pixels of the windows
    """
    height = image.shape[0]
    width = image.shape[1]
    
    pixels = np.zeros((window_quantity, 2))
    
    #get random window start pixels
    x = np.random.randint(0, height-px, window_quantity)
    y = np.random.randint(0, width-px, window_quantity)
    
    for i in range(window_quantity):
        window = get_window(image, x[i], y[i], px)
        window = window.reshape((1, -1))
        pixels[i] = [x[i], y[i]] #saving the start pixels of the windows
        
        if i == 0:
            result = np.array(window)
        else:
            result = np.vstack([result, window])
    
    return result, pixels

def get_label_map(height, width, pixels, labels, px):
    """
    Method for getting the label to each pixel of the windows. 
    :param height: the height of the image that contains the windows
    :param width: the width of the image that contains the windows
    :param pixels: the starting pixels of the previously calculated windows
    :param px: the width and height of the windows
    :return: an array in the shape of the image, the values are -1 where the pixel is not segmented, the labels elswhere
    """
    label_map = np.full((height, width), -1)

    for i in range(len(pixels)):
        for j in range(0, px):
            for k in range(0, px):
                label_map[int(pixels[i][0]+j)][int(pixels[i][1]+k)] =\
                    labels[i]
                
    return label_map

def color_image(image, label_map):
    """
    Method for colorizing the image based on the label_map. 
    :param image: the image which I want to colorize
    :param label_map: the map of the labels, I choose the color of the pixels by the label given to the pixel
    :return: the colorized image
    """
    height = image.shape[0]
    width = image.shape[1]
    
    colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    colors = np.array([
        [255, 0, 0],     #red
        [255, 165, 0],   #orange
        [255, 255, 0],   #yellow
        [0, 255, 0],     #lime
        [0, 128, 0],     #green
        [0, 255, 180],   #aquamarine
        [0, 255, 255],   #cyan
        [0, 100, 255],   #light blue
        [0, 0, 255],     #blue
        [128, 0, 128],   #purple
        [255, 0, 255],   #magenta
        [0, 0, 0],       #black
        [255, 255, 255]  #white
       ])

    for i in range(height):
        for j in range(width):
            if(label_map[i][j] != -1):
                colored_image[i][j] = colors[label_map[i][j]]
                
    return colored_image

def determine_K_value(values, labels):
    """
    Method for determining the best K value for KNN method.
    :param values: the values that I want to use to teach the modell
    :param labels: the labels to the values
    :return: the best K value, the training values and labels for teaching the model
    """
    values_train, values_test, labels_train, labels_test = train_test_split(values, labels, stratify = labels, train_size = 0.8)

    root =  int(math.sqrt(values_train.shape[0]))

    neighbors = np.arange(root-10, root+10)
    
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    for i, k in enumerate(neighbors):
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(values_train, labels_train)

        train_accuracy[i] = knn_model.score(values_train, labels_train)
        test_accuracy[i] = knn_model.score(values_test, labels_test)

    max_train_k = neighbors[np.where(train_accuracy == max(train_accuracy))]
    max_test_k = neighbors[np.where(test_accuracy == max(test_accuracy))]

    chosen_k = -1
    if(len(max_test_k) > 1):
        for k in max_test_k:
            if(k in max_train_k):
                chosen_k = k
                break
    else:
        chosen_k = max_test_k[0]

    if(chosen_k == -1):
        chosen_k = np.random.choice(max_test_k)
    
    return chosen_k, values_train, labels_train