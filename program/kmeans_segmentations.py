import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#my modules
import segmentation_methods as sm
import commonmethods.image_modification as im
import commonmethods.optimal_cluster_number as ocn

def kmeans_segmentation(image, image_index):
    """
    Method for 
    :param image: the image that I want to get the window from
    :param image_index: the row where the window left corner pixel should start
    """
    pixel_values = image.reshape((-1, 1))
    
    #Getting the optimal cluster number
    db_min = [1000000, 2]
    for k in range(2, 11):
        _,labels,_ = im.kmeans_segmentation(pixel_values, k)
        
        _, db_score = ocn.davies_bouldin_method(pixel_values, labels)
        
        if(db_score < db_min[0]):
            db_min = [db_score, k]
            
    k = db_min[1]
    
    _,labels,(centers) = im.kmeans_segmentation(pixel_values, k)
    
    centers = np.uint8(centers)
    
    segmented_image = centers[labels.flatten()]
    
    segmented_image = segmented_image.reshape(image.shape)
    
    plt.figure(figsize=(6,6))
    
    plt.imshow(segmented_image, cmap="gray", vmin=0, vmax=255)
    plt.title("Image after K-means clustering")
    plt.axis("off")
    plt.savefig('results/' + format(image_index) + '/01_kmeans_segmented.png')

def kmeans_texture_segmentation(image, k, px, image_index, cnn_model):
    """
    Method for segmenting the image by texture and saving the results.
    :param image: the image that I want to segment
    :param k: number of the clusters
    :param px: the width and height of the windows
    :param image_index: the index of the image for saving the result
    :param cnn_model: the CNN modell for predicting the colors to the windows
    :return: the map of labels and the labels of the colors
    """
    window_values, pixels = sm.get_window_values(image, 3000, px)

    _, labels, _ = im.kmeans_segmentation(window_values, k)
    
    color_labels = predict_color_with_cnn(cnn_model, window_values, labels, k)
    
    chosen_k, values_train, labels_train  = sm.determine_K_value(window_values, labels)
    
    height = image.shape[0]
    width = image.shape[1]
    
    knn_model = KNeighborsClassifier(n_neighbors=chosen_k)
    knn_model.fit(values_train, labels_train)
    
    label_map = np.full((height, width), -1)

    for i in range(height-px):
        for j in range(width-px):
            if(label_map[i][j] == -1):
                test_window = sm.get_window(image, i, j, px)
    
                test_window = test_window.reshape((1, -1))
    
                predicted = knn_model.predict(test_window)
                
                for k in range(0, px):
                    for l in range(0, px):
                        label_map[i+k][j+l] = predicted

    colored_image_windows = sm.color_image(image, label_map)

    plt.figure(figsize=(6,6))
    plt.title("Image after K-means texture based clustering, \n classified by windows")
    plt.axis("off")
    plt.imshow(colored_image_windows)
    plt.savefig('results/' + format(image_index) + '/02_kmeans_texture_windows.png')
    
    window_values = window_values.reshape((-1, 1))

    label_map = np.full((height, width), -1)
    train_labels = []
    
    for i in range(len(pixels)):
        for j in range(0, px):
            for k in range(0, px):
                train_labels.append(labels[i])
    
    chosen_k = int(math.sqrt(window_values.shape[0]))
    
    knn_model = KNeighborsClassifier(n_neighbors=chosen_k)
    knn_model.fit(window_values, train_labels)
                
    for i in range(height):
        for j in range(width):
            if(label_map[i][j] == -1):
                test_pixel = image[i][j]
                test_pixel = test_pixel.reshape((1, -1))
    
                predicted = knn_model.predict(test_pixel)
                
                label_map[i][j] = predicted
    
    colored_image_pixels = sm.color_image(image, label_map)
    
    plt.figure(figsize=(6,6))
    plt.title("Image after K-means texture based clustering, \n classified by pixels")
    plt.axis("off")
    plt.imshow(colored_image_pixels)
    plt.savefig('results/' + format(image_index) + '/03_kmeans_texture_pixels.png')
    
    return label_map, color_labels

def predict_color_with_cnn(cnn_model, window_values, labels, k):
    """
    Method for predicting the colors to the windows with CNN.
    :param cnn_model: the CNN modell for predicting the colors to the windows
    :param window_values: the windows that I want to predict the colors for
    :param labels: the labels of the windows
    :param k: number of the clusters
    :return: the list of predicted colors to the labels
    """
    color_labels = np.zeros((k, 13), dtype = int)
    
    for i in range(len(labels)):
        test_window = window_values[i]
        
        test_window = test_window/255
        
        test_window = test_window.reshape((15, 15))
        test_window = np.expand_dims(test_window, axis = 0)
        
        color_label = np.argmax(cnn_model.predict(test_window))
                
        color_labels[labels[i]][color_label] = color_labels[labels[i]][color_label] + 1
        
    print(color_labels)
    return color_labels