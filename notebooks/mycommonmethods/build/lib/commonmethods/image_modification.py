import cv2
import numpy as np

def load_image_grayscale(name):
    """
    Loading the image from the images folder using opencv-python
    :param name: the name of the image I want to load, the method doesn't need the path or the extension
    :return: the loaded image in grayscale color
    """
    image = cv2.imread('../images/' + format(name) + '.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return image

def load_image_rgb(name):
    """
    Loading the image from the images folder using opencv-python
    :param name: the name of the image I want to load, the method doesn't need the path or the extension
    :return: the loaded image in rgb color
    """
    image = cv2.imread('../images/' + format(name) + '.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image

def resize_image(image, desired_height) :
    """
    Resizing the image for the given height, without disortion, using opencv-python
    :param image: the image that I want to resize
    :param desired_height: the height in pixel that I want to have in the resized image
    :return: the resized image
    """
    height = image.shape[0]
    # calculating the amount with I need to change the width
    scale_percent = height / desired_height
    width = int(image.shape[1] / scale_percent)
    dim = (width, desired_height)
    
    resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    
    return resized_image

def kmeans_segmentation(values, k):
    """
    Segmenting the given values using opencv-python's k-means method 
    :param values: the values at which I want to perform segmentation
    :param k: the number of the clusters
    :return: the compactness, the labels and the centers from the k-means method
    """
    values = np.float32(values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    compactness, labels, (centers) = cv2.kmeans(values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    labels = labels.flatten()
    
    return compactness, labels, (centers)