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

def resize_image(image, wanted_height) :
    """
    Resizing the image for the given height, without disortion, using opencv-python
    :param image: the image that I want to resize
    :param wanted_height: the height in pixel that I want to have in the resized image
    :return: the resized image
    """
    height = image.shape[0]
    # calculating the amount with I need to change the width
    scale_percent = height / wanted_height
    width = int(image.shape[0] / scale_percent)
    dim = (width, wanted_height)
    
    resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    
    return resized_image

def kmeans_segmentation(image, k):
    """
    Segmenting the image using opencv-python's k-means method 
    :param image: the image that I want to carry out the segmentation
    :param k: the number of the clusters
    :return: the compactness, the pixel_values and the labels from the k-means method
    """
    pixel_values = image.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    compactness, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    labels = labels.flatten()
    
    return compactness, pixel_values, labels