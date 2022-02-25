import cv2
import numpy as np

#Method for importing the image by the given image name
def load_image(name):
    image = cv2.imread('../images/' + format(name) + '.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return image

# Method for resizing the image by the given height
def resize_image(image, wanted_height) :
    height = image.shape[0]
    scale_percent = height / wanted_height
    width = int(image.shape[0] / scale_percent)
    dim = (width, wanted_height)
    
    resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    
    return resized_image

# Method for grayscale image segmentation
def image_segmentation(image, k):
    pixel_values = image.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    labels = labels.flatten()
    
    return pixel_values, labels