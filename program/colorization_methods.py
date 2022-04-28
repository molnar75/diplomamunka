import cv2
import colorsys
import numpy as np
from matplotlib import pyplot as plt

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

def color_rgb(image, label_map, k, image_index):
    """
    Coloring the RGB image and saving it to the results folder.
    :param image: the image that I want to color
    :param label_map: the label map for the image
    :param k: the number of the clusters
    :param image_index: the index of the image for saving the result
    """
    colored_image_rgb_multiply = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    r, g, b = cv2.split(colored_image_rgb_multiply)
    
    for i in range(0, k):
        np.multiply(r, colors[i][0]/255, out=r, where=label_map==i, casting="unsafe")
        np.multiply(g, colors[i][1]/255, out=g, where=label_map==i, casting="unsafe")
        np.multiply(b, colors[i][2]/255, out=b, where=label_map==i, casting="unsafe")
    
    colored_image_rgb = cv2.merge([r, g, b])
    
    plt.figure(figsize=(6,6))
    plt.title("Image colorized in RGB colorspace")
    plt.axis("off")
    plt.imshow(colored_image_rgb)
    plt.savefig('results/' + format(image_index) + '/04_rgb_colorization.png')
    
    
def color_by_predicted_colors(image, label_map, color_labels, k, image_index):
    """
    Coloring the RGB image with the predicted colors and saving the image to the results folder.
    :param image: the image that I want to color
    :param label_map: the label map for the image
    :param color_labels: list of predicted colors to the labels
    :param k: the number of the clusters
    :param image_index: the index of the image for saving the result
    """
    colored_image_rgb_multiply = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    r, g, b = cv2.split(colored_image_rgb_multiply)
    
    for i in range(0, k):
        color_label = np.argmax(color_labels[i])
        
        np.multiply(r, colors[color_label][0]/255, out=r, where=label_map==i, casting="unsafe")
        np.multiply(g, colors[color_label][1]/255, out=g, where=label_map==i, casting="unsafe")
        np.multiply(b, colors[color_label][2]/255, out=b, where=label_map==i, casting="unsafe")
    
    colored_image_rgb = cv2.merge([r, g, b])
    
    plt.figure(figsize=(6,6))
    plt.title("Image colorized by predicted colors \n in RGB colorspace")
    plt.axis("off")
    plt.imshow(colored_image_rgb)
    plt.savefig('results/' + format(image_index) + '/06_rgb_predicted_colorization.png')
    
def color_hsv(image, label_map, k, image_index):
    """
    Coloring the HSV image and saving it to the results folder.
    :param image: the image that I want to color
    :param label_map: the label map for the image
    :param k: the number of the clusters
    :param image_index: the index of the image for saving the result
    """
    colored_image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    hsv_image = cv2.cvtColor(colored_image_rgb, cv2.COLOR_RGB2HSV)
    
    colored_image_hsv = hsv_image
    
    height = image.shape[0]
    width = image.shape[1]
    
    for label in range(k):
        color_hsv = colorsys.rgb_to_hsv(colors[label][0]/255, colors[label][1]/255, colors[label][2]/255)
        for i in range(height):
            for j in range(width):
                if(label_map[i][j] == label):
                    colored_image_hsv[i][j] = [color_hsv[0]*180, color_hsv[1]*255, hsv_image[i][j][2]]
    
    plt.figure(figsize=(6,6))
    plt.title("Image colorized in HSV colorspace")
    plt.axis("off")
    colored_rgb = cv2.cvtColor(colored_image_hsv, cv2.COLOR_HSV2RGB)
    plt.imshow(colored_rgb)
    plt.savefig('results/' + format(image_index) + '/05_hsv_colorization.png')