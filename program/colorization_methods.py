import cv2
import colorsys
import numpy as np
from matplotlib import pyplot as plt

colors = np.array([
        [0, 0, 255],     #blue
        [255, 255, 0],   #yellow
        [0, 255, 255],   #cyan
        [0, 255, 0],     #lime
        [255, 0, 0],     #red
        [255, 150, 0]    #orange
    ])

def color_rgb(image, label_map, k, image_index):
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
    
def color_hsv(image, label_map, k, image_index):
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