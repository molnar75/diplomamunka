import matplotlib.pyplot as plt
import numpy as np
import cv2

#modules to get segments
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries

def get_segments(image, image_index):
    """
    Method for segmenting the image with 4 Superpixel algorithms: Felzenszwalbs, SLIC, Quickshift and Watershed.
    :param image: the image that I want to sgement with superpixel methods
    :param image_index: the index of the image for saving the result
    :return: the segments from SLIC and Watershed methods
    """
    segments_fz = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
    
    segments_slic = slic(image, n_segments=250, compactness=.1, sigma=5, start_label=0)
    
    grayscaled_color = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    segments_quick = quickshift(grayscaled_color, kernel_size=3, max_dist=6, ratio=0.5)
    
    gradient = sobel(image)
    segments_watershed = watershed(gradient, markers=250, compactness=0.001)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    
    ax[0, 0].imshow(mark_boundaries(image, segments_fz, color=(1, 0, 0)))
    ax[0, 0].set_title("Felzenszwalbs's method")
    ax[0, 1].imshow(mark_boundaries(image, segments_slic, color=(1, 0, 0)))
    ax[0, 1].set_title('SLIC')
    ax[1, 0].imshow(mark_boundaries(image, segments_quick, color=(1, 0, 0)))
    ax[1, 0].set_title('Quickshift')
    ax[1, 1].imshow(mark_boundaries(image, segments_watershed, color=(1, 0, 0)))
    ax[1, 1].set_title('Compact watershed')
    
    for a in ax.ravel():
        a.set_axis_off()
    
    plt.tight_layout()
    plt.savefig('results/' + format(image_index) + '/08_superpixel_segmentations.png')
    
    return segments_slic, segments_watershed
    
def get_segments_color(image, segments, cnn_model):
    """
    Method for predicting the colors to each segments with CNN modell.
    :param image: the image that I want to determine the colors
    :param segments: the segments of the image
    :param cnn_model: the CNN model for the prediction
    :return: the color label map and the quantity of the segments for colorization purposes
    """
    
    segments_quantity = len(np.unique(segments))
    
    color_labels = np.zeros((segments_quantity, 13))
    
    px = 15
    
    height = image.shape[0]
    width = image.shape[1]
    
    for (i, segVal) in enumerate(np.unique(segments)):
        #getting the segment
        mask = np.zeros(image.shape[:2], dtype = "uint8")
        mask[segments == segVal] = 255
        
        
        #getting the indexes of the pixels from the segment
        indexes = np.where(mask == 255)
        x = indexes[0][0]
        y = indexes[1][0]
        
        if(x <= (height-px) and y <= (width-px)):
            test_window = image[x:x+px, y:y+px]
            
            test_window = test_window/255
            
            test_window = test_window.reshape((15, 15))
            test_window = np.expand_dims(test_window, axis = 0)
            
            color_label = np.argmax(cnn_model.predict(test_window))
                    
            color_labels[i][color_label] = color_labels[i][color_label] + 1
        
    return color_labels, segments_quantity