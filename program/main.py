from matplotlib import pyplot as plt
#my modules
import manage_directories as mdir
import kmeans_segmentations as ks
import commonmethods.image_modification as im
import colorization_methods as cm
import cnn_model as cnn

if __name__ == '__main__':
    
    number_of_images = 6
    k_texture_values = [8, 5, 6, 3, 6, 5]
    
    cnn_model = cnn.get_model()
    print('CNN model configuration done!')
    
    for i in range(number_of_images):
        print(format(i) + '.jpg: ')
        
        mdir.create_results_directory
        mdir.manage_directories(format(i))
        
        image = im.load_image_grayscale(format(i))
        
        resized_image = im.resize_image(image, 1024)
        
        ks.kmeans_segmentation(resized_image, i)
        print('   - image kmeans segmentation done!')
        
        label_map, color_labels = ks.kmeans_texture_segmentation(resized_image, k_texture_values[i], 15, i, cnn_model)
        print('   - image texture kmeans segmentation done!')
        
        cm.color_rgb(resized_image, label_map, k_texture_values[i], i)
        print('   - image rgb colorization done!')
        
        cm.color_hsv(resized_image, label_map, k_texture_values[i], i)
        print('   - image hsv colorization done!')
        
        cm.color_rgb_by_predicted_colors(resized_image, label_map, color_labels, k_texture_values[i], i)
        print('   - image rgb colorization with predicted colors done!')
        
        cm.color_hsv_by_predicted_colors(resized_image, label_map, color_labels, k_texture_values[i], i)
        print('   - image hsv colorization with predicted colors done!')
        
        plt.close('all') #closing all figures