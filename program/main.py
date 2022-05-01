from matplotlib import pyplot as plt

#my modules
import manage_directories as mdir
import kmeans_segmentations as ks
import commonmethods.image_modification as im
import colorization_methods as cm
import cnn_model as cnn
import superpixel_segmentations as ss

if __name__ == '__main__':
    
    number_of_images = 6
    k_texture_values = [8, 5, 6, 5, 6, 5]
    
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
        print('   - image rgb kmeans colorization done!')
        
        cm.color_hsv(resized_image, label_map, k_texture_values[i], i)
        print('   - image hsv kmeans colorization done!')
        
        cm.color_rgb_by_predicted_colors(resized_image, label_map, color_labels, k_texture_values[i], i, 'kmeans')
        print('   - image rgb kmeans colorization with predicted colors done!')
        
        cm.color_hsv_by_predicted_colors(resized_image, label_map, color_labels, k_texture_values[i], i, 'kmeans')
        print('   - image hsv kemans colorization with predicted colors done!')
        
        segments_slic, segments_watershed = ss.get_segments(resized_image, i)
        print('   - image superpixel segmentation done!')
        
        slic_color_labels, segments_slic_quantity = ss.get_segments_color(resized_image, segments_slic, cnn_model)
        print('   - image slic color prediction done!')
        
        watershed_color_labels, segments_watershed_quantity = ss.get_segments_color(resized_image, segments_watershed, cnn_model)
        print('   - image watershed color prediction done!')
        
        cm.color_rgb_by_predicted_colors(resized_image, segments_slic, slic_color_labels, segments_slic_quantity, i, 'superpixel_slic')
        print('   - image rgb slic colorization with predicted colors done!')
        
        cm.color_hsv_by_predicted_colors(resized_image, segments_slic, slic_color_labels, segments_slic_quantity, i, 'superpixel_slic')
        print('   - image hsv slic colorization with predicted colors done!')
        
        cm.color_rgb_by_predicted_colors(resized_image, segments_watershed, watershed_color_labels, segments_watershed_quantity, i, 'superpixel_watershed')
        print('   - image rgb watershed colorization with predicted colors done!')
        
        cm.color_hsv_by_predicted_colors(resized_image, segments_watershed, watershed_color_labels, segments_watershed_quantity, i, 'superpixel_watershed')
        print('   - image hsv watershed colorization with predicted colors done!')
        
        plt.close('all') #closing all figures