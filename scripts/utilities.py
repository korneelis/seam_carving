from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image from the disk and make sure it is RGB. Throw exception when image cannot be loaded.
def load_rgb_image(file_path):
    try:
        # Open image file
        image = Image.open(file_path)
        # If image file is not in RGB mode then convert it
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return image
    
    except Exception as e:
        # Print out error when image could not be loaded
        print(f"Error loading the image: {str(e)}")
        return None

def combine_feature_maps(cam, depth_map, cnn_ratio):
    # Combine the cam with the depth map according to the cnn ratio
    combined_cam = cnn_ratio * cam + (1 - cnn_ratio) * depth_map
    return combined_cam

def display_heatmap_on_image(background_image, feature_map):
    # Prepare heatmap by applying color to class activation map
    heatmap = cv2.applyColorMap(np.uint8(255*feature_map), cv2.COLORMAP_JET)

    # Normalize and combine image and heatmap
    heatmap = np.float32(heatmap)/255
    background_image = np.float32(background_image/255)
    heatmap_on_image = heatmap + background_image
    
    # Normalize result and prepare for display
    heatmap_on_image = heatmap_on_image / np.max(heatmap_on_image)
    heatmap_on_image = np.uint8(255*heatmap_on_image)

    return heatmap_on_image, heatmap

def prepare_image(image_path):
    # Prepare background image
    background_image = cv2.imread(image_path)[..., ::-1]
    background_image = cv2.resize(background_image, (224, 224))

    return background_image

def visualize_image(image, title):
    # Visualize image with given title
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()
