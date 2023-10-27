# 1. Load RGB image from disk --------------------------------------------------

from PIL import Image

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


# 2. Run pre-trained CNN for image detection ----------------------------------

from resnet import ResNet50
import torch

image = load_rgb_image('../data/images/cat.jpg')

# Creat ResNet50 instance
resnet = ResNet50()
# Preprocess image and make a prediction
input_batch = resnet.prepare_image(image)
category = resnet.predict(input_batch)

# 3. Extract feature map from a CNN using Grad-CAM ----------------------------




# 4. Modify feature map by painting


# 5. Use map for seam carving and remove pixel columns with low values

# 6. Vectorize remaining pixels by replacing them by triangle pairs

# 7. Move vectors back to original positions by "uncarving" the previously removed columns

# 8. Smoothly interpolate the colors in the stretched vector graphics and rasterize it back to an image

# 9. Save and display result


# EXTRA FEATURES

# 10. Visualize the steps of the carving (step 5)
# 11. Add another CNN with features conditioned on a different type of user input (text, depth map, image, sketch,...)
# 12. Devise strategy for orientation of the triangle diagonals
