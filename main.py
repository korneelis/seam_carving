# 1. Load RGB image from disk -----------------------------------------------------------------------------------------------------------

from scripts.utilities import load_rgb_image

# Load RGB image from disk
image_path = './data/images/cat.jpg'
raw_image = load_rgb_image(image_path)
# raw_image_height, raw_image_width = raw_image.size

# 2. Run pre-trained CNN for image detection -------------------------------------------------------------------------------------------

from scripts.resnet import ResNet50

# Creat ResNet50 instance
resnet_model = ResNet50()

# Preprocess image and make a prediction
input_image = resnet_model.prepare_image(raw_image)
prediction = resnet_model.predict(input_image)

# 3. Extract feature map from a CNN using Grad-CAM -------------------------------------------------------------------------------------

import cv2
import numpy as np
from scripts.gradcam import GradCam, prepare_image, create_heatmap_on_image

# Choose specific class_id or leave 'None' to use class with highest score
class_id = None

# Create GradCam instance and compute cam
grad_cam = GradCam(resnet_model.model, "layer4.2")
cam = grad_cam(input_image, index=class_id)
# Prepare background image
background_image = prepare_image(image_path)

# Generate and save heatmap image
heatmap_on_image, heatmap = create_heatmap_on_image(background_image, cam)
cv2.imwrite("./data/feature_maps/heatmap_" + image_path.split('/')[-1], heatmap_on_image)

# 4. Modify feature map by painting ---------------------------------------------------------------------------------------------------

from scripts.painting import HeatmapPainter

# Create HeatmapPainter instance
painter = HeatmapPainter(cam)
# Start painting and save result
updated_cam = painter.start(create_heatmap_on_image, background_image, image_path)

# 5. Use map for seam carving and remove pixel columns with low values -------------------------------------------------------------------

from scripts.seam_carving import SeamCarver

# Choose number of seams that should be removed
num_seams = 10

# Create SeamCarver instance
carver = SeamCarver(background_image, updated_cam)
# Generate carved image and visualize
carved_image = carver.seam_carve(num_seams=num_seams)
cv2.imshow('Carved Image', carved_image)
cv2.waitKey(0)

# 6. Vectorize remaining pixels by replacing them by triangle pairs -----------------------------------------------------------------------

from scripts.vectorization import vectorization, visualize_triangles

# Vectorize the carved_image
vertices, triangles = vectorization(carved_image)
# Visualize triangle pairs
vector_visualisation = visualize_triangles(vertices, triangles)

# 7. Move vectors back to original positions by "uncarving" the previously removed columns ------------------------------------------------



# 8. Smoothly interpolate the colors in the stretched vector graphics and rasterize it back to an image -----------------------------------

# 9. Save and display result -------------------------------------------------------------------------------------------------------------


# EXTRA FEATURES

# 10. Visualize the steps of the carving (step 5)



# 11. Add another CNN with features conditioned on a different type of user input (text, depth map, image, sketch,...)

# 12. Devise strategy for orientation of the triangle diagonals
