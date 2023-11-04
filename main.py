
# -------------------- 1. Load RGB image from disk -----------------------------------------------------------------------------------------------------------
from scripts.utilities import load_rgb_image

# Load RGB image from disk
image_path = './data/images/jellyfish_tigershark.jpg'
raw_image = load_rgb_image(image_path)

# -------------------- 2. Run pre-trained CNN for image detection -------------------------------------------------------------------------------------------

from scripts.object_recognition import ResNet50
from scripts.depth_estimation import MidasEstimation


# Creat ResNet50 instance
resnet_model = ResNet50()

# Preprocess image and make a prediction
input_image = resnet_model.prepare_image(raw_image)
prediction = resnet_model.predict(input_image)

# Creat MidasEstimation instance
depth_estimator = MidasEstimation()

# Predict depth
depth_map = depth_estimator.predict_depth(raw_image)

# --------------------- 3. Extract feature map from a CNN using Grad-CAM -------------------------------------------------------------------------------------

import cv2
from scripts.gradcam import GradCam
from scripts.utilities import load_rgb_image, prepare_image, display_heatmap_on_image, combine_feature_maps, visualize_image
import matplotlib.pyplot as plt

# Choose specific class_id or leave 'None' to use class with highest score from ResNet50
class_id = 107
# Choose cnn ratio. A higher value will give more weight to resnet, a lower value to the depth estimation
cnn_ratio = 0.5

# Create GradCam instance and compute class activation map
grad_cam = GradCam(resnet_model.model, "layer4.2")
cam = grad_cam(input_image, index=class_id)

# Combine the class activation map and the depth map
feature_maps = combine_feature_maps(cam, depth_map, cnn_ratio)
# Prepare background image
background_image = prepare_image(image_path)

# Generate heatmaps
_, heatmap_depth = display_heatmap_on_image(background_image, depth_map)
_, heatmap_cam = display_heatmap_on_image(background_image, cam)
heatmap_on_image, heatmap = display_heatmap_on_image(background_image, feature_maps)
# Save combined heatmaps on image
cv2.imwrite("./data/feature_maps/heatmap_" + image_path.split('/')[-1], heatmap_on_image)
# Display all heatmaps
plt.subplot(131), plt.imshow(heatmap_depth), plt.title('Depth Heatmap')
plt.subplot(132), plt.imshow(heatmap_cam), plt.title('Cam Heatmap')
plt.subplot(133), plt.imshow(heatmap), plt.title('Combined Heatmap')
plt.show()

# --------------------- 4. Modify feature map by painting ---------------------------------------------------------------------------------------------------

from scripts.painting import HeatmapPainter

# Create HeatmapPainter instance
painter = HeatmapPainter(feature_maps)
# Start painting and save result
updated_map = painter.start(display_heatmap_on_image, background_image, image_path)

# ---------------------- 5. Use map for seam carving and remove pixel columns with low values -------------------------------------------------------------------

from scripts.seam_carving import SeamCarver, visualize_uncarving

# Choose number of seams that should be removed
num_seams = 50

# Create SeamCarver instance
carver = SeamCarver(background_image, updated_map)
# carved_image = carver.crop_c(scale_c=0.5)

# Generate carved image and visualize
carved_image = carver.seam_carve(num_seams)
visualize_image(carved_image, 'Carved Image')

# ---------------------- 6. Vectorize remaining pixels by replacing them by triangle pairs -----------------------------------------------------------------------

from scripts.vectorization import vectorization, visualize_triangles

# Vectorize the carved_image
vertices, triangles = vectorization(carved_image)
# Visualize triangle pairs
visualize_triangles(vertices, triangles)

# ---------------------- 7. Move vectors back to original positions by "uncarving" the previously removed columns ------------------------------------------------

# Uncarving of vertices
uncarving_visualisation_input = prepare_image(image_path)
vertices_updated = carver.uncarve_vertices(vertices)

# Visualizations
visualize_uncarving(uncarving_visualisation_input, vertices_updated)
visualize_triangles(vertices_updated, triangles)


# ---------------------- 8. Smoothly interpolate the colors in the stretched vector graphics and rasterize it back to an image -----------------------------------
from scripts.color_interpolation import ColorInterpolator

# Create ColorInterpolator instance
color_interpolator = ColorInterpolator(background_image)
# Interpolate colors for each triangle
rasterized_image = color_interpolator.interpolate_colors(vertices_updated, triangles)

# Display the final rasterized image
visualize_image(rasterized_image, 'Rasterized Image')

# ---------------------- 9. Save and display result -------------------------------------------------------------------------------------------------------------


# EXTRA FEATURES

# --------------------- 10. Visualize the steps of the carving (step 5) -----------------------------------------------------------------------------------------

from scripts.seam_carving import seam_carving_gui

seam_carving_gui(background_image, updated_map, num_seams)

# --------------------- 11. Add another CNN with features conditioned on a different type of user input (text, depth map, image, sketch,...) ---------------------------

# See CNN section above

# --------------------12. Devise strategy for orientation of the triangle diagonals -----------------------------------------------------------------------------------------


