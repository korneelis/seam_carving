
from scripts.utilities import load_rgb_image
from scripts.object_recognition import ResNet50
from scripts.depth_estimation import MidasEstimation
from scripts.gradcam import GradCam
from scripts.painting import HeatmapPainter
from scripts.seam_carving import SeamCarver, seam_carving_gui
from scripts.vectorization import vectorization, visualize_triangles_carved, visualize_triangles_uncarved
from scripts.color_interpolation import ColorInterpolator
from scripts.utilities import load_rgb_image, prepare_image, display_heatmap_on_image, combine_feature_maps, visualize_image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

# -------------------- Load RGB image from disk -----------------------------

# Load RGB image from disk
image_path_input = input("Enter the image path or leave blank for default image: ")
if image_path_input and os.path.isfile(image_path_input):
    image_path = image_path_input
else:
    print("Invalid or no input. Using default image")
    image_path = './data/images/jellyfish_tigershark.jpg'
    
raw_image = load_rgb_image(image_path)
visualize_image(raw_image, 'Raw Image')

# ------------- Run pre-trained CNN for image detection ---------------------

# Creat ResNet50 instance
resnet_model = ResNet50()
# Preprocess image and make a prediction
input_image = resnet_model.prepare_image(raw_image)
prediction = resnet_model.predict(input_image)

# ------------ Extract Class Activation Map ---------------------------------

# User input for class_id
class_id_input = input("Enter a specific class_id or leave blank to use class with highest score: ")
class_id = int(class_id_input) if class_id_input.isdigit() else None

# User input for cnn_ratio
cnn_ratio_input = input("Enter a CNN ratio (higher values give more weight to object recognition, lower values to depth map): ")
try:
    cnn_ratio = float(cnn_ratio_input)
except ValueError:
    print("Invalid input for cnn ratio. Using default value 0.5")
    cnn_ratio = 0.5

# Create GradCam instance and compute class activation map
grad_cam = GradCam(resnet_model.model, "layer4.2")
cam = grad_cam(input_image, index=class_id)

# ----------------- Predict depth map with pre-trained CNN ---------------------

# Creat MidasEstimation instance
depth_estimator = MidasEstimation()
# Predict depth
depth_map = depth_estimator.predict_depth(raw_image)

# ----------------------- Generate Combined Heatmap -----------------------------

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

# ---------------------- Paint on Heatmap to Modify -----------------------------

# Create HeatmapPainter instance
painter = HeatmapPainter(feature_maps)
# Start painting and save result
updated_map = painter.start(display_heatmap_on_image, background_image, image_path)

# ---------------------- Perform Seam Carving -----------------------------------

# User input for number of seams
num_seams_input = input("Enter the number of seams: ")
try:
    num_seams = int(num_seams_input)
except ValueError:
    print("Invalid input for the number of seams. Using default value 10")
    num_seams = 10

# Create SeamCarver instance
carver = SeamCarver(background_image, updated_map)
# Generate carved image and visualize
carved_image = carver.seam_carve(num_seams)
cv2.imwrite("./data/carved_images/carved_" + image_path.split('/')[-1], carved_image)

# Visualize seam carving
seam_carving_gui(background_image, updated_map, num_seams)

# ---------------------- Vectorize Carved Image --------------------------------

# Vectorize the carved_image
vertices, triangles = vectorization(carved_image)
# Visualize triangle pairs
visualize_triangles_carved(carved_image, vertices, triangles)

# ---------------------- Uncarve Vectors ---------------------------------------

# Uncarving of vertices
uncarving_visualisation_input = prepare_image(image_path)
vertices_updated = carver.uncarve_vertices(vertices)
# Visualize uncarved vertices
visualize_triangles_uncarved(uncarving_visualisation_input, vertices_updated, triangles)

# ---------------------- Interpolate Colors ------------------------------------

# Create ColorInterpolator instance
color_interpolator = ColorInterpolator(background_image)
# Interpolate colors for each triangle
rasterized_image = color_interpolator.interpolate_colors(vertices_updated, triangles)
rasterized_image_array = np.array(rasterized_image)

cv2.imwrite("./data/output_images/rasterized_" + image_path.split('/')[-1], rasterized_image_array)

# Display the final rasterized image
visualize_image(rasterized_image, 'Rasterized Image')







