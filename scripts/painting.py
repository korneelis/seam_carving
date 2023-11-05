import cv2
import numpy as np

class HeatmapPainter:
    def __init__(self, initial_cam):
        # Get the initial class activation map
        self.cam = initial_cam

        # Set painting variable to 'False' and open the heatmap_on_image to start painting
        self.painting = False
        cv2.namedWindow('Heatmap on image')

        # Call update function when mouse
        cv2.setMouseCallback('Heatmap on image', self.update_heatmap)

    def update_heatmap(self, event, x, y, flags, param):
        # Determine the max value to cap the intensity    
        max_value = np.max(self.cam)
        # Choose radius for brush size and sigma for gaussian blur
        radius = 5
        sigma = 10.0

        # Start painting when left mouse button is pressed
        if event == cv2.EVENT_LBUTTONDOWN:
            self.painting = True
        # Stop painting when left mouse button is lifted
        elif event == cv2.EVENT_LBUTTONUP:
            self.painting = False
        # Update heatmap while painting
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.painting:
                # Create mask to save paint
                mask = np.zeros_like(self.cam)
                cv2.circle(mask, (x, y), radius, float(min(self.cam[y, x] * 0.5, max_value)), -1)
                # Apply gaussian blur
                mask = cv2.GaussianBlur(mask, (radius * 2 + 1, radius * 2 + 1), sigma)
                # Update the class activation map with the mask
                self.cam = np.minimum(self.cam + mask, max_value)

    def start(self, create_heatmap_on_image, background_image, image_path):
        # Start a loop to keep displaying and updating the image
        while True:
            updated_heatmap_on_image, updated_heatmap = create_heatmap_on_image(background_image, self.cam)
            cv2.imshow('Heatmap on image', updated_heatmap_on_image)
            # If ESC is pressed, the display closes
            if cv2.waitKey(1) & 0xFF == 27:
                # Image and cam of final updated feature map are saved
                cv2.imwrite("./data/feature_maps_updated/heatmap_" + image_path.split('/')[-1], updated_heatmap_on_image)
                return self.cam
