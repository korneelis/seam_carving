# Source https://karthikkaranth.me/blog/implementing-seam-carving-with-python/

import numpy as np
import cv2

class SeamCarver:
    def __init__(self, image, cam):
        # Initialize image and class activation map
        self.image = image
        self.cam = cam


    def compute_min_seam(self):
        # Use cam dimensions to initialize placeholders
        rows, cols = self.cam.shape
        min_cam = self.cam.copy()
        backtrack = np.zeros((rows, cols), dtype=np.int32)
        # Create array that will store minimum seam
        seam = np.zeros(rows, dtype=np.int32)
        
        for i in range(1, rows):
            for j in range(cols):
                if j == 0:
                    idx = np.argmin(min_cam[i - 1, j:j + 2])
                    min_energy = min_cam[i - 1, idx + j]
                else:
                    idx = np.argmin(min_cam[i - 1, j - 1:j + 2])
                    min_energy = min_cam[i - 1, idx + j - 1]

                min_cam[i, j] += min_energy
                backtrack[i, j] = idx + j - 1 if j != 0 else idx + j

        seam[-1] = np.argmin(min_cam[-1])
        for i in range(rows - 2, -1, -1):
            seam[i] = backtrack[i + 1, seam[i + 1]]

        return seam

    def remove_min_seam(self, seam):
        # Get image dimensions and create placeholder for output image (with one less column)
        height, width, _ = self.image.shape
        output_image = np.zeros((height, width-1, 3))
        
        # Loop through the rows and remove pixel that is specified by seam
        for i in range(height):
            remove_pixel = seam[i]
            output_image[i, :, :] = np.delete(self.image[i, :, :], remove_pixel, axis=0)
        
        # Convert output image
        output_image = output_image.astype(np.uint8)
        
        return output_image
    
    def mark_seam(self, seam):
        # mark the seam for visualization
        for row, col in enumerate(seam):
            self.image[row, col, :] = [0, 0, 255]

    def seam_carve(self, num_seams, mark=False):
        # Find minimum seam and remove it (num_seams times)
        for i in range(num_seams):
            min_seam = self.compute_min_seam()
            self.image = self.remove_min_seam(min_seam)
            # Only when requested, mark the seam
            if mark and i == num_seams-1:
                self.mark_seam(min_seam)
        # Convert image from BGR to RGB
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        return self.image
    

