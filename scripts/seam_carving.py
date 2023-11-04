# Source https://karthikkaranth.me/blog/implementing-seam-carving-with-python/

import numpy as np
import cv2
import matplotlib.pyplot as plt

class SeamCarver:
    def __init__(self, image, cam):
        # Initialize image and class activation map
        self.image = image
        self.cam = cam
        self.removed_seams = []

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
                elif j == cols - 1:
                    idx = np.argmin(min_cam[i - 1, j - 1:j + 1])
                    min_energy = min_cam[i - 1, idx + j - 1]
                    backtrack[i, j] = idx + j - 1
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
        output_cam = np.zeros((height, width - 1))

        # Loop through the rows and remove pixel that is specified by seam
        for i in range(height):
            remove_pixel = seam[i]
            output_image[i, :, :] = np.delete(self.image[i, :, :], remove_pixel, axis=0)
            output_cam[i, :] = np.delete(self.cam[i, :], remove_pixel, axis=0)
        
        #Keep track of every seam that is removed to use for uncarving
        self.removed_seams.append(seam.copy())
        
        # Update the cam
        self.cam = output_cam
        self.image = output_image

        
        # Convert output image
        output_image = output_image.astype(np.uint8)
        
        return output_image
    
    def mark_seam(self, seam):
        # mark the seam for visualization
        for row, col in enumerate(seam):
            self.image[row, col-1, :] = [0, 0, 255]

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
    
    def uncarve_vertices(self, vertices):
        adjusted_vertices = vertices.copy()

        for seam in reversed(self.removed_seams):
            for i, (x, y) in enumerate(adjusted_vertices):
                # Convert to int, in case they are not
                x = int(x)
                y = int(y)

                # Check and adjust the vertices based on the seam
                if x >= seam[y]:
                    adjusted_vertices[i] = (x + 1, y)

        return adjusted_vertices
    
        
from tkinter import Tk, Scale
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def seam_carving_gui(image, cam, num_seams):
    def update_image(val):
        num_seams = int(val)
        seam_carver = SeamCarver(image.copy(), cam.copy())
        updated_img = seam_carver.seam_carve(num_seams, mark=True)
        updated_img = cv2.cvtColor(updated_img, cv2.COLOR_BGR2RGB)
        ax.clear()
        ax.imshow(updated_img)
        canvas.draw()

    root = Tk()
    root.title("Seam Carving GUI")

    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)

    canvas = FigureCanvasTkAgg(fig, master=root) 
    canvas.draw()
    canvas.get_tk_widget().pack()

    slider = Scale(root, from_=0, to=num_seams, orient="horizontal", label="Number of Seams", command=update_image)
    slider.pack()

    root.mainloop()

def visualize_uncarving(image, updated_vertices):
    # Create a final image with adjusted vertices
    for x, y in updated_vertices:
        image[int(y), int(x)] = [50, 50, 50]

    # Show final visualization
    plt.imshow(image)
    plt.title('Carved Parts')
    plt.show()