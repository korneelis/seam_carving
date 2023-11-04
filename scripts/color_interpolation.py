import numpy as np
from PIL import Image, ImageDraw

class ColorInterpolator:
    def __init__(self, image):
        # Initialize two representations of the image
        self.image_array = image
        self.image = Image.fromarray(image)
    
    def interpolate_colors(self, vertices, triangles):
        # Interpolate colors for each triangle
        # Start by getting the color data for each vertex
        color_data = self.image_array[vertices[:, 1].astype(np.int32), vertices[:, 0].astype(np.int32)]
        # Placeholders for the interpolated colors
        raster_image = Image.new('RGB', self.image.size)
        # Create a draw object
        draw = ImageDraw.Draw(raster_image)

        # Loop over all triangles and rasterize them
        for triangle in triangles:
            self.rasterize_triangle(vertices, color_data, draw, triangle)

        return raster_image

    def rasterize_triangle(self, vertices, color_data, draw, triangle_indices):
        # Get the vertices of the triangle
        triangle = [int(index) for index in triangle_indices]
        v0, v1, v2 = [vertices[i] for i in triangle]
        # Get the bounding box of the triangle
        min_x, max_x, min_y, max_y = self.bounding_box(v0, v1, v2)

        # Loop over all pixels in the bounding box 
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                # Get the barycentric weights for the current pixel
                w0, w1, w2 = self.barycentric_weights(x, y, v0, v1, v2)
                # Check if the pixel is inside the triangle
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    # If it is then interpolate the color for the current pixel
                    color = w0 * color_data[triangle[0]] + w1 * color_data[triangle[1]] + w2 * color_data[triangle[2]]
                    draw.point((x, y), fill=tuple(color.astype(np.int32)))

    def bounding_box(self, v0, v1, v2):
        # Get the bounding box of the triangle
        min_x = max(int(np.min([v0[0], v1[0], v2[0]])), 0)
        max_x = min(int(np.max([v0[0], v1[0], v2[0]])), self.image.width - 1)
        min_y = max(int(np.min([v0[1], v1[1], v2[1]])), 0)
        max_y = min(int(np.max([v0[1], v1[1], v2[1]])), self.image.height - 1)
        return min_x, max_x, min_y, max_y

    def barycentric_weights(self, x, y, v0, v1, v2):
        # Calculate the barycentric weights for the a given pixel
        det = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
        w0 = ((v1[1] - v2[1]) * (x - v2[0]) + (v2[0] - v1[0]) * (y - v2[1])) / det
        w1 = ((v2[1] - v0[1]) * (x - v2[0]) + (v0[0] - v2[0]) * (y - v2[1])) / det
        w2 = 1 - w0 - w1
        return w0, w1, w2