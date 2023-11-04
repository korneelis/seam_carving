# source: assignment 3

import numpy as np

def vectorization(image):
    # Image dimensions
    height, width, _ = image.shape

    # Build vertex buffer
    num_vertices = width*height
    vertices = np.zeros((num_vertices, 2))
    
    # Assign the vertex coordinates to each pixelcenter
    for y in range(height):
        for x in range(width):
            vertices[y * width + x] = [x+0.5, y+0.5]
    
    # Build index buffer
    num_pixels = (width-1) * (height-1)
    num_triangles = num_pixels * 2
    triangles = np.zeros((num_triangles, 3))
    
    # Assign triangles 
    for y in range(height-1):
        for x in range(width-1):
            index = y * (width-1) + x
            triangles[index * 2] = [y * width + x, y * width + x + 1, (y + 1) * width + x + 1]
            triangles[index * 2 + 1] = [y * width + x, (y + 1) * width + x + 1, (y + 1) * width + x]
    
    return vertices, triangles

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def visualize_triangles(vertices, triangles):
    #Retrieve the coordinates of the vertices to create Triangulation object
    x = vertices[:, 0]
    y = vertices[:, 1]
    tri = mtri.Triangulation(x, y, triangles=triangles)
    
    #Visualize triangulation
    plt.figure()
    plt.triplot(tri)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()