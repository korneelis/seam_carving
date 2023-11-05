import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import cv2

# Source: assignment 3

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


def visualize_triangles_carved(carved_image, vertices, triangles):
    # Set up the figure and subplots
    plt.figure(figsize=(10, 5)) 

    carved_image = cv2.cvtColor(carved_image, cv2.COLOR_BGR2RGB)

    # First subplot: Carved Image
    plt.subplot(1, 3, 1)
    plt.imshow(carved_image)
    plt.title('Carved Image')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')

    # Second subplot: Carved Image with Triangles
    # Create triangulation object from vertices
    x = vertices[:, 0]
    y = vertices[:, 1]
    tri = mtri.Triangulation(x, y, triangles=triangles)
    plt.subplot(1, 3, 2)
    plt.imshow(carved_image)
    plt.triplot(tri, 'g-')
    plt.title('Carved Image with Triangles')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')

    # Third subplot: Triangles Only
    plt.subplot(1, 3, 3)
    plt.triplot(tri, 'g-')
    plt.title('Triangles Only')
    plt.gca().set_aspect('equal', adjustable='box')

    # Show the complete figure
    plt.tight_layout()
    plt.show()

def visualize_triangles_uncarved(image, updated_vertices, triangles):
    plt.subplots(1, 2, figsize=(10, 5)) 

    # First subplot - Carved Parts with Updated Vertices
    for x, y in updated_vertices:
        image[int(y), int(x)] = [50, 50, 50]

    # Show final visualization
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Carved Parts')
    plt.axis('off')

    # Second subplot - Triangulation
    x = updated_vertices[:, 0]
    y = updated_vertices[:, 1]
    tri = mtri.Triangulation(x, y, triangles=triangles)
    
    #Visualize triangulation
    plt.subplot(1, 2, 2)
    plt.triplot(tri)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Triangulation')

    plt.tight_layout()
    plt.show()