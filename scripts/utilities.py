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
