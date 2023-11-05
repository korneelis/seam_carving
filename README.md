# Seam Carved Vectorization
**Author:** Korneel Somers
**Date:** November 2023
**Git Repository:** [Seam Carved Vectorization Project](https://gitlab.ewi.tudelft.nl/cgv/cs4365/student-repositories/2023-2024/cs436523ksomers.git)

This project performs content aware resizing for images. It uses ResNet50 and GradCam to generate the feature map of a certain class of objects in an image and Midas for a depth map. These maps are combined and can be edited by the user, to then be used as input for seam carving. Eventually, the image is upsized again by 'uncarving' it and interpolating the pixel colors.

## 1. Environment Setup

To get started you need to set up an environment and install the necessary dependencies. Run these commands in your terminal:

```bash
python -m venv seam_carving_project
source venv/bin/activate
pip install -r requirements.txt
```
To deactivate the environment, simply write `deactivate` in your terminal.

## 2. File Structure

The repository structure of the project is outlined below. The main script is located in the root folder and in that script various python scripts are called for the image processing tasks. These scripts are located in a correspondingly named folder. The data folder contains the input, some intermediate and ouput images of the project.

```
Project_Root/
├── data/
│   ├── carved_images/
│   ├── feature_maps/
│   ├── feature_maps_updated/
│   ├── images/
│   └── output_images/
├── scripts/
│   ├── color_interpolation.py
│   ├── depth_estimation.py
│   ├── gradcam.py
│   ├── object_recognition.py
│   ├── painting.py
│   ├── seam_carving.py
│   ├── utilities.py
│   └── vectorization.py
├── main.py
├── README.md
└── requirements.txt
```

## 3. Running the Project

Navigate to the project directory in your terminal and execute the main script:
```bash
python main.py
```
### User Interaction Guide

The main script interacts with the user to customize the processing flow and visualize intermediate steps. Read below how you should deal with the prompts and visualizations.

<details>
  <summary><b>Prompts</b></summary>

When you run the main script, it will prompt you for various inputs. If the input is invalid or left blank, the default will be used. As the script runs, you will be asked for the following:

- **Image Path:**
   - **Input:** A valid file path to an image.
   - **Default:** `'./data/images/jellyfish_tigershark.jpg'` 

- **Class ID for Grad-CAM:**
   - **Input:** The class ID for which you want to determine the class activation map ([click here](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/) for the list of Imagenet classes with their IDs).
   - **Default:** The class with the highest confidence score.

- **CNN Ratio:**
   - **Input:** A value between 0 and 1 that indicates the weight ratio between the two CNN's for the combined feature map. Higher values give more weight to object recognition, lower values to depth map.
   - **Default:** `0.5` (this value gives both CNNs the same weight)

- **Number of Seams:**
   - **Input:** The number of seams to remove.
   - **Default:** `10`
</details>

<details>
  <summary><b>Visualizations</b></summary>
    
While the main script is running, several visualizations will pop-up of intermediate step. Deal with them as follows:
    
- **Visualization Windows:** Close the window to proceed to the next step of the script.
- **Painting Interface:** After painting, press **ESC** to continue with the script.

</details>

### Output

After running the script, check the data folder to find generated outputs at various stages. Below you find the results of the example image.



## 4. Algorithmic Steps 

Below are the algorithmic steps of this project, along with links to the relevant scripts:

1. **Load RGB image from disk** - See [load_rgb_image](https://gitlab.ewi.tudelft.nl/cgv/cs4365/student-repositories/2023-2024/cs436523ksomers/-/blob/main/scripts/utilities.py?ref_type=heads#L7) in [utilities.py](scripts/utilities.py)
2. **Run pre-trained CNN for image detection** - [object_recognition.py](scripts/object_recognition.py)
3. **Extract feature map from a CNN using Grad-CAM** - [gradcam.py](scripts/gradcam.py)
4. **Modify feature map by painting** - [painting.py](scripts/painting.py)
5. **Use map for seam carving and remove pixel columns with low values** - [seam_carving.py](scripts/seam_carving.py)
6. **Vectorize remaining pixels by replacing them by triangle pairs** - [vectorization.py](scripts/vectorization.py)
7. **Move vectors back to original positions by "uncarving" the previously removed columns** - See [uncarve_vertices](https://gitlab.ewi.tudelft.nl/cgv/cs4365/student-repositories/2023-2024/cs436523ksomers/-/blob/main/scripts/seam_carving.py?ref_type=heads#L97) in [seam_carving.py](scripts/seam_carving.py)
8. **Smoothly interpolate the colors in the stretched vector graphics and rasterize it back to an image** - [color_interpolation.py](scripts/color_interpolation.py)
9. **Save and display result** - Displaying results is managed within the relevant scripts according to what is being displayed. The saving is handled through OpenCV's `cv2.imwrite()` function.
10. **Visualize the steps of the carving** - See [seam_carving_gui](https://gitlab.ewi.tudelft.nl/cgv/cs4365/student-repositories/2023-2024/cs436523ksomers/-/blob/main/scripts/seam_carving.py?ref_type=heads#L112) in [seam_carving.py](scripts/seam_carving.py)
11. **Add another CNN with features conditioned on different types of user input** - [depth_estimation.py](scripts/depth_estimation.py)
12. **Devise strategy for orientation of the triangle diagonals**



