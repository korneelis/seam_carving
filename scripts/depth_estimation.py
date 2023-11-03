# Sources: 
# https://pytorch.org/hub/intelisl_midas_v2/ 
# https://github.com/isl-org/MiDaS
# https://www.kaggle.com/code/rajeevsharma993/depth-detection-using-intel-s-midas

import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import numpy as np

class MidasEstimation:
    def __init__(self):
        # Load the pre-trained model from PyTorch Hub and set to evaluation mode
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        self.model.eval()

        # Define the transforms to get correct model input
        self.transform = Compose([
            Resize((224, 224)),  
            ToTensor(),  
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def predict_depth(self, raw_image):
        # Preprocess the raw image
        input_batch = self.transform(raw_image).unsqueeze(0)

        # Predict depth
        with torch.no_grad():
            prediction = self.model(input_batch)
        depth_map = prediction.squeeze().cpu().detach().numpy()

        # Get the minimum and maximum values from the depth map
        min_val = np.min(depth_map)
        max_val = np.max(depth_map)

        # Normalize the depth map
        normalized_depth_map = (depth_map - min_val) / (max_val - min_val)

        return normalized_depth_map