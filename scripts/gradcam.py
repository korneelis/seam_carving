import cv2
import numpy as np
import torch

# Source: https://github.com/kamata1729/visualize-pytorch/blob/master/src/gradCAM.py
    
class GradCam():
    def __init__(self, model, target_layer):
        # Set model to evaluation mode
        self.model = model.eval()
        
        # Go through the layer of the CNN model
        for module in self.model.named_modules():
            if module[0] == target_layer:
                # save feature maps and gradients of target layer
                module[1].register_forward_hook(self.save_feature_map)
                module[1].register_full_backward_hook(self.save_grad)
    
    # Function to save the output feature map during forward pass 
    def save_feature_map(self, module, input, output):
        self.feature_map = output.detach()

    # Function to save the gradient of the output feature map during backward pass    
    def save_grad(self, module, grad_in, grad_out):
        self.grad = grad_out[0].detach()
        
    def __call__(self, x, index=None):
        # Make clone of input data to preserve it
        x = x.clone()

        # Forward pass to get output logits    
        output = self.model(x)

        # If no class_id is specified, use class with highest score
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        # Create vector to store target class scores
        target_class_vector = np.zeros((1, output.size()[-1]))
        target_class_vector[0][index] = 1
        target_class_vector = torch.from_numpy(target_class_vector)
        target_class_vector.requires_grad_()
        target_class_vector = torch.sum(target_class_vector * output)

        # Remove old gradients    
        self.model.zero_grad()
        
        # Backpropagate and store gradients
        target_class_vector.backward()
        
        # Make feature map an array and define its weights
        self.feature_map = self.feature_map.cpu().numpy()[0]
        self.weights = np.mean(self.grad.cpu().numpy(), axis = (2, 3))[0, :]
        
        # Calculate class activation map (cam)
        cam = np.sum(self.feature_map * self.weights[:, None, None], axis=0)
        # Apply ReLU and resize to original image size
        cam = np.maximum(cam, 0)    
        cam = cv2.resize(cam, (x.size()[-1], x.size()[-2]))

        # Find minimum and maximum cam values
        cam_min = np.min(cam)
        cam_max = np.max(cam)
        # Normalize cam
        cam_normalized = (cam - cam_min) / (cam_max - cam_min)

        return cam_normalized


# Other sources:
# https://github.com/ismailuddin/gradcam-tensorflow-2/blob/master/notebooks/GradCam.ipynb
# https://medium.com/the-owl/gradcam-in-pytorch-7b700caa79e5
# https://github.com/Caoliangjie/pytorch-gradcam-resnet50/blob/master/grad-cam.py
# https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
# https://github.com/jacobgil/pytorch-grad-cam
