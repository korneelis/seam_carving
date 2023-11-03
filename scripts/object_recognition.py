import torch
import torchvision.models as models

# Sources: https://pytorch.org/vision/stable/models.html#id

class ResNet50:
    def __init__(self):
        # Load pre-trained ResNet-50 model
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        self.model = models.resnet50(weights=weights)
        self.model.eval()

        # Image preprocessing pipeline (standardize input to meet model requirements/improve performance)
        self.preprocess = weights.transforms()

        # self.preprocess = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

    def prepare_image(self, raw_image):
        # Preprocess image
        input_tensor = self.preprocess(raw_image)
        return input_tensor.unsqueeze(0)

    def predict(self, input_batch):
        # Apply model to input
        with torch.no_grad():
            prediction = self.model(input_batch).squeeze(0).softmax(0)

        # Determine the classification categories
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()

        # Map class_id to category name
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        category_name = weights.meta["categories"][class_id]
        print(f"{category_name} ({class_id}): {100 * score:.1f}%")   

        return category_name 
    

# source https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50
# source https://keras.io/api/applications/#usage-examples-for-image-classification-models
