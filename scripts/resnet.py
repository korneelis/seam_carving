import torch
import torchvision.models as models
import torchvision.transforms as transforms

# Sources: https://pytorch.org/vision/stable/models.html#id3


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

    def prepare_image(self, image):
        # Preprocess image
        input_tensor = self.preprocess(image)
        return input_tensor.unsqueeze(0)

    def predict(self, input_batch):
        # Apply model to input
        
        # with torch.no_grad():
        #     prediction = self.model(input_batch).squeeze(0).softmax(0)
        prediction = self.model(input_batch).squeeze(0).softmax(0)

        # Determine the classification categories
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()

        # Map class_id to category name
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        category_name = weights.meta["categories"][class_id]
        print(f"{category_name}: {100 * score:.1f}%")   
        print(class_id)     

        return category_name 