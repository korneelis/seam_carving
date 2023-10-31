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
        #prediction = self.model(input_batch).squeeze(0).softmax(0)

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

# import tensorflow as tf
# from keras.applications import ResNet50
# from keras.applications.resnet50 import preprocess_input, decode_predictions
# from keras.preprocessing.image import load_img, img_to_array
# from keras.layers import Input

# class ResNet50_pretrained:
#     def __init__(self):
#         # Load pre-trained ResNet-50 model
#         self.model = ResNet50(weights='imagenet')
#         self.model.trainable = False
#         self.inputs = Input(shape=(224, 224, 3))


#     def prepare_image(self, raw_image):
#         # Load and preprocess the image to fit the ResNet50 model
#         input_image = raw_image.resize((224, 224))
#         input_image = img_to_array(input_image)
#         input_image = preprocess_input(input_image)
#         input_image = tf.expand_dims(input_image, axis=0)
#         return input_image

#     def get_last_conv_layer(self):
#         # Access the last convolutional layer in the ResNet50 model
#         last_conv_layer = self.model.get_layer("conv5_block3_out")
#         return last_conv_layer

#     def predict(self, input_image):
#         # Apply model to input
#         predictions = self.model(input_image)

#         # Determine class with highest score
#         class_id = tf.argmax(predictions, axis=1)
#         class_name = decode_predictions(predictions.numpy())[0][0][1]
#         class_score = tf.reduce_max(predictions)

#         print(f"{class_name} ({class_id.numpy()[0]}): {100 * class_score:.1f}%")

#         return class_name