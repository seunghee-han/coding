import timm
import torch
from skimage import io
from torchvision.models import vgg19, resnet18
from torchsummary import summary
import numpy as np
import cv2
import os

timm.list_models('vit_*')

class GradCam:
    def __init__(self, model, target):
        self.model = model.eval()
        self.feature = None
        self.gradient = None
        self.handlers = []
        self.target = target
        self._get_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = self.reshape_transform(output)

    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient = self.reshape_transform(output_grad)

        def _store_grad(grad):  
            self.gradient = self.reshape_transform(grad)

        output_grad.register_hook(_store_grad)
    
    def _get_hook(self):
        self.target.register_forward_hook(self._get_features_hook)
        self.target.register_forward_hook(self._get_grads_hook)

    def reshape_transform(self, tensor, height=14, width=14):
        result = tensor[:, 1:, :].reshape(tensor.size(0),
                                          height, 
                                          width, 
                                          tensor.size(2))

        result = result.transpose(2, 3).transpose(1, 2)
        return result

    def __call__(self, inputs):
            
        self.model.zero_grad()
        output = self.model(inputs) 
        
        index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]
        target.backward()

        gradient = self.gradient[0].cpu().data.numpy()
        weight = np.mean(gradient, axis=(1, 2))
        feature = self.feature[0].cpu().data.numpy()

        cam = feature * weight[:, np.newaxis, np.newaxis]
        cam = np.sum(cam, axis=0) 
        cam = np.maximum(cam, 0) 

        cam -= np.min(cam)
        cam /= np.max(cam)
        cam = cv2.resize(cam, (224, 224))
        return cam
    
def prepare_input(image):
    image = image.copy()
    means = np.array([0.5, 0.5, 0.5])
    stds = np.array([0.5, 0.5, 0.5])
    image = (image - means) / stds
    image = np.transpose(image, (2, 0, 1))
    image = image[np.newaxis, ...]
    return torch.tensor(image, dtype=torch.float32, requires_grad=True)

def gen_cam(image, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = (1 - 0.5) * heatmap + 0.5 * image
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def process_image(img_path, model, target_layer, output_dir):
    # Load the image
    original_img = io.imread(img_path)
    
    # Preserve original image size
    h, w = original_img.shape[:2]
    
    # Prepare input for the model (224x224)
    img_224 = cv2.resize(original_img, (224, 224))
    img_224 = np.float32(img_224) / 255
    inputs = prepare_input(img_224)

    # Generate Grad-CAM
    grad_cam = GradCam(model, target_layer)
    mask = grad_cam(inputs)
    
    # Resize mask to original image size
    mask_resized = cv2.resize(mask, (w, h))
    
    # Generate CAM on original sized image
    original_img_float = np.float32(original_img) / 255
    result = gen_cam(original_img_float, mask_resized)
    
    # Save the result
    output_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(output_path, result)

if __name__ == '__main__':
    # Set input and output directories
    input_dir = "/workspace/datasets/dukemtmcreid/DukeMTMC-reID/bounding_box_test" ##### 이미지가 있는 폴더
    output_dir = "/workspace/grad_cam_results" ##### 저장될 위치
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create model and set target layer
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    target_layer = model.blocks[-1].norm1

    # Process all images in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            process_image(img_path, model, target_layer, output_dir)

    print(f"Grad-CAM results have been saved to {output_dir}")
