import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
class Single_Stem_Classifier:
    def __init__(self, model_path, num_classes):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path, num_classes)
        self.model.eval()  # Set the model to evaluation mode
        self.transform = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,))
        ])
    def load_model(self, model_path, num_classes):        
        class SimpleCNN(nn.Module):
            def __init__(self, num_classes):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                self.fc1 = nn.Linear(64 * 8 * 8, 512)  # Adjusted for 64x64 input size -> 64:16, 32:8, 16:4 etc
                self.fc2 = nn.Linear(512, num_classes)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = x.view(-1, 64 * 8 * 8) # also have to change here
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        model = SimpleCNN(num_classes).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model
    def proprocess_image(self, img:np.ndarray):
        w = img.shape[1]
        h = img.shape[0]
        if h>=w:
            left = (h-w)//2
            right = h-w-left
            normalize_transform = transforms.Compose([
                transforms.Normalize((0.5,), (0.5,))
            ])
            square = cv2.copyMakeBorder(img, 0,0,left, right, cv2.BORDER_CONSTANT, value = (0,0,0))
            image = cv2.cvtColor(square, cv2.COLOR_GRAY2RGB)
            image = cv2.resize(image, (32,32))  # Resize image to 64x64
            image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W) format, pytorch wants
            image = torch.tensor(image, dtype=torch.float32) / 255.0  # Convert to tensor and scale to [0, 1]
            image = normalize_transform(image)  # Normalize the image
            image = image.unsqueeze(0)  # Add batch dimension
        else:
            top = (w-h)//2
            bottom = w-h-top
            normalize_transform = transforms.Compose([
                transforms.Normalize((0.5,), (0.5,))
            ])
            square = cv2.copyMakeBorder(img, top, bottom, 0,0,cv2.BORDER_CONSTANT, value = (0,0,0))
            image = cv2.cvtColor(square, cv2.COLOR_GRAY2RGB)
            image = cv2.resize(image, (32,32))  # Resize image to 64x64
            image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W) format, pytorch wants
            image = torch.tensor(image, dtype=torch.float32) / 255.0  # Convert to tensor and scale to [0, 1]
            image = normalize_transform(image)  # Normalize the image
            image = image.unsqueeze(0)  # Add batch dimension
        return image
    def predict(self, img:np.ndarray):
        img_sq = self.proprocess_image(img).to(self.device)
        with torch.no_grad():
            output = self.model(img_sq)
            _, predicted = torch.max(output.data, 1)
            class_idx = predicted.item()
        return class_idx
    def get_prediction_vector(self, img:np.ndarray):
        img_sq = self.proprocess_image(img).to(self.device)
        with torch.no_grad():
            output = self.model(img_sq)
            _, sorted_idx = torch.sort(output, descending = True)
            idx = sorted_idx.tolist()
            assert len(idx)==1
        return idx[0]
    

if __name__ == '__main__':
    print('stem down prediction')
    # for stemDown's classification
    img = cv2.imread(r"C:\Ellie\ellie2023~2024\iis\omr-iis\training\single_stem\StemDown_img200\n4\bach_209.jpg", cv2.IMREAD_GRAYSCALE)
    pth_path = 'training\stemdownImg32x32_best.pth'
    num_classes = 3
    classifier = Single_Stem_Classifier(pth_path, num_classes)
    idx = classifier.predict(img)
    vect = classifier.get_prediction_vector(img)
    class_names = ['n2', 'n4', 'n816']  # Replace with actual class names
    print(f'The predicted class is: {class_names[idx]}, class: {idx}')

    img2 = cv2.imread(r"C:\Ellie\ellie2023~2024\iis\omr-iis\training\single_stem\StemDown_img200\n16\bach_74.jpg", cv2.IMREAD_GRAYSCALE)
    idx2 = classifier.predict(img2)
    print(f'The predicted class is: {class_names[idx2]}, class: {idx2}')
    print()

    print('stem up prediction')
    # for stemUp's classification
    img = cv2.imread(r"C:\Ellie\ellie2023~2024\iis\omr-iis\training\single_stem\StemUp_img200\n4\bach_355.jpg", cv2.IMREAD_GRAYSCALE)
    pth_path = 'training\stemupImg32x32_best.pth'
    num_classes = 3
    classifier = Single_Stem_Classifier(pth_path, num_classes)
    idx = classifier.predict(img)
    vect = classifier.get_prediction_vector(img)
    class_names = ['n2', 'n4', 'n816']  # Replace with actual class names
    print(f'The predicted class is: {class_names[idx]}, class: {idx}')

    img2 = cv2.imread(r"C:\Ellie\ellie2023~2024\iis\omr-iis\training\single_stem\StemUp_img200\n8\bach_292.jpg", cv2.IMREAD_GRAYSCALE)
    idx2 = classifier.predict(img2)
    print(f'The predicted class is: {class_names[idx2]}, class: {idx2}')

    img2 = cv2.imread(r"C:\Ellie\ellie2023~2024\iis\omr-iis\training\single_stem\StemUp_img200\n1\fant_switch_91.jpg", cv2.IMREAD_GRAYSCALE)
    idx2 = classifier.predict(img2)
    print(f'The predicted class is: {class_names[idx2]}, class: {idx2}')
    print()

