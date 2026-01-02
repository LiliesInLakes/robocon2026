# TRAINING SCRIPT (Run this in Google Colab)
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
import torch.onnx

# 1. SETUP: Define your classes
# Adjust this to match your folder names exactly
CLASS_NAMES = ['Robocon Logo', 'Oracle Bone', 'Random Symbols', 'background']

# 2. PREPARE THE MODEL (MobileNetV3-Small)
# We download a pre-built smart brain and adjust the last layer to output YOUR 4 classes
model = torchvision.models.mobilenet_v3_small(pretrained=True)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASS_NAMES))

# 3. LOAD DATA
# Assume you unzipped your dataset into a folder called 'dataset'
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize images to standard size
    transforms.ToTensor(),
])
dataset = datasets.ImageFolder(root='desktop/mobilenet/my_data', transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# 4. TRAIN (Simple Loop)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("Training started... (this takes about 5 mins)")
for epoch in range(5):  # Train for 5 loops (epochs)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
print("Training finished!")

# 5. EXPORT TO ONNX (The "Deployment" Step)
# This creates the tiny file you need for your local device
dummy_input = torch.randn(1, 3, 224, 224).to(device)
onnx_path = "desktop/mobilenet/symbol_classifier.onnx"

torch.onnx.export(
    model, 
    dummy_input, 
    onnx_path,
    input_names=['input'], 
    output_names=['output'],
    opset_version=11
)

print(f"Model saved as {onnx_path}. Download this file!")
if