import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(torch.load("models/brain_tumor_model.pth", map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

def predict(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    model = load_model()
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)

    return CLASS_NAMES[predicted.item()], confidence.item()

if __name__ == "__main__":
    # Quick test - pick any image from the test set
    test_image = "data/Testing/glioma/Te-gl_1.jpg"
    label, confidence = predict(test_image)
    print(f"Predicted: {label} ({confidence*100:.1f}% confidence)")
