import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(torch.load("models/brain_tumor_model.pth", map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

def generate_gradcam(image_path):
    model = load_model()

    # Hook to capture gradients and activations
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Attach hooks to the last conv layer
    target_layer = model.layer4[2].conv3
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Forward pass
    output = model(tensor)
    pred_class = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1).max().item()

    # Backward pass
    model.zero_grad()
    output[0, pred_class].backward()

    # Generate heatmap
    grad = gradients[0].cpu().detach().numpy()[0]
    act = activations[0].cpu().detach().numpy()[0]
    weights = grad.mean(axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= cam.min()
    cam /= cam.max()

    # Overlay on original image
    orig = np.array(image.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * orig + 0.5 * heatmap).astype(np.uint8)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig)
    axes[0].set_title("Original MRI")
    axes[0].axis("off")

    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Predicted: {CLASS_NAMES[pred_class]}\nConfidence: {confidence*100:.1f}%")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("gradcam_output.png")
    print(f"Predicted: {CLASS_NAMES[pred_class]} ({confidence*100:.1f}% confidence)")
    print("Grad-CAM saved to gradcam_output.png")

if __name__ == "__main__":
    generate_gradcam("data/Testing/glioma/Te-gl_1.jpg")