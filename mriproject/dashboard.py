import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import io

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(torch.load("models/brain_tumor_model.pth", map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_model()

def analyze(image):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.layer4[1].conv2
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    img = Image.fromarray(image).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    output = model(tensor)
    pred_class = output.argmax(dim=1).item()
    probs = torch.softmax(output, dim=1)[0].cpu().detach().numpy()

    model.zero_grad()
    output[0, pred_class].backward()

    grad = gradients[0].cpu().detach().numpy()[0]
    act = activations[0].cpu().detach().numpy()[0]
    weights = grad.mean(axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= cam.min()
    cam /= cam.max() + 1e-8

    orig = np.array(img.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * orig + 0.5 * heatmap).astype(np.uint8)

    fh.remove()
    bh.remove()

    # Confidence scores
    confidence_text = "\n".join([
        f"{CLASS_NAMES[i]}: {probs[i]*100:.1f}%" for i in range(4)
    ])
    result = f"Predicted: {CLASS_NAMES[pred_class].upper()}\nConfidence: {probs[pred_class]*100:.1f}%\n\n{confidence_text}"

    return overlay, result

with gr.Blocks(title="Brain Tumor MRI Classifier") as app:
    gr.Markdown("# 🧠 Brain Tumor MRI Classifier")
    gr.Markdown("Upload an MRI scan to classify the tumor type and visualize where the model is looking.")

    with gr.Row():
        input_image = gr.Image(label="Upload MRI Image")
        output_image = gr.Image(label="Grad-CAM Overlay")

    output_text = gr.Textbox(label="Prediction")
    btn = gr.Button("Analyze", variant="primary")
    btn.click(fn=analyze, inputs=input_image, outputs=[output_image, output_text])

app.launch()
