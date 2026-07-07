import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models, transforms
from PIL import Image
from supabase import create_client
import datetime
import bcrypt

# Supabase setup
SUPABASE_URL = "https://ewinyadbhqkskhlolmid.supabase.co"
SUPABASE_KEY = "sb_publishable_WX8DACaZIEI84VvST7uW6g_aLheUnGi"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# In-memory session store
sessions = {}

def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(torch.load("models/brain_tumor_model.pth", map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ── Auth functions ──────────────────────────────────────────

def register_user(username, password, confirm_password):
    if not username or not password:
        return "ERROR: Username and password are required.", gr.update(visible=True), gr.update(visible=False)
    if password != confirm_password:
        return "ERROR: Passwords do not match.", gr.update(visible=True), gr.update(visible=False)
    if len(password) < 6:
        return "ERROR: Password must be at least 6 characters.", gr.update(visible=True), gr.update(visible=False)

    try:
        existing = supabase.table("users").select("id").eq("username", username).execute()
        if existing.data:
            return "ERROR: Username already exists.", gr.update(visible=True), gr.update(visible=False)

        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        supabase.table("users").insert({
            "username": username,
            "password_hash": password_hash
        }).execute()
        return f"Account created successfully. You can now log in as '{username}'.", gr.update(visible=True), gr.update(visible=False)
    except Exception as e:
        return f"ERROR: {e}", gr.update(visible=True), gr.update(visible=False)

def login_user(username, password, request: gr.Request):
    if not username or not password:
        return "ERROR: Username and password are required.", gr.update(visible=True), gr.update(visible=False), ""

    try:
        result = supabase.table("users").select("*").eq("username", username).execute()
        if not result.data:
            return "ERROR: Username not found.", gr.update(visible=True), gr.update(visible=False), ""

        user = result.data[0]
        if not bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
            return "ERROR: Incorrect password.", gr.update(visible=True), gr.update(visible=False), ""

        sessions[request.session_hash] = {"user_id": user["id"], "username": username}
        return "", gr.update(visible=False), gr.update(visible=True), username

    except Exception as e:
        return f"ERROR: {e}", gr.update(visible=True), gr.update(visible=False), ""

def logout_user(request: gr.Request):
    if request.session_hash in sessions:
        del sessions[request.session_hash]
    return gr.update(visible=True), gr.update(visible=False), "", ""

def get_current_user(request: gr.Request):
    session = sessions.get(request.session_hash)
    if session:
        return session["username"], session["user_id"]
    return None, None

# ── ML functions ────────────────────────────────────────────

def analyze_single(image):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.layer4[2].conv3
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

    return pred_class, probs, overlay

def analyze_all(patient_name, patient_age, images, request: gr.Request):
    username, user_id = get_current_user(request)
    if not user_id:
        return None, None, None, "ERROR: Not logged in.", ""
    if not patient_name:
        return None, None, None, "ERROR: Patient name is required.", ""
    if not images:
        return None, None, None, "ERROR: No images uploaded.", ""

    overlays = []
    all_results = []
    history_rows = []

    for i, image in enumerate(images):
        img_array = image[0] if isinstance(image, tuple) else image
        pred_class, probs, overlay = analyze_single(img_array)
        overlays.append(overlay)

        result_text = (
            f"[ SCAN {i+1} ]\n"
            f"Classification  :  {CLASS_NAMES[pred_class].upper()}\n"
            f"Confidence      :  {probs[pred_class]*100:.1f}%\n"
            f"\n"
            f"Glioma          :  {probs[0]*100:.1f}%\n"
            f"Meningioma      :  {probs[1]*100:.1f}%\n"
            f"No Tumor        :  {probs[2]*100:.1f}%\n"
            f"Pituitary       :  {probs[3]*100:.1f}%\n"
        )
        all_results.append(result_text)

        try:
            supabase.table("analyses").insert({
                "user_id": user_id,
                "patient_name": patient_name,
                "patient_age": int(patient_age) if patient_age else None,
                "scan_date": datetime.datetime.now().isoformat(),
                "predicted_class": CLASS_NAMES[pred_class],
                "confidence": float(probs[pred_class]),
                "glioma_prob": float(probs[0]),
                "meningioma_prob": float(probs[1]),
                "notumor_prob": float(probs[2]),
                "pituitary_prob": float(probs[3]),
                "image_filename": f"scan_{i+1}"
            }).execute()
            save_status = "SAVED"
        except Exception as e:
            save_status = f"FAILED: {e}"

        history_rows.append([
            patient_name,
            str(patient_age),
            CLASS_NAMES[pred_class].upper(),
            f"{probs[pred_class]*100:.1f}%",
            save_status
        ])

    combined_results = "\n\n".join(all_results)
    session_text = "\n".join([
        f"{r[0]}  |  Age {r[1]}  |  {r[2]}  |  {r[3]}  |  {r[4]}"
        for r in history_rows
    ])

    out1 = overlays[0] if len(overlays) > 0 else None
    out2 = overlays[1] if len(overlays) > 1 else None
    out3 = overlays[2] if len(overlays) > 2 else None
    return out1, out2, out3, combined_results, session_text

def load_history(request: gr.Request):
    username, user_id = get_current_user(request)
    if not user_id:
        return "ERROR: Not logged in."
    try:
        response = supabase.table("analyses").select("*").eq("user_id", user_id).order("scan_date", desc=True).limit(50).execute()
        rows = response.data
        if not rows:
            return "No records found for this account."
        lines = [
            f"{'DATE':<12} {'PATIENT':<22} {'AGE':<6} {'CLASSIFICATION':<18} {'CONFIDENCE'}",
            "-" * 75
        ]
        for r in rows:
            lines.append(
                f"{r['scan_date'][:10]:<12} {r['patient_name']:<22} {str(r['patient_age']):<6} "
                f"{r['predicted_class'].upper():<18} {r['confidence']*100:.1f}%"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"

# ── CSS ─────────────────────────────────────────────────────

css = """
* {
    font-family: 'Consolas', 'JetBrains Mono', 'Courier New', monospace !important;
    box-sizing: border-box;
}

body, .gradio-container {
    background-color: #1e1e2e !important;
    color: #cdd6f4 !important;
    margin: 0 !important;
    padding: 0 !important;
    min-height: 100vh !important;
}

.gradio-container {
    max-width: 100% !important;
    padding: 8px 16px !important;
}

footer { display: none !important; }

h1, h2, h3 {
    color: #89b4fa !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    margin: 0 0 8px 0 !important;
}

h1 { font-size: 1rem !important; color: #cdd6f4 !important; }
h2 { font-size: 0.72rem !important; color: #6c7086 !important; }

.block, .panel {
    background-color: #181825 !important;
    border: 1px solid #313244 !important;
    border-radius: 4px !important;
    padding: 12px !important;
    box-shadow: none !important;
}

label, .label-wrap span {
    color: #89b4fa !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-weight: 600 !important;
}

input[type="text"],
input[type="number"],
input[type="password"],
textarea {
    background-color: #11111b !important;
    border: 1px solid #313244 !important;
    border-radius: 3px !important;
    color: #cdd6f4 !important;
    font-family: 'Consolas', monospace !important;
    font-size: 0.85rem !important;
    padding: 8px 10px !important;
}

input:focus, textarea:focus {
    border-color: #89b4fa !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(137, 180, 250, 0.1) !important;
}

button {
    font-family: 'Consolas', monospace !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    border-radius: 3px !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}

button.primary {
    background-color: #89b4fa !important;
    color: #11111b !important;
    border: none !important;
    padding: 10px 20px !important;
    font-weight: 700 !important;
}

button.primary:hover { background-color: #b4befe !important; }

button.secondary {
    background-color: transparent !important;
    color: #89b4fa !important;
    border: 1px solid #89b4fa !important;
    padding: 8px 16px !important;
}

button.secondary:hover { background-color: rgba(137, 180, 250, 0.1) !important; }

.tab-nav button {
    background-color: #181825 !important;
    color: #6c7086 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 8px 16px !important;
    font-size: 0.75rem !important;
}

.tab-nav button.selected {
    color: #89b4fa !important;
    border-bottom: 2px solid #89b4fa !important;
    background-color: #1e1e2e !important;
}

textarea {
    background-color: #11111b !important;
    border: 1px solid #313244 !important;
    color: #a6e3a1 !important;
    font-size: 0.8rem !important;
    line-height: 1.6 !important;
}

hr { border-color: #313244 !important; margin: 16px 0 !important; }
"""

# ── UI ──────────────────────────────────────────────────────

with gr.Blocks(title="NeuroScan AI", css=css) as app:

    current_user = gr.State("")

    # ── Auth panel ──
    with gr.Column(visible=True) as auth_panel:
        gr.Markdown("# NEUROSCAN AI  --  Brain Tumor MRI Classification System")
        gr.Markdown("## Secure Login  |  ResNet50  |  Grad-CAM  |  95.4% Accuracy")
        gr.Markdown("---")

        with gr.Tabs():
            with gr.Tab("Login"):
                login_username = gr.Textbox(label="Username", placeholder="Enter username")
                login_password = gr.Textbox(label="Password", placeholder="Enter password", type="password")
                login_btn = gr.Button("Login", variant="primary")
                login_msg = gr.Textbox(label="Status", show_label=False, interactive=False, lines=1)

            with gr.Tab("Register"):
                reg_username = gr.Textbox(label="Username", placeholder="Choose a username")
                reg_password = gr.Textbox(label="Password", placeholder="Choose a password (min 6 chars)", type="password")
                reg_confirm = gr.Textbox(label="Confirm Password", placeholder="Repeat password", type="password")
                reg_btn = gr.Button("Create Account", variant="primary")
                reg_msg = gr.Textbox(label="Status", show_label=False, interactive=False, lines=1)

    # ── Main app panel ──
    with gr.Column(visible=False) as main_panel:
        with gr.Row():
            gr.Markdown("# NEUROSCAN AI  --  Brain Tumor MRI Classification System  v1.0")
            logout_btn = gr.Button("Logout", variant="secondary", scale=0)

        user_label = gr.Markdown("## Logged in as: —")
        gr.Markdown("---")

        with gr.Tabs():
            with gr.Tab("Analysis"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=280):
                        gr.Markdown("## Patient Record")
                        patient_name = gr.Textbox(label="Patient Name", placeholder="Enter full name")
                        patient_age = gr.Number(label="Patient Age", precision=0)
                        gr.Markdown("## MRI Input")
                        images = gr.Gallery(label="Upload Scans (max 3)", type="numpy", columns=1, rows=3)
                        analyze_btn = gr.Button("Run Analysis", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown("## Grad-CAM Overlays")
                        output_image1 = gr.Image(label="Scan 1  --  Grad-CAM", type="numpy")
                        output_image2 = gr.Image(label="Scan 2  --  Grad-CAM", type="numpy")
                        output_image3 = gr.Image(label="Scan 3  --  Grad-CAM", type="numpy")

                    with gr.Column(scale=1):
                        gr.Markdown("## Classification Output")
                        results_text = gr.Textbox(lines=18, show_label=False, placeholder="Results will appear here...")
                        gr.Markdown("## Session Log")
                        session_text = gr.Textbox(lines=6, show_label=False)

            with gr.Tab("Patient History"):
                gr.Markdown("## Your Records")
                history_btn = gr.Button("Load My Records", variant="secondary")
                history_text = gr.Textbox(lines=20, show_label=False, placeholder="Click Load My Records...")

            with gr.Tab("System Info"):
                gr.Markdown("## System Information")
                gr.Textbox(
                    value=(
                        "MODEL          :  ResNet50 (pretrained ImageNet, fine-tuned)\n"
                        "DATASET        :  Masoud Nickparvar Brain Tumor MRI Dataset\n"
                        "TRAINING       :  5,600 images  |  20 epochs\n"
                        "TEST SET       :  1,600 images\n"
                        "ACCURACY       :  95.4% (test)  |  99.8% (train)\n"
                        "CLASSES        :  Glioma  |  Meningioma  |  No Tumor  |  Pituitary\n"
                        "EXPLAINABILITY :  Grad-CAM (layer4[2].conv3)\n"
                        "DATABASE       :  Supabase PostgreSQL\n"
                        "DEVICE         :  Apple MPS (Metal Performance Shaders)\n"
                        "FRAMEWORK      :  PyTorch 2.12.1  |  Gradio 6.19.0\n"
                    ),
                    lines=12,
                    show_label=False,
                    interactive=False
                )

        gr.Markdown("---")
        gr.Markdown("## NeuroScan AI  |  For research use only  |  Not for clinical diagnosis")

    # ── Event handlers ──

    def handle_login(username, password, request: gr.Request):
        msg, auth_vis, main_vis, uname = login_user(username, password, request)
        user_label_text = f"## Logged in as:  {uname.upper()}" if uname else "## Logged in as: —"
        return msg, auth_vis, main_vis, uname, user_label_text

    def handle_logout(request: gr.Request):
        auth_vis, main_vis, uname, label = logout_user(request)
        return auth_vis, main_vis, uname, "## Logged in as: —"

    login_btn.click(
        fn=handle_login,
        inputs=[login_username, login_password],
        outputs=[login_msg, auth_panel, main_panel, current_user, user_label]
    )

    reg_btn.click(
        fn=register_user,
        inputs=[reg_username, reg_password, reg_confirm],
        outputs=[reg_msg, auth_panel, main_panel]
    )

    logout_btn.click(
        fn=handle_logout,
        outputs=[auth_panel, main_panel, current_user, user_label]
    )

    analyze_btn.click(
        fn=analyze_all,
        inputs=[patient_name, patient_age, images],
        outputs=[output_image1, output_image2, output_image3, results_text, session_text]
    )

    history_btn.click(fn=load_history, outputs=history_text)

app.launch()