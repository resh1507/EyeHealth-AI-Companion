import os
import numpy as np
import pickle
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import cv2
from io import BytesIO

# ------------------------------
# Keras / TensorFlow
# ------------------------------
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.applications.efficientnet import preprocess_input as enet_preprocess
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from gradcam_utils import get_gradcam_heatmap, overlay_gradcam

# ------------------------------
# PyTorch
# ------------------------------
import torch
from torchvision import transforms, models
from torch import nn
from timm import create_model
from transformers import ViTModel

# ------------------------------
# LightGBM
# ------------------------------
import lightgbm as lgb

# ------------------------------
# Paths
# ------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project root
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")


MODEL_PATH = os.path.join("models/fused_lightgbm.txt")
CLASS_INDICES_PATH = os.path.join("models/class_indices.pkl")

# ------------------------------
# Load LightGBM model
# ------------------------------
lgb_model = lgb.Booster(model_file=MODEL_PATH)

# ------------------------------
# Load class indices
# ------------------------------
with open(CLASS_INDICES_PATH, "rb") as f:
    class_indices = pickle.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}

# ------------------------------
# Load EfficientNetB5
# ------------------------------
enet_base = EfficientNetB5(weights="imagenet", include_top=False, input_shape=(456, 456, 3))
x = GlobalAveragePooling2D()(enet_base.output)
enet_model = Model(inputs=enet_base.input, outputs=x)

# ------------------------------
# Load ResNet50
# ------------------------------
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet.fc = nn.Identity()
resnet.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device)

# ------------------------------
# Load Swin-T
# ------------------------------
swin = create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
swin.eval()
swin.to(device)

# ------------------------------
# Load ViT-B/16
# ------------------------------
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model.eval()
vit_model.to(device)



# ------------------------------
# Flask app
# ------------------------------
app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)  # enable CORS

# ------------------------------
# Serve frontend
# ------------------------------
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(app.static_folder, path)

# ------------------------------
# Transforms
# ------------------------------
resnet_swin_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# ------------------------------
# Feature extraction
# ------------------------------
def extract_features(img_path):
    img_enet = Image.open(img_path).convert("RGB").resize((456, 456))
    x_enet = np.expand_dims(enet_preprocess(np.array(img_enet).astype(np.float32)), axis=0)
    enet_feats = enet_model.predict(x_enet)

    x_resnet = resnet_swin_transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        resnet_feats = resnet(x_resnet).cpu().numpy()

    x_swin = resnet_swin_transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = swin(x_swin)
        if len(feats.shape) == 3:
            feats = feats.mean(dim=1)
        swin_feats = feats.cpu().numpy()

    x_vit = vit_transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = vit_model(pixel_values=x_vit).last_hidden_state
        vit_feats = outputs[:, 0, :].cpu().numpy()

    fused = np.concatenate([enet_feats, resnet_feats, vit_feats, swin_feats], axis=1)
    return fused

# ------------------------------
# Prediction route
# ------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    temp_path = "temp.jpg"
    file.save(temp_path)

    # Extract features & predict
    fused_feats = extract_features(temp_path)
    preds = lgb_model.predict(fused_feats)
    predicted_class_idx = np.argmax(preds, axis=1)[0]
    predicted_class = idx_to_class[predicted_class_idx]

    prob_dict = {idx_to_class[i]: float(preds[0][i]) for i in range(len(preds[0]))}

    # Grad-CAM using EfficientNetB5
    img_enet = Image.open(temp_path).convert("RGB").resize((456, 456))
    x_enet = np.expand_dims(enet_preprocess(np.array(img_enet).astype(np.float32)), axis=0)
    enet_full = EfficientNetB5(weights="imagenet", include_top=True)
    heatmap = get_gradcam_heatmap(enet_full, x_enet, last_conv_layer_name="top_conv")
    img_cv = cv2.cvtColor(np.array(img_enet), cv2.COLOR_RGB2BGR)
    gradcam_img = overlay_gradcam(img_cv, heatmap)

    _, buffer = cv2.imencode('.jpg', gradcam_img)
    gradcam_base64 = base64.b64encode(buffer).decode('utf-8')

    with open(temp_path, "rb") as f:
        original_base64 = base64.b64encode(f.read()).decode('utf-8')

    return jsonify({
        'prediction': predicted_class,
        'gradcam_image': gradcam_base64,
        'original_image': original_base64
    })

# ------------------------------
# Run server
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
