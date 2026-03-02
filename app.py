import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
YOLO_PATH = os.path.join(MODEL_DIR, "yolo_best.pt")
SEG_MODEL_PATH = os.path.join(MODEL_DIR, "connected_unet_best.pth")
RESNET18_PATH = os.path.join(MODEL_DIR, "ResNet18_best.pth")
RESNET34_PATH = os.path.join(MODEL_DIR, "ResNet34_best.pth")
RESNET50_PATH = os.path.join(MODEL_DIR, "ResNet50_best.pth")

# Label Maps (assuming standard labels from notebook)
PATHOLOGY_LABELS = ['BENIGN', 'MALIGNANT']
BIRADS_LABELS = ['1', '2', '3', '4', '5', '6']
SHAPE_LABELS = ['IRREGULAR', 'OVAL', 'ROUND', 'LOBULATED', 'ARCHITECTURAL_DISTORTION', 'SPICULATED', 'MACROCALCIFICATION', 'ASYMMETRY', 'MICROCALCIFICATION'] # Updated with common names, will index by index anyway

# ==========================================
# MODEL DEFINITIONS (MATCHING NOTEBOOK)
# ==========================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True)
        )
    def forward(self, x): return self.conv(x)

class ASPPModule(nn.Module):
    def __init__(self, in_ch, out_ch, rates=[6, 12, 18]):
        super().__init__()
        self.conv1x1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True))
        self.atrous = nn.ModuleList([nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True)) for r in rates])
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True))
        self.project = nn.Sequential(nn.Conv2d(out_ch*(len(rates)+2), out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True), nn.Dropout(0.5))
    def forward(self, x):
        res = [self.conv1x1(x)] + [c(x) for c in self.atrous]
        res.append(F.interpolate(self.gap(x), size=x.shape[2:], mode='bilinear', align_corners=False))
        return self.project(torch.cat(res, 1))

class ConnectedUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, feats=[64, 128, 256, 512]):
        super().__init__()
        self.encoders, self.pools = nn.ModuleList(), nn.ModuleList()
        ch = in_ch
        for f in feats: 
            self.encoders.append(ConvBlock(ch, f))
            self.pools.append(nn.MaxPool2d(2))
            ch = f
        self.bottleneck = ASPPModule(feats[-1], feats[-1]*2)
        self.up1, self.dec1 = nn.ModuleList(), nn.ModuleList()
        ch = feats[-1]*2
        for f in reversed(feats): 
            self.up1.append(nn.ConvTranspose2d(ch, f, 2, stride=2))
            self.dec1.append(ConvBlock(f*2, f))
            ch = f
        self.up2, self.dec2 = nn.ModuleList(), nn.ModuleList()
        ch = feats[-1]*2
        for f in reversed(feats): 
            self.up2.append(nn.ConvTranspose2d(ch, f, 2, stride=2))
            self.dec2.append(ConvBlock(f*3, f))
            ch = f
        self.out1 = nn.Conv2d(feats[0], out_ch, 1)
        self.out2 = nn.Conv2d(feats[0], out_ch, 1)
        self.final = nn.Conv2d(out_ch*2, out_ch, 1)
        
    def forward(self, x):
        skips = []; ch = x
        for enc, pool in zip(self.encoders, self.pools): 
            ch = enc(ch)
            skips.append(ch)
            ch = pool(ch)
        x = self.bottleneck(ch); bn = x; d1_out = []
        for i, (u, d) in enumerate(zip(self.up1, self.dec1)):
            x = u(x); s = skips[-(i+1)]
            if x.shape != s.shape: x = F.interpolate(x, size=s.shape[2:], mode='bilinear', align_corners=False)
            x = d(torch.cat([x, s], 1)); d1_out.append(x)
        o1 = self.out1(x); x = bn
        for i, (u, d) in enumerate(zip(self.up2, self.dec2)):
            x = u(x); s = skips[-(i+1)]
            if x.shape != s.shape: x = F.interpolate(x, size=s.shape[2:], mode='bilinear', align_corners=False)
            x = d(torch.cat([x, s, d1_out[i]], 1))
        o2 = self.out2(x)
        return torch.sigmoid(self.final(torch.cat([o1, o2], 1)))

class ResNetClassifier(nn.Module):
    def __init__(self, num_path=2, num_bir=6, num_shp=18, backbone='resnet50'):
        super().__init__()
        if backbone=='resnet50': self.backbone = models.resnet50(weights=None)
        elif backbone=='resnet34': self.backbone = models.resnet34(weights=None)
        else: self.backbone = models.resnet18(weights=None)
        in_f = self.backbone.fc.in_features; self.backbone.fc = nn.Identity()
        self.shared = nn.Sequential(nn.Linear(in_f, 512), nn.ReLU(), nn.Dropout(0.3))
        self.head_path = nn.Linear(512, num_path)
        self.head_bir = nn.Linear(512, num_bir)
        self.head_shp = nn.Linear(512, num_shp)
    def forward(self, x):
        f = self.shared(self.backbone(x))
        return self.head_path(f), self.head_bir(f), self.head_shp(f)

# ==========================================
# MODEL LOADING
# ==========================================

print("Loading YOLO model...")
yolo_model = YOLO(YOLO_PATH)

print("Loading Segmentation model...")
seg_model = ConnectedUNet(in_ch=3, out_ch=1).to(DEVICE)
if os.path.exists(SEG_MODEL_PATH):
    seg_model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=DEVICE))
    seg_model.eval()

print("Loading ResNet ensemble...")
r18 = ResNetClassifier(num_path=2, num_bir=6, num_shp=18, backbone='resnet18').to(DEVICE)
r34 = ResNetClassifier(num_path=2, num_bir=6, num_shp=18, backbone='resnet34').to(DEVICE)
r50 = ResNetClassifier(num_path=2, num_bir=6, num_shp=18, backbone='resnet50').to(DEVICE)

for m, p in zip([r18, r34, r50], [RESNET18_PATH, RESNET34_PATH, RESNET50_PATH]):
    if os.path.exists(p):
        m.load_state_dict(torch.load(p, map_location=DEVICE))
        m.eval()

ensemble = [r18, r34, r50]

# ==========================================
# UTILS
# ==========================================

def img_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

# ==========================================
# ROUTES
# ==========================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    file = request.files['image']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    orig_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w = orig_img.shape[:2]

    # Stage 1: YOLO Detection
    results = yolo_model(orig_img, conf=0.25)
    
    det_img = orig_img.copy()
    seg_img = np.zeros_like(orig_img) # black background or we could overlay
    
    pathology = "No lesion detected"
    birads = "-"
    shape = "-"
    confidence = 0.0
    
    # Just process the highest confidence detection for simplicity
    best_conf = 0
    best_crop = None
    best_box = None
    
    for r in results:
        for box in r.boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf)
            if conf > best_conf:
                best_conf = conf
                best_box = b
                x1, y1, x2, y2 = b
                best_crop = orig_img[y1:y2, x1:x2]

    if best_crop is not None and best_crop.size > 0:
        x1, y1, x2, y2 = best_box
        confidence = best_conf
        
        cv2.rectangle(det_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(det_img, f"Lesion {confidence:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Stage 2: Segmentation
        input_seg = cv2.resize(best_crop, (256, 256))
        
        # Note: Depending on the model, it might need normalization.
        # Connected-UNet from the notebook likely expects [0,1] normalization.
        input_seg = transforms.ToTensor()(input_seg).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            mask_pred = seg_model(input_seg).squeeze().cpu().numpy()
        
        mask_resized = cv2.resize((mask_pred > 0.5).astype(np.uint8), (x2-x1, y2-y1))
        
        # Overlay mask on seg_img
        seg_img = orig_img.copy()
        color_mask = np.zeros_like(best_crop)
        color_mask[mask_resized == 1] = [0, 0, 255] # Red mask
        
        # Blend
        seg_img[y1:y2, x1:x2] = cv2.addWeighted(best_crop, 0.6, color_mask, 0.4, 0)
        
        # Stage 3: Classification (Ensemble)
        input_cls = cv2.resize(best_crop, (224, 224))
        # Note: ResNet commonly uses ImageNet normalization
        transform_cls = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_cls = transform_cls(best_crop).unsqueeze(0).to(DEVICE)
        
        p_path, p_bir, p_shp = 0, 0, 0
        with torch.no_grad():
            for model in ensemble:
                out_p, out_b, out_s = model(input_cls)
                p_path += torch.softmax(out_p, dim=1)
                p_bir += torch.softmax(out_b, dim=1)
                p_shp += torch.softmax(out_s, dim=1)
        
        p_path /= len(ensemble)
        p_bir /= len(ensemble)
        p_shp /= len(ensemble)
        
        path_idx = torch.argmax(p_path).item()
        bir_idx = torch.argmax(p_bir).item()
        shp_idx = torch.argmax(p_shp).item()
        
        pathology = PATHOLOGY_LABELS[path_idx] if path_idx < len(PATHOLOGY_LABELS) else f"Class {path_idx}"
        birads = BIRADS_LABELS[bir_idx] if bir_idx < len(BIRADS_LABELS) else f"Level {bir_idx}"
        shape = SHAPE_LABELS[shp_idx] if shp_idx < len(SHAPE_LABELS) else f"Shape {shp_idx}"
    else:
        # If no lesion is found, seg_img should just be the original image
        seg_img = orig_img.copy()
        pathology = "No lesion detected"
        birads = "-"
        shape = "-"
        confidence = 0.0

    return jsonify({
        'det_img': img_to_base64(det_img),
        'seg_img': img_to_base64(seg_img),
        'pathology': pathology,
        'birads': birads,
        'shape': shape,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)