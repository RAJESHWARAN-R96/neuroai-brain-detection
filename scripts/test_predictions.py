import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.models import build_model, MetaheuristicEnsembleModel
from core.preprocessing import preprocess_mri_pipeline, ensure_numpy_image
from core.dataset import CLASS_NAMES
import torchvision.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)

models = {}
for mname in ['densenet121', 'resnet50', 'efficientnet_b0']:
    m = build_model(mname, num_classes=4, pretrained=False).to(device)
    ckpt = os.path.join(BASE_DIR, 'weights', f'{mname}_best.pth')
    if os.path.exists(ckpt):
        m.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded {mname} checkpoint", flush=True)
    m.eval()
    models[mname] = m

# Build ensemble with the trained weights
weights = [0.3419, 0.0067, 0.6514] # densenet121, resnet50, efficientnet_b0
ens_models = [models['densenet121'], models['resnet50'], models['efficientnet_b0']]
ensemble = MetaheuristicEnsembleModel(ens_models, weights=weights).to(device)
ensemble.eval()
models['ensemble'] = ensemble

raw_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

sample_root = os.path.join(BASE_DIR, 'data', 'sample_mris')

print("\n" + "="*50, flush=True)
print("TESTING SAMPLES WITH DIRECT RESIZE + NORMALIZE", flush=True)
print("="*50, flush=True)
for cname in CLASS_NAMES:
    cdir = os.path.join(sample_root, cname)
    if not os.path.exists(cdir): continue
    for fname in sorted(os.listdir(cdir))[:2]:
        fpath = os.path.join(cdir, fname)
        img = Image.open(fpath).convert('RGB')
        t = raw_tf(img).unsqueeze(0).to(device)
        print(f"\nGround Truth: {cname.upper()} ({fname})", flush=True)
        for mname, m in models.items():
            out = m(t)
            probs = F.softmax(out, dim=-1)[0].detach().cpu().numpy()
            pred = CLASS_NAMES[np.argmax(probs)]
            conf = np.max(probs) * 100
            print(f"  {mname:16}: {pred:12} ({conf:5.1f}%) [all: {[round(float(p),2) for p in probs]}]", flush=True)

print("\n" + "="*50, flush=True)
print("TESTING SAMPLES WITH PREPROCESS PIPELINE", flush=True)
print("="*50, flush=True)
for cname in CLASS_NAMES:
    cdir = os.path.join(sample_root, cname)
    if not os.path.exists(cdir): continue
    for fname in sorted(os.listdir(cdir))[:2]:
        fpath = os.path.join(cdir, fname)
        raw_np = ensure_numpy_image(fpath)
        prep = preprocess_mri_pipeline(raw_np, return_tensor=True)
        t = prep['tensor'].to(device)
        print(f"\nGround Truth: {cname.upper()} ({fname})", flush=True)
        for mname, m in models.items():
            out = m(t)
            probs = F.softmax(out, dim=-1)[0].detach().cpu().numpy()
            pred = CLASS_NAMES[np.argmax(probs)]
            conf = np.max(probs) * 100
            print(f"  {mname:16}: {pred:12} ({conf:5.1f}%) [all: {[round(float(p),2) for p in probs]}]", flush=True)
