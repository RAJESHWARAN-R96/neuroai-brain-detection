import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.server import app, load_all_models, CLASS_NAMES

print("Initializing models...", flush=True)
load_all_models()
client = app.test_client()

# Test sample list endpoint
res = client.get('/api/sample-images')
print("Sample images status:", res.status_code, flush=True)
samples = res.get_json()['samples']
for k, v in samples.items():
    print(f"  {k}: {len(v)} samples", flush=True)

print("\n" + "="*60, flush=True)
print("TESTING PREDICT API ACROSS ALL 4 CLASSES (5 SAMPLES EACH)", flush=True)
print("="*60, flush=True)

total_tested = 0
total_correct = 0

for cname in CLASS_NAMES:
    print(f"\n--- Class: {cname.upper()} ---", flush=True)
    for spath in samples[cname]:
        r = client.post('/api/predict', data={'sample_path': spath, 'model_name': 'ensemble'})
        data = r.get_json()
        pred_class = data.get('predicted_class', 'UNKNOWN')
        conf = data.get('confidence', 0.0)
        is_correct = (pred_class == cname)
        total_tested += 1
        if is_correct: total_correct += 1
        mark = "[PASS]" if is_correct else "[FAIL]"
        print(f"  {mark} Sample: {os.path.basename(spath)} -> Pred: {pred_class.upper():12} ({conf:5.1f}%)", flush=True)

print("\n" + "="*60, flush=True)
print(f"VERIFICATION SUMMARY: {total_correct}/{total_tested} ({total_correct/total_tested*100:.1f}%) Correct", flush=True)
print("="*60, flush=True)
