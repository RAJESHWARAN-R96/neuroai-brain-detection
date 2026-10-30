import urllib.request
import json

with open('uploads/750adf6a5f_raw.jpg', 'rb') as f:
    img_bytes = f.read()

boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
body = bytearray()
body.extend(f'--{boundary}\r\n'.encode('utf-8'))
body.extend(b'Content-Disposition: form-data; name="file"; filename="user_glioma.jpg"\r\n')
body.extend(b'Content-Type: image/jpeg\r\n\r\n')
body.extend(img_bytes)
body.extend(f'\r\n--{boundary}--\r\n'.encode('utf-8'))

req = urllib.request.Request(
    'http://127.0.0.1:5000/api/predict',
    data=bytes(body),
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
)

with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))

print("=== PREDICTION RESPONSE ===")
print("Status:", data.get('status'))
print("Predicted Class:", data.get('predicted_class'))
print("Confidence:", data.get('confidence'))
print("Probabilities:", data.get('probabilities'))
print("Tumor Metrics:", data.get('tumor_metrics'))
print("Keys in clinical_guidance:", list(data.get('clinical_guidance', {}).keys()))
cur = data.get('clinical_guidance', {}).get('curability', {})
print(f"Curability: Curable = {cur.get('curable_pct')}%, Non-curable = {cur.get('non_curable_pct')}% ({cur.get('prognosis_category')})")
print("Primary Solution:", data.get('clinical_guidance', {}).get('primary_solution'))
meds = data.get('clinical_guidance', {}).get('recommended_medicines', [])
print(f"Recommended Medicines ({len(meds)}):")
for med in meds:
    print(f"  - {med['name']} ({med['category']}): {med['dosage']} [{med['purpose']}]")
