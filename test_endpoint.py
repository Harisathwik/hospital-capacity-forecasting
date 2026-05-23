import sys
sys.path.append('D:/Sathwik/Ayush/MLOps-Github/src')
from zenml.client import Client
import requests
import json

client = Client()
run_id = "7ecc5275-4b07-4ffb-bfd7-2ceebd87eccc"
run = client.get_pipeline_run(run_id)

# Load feature names from trainer_step_2::output_2
matching_avs = [av for av in run.artifact_versions if av.name.endswith('trainer_step_2::output_2')]
if not matching_avs:
    print("ERROR: Could not find feature names artifact")
    sys.exit(1)
feat_av = matching_avs[0]
feature_names = client.get_artifact_version(feat_av.id).load()
print(f'Loaded {len(feature_names)} feature names')

# Create a dummy input with zeros (or maybe we can use some real values from the data)
# For now, zeros
dummy_input = {name: 0.0 for name in feature_names}
print('Sample input keys:', list(dummy_input.keys())[:5])

# Now we can test the endpoint
url = 'http://127.0.0.1:8002/predict'
payload = {'features': dummy_input}
print('Sending request...')
try:
    response = requests.post(url, json=payload, timeout=10)
    print('Status code:', response.status_code)
    print('Response:', response.json())
except Exception as e:
    print('Error:', e)