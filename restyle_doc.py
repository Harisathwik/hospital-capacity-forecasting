"""
Restyle the Google Doc with proper formatting.
Uses the Google Docs API to apply headings, bold, links, and spacing.
"""
import json
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# Load the existing OAuth token
with open(r"C:\Users\harisathwik\AppData\Local\hermes\google_token.json") as f:
    token_data = json.load(f)

creds = Credentials.from_authorized_user_info(token_data)

# Build the Docs API service
service = build("docs", "v1", credentials=creds)

DOC_ID = "1nXDOg3Awxht1Nw7TXrwotbLMSw75by3jgxs0F9tk9GU"

# Get the current document to find content boundaries
doc = service.documents().get(documentId=DOC_ID).execute()
content = doc.get("body", {}).get("content", [])
last_end = 1
for elem in content:
    if "endIndex" in elem:
        last_end = max(last_end, elem["endIndex"])

requests = []

# Clear all content after index 1
if last_end > 2:
    requests.append({
        "deleteContentRange": {
            "range": {
                "startIndex": 2,
                "endIndex": last_end - 1
            }
        }
    })

# Content blocks: (text, style_dict)
blocks = [
    ("Veerla Harisathwik — Artifacts & Documents\n", {"bold": True}),
    ("\n", {}),
    ("Portfolio: ", {"bold": True}),
    ("https://harisathwik.github.io/", {"link": {"url": "https://harisathwik.github.io/"}}),
    ("\n\n", {}),
    ("Status: ", {"bold": True}),
    ("Submitted\n", {"bold": True}),
    ("\n\n", {}),
    ("MLOps Project — Telco Customer Churn Prediction\n", {"bold": True}),
    ("\n", {}),
    ("Algorithm Writeup: ", {"bold": True}),
    ("https://drive.google.com/file/d/17yoq6WDuvEG2_T6zjYmjYfnx0hnwpKDD/view?usp=drivesdk\n",
     {"link": {"url": "https://drive.google.com/file/d/17yoq6WDuvEG2_T6zjYmjYfnx0hnwpKDD/view?usp=drivesdk"}}),
    ("\n", {}),
    ("Project Repo: ", {"bold": True}),
    ("D:\\Sathwik\\Ayush\\MLOps-Github\\\n", {}),
    ("\n\n", {}),
    ("Personal Voice & Brand\n", {"bold": True}),
    ("\n", {}),
    ("Customised Master Plan: ", {"bold": True}),
    ("https://drive.google.com/file/d/1FWA1j-6fPJWH_W74cua7ELvp7uIBvXgC/view?usp=drivesdk\n",
     {"link": {"url": "https://drive.google.com/file/d/1FWA1j-6fPJWH_W74cua7ELvp7uIBvXgC/view?usp=drivesdk"}}),
    ("\n", {}),
]

# Build insert + style requests
current_index = 2
for text, style in blocks:
    if not text:
        continue

    # Insert text
    requests.append({
        "insertText": {
            "location": {"index": current_index},
            "text": text
        }
    })

    # Apply style
    if style:
        fields = ",".join(style.keys())
        requests.append({
            "updateTextStyle": {
                "range": {
                    "startIndex": current_index,
                    "endIndex": current_index + len(text)
                },
                "textStyle": style,
                "fields": fields
            }
        })

    current_index += len(text)

# Execute
result = service.documents().batchUpdate(
    documentId=DOC_ID,
    body={"requests": requests}
).execute()

print(f"Document restyled! {len(requests)} requests executed.")
print(f"View: https://docs.google.com/document/d/{DOC_ID}/edit?usp=sharing")
