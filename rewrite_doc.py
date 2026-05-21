"""
Rewrite the entire Google Doc with proper file titles as clickable links.
"""
import json
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

with open(r"C:\Users\harisathwik\AppData\Local\hermes\google_token.json") as f:
    token_data = json.load(f)

creds = Credentials.from_authorized_user_info(token_data)
service = build("docs", "v1", credentials=creds)

DOC_ID = "1nXDOg3Awxht1Nw7TXrwotbLMSw75by3jgxs0F9tk9GU"

# Get current doc to find content boundaries
doc = service.documents().get(documentId=DOC_ID).execute()
content = doc.get("body", {}).get("content", [])
last_end = 1
for elem in content:
    if "endIndex" in elem:
        last_end = max(last_end, elem["endIndex"])

# Step 1: Clear all content after index 1
requests = []
if last_end > 2:
    requests.append({
        "deleteContentRange": {
            "range": {
                "startIndex": 2,
                "endIndex": last_end - 1
            }
        }
    })

# Step 2: Insert all content with proper formatting
# Format: (text, style_dict)
blocks = [
    # Header
    ("Veerla Harisathwik — Artifacts & Documents\n", {"bold": True}),
    ("\n", {}),
    ("Portfolio: ", {"bold": True}),
    ("https://harisathwik.github.io/", {"link": {"url": "https://harisathwik.github.io/"}}),
    ("\n\n", {}),
    ("Status: ", {"bold": True}),
    ("Submitted\n", {"bold": True}),
    ("\n\n", {}),

    # Section 1: MLOps Project
    ("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", {}),
    ("\n", {}),
    ("MLOps Project — Telco Customer Churn Prediction\n", {"bold": True}),
    ("\n", {}),
    ("Algorithm Writeup\n", {"bold": True}),
    ("https://drive.google.com/file/d/17yoq6WDuvEG2_T6zjYmjYfnx0hnwpKDD/view?usp=drivesdk\n",
     {"link": {"url": "https://drive.google.com/file/d/17yoq6WDuvEG2_T6zjYmjYfnx0hnwpKDD/view?usp=drivesdk"}}),
    ("\n", {}),
    ("System Design & Architecture\n", {"bold": True}),
    ("https://drive.google.com/file/d/15HXfZNtgIuXFkGsbI04PtLIUebvMYMK-/view?usp=drivesdk\n",
     {"link": {"url": "https://drive.google.com/file/d/15HXfZNtgIuXFkGsbI04PtLIUebvMYMK-/view?usp=drivesdk"}}),
    ("\n", {}),
    ("Project Repo: ", {"bold": True}),
    ("D:\\Sathwik\\Ayush\\MLOps-Github\\\n", {}),
    ("\n\n", {}),

    # Section 2: Personal Voice & Brand
    ("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", {}),
    ("\n", {}),
    ("Personal Voice & Brand\n", {"bold": True}),
    ("\n", {}),
    ("Customised Master Plan\n", {"bold": True}),
    ("https://drive.google.com/file/d/1FWA1j-6fPJWH_W74cua7ELvp7uIBvXgC/view?usp=drivesdk\n",
     {"link": {"url": "https://drive.google.com/file/d/1FWA1j-6fPJWH_W74cua7ELvp7uIBvXgC/view?usp=drivesdk"}}),
    ("\n\n", {}),

    # Section 3: Outreach & Business Development
    ("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", {}),
    ("\n", {}),
    ("Outreach & Business Development\n", {"bold": True}),
    ("\n", {}),
    ("Outreach Message Template\n", {"bold": True}),
    ("https://drive.google.com/file/d/1ilmGwl3tD2DzjzZM2gTmluIFjXsLY586/view?usp=drivesdk\n",
     {"link": {"url": "https://drive.google.com/file/d/1ilmGwl3tD2DzjzZM2gTmluIFjXsLY586/view?usp=drivesdk"}}),
    ("\n\n", {}),

    # Section 4: Thought Leadership
    ("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", {}),
    ("\n", {}),
    ("Thought Leadership\n", {"bold": True}),
    ("\n", {}),
    ("Why 91% of AI Agents Fail in Production (And What the 9% Do Differently)\n", {"bold": True}),
    ("https://drive.google.com/file/d/1YYdmRtPX1nqVxRyjHXZhJ8CFnhV3V6on/view?usp=drivesdk\n",
     {"link": {"url": "https://drive.google.com/file/d/1YYdmRtPX1nqVxRyjHXZhJ8CFnhV3V6on/view?usp=drivesdk"}}),
    ("\n", {}),
]

# Build requests
current_index = 2
for text, style in blocks:
    if not text:
        continue
    requests.append({
        "insertText": {
            "location": {"index": current_index},
            "text": text
        }
    })
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

print(f"Doc rewritten! {len(requests)} requests executed.")
print(f"View: https://docs.google.com/document/d/{DOC_ID}/edit?usp=sharing")
