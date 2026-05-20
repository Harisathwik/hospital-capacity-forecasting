"""
Update the Google Doc to add the System Design section.
"""
import json
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

with open(r"C:\Users\harisathwik\AppData\Local\hermes\google_token.json") as f:
    token_data = json.load(f)

creds = Credentials.from_authorized_user_info(token_data)
service = build("docs", "v1", credentials=creds)

DOC_ID = "1nXDOg3Awxht1Nw7TXrwotbLMSw75by3jgxs0F9tk9GU"

doc = service.documents().get(documentId=DOC_ID).execute()
content = doc.get("body", {}).get("content", [])
last_end = 1
for elem in content:
    if "endIndex" in elem:
        last_end = max(last_end, elem["endIndex"])

blocks = [
    ("\n", {}),
    ("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", {}),
    ("\n", {}),
    ("System Design & Architecture\n", {"bold": True}),
    ("\n", {}),
    ("System Design Doc: ", {"bold": True}),
    ("https://drive.google.com/file/d/15HXfZNtgIuXFkGsbI04PtLIUebvMYMK-/view?usp=drivesdk\n",
     {"link": {"url": "https://drive.google.com/file/d/15HXfZNtgIuXFkGsbI04PtLIUebvMYMK-/view?usp=drivesdk"}}),
    ("\n", {}),
]

requests = []
current_index = last_end - 1

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

result = service.documents().batchUpdate(
    documentId=DOC_ID,
    body={"requests": requests}
).execute()

print(f"Doc updated! {len(requests)} requests executed.")
