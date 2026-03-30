import torch
import joblib
import torch.nn.functional as F
from reportlab.lib.styles import getSampleStyleSheet
from model_definition import HybridInLegal, INLEGAL
from transformers import AutoTokenizer


# -------------------------------
# STEP 1: DEVICE
# -------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# STEP 2: LOAD FILES
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(INLEGAL)
le_cat = joblib.load("le_cat.pkl")
le_section = joblib.load("le_section.pkl")

# -------------------------------
# STEP 3: LOAD MODEL
# -------------------------------
model = HybridInLegal(transformer_name=INLEGAL)

model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -------------------------------
# STEP 4: USER INPUT
# -------------------------------
print("\nENTER THE COMPLAINT BELOW:")


# -------------------------------
# STEP 5: TOKENIZE INPUT
# -------------------------------
inputs = tokenizer(
    complaint_text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=512
)

inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

# -------------------------------
# STEP 6: MODEL PREDICTION
# -------------------------------
with torch.no_grad():
    logits_cat, logits_sec = model(**inputs)

# -------------------------------
# STEP 7: DECODE CATEGORY
# -------------------------------
cat_id = torch.argmax(logits_cat, dim=1).item()
predicted_category = le_cat.inverse_transform([cat_id])[0]

# -------------------------------
# STEP 8: TOP-3 BNS SECTIONS
# -------------------------------
top_k = 3
sec_probs = F.softmax(logits_sec, dim=1)

top_sec_ids = torch.topk(logits_sec, top_k, dim=1).indices[0].tolist()
top_sections = le_section.inverse_transform(top_sec_ids)
top_confidence = sec_probs[0][top_sec_ids].tolist()

# -------------------------------
# STEP 9: SHOW OUTPUT IN TERMINAL
# -------------------------------
print("\nPREDICTED CATEGORY:", predicted_category)
print("\nPREDICTED BNS SECTIONS:")
for sec, prob in zip(top_sections, top_confidence):
    print(f"Section {sec}  →  Confidence: {prob:.2f}")

# -------------------------------
# STEP 10: FIR TEXT
# -------------------------------
fir_text = f"""
FIRST INFORMATION REPORT (FIR)

Based on the complaint provided by the informant, the nature of the offence
falls under the category "{predicted_category}".

Upon preliminary legal analysis using an AI-based decision support system,
the incident attracts the following provisions of the Bharatiya Nyaya
Sanhita (BNS):

Sections: {", ".join(top_sections)}

Accordingly, FIR may be registered under the above-mentioned sections
for further investigation as per law.
"""

# -------------------------------
# STEP 11: GENERATE PDF
# -------------------------------
pdf_file = "Generated_FIR.pdf"
doc = SimpleDocTemplate(pdf_file)
styles = getSampleStyleSheet()
story = []

for line in fir_text.split("\n"):
    story.append(Paragraph(line, styles["Normal"]))

doc.build(story)

print("\nFIR PDF GENERATED SUCCESSFULLY")
print("Saved as:", pdf_file)
