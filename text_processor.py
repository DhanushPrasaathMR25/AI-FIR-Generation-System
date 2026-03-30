from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def professional_rephrase(text):

    prompt = f"""
You are a legal assistant helping police officers write FIR complaints.

Rewrite the following complaint into clear and professional police report language.

STRICT RULES:
- Do NOT summarize.
- Do NOT remove any information.
- Preserve all names, locations, dates, times, numbers, and stolen items.
- Keep the same meaning.
- Only improve grammar and convert to formal legal style.

Example:

Input:
sir my bike missing yesterday near bus stand

Output:
The complainant reported that his motorcycle went missing near the bus stand yesterday.

Now rewrite the complaint.

Complaint:
{text}

Professional Complaint:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.2
        )

    rewritten = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return rewritten.strip()