
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st

model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5_paraphraser")
tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5_paraphraser")

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write("device ",device)
model = model.to(device)

# Beam Search
st.title('Automatic Paraphraser')
context = "Accurate, “realistic information” in marketing was important in these early conceptions because the opinion leader was assumed to transmit marketing messages more or less faithfully, without substantially altering them or having them altered by ongoing communications with other consumers"
text = "paraphrase: "+context + " </s>"

encoding = tokenizer.encode_plus(text,max_length =1000, padding=True, return_tensors="pt")
input_ids,attention_mask  = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

model.eval()
beam_outputs = model.generate(
    input_ids=input_ids,attention_mask=attention_mask,
    max_length=1000,
    early_stopping=True,
    num_beams=15,
    num_return_sequences=3

)


for beam_output in beam_outputs:
    sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
    st.write(sent)

