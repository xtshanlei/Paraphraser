
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

st.subheader('Original sentence:')
context = st.text_input('Please type the sentence you want to paraphrase here')

@cache
def para(context):
        text = "paraphrase: "+context + " </s>"

        encoding = tokenizer.encode_plus(text,max_length =1000, padding=True, return_tensors="pt")
        input_ids,attention_mask  = encoding["input_ids"].to(device),encoding["attention_mask"].to(device)

        model.eval()
        beam_outputs = model.generate(
            input_ids=input_ids,attention_mask=attention_mask,
            max_length=1000,
            early_stopping=True,
            num_beams=15,
            num_return_sequences=3

        )
        return beam_outputs
if context:
        beam_outputs = para(context)

        st.subheader('Results:')
        for beam_output in beam_outputs:
            sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
            st.write(sent)

