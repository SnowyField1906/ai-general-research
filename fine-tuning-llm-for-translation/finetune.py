import transformers
import os
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import json

absolute_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(absolute_path, 'llm-jp-13b-v1.0')
token_path = os.path.join(absolute_path, 'llm-jp-13b-v1.0', 'tokenizer.json')
model_id = 'llm-jp/llm-jp-13b-v1.0'

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, local_files_only=True)

text = "自然言語処理とは何か"
device = torch.device("mps")
model = model.to(device)
model.eval()

tokenized_input = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
tokenized_input = tokenized_input.to(device)

with torch.no_grad():
    output = model.generate(
        tokenized_input,
        num_beams=5,
        early_stopping=True,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.95,
        temperature=0.7
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))

