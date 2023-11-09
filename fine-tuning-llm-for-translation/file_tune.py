import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

absolute_path = os.path.dirname(os.path.abspath(__file__))
model_id = 'llm-jp/llm-jp-13b-v1.0'
model_name = 'llmjp'
# model_id = "stabilityai/japanese-stablelm-base-ja_vocab-beta-7b"
# model_name = 'stablelm'

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    f'{absolute_path}/{model_name}-tokenizer',
    local_files_only=True
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    f'{absolute_path}/{model_name}-model',
    local_files_only=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
)

print("Encoding input...")
text = "自然言語処理とは何か"
device = torch.device("mps")
model = model.to(device)
model.half()
model.eval()

input_ids = tokenizer.encode(
    text,
    add_special_tokens=False,
    return_tensors="pt"
)

print("Generating output...")
output = model.generate(
    input_ids,
    # temperature=0.7,
    early_stopping=True,
    max_new_tokens=100,
    num_beams=5,
    # top_p=0.95,
    # do_sample=True,
)

print("Decoding output...")
out = tokenizer.decode(
    output[0],
    skip_special_tokens=True
)
print(out)

