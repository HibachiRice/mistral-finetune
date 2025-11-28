from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_dir = "/home/zhoulab/Downloads/mtllm/mistral"

tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

prompt = "You are an expert in microbial metagenomics and microbial traits. You are tasked with answering the question provided by the user. All information required to answer the question will be provided to you in a large text format as a list. Each entry in the list will contain an article summary, a grade for how well the article answers the user question, and the citation for the article in apa format. Your answer should be a detailed paragraph consisting of 5–10 sentences answering the user question. Additionally, when writing your response, every source you use should be cited in an in-text format within your response in apa format. You should never have a references section at the end of your response. If you cannot cite the given information in apa format, please instead list the article title. You should prioritize using the articles with a higher grade over the ones with a lower grade. You must answer the user question to the best of your ability without using any other information besides the information provided to you. The Question: What are some bacterial strains associated with ear infections?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=1600)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
