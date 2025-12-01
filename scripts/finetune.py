import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# --- 1. Configuration ---
MODEL_NAME = "./mistral"
OUTPUT_DIR = "./adapters/MicroTraitLLM-LoRA" # This is where the adapter will be created
DATA_PATH = "./data/train_data.json"         # Your 15 training questions

# Quantization Config (For RTX 4090 memory efficiency)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# --- 2. Load Base Model & Tokenizer ---
print("Loading Base Model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False # Disable cache for training
)
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 3. LoRA Configuration (The "Adapter" Settings) ---
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,  # Rank: Higher = smarter but more VRAM. 64 is good for 4090.
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", # Attention layers
        "gate_proj", "up_proj", "down_proj"     # MLP layers
    ],
)

# --- 4. Data Formatting (The "Teacher") ---
# This forces the model to learn your specific prompt structure
def formatting_prompts_func(examples):
    output_texts = []
    
    # System Prompt from your report
    system_prompt = (
        "You are an expert in microbial metagenomics and microbial traits. "
        "You are tasked with answering the question provided by the user using only the provided list of articles. "
        "Prioritize high-grade articles. Cite every source in-text (APA). "
        "Do NOT include a references section at the end."
    )

    for i in range(len(examples['question'])):
        context = examples['context_list'][i]
        question = examples['question'][i]
        answer = examples['answer'][i]

        # The Prompt Structure
        # Note the </s> at the end. This is the "STOP" signal.
        # It teaches the model: "After the answer, STOP WRITING immediately."
        text = f"<s>[INST] {system_prompt}\n\nContext:\n{context}\n\nQuestion: {question} [/INST] {answer} </s>"
        output_texts.append(text)
        
    return output_texts

# Load Dataset
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# --- 5. Training Arguments ---
training_arguments = TrainingArguments(
    output_dir="./results",       # Temporary checkpoints
    num_train_epochs=3,           # How many times to loop over the 15 questions
    per_device_train_batch_size=4,# Batch size for RTX 4090
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=5,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True,                    # Use BFloat16 for RTX 40xx series
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)

# --- 6. The Trainer (The "Oven") ---
print("Starting Training...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
    formatting_func=formatting_prompts_func,
)

# --- 7. Run Training & Save Adapter ---
trainer.train()

# This specific command generates the "Adapter" files you need
print(f"Saving adapter to {OUTPUT_DIR}...")
trainer.model.save_pretrained(OUTPUT_DIR)
print("Done! Adapter created.")