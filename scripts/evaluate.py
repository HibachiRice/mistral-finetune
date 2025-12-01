import torch
import json
import re
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

# --- 1. Configuration ---
BASE_MODEL = "/home/zhoulab/Downloads/mtllm/mistral"
ADAPTER_PATH = "./adapters/MicroTraitLLM-LoRA"
TEST_DATA_PATH = "./data/test_data.json"

# --- 2. Load Model & Adapter ---
print(f"Loading base model from {BASE_MODEL}...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 3. Evaluation Functions ---

def check_citations(text):
    """
    Checks if the text contains APA style citations.
    UPDATED: Now allows '&', '.', and '-' inside names.
    """
    # Allows "Smith & Jones", "St. John", "O'Neil", etc.
    citation_pattern = r"\([A-Za-z\s&\.-]+(?:et al\.)?,?\s\d{4}\)"
    return len(re.findall(citation_pattern, text)) > 0

def check_structure(text):
    forbidden_headers = ["References:", "Bibliography:", "Works Cited:", "Sources:"]
    ending_text = text[-300:]
    for header in forbidden_headers:
        if header in ending_text:
            return False
    return True

def calculate_entity_recall(generated_text, required_entities):
    if not required_entities:
        return 1.0 
    found_count = 0
    generated_lower = generated_text.lower()
    for entity in required_entities:
        if entity.lower() in generated_lower:
            found_count += 1
    return found_count / len(required_entities)

# --- 4. Main Loop ---

with open(TEST_DATA_PATH, "r") as f:
    test_data = json.load(f)

print(f"\nStarting evaluation on {len(test_data)} questions...\n")
print("="*60)

total_recall = 0
total_citations = 0
total_clean_structure = 0

for i, entry in enumerate(test_data):
    question = entry['question']
    context = entry['context_list']
    required_entities = entry.get('required_entities', [])

    system_prompt = (
        "You are an expert in microbial metagenomics and microbial traits. "
        "You are tasked with answering the question provided by the user using only the provided list of articles. "
        "Prioritize high-grade articles. Cite every source in-text (APA). "
        "Do NOT include a references section at the end."
    )
    
    formatted_prompt = f"<s>[INST] {system_prompt}\n\nContext:\n{context}\n\nQuestion: {question} [/INST]"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_tokens = model.generate(
            **inputs, 
            max_new_tokens=300, 
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(output_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    # --- NEW CLEANER: REMOVE 'INST' SPAM ---
    # This chops off the text if the model starts hallucinating a new instruction tag
    if "INST" in generated_text:
        generated_text = generated_text.split("INST")[0].strip()
        # Also clean up any trailing bracket if it got cut off weirdly
        if generated_text.endswith("["):
            generated_text = generated_text[:-1].strip()

    # --- Scoring ---
    recall = calculate_entity_recall(generated_text, required_entities)
    has_citations = check_citations(generated_text)
    is_clean = check_structure(generated_text)
    
    total_recall += recall
    if has_citations: total_citations += 1
    if is_clean: total_clean_structure += 1

    print(f"Q{i+1}: {question}")
    print("-" * 20)
    print(f"FULL MODEL OUTPUT:\n{generated_text}")
    print("-" * 20)
    print(f" > Entities Found: {int(recall * len(required_entities))}/{len(required_entities)}")
    print(f" > Citations: {'✅ Found' if has_citations else '❌ Missing'}")
    print(f" > No Footer: {'✅ Clean' if is_clean else '❌ Failed (Footer detected)'}")
    print("=" * 60)

# --- 5. Final Report ---
num_q = len(test_data)
print("\n=== FINAL EVALUATION REPORT ===")
print(f"1. Entity Recall (Accuracy):   {total_recall / num_q:.1%}")
print(f"2. Citation Usage:             {total_citations / num_q:.1%}")
print(f"3. Structural Adherence:       {total_clean_structure / num_q:.1%}")
print("="*60)
