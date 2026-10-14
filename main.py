"""
Now extremely deprecated. Code added below is from May-June 2026
"""
# 1. Install pip upgrade
!pip install --upgrade pip

# 1. FIXED INSTALLATION
# Thoroughly uninstall unsloth and all related packages to ensure a clean slate.
# This prevents potential conflicts from previous installations, including PyTorch.
!pip uninstall -y unsloth unsloth_zoo transformers datasets trl peft accelerate bitsandbytes xformers torch torchvision torchaudio

# Then, install unsloth with its recommended Colab configuration.
# This variant ensures unsloth handles its own PyTorch, xformers, and bitsandbytes dependencies optimally for Colab.
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install other common libraries if they are not explicitly pulled by unsloth[colab-new]
# Explicitly install torchvision as it seems to be an implicit dependency for unsloth_zoo.
!pip install trl peft accelerate bitsandbytes datasets torchvision

# Standard imports
import torch
import requests
import re
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from datasets import Dataset


# ==========================================
# 2. DATA ACQUISITION & DEEP CLEANING
# ==========================================
urls = [
    "https://www.gutenberg.org/files/98/98-0.txt", #A Tale of Two Cities
    "https://www.gutenberg.org/files/1400/1400-0.txt", #Great Expectations
    "https://www.gutenberg.org/files/46/46-0.txt", #A Christmas Carol
    "https://www.gutenberg.org/files/1023/1023-0.txt", #Bleak House
    "https://www.gutenberg.org/files/730/730-0.txt", #Oliver Twist
    "https://www.gutenberg.org/files/766/766-0.txt", #David Copperfield
    "https://www.gutenberg.org/files/786/786-0.txt", #Hard Times
    "https://www.gutenberg.org/files/564/564-0.txt", #The Mystery of Edwin Drood
    "https://www.gutenberg.org/files/345/345-0.txt", #Dracula
    "https://www.gutenberg.org/files/110/110-0.txt", #Tess of the D'Urbervilles
    "https://www.gutenberg.org/files/107/107-0.txt", #Far From the Madding Crowd
    "https://www.gutenberg.org/files/174/174-0.txt", #The Picture of Dorian Gray
    "https://www.gutenberg.org/files/2701/2701-0.txt", #Moby Dick
    "https://www.gutenberg.org/files/43/43-0.txt", #The Strange case of Dr. Jekyll and Mr. Hyde
    "https://www.gutenberg.org/files/1661/1661-0.txt", #The Adventures of Sherlock Holmes
    "https://www.gutenberg.org/files/76/76-0.txt", #Adventures of Huckleberry Finn
    "https://www.gutenberg.org/files/74/74-0.txt", #The Adventures of Tom Sawyer
    "https://www.gutenberg.org/files/25344/25344-0.txt", #The Scarlett Letter
    "https://www.gutenberg.org/files/244/244-0.txt", #A Study in Scarlett
    "https://www.gutenberg.org/files/219/219-0.txt", #Heart of Darkness
    "https://www.gutenberg.org/files/120/120-0.txt", #Treasure Island
    "https://www.gutenberg.org/files/1998/1998-0.txt", #Thus Spake Zarathustra:..
    "https://www.gutenberg.org/files/209/209-0.txt", #The Turn of the Screw
    "https://www.gutenberg.org/files/208/208-0.txt", #Daisy Miller
    "https://www.gutenberg.org/files/2097/2097-0.txt", #The Sign of the Four
    "https://www.gutenberg.org/files/35/35-0.txt", #The Time Machine
    "https://www.gutenberg.org/files/36/36-0.txt", #The War of the Worlds
    "https://www.gutenberg.org/files/599/599-0.txt", #Vanity Fair


]

def clean_classic_text(text):
    start_markers = ["*** START OF THIS PROJECT", "*** START OF THE PROJECT"]
    end_markers = ["*** END OF THIS PROJECT", "*** END OF THE PROJECT"]
    start_idx = 0
    for m in start_markers:
        found = text.find(m)
        if found != -1:
            start_idx = text.find("\n", found) + 1
            break
    end_idx = len(text)
    for m in end_markers:
        found = text.find(m)
        if found != -1:
            end_idx = found
            break
    clean = text[start_idx:end_idx]
    clean = re.sub(r'CHAPTER\s+[IVXLCDM]+\.?', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'Chapter\s+\d+\.?', '', clean)
    clean = re.sub(r'\n\s*\n', '\n\n', clean)
    return clean.strip()

print("Downloading the Full Library...")
full_library_text = ""
for url in urls:
    r = requests.get(url)
    r.encoding = 'utf-8-sig'
    full_library_text += clean_classic_text(r.text) + "\n\n"

# ==========================================
# 3. LOAD LLAMA-3 & PREPARE
# ==========================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Higher rank to capture more complex stylistic nuances
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
)

def format_prompt(text_chunk):
    # This instruction is now optimized for the "Classic Male Lit" persona
    return {
        "text": f"### Instruction:\nCompose a narrative that embodies the vocabulary, depth, awareness, sentence structure, and philosophical reflection of 19th-century male literary classics, using the styles characterized by ornate vocabulary, winding sentences, and striking introspection.\n\n### Input:\n{text_chunk[:200]}\n\n### Response:\n{text_chunk[200:]}"
    }

chunks = [full_library_text[i:i + 1100] for i in range(0, len(full_library_text), 1100)]
dataset = Dataset.from_dict({"raw_text": chunks})
dataset = dataset.map(lambda x: format_prompt(x["raw_text"]), remove_columns=["raw_text"])

# ==========================================
# 4. MONITORING CALLBACK (The Progress Mirror)
# ==========================================
class ProgressMonitor(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0 and state.global_step > 0:
            print(f"\n--- [LITERARY CHECK-IN: STEP {state.global_step}] ---")
            FastLanguageModel.for_inference(model)
            test_prompt = "The drawing room was silent, save for the crackle of the fire and the realization that"
            inputs = tokenizer([f"### Instruction:\nCompose a narrative that embodies the vocabulary, depth, awareness, sentence structure, and philosophical reflection of 19th-century male literary classics, using the styles characterized by ornate vocabulary, winding sentences, and striking introspection.  \n\n### Input:\n{test_prompt}\n\n### Response:\n"], return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=60, temperature=0.8)
            print(test_prompt,tokenizer.decode(outputs[0], skip_special_tokens=True).split("### Response:\n")[-1])
            print("-" * 50)
            model.train() # Resume learning

# ==========================================
# 5. OPTIMIZED TRAINING (3 Epochs)
# ==========================================
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    callbacks = [ProgressMonitor()],
    args = TrainingArguments(
        num_train_epochs = 3,         # Reads the library 3 times for deep stylistic immersion
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,           # Slowly ramps up learning to protect the model's "brain"
        learning_rate = 1e-4,         # The sweet spot for fine-tuning 8B models
        lr_scheduler_type = "cosine", # Smoothly winds down training
        weight_decay = 0.01,
        fp16 = not torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        output_dir = "classics_model_final",
    ),
)

trainer.train()

#GENERATION CELL
FastLanguageModel.for_inference(model)
test_prompt = input("")
inputs = tokenizer([f"### Instruction:\nCompose a cohesive, flowing narrative that embodies the wit, social precision, vocabulary, and emotional depth of 19th-century female literary classics, using the styles characterized by ornate vocabulary and winding sentences and paying close attention to logical sequences of events in the narrative.\n\n### Input:\n{test_prompt}\n\n### Response:\n"], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.55)
print(test_prompt)
print(tokenizer.decode(outputs[0], skip_special_tokens=True).split("### Response:\n")[-1])
print("-" * 50)
