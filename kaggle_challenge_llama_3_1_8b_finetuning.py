# -*- coding: utf-8 -*-
"""This script is used to fine-tune the Llama 3.1 8B model on the Alpaca dataset for the Kaggle competition. The script is based on the [Llama 3.1 8B finetuning notebook](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-bnb-4bit/blob/main/Llama_3_1_8B_finetuning.ipynjson) by [unsloth](https://huggingface.co/unsloth)."""


# Create .kaggle directory
from datasets import Dataset
from unsloth import FastLanguageModel
import os
import pandas as pd
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
os.makedirs('/root/.kaggle', exist_ok=True)

# # Move kaggle.json to the .kaggle directory
# !mv kaggle.json /root/.kaggle/

# # Set permissions
# !chmod 600 /root/.kaggle/kaggle.json

# !kaggle competitions download -c generating-graphs-with-specified-properties
# !unzip -q generating-graphs-with-specified-properties.zip

# Get the max_seq_length we need


max_seq_length = 6000  # 10 000 tokens ~ 20 000 char max
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
# Use 4bit quantization to reduce memory usage. Can be False.
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",  # Change to non-intruct
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

os.getcwd()


def load_data(train=True):
    path = "data/data/valid"
    if train:
        path = "data/data/train/"

    # load description (listdir)
    descriptions = []
    graphs = []

    for description in os.listdir(os.path.join(path, "description")):
        with open(os.path.join(path, "description", description), 'r') as f:
            descriptions.append(f.read())

    for graph in os.listdir(os.path.join(path, "graph")):
        with open(os.path.join(path, "graph", graph), 'r') as f:
            graphs.append(f.read())

    # to df
    instructions = [
        "Generate a list of all node connections and their relationships based on the given graph properties."] * len(descriptions)
    df = pd.DataFrame({'instruction': instructions,
                      'input': descriptions, 'output': graphs})
    return df


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts, }


pass

df = load_data(False)
# to dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.map(formatting_prompts_func, batched=True,)


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1,  # Set this for 1 full training run.
        # max_steps = 60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

trainer_stats = trainer.train()

"""## Save Model"""
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
