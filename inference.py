
from unsloth import FastLanguageModel
import torch
import pandas as pd
from datasets import Dataset

max_seq_length = 6000  # 10 000 tokens ~ 20 000 char max
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
# Use 4bit quantization to reduce memory usage. Can be False.
load_in_4bit = True
peft_model_id = "lora_model"
model, tokenizer = FastLanguageModel.from_pretrained(peft_model_id,
                                                     max_seq_length=max_seq_length,
                                                     dtype=None,
                                                     load_in_4bit=load_in_4bit)


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = ""  # tokenizer.eos_token # Must add EOS_TOKEN


def formatting_prompts_func_test(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    texts = []
    for instruction, input in zip(instructions, inputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, "")
        texts.append(text)
    return {"text": texts, }


def load_test_data():
    path = "data/data/test/test.txt"

    # load description
    descriptions = []

    with open(path, 'r') as f:
        for line in f:
            description = line.strip()
            descriptions.append(description)
    # to df
    instructions = [
        "Generate a list of all node connections and their relationships based on the given graph properties."] * len(descriptions)
    df = pd.DataFrame({'instruction': instructions, 'input': descriptions})
    return df


df = load_test_data()
dataset = Dataset.from_pandas(df)
dataset = dataset.map(formatting_prompts_func_test, batched=True,)


FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

for i in range(len(dataset)):
    inputs = tokenizer(dataset[i]["text"], return_tensors="pt").to("cuda")
    output = model.generate(**inputs, use_cache=True)
    with open(f"/pred/{i}.txt", "w") as f:
        f.write(tokenizer.batch_decode(
            output[:, inputs["input_ids"].shape[1]:-1])[0])
