from transformers import DataCollatorForLanguageModeling, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import os
import random
import argparse

from continual_pretraining_utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description="Run classification experiments with different models and output directories.")
parser.add_argument("--model_name", type=str, help="Path to the pretrained model or model name.")
parser.add_argument("--input_file", type=str, help="File for pretraining data.")
parser.add_argument("--validation_file", type=str, help="File for validation data.")
parser.add_argument("--output_dir", type=str, help="Directory to save results and logs.")
parser.add_argument("--sequence_length", type=int, help="Max sequence length when preprocessing data.")
parser.add_argument("--data_preprocess_type", type=str, default="sp")
parser.add_argument("--repetition_ratio", type=float, default=0.3)
parser.add_argument("--bin_extra_capacity", type=int, default=0)

args, _ = parser.parse_known_args()

if args.model_name:
    model_name = args.model_name
else:
    model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None and "llama" in model_name:
    tokenizer.pad_token = "<|reserved_special_token_0|>"
    print(tokenizer.pad_token_id)

if args.input_file:
    input_file = args.input_file
else:
    input_file = "data/toy_train.txt"
if args.validation_file:
    validation_file = args.validation_file
else:
    validation_file = "data/toy_validation.txt"
with open(input_file, "r", encoding="utf-8") as f:
    texts = f.readlines()
with open(validation_file, "r", encoding="utf-8") as f:
    validation_texts = f.readlines()

if args.output_dir:
    output_dir = args.output_dir
else:
    output_dir = "models/Llama-3.2-1B_sp"
os.makedirs(output_dir, exist_ok=True)

if args.sequence_length:
    sequence_length = args.sequence_length
else:
    sequence_length = 512

print("Tokenizing and chunking...")
padding = False
if args.data_preprocess_type == "sp":
    chunks = tokenize_and_chunk_by_seamless_packing(
        texts,
        tokenizer,
        sequence_length,
        repetition=args.repetition_ratio,
        bin_extra_capacity=args.bin_extra_capacity,
        padding=False
    )
    validation_chunks = tokenize_and_chunk_by_seamless_packing(
        validation_texts, 
        tokenizer, 
        sequence_length, 
        repetition=args.repetition_ratio,
        bin_extra_capacity=args.bin_extra_capacity,
        padding=False
    )

random.shuffle(chunks)
dataset = ChunkedDataset(chunks, tokenizer, sequence_length)
random.shuffle(validation_chunks)
validation_dataset = ChunkedDataset(validation_chunks, tokenizer, sequence_length)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_strategy="epoch",
    save_total_limit=3,
    logging_dir=f"{output_dir}/logs",
    evaluation_strategy="epoch",
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
    fp16=True,
    deepspeed="config/ds_config_zero1_hf.json",
)

model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.cuda()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
callback = SaveMetricsCallback(output_dir=output_dir, training_args=training_args)
trainer.add_callback(callback)

trainer.train()

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)