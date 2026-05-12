# -*- coding: utf-8 -*-
import os
import sys
import argparse
import json
import random
from pathlib import Path
import gc
import torch
import logging
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig, TaskType
import re

from modeling_llada import LLaDAModelLM, LLaDAConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from transformers import AutoTokenizer, TrainingArguments, Trainer, default_data_collator
from peft import LoraConfig, get_peft_model
from modeling_llada import LLaDAModelLM
from datasets import Dataset
from tqdm import tqdm 
from accelerate import Accelerator

from generate_parallel import generate_parallel

import torch
import flash_attn
import numpy as np


from generate_parallel_new import generate_parallel
from dllm.core.trainers.mdlm import MDLMTrainer, StructuredMDLMTrainer

from dllm.core.schedulers import BaseAlphaScheduler, LinearAlphaScheduler
from generate_structure import generate_sns_global


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./finetuned_llada")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--block_length", type=int, default=64)
    parser.add_argument("--grad_acc_steps", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--log_interval", type=int, default=50)
    return parser.parse_args()
    
    
from glob import glob

def load_json_folder(file_path, max_samples=None):
    if os.path.isfile(file_path):
        files = [file_path]
    else:
        raise FileNotFoundError(f"Dataset path does not exist or is not a file: {file_path}")

    if len(files) == 0:
        raise FileNotFoundError(f"No valid files found in {file_path}")
    
    logger.info(f"Found {len(files)} file in {file_path}")
    all_records = []

    for fp in files:
        logger.info(f"Processing file: {fp}")
        try:
            if fp.endswith('.jsonl'):
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line: 
                            try:
                                all_records.append(json.loads(line))
                            except json.JSONDecodeError:
                                logger.warning(f"Skipping invalid JSON line in {fp}")
            else:
                raise ValueError(f"Unsupported file format for {fp}, only '.jsonl' is supported.")
        except Exception as e:
            logger.error(f"Error processing file {fp}: {str(e)}")

    if not all_records:
        raise ValueError(f"No valid data found in the file: {file_path}")

    if max_samples:
        all_records = random.sample(all_records, min(max_samples, len(all_records)))

    logger.info(f"Total samples: {len(all_records)}")
    return Dataset.from_list(all_records)

def prepare_examples(raw, tokenizer, pad_token_id, mask_len_per_step=128, mask_id=126336, device="cpu"):
    examples = []
    for idex, ex in enumerate(tqdm(raw, desc="Preparing examples")):
        input_ids, step_list, semantic_block_lengths, prompt_len, detailed_range, clean_input, draft_mask_pos, detail_mask_pos = _tokenize_with_mask(
            ex, tokenizer, pad_token_id, mask_id, mask_len_per_step
        )
        clean_input_list = clean_input.tolist() if isinstance(clean_input, torch.Tensor) else list(clean_input)

        detailed_range_list = [[int(a), int(b)] for a, b in detailed_range]
        draft_mask_pos_list = [[int(a), int(b)] for a, b in draft_mask_pos]
        detail_mask_pos_list = [[int(a), int(b)] for a, b in detail_mask_pos]

        examples.append({
            "input_ids": clean_input_list,
            "steps": step_list,
            "num_step_count": len(step_list),
            "mask_pos": detailed_range_list,
            "draft_mask_pos": draft_mask_pos_list,      # 新增
            "detail_mask_pos": detail_mask_pos_list,    
            "prompt_len": prompt_len,
            "detailed_process": ex.get("detailed_process", ""),
            "detailed_range": detailed_range_list,
        })
    return examples

def _tokenize_with_mask(entry, tokenizer, pad_token_id, mask_token_id, mask_len_per_step=128):
    import re
    import torch

    prompt_text = (
        "You are an assistant that solves problems by writing a step-by-step detailed reasoning.\n"
        "First, plan the necessary steps based on the question, then expand each step into a "
        "complete reasoning paragraph that explicitly explains the logical calculations.\n\n"
        "Task Requirements:\n"
        "- Identify the key steps needed to solve the problem and order them logically.\n"
        "- For each step, provide a clear heading (e.g., 'Step 1 - ...') followed by detailed reasoning.\n"
        "- Each paragraph must explicitly show the calculations and reasoning for that step.\n"
        "- ONLY output the reasoning; do NOT repeat the question.\n"
        "- Keep reasoning concise, coherent, and strictly focused on solving the question.\n"
        "- Before giving the final answer, verify your calculations for correctness.\n"
        "- Finally, output the numeric answer on a new line in this exact format:\n"
        "    #### <number>\n"
        "- Do not output anything after the final numeric answer."
    )

    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0]
    input_ids = prompt_ids.clone()
    semantic_block_length = []

    if isinstance(entry, dict):
        question = entry.get("question", "")
        steps_list = entry.get("steps", [])  # ?? 保
        detailed_process = entry.get("detailed_process", "")
    else:
        raise ValueError("Unsupported entry type")

    if question.strip():
        question_text = f"Question:{question}\n"
        question_ids = tokenizer(question_text, return_tensors="pt").input_ids[0]
        input_ids = torch.cat([input_ids, question_ids], dim=0)

    prompt_len = len(input_ids)

    x = input_ids.clone()
    detailed_range = []
    draft_mask_pos = []
    detail_mask_pos = []

    if detailed_process.strip():
        prefix_text = "Detailed Explanation:\n\n"
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
        x = torch.cat([x, torch.tensor(prefix_ids)], dim=0)

        encoding = tokenizer(
            detailed_process,
            return_offsets_mapping=True,
            add_special_tokens=False,
            return_tensors=None
        )

        detailed_ids = torch.tensor(encoding["input_ids"])
        offsets = encoding["offset_mapping"]

        x = torch.cat([x, detailed_ids], dim=0)

        prefix_len = len(prefix_ids)
        offset_base = len(input_ids) + prefix_len

        text = detailed_process

        matches = list(re.finditer(r"Step\s*\d+\s*-", text))

        for i, m in enumerate(matches):
            start_char = m.start()

            if i + 1 < len(matches):
                end_char = matches[i + 1].start()
            else:
                end_char = len(text)

            # token span
            token_indices = [
                idx for idx, (s, e) in enumerate(offsets)
                if not (e <= start_char or s >= end_char)
            ]

            if not token_indices:
                continue

            S = offset_base + token_indices[0]
            E = offset_base + token_indices[-1] + 1

            detailed_range.append((S, E))

            # split header / detail
            colon_pos = text.find(":", start_char, end_char)

            if colon_pos != -1:
                header_tokens = [
                    idx for idx, (s, e) in enumerate(offsets)
                    if not (e <= start_char or s >= colon_pos)
                ]
                body_tokens = [
                    idx for idx, (s, e) in enumerate(offsets)
                    if not (e <= colon_pos or s >= end_char)
                ]

                if header_tokens:
                    draft_mask_pos.append((
                        offset_base + header_tokens[0],
                        offset_base + header_tokens[-1] + 1
                    ))

                if body_tokens:
                    detail_mask_pos.append((
                        offset_base + body_tokens[0],
                        offset_base + body_tokens[-1] + 1
                    ))

    # =========================
    # block mask（保持不变）
    # =========================
    for i, step_text in enumerate(steps_list):
        step_ids = tokenizer(step_text, return_tensors="pt").input_ids[0]
        mask_len = max(mask_len_per_step, len(step_ids))
        if i == len(steps_list) - 1:
            mask_len += mask_len_per_step

        mask_ids = torch.full((mask_len,), mask_token_id, dtype=torch.long)
        input_ids = torch.cat([input_ids, step_ids, mask_ids], dim=0)

        if not semantic_block_length:
            semantic_block_length.append(len(input_ids))
        else:
            semantic_block_length.append(len(step_ids) + mask_len)

    return input_ids, steps_list, semantic_block_length, prompt_len, detailed_range, x, draft_mask_pos, detail_mask_pos

def custom_collator(features, pad_token_id):
    input_ids_list = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    max_len = max(x.size(0) for x in input_ids_list)

    input_ids_padded = torch.stack([
        F.pad(x, (0, max_len - x.size(0)), value=pad_token_id) for x in input_ids_list
    ])
    labels = input_ids_padded.clone()
    labels[labels == pad_token_id] = -100
    attention_mask = (input_ids_padded != pad_token_id).long()

    return {
        "input_ids": input_ids_padded,
        "labels": labels,
        "attention_mask": attention_mask,
        "mask_pos": [f["mask_pos"] for f in features],
        "draft_mask_pos": [f["draft_mask_pos"] for f in features],      
        "detail_mask_pos": [f["detail_mask_pos"] for f in features],    
    }
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():

    args = parse_args()
    accelerator = Accelerator()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    
    # 1. tokenizer
    accelerator.print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    pad_token_id = tokenizer.convert_tokens_to_ids("<|reserved_token_0|>")
    #使用tokenizer中的空special token作为换行符
    print(f"Mask token id: {pad_token_id}")
    

    print("Tokenizer loaded.")
    print(tokenizer.mask_token_id)
    
    mask_token_id = tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
    tokenizer.mask_token_id = mask_token_id
    print(f"Mask token id: {mask_token_id}")


    # 2. load & prepare dataset
    accelerator.print(" Loading dataset...")
    raw = load_json_folder(args.data_path, max_samples=args.max_samples)
    print(f"The raw dataset has {len(raw)} examples.")
    dataset = prepare_examples(raw, tokenizer, pad_token_id, args.block_length, mask_token_id, device)
    #raw, tokenizer, pad_token_id, mask_len_per_step=128, mask_id=126336, device="cpu"

    from datasets import Dataset as HFDataset
    


    hf_dataset = HFDataset.from_list(dataset)   

    # after hf_dataset = HFDataset.from_list(dataset)
    print("HF Dataset column names:", hf_dataset.column_names)
    print("HF Dataset example keys:", list(hf_dataset[0].keys()))
    # show the first example content summary:
    ex = hf_dataset[0]
    for k,v in ex.items():
        print(k, type(v), (v if (isinstance(v, (int, str)) or len(str(v))<80) else f"<{len(v)}-items>"))
    

        
    split = int(0.9 * len(dataset))
    train_dataset = hf_dataset.select(range(0, split))
    eval_dataset  = hf_dataset.select(range(split, len(hf_dataset)))

    accelerator.print(f"Train: {len(train_dataset)} examples, Val: {len(eval_dataset)} examples.")


    # 3. load base model
    accelerator.print(" Loading model...")
    base_model = LLaDAModelLM.from_pretrained(args.model_path)
    base_model.resize_token_embeddings(len(tokenizer))
    print("PAD id:", pad_token_id)



    # 4. LoRA 
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj"],
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    model.to(device)



    '''
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01, betas=(0.9, 0.95))

    #wrap
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    '''
    
    prefix_text = "Detailed Explanation:\n\n"
    prefix_len = len(tokenizer(prefix_text, add_special_tokens=False).input_ids)
    
    global_step = 0   
    eps=1e-6
    

    trainer = StructuredMDLMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda f: custom_collator(f, pad_token_id),
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            learning_rate=2e-6,
            num_train_epochs=args.epochs,
            logging_steps=args.log_interval,   
            bf16=True,
            remove_unused_columns=False,
        ),
        scheduler=LinearAlphaScheduler(),  

    )
    
    trainer = accelerator.prepare(trainer)

    trainer.train()
    

    val_loss = trainer.evaluate()
    print("Validation loss:", val_loss["eval_loss"])
    for i, ex in enumerate(eval_dataset.select(range(3))):
        prompt_len = ex["prompt_len"]
        prompt_ids = torch.tensor(ex["input_ids"][:prompt_len], dtype=torch.long)
        gen_len = 512
        mask_tokens = torch.full((gen_len,), mask_token_id, dtype=torch.long)
        gen_input = torch.cat([prompt_ids, mask_tokens], dim=0).unsqueeze(0).to(device)

        gen_output = generate_sns_global(
            model,
            gen_input,
            steps=128,                  
            temperature=0.0,
            cfg_scale=0.0,
            mask_id=mask_token_id

        )

        gen_text = tokenizer.decode(
            gen_output[0][prompt_len:], 
            skip_special_tokens=True
        )

        print(f"=== Sample {i} ===")
        print("Generated text (first 200 chars):", gen_text)
        print("Target (first 200 chars):", ex.get("detailed_process", ""))
    
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
