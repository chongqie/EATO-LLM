# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np
import json
import re
from transformers import AutoTokenizer
from modeling_llada import LLaDAModelLM
from peft import PeftModel

BASE_MODEL_PATH = "/path/to/LLaDA-8B-Instruct"
ADAPTER_PATH = "/path/to/your/finetuned_lora"     
MASK_TOKEN = "<|mdm_mask|>"
PAD_TOKEN = "<|reserved_token_0|>"

# prompt just the same as training
PROMPT_TEXT = (
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

FIXED_DRAFT_LEN = 15        # 每个步骤标题句的 token 数
FIXED_DETAIL_LEN = 128      # 每个步骤详细内容的 token 数
DRAFT_STEPS = 16             # 生成 Draft 的去噪步数（本身不再使用，保留以兼容）
DETAIL_STEPS = 64            # 生成 Details 的去噪步数
DEVICE = "cuda"

def extract_final_answer(text: str) -> str:
    """从生成文本中提取 #### 后面的最终数字答案"""
    match = re.search(r"####\s*([\d\.]+)", text)
    if match:
        return match.group(1)
    return ""

#全序列扩散
@torch.no_grad()
def generate_sns_global(
    model,
    input_ids,
    steps=64,
    temperature=0.0,
    cfg_scale=0.0,
    mask_id=126336,
    entropy_tau_start=0.1,           # inital entropy threshold, start with a low value to fix only very certain tokens (the "skeleton")
    entropy_tau_end=2.0,             # final entropy threshold, gradually increase to allow more tokens to be fixed (the "details")
    max_fix_frac=0.4,                # entropy threshold schedule controls the fraction of masked tokens that can be fixed at each step, start strict and become more lenient
    min_fix_abs=1,                   # every step must fix at least this many tokens (if available

):
    device = model.device
    x = input_ids.clone().to(device)
    prompt_index = (x != mask_id)
    B, L = x.shape
    trajectory = {
        'step': [],
        'mask_positions': [],   
        'entropy': [],          
        'input_ids': []         
    }
    for t in range(steps):
        if cfg_scale > 0.0:
            un_x = x.clone()
            un_x[prompt_index] = mask_id
            x_cat = torch.cat([x, un_x], dim=0)
            logits = model(x_cat).logits
            cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
            logits = uncond_logits + (cfg_scale + 1) * (cond_logits - uncond_logits)
        else:
            logits = model(x).logits

        if temperature > 0:
            noise = torch.rand_like(logits)
            gumbel = -torch.log(-torch.log(noise + 1e-9) + 1e-9)
            logits = logits + gumbel * temperature

        probs = F.softmax(logits.float(), dim=-1)
        log_probs = torch.log(probs + 1e-9)
        entropy = -torch.sum(probs * log_probs, dim=-1)          # (B, L)

        x0 = torch.argmax(logits, dim=-1)

        mask_index = (x == mask_id) & ~prompt_index
        if not mask_index.any():
            break
        x0 = torch.where(prompt_index, x, x0)


        # calculate entropy threshold, schedule only control max_fix_frac, but not min_fix_abs, to ensure a minimum number of tokens are fixed each step
        tau = entropy_tau_start + (entropy_tau_end - entropy_tau_start) * (t / max(1, steps-1))

        fix_candidate = mask_index & (entropy < tau)

        num_masked = mask_index.sum(dim=1)  # (B,)
        # based on the current number of masked tokens and the schedule, calculate how many tokens can be fixed this step, with a minimum guarantee
        max_fix = torch.clamp((num_masked.float() * max_fix_frac).long(), min=min_fix_abs)
        min_fix = torch.ones_like(num_masked) * min_fix_abs

        transfer_index = torch.zeros_like(x, dtype=torch.bool)
        
        """
        # save the trajectory for analysis, can remove this part if not needed
        trajectory['step'].append(t)
        trajectory['mask_positions'].append(mask_index.cpu().clone())
        trajectory['entropy'].append(entropy.cpu().clone())
        """
        for i in range(B):
            if num_masked[i] == 0:
                continue

            candidate_idx = fix_candidate[i]
            n_candidate = candidate_idx.sum().item()

            #for candidate tokens'number consider 3 cases
            if n_candidate > max_fix[i]:
                # if too many candidates, only fix the most certain ones (the "skeleton"), to avoid fixing too many tokens at early steps which may lead to error accumulation
                ent_masked = entropy[i].clone()
                ent_masked[~mask_index[i]] = 1e9
                _, topk_idx = torch.topk(ent_masked, k=max_fix[i], largest=False)
                transfer_index[i, topk_idx] = True
            elif n_candidate >= min_fix[i]:
                # candidate tokens are within the acceptable range, fix them all (the "details")
                transfer_index[i, candidate_idx] = True
            else:
                # if too few candidates, still fix the most certain ones to ensure a minimum number of tokens are fixed each step, to maintain generation momentum and avoid stagnation
                ent_masked = entropy[i].clone()
                ent_masked[~mask_index[i]] = 1e9
                kk = min(min_fix[i].item(), num_masked[i].item())
                _, topk_idx = torch.topk(ent_masked, k=kk, largest=False)
                transfer_index[i, topk_idx] = True

        # 8. 更新 tokens
        x[transfer_index] = x0[transfer_index]

    return x
#example
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, use_fast=False)
    mask_token_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
    if mask_token_id == tokenizer.unk_token_id:
        mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)

    base_model = LLaDAModelLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16)
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model = model.merge_and_unload().to(DEVICE).eval()

    eval_file = "your_eval.jsonl"
    correct = 0
    total = 0
    with open(eval_file, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            question = entry["question"]
            true_answer = entry.get("answer_text", "").strip()

            prompt_ids = tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids[0]
            question_text = f"Question:{question}\n" 
            q_ids = tokenizer(question_text, return_tensors="pt").input_ids[0]
            input_ids = torch.cat([prompt_ids, q_ids], dim=0)

            gen_length = 512  
            mask_token_tensor = torch.full((gen_length,), mask_token_id, dtype=torch.long)
            input_ids = torch.cat([input_ids, mask_token_tensor], dim=0).unsqueeze(0)

            gen_ids = generate_sns_global(
                model,
                input_ids,
                steps=DETAIL_STEPS,
                temperature=0.0,
                cfg_scale=0.0,
                mask_id=mask_token_id,
                schedule="cosine"
            )
            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            pred_answer = extract_final_answer(gen_text)

            if pred_answer == true_answer:
                correct += 1
            total += 1

            print(f"Q: {question[:50]}...")
            print(f"Generated answer: {pred_answer} | True: {true_answer}")
            print("---")

    print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")