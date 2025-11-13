import os, math, re, ast
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd
import torch
from torch.nn.functional import pad
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from latex2sympy2 import latex2sympy
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


# ───────────────────────────── helper utilities ──────────────────────────────
def load_model_and_tokenizer(ckpt: str, mp: str = "bf16"):
    """Return (accelerator, model, tokenizer) - model already on right device(s)."""
    accelerator = Accelerator(mixed_precision=mp)        

    tok = AutoTokenizer.from_pretrained(ckpt, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    config = AutoConfig.from_pretrained(ckpt)
    # accelerator.print(f"config: {config}")
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        ckpt,
        config=config,
        torch_dtype=torch.bfloat16 if mp == "bf16" else None,
    )
    model = accelerator.prepare(model)
    model.eval()
    return accelerator, model, tok


def extract_prompt(prompt_raw):
    """Extract plain user text from merged CSV format.

    CSV stores a stringified Python list of dicts:
        "[{ 'content': '...', 'role': 'user'}]"
    We parse with ast.literal_eval and return the first element's 'content'.
    Legacy ndarray support retained.
    """
    if isinstance(prompt_raw, np.ndarray):
        return prompt_raw[0]["content"]
    if isinstance(prompt_raw, str):
        text = prompt_raw.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    return parsed[0].get("content", prompt_raw)
            except Exception:
                return prompt_raw
        return prompt_raw
    if isinstance(prompt_raw, dict):  # unexpected but try common key
        return prompt_raw.get("content", str(prompt_raw))
    raise ValueError(f"Unsupported prompt_raw type: {type(prompt_raw)}")


def extract_answer(answer_raw):
    """Return ground truth numeric/string value.

    CSV stores a stringified dict: "{'ground_truth': '12.8', 'style': 'rule'}".
    Handles LaTeX fractions (e.g. '\\frac{4}{3}').
    """
    if isinstance(answer_raw, dict):
        return str(answer_raw.get("ground_truth", list(answer_raw.values())[0]))
    if isinstance(answer_raw, str):
        text = answer_raw.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, dict):
                    return str(parsed.get("ground_truth", next(iter(parsed.values()), text)))
            except Exception:
                return text
        return text
    return str(answer_raw)


def chat_wrap(tok, user_text: str) -> str:
    """Apply model's chat template if it exists."""
    if hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return user_text


def remove_prompt(text: str) -> str:
    ## find the first "assistant\n" and remove everything after it
    match = re.search(r"assistant\s*\n", text)
    if match:
        return text[match.end():].strip()
    else:
        raise ValueError(f"Text does not contain 'assistant\\n': {text}")


def sample_many(accelerator, model, tok, prompt: str, n: int,
                batch: int = 8, gen_kwargs: dict | None = None):
    """Generate `n` continuations of the same prompt using multi-GPU if available."""
    gen_kwargs = gen_kwargs or dict(
        max_new_tokens=1280,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        pad_token_id=tok.pad_token_id,
    )

    num_processes = accelerator.num_processes
    samples_per_process = math.ceil(n / num_processes)
    actual_total_samples = samples_per_process * num_processes
    
    accelerator.print(f"Multi-GPU sampling: {num_processes} processes, "
                     f"{samples_per_process} samples per process, "
                     f"total {actual_total_samples} samples (requested {n})")
    
    enc = tok(prompt, return_tensors="pt").to(accelerator.device)
    ids = []
    steps = math.ceil(samples_per_process / batch)
    
    # Set different seeds for each process to ensure diversity
    process_seed = accelerator.process_index * 1000
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)
    np.random.seed(process_seed)
    
    for step in tqdm(range(steps), desc=f"Sampling on GPU {accelerator.process_index}"):
                    #  disable=not accelerator.is_local_main_process, 
        # Calculate actual batch size for this step
        remaining_samples = samples_per_process - len(ids)
        current_batch_size = min(batch, remaining_samples)
        
        if current_batch_size <= 0:
            break
            
        batch_enc = {k: v.repeat(current_batch_size, 1) for k, v in enc.items()}
        with torch.no_grad():
            current_ids = accelerator.unwrap_model(model).generate(**batch_enc, **gen_kwargs)
            # print(f"local rank: {accelerator.process_index}, current_ids type: {type(current_ids)}, current_ids shape: {current_ids.shape}")
            ids.append(current_ids)
        
    pad_id = tok.pad_token_id
    max_len_local = max(t.shape[1] for t in ids)
    ids = [pad(t, (0, max_len_local - t.shape[1]), value=pad_id) for t in ids]    
    ids = torch.cat(ids, dim=0)
    # print(f"local rank: {accelerator.process_index}, ids.shape, {ids.shape}")    
    # accelerator.print(f"type(ids): {type(ids)}")
        
    ids = accelerator.pad_across_processes(ids, dim=1, pad_index=pad_id)
    gathered_ids = accelerator.gather(ids)[:n]
    accelerator.print(f"gathered_ids.shape, {gathered_ids.shape}")
    
    if accelerator.is_main_process:
        batch_results = tok.batch_decode(gathered_ids, skip_special_tokens=True)
        batch_results = [remove_prompt(text) for text in batch_results]
        return batch_results
    else:
        return []


def extract_last_boxed_content(text: str) -> str:
    """
    Extract the content of the last \boxed{} in the text, handling nested braces.
    """
    # Find the last occurrence of \boxed{
    last_boxed_pos = text.rfind(r'\boxed{')
    if last_boxed_pos == -1:
        return None
    
    # Start from the opening brace after \boxed
    start_pos = last_boxed_pos + 7  # len(r'\boxed{') = 7
    brace_count = 0  
    current_pos = start_pos
    
    while current_pos < len(text) and brace_count >= 0:
        if text[current_pos] == '{':
            brace_count += 1
        elif text[current_pos] == '}':
            brace_count -= 1
        current_pos += 1
    
    if brace_count == -1:
        # Successfully found matching closing brace
        return text[start_pos:current_pos-1]
    else:
        # No matching closing brace found
        return None
    

def keep_correct_answers(texts: List[str], answer: str) -> List[str]:
    """Retain responses whose last boxed value numerically matches answer within ±10%.

    Attempts to interpret answer as float first; if that fails, try latex2sympy.
    """
    correct_answers = []
    try:
        answer_value = float(answer)
    except Exception:
        try:
            answer_value = float(latex2sympy(answer).evalf())
        except Exception:
            # fallback: no numeric comparison possible
            answer_value = None
    if answer_value is not None:
        lower_bound = answer_value * 0.9
        upper_bound = answer_value * 1.1
        if lower_bound > upper_bound:
            lower_bound, upper_bound = upper_bound, lower_bound
    for id, text in enumerate(tqdm(texts, desc="Filtering valid responses")):
        prediction = extract_last_boxed_content(text)
        if prediction is None:
            continue
        try:
            sympy_expr = latex2sympy(prediction)
            predicted_value = sympy_expr.evalf()
            if answer_value is not None and lower_bound <= predicted_value <= upper_bound:
                correct_answers.append(text)
        except Exception as e:
            # ignore malformed predictions
            continue
    return correct_answers


def parse_args():
    parser = argparse.ArgumentParser(description="Generate responses for ALL prompts in a merged CSV dataset.")
    # Model and data arguments
    parser.add_argument("--model_path", type=str, default="/cephfs/lxh/models/qwen2.5-math-1.5b")
    parser.add_argument("--input_data", type=str, required=True, help="Input merged CSV (pi_merged.csv format).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs (correct/incorrect).")

    # Sampling arguments
    parser.add_argument("--total_samples", type=int, default=16000, help="Total responses to generate across ALL prompts.")
    parser.add_argument("--per_prompt_max", type=int, default=None, help="Optional cap per prompt; if unset distribute evenly.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Accelerate")

    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=1280)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)

    # Model arguments
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--dry_run", action="store_true", help="Parse CSV and plan generation without model load or sampling.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    ckpt       = args.model_path
    input_path = args.input_data
    out_dir    = args.output_dir
    total      = args.total_samples
    batch_size = args.batch_size
    precision  = args.precision

    os.makedirs(out_dir, exist_ok=True)

    # Enforce CSV-only input
    if not input_path.endswith('.csv'):
        raise ValueError("input_data must be a CSV file (merged format)")
    df = pd.read_csv(input_path)

    required_cols = {"prompt", "reward_model"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input dataset must contain columns {required_cols}, found {df.columns}")

    num_prompts = len(df)
    if num_prompts == 0:
        raise ValueError("Input dataset is empty")

    # Distribute samples across prompts
    if args.per_prompt_max is not None:
        per_prompt_target = min(args.per_prompt_max, math.ceil(total / num_prompts))
    else:
        per_prompt_target = math.ceil(total / num_prompts)
    # Adjust total to actual number we'll generate
    actual_total = per_prompt_target * num_prompts
    print(f"Planning to generate {actual_total} responses: {per_prompt_target} per prompt * {num_prompts} prompts")

    if args.dry_run:
        print("[dry-run] Loaded CSV with", len(df), "rows")
        for i, row in df.head(3).iterrows():
            p_txt = extract_prompt(row["prompt"])[:100].replace("\n", " ")
            a_txt = extract_answer(row["reward_model"])[:50]
            print(f"  Row {i}: prompt='{p_txt}...' answer='{a_txt}'")
        return

    acc, model, tok = load_model_and_tokenizer(ckpt, mp=precision)
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "pad_token_id": tok.pad_token_id,
    }

    all_records: List[Dict[str, Any]] = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Prompts"):
        prompt_text = extract_prompt(row["prompt"])
        answer_text = extract_answer(row["reward_model"])
        formatted_prompt = chat_wrap(tok, prompt_text)

        gen_texts = sample_many(acc, model, tok, formatted_prompt,
                                n=per_prompt_target, batch=batch_size, gen_kwargs=gen_kwargs)
        if acc.is_main_process:
            valid_texts = keep_correct_answers(gen_texts, answer_text)
            cnt = Counter(valid_texts)
            for t in gen_texts:
                is_correct = cnt[t] > 0
                if is_correct:
                    cnt[t] -= 1
                all_records.append({
                    "prompt": formatted_prompt,
                    "response": t,
                    "correct": int(is_correct),
                    "source_row": int(idx),
                })

    # Only main process writes output
    if acc.is_main_process:
        out_df = pd.DataFrame(all_records)
        correct_df = out_df[out_df.correct == 1].copy()
        incorrect_df = out_df[out_df.correct == 0].copy()
    correct_df.to_csv(os.path.join(out_dir, "correct.csv"), index=False)
    incorrect_df.to_csv(os.path.join(out_dir, "incorrect.csv"), index=False)
    out_df.to_csv(os.path.join(out_dir, "all.csv"), index=False)
    print(f"Saved CSV files: correct={len(correct_df)} incorrect={len(incorrect_df)} total={len(out_df)} in {out_dir}")


if __name__ == "__main__":
    main()
