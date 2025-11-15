import os, math, re
import pandas as pd
import torch
from torch.nn.functional import pad
import argparse
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator           
from accelerate.utils import set_seed
from latex2sympy2 import latex2sympy
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


# ───────────────────────────── helper utilities ──────────────────────────────
def load_model_and_tokenizer(ckpt: str, mp: str = "bf16"):
    """Return (accelerator, model, tokenizer) - model already on right device(s)."""
    accelerator = Accelerator(mixed_precision=mp)        

    tok = AutoTokenizer.from_pretrained(ckpt, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    config = AutoConfig.from_pretrained(ckpt)
    accelerator.print(f"config: {config}")
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
    ## format of prompt_raw: np.ndarray[dict[str, str]]
    if not isinstance(prompt_raw, np.ndarray):
        raise ValueError(f"prompt_raw is of type {type(prompt_raw)} instead of np.ndarray")
    else:
        return prompt_raw[0]["content"]


def extract_answer(answer_raw):
    ## format of answer_raw: dict[str, str]
    if not isinstance(answer_raw, dict):
        raise ValueError(f"answer_raw is of type {type(answer_raw)} instead of dict")
    else:
        return answer_raw["ground_truth"]


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


def sample_many(accelerator, model, tok, prompt: str, n: int, \
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
    

def judge_answers(texts: list[str], answer: str) -> tuple[list[str], list[str]]:
    """Retain responses that contain the reference answer as a substring."""
    correct_answers = []
    incorrect_answers = []
    answer_value = float(answer) ## maybe can consider latex2sympy later
    lower_bound = answer_value * 0.9
    upper_bound = answer_value * 1.1
    if lower_bound > upper_bound:
        lower_bound, upper_bound = upper_bound, lower_bound
    for id, text in enumerate(tqdm(texts, desc="Filtering valid responses")):
        print(f"current candidate id: {id}")
        prediction = extract_last_boxed_content(text)
        if prediction is not None:
            print(f"################################# format correct, answer: {answer}, prediction: {prediction}")
            try:
                sympy_expr = latex2sympy(prediction)
                predicted_value = sympy_expr.evalf()
                print(f"################################# predicted_value: {predicted_value}")
                if lower_bound <= predicted_value <= upper_bound:
                    correct_answers.append(text)
                else:
                    print(f"################################# value out of range, answer: {answer}, prediction: {prediction}, predicted_value: {predicted_value}")
                    incorrect_answers.append(text)
            except TypeError:
                print(f"TypeError, prediction: {prediction}")
                incorrect_answers.append(text)
            except ValueError:
                print(f"ValueError, prediction: {prediction}")
                incorrect_answers.append(text)
            except Exception as e:
                print(f"Other error: {e}, prediction: {prediction}")
                incorrect_answers.append(text)
        else:
            print(f"################################# format incorrect, answer: {answer}, prediction: {prediction}")
            incorrect_answers.append(text)

    return correct_answers, incorrect_answers


def parse_args():
    parser = argparse.ArgumentParser()    
    # Model and data arguments
    parser.add_argument("--model_path", type=str, default="/homes/gws/lxh22/models/Qwen2.5-Math-1.5B")
    parser.add_argument("--input_data", type=str, default="input-data/pi1_r128.parquet")
    parser.add_argument("--output_parquet", type=str, default="../train-data/pi1_r128_pm_responses_16000.parquet")
    parser.add_argument("--output_csv", type=str, default="../train-data/pi1_r128_pm_responses_16000.csv")
    
    # Sampling arguments
    parser.add_argument("--n_samples", type=int, default=16000, help="Number of samples to generate")
    parser.add_argument("--n_train", type=int, default=15000, help="Number of samples to keep for training; the rest will be saved as validation data")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Accelerate")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=1280)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    
    # Model arguments
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    assert args.n_train <= args.n_samples, \
        f"n_train ({args.n_train}) must be less than or equal to n_samples ({args.n_samples})"

    ckpt            = args.model_path
    parquet_in      = args.input_data
    parquet_out     = args.output_parquet
    csv_out         = args.output_csv
    n_samples       = args.n_samples
    batch_size      = args.batch_size
    precision       = args.precision
    parquet_out_val = parquet_out.replace(".parquet", "_valid.parquet")
    csv_out_valid   = csv_out.replace(".csv", "_valid.csv")
    
    acc, model, tok = load_model_and_tokenizer(ckpt, mp=precision)
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "pad_token_id": tok.pad_token_id,
    }

    ## read the input
    df             = pd.read_parquet(parquet_in)
    row            = df.iloc[0]   ## the single example
    prompt         = extract_prompt(row["prompt"])
    answer         = extract_answer(row["reward_model"])
    # print(f"data loaded. prompt: {prompt}, answer: {answer}")

    ## generation + filtering
    formatted_prompt = chat_wrap(tok, prompt)
    # print(f"formatted_prompt: {formatted_prompt}")
    # print(f"Using {acc.num_processes} GPUs")
    gen_texts    = sample_many(acc, model, tok, formatted_prompt,
                               n=n_samples, batch=batch_size, gen_kwargs=gen_kwargs)
    
    ## Under main process, filter out valid samples and save them
    if acc.is_main_process:
        valid_texts, invalid_texts  = judge_answers(gen_texts, answer)
        print(len(valid_texts), "valid samples, ", len(invalid_texts), "invalid samples")
        assert len(valid_texts) + len(invalid_texts) == n_samples
        y = len(valid_texts)
        x = len(invalid_texts)
        positive_reward = np.sqrt(x / y)
        negative_reward = -np.sqrt(y / x) 
        # positive_reward = len(invalid_texts) / n_samples
        # negative_reward = -len(valid_texts) / n_samples
        acc.print(f"Positive reward: {positive_reward}, Negative reward: {negative_reward}")
        df_valid = pd.DataFrame({"prompt": [formatted_prompt] * len(valid_texts),
                           "response": valid_texts,
                           "reward": [positive_reward] * len(valid_texts)})

        df_invalid = pd.DataFrame({"prompt": [formatted_prompt] * len(invalid_texts),
                           "response": invalid_texts,
                           "reward": [negative_reward] * len(invalid_texts)})
        
        ## merge valid and invalid samples, and random shuffle
        df_all = pd.concat([df_valid, df_invalid], ignore_index=True)
        df_all = df_all.sample(frac=1, random_state=args.seed).reset_index(drop=True)

        ## take the first `n_train` samples for training, and the rest for validation
        out_df = df_all.iloc[:args.n_train]
        out_df_val = df_all.iloc[args.n_train:]
    
        os.makedirs(os.path.dirname(parquet_out), exist_ok=True)
        out_df.to_parquet(parquet_out, index=False)
        out_df.to_csv(csv_out, index=False)
        out_df_val.to_parquet(parquet_out_val, index=False)
        out_df_val.to_csv(csv_out_valid, index=False)
        acc.print(f"Saved {args.n_train} valid samples to {parquet_out} and {csv_out}, "
               f"and {args.n_samples - args.n_train} valid samples to {parquet_out_val} and {csv_out_valid}.")


if __name__ == "__main__":
    main()
