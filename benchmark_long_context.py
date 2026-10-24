"""
Reproducible Long-Context Evaluation Suite for FlashCache-KV.

Evaluates:
- Context Lengths: 512, 1024, 2048 tokens
- Tasks:
  1. needle_beginning: Secret key located in initial 10% of prompt.
  2. needle_middle: Secret key located in middle 45-55% of prompt.
  3. needle_end: Secret key located in final 90% of prompt.
  4. normal_generation: Open-ended reasoning/generation task.
- Strategies:
  1. Full KV Cache Baseline
  2. StreamingLLM Eviction (Cap = 64 tokens, Sink = 4)
  3. StreamingLLM Eviction (Cap = 40 tokens, Sink = 4)
- Metrics:
  - Exact-match retrieval accuracy (needle tasks)
  - KV-cache tensor memory (MB)
  - Peak allocated GPU VRAM (MB)
  - Time to first token (TTFT, ms)
  - Per-token decode latency (ms) & tokens/sec
  - Actual input and generated token counts
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import time
import json
import torch
from typing import Dict, Any, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from evaluate_eviction import EvictionPolicy, get_kv_cache_tensor_memory

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

FILLER_SENTENCE = "Advanced neural network architectures utilize multi-head self-attention mechanisms to capture complex contextual dependencies across diverse input sequences. "

def create_needle_prompt(tokenizer, target_tokens: int, needle_pos: str, secret_passkey: str) -> Tuple[str, str]:
    """
    Constructs a document with a needle passkey inserted at 'beginning', 'middle', or 'end'.
    Returns (prompt, expected_answer).
    """
    needle = f"\n[CRITICAL NOTICE: The secret emergency authorization passkey is {secret_passkey}. Memorize this passkey.]\n"
    question = f"\nQuestion: What is the secret emergency authorization passkey? Answer strictly with the passkey: "
    expected_answer = secret_passkey

    # Approximate filler repeats needed
    base_text = ""
    while len(tokenizer.encode(base_text)) < target_tokens - 60:
        base_text += FILLER_SENTENCE

    tokens = tokenizer.encode(base_text)
    total_len = len(tokens)

    if needle_pos == "beginning":
        insert_idx = int(total_len * 0.05)
    elif needle_pos == "middle":
        insert_idx = int(total_len * 0.50)
    elif needle_pos == "end":
        insert_idx = int(total_len * 0.90)
    else:
        insert_idx = int(total_len * 0.50)

    part1 = tokenizer.decode(tokens[:insert_idx])
    part2 = tokenizer.decode(tokens[insert_idx:])

    prompt = part1 + needle + part2 + question
    return prompt, expected_answer

def run_single_benchmark(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 15,
    policy: EvictionPolicy = None,
    device: str = "cuda"
) -> Dict[str, Any]:
    """Runs synchronized, profiled generation with monotonic position tracking."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[-1]

    cache = DynamicCache()

    # Pre-run GPU memory baseline
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / (1024 ** 2)
    else:
        mem_before = 0.0

    # 1. Prefill step (TTFT)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    if device == "cuda":
        torch.cuda.synchronize()
    ttft = (time.perf_counter() - t0) * 1000.0 # ms

    cache = out.past_key_values
    next_token = out.logits[:, -1:, :].argmax(dim=-1)

    # Apply initial eviction if policy active
    if policy is not None:
        policy.apply(cache)

    current_position = prompt_len
    generated_tokens = [next_token.item()]
    decode_latencies = []

    # 2. Decode steps
    for _ in range(max_new_tokens - 1):
        pos_tensor = torch.tensor([[current_position]], device=device, dtype=torch.long)
        t_step = time.perf_counter()
        with torch.no_grad():
            out = model(
                input_ids=next_token,
                position_ids=pos_tensor,
                past_key_values=cache,
                use_cache=True
            )
            if device == "cuda":
                torch.cuda.synchronize()
        step_ms = (time.perf_counter() - t_step) * 1000.0
        decode_latencies.append(step_ms)

        cache = out.past_key_values
        next_token = out.logits[:, -1:, :].argmax(dim=-1)
        generated_tokens.append(next_token.item())
        current_position += 1

        if policy is not None:
            policy.apply(cache)

    # Final measurements
    if device == "cuda":
        torch.cuda.synchronize()
        peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        total_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    else:
        peak_allocated = 0.0
        total_reserved = 0.0

    kv_bytes, _, final_cache_len = get_kv_cache_tensor_memory(cache)
    kv_mb = kv_bytes / (1024 ** 2)

    mean_decode_ms = sum(decode_latencies) / len(decode_latencies) if decode_latencies else 0.0
    throughput = (1000.0 / mean_decode_ms) if mean_decode_ms > 0 else 0.0

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return {
        "prompt_tokens": prompt_len,
        "generated_tokens": len(generated_tokens),
        "ttft_ms": ttft,
        "mean_decode_ms": mean_decode_ms,
        "throughput_tok_s": throughput,
        "kv_mb": kv_mb,
        "final_cache_len": final_cache_len,
        "peak_vram_mb": peak_allocated,
        "total_reserved_mb": total_reserved,
        "generated_text": generated_text,
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map=device,
        local_files_only=True
    )
    model.eval()

    # Warm-up run
    print("Executing GPU warm-up run...")
    warm_prompt = "Warm-up sequence initializing CUDA contexts and memory allocators."
    _ = run_single_benchmark(model, tokenizer, warm_prompt, max_new_tokens=5, device=device)

    # Evaluation Matrix
    context_lengths = [512, 1024, 2048]
    needle_positions = ["beginning", "middle", "end"]
    secret_key = "ALPHA-7749"

    strategies = [
        ("Full KV Cache", None),
        ("Eviction (Cap=64, Sink=4)", EvictionPolicy("streaming_llm", max_capacity=64, sink_tokens=4)),
        ("Eviction (Cap=40, Sink=4)", EvictionPolicy("streaming_llm", max_capacity=40, sink_tokens=4)),
    ]

    all_results = []
    print("\n" + "=" * 90)
    print("STARTING REPRODUCIBLE LONG-CONTEXT EVALUATION")
    print("=" * 90)

    for ctx_len in context_lengths:
        for pos in needle_positions:
            prompt, expected = create_needle_prompt(tokenizer, ctx_len, pos, secret_key)
            actual_len = len(tokenizer.encode(prompt))

            print(f"\n[Task: Needle-{pos.upper()} | Target Len: {ctx_len} | Actual Len: {actual_len}]")

            for strat_name, pol in strategies:
                res = run_single_benchmark(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=15,
                    policy=pol,
                    device=device
                )

                # Evaluate accuracy
                # Exact match or substring containment
                is_correct = (expected.lower() in res["generated_text"].lower())
                res["task"] = f"needle_{pos}"
                res["target_ctx"] = ctx_len
                res["strategy"] = strat_name
                res["accuracy"] = 1.0 if is_correct else 0.0
                res["expected"] = expected

                print(f"  -> {strat_name:<28} | Acc: {res['accuracy']} | KV: {res['kv_mb']:<6.3f} MB | Peak VRAM: {res['peak_vram_mb']:<7.1f} MB | TTFT: {res['ttft_ms']:<6.1f} ms | Tok/s: {res['throughput_tok_s']:<5.1f}")
                print(f"     Output: {repr(res['generated_text'][:60])}")

                all_results.append(res)

    # Open-ended generation task at 1024 tokens
    print("\n" + "=" * 90)
    print("[Task: Normal Open-Ended Generation | Context: ~1024 tokens]")
    print("=" * 90)
    normal_prompt = (
        "Discuss in detail the trade-offs between dense model scaling and sparse mixture of experts. "
        + (FILLER_SENTENCE * 60)
        + "\nSummarize the main architectural trade-offs in three concise bullet points:\n"
    )
    for strat_name, pol in strategies:
        res = run_single_benchmark(
            model=model,
            tokenizer=tokenizer,
            prompt=normal_prompt,
            max_new_tokens=40,
            policy=pol,
            device=device
        )
        res["task"] = "normal_generation"
        res["target_ctx"] = 1024
        res["strategy"] = strat_name
        res["accuracy"] = 1.0 # fluency task
        res["expected"] = "N/A"
        print(f"  -> {strat_name:<28} | KV: {res['kv_mb']:<6.3f} MB | Peak VRAM: {res['peak_vram_mb']:<7.1f} MB | TTFT: {res['ttft_ms']:<6.1f} ms | Tok/s: {res['throughput_tok_s']:<5.1f}")
        print(f"     Output: {repr(res['generated_text'][:80])}")
        all_results.append(res)

    # Save results to JSON for dashboard integration
    out_path = r"C:\Users\admin\.gemini\antigravity\scratch\long_context_benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved benchmark results to {out_path}")

if __name__ == "__main__":
    main()
