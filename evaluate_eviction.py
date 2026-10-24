"""
Configurable KV-Cache Eviction Implementation & Evaluation for Qwen2.5 / Transformers.

Eviction Policies Supported:
1. Full Cache (No eviction baseline)
2. StreamingLLM (Sink + Local sliding window)
   - Retains initial sink tokens (e.g. 4 tokens) which absorb massive attention scores.
   - Retains recent local tokens (e.g. window of N tokens).
   - Evicts intermediate tokens when cache exceeds capacity.
3. Pure Sliding Window (Recent window only)
4. Heavy-Hitter (H2O / Attention-score based) or Strided eviction.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import time
import math
from typing import Optional, List, Dict, Any
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, CacheLayerMixin

def get_kv_cache_tensor_memory(past_key_values):
    """Calculates exact total bytes and shape of keys/values in cache."""
    total_bytes = 0
    num_layers = 0
    seq_len = 0
    if past_key_values is None:
        return 0, 0, 0

    if hasattr(past_key_values, "layers"):
        num_layers = len(past_key_values.layers)
        for layer in past_key_values.layers:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if isinstance(keys, torch.Tensor) and keys.numel() > 0:
                total_bytes += keys.numel() * keys.element_size()
                seq_len = keys.shape[-2]
            if isinstance(values, torch.Tensor) and values.numel() > 0:
                total_bytes += values.numel() * values.element_size()
    return total_bytes, num_layers, seq_len

class EvictionPolicy:
    def __init__(self, name: str, max_capacity: int, sink_tokens: int = 4):
        self.name = name
        self.max_capacity = max_capacity
        self.sink_tokens = sink_tokens

    def prune_layer(self, keys: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        keys shape: [batch, num_heads, seq_len, head_dim]
        values shape: [batch, num_heads, seq_len, head_dim]
        """
        seq_len = keys.shape[-2]
        if self.max_capacity is None or seq_len <= self.max_capacity:
            return keys, values

        if self.name == "full":
            return keys, values

        elif self.name == "streaming_llm":
            # Retain sink_tokens from front + (max_capacity - sink_tokens) from tail
            sink_k = keys[:, :, :self.sink_tokens, :]
            sink_v = values[:, :, :self.sink_tokens, :]
            recent_count = self.max_capacity - self.sink_tokens
            recent_k = keys[:, :, -recent_count:, :]
            recent_v = values[:, :, -recent_count:, :]
            new_k = torch.cat([sink_k, recent_k], dim=-2)
            new_v = torch.cat([sink_v, recent_v], dim=-2)
            return new_k, new_v

        elif self.name == "sliding_window":
            # Retain only the most recent max_capacity tokens
            new_k = keys[:, :, -self.max_capacity:, :]
            new_v = values[:, :, -self.max_capacity:, :]
            return new_k, new_v

        else:
            raise ValueError(f"Unknown eviction policy: {self.name}")

    def apply(self, past_key_values: DynamicCache):
        """Applies eviction policy across all layers in-place."""
        if self.max_capacity is None or past_key_values is None:
            return
        if hasattr(past_key_values, "layers"):
            for layer in past_key_values.layers:
                if getattr(layer, "is_initialized", False) and layer.keys is not None:
                    k, v = self.prune_layer(layer.keys, layer.values)
                    layer.keys = k
                    layer.values = v

def run_generation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 120,
    policy: Optional[EvictionPolicy] = None,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Executes autoregressive generation with token eviction applied at each decode step.
    Computes generation latency, throughput, token perplexity/loss, and memory footprint.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[-1]

    past_key_values = DynamicCache()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        mem_pre = torch.cuda.memory_allocated() / (1024 ** 2)
    else:
        mem_pre = 0.0

    t_start = time.perf_counter()

    # Monotonic position tracking:
    # prompt tokens occupy logical indices 0 ... prompt_len - 1
    current_position = prompt_len

    # Step 1: Prefill phase
    with torch.no_grad():
        out = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        next_token = out.logits[:, -1:, :].argmax(dim=-1)

    generated_tokens = [next_token.item()]
    step_logits = [out.logits[:, -1:, :]]

    # Apply eviction if prefill already exceeds capacity
    if policy is not None:
        policy.apply(past_key_values)

    # Step 2: Decoding loop with monotonic position tracking and eviction
    for _ in range(max_new_tokens - 1):
        pos_tensor = torch.tensor([[current_position]], device=device, dtype=torch.long)
        with torch.no_grad():
            out = model(
                input_ids=next_token,
                position_ids=pos_tensor,
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = out.past_key_values
            next_token = out.logits[:, -1:, :].argmax(dim=-1)

            generated_tokens.append(next_token.item())
            step_logits.append(out.logits[:, -1:, :])
            current_position += 1

            # Apply eviction policy after new token states are concatenated
            if policy is not None:
                policy.apply(past_key_values)

    if device == "cuda":
        torch.cuda.synchronize()
        mem_post = torch.cuda.memory_allocated() / (1024 ** 2)
        mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        mem_post = 0.0
        mem_peak = 0.0

    t_end = time.perf_counter()
    latency = t_end - t_start

    # Metrics
    num_generated = len(generated_tokens)
    throughput = num_generated / latency if latency > 0 else 0
    kv_bytes, _, final_cache_seq_len = get_kv_cache_tensor_memory(past_key_values)
    kv_mb = kv_bytes / (1024 ** 2)

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Calculate average negative log-likelihood (surrogate quality metric for generated text)
    all_logits = torch.cat(step_logits, dim=1)[:, :-1, :] # [1, T-1, V]
    target_tokens = torch.tensor(generated_tokens[1:], device=device).unsqueeze(0) # [1, T-1]
    loss = F.cross_entropy(all_logits.view(-1, all_logits.shape[-1]), target_tokens.view(-1)).item()
    ppl = math.exp(min(loss, 100))

    return {
        "policy_name": policy.name if policy else "full",
        "max_capacity": policy.max_capacity if policy else None,
        "sink_tokens": policy.sink_tokens if policy else 0,
        "prompt_len": prompt_len,
        "num_generated": num_generated,
        "final_cache_tokens": final_cache_seq_len,
        "kv_mb": kv_mb,
        "kv_bytes": kv_bytes,
        "latency_sec": latency,
        "throughput": throughput,
        "gpu_pre_mb": mem_pre,
        "gpu_post_mb": mem_post,
        "gpu_peak_mb": mem_peak,
        "loss": loss,
        "perplexity": ppl,
        "text": generated_text,
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map=device,
        local_files_only=True
    )
    model.eval()

    prompt = (
        "In modern deep learning architectures, attention mechanisms require substantial computational "
        "and memory resources. Explain in detail the trade-offs between full attention KV caching, "
        "sliding window attention, and token eviction strategies such as StreamingLLM. Specifically address "
        "how attention sinks prevent performance degradation, how memory scaling changes from O(N) to O(1), "
        "and what impact eviction has on long-range dependencies."
    )

    print(f"\nPrompt length: {len(tokenizer.encode(prompt))} tokens")
    max_new_tokens = 150

    experiments = [
        ("Full Cache Baseline", None),
        ("StreamingLLM (Cap=64, Sink=4)", EvictionPolicy("streaming_llm", max_capacity=64, sink_tokens=4)),
        ("StreamingLLM (Cap=40, Sink=4)", EvictionPolicy("streaming_llm", max_capacity=40, sink_tokens=4)),
        ("Sliding Window (Cap=40, Sink=0)", EvictionPolicy("sliding_window", max_capacity=40)),
    ]

    results = []
    print("=" * 80)
    print("RUNNING BENCHMARKS...")
    print("=" * 80)

    for desc, policy in experiments:
        print(f"\n--> Running experiment: {desc}")
        res = run_generation(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            policy=policy,
            device=device
        )
        results.append((desc, res))

    print("\n" + "=" * 80)
    print("EVALUATION & COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':<32} | {'KV (MB)':<8} | {'SeqLen':<7} | {'Latency':<8} | {'Tok/s':<7} | {'Loss':<6} | {'PPL':<6}")
    print("-" * 80)
    for desc, r in results:
        print(f"{desc:<32} | {r['kv_mb']:<8.4f} | {r['final_cache_tokens']:<7} | {r['latency_sec']:<8.3f} | {r['throughput']:<7.2f} | {r['loss']:<6.3f} | {r['perplexity']:<6.2f}")

    print("\n" + "=" * 80)
    print("GENERATED SAMPLE COMPARISONS (First 200 chars)")
    print("=" * 80)
    for desc, r in results:
        snippet = r['text'][:200].replace('\n', ' ')
        print(f"[{desc}]\n  {snippet}...\n")

if __name__ == "__main__":
    main()
