"""
Comparative Benchmark: Eviction vs Experimental Similarity-Based KV Merging.

Under identical memory budgets (e.g. capacity = 40 and 64 tokens):
1. Full Cache Baseline (unbounded)
2. Token Eviction (StreamingLLM with 4 sink tokens + local window)
3. Similarity-Based KV Merging (Cosine similarity between adjacent key vectors
   weighted by magnitude or simple weighted average, preserving past information
   rather than discarding it).
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import time
import math
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

def get_kv_cache_tensor_memory(past_key_values) -> Tuple[int, int]:
    """Calculates exact total bytes and seq length in cache."""
    total_bytes = 0
    seq_len = 0
    if past_key_values is None or not hasattr(past_key_values, "layers"):
        return 0, 0

    for layer in past_key_values.layers:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if isinstance(keys, torch.Tensor) and keys.numel() > 0:
            total_bytes += keys.numel() * keys.element_size()
            seq_len = keys.shape[-2]
        if isinstance(values, torch.Tensor) and values.numel() > 0:
            total_bytes += values.numel() * values.element_size()
    return total_bytes, seq_len

class KVCompressor:
    """
    KV-Cache Compression Strategy.
    
    Supported:
    - 'eviction': StreamingLLM-style sink + local sliding window eviction.
    - 'merging': Experimental adjacent key-value cosine similarity blending.
      NOTE ON CORRECTNESS & POSITIONAL SEMANTICS:
      Adjacent KV merging blends embeddings that were computed with different RoPE
      phase angles at distinct time steps. Because RoPE applies non-linear rotations,
      averaging key vectors causes semantic smearing and positional distortion.
      Therefore, while mathematically computable and dimensionally compatible,
      similarity merging is marked EXPERIMENTAL / NOT VERIFIED for production serving.
    """
    def __init__(self, mode: str, max_capacity: int, sink_tokens: int = 4):
        self.mode = mode
        self.max_capacity = max_capacity
        self.sink_tokens = sink_tokens

    def merge_layer(self, keys: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        keys: [batch, num_heads, seq_len, head_dim]
        values: [batch, num_heads, seq_len, head_dim]

        When seq_len > max_capacity, similarity-based merging computes cosine similarity
        among candidates (protecting sink tokens and the most recent token), identifies
        the pair with highest cosine similarity, and blends their keys and values.
        """
        seq_len = keys.shape[-2]
        if seq_len <= self.max_capacity:
            return keys, values

        # 1. Standard Eviction (StreamingLLM)
        if self.mode == "eviction":
            sink_k = keys[:, :, :self.sink_tokens, :]
            sink_v = values[:, :, :self.sink_tokens, :]
            recent_count = self.max_capacity - self.sink_tokens
            recent_k = keys[:, :, -recent_count:, :]
            recent_v = values[:, :, -recent_count:, :]
            return torch.cat([sink_k, recent_k], dim=-2), torch.cat([sink_v, recent_v], dim=-2)

        # 2. Similarity-Based KV Merging
        elif self.mode == "merging":
            # Target is to reduce seq_len down to max_capacity
            # Protect sink tokens at the front and recent tokens at the end
            # Merge candidate tokens in the middle region: [sink_tokens, -protect_recent]
            protect_recent = 8
            if seq_len - (self.sink_tokens + protect_recent) < 2:
                # Fallback to eviction if merge region is too narrow
                sink_k = keys[:, :, :self.sink_tokens, :]
                sink_v = values[:, :, :self.sink_tokens, :]
                recent_count = self.max_capacity - self.sink_tokens
                return torch.cat([sink_k, keys[:, :, -recent_count:, :]], dim=-2), \
                       torch.cat([sink_v, values[:, :, -recent_count:, :]], dim=-2)

            k_curr = keys
            v_curr = values

            # Iteratively or block-merge until size is max_capacity
            num_to_merge = k_curr.shape[-2] - self.max_capacity
            for _ in range(num_to_merge):
                cand_len = k_curr.shape[-2]
                cand_start = self.sink_tokens
                cand_end = cand_len - protect_recent
                if cand_end <= cand_start + 1:
                    break

                # Compute cosine similarities between consecutive adjacent candidates
                cand_k = k_curr[:, :, cand_start:cand_end, :] # [B, H, N, D]
                k1 = cand_k[:, :, :-1, :]
                k2 = cand_k[:, :, 1:, :]

                # Mean similarity across heads and batch
                sim = F.cosine_similarity(k1, k2, dim=-1).mean(dim=(0, 1)) # [N-1]
                best_idx = torch.argmax(sim).item() # index in cand_k
                merge_pos = cand_start + best_idx

                # Average the best pair
                merged_k = 0.5 * (k_curr[:, :, merge_pos:merge_pos+1, :] + k_curr[:, :, merge_pos+1:merge_pos+2, :])
                merged_v = 0.5 * (v_curr[:, :, merge_pos:merge_pos+1, :] + v_curr[:, :, merge_pos+1:merge_pos+2, :])

                # Reconstruct cache without increasing size
                k_curr = torch.cat([
                    k_curr[:, :, :merge_pos, :],
                    merged_k,
                    k_curr[:, :, merge_pos+2:, :]
                ], dim=-2)

                v_curr = torch.cat([
                    v_curr[:, :, :merge_pos, :],
                    merged_v,
                    v_curr[:, :, merge_pos+2:, :]
                ], dim=-2)

            # Safety clamp in case candidate region was constrained
            if k_curr.shape[-2] > self.max_capacity:
                sink_k = k_curr[:, :, :self.sink_tokens, :]
                sink_v = v_curr[:, :, :self.sink_tokens, :]
                recent_count = self.max_capacity - self.sink_tokens
                k_curr = torch.cat([sink_k, k_curr[:, :, -recent_count:, :]], dim=-2)
                v_curr = torch.cat([sink_v, v_curr[:, :, -recent_count:, :]], dim=-2)

            return k_curr, v_curr

        else:
            return keys, values

    def apply(self, past_key_values: DynamicCache):
        if self.max_capacity is None or past_key_values is None:
            return
        if hasattr(past_key_values, "layers"):
            for layer in past_key_values.layers:
                if getattr(layer, "is_initialized", False) and layer.keys is not None:
                    layer.keys, layer.values = self.merge_layer(layer.keys, layer.values)

def benchmark_compression(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 120,
    compressor: Optional[KVCompressor] = None,
    device: str = "cuda"
) -> Dict[str, Any]:
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

    current_position = prompt_len

    # Prefill step
    with torch.no_grad():
        out = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        next_token = out.logits[:, -1:, :].argmax(dim=-1)

    generated_tokens = [next_token.item()]
    step_logits = [out.logits[:, -1:, :]]

    if compressor is not None:
        compressor.apply(past_key_values)

    # Decoding steps with monotonic position tracking
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

            if compressor is not None:
                compressor.apply(past_key_values)

    if device == "cuda":
        torch.cuda.synchronize()
        mem_post = torch.cuda.memory_allocated() / (1024 ** 2)
        mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        mem_post = 0.0
        mem_peak = 0.0

    t_end = time.perf_counter()
    latency = t_end - t_start

    num_generated = len(generated_tokens)
    throughput = num_generated / latency if latency > 0 else 0
    kv_bytes, final_cache_seq_len = get_kv_cache_tensor_memory(past_key_values)
    kv_mb = kv_bytes / (1024 ** 2)

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    all_logits = torch.cat(step_logits, dim=1)[:, :-1, :]
    target_tokens = torch.tensor(generated_tokens[1:], device=device).unsqueeze(0)
    loss = F.cross_entropy(all_logits.view(-1, all_logits.shape[-1]), target_tokens.view(-1)).item()
    ppl = math.exp(min(loss, 100))

    return {
        "strategy": compressor.mode if compressor else "baseline",
        "capacity": compressor.max_capacity if compressor else prompt_len + num_generated,
        "prompt_len": prompt_len,
        "num_generated": num_generated,
        "final_cache_tokens": final_cache_seq_len,
        "kv_mb": kv_mb,
        "latency_sec": latency,
        "throughput": throughput,
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

    max_new_tokens = 150

    configs = [
        ("Full Cache Baseline", None),
        ("Token Eviction (Cap=40, Sink=4)", KVCompressor(mode="eviction", max_capacity=40, sink_tokens=4)),
        ("Similarity Merging (Cap=40, Sink=4)", KVCompressor(mode="merging", max_capacity=40, sink_tokens=4)),
        ("Token Eviction (Cap=64, Sink=4)", KVCompressor(mode="eviction", max_capacity=64, sink_tokens=4)),
        ("Similarity Merging (Cap=64, Sink=4)", KVCompressor(mode="merging", max_capacity=64, sink_tokens=4)),
    ]

    results = []
    print("=" * 80)
    print("STARTING EVICITON VS SIMILARITY MERGING EVALUATION")
    print("=" * 80)

    for name, comp in configs:
        print(f"--> Running: {name} ...")
        res = benchmark_compression(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            compressor=comp,
            device=device
        )
        results.append((name, res))

    print("\n" + "=" * 80)
    print("EMPIRICAL COMPARISON RESULTS")
    print("=" * 80)
    header = f"{'Strategy / Configuration':<36} | {'KV (MB)':<8} | {'SeqLen':<7} | {'Latency':<8} | {'Tok/s':<7} | {'PPL':<7}"
    print(header)
    print("-" * len(header))
    for name, r in results:
        print(f"{name:<36} | {r['kv_mb']:<8.4f} | {r['final_cache_tokens']:<7} | {r['latency_sec']:<8.3f} | {r['throughput']:<7.2f} | {r['perplexity']:<7.2f}")

    print("\n" + "=" * 80)
    print("DECODED TEXT QUALITY COMPARISON")
    print("=" * 80)
    for name, r in results:
        print(f"[{name}]")
        print(f"  {r['text'][:220].strip()}...\n")

if __name__ == "__main__":
    main()
