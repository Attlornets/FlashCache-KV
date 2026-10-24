import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_kv_cache_size_bytes(past_key_values):
    """
    Computes total memory occupied by KV cache tensors in bytes.
    Handles:
    - Transformers 5.x DynamicCache with .layers (each with .keys and .values)
    - Transformers 4.x DynamicCache with .key_cache and .value_cache
    - Legacy tuple of tuples / lists
    """
    total_bytes = 0
    if past_key_values is None:
        return 0

    # Transformers 5.x DynamicCache (has .layers list of DynamicLayer objects)
    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if isinstance(keys, torch.Tensor):
                total_bytes += keys.numel() * keys.element_size()
            if isinstance(values, torch.Tensor):
                total_bytes += values.numel() * values.element_size()
        if total_bytes > 0:
            return total_bytes

    # Transformers 4.x DynamicCache
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        for k, v in zip(past_key_values.key_cache, past_key_values.value_cache):
            if isinstance(k, torch.Tensor):
                total_bytes += k.numel() * k.element_size()
            if isinstance(v, torch.Tensor):
                total_bytes += v.numel() * v.element_size()
        if total_bytes > 0:
            return total_bytes

    # Legacy tuple/list structure
    if isinstance(past_key_values, (tuple, list)):
        for layer in past_key_values:
            if isinstance(layer, (tuple, list)):
                for tensor in layer:
                    if isinstance(tensor, torch.Tensor):
                        total_bytes += tensor.numel() * tensor.element_size()
            elif isinstance(layer, torch.Tensor):
                total_bytes += layer.numel() * layer.element_size()

    return total_bytes

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Using a small, fast model suitable for local testing without large downloads
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading tokenizer & model: {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Load model with float16 on GPU, float32 on CPU
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    model.eval()

    prompt = "Explain the concept of key-value cache in large language models and why it is important for inference efficiency."
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)
    prompt_len = inputs["input_ids"].shape[-1]
    print(f"Prompt tokens: {prompt_len}")

    if device == "cuda":
        torch.cuda.synchronize()
        mem_before_gen = torch.cuda.memory_allocated() / (1024 ** 2)
    else:
        mem_before_gen = 0.0

    start_time = time.perf_counter()

    current_position = prompt_len

    # Generate with KV cache returned and monotonic position tracking
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
        )

    if device == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    latency = end_time - start_time

    generated_tokens = outputs.sequences[0][prompt_len:]
    num_generated = len(generated_tokens)
    tokens_per_sec = num_generated / latency if latency > 0 else 0

    # Calculate KV Cache size
    kv_cache = outputs.past_key_values
    kv_bytes = get_kv_cache_size_bytes(kv_cache)
    kv_mb = kv_bytes / (1024 ** 2)

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print("\n--- Generation Output ---")
    print(generated_text)
    print("\n--- Benchmark & Memory Metrics ---")
    print(f"Prompt length:           {prompt_len} tokens")
    print(f"Generated tokens:        {num_generated} tokens")
    print(f"Total sequence length:   {prompt_len + num_generated} tokens")
    print(f"Latency:                 {latency:.4f} s")
    print(f"Throughput:              {tokens_per_sec:.2f} tokens/s")
    print(f"KV Cache Tensor Memory:  {kv_mb:.4f} MB ({kv_bytes:,} bytes)")

    if device == "cuda":
        mem_after_gen = torch.cuda.memory_allocated() / (1024 ** 2)
        peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"GPU Allocated (Pre-gen): {mem_before_gen:.2f} MB")
        print(f"GPU Allocated (Current): {mem_after_gen:.2f} MB")
        print(f"GPU Peak Allocated:      {peak_gpu_mem:.2f} MB")

if __name__ == "__main__":
    main()
