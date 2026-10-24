# Contributing to FlashCache-KV

Contributions to FlashCache-KV are welcome.

## Guidelines
1. **Mathematical Honesty**: Always separate structurally valid implementations from quality-preserving ones. If a compression or merging strategy exhibits semantic distortion or RoPE phase drift, clearly document it as experimental.
2. **Deterministic & Profiling Standards**: Wrap timed GPU sections with `torch.cuda.synchronize()` and run a warm-up pass before collecting latency metrics.
3. **Regression Tests**: Before opening a PR, ensure all unit and regression tests pass:
   ```bash
   python -m unittest discover -s . -p "test_*.py"
   ```
4. **Environment Hygiene**: Never commit model weights, local cache files, or environment files (`.env`).
