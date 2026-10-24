# FlashCache-KV

**Adaptive KV-cache compaction for memory-efficient LLM inference.**

FlashCache-KV is an experimental project exploring how to reduce the memory occupied by a language model's key-value (KV) cache during autoregressive text generation.

The project currently investigates token eviction, cache-size limits, attention sink retention, and experimental similarity-based merging. It includes benchmark scripts, a Streamlit dashboard, and tests for the implemented cache-management logic.

> **Research prototype:** Memory reduction does not guarantee equivalent generation quality. Token eviction can remove information required to answer questions about earlier parts of a conversation. Similarity-based merging is experimental and is not recommended for production use.

## Features

- **KV-cache eviction:** Bound the number of retained cache entries using configurable limits.
- **Attention sink retention:** Preserve a configurable number of initial cache entries.
- **Benchmarking:** Compare cache size, generation throughput, latency, and selected quality checks.
- **Long-context evaluation:** Test retrieval of information placed at different positions in a prompt.
- **Interactive dashboard:** Explore reference measurements and cache-compaction trade-offs.
- **Automated tests:** Run the project's test suite before making changes.

## How it works

During autoregressive inference, a transformer stores key and value tensors from previous tokens so that they can be reused when generating subsequent tokens. The cache can grow with sequence length.

FlashCache-KV explores limiting this growth by retaining a subset of cached entries. This reduces the number of stored KV entries, but discarded entries may contain information that the model later needs.

The project distinguishes three approaches:

1. **Full KV cache:** Retain the model's cache without the project's eviction policy.
2. **Token eviction:** Retain a bounded set of entries, optionally including initial attention-sink entries.
3. **Similarity-based merging:** Experimental attempts to combine similar entries. This approach requires further correctness and quality evaluation.

## Repository structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard |
| `benchmark_kv.py` | KV-cache benchmarking |
| `benchmark_long_context.py` | Long-context benchmark |
| `benchmark_merging.py` | Similarity-merging experiments |
| `evaluate_eviction.py` | Eviction evaluation |
| `test_flashcache.py` | Automated tests |
| `long_context_benchmark_results.json` | Saved benchmark results |
| `requirements.txt` | Python dependencies |
| `pyproject.toml` | Project metadata and development dependencies |

## Installation

Python 3.10 or newer is specified in the project metadata. Use a virtual environment to isolate dependencies.

### Windows (PowerShell)

```powershell
git clone https://github.com/Attlornets/FlashCache-KV.git
cd FlashCache-KV

py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Python 3.10 is not installed, use another Python version compatible with the project's dependencies.

### Linux or macOS

```bash
git clone https://github.com/Attlornets/FlashCache-KV.git
cd FlashCache-KV

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Hardware note:** PyTorch installation differs by operating system and accelerator. For NVIDIA GPUs, install a PyTorch build compatible with your system's driver and supported CUDA runtime, following the official [PyTorch installation selector](https://pytorch.org/get-started/locally/). CPU-only execution may be possible, but model loading and benchmarks can be slower. Apple Silicon support must be validated with the installed PyTorch and model dependencies.

These installation steps should be considered provisional until tested on clean Windows, Linux, and macOS environments.

## Run the dashboard

From the repository directory, with the virtual environment activated:

```bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal, typically `http://localhost:8501`.

## Run tests and benchmarks

Run the automated tests:

```bash
python -m pytest -q
```

Run individual benchmark scripts only after checking their command-line arguments and resource requirements:

```bash
python benchmark_kv.py --help
python benchmark_long_context.py --help
python benchmark_merging.py --help
python evaluate_eviction.py --help
```

Some scripts may require a model download, an internet connection, and sufficient RAM or GPU memory. Benchmark values depend on the model, prompt, sequence length, device, precision, software versions, and measurement methodology.

## Benchmark interpretation

The repository includes historical measurements collected on a specific test setup. Treat these as **reference results, not predictions of performance on your own machine**.

A meaningful comparison should record:

- CPU and GPU model, RAM, and available accelerator memory.
- Operating system, Python, PyTorch, and Transformers versions.
- Model identifier, model precision, prompt length, and generated-token count.
- Cache policy, capacity, and attention-sink configuration.
- KV-cache memory, peak device memory, latency, and throughput.
- Quality checks, including whether expected information can still be retrieved.

KV-cache memory is only one part of total inference memory. Reducing the cache does not necessarily reduce total GPU memory by the same percentage, nor does it guarantee faster generation.

### Known limitations

- **Information loss:** Eviction may discard older context needed for accurate responses.
- **Long-context retrieval:** The included reference evaluation reports failures to retrieve a test passkey under the tested eviction configurations, despite successful retrieval with the full cache. See `long_context_benchmark_results.json`.
- **Similarity merging:** Experimental; merging keys from different positional contexts may introduce correctness issues.
- **Hardware portability:** Dependency and accelerator compatibility must be tested on each supported platform.
- **Benchmark reproducibility:** Historical measurements should not be presented as live measurements from the current machine.

## Development

1. Create and activate a virtual environment.
2. Install the project dependencies.
3. Run the test suite.
4. Reproduce benchmarks on your own hardware.
5. Document environment details and any changes in results.

Please report reproducible bugs with the operating system, Python version, package versions, hardware, command executed, and complete error output.

## Roadmap

- [ ] Expand documentation and reproducible setup instructions.
- [ ] Separate historical reference results from current-machine measurements.
- [ ] Improve hardware and accelerator detection.
- [ ] Establish tested dependency versions and compatibility checks.
- [ ] Validate dashboard behavior with automated UI tests.
- [ ] Expand long-context quality evaluations.
- [ ] Validate installation on Windows, Linux, and macOS.
- [ ] Further investigate position-aware similarity-based cache merging.

## Disclaimer

FlashCache-KV is an experimental research and engineering project. Benchmark results are workload-specific and should not be interpreted as guarantees of production performance, memory savings, or output quality.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
