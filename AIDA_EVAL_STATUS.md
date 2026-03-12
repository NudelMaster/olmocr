# AIDA Eval Status

This note documents the AIDA-Calculus evaluation work added on top of the existing olmOCR codebase and the reporting artifacts produced for the completed full eval run.

## What Already Existed In The Codebase

- `olmocr/bench/katex/render.py` already provided KaTeX rendering, symbol extraction, and spatial symbol matching.
- `olmocr/bench/tests.py` already provided `MathTest`, which extracts LaTeX from model output and checks it with the KaTeX-based pass/fail pipeline.
- `olmocr/bench/runners/run_transformers.py` already provided local Hugging Face inference for PDF-to-markdown OCR.
- `olmocr/bench/runners/run_server.py` already provided an OpenAI-compatible runner for server-backed inference.
- `olmocr/data/aida-data/eval-10k/` already contained the 1,000-sample evaluation subset.

## What Was Added

### 1. AIDA evaluation entrypoint

Added `olmocr/bench/eval_aida.py`.

This script:

- loads the paired `.pdf` and `.md` files from the AIDA eval set
- extracts the single ground-truth LaTeX expression from each markdown file
- runs inference with either local Transformers or an external vLLM/OpenAI-compatible server
- evaluates each prediction with `MathTest`
- writes per-sample output to JSONL
- writes an aggregate text summary with total processed count and pass rate

Expected outputs:

- `evaluation_results.jsonl`
- `run_summary.txt`

### 2. Inference-path updates for the eval

Updated `olmocr/bench/runners/run_transformers.py` to better match the finetuned inference path used elsewhere in the repo.

Changes made:

- added support for `prompt_template="yaml_v4"`
- wired `yaml_v4` to `build_no_anchoring_v4_yaml_prompt()`
- reordered multimodal message content to `text` then `image`, matching the main pipeline pattern
- kept a fallback so inference still works when FlashAttention 2 is unavailable

These changes were needed so the AIDA eval uses the same prompt family and a compatible local inference setup.

### 3. Server-backed vLLM path for eval

Updated `olmocr/bench/runners/run_server.py` and `olmocr/bench/eval_aida.py` so the AIDA eval can use the same external-server pattern described in `README.md`.

Changes made:

- added `yaml` and `yaml_v4` prompt support to `run_server.py`
- added YAML response parsing support to `run_server.py`
- normalized server URLs so `localhost:30024`, `http://localhost:30024`, and `http://localhost:30024/v1` all work
- added `--backend transformers|vllm` to `eval_aida.py`
- added `--server`, `--server-model`, and `--max-concurrent` to `eval_aida.py`
- kept the vLLM flow external to the evaluator instead of teaching `eval_aida.py` to spawn and manage a server itself

This is the lowest-friction way to add vLLM support while staying close to the repo's documented inference pattern.

## Sanity Check Status

- A one-sample run completed successfully on GPU 0 using `CUDA_VISIBLE_DEVICES=0`
- the single-sample sanity check passed at `100.00%`
- the output path under `inference_workspace/results/aida_eval/` was validated with a transformers smoke test
- a one-sample vLLM smoke test completed successfully and wrote outputs to `./inference_workspace/results/aida_eval/`

## Full Eval Status

- the full 1,000-sample AIDA eval has finished running
- results are stored in `./inference_workspace/results/aida_eval/`
- final output files:
  - `./inference_workspace/results/aida_eval/evaluation_results.jsonl`
  - `./inference_workspace/results/aida_eval/run_summary.txt`
- reported summary:
  - total examples processed: `1000`
  - tests passed: `537`
  - pass rate: `53.70%`

## Quantitative Results

| Metric | Value |
| --- | --- |
| Dataset | `olmocr/data/aida-data/eval-10k/` |
| Samples | `1000` |
| Passed | `537` |
| Failed | `463` |
| Pass rate | `53.70%` |
| Evaluation rule | Generated math must match the reference symbol layout under KaTeX rendering + spatial comparison |
| Normalization | Aggressive string cleanup is applied before scoring, but the final metric remains binary pass/fail |
| Per-sample outputs | `inference_workspace/results/aida_eval/evaluation_results.jsonl` |
| Summary file | `inference_workspace/results/aida_eval/run_summary.txt` |
| Report notebook | `inference_workspace/results/aida_eval/aida_eval_report.ipynb` |

## Base-Model Baseline Results

| Metric | Value |
| --- | --- |
| Model | `allenai/olmOCR-2-7B-1025` |
| Samples | `1000` |
| Passed | `237` |
| Failed | `763` |
| Pass rate | `23.70%` |
| Per-sample outputs | `inference_workspace/results/aida_eval_base/evaluation_results.jsonl` |
| Summary file | `inference_workspace/results/aida_eval_base/run_summary.txt` |
| Report notebook | `inference_workspace/results/aida_eval_base/aida_eval_base_report.ipynb` |

## Finetuned Vs Base Comparison

| Comparison metric | Value |
| --- | --- |
| Base model pass rate | `23.70%` (`237/1000`) |
| Finetuned model pass rate | `53.70%` (`537/1000`) |
| Absolute gain from finetuning | `+30.00` percentage points |
| Additional passed samples | `+300` |
| Relative lift in pass rate | about `2.27x` |
| Finetuned-only wins | `324` samples |
| Base-only wins | `24` samples |
| Shared passes | `213` samples |
| Shared failures | `439` samples |

## Reporting Artifacts

- `inference_workspace/results/aida_eval/aida_eval_report.ipynb` now contains the finetuned-model report notebook with quantitative tables, random passed/failed side-by-side visuals, failure-category plots, throughput estimates, and comparative notes against the paper's benchmark.
- `inference_workspace/results/aida_eval_base/aida_eval_base_report.ipynb` now contains the matching base-model analysis notebook with the same reporting structure for direct comparison.
- `inference_workspace/results/aida_eval/evaluation_results.jsonl` remains the canonical per-sample output for this run.
- `inference_workspace/results/aida_eval/run_summary.txt` remains the canonical aggregate text summary for this run.
- `inference_workspace/results/aida_eval_base/evaluation_results.jsonl` remains the canonical per-sample output for the base-model baseline.
- `inference_workspace/results/aida_eval_base/run_summary.txt` remains the canonical aggregate text summary for the base-model baseline.

## Base-Model Baseline On The Same 1K Eval

This is already supported by `olmocr/bench/eval_aida.py`.

### Recommended path: vLLM with the base model

This runs the exact same 1,000-sample `eval-10k` subset, but swaps the served model from the finetuned checkpoint to the raw base model.

Terminal 1:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve allenai/olmOCR-2-7B-1025 \
  --port 8000 \
  --served-model-name olmocr-base \
  --max-model-len 16384
```

Terminal 2:

```bash
python -m olmocr.bench.eval_aida \
  --backend vllm \
  --server http://localhost:8000/v1 \
  --server-model olmocr-base \
  --max-concurrent 2 \
  --results-path ./inference_workspace/results/aida_eval_base/evaluation_results.jsonl \
  --summary-path ./inference_workspace/results/aida_eval_base/run_summary.txt
```

Notes:

- `eval_aida.py` already defaults to `olmocr/data/aida-data/eval-10k/`, so this command evaluates the same 1K subset.
- A separate output directory (`inference_workspace/results/aida_eval_base/`) keeps the base-model baseline isolated from the finetuned run outputs.

### Local Transformers path: supported if the base model exists as a local directory

If the base model has already been downloaded locally, point `--model-path` at that local directory:

```bash
CUDA_VISIBLE_DEVICES=0 python -m olmocr.bench.eval_aida \
  --model-path /absolute/path/to/local/allenai_olmOCR-2-7B-1025 \
  --results-path ./inference_workspace/results/aida_eval_base/evaluation_results.jsonl \
  --summary-path ./inference_workspace/results/aida_eval_base/run_summary.txt
```

Important limitation:

- The current `--model-path` argument is parsed as a filesystem `Path` and `eval_aida.py` checks that it exists locally before running the Transformers backend.
- That means the Transformers path does **not** currently accept a Hugging Face repo ID like `allenai/olmOCR-2-7B-1025` directly.
- If direct repo-ID support is needed for the Transformers backend, change `--model-path` in `olmocr/bench/eval_aida.py` from `Path` to `str` and relax the local existence check so remote model IDs can pass through to `run_transformers()`.

## How To Run The Full 1K Eval

### Transformers backend

Use GPU 0 only:

```bash
CUDA_VISIBLE_DEVICES=0 python -m olmocr.bench.eval_aida \
  --results-path ./inference_workspace/results/aida_eval/evaluation_results.jsonl \
  --summary-path ./inference_workspace/results/aida_eval/run_summary.txt
```

### vLLM backend (recommended for the full 1K run)

This follows the documented repo pattern from `README.md`: run `vllm serve` separately, then point the evaluator at that OpenAI-compatible endpoint.

Terminal 1:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve ./olmocr-finetuned-model \
  --port 30024 \
  --served-model-name olmocr \
  --max-model-len 16384
```

Terminal 2:

```bash
python -m olmocr.bench.eval_aida \
  --backend vllm \
  --server http://localhost:30024/v1 \
  --server-model olmocr \
  --max-concurrent 4 \
  --results-path ./inference_workspace/results/aida_eval/evaluation_results.jsonl \
  --summary-path ./inference_workspace/results/aida_eval/run_summary.txt
```

If you want the output somewhere else, replace the `./inference_workspace/results/aida_eval` paths with your preferred destination.
If you used a different port or a different `--served-model-name` in Terminal 1, replace `http://localhost:30024/v1` and `olmocr` to match.

## Recommended vLLM Smoke Test

The simplest smoke test is to follow the same two-terminal pattern above and reduce the eval to one sample.

Terminal 1:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve ./olmocr-finetuned-model \
  --port 30024 \
  --served-model-name olmocr \
  --max-model-len 16384
```

Optional readiness check:

```bash
curl http://localhost:30024/v1/models
```

Terminal 2:

```bash
python -m olmocr.bench.eval_aida \
  --backend vllm \
  --server http://localhost:30024/v1 \
  --server-model olmocr \
  --limit 1 \
  --max-concurrent 1 \
  --results-path ./inference_workspace/results/aida_eval/smoke_vllm_results.jsonl \
  --summary-path ./inference_workspace/results/aida_eval/smoke_vllm_summary.txt
```

For a slightly stronger smoke test, increase to `--limit 8 --max-concurrent 4`.

## Terminal 2 Command For The Full Eval

Once the vLLM server is already running, use this command in Terminal 2 to evaluate the full AIDA `eval-10k` subset (all 1,000 samples):

```bash
python -m olmocr.bench.eval_aida \
  --backend vllm \
  --server http://localhost:30024/v1 \
  --server-model olmocr \
  --max-concurrent 4 \
  --results-path ./inference_workspace/results/aida_eval/evaluation_results.jsonl \
  --summary-path ./inference_workspace/results/aida_eval/run_summary.txt
```

This runs all files because it does not pass `--limit`.

If you started vLLM in a README-style setup on port `8000`, then use:

```bash
python -m olmocr.bench.eval_aida \
  --backend vllm \
  --server http://localhost:8000/v1 \
  --server-model olmocr \
  --max-concurrent 4 \
  --results-path ./inference_workspace/results/aida_eval/evaluation_results.jsonl \
  --summary-path ./inference_workspace/results/aida_eval/run_summary.txt
```

## Remaining Optional Follow-up

1. If desired, export the two executed notebooks as HTML or static slide assets for presentations.
2. For future consideration, re-run eval with image-render settings matched to finetuning if a controlled ablation is needed.
3. For future consideration, test longer finetuning and/or more AIDA training data if the base-vs-finetuned comparison shows headroom.
4. If desired, refine the heuristic failure taxonomy in `inference_workspace/results/aida_eval/aida_eval_report.ipynb` and `inference_workspace/results/aida_eval_base/aida_eval_base_report.ipynb` for a more granular paper-ready error analysis.
5. If desired, add a CLI flag for selecting a CUDA device directly in `eval_aida.py` instead of relying on `CUDA_VISIBLE_DEVICES`.

## Notes

- The evaluator uses the existing binary math pass/fail logic from `MathTest`; it does not use edit distance, ROUGE, or any soft matching metric.
- The transformers backend evaluates sequentially and benefits from model caching after the first sample is loaded.
- The vLLM backend is intentionally modeled after the repo's documented external-server usage rather than trying to embed server lifecycle management into `eval_aida.py`.
- Local environment warnings may still appear during startup (for example tokenizer or processor warnings), but they did not block the one-sample sanity check.
