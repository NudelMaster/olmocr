# olmocr Proof & Text Finetuning Plan

## Goal

Extend the finetuned olmocr model beyond single mathematical equations to handle **entire proofs, mixed prose+math, and full handwritten text pages**. Use knowledge distillation (GPT-4o as teacher) to create reliable training data for the Qwen2.5-VL 7B student model.

---

## Architecture Context

The olmocr training pipeline follows this pattern:

```
Source Data (images/PDFs)
    |
    v
prepare_*.py  -->  paired {ID}.pdf + {ID}.md files
    |                  (single-page PDF + YAML front matter markdown)
    v
buildsilver.py + runopenaibatch.py + process_openai_batch_results.py
    |                  (GPT-4o knowledge distillation, optional)
    v
train.py       -->  SFT with YAML config + LoRA
    |
    v
grpo_train.py  -->  RL polish with reward functions (optional)
    |
    v
pipeline.py    -->  Production batch inference
dolmaviewer.py -->  Visual verification
```

### Critical Format Requirement

The training pipeline (`BaseMarkdownPDFDataset` in `olmocr/train/dataloader.py`) expects:

- A directory of **paired files**: `{ID}.pdf` + `{ID}.md`
- Each PDF must be **exactly 1 page**
- Each `.md` file must have **YAML front matter** matching the `PageResponse` dataclass:

```markdown
---
primary_language: en
is_rotation_valid: True
rotation_correction: 0
is_table: False
is_diagram: False
---
The actual text content with \( inline \) and \[ block \] math...
```

- LaTeX delimiters: `\(` `\)` for inline math, `\[` `\]` for block math (NOT `$` / `$$`)
- Tables: HTML format (NOT markdown tables)
- The existing `prepare_aida_calculus.py` (`olmocr/data/prepare_aida_calculus.py`) is the reference implementation for converting a HuggingFace image+text dataset into this format

---

## Phase 1: Data Gathering & Pre-processing

### Goal
Assemble a high-quality, mixed-modality pool of images converted into the olmocr training format (`.pdf` + `.md` pairs).

### Original Plan
> Use `filter.py` to standardize image resolutions, remove corrupted files, and filter non-English text.

### What Actually Needs to Happen

**`filter.py` (`olmocr/filter/filter.py`) is the wrong tool.** It operates exclusively on PDF files (language detection via extracted text, form detection, spam filtering). It has no image preprocessing capability and no resolution standardization.

**What to do instead:** Write a `prepare_proof_dataset.py` script modeled after `prepare_aida_calculus.py`. This script should:

1. Accept multiple HuggingFace dataset identifiers as input
2. For each `(image, ground_truth_text)` pair:
   - Convert the image to a single-page PDF using `convert_image_to_pdf_bytes()` from `olmocr/image_utils.py`
   - Write the ground truth as a `.md` file with YAML front matter
3. Output the standard `train/` + `eval/` directory structure

### Source Datasets

| Dataset | Role | Notes |
|---------|------|-------|
| English-Handwritten-Math-Notes | Structural core (prose + math interleaved) | Ground truth must be formatted with `\(` `\)` / `\[` `\]` delimiters |
| MathWriting / Aida Calculus | Math symbol dictionary | Aida already has a prepare script (`prepare_aida_calculus.py`); can reuse |
| HME100K | Noise injection (camera distortions) | Image quality varies; good for robustness |

### Key Decisions

- **Image resolution**: The training pipeline handles this via the `PDFRenderer` pipeline step (`target_longest_image_dim: 1288`), so no pre-resize is needed
- **Corruption filtering**: Add basic PIL image validation in the prepare script (try to open, verify mode/size)
- **Language filtering**: Not needed at prepare time since all source datasets are English; the pipeline's `FilterOutRotatedDocuments` step handles rotation issues
- **Ground truth format for proofs**: Unlike Aida (which wraps everything in `\[ ... \]` display math), proof datasets will need mixed content — paragraphs of English text with inline `\( ... \)` and block `\[ ... \]` math interspersed

### Script Location
`olmocr/data/prepare_proof_dataset.py`

### Estimated Output
- Combined pool of 50K-100K paired `.pdf` + `.md` files across all sources

---

## Phase 2: Synthetic Edge-Case Augmentation

### Goal
Fill gaps where real-world data falls short with programmatically generated examples.

### Original Plan
> Use `mine_html_templates.py` to procedurally generate synthetic images of edge-case mathematics.

### What Actually Needs to Happen

**`mine_html_templates.py` (`olmocr/bench/synth/mine_html_templates.py`) is the wrong tool.** It:
- Takes real PDFs as input (not a generator)
- Uses the Claude API to extract HTML structure from existing documents
- Outputs bench-format JSONL (for `grpo_train.py`), NOT `.pdf` + `.md` pairs (for `train.py`)
- Costs money per call (Anthropic API)

**What to do instead:** Write a `generate_synthetic_proofs.py` script that:

1. Uses LaTeX templates to generate synthetic proof documents
2. Compiles them with `pdflatex` to produce single-page PDFs
3. Writes the `.md` ground truth derived directly from the LaTeX source (perfect labels, zero API cost)

### Template Categories

| Category | Example Content | Why It's Needed |
|----------|----------------|-----------------|
| Large matrices | 4x4, 5x5 matrices with mixed entries | Rare in handwriting datasets |
| Piecewise functions | Nested cases with conditions | Complex vertical layout |
| Multi-step proofs | "Given... Prove... Proof: Step 1... Step 2..." | Core use case |
| Theorem/Lemma blocks | Statement + proof structure | Formal math document structure |
| Mixed prose + equations | "Substituting (3) into (2), we get..." | Natural proof writing style |
| Long chain equalities | `a = b = c = ... = z` with justifications | Common in student work |

### Key Decisions

- **Rendering approach**: `pdflatex` -> single-page PDF (cleanest approach, guaranteed valid LaTeX)
- **Handwriting simulation**: Consider using `augraphy` (already a dependency in `[train]` extras) to add realistic handwriting-like distortions to the rendered PDFs
- **Volume**: Generate 2K-5K synthetic samples to supplement real data

### Script Location
`olmocr/data/generate_synthetic_proofs.py`

### Alternative Approach
If you want distorted/handwritten-style output rather than typeset LaTeX, you could:
- Render LaTeX to images
- Apply `augraphy` transformations (paper texture, ink bleed, skew, etc.)
- Convert back to PDF via `convert_image_to_pdf_bytes()`

---

## Phase 3: Knowledge Distillation (The "Teacher" Phase)

### Goal
Use GPT-4o to generate high-quality ground truth labels for images that lack them or have noisy labels.

### Original Plan
> Inject a "Golden Prompt" (instructing `$` / `$$` delimiters) into `buildsilver.py`, which formats outputs into Base64 JSONL.

### What Actually Needs to Happen

**`buildsilver.py` (`olmocr/data/buildsilver.py`) IS the right tool**, but the workflow is a 3-step pipeline, not a single script:

```
buildsilver.py          ->  OpenAI Batch API request JSONL files
runopenaibatch.py       ->  Submits batches to OpenAI, monitors completion
process_openai_batch_results.py  ->  Converts results to .pdf + .md pairs
```

### Corrections to the Original Plan

1. **Do NOT use `$` / `$$` delimiters.** The entire olmocr ecosystem uses `\(` `\)` for inline and `\[` `\]` for block math. The existing silver prompt `v3_simple` (`olmocr/prompts/prompts.py:50-63`) already enforces this convention. The `ReformatLatexBoldItalic` pipeline step in training also expects this format.

2. **The output is NOT direct "Base64 JSONL for training."** `buildsilver.py` produces batch API request files. The final training-ready `.pdf` + `.md` pairs come from `process_openai_batch_results.py`.

3. **You may want a custom prompt for proof-heavy content.** Add a new function to `olmocr/prompts/prompts.py`:

```python
def build_openai_silver_data_prompt_v4_proofs(page_width: int, page_height: int) -> str:
    return (
        f"Attached is the image of one page of a document containing a mathematical proof or worked solution."
        f"Return the plain text representation of this document as if you were reading it naturally.\n"
        f"Preserve the logical structure: theorem statements, proof steps, and explanations.\n"
        f"Turn equations and math symbols into LaTeX, using \\( and \\) for inline math and \\[ and \\] for block math. "
        f"Do NOT use $ or $$ delimiters. Do NOT use unicode math symbols.\n"
        f"Convert tables into HTML format.\n"
        f"Read any natural handwriting carefully.\n"
        f"If there is no text at all, output null.\n"
        f"Do not hallucinate.\n"
        f"Page width: {page_width}, Page height: {page_height}"
    )
```

### Workflow

1. Randomly sample 5K-10K images from the Phase 1 + Phase 2 pool (the ones that need better labels or lack labels entirely)
2. These must already be in PDF format (done by Phase 1 prepare script)
3. Run `buildsilver.py` with `--path_list` pointing to the sampled PDFs
4. Run `runopenaibatch.py` to submit and monitor the OpenAI batch
5. Run `process_openai_batch_results.py` to convert results to `.pdf` + `.md` pairs
6. Merge these silver-labeled pairs back into the training pool

### Key Decisions

- **When to use distillation vs. existing labels**: Use distillation only for samples where ground truth is missing, noisy, or in the wrong format. Datasets with clean ground truth (like Aida Calculus) can skip this phase.
- **Cost estimate**: GPT-4o batch API is ~$1.25/1M input tokens + $5/1M output tokens (half price of real-time). 10K images at ~2K tokens each = ~$25 input + ~$50 output = **~$75 total**.

### Script Locations
- `olmocr/data/buildsilver.py` (existing)
- `olmocr/data/runopenaibatch.py` (existing)
- `olmocr/data/process_openai_batch_results.py` (existing)
- `olmocr/prompts/prompts.py` (add new prompt function)

---

## Phase 4: Model Training (The "Student" Phase)

### Goal
Train the Qwen2.5-VL 7B model on the combined dataset.

### Step A: Supervised Fine-Tuning (SFT)

**Script:** `olmocr/train/train.py` (correct tool)

**What to do:** Create a new YAML config derived from the existing Aida calculus config (`olmocr/train/configs/v0.4.0/qwen25_vl_olmocrv4_finetuning_aida_calculus.yaml`).

Key changes from the Aida config:

| Setting | Aida Config | New Config | Reason |
|---------|-------------|------------|--------|
| `root_dir` (train) | `aida-data/train-10k` | Path to combined Phase 1/2/3 data | New dataset |
| `root_dir` (eval) | `aida-data/eval-10k` | 10% split of combined data | New dataset |
| `collator_max_token_len` | `8192` | `16384` | Full proofs are much longer than single equations |
| `gradient_accumulation_steps` | `32` | `16` or `32` | Adjust based on dataset size |
| `num_train_epochs` | `1` | `2-3` | May need more passes for diverse data |
| `eval_steps` / `save_steps` | `100` | Adjust to dataset size | ~3 evals per epoch is ideal |

The data pipeline stays the same:
```yaml
pipeline:
  - name: FrontMatterParser
    front_matter_class: PageResponse
  - name: FilterOutRotatedDocuments
  - name: ReformatLatexBoldItalic
  - name: PDFRenderer
    target_longest_image_dim: 1288
  - name: RotationAugmentation
    probability: 0.02
  - name: NewYamlFinetuningPromptWithNoAnchoring
  - name: FrontMatterOutputFormat
  - name: InstructUserMessages
    prompt_first: true
  - name: Tokenizer
    masking_index: -100
    end_of_message_token: "<|im_end|>"
```

**Config location:** `olmocr/train/configs/v0.4.0/qwen25_vl_olmocrv4_finetuning_proofs.yaml`

**Run command:**
```bash
python -m olmocr.train.train --config olmocr/train/configs/v0.4.0/qwen25_vl_olmocrv4_finetuning_proofs.yaml
```

### Step B: GRPO Reinforcement Learning (Optional Polish)

**Script:** `olmocr/train/grpo_train.py` (correct tool)

**Important format difference:** GRPO training uses **bench-format data**, not `.pdf` + `.md` pairs:
```
bench_data/
├── dataset.jsonl       # Test definitions (PDF paths + expected content checks)
├── pdfs/               # PDF files referenced by the JSONL
└── claude_original/    # (Optional) Claude-generated reference transcriptions
```

This is a different format from the SFT training data. You would need to:
1. Set aside a hold-out set of math-heavy PDFs
2. Convert them to bench format using `olmocr/bench/scripts/workspace_to_bench.py` or manually
3. Optionally generate Claude reference transcriptions for the `claude_original/` directory

**Available reward functions** (configurable via CLI weights):

| Reward | Flag | Description |
|--------|------|-------------|
| Bench tests | `--reward_bench` | Runs content-matching tests from the JSONL |
| Edit distance | `--reward_bench_edit_distance` | Character-level similarity to reference |
| Medoid | `--reward_medoid` | Similarity to the median of multiple generations |
| Front matter | `--reward_front_matter` | Validates YAML front matter structure |
| Element count | `--reward_element_count` | Checks for expected math/table elements |
| EOS token | `--reward_eos` | Ensures proper end-of-sequence |

**Missing reward (to implement):** A `reward_latex_validity` function that checks whether generated LaTeX compiles without errors. This would go in `grpo_train.py` alongside the existing reward functions.

**Run command:**
```bash
python -m olmocr.train.grpo_train \
    --train_bench_data_folder /path/to/bench_data \
    --model_name /path/to/sft_checkpoint \
    --reward_bench 1.0 \
    --reward_front_matter 0.5 \
    --reward_element_count 0.5 \
    --num_generations 16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8
```

---

## Phase 5: Evaluation & Production

### Goal
Verify accuracy on unseen data and set up batch processing.

### Original Plan
> Take 50 phone photos, run `pipeline.py`, view with `dolmaviewer.py`.

### What Actually Needs to Happen

**`pipeline.py` processes PDFs, not raw images.** Phone photos must be wrapped as single-page PDFs first.

**The finetuned model (LoRA adapter) must be merged** before vLLM can serve it. Use `olmocr/train/compress_checkpoint.py` to merge LoRA weights into the base model.

### Workflow

1. **Convert phone photos to PDFs:**
   ```python
   from olmocr.image_utils import convert_image_to_pdf_bytes
   # For each photo:
   pdf_bytes = convert_image_to_pdf_bytes("photo.jpg")
   with open("photo.pdf", "wb") as f:
       f.write(pdf_bytes)
   ```

2. **Merge LoRA weights** (if using LoRA finetuning):
   ```bash
   python -m olmocr.train.compress_checkpoint \
       --input /path/to/checkpoint \
       --output /path/to/merged_model
   ```

3. **Option A — Small-scale eval (recommended for 50 images):**
   Use the existing `olmocr_test.ipynb` notebook with the new local inference cell. This avoids the vLLM server requirement and is simpler for small batches.

4. **Option B — Production-scale eval:**
   ```bash
   python -m olmocr.pipeline /path/to/workspace \
       --pdfs /path/to/phone_photos_as_pdfs/ \
       --model /path/to/merged_model \
       --markdown
   ```

5. **Visual verification:**
   ```bash
   python -m olmocr.viewer.dolmaviewer /path/to/workspace/results/*.jsonl \
       --output_dir /path/to/html_previews \
       --merge
   ```

---

## New Files to Create

| File | Phase | Purpose |
|------|-------|---------|
| `olmocr/data/prepare_proof_dataset.py` | 1 | Convert image+text datasets to `.pdf` + `.md` pairs |
| `olmocr/data/generate_synthetic_proofs.py` | 2 | Generate synthetic proof documents via LaTeX templates |
| `olmocr/train/configs/v0.4.0/qwen25_vl_olmocrv4_finetuning_proofs.yaml` | 4A | Training config for the proof dataset |
| (Optional) New prompt function in `olmocr/prompts/prompts.py` | 3 | Proof-specific silver data prompt |
| (Optional) LaTeX validity reward in `olmocr/train/grpo_train.py` | 4B | Reward function for GRPO |

## Existing Files to Reuse As-Is

| File | Phase | Purpose |
|------|-------|---------|
| `olmocr/data/prepare_aida_calculus.py` | 1 | Reference implementation / direct reuse for Aida data |
| `olmocr/data/buildsilver.py` | 3 | GPT-4o batch request generation |
| `olmocr/data/runopenaibatch.py` | 3 | OpenAI batch submission |
| `olmocr/data/process_openai_batch_results.py` | 3 | Batch result conversion to training format |
| `olmocr/train/train.py` | 4A | Supervised fine-tuning |
| `olmocr/train/grpo_train.py` | 4B | GRPO reinforcement learning |
| `olmocr/pipeline.py` | 5 | Batch inference |
| `olmocr/viewer/dolmaviewer.py` | 5 | Visual verification |
| `olmocr/image_utils.py` | 1, 5 | Image-to-PDF conversion utility |
| `olmocr/train/compress_checkpoint.py` | 5 | LoRA merge for vLLM serving |

---

## Estimated Timeline

| Phase | Work | Estimate |
|-------|------|----------|
| Phase 1 | Write prepare script, download datasets, convert | 2-3 days |
| Phase 2 | Write synthetic generator with LaTeX templates | 2-3 days |
| Phase 3 | Run distillation pipeline (mostly waiting for API) | 1-2 days |
| Phase 4A | Create config, run SFT training | 1 day setup + training time |
| Phase 4B | Set up GRPO (optional) | 1-2 days |
| Phase 5 | Evaluate and iterate | 1 day |
