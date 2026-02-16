"""
Prepare the deepcopy/Aida-Calculus-Math-Handwriting HuggingFace dataset
for olmOCR finetuning.

Converts each (image, latex) pair into the expected training format:
  - Single-page PDF of the handwritten math image
  - Markdown file with YAML front matter + ground truth LaTeX

Output structure:
    destination/
    ├── train/
    │   ├── 000000.pdf
    │   ├── 000000.md
    │   ├── 000001.pdf
    │   ├── 000001.md
    │   └── ...
    └── eval/
        ├── 000000.pdf
        ├── 000000.md
        └── ...

Usage:
    python -m olmocr.data.prepare_aida_calculus \
        --destination ~/aida-calculus-data \
        --limit 1000 \
        --eval-ratio 0.1 \
        --seed 42 \
        --parallel 4
"""

import argparse
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Set, Tuple

from tqdm import tqdm

from olmocr.image_utils import convert_image_to_pdf_bytes

# Front matter columns, matching prepare_workspace.py and PageResponse dataclass
PAGE_RESPONSE_COLUMNS = [
    "primary_language",
    "is_rotation_valid",
    "rotation_correction",
    "is_table",
    "is_diagram",
]


def write_front_matter_md(latex: str, md_path: Path) -> None:
    """Write a markdown file with YAML front matter and LaTeX content.

    Follows the exact format expected by FrontMatterParser with
    front_matter_class=PageResponse:
      - 5 metadata fields in the YAML front matter block
      - natural_text as the body after the closing '---'

    The LaTeX is wrapped in \\[ ... \\] display math delimiters per
    olmOCR conventions (the pipeline normalizes $ to \\( \\) anyway).
    """
    # Some dataset entries have a leading '=' sign (e.g. "=\lim_{...}"),
    # strip it for cleaner LaTeX
    latex_clean = latex.strip()
    if latex_clean.startswith("="):
        latex_clean = latex_clean[1:].strip()

    # Wrap in display math delimiters
    natural_text = f"\\[ {latex_clean} \\]"

    # Write to a temp file first, then atomically rename.
    # This prevents a partial .md from being left behind on interrupt.
    tmp_md = md_path.with_suffix(".md.tmp")
    with open(tmp_md, "w", encoding="utf-8") as f:
        # Write YAML front matter matching prepare_workspace.py format (lines 243-279)
        f.write("---\n")
        f.write("primary_language: en\n")
        f.write("is_rotation_valid: True\n")
        f.write("rotation_correction: 0\n")
        f.write("is_table: False\n")
        f.write("is_diagram: False\n")
        f.write("---\n")
        f.write(natural_text)
    tmp_md.rename(md_path)


def save_image_as_pdf(image, pdf_path: Path) -> None:
    """Convert a PIL Image to a single-page PDF file.

    Saves the image to a temporary JPEG, then uses convert_image_to_pdf_bytes()
    (wrapping img2pdf) to produce a valid PDF. This is the same approach used
    by prepare_loc_transcripts.py and prepare_national_archive_transcripts.py.

    Args:
        image: PIL Image object from the HuggingFace dataset
        pdf_path: Destination path for the output PDF

    Raises:
        RuntimeError: If image conversion or PDF creation fails
    """
    tmp_path = None
    try:
        # img2pdf requires a file on disk; use a named temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)

            # Convert to RGB if necessary (handles grayscale, RGBA, palette modes)
            rgb_image = image.convert("RGB") if image.mode != "RGB" else image
            rgb_image.save(tmp_path, "JPEG", quality=95)

        # Convert to PDF using the same utility as other prepare_* scripts.
        # Write to a temp file first, then atomically rename to the final path.
        # This prevents a partial/corrupt PDF from being left behind if the
        # process is interrupted mid-write.
        pdf_bytes = convert_image_to_pdf_bytes(str(tmp_path))
        tmp_pdf = pdf_path.with_suffix(".pdf.tmp")
        with open(tmp_pdf, "wb") as f:
            f.write(pdf_bytes)
        tmp_pdf.rename(pdf_path)

    finally:
        # Always clean up temp files
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        # Clean up temp PDF if rename didn't happen
        tmp_pdf_path = pdf_path.with_suffix(".pdf.tmp")
        tmp_pdf_path.unlink(missing_ok=True)


def process_single_sample(
    local_idx: int,
    image,
    latex: str,
    output_dir: Path,
    processed_lock: threading.Lock,
    processed_set: Set[str],
) -> Tuple[str, bool, Optional[str]]:
    """Process a single (image, latex) sample into a PDF+MD pair.

    Creates:
      output_dir/{local_idx:06d}.pdf  (single-page PDF of the handwritten image)
      output_dir/{local_idx:06d}.md   (YAML front matter + LaTeX in \\[...\\])

    Args:
        local_idx: Index of this sample within its split (train or eval)
        image: PIL Image from the HuggingFace dataset
        latex: Ground truth LaTeX string
        output_dir: Directory to write the PDF and MD files
        processed_lock: Thread lock for the processed set
        processed_set: Set of already-processed sample IDs

    Returns:
        Tuple of (sample_id, success, error_message)
    """
    sample_id = f"{local_idx:06d}"

    # Skip if already processed (resume support)
    with processed_lock:
        if sample_id in processed_set:
            return (sample_id, True, None)

    pdf_path = output_dir / f"{sample_id}.pdf"
    md_path = output_dir / f"{sample_id}.md"

    # Skip if both files already exist and PDF is non-empty
    if pdf_path.exists() and md_path.exists() and pdf_path.stat().st_size > 0:
        with processed_lock:
            processed_set.add(sample_id)
        return (sample_id, True, None)

    try:
        # Convert image to single-page PDF
        save_image_as_pdf(image, pdf_path)

        # Write markdown with front matter + ground truth LaTeX
        write_front_matter_md(latex, md_path)

        # Verify the PDF was written correctly
        if not pdf_path.exists() or pdf_path.stat().st_size == 0:
            raise RuntimeError("PDF file is empty or missing after conversion")

        with processed_lock:
            processed_set.add(sample_id)
        return (sample_id, True, None)

    except Exception as e:
        # Clean up partial and temp files on failure
        pdf_path.unlink(missing_ok=True)
        md_path.unlink(missing_ok=True)
        pdf_path.with_suffix(".pdf.tmp").unlink(missing_ok=True)
        md_path.with_suffix(".md.tmp").unlink(missing_ok=True)
        return (sample_id, False, str(e))


def scan_existing_outputs(output_dir: Path) -> Set[str]:
    """Scan output directory to find already-processed samples.

    Follows the same pattern as prepare_loc_transcripts.py:
    looks for matching .pdf + .md pairs where the PDF is non-empty.
    """
    processed = set()
    if not output_dir.exists():
        return processed

    pdf_stems = {f.stem for f in output_dir.glob("*.pdf")}
    md_stems = {f.stem for f in output_dir.glob("*.md")}
    complete = pdf_stems.intersection(md_stems)

    for stem in complete:
        pdf_path = output_dir / f"{stem}.pdf"
        if pdf_path.stat().st_size > 0:
            processed.add(stem)

    return processed


def process_split(
    dataset,
    start_idx: int,
    end_idx: int,
    split_dir: Path,
    split_name: str,
    parallel: int,
) -> Tuple[int, int]:
    """Process a range of dataset samples into a split directory.

    Args:
        dataset: The HuggingFace dataset object
        start_idx: Start index in the dataset (inclusive)
        end_idx: End index in the dataset (exclusive)
        split_dir: Output directory for this split
        split_name: Name of the split (for logging)
        parallel: Number of parallel workers

    Returns:
        Tuple of (success_count, error_count)
    """
    count = end_idx - start_idx
    print(f"\nProcessing {split_name} split ({count} samples)...")

    # Scan for already-processed samples (resume support)
    processed_lock = threading.Lock()
    processed_set = scan_existing_outputs(split_dir)
    if processed_set:
        print(f"  Found {len(processed_set)} already processed, resuming...")

    success = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {}
        for i in range(start_idx, end_idx):
            local_idx = i - start_idx
            sample = dataset[i]
            future = executor.submit(
                process_single_sample,
                local_idx,
                sample["image"],
                sample["latex"],
                split_dir,
                processed_lock,
                processed_set,
            )
            futures[future] = local_idx

        with tqdm(total=count, desc=f"Converting {split_name}") as pbar:
            for future in as_completed(futures):
                sample_id, ok, err = future.result()
                if ok:
                    success += 1
                else:
                    errors += 1
                    if err:
                        tqdm.write(f"  Error on {sample_id}: {err}")
                pbar.update(1)

    print(f"  {split_name}: {success} succeeded, {errors} failed")
    return success, errors


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Aida-Calculus-Math-Handwriting dataset for olmOCR finetuning"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="deepcopy/Aida-Calculus-Math-Handwriting",
        help="HuggingFace dataset path (default: deepcopy/Aida-Calculus-Math-Handwriting)",
    )
    parser.add_argument(
        "--destination",
        type=str,
        required=True,
        help="Output directory. Will create train/ and eval/ subdirectories.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max samples to process. 0 means all. (default: 0)",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="Fraction of samples for eval split. (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling. (default: 42)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Number of parallel workers. (default: 4)",
    )

    args = parser.parse_args()

    if args.eval_ratio < 0 or args.eval_ratio >= 1:
        print("Error: --eval-ratio must be in [0, 1)")
        return

    if args.parallel < 1:
        print("Error: --parallel must be at least 1")
        return

    dest = Path(args.destination)
    train_dir = dest / "train"
    eval_dir = dest / "eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset from HuggingFace
    print(f"Loading dataset: {args.dataset_path}")
    from datasets import load_dataset

    dataset = load_dataset(args.dataset_path, split="train")

    # Shuffle with fixed seed for reproducible splits
    dataset = dataset.shuffle(seed=args.seed)

    # Apply sample limit
    total = len(dataset)
    if args.limit > 0:
        total = min(args.limit, total)
        dataset = dataset.select(range(total))

    # Compute split sizes
    eval_count = max(1, int(total * args.eval_ratio))
    train_count = total - eval_count

    print(f"Dataset size:   {total}")
    print(f"Train samples:  {train_count}")
    print(f"Eval samples:   {eval_count}")
    print(f"Workers:        {args.parallel}")
    print(f"Output:         {dest}")

    # Process train split
    train_ok, train_err = process_split(
        dataset, 0, train_count, train_dir, "train", args.parallel
    )

    # Process eval split
    eval_ok, eval_err = process_split(
        dataset, train_count, total, eval_dir, "eval", args.parallel
    )

    # Summary
    print(f"\nDone!")
    print(f"  Train: {train_dir}  ({train_ok} ok, {train_err} errors)")
    print(f"  Eval:  {eval_dir}  ({eval_ok} ok, {eval_err} errors)")
    print(f"\nNext steps:")
    print(f"  1. Update your training config's dataset.train[].root_dir to: {train_dir}")
    print(f"  2. Update your training config's dataset.eval[].root_dir to:  {eval_dir}")
    print(f"  3. Run: python -m olmocr.train.train --config <your_config.yaml>")


if __name__ == "__main__":
    main()
