#!/usr/bin/env python3

# Usage example for the full 1,000-sample eval on GPU 0 with local Transformers:
# CUDA_VISIBLE_DEVICES=0 python -m olmocr.bench.eval_aida \
#   --results-path ./inference_workspace/results/aida_eval/evaluation_results.jsonl \
#   --summary-path ./inference_workspace/results/aida_eval/run_summary.txt
#
# Usage example for the full 1,000-sample eval with a running vLLM server on GPU 0:
# Terminal 1:
# CUDA_VISIBLE_DEVICES=0 vllm serve ./olmocr-finetuned-model --port 30024 --served-model-name olmocr --max-model-len 16384
# Terminal 2:
# python -m olmocr.bench.eval_aida --backend vllm \
#   --server http://localhost:30024/v1 \
#   --server-model olmocr \
#   --results-path ./inference_workspace/results/aida_eval/evaluation_results.jsonl \
#   --summary-path ./inference_workspace/results/aida_eval/run_summary.txt

from __future__ import annotations

import asyncio
import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from olmocr.bench.runners.run_server import run_server
from olmocr.bench.runners.run_transformers import run_transformers
from olmocr.bench.tests import MathTest, TestType
from olmocr.train.dataloader import FrontMatterParser

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "olmocr" / "data" / "aida-data" / "eval-10k"
DEFAULT_MODEL_PATH = REPO_ROOT / "olmocr-finetuned-model"
DEFAULT_RESULTS_DIR = REPO_ROOT / "inference_workspace" / "results" / "aida_eval"
DEFAULT_RESULTS_PATH = DEFAULT_RESULTS_DIR / "evaluation_results.jsonl"
DEFAULT_SUMMARY_PATH = DEFAULT_RESULTS_DIR / "run_summary.txt"
DEFAULT_VLLM_SERVER = "http://localhost:30024/v1"
DEFAULT_VLLM_MODEL = "olmocr"

MATH_PATTERNS = (
    r"\\\[(.+?)\\\]",
    r"\\\((.+?)\\\)",
    r"\$\$(.+?)\$\$",
    r"\$(.+?)\$",
)

FRONT_MATTER_PARSER = FrontMatterParser()


@dataclass(frozen=True, slots=True)
class AIDASample:
    image_id: str
    pdf_path: Path
    ground_truth_latex: str


def extract_ground_truth_latex(md_path: Path) -> str:
    markdown_content = md_path.read_text(encoding="utf-8")
    _, body_text = FRONT_MATTER_PARSER._extract_front_matter_and_text(markdown_content)

    equations: list[str] = []
    for pattern in MATH_PATTERNS:
        equations.extend(match.strip() for match in re.findall(pattern, body_text, flags=re.DOTALL) if match.strip())

    if len(equations) != 1:
        raise ValueError(f"Expected exactly one math expression in {md_path}, found {len(equations)}")

    return equations[0]


def load_samples(dataset_dir: Path, limit: int | None) -> list[AIDASample]:
    pdf_paths = sorted(dataset_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {dataset_dir}")

    if limit is not None:
        pdf_paths = pdf_paths[:limit]

    samples: list[AIDASample] = []
    for pdf_path in pdf_paths:
        md_path = pdf_path.with_suffix(".md")
        if not md_path.exists():
            raise FileNotFoundError(f"Missing ground truth markdown for {pdf_path.name}")

        samples.append(
            AIDASample(
                image_id=pdf_path.stem,
                pdf_path=pdf_path,
                ground_truth_latex=extract_ground_truth_latex(md_path),
            )
        )

    return samples


def build_math_test(sample: AIDASample) -> MathTest:
    return MathTest(
        pdf=sample.pdf_path.name,
        page=1,
        id=sample.image_id,
        type=TestType.MATH.value,
        math=sample.ground_truth_latex,
    )


def score_model_output(sample: AIDASample, model_output: str) -> bool:
    passed, _ = build_math_test(sample).run(model_output)
    return passed


def build_record(sample: AIDASample, model_output: str, passed: bool) -> dict[str, object]:
    return {
        "image_id": sample.image_id,
        "ground_truth_latex": sample.ground_truth_latex,
        "model_output": model_output,
        "test_passed": passed,
    }


def evaluate_sample_transformers(sample: AIDASample, args: argparse.Namespace) -> dict[str, object]:
    model_output = ""

    try:
        model_output = run_transformers(
            pdf_path=str(sample.pdf_path),
            page_num=1,
            model_name=str(args.model_path),
            temperature=args.temperature,
            target_longest_image_dim=args.target_longest_image_dim,
            prompt_template=args.prompt_template,
            response_template="plain",
        )
        passed = score_model_output(sample, model_output)
    except Exception as exc:
        print(f"Failed on {sample.image_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
        passed = False

    return build_record(sample, model_output, passed)


async def evaluate_sample_vllm(sample: AIDASample, args: argparse.Namespace) -> dict[str, object]:
    model_output = ""

    try:
        model_output = await run_server(
            pdf_path=str(sample.pdf_path),
            page_num=1,
            server=args.server,
            model=args.server_model,
            temperature=args.temperature,
            target_longest_image_dim=args.target_longest_image_dim,
            prompt_template=args.prompt_template,
            response_template="plain",
        )
        passed = await asyncio.to_thread(score_model_output, sample, model_output)
    except Exception as exc:
        print(f"Failed on {sample.image_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
        passed = False

    return build_record(sample, model_output, passed)


def ensure_output_dirs(args: argparse.Namespace) -> None:
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)


def write_record(results_file, record: dict[str, object]) -> None:
    results_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    results_file.flush()


def evaluate_samples_transformers(samples: list[AIDASample], args: argparse.Namespace) -> tuple[int, int]:
    processed = 0
    passed = 0

    with args.results_path.open("w", encoding="utf-8") as results_file:
        for sample in tqdm(samples, desc="Evaluating AIDA", unit="sample"):
            record = evaluate_sample_transformers(sample, args)
            processed += 1
            passed += int(record["test_passed"])
            write_record(results_file, record)

    return processed, passed


async def evaluate_samples_vllm(samples: list[AIDASample], args: argparse.Namespace) -> tuple[int, int]:
    processed = 0
    passed = 0
    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def evaluate_with_limit(sample: AIDASample) -> dict[str, object]:
        async with semaphore:
            return await evaluate_sample_vllm(sample, args)

    tasks = [asyncio.create_task(evaluate_with_limit(sample)) for sample in samples]

    with args.results_path.open("w", encoding="utf-8") as results_file:
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating AIDA", unit="sample"):
            record = await future
            processed += 1
            passed += int(record["test_passed"])
            write_record(results_file, record)

    return processed, passed


def write_summary(summary_path: Path, processed: int, passed: int) -> float:
    pass_rate = (passed / processed * 100.0) if processed else 0.0
    summary_lines = [
        f"Total examples processed: {processed}",
        f"Tests passed: {passed}",
        f"Pass rate: {pass_rate:.2f}%",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return pass_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AIDA-Calculus with olmOCR's KaTeX math verification pipeline.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR, help=f"Path to AIDA eval set (default: {DEFAULT_DATASET_DIR})")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help=f"Path to local finetuned model (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument(
        "--backend",
        choices=("transformers", "vllm"),
        default="transformers",
        help="Inference backend to use for evaluation (default: transformers)",
    )
    parser.add_argument(
        "--server",
        type=str,
        default=DEFAULT_VLLM_SERVER,
        help=f"OpenAI-compatible server URL for vLLM mode (default: {DEFAULT_VLLM_SERVER})",
    )
    parser.add_argument(
        "--server-model",
        type=str,
        default=DEFAULT_VLLM_MODEL,
        help=f"Served model name to request in vLLM mode (default: {DEFAULT_VLLM_MODEL})",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help=f"Where to write per-sample JSONL results (default: {DEFAULT_RESULTS_PATH})",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help=f"Where to write the run summary (default: {DEFAULT_SUMMARY_PATH})",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of samples to evaluate")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature for generation")
    parser.add_argument(
        "--target-longest-image-dim",
        type=int,
        default=1024,
        help="Longest image dimension used during PDF rendering",
    )
    parser.add_argument(
        "--prompt-template",
        choices=("yaml", "yaml_v4"),
        default="yaml_v4",
        help="Prompt template to use for inference (default: yaml_v4)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=8,
        help="Maximum concurrent in-flight requests in vLLM mode (default: 8)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.dataset_dir.exists():
        print(f"Dataset directory does not exist: {args.dataset_dir}", file=sys.stderr)
        return 1

    if args.backend == "transformers" and not args.model_path.exists():
        print(f"Model path does not exist: {args.model_path}", file=sys.stderr)
        return 1

    samples = load_samples(args.dataset_dir, args.limit)

    ensure_output_dirs(args)

    if args.backend == "vllm":
        processed, passed = asyncio.run(evaluate_samples_vllm(samples, args))
    else:
        processed, passed = evaluate_samples_transformers(samples, args)

    pass_rate = write_summary(args.summary_path, processed, passed)

    print(f"Processed {processed} samples")
    print(f"Pass rate: {pass_rate:.2f}%")
    print(f"Backend: {args.backend}")
    print(f"Results written to {args.results_path}")
    print(f"Summary written to {args.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
