#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
# ]
# ///
"""
Prepare Dolly 2.0 dataset for fine-tuning.

Downloads databricks/databricks-dolly-15k from HuggingFace and formats it
as a plain text file with instruction/input/response markers.

Usage:
    uv run scripts/prepare_dolly.py
"""

import os
from datasets import load_dataset

OUTPUT_DIR = "corpus"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dolly.txt")


def format_example(example):
    """Format a single Dolly example with instruction markers."""
    parts = [f"### Instruction:\n{example['instruction']}"]

    context = example.get("context", "").strip()
    if context:
        parts.append(f"\n\n### Input:\n{context}")

    parts.append(f"\n\n### Response:\n{example['response']}")
    parts.append("\n\n<|endoftext|>")

    return "".join(parts)


def main():
    print("Downloading Dolly 2.0 dataset...")
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    print(f"Loaded {len(dataset)} examples")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        for i, example in enumerate(dataset):
            if i > 0:
                f.write("\n\n")
            f.write(format_example(example))

    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"Done! Wrote {len(dataset)} examples to {OUTPUT_FILE} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
