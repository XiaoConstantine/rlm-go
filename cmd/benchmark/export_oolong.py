#!/usr/bin/env python3
"""Export OOLONG dataset to JSON for Go benchmark."""

import json
import sys
from pathlib import Path


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library not installed")
        print("Install with: pip install datasets")
        sys.exit(1)

    num_tasks = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    output_file = sys.argv[2] if len(sys.argv) > 2 else "oolong_tasks.json"

    print(f"Loading OOLONG dataset ({num_tasks} tasks)...")
    dataset = load_dataset("oolongbench/oolong-synth", split="test")

    tasks = []
    for i, item in enumerate(dataset):
        if i >= num_tasks:
            break

        answer = item.get("answer", "")
        if isinstance(answer, list) and len(answer) > 0:
            answer = str(answer[0])
        elif isinstance(answer, str):
            answer = answer.strip()

        tasks.append({
            "task_id": f"oolong_{i}",
            "context": item.get("context_window_text", ""),
            "question": item.get("question", ""),
            "answer": answer,
        })

    print(f"Loaded {len(tasks)} tasks")

    with open(output_file, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"Exported to: {output_file}")


if __name__ == "__main__":
    main()
