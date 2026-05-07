"""Filter training data to keep only complete (non-truncated) samples.

A sample is considered complete if:
- It has violations in ground_truth → output must contain '异常判断' OR '结论' OR '违规' (conclusion section)
- It has no violations → output must contain 'CPK' (CPK calculation must be present)
"""
import argparse
import json
from pathlib import Path


def is_complete(data: dict) -> bool:
    out = data.get("output", "")
    gt = data.get("ground_truth", {})
    violations = gt.get("violations", []) if isinstance(gt, dict) else []

    if violations:
        return "异常判断" in out or "结论" in out or "违规" in out
    else:
        return "CPK" in out.upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    lines = Path(args.input).read_text().splitlines()
    kept = []
    dropped = 0
    for line in lines:
        if not line.strip():
            continue
        d = json.loads(line)
        if is_complete(d):
            kept.append(line)
        else:
            dropped += 1

    Path(args.output).write_text("\n".join(kept) + "\n")
    print(f"  {Path(args.input).name}: {len(lines)} → kept {len(kept)}, dropped {dropped} ({100*dropped//len(lines)}%)")


if __name__ == "__main__":
    main()
