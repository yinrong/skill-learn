"""Convert with_skill teacher data to no_skill format (change system prompt)."""
import json
import argparse
from pathlib import Path

NO_SKILL_SYSTEM = "你是一名 SPC 工程师。"

def convert(input_path: str, output_path: str) -> int:
    count = 0
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as fout:
        for line in open(input_path):
            d = json.loads(line)
            # Check if it has a long system (with_skill)
            sys_val = d.get('system', '')
            if len(sys_val) > 100:  # with_skill has long system
                d['system'] = NO_SKILL_SYSTEM
            fout.write(json.dumps(d, ensure_ascii=False) + '\n')
            count += 1
    return count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    n = convert(args.input, args.output)
    print(f"Converted {n} samples: {args.input} → {args.output}")

if __name__ == '__main__':
    main()
