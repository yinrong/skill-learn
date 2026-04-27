"""向 LLaMA-Factory 的 dataset_info.json 追加数据集条目。

用法：
    python tools/train/register_dataset.py \\
        --name spc_N4 \\
        --file data/demo/train_N4.jsonl
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path


def find_dataset_info_path() -> Path:
    """在常见位置查找 LLaMA-Factory 的 dataset_info.json。"""
    candidates = [
        # LLaMA-Factory 安装后的 data 目录
        Path(os.environ.get("LLAMAFACTORY_DATA_DIR", "")) / "dataset_info.json",
        Path.home() / "LLaMA-Factory" / "data" / "dataset_info.json",
        Path("/opt/LLaMA-Factory/data/dataset_info.json"),
        Path("/usr/local/lib/python3.11/site-packages/llamafactory/data/dataset_info.json"),
        # 通过 pip install llamafactory 安装的路径
    ]
    # 通过 importlib 查找包路径
    try:
        import importlib.util
        spec = importlib.util.find_spec("llamafactory")
        if spec and spec.origin:
            pkg_dir = Path(spec.origin).parent
            candidates.insert(0, pkg_dir / "data" / "dataset_info.json")
    except Exception:
        pass

    for c in candidates:
        if c.exists():
            return c

    # 最后尝试在当前工作目录查找
    local_path = Path("data/dataset_info.json")
    if local_path.exists():
        return local_path

    raise FileNotFoundError(
        "找不到 LLaMA-Factory 的 dataset_info.json。\n"
        "请设置环境变量 LLAMAFACTORY_DATA_DIR 指向 LLaMA-Factory 的 data 目录。"
    )


def register_dataset(
    name: str,
    file_path: str,
    dataset_info_path: Optional[str] = None,
    alpaca: bool = True,
) -> None:
    """将数据集条目追加到 dataset_info.json。"""
    if dataset_info_path:
        info_path = Path(dataset_info_path)
    else:
        info_path = find_dataset_info_path()

    # 读取现有配置
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)

    if name in info:
        print(f"⚠ 数据集 '{name}' 已存在，跳过注册（如需更新请先手动删除该条目）")
        return

    # 转为绝对路径（LLaMA-Factory 需要绝对路径或相对于其 data 目录的路径）
    abs_file = str(Path(file_path).resolve())

    if alpaca:
        entry = {
            "file_name": abs_file,
            "formatting": "alpaca",
            "columns": {
                "system": "system",
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        }
    else:
        entry = {"file_name": abs_file}

    info[name] = entry

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"✓ 已注册数据集 '{name}' → {abs_file}")
    print(f"  写入：{info_path}")


from typing import Optional


def main() -> None:
    parser = argparse.ArgumentParser(description="注册数据集到 LLaMA-Factory")
    parser.add_argument("--name", required=True, help="数据集名称，如 spc_N4")
    parser.add_argument("--file", required=True, help="JSONL 文件路径")
    parser.add_argument("--dataset_info", default=None,
                        help="dataset_info.json 路径（不填则自动查找）")
    parser.add_argument("--no_alpaca", action="store_true",
                        help="不使用 Alpaca 格式（使用原始 JSON 列名）")
    args = parser.parse_args()

    register_dataset(
        name=args.name,
        file_path=args.file,
        dataset_info_path=args.dataset_info,
        alpaca=not args.no_alpaca,
    )


if __name__ == "__main__":
    main()
