"""LoRA adapter 合并工具：llamafactory-cli export 的封装。

用法：
    python tools/train/merge_adapter.py \\
        --base /data/models/Qwen3-32B \\
        --adapter /data/checkpoints/demo-N1 \\
        --output /data/checkpoints/demo-N1-merged
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def merge_adapter(
    base_model: str,
    adapter_path: str,
    output_path: str,
    template: str = "qwen",
    finetuning_type: str = "lora",
) -> None:
    """调用 llamafactory-cli export 完成 adapter 合并。"""
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    # 写一个临时 YAML export 配置
    export_yaml = f"""
model_name_or_path: {base_model}
adapter_name_or_path: {adapter_path}
template: {template}
finetuning_type: {finetuning_type}
export_dir: {output_path}
export_size: 2
export_device: cpu
export_legacy_format: false
""".strip()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
        tf.write(export_yaml)
        tmp_path = tf.name

    print(f"合并配置：")
    print(f"  base:    {base_model}")
    print(f"  adapter: {adapter_path}")
    print(f"  output:  {output_path}")

    import os
    env = os.environ.copy()
    env["DISABLE_VERSION_CHECK"] = "1"

    cmd = ["llamafactory-cli", "export", tmp_path]
    print(f"执行：{' '.join(cmd)}")

    result = subprocess.run(cmd, check=False, env=env)
    if result.returncode != 0:
        print(f"✗ 合并失败（exit code {result.returncode}），请检查 adapter 路径和 base 模型版本")
        sys.exit(result.returncode)

    # 验证输出文件
    safetensors = list(out.glob("*.safetensors"))
    if not safetensors:
        print("⚠ 输出目录中未找到 safetensors 文件，合并可能未成功")
    else:
        print(f"✓ 合并完成，safetensors 文件数：{len(safetensors)}")
        print(f"  输出目录：{output_path}")

    Path(tmp_path).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="合并 LoRA adapter 到 base 模型")
    parser.add_argument("--base", required=True, help="基座模型路径")
    parser.add_argument("--adapter", required=True, help="LoRA checkpoint 目录")
    parser.add_argument("--output", required=True, help="合并后模型输出目录")
    parser.add_argument("--template", default="qwen", help="LLaMA-Factory 模板名（默认 qwen）")
    args = parser.parse_args()

    merge_adapter(
        base_model=args.base,
        adapter_path=args.adapter,
        output_path=args.output,
        template=args.template,
    )


if __name__ == "__main__":
    main()
