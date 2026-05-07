#!/usr/bin/env python3
"""
并发预下载 benchmark 所需的 HuggingFace 数据集。

策略：
- 多仓库并行：ThreadPoolExecutor，每个仓库一个线程
- 仓库内多文件并行：snapshot_download(max_workers=16)
- 断点续传：已缓存文件自动跳过
- HF_TRANSFER 加速（若已安装 hf-transfer）

用法：
    python prefetch_datasets.py [--endpoint https://hf-mirror.com] [--jobs 4]
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("pip install huggingface_hub")
    sys.exit(1)

DATASETS = [
    "cais/mmlu",
    "openai/gsm8k",
    "allenai/ai2_arc",
    "Rowan/hellaswag",
    "allenai/winogrande",
    "truthfulqa/truthful_qa",
    "haonan-li/cmmlu",
]


def download_one(repo_id: str, endpoint: str, file_workers: int) -> tuple[str, bool, str]:
    """下载单个数据集仓库，返回 (repo_id, ok, message)。"""
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            endpoint=endpoint,
            max_workers=file_workers,   # 仓库内并发文件数
            local_files_only=False,
        )
        return repo_id, True, "ok"
    except Exception as e:
        return repo_id, False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--jobs", type=int, default=4,
                        help="同时下载的仓库数（默认4）")
    parser.add_argument("--file-workers", type=int, default=16,
                        help="每个仓库内的并发文件下载线程数（默认16）")
    args = parser.parse_args()

    os.environ["HF_ENDPOINT"] = args.endpoint
    # 若安装了 hf-transfer，自动启用（Rust 多段下载，比 requests 快 3~5x）
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") != "0":
        try:
            import hf_transfer  # noqa: F401
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            print("[info] hf_transfer 已启用")
        except ImportError:
            pass

    print(f"下载 {len(DATASETS)} 个数据集，仓库并发={args.jobs}，文件并发={args.file_workers}")
    print(f"端点：{args.endpoint}\n")

    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(download_one, repo, args.endpoint, args.file_workers): repo
            for repo in DATASETS
        }
        for fut in as_completed(futures):
            repo_id, ok, msg = fut.result()
            status = "✓" if ok else "✗"
            print(f"  [{status}] {repo_id}: {msg}")
            results.append((repo_id, ok))

    failed = [r for r, ok in results if not ok]
    if failed:
        print(f"\n{len(failed)} 个仓库下载失败，串行重试（绕过 tqdm 并发锁问题）...")
        for repo_id in failed:
            _, ok, msg = download_one(repo_id, args.endpoint, file_workers=1)
            status = "✓" if ok else "✗"
            print(f"  [{status}] {repo_id} (retry): {msg}")

    still_failed = [r for r, ok in results if not ok]
    if still_failed:
        print(f"\n以下仓库最终失败（cmmlu 在部分镜像不可用属正常）：{still_failed}")
        sys.exit(1)
    else:
        print("\n所有数据集下载完成")


if __name__ == "__main__":
    main()
