"""启动 vLLM 推理服务，阻塞直到健康检查通过后返回 endpoint URL。

用法：
    python tools/train/deploy_vllm.py --model /data/models/Qwen3-32B --port 8000
    python tools/train/deploy_vllm.py --teardown --port 8000

接口：
    deploy(model_path, port) -> str   # 返回 URL
    teardown(port) -> None
"""
from __future__ import annotations
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional


# 记录已启动的 vLLM 进程（port -> pid）
_PID_FILE = Path("/tmp/vllm_pids.json")


def _load_pids() -> dict[str, int]:
    if _PID_FILE.exists():
        try:
            return json.loads(_PID_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_pids(pids: dict[str, int]) -> None:
    _PID_FILE.write_text(json.dumps(pids))


def _health_check(url: str, timeout: int = 300, interval: float = 5.0) -> bool:
    """轮询健康检查，超时返回 False。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        print(f"  等待 vLLM 启动... ({url}/health)", flush=True)
        time.sleep(interval)
    return False


def deploy(
    model_path: str,
    port: int,
    gpu_memory_utilization: float = 0.90,
    tensor_parallel_size: int = 1,
    max_model_len: int = 7168,
    enable_thinking: bool = True,
    dtype: str = "bfloat16",
    extra_args: Optional[list[str]] = None,
) -> str:
    """启动 vLLM 服务并等待就绪，返回 endpoint URL。

    Args:
        model_path:              模型权重目录
        port:                    监听端口
        gpu_memory_utilization:  显存使用率（0~1）
        tensor_parallel_size:    TP 并行度（多卡推理时调整）
        max_model_len:           最大序列长度
        enable_thinking:         是否开启 <think> 模式（Qwen3 系列）
        dtype:                   权重精度
        extra_args:              其他 vllm serve 参数

    Returns:
        endpoint URL，如 "http://localhost:8000"
    """
    url = f"http://localhost:{port}"

    # 检查端口是否已在使用
    pids = _load_pids()
    if str(port) in pids:
        old_pid = pids[str(port)]
        try:
            os.kill(old_pid, 0)  # 检查进程是否存活
            print(f"⚠ 端口 {port} 已有 vLLM 进程（pid={old_pid}），先停止...")
            teardown(port)
        except ProcessLookupError:
            pass

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--max-model-len", str(max_model_len),
        "--dtype", dtype,
        "--trust-remote-code",
        "--served-model-name", "default",
    ]
    if enable_thinking:
        # Qwen3 需要在推理时开启 thinking/reasoning 模式（vllm 0.19.1）
        cmd += ["--reasoning-parser", "deepseek_r1"]

    if extra_args:
        cmd += extra_args

    log_file = Path(f"/tmp/vllm_{port}.log").open("w")
    print(f"启动 vLLM：port={port}  model={model_path}")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    pids = _load_pids()
    pids[str(port)] = proc.pid
    _save_pids(pids)
    print(f"  pid={proc.pid}  日志：/tmp/vllm_{port}.log")

    # 等待健康检查
    print(f"  等待健康检查（最多300秒）...")
    if not _health_check(url):
        print(f"✗ vLLM 启动超时，请检查 /tmp/vllm_{port}.log")
        sys.exit(1)

    print(f"✓ vLLM 已就绪：{url}")
    return url


def teardown(port: int) -> None:
    """停止指定端口的 vLLM 进程。"""
    pids = _load_pids()
    pid = pids.get(str(port))
    if pid is None:
        print(f"⚠ 端口 {port} 无记录的 vLLM 进程")
        return

    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        print(f"✓ 已发送 SIGTERM 给 vLLM 进程组（pid={pid}，port={port}）")
    except ProcessLookupError:
        print(f"⚠ 进程 {pid} 不存在（可能已退出）")
    except Exception as e:
        print(f"⚠ 停止进程失败：{e}")

    del pids[str(port)]
    _save_pids(pids)


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM 服务管理")
    parser.add_argument("--model", default=None, help="模型路径（启动时必须）")
    parser.add_argument("--port", type=int, required=True, help="端口号")
    parser.add_argument("--teardown", action="store_true", help="停止该端口的服务")
    parser.add_argument("--gpu_util", type=float, default=0.90)
    parser.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    parser.add_argument("--max_len", type=int, default=7168)
    parser.add_argument("--no_thinking", action="store_true", help="禁用 <think> 模式")
    args = parser.parse_args()

    if args.teardown:
        teardown(args.port)
    else:
        if not args.model:
            parser.error("启动服务时必须指定 --model")
        deploy(
            model_path=args.model,
            port=args.port,
            gpu_memory_utilization=args.gpu_util,
            tensor_parallel_size=args.tp,
            max_model_len=args.max_len,
            enable_thinking=not args.no_thinking,
        )


if __name__ == "__main__":
    main()
