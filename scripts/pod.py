"""Launch all pod workers together, using existing SSH aliases and runtime."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import queue
import shlex
import signal
import subprocess
import sys
import threading
import time

SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]


def run(hosts, root, python, module, arguments, coordinator, lock, logs, timeout):
    logs.mkdir(parents=True, exist_ok=True)
    events, processes, handles = queue.Queue(), [], []
    started = time.monotonic()
    previous_handlers = {}
    def cancelled(*_):
        raise KeyboardInterrupt("Pod launcher cancelled")
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.signal(signum, cancelled)

    def read(index, process, stream):
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            events.put((index, line.rstrip()))

    try:
        for index, host in enumerate(hosts):
            command = [str(python), "-u", "-m", module, *arguments, "--distributed", "--coordinator", coordinator,
                       "--processes", str(len(hosts)), "--process-id", str(index)]
            worker = [str(python), str(root / "scripts/pod_worker.py"), "--lock", lock,
                      "--command-json", json.dumps(command)]
            remote = "cd " + shlex.quote(str(root)) + " && env JAX_PLATFORMS=tpu OPENBLAS_NUM_THREADS=1 " + shlex.join(worker)
            process = subprocess.Popen([*SSH, host, remote], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            processes.append(process)
            handle = (logs / f"worker-{index}.log").open("w")
            handles.append(handle)
            threading.Thread(target=read, args=(index, process, handle), daemon=True).start()
        ready = set()
        while len(ready) < len(hosts):
            if time.monotonic() - started > 45 or any(p.poll() is not None for p in processes):
                raise RuntimeError("Pod worker did not become ready; no training process was released")
            try:
                index, line = events.get(timeout=1)
                print(f"[{hosts[index]}] {line}", flush=True)
                if line == "COMPASS_WORKER_BUSY":
                    raise RuntimeError(f"{hosts[index]} is busy; leave the existing workload untouched")
                if line == "COMPASS_WORKER_READY":
                    ready.add(index)
            except queue.Empty:
                continue
        for process in processes:
            process.stdin.write("START\n")
            process.stdin.flush()
        last_heartbeat = time.monotonic()
        while any(p.poll() is None for p in processes):
            if time.monotonic() - started > timeout:
                raise TimeoutError("Bounded pod validation deadline exceeded")
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError("A pod worker failed; cancelling only this launcher's workers")
            try:
                index, line = events.get(timeout=1)
                print(f"[{hosts[index]}] {line}", flush=True)
            except queue.Empty:
                pass
            if time.monotonic() - last_heartbeat >= 30:
                print(f"Pod job remains live ({time.monotonic() - started:.0f}s)", flush=True)
                last_heartbeat = time.monotonic()
        while not events.empty():
            index, line = events.get_nowait()
            print(f"[{hosts[index]}] {line}", flush=True)
        codes = [p.wait() for p in processes]
        if any(codes):
            raise RuntimeError(f"Pod exit codes: {codes}")
        return codes
    finally:
        for process in processes:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=10)
        for handle in handles:
            handle.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", nargs="+", default=["w0", "w1", "w2", "w3"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--python", type=Path)
    parser.add_argument("--coordinator")
    parser.add_argument("--port", type=int, default=12392)
    parser.add_argument("--lock", default="/home/tasma/.cache/rocora/pod-worker.lock",
                        help="Use the existing cooperative lock shared with the other workload on this pod")
    parser.add_argument("--logs", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("module")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = args.root.resolve()
    python = args.python or root / ".venv/bin/python"
    coordinator = args.coordinator
    if coordinator is None:
        config = subprocess.check_output(["ssh", "-G", args.hosts[0]], text=True, stderr=subprocess.DEVNULL)
        host = next(line.split(maxsplit=1)[1] for line in config.splitlines() if line.startswith("hostname "))
        coordinator = f"{host}:{args.port}"
    run(args.hosts, root, python, args.module, args.arguments, coordinator, args.lock, args.logs, args.timeout)


if __name__ == "__main__":
    main()
