"""Hold a cooperative worker lock and reap only this launcher's child group."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--command-json", required=True)
    args = parser.parse_args()
    child = None
    lock_path = Path(args.lock).expanduser()
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    class Cancelled(Exception):
        pass

    def cancel(*_):
        raise Cancelled()

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, cancel)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("COMPASS_WORKER_BUSY", flush=True)
            return 75
        try:
            # Some pre-existing jobs do not participate in the cooperative
            # lock. Device ownership is the authoritative additional check.
            devices = sorted(Path("/dev").glob("accel[0-9]*"))
            if devices and subprocess.run(["fuser", "-s", *map(str, devices)], check=False).returncode == 0:
                print("COMPASS_WORKER_BUSY", flush=True)
                return 75
            print("COMPASS_WORKER_READY", flush=True)
            if sys.stdin.buffer.readline() != b"START\n":
                return 130
            child = subprocess.Popen(json.loads(args.command_json), start_new_session=True)

            def disconnected():
                while os.read(sys.stdin.fileno(), 1):
                    pass
                os.kill(os.getpid(), signal.SIGTERM)

            threading.Thread(target=disconnected, daemon=True).start()
            return child.wait()
        except Cancelled:
            return 130
        finally:
            for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                signal.signal(signum, signal.SIG_IGN)
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


if __name__ == "__main__":
    sys.exit(main())
