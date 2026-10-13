"""A bounded distributed device/topology probe."""
import argparse
from pathlib import Path
import numpy as np
from .artifacts import atomic_json, environment
from .runtime import initialize, mesh_and_shardings, host_array, sync


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--coordinator")
    parser.add_argument("--processes", type=int)
    parser.add_argument("--process-id", type=int)
    args = parser.parse_args()
    jax = initialize(args.distributed, args.coordinator, args.processes, args.process_id)
    mesh, replicated, sharded = mesh_and_shardings()
    value = jax.device_put(np.ones(32, np.float32), replicated)
    result = jax.jit(lambda x: (x @ x), in_shardings=replicated, out_shardings=replicated)(value)
    result.block_until_ready()
    if float(host_array(result)) != 32:
        raise ValueError("TPU calculation failed")
    record = {"environment": environment(), "sum": float(host_array(result)),
              "process_index": jax.process_index(), "device_memory_stats": [d.memory_stats() for d in jax.local_devices()]}
    atomic_json(args.output.with_name(args.output.stem + f"-worker-{jax.process_index()}.json"), record)
    print(record, flush=True)
    sync("probe-complete")


if __name__ == "__main__":
    main()
