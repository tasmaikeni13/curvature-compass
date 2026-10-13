"""Controlled LM execution: bounded smoke checks or a gated full comparison."""
import argparse
import json
import math
from pathlib import Path
import signal
import time
import traceback
import numpy as np
from . import checkpoint
from .artifacts import atomic_json, environment, sha256, source_hashes
from .calibrate import load_prior
from .config import load_config
from .data import TokenBlocks, verify_data
from .gate import validate_gate
from .model import Decoder
from .optim import TomographyState, clip_gradient, empty_state, learning_rate, make_gradient, make_refresh, make_update
from .runtime import any_host_stop, global_batch, host_array, host_stats, initialize, mesh_and_shardings, sync


def memory_analysis(executable):
    record = executable.memory_analysis()
    if record is None:
        return None
    names = ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes",
             "host_argument_size_in_bytes", "host_output_size_in_bytes", "host_temp_size_in_bytes")
    result = {name: int(getattr(record, name, 0)) for name in names}
    result["estimated_total_bytes"] = result["argument_size_in_bytes"] + result["output_size_in_bytes"] + result["temp_size_in_bytes"] - result["alias_size_in_bytes"]
    result["method"] = "XLA executable memory analysis; not the peak of the entire training process"
    return result


def compile_timed(function, *args):
    started = time.perf_counter()
    compiled = function.lower(*args).compile()
    return compiled, {"compile_seconds": time.perf_counter() - started, "memory": memory_analysis(compiled)}


def execute(config, data, prior_path, output, smoke_steps=None, gate=None, resume=False, smoke_validation_tokens=8192):
    import jax
    import jax.numpy as jnp
    if smoke_steps is None and gate is None:
        raise ValueError("Full training requires the infrastructure gate; select bounded smoke mode otherwise")
    if smoke_steps is not None and not 1 <= smoke_steps <= 8:
        raise ValueError("Infrastructure smoke checks are limited to 1–8 steps")
    startup_started = time.perf_counter()
    decoder = Decoder(config.model)
    data = Path(data)
    manifest = verify_data(data, full=True)
    if manifest["vocab_size"] != config.model.vocab_size:
        raise ValueError("Model and tokenizer vocabulary differ")
    data_hash = sha256(data / "manifest.json")
    frame, prior_record = load_prior(prior_path, decoder, config.optimizer.cap, data_hash)
    if prior_record["seed"] == config.seed:
        raise ValueError("Fixed prior must use an independent model initialization")
    prior_hash = sha256(prior_path)
    env, sources = environment(), source_hashes()
    if smoke_steps is None:
        if not manifest["complete_source_scan"] or config.token_budget < 900_000_000:
            raise ValueError("Full comparison requires the approximately 1B-token prepared corpus")
        validate_gate(gate, config, data_hash, prior_hash, env)
    mesh, replicated, batch_sharding = mesh_and_shardings()
    if config.global_batch_size % (config.accumulation_steps * jax.device_count()):
        raise ValueError("Each gradient microbatch must divide the global device mesh")
    if config.curvature_batch_size % jax.device_count():
        raise ValueError("Curvature batch must divide the global device mesh")
    train = TokenBlocks(data / "train.bin", config.model.sequence_length, config.token_budget, config.data_seed)
    validation_budget = min(config.validation_tokens, smoke_validation_tokens) if smoke_steps is not None else config.validation_tokens
    validation = TokenBlocks(data / "validation.bin", config.model.sequence_length, validation_budget)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    host_index = jax.process_index()
    log_path = output / f"metrics-worker-{host_index}.jsonl"
    run_path = output / f"run-worker-{host_index}.json"
    checkpoint_dir = output / "checkpoints"
    if run_path.exists() and not resume:
        raise ValueError("Existing run output: use --resume or a new output directory")
    step, cursor, tokens, total_hvps, previous_seconds = 0, 0, 0, 0, 0.
    recovery, previous_evaluations, previous_compile_seconds = None, [], 0.
    if resume:
        arrays, metadata, recovery = checkpoint.load(checkpoint_dir, config.fingerprint, data_hash, prior_hash, sources)
        if metadata["mode"] != ("smoke" if smoke_steps is not None else "comparison"):
            raise ValueError("A smoke checkpoint cannot be promoted to a full comparison run")
        for key in ("jax", "jaxlib", "backend", "device_kind", "process_count", "local_device_count"):
            if metadata["environment"][key] != env[key]:
                raise ValueError(f"Checkpoint numerical runtime changed: {key}")
        if arrays["parameters"].shape != (decoder.parameter_count,) or arrays["basis"].shape != (config.optimizer.cap, decoder.parameter_count):
            raise ValueError("Checkpoint tensor shape mismatch")
        parameters = jax.device_put(arrays["parameters"], replicated)
        state = jax.device_put(TomographyState(arrays["basis"], arrays["inverse"], arrays["count"], arrays["refresh_step"]), replicated)
        step, cursor, tokens = metadata["step"], metadata["cursor"], metadata["tokens"]
        total_hvps, previous_seconds = metadata["total_hvps"], metadata["elapsed_seconds"]
        previous_evaluations = metadata["evaluations"]
        previous_compile_seconds = metadata.get("compile_seconds", 0.)
        if cursor != step * config.global_batch_size or tokens > config.token_budget:
            raise ValueError("Checkpoint data cursor/token invariant violated")
        if log_path.exists():
            rows = [json.loads(line) for line in log_path.read_text().splitlines()]
            with log_path.open("w") as stream:
                for row in rows:
                    if row["step"] <= step:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
    else:
        parameters = jax.device_put(decoder.initialize(config.seed), replicated)
        state = jax.device_put(empty_state(decoder.parameter_count, config.optimizer.cap), replicated)
    prior = jax.device_put(np.asarray(frame) if not config.optimizer.adaptive_alignment else np.zeros((1, 1), np.float32), replicated)
    del frame
    record = {"schema": 1, "mode": "smoke" if smoke_steps is not None else "comparison", "config": config.to_dict(),
              "config_hash": config.fingerprint, "data_hash": data_hash, "prior_hash": prior_hash,
              "source_sha256": sources, "environment": env, "parameter_count": decoder.parameter_count,
              "recovery": recovery, "smoke_step_limit": smoke_steps, "validation_tokens": validation_budget,
              "rng": "No training randomness: dropout=0; initialization seed and deterministic block permutation fully specify RNG state."}
    atomic_json(run_path, record)
    requested = [False]
    old_handlers = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        old_handlers[signum] = signal.signal(signum, lambda *_: requested.__setitem__(0, True))
    total_steps = math.ceil(len(train.order) / config.global_batch_size)
    stop_step = total_steps if smoke_steps is None else min(total_steps, smoke_steps)
    gradient_function = jax.jit(make_gradient(decoder, config), in_shardings=(replicated, (batch_sharding,) * 3), out_shardings=replicated)
    refresh_function = jax.jit(make_refresh(decoder, config), in_shardings=(replicated, replicated, (batch_sharding,) * 3, replicated, replicated), out_shardings=replicated)
    update_function = jax.jit(make_update(decoder, config), in_shardings=(replicated, replicated, replicated, replicated), out_shardings=replicated, donate_argnums=(0,))
    clip_function = jax.jit(lambda g: clip_gradient(g, config.gradient_clip), in_shardings=replicated, out_shardings=replicated)
    eval_function = jax.jit(lambda p, b: decoder.loss_sum(p, b), in_shardings=(replicated, (batch_sharding,) * 3), out_shardings=replicated)
    # Compilations are separate from steady-state timings and do not update the model.
    first_batch = global_batch(train, cursor, config.global_batch_size, batch_sharding)
    first_curvature = global_batch(train, cursor, config.curvature_batch_size, batch_sharding, config.curvature_sequence_length)
    example_gradient = jax.device_put(np.zeros(decoder.parameter_count, np.float32), replicated)
    compilation = {}
    gradient_function, compilation["gradient"] = compile_timed(gradient_function, parameters, first_batch)
    refresh_function, compilation["refresh"] = compile_timed(refresh_function, parameters, example_gradient, first_curvature, prior, jax.device_put(np.array(step, np.int32), replicated))
    update_function, compilation["update"] = compile_timed(update_function, parameters, example_gradient, state, jax.device_put(np.array(config.learning_rate, np.float32), replicated))
    first_eval = global_batch(validation, 0, config.global_batch_size, batch_sharding)
    eval_function, compilation["validation"] = compile_timed(eval_function, parameters, first_eval)
    record["compilation"] = compilation
    record["startup_seconds"] = time.perf_counter() - startup_started
    atomic_json(run_path, record)
    started, evaluations, last_saved = time.perf_counter(), previous_evaluations, -1

    def evaluate():
        t = time.perf_counter()
        loss_sum, predicted = 0., 0
        for position in range(0, len(validation.order), config.global_batch_size):
            batch = global_batch(validation, position, config.global_batch_size, batch_sharding)
            value, count = eval_function(parameters, batch)
            jax.block_until_ready((value, count))
            loss_sum += float(host_array(value))
            predicted += int(host_array(count))
        if predicted != validation_budget:
            raise ValueError("Validation token budget was not evaluated exactly")
        mean = loss_sum / predicted
        result = {"step": step, "tokens": tokens, "validation_loss": mean, "perplexity": math.exp(mean),
                  "validation_tokens": predicted, "seconds": time.perf_counter() - t}
        if not np.isfinite(mean):
            raise FloatingPointError("Nonfinite validation loss")
        evaluations.append(result)
        return result

    def save_checkpoint():
        nonlocal last_saved
        sync(f"checkpoint-before-{step}")
        t = time.perf_counter()
        metadata = {"step": step, "cursor": cursor, "tokens": tokens, "total_hvps": total_hvps,
                    "elapsed_seconds": previous_seconds + time.perf_counter() - started,
                    "config_hash": config.fingerprint, "data_hash": data_hash, "prior_hash": prior_hash,
                    "source_sha256": sources, "environment": env, "config": config.to_dict(),
                    "mode": record["mode"], "evaluations": evaluations}
        metadata["compile_seconds"] = previous_compile_seconds + sum(v["compile_seconds"] for v in compilation.values())
        checkpoint.save(checkpoint_dir, parameters, state, metadata, config.keep_checkpoints)
        sync(f"checkpoint-after-{step}")
        last_saved = step
        return time.perf_counter() - t

    status = "finished"
    try:
        with log_path.open("a", buffering=1) as stream:
            while step < stop_step:
                if any_host_stop(requested[0]):
                    status = "interrupted"
                    break
                wall_start = time.perf_counter()
                batch = global_batch(train, cursor, config.global_batch_size, batch_sharding)
                data_seconds = time.perf_counter() - wall_start
                t = time.perf_counter()
                loss, gradient = gradient_function(parameters, batch)
                gradient, gradient_norm = clip_function(gradient)
                jax.block_until_ready((loss, gradient, gradient_norm))
                gradient_seconds = time.perf_counter() - t
                if not np.isfinite(float(host_array(loss))) or not np.isfinite(float(host_array(gradient_norm))):
                    raise FloatingPointError("Nonfinite loss/gradient; last checkpoint remains intact")
                refresh = step % config.optimizer.refresh_interval == 0 or int(host_array(state.refresh_step)) < 0
                curvature_seconds, acquisition = 0., {"hvp_count": 0, "finite": True}
                if refresh:
                    curvature = global_batch(train, cursor, config.curvature_batch_size, batch_sharding, config.curvature_sequence_length)
                    t = time.perf_counter()
                    state, raw = refresh_function(parameters, gradient, curvature, prior, jax.device_put(np.array(step, np.int32), replicated))
                    jax.block_until_ready((state, raw))
                    curvature_seconds = time.perf_counter() - t
                    acquisition = host_stats(raw)
                    if not acquisition["finite"]:
                        raise FloatingPointError("Nonfinite curvature acquisition")
                    total_hvps += acquisition["hvp_count"]
                t = time.perf_counter()
                rate = float(learning_rate(step, total_steps, config))
                parameters, raw_update = update_function(parameters, gradient, state, jax.device_put(np.array(rate, np.float32), replicated))
                jax.block_until_ready((parameters, raw_update))
                update_seconds = time.perf_counter() - t
                update_stats = host_stats(raw_update)
                if not update_stats["parameters_finite"] or not update_stats["direction_finite_and_descent"]:
                    raise FloatingPointError("Invalid update; recover from last committed checkpoint")
                # Host count is derived independently from the deterministic permutation.
                predicted = int(sum(train.batch(cursor, config.global_batch_size)[2].sum(axis=1)))
                cursor += config.global_batch_size
                tokens += predicted
                step += 1
                row = {"step": step, "tokens": tokens, "predicted_tokens": predicted,
                       "training_loss": float(host_array(loss)), "learning_rate": rate,
                       "gradient_norm": float(host_array(gradient_norm)), "refresh": refresh,
                       "curvature_age": step - 1 - int(host_array(state.refresh_step)),
                       "curvature_tokens": int(train.batch(cursor - config.global_batch_size, config.curvature_batch_size, prefix_length=config.curvature_sequence_length)[2].sum()) if refresh else 0,
                       "curvature_positions_processed": config.curvature_batch_size * config.curvature_sequence_length if refresh else 0,
                       "total_hvps": total_hvps, "data_seconds": data_seconds,
                       "gradient_seconds": gradient_seconds, "curvature_seconds": curvature_seconds,
                       "update_seconds": update_seconds, "optimizer_seconds": curvature_seconds + update_seconds,
                       "compute_step_seconds": time.perf_counter() - wall_start,
                       "elapsed_seconds": previous_seconds + time.perf_counter() - started,
                       **acquisition, **update_stats}
                row["training_tokens_per_second"] = predicted / max(row["compute_step_seconds"], 1e-30)
                if step % config.eval_interval == 0 or step == stop_step:
                    row["evaluation"] = evaluate()
                if step % config.checkpoint_interval == 0:
                    row["checkpoint_seconds"] = save_checkpoint()
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                if host_index == 0:
                    print(json.dumps({k: row[k] for k in ("step", "tokens", "training_loss", "hvp_count", "compute_step_seconds")}), flush=True)
            if last_saved != step:
                final_checkpoint_seconds = save_checkpoint()
            else:
                final_checkpoint_seconds = 0.
        complete_budget = tokens == config.token_budget
        if step == total_steps and not complete_budget:
            raise ValueError("Completed block stream did not match exact training token budget")
        elapsed = previous_seconds + time.perf_counter() - started
        summary = {"schema": 1, "status": status, "mode": record["mode"], "configuration": config.optimizer.configuration,
                   "config_hash": config.fingerprint, "parameter_count": decoder.parameter_count,
                   "step": step, "tokens": tokens, "target_tokens": config.token_budget,
                   "complete_budget": complete_budget, "total_hvps": total_hvps,
                   "elapsed_seconds": elapsed, "overall_tokens_per_second": tokens / max(elapsed, 1e-30),
                   "compile_seconds": previous_compile_seconds + sum(v["compile_seconds"] for v in compilation.values()),
                   "startup_seconds_this_invocation": record["startup_seconds"],
                   "final_checkpoint_seconds": final_checkpoint_seconds, "evaluations": evaluations,
                   "device_memory_stats": [d.memory_stats() for d in jax.local_devices()],
                   "data_manifest_sha256": data_hash, "prior_sha256": prior_hash, "source_sha256": sources,
                   "environment": env, "note": "Smoke outcomes validate infrastructure; they are not full-comparison optimizer results." if smoke_steps is not None else None}
        atomic_json(output / f"summary-worker-{host_index}.json", summary)
        return summary
    except BaseException as error:
        atomic_json(output / f"failure-worker-{host_index}.json", {"step": step, "tokens": tokens,
                    "error": str(error), "traceback": traceback.format_exc(), "mode": record["mode"]})
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--prior", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke-steps", type=int)
    mode.add_argument("--full-run", action="store_true")
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-validation-tokens", type=int, default=8192)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--coordinator")
    parser.add_argument("--processes", type=int)
    parser.add_argument("--process-id", type=int)
    args = parser.parse_args()
    if args.full_run and args.gate is None:
        parser.error("--full-run requires --gate from completed infrastructure validation")
    initialize(args.distributed, args.coordinator, args.processes, args.process_id)
    execute(load_config(args.config), args.data, args.prior, args.output, args.smoke_steps, args.gate, args.resume, args.smoke_validation_tokens)


if __name__ == "__main__":
    main()
