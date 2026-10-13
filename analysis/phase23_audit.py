"""Bind prepared inputs, small-model correctness and full-shape timing evidence.

No full-size training run is required or claimed. Historical timing artifacts
retain the source hashes of their measured snapshots; final numerical tests
check the cleaned standard-XLA implementation.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from curvature_compass.lm.artifacts import atomic_json, sha256, source_hashes
from curvature_compass.lm.calibrate import load_prior
from curvature_compass.lm.config import load_config
from curvature_compass.lm.data import verify_data
from curvature_compass.lm.gate import REQUIRED_CHECKS
from curvature_compass.lm.model import Decoder

ROOT = Path('analysis/results/phase23')


def check(condition, reason):
    if not condition:
        raise ValueError(reason)


def audit(data, prior, output):
    evidence, checks = {}, dict.fromkeys(REQUIRED_CHECKS, False)

    def witness(path):
        path = Path(path)
        evidence[str(path)] = sha256(path)
        return json.loads(path.read_text())

    manifest = verify_data(data, full=True)
    data_hash = sha256(Path(data) / 'manifest.json')
    configs = [load_config(f'configs/lm60m_{cell}.json') for cell in 'ABCD']
    decoder = Decoder(configs[0].model)
    check(manifest['complete_source_scan'] and manifest['rows_read'] == 970397, 'Pinned corpus scan is incomplete')
    check(decoder.parameter_count == 60_957_184, 'Architecture differs from the measured workload')
    check(all(c.token_budget == manifest['predicted_tokens']['train'] and .9e9 <= c.token_budget <= 1.1e9 for c in configs), 'Token controls differ')
    frame, prior_record = load_prior(prior, decoder, configs[0].optimizer.cap, data_hash)
    check(prior_record['data_split'] == 'calibration' and prior_record['snapshots'] == 16 and prior_record['seed'] not in [c.seed for c in configs], 'Prior lacks independent calibration')
    del frame
    check(manifest['predicted_tokens']['validation'] >= configs[0].validation_tokens, 'Insufficient held-out validation tokens')
    witness(ROOT / 'data_manifest.json')
    witness(ROOT / 'prior_manifest.json')
    checks['data_integrity'] = True
    common = []
    for c in configs:
        value = c.to_dict()
        value.pop('seed')
        value['optimizer'].pop('configuration')
        common.append(value)
    check(len({c.seed for c in configs}) == 4 and all(v == common[0] for v in common), 'Four configurations have uncontrolled differences')
    checks['four_controls'] = True

    tests = ROOT / 'cpu_tests.xml'
    tree = ET.parse(tests)
    cases = tree.findall('.//testcase')
    check(len(cases) >= 31 and not any(tree.findall('.//' + tag) for tag in ('failure', 'error', 'skipped')), 'CPU suite has missing or failed checks')
    names = [t.attrib['name'] for t in cases]
    for name in ('test_bitwise_resumed_training_and_exact_partial_final_budget', 'test_full_run_and_smoke_bound_guard',
                 'test_full_run_gate_rejects_unvalidated_or_changed_experiments', 'test_atomic_checkpoint_rollback_and_experiment_binding'):
        check(any(n.startswith(name) for n in names), 'Required training/recovery coverage missing: ' + name)
    for cell in 'ABCD':
        check(any(n.startswith('test_quadratic_matches_independent_projected_dense_inverse') and '[' + cell + ']' in n for n in names), 'Missing optimizer oracle: ' + cell)
        check(any(n.startswith('test_all_cells_execute_real_model_hvps_and_cached_updates') and '[' + cell + ']' in n for n in names), 'Missing model integration: ' + cell)
    evidence[str(tests)] = sha256(tests)
    checks['cpu_tests'] = checks['all_four_implementations'] = True

    current = source_hashes()
    core = ['src/curvature_compass/lm/' + n + '.py' for n in ('model', 'optim', 'config', 'runtime', 'numerics')]
    for worker in range(4):
        numeric = witness(ROOT / f'tpu_numerics_final-worker-{worker}.json')
        check(numeric['status'] == 'passed' and len(numeric['checks']) >= 21, 'TPU numerical coverage failed')
        check(all(v.get('passed', v.get('error', 1) <= v.get('limit', 0)) for v in numeric['checks'].values()), 'TPU artifact contradicts its pass label')
        check(all(numeric['source_sha256'].get(name) == current[name] for name in core), 'Final TPU numerical source changed')
        check(numeric['checks']['bitwise_device_checkpoint_continuation']['passed'], 'Device checkpoint continuation differs')
        env = numeric['environment']
        check(env['backend'] == 'tpu' and env['process_count'] == 4 and env['local_device_count'] == 4 and len(env['devices']) == 16, 'Wrong TPU topology')
    checks['tpu_numerics'] = checks['checkpoint_recovery'] = True

    selection = witness(ROOT / 'acceleration_selection.json')
    check(selection['status'] == 'frozen' and selection['custom_kernel_code_removed'], 'Execution selection is incomplete')
    for name in (selection['reference_benchmark'], selection['selected_benchmark']):
        record = witness(ROOT / (name + '-worker-0.json'))
        check(record['status'] == 'passed' and len(record['raw']) >= 3, 'Benchmark lacks repeated observations')
        check(record['parameter_count'] == decoder.parameter_count and record['batch_shape'] == [64, 1024] and record['curvature_shape'] == [16, 256], 'Wrong benchmark shape')
        check(all(r['hvp_count'] == 8 for r in record['raw']), 'Fixed-count benchmark did not execute eight HVPs')
        check(all(np.isfinite(r[k]) and r[k] > 0 for r in record['raw'] for k in ('gradient_seconds', 'curvature_seconds', 'update_seconds')), 'Incomplete timing accounting')
        check(bool(record['source_sha256']), 'Historical benchmark source provenance missing')
    for c in configs:
        check(all(c.to_dict()[k] == v for k, v in selection['selected_options'].items()), 'Final execution options differ from selection')
    checks['full_shape_benchmarks'] = checks['cost_accounting'] = checks['selected_acceleration'] = True
    evidence['analysis/phase23_audit.py'] = sha256('analysis/phase23_audit.py')
    evidence['analysis/phase23_protocol.md'] = sha256('analysis/phase23_protocol.md')
    gate = {'schema': 1, 'status': 'passed', 'checks': checks,
            'scope': 'Preparation only: small-model integration/recovery, final TPU derivatives, historical full-shape stationary timing. No full-size training or optimizer winner claimed.',
            'limitations': 'Benchmark snapshots precede removal of unused prototype branches; their original source hashes are preserved. Cached costs are stage sums, not a measured full training run.',
            'validated_config_hashes': [c.fingerprint for c in configs], 'data_manifest_sha256': data_hash,
            'prior_sha256': sha256(prior), 'source_sha256': current, 'environment': env, 'evidence_sha256': evidence}
    atomic_json(output, gate)
    print(json.dumps({'status': 'passed', 'checks': checks, 'gate': str(output)}, indent=2))
    return gate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/fineweb'))
    parser.add_argument('--prior', type=Path, default=Path('artifacts/lm60m_prior.npy'))
    parser.add_argument('--output', type=Path, default=ROOT / 'gate.json')
    args = parser.parse_args()
    audit(args.data, args.prior, args.output)
