"""Backfill existing confirmation without changing its frozen training config."""
from pathlib import Path

from tools.simvla import geometry_target_confirmation as confirmation
from tools.simvla import attention_query_eval as evaluation
from tools.simvla.compile_benchmark import read_json, sha


def runtime_seed_builder(original):
    def build(mode, config):
        stored = read_json(Path(config['output'])/'contract.json')['config']
        changed = {k for k in set(config)|set(stored) if config.get(k)!=stored.get(k)}
        if changed - {'action_noise_seed_base'}:
            raise RuntimeError('Unexpected reference config change: '+str(changed))
        seed = config['action_noise_seed_base']
        policy, compiler = original(mode, stored)
        policy.action_noise_seed_base = seed
        return policy, compiler
    return build


def main():
    original_builder, original_writer = evaluation.prior.build_policy, evaluation.write_json
    source = sha(__file__)
    def writer(path, value):
        if Path(path).name == 'runtime.json' or Path(path).parent.name == 'episodes':
            value = dict(value, reference_seed_repair_sha256=source)
        original_writer(path,value)
    evaluation.prior.build_policy = runtime_seed_builder(original_builder)
    evaluation.write_json = writer
    try:
        return confirmation.main()
    finally:
        evaluation.prior.build_policy, evaluation.write_json = original_builder, original_writer


if __name__=='__main__':
    raise SystemExit(main())
