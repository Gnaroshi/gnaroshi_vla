"""CPU-only capacity diagnostic: teacher-assisted camera-local token selection."""
import torch

from tools.simvla import visual_transport_train as study
from tools.simvla.compile_benchmark import write_json


def camera_local_distances(reference, target):
    """Relative K/V MSE for each current-token/reference-token pair.

    Reference and target are each a pair of [2,36,64] tensors. The two
    cameras remain independent. Teachers select tokens only in this diagnostic.
    """
    distances = []
    for old, current in zip(reference, target):
        squared = (current[:, :, None] - old[:, None]).square().mean(-1)
        distances.append(squared / current.square().mean(-1).clamp_min(1e-6)[..., None])
    return sum(distances) / len(distances)


def main():
    from tools.simvla.visual_midpoint_features import datasets, CachedFeatures
    from architectures.simvla.adapters.latentloop.native_v0_checkpoint import load_native_v0_checkpoint
    from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import _balanced_indices
    from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import collate_exact_teacher_sequences
    torch.set_num_threads(1)
    c = study.checked()
    native, _ = load_native_v0_checkpoint(c['model']['config']['condition_checkpoint'], device='cpu')
    model = study.initial_model(c, 'transport', native).eval()
    heldout = datasets(c['model']['config'])[1]
    features = CachedFeatures(study.previous.parent.OUTPUT / 'features', study.previous.parent.checked()['identity'])
    records = []
    with torch.no_grad():
        for index in _balanced_indices(heldout.identities, limit=30, seed=c['seed']):
            seq = collate_exact_teacher_sequences([heldout[index]]); features.bind(seq)
            reference = tuple(x.reshape(2,36,64) for x in model.visual.target_readout(features(4,seq['image_sequence'][:,4])))
            for age in (5,6,7):
                target = tuple(x.reshape(2,36,64) for x in model.visual.target_readout(features(age,seq['image_sequence'][:,age])))
                distances = camera_local_distances(reference, target)
                held = distances.diagonal(dim1=1,dim2=2)
                best, selected = distances.min(-1)
                if not (best <= held + 1e-6).all(): raise RuntimeError('Invalid token-selection diagnostic')
                records.append(dict(window=index, age=age, query_ids=seq['query_ids'],
                    held_relative_mse=float(held.mean()), teacher_selected_relative_mse=float(best.mean()),
                    different_position_fraction=float((selected != torch.arange(36)[None]).float().mean())))
    summary = {str(age): {key:sum(r[key] for r in records if r['age']==age)/30
        for key in ('held_relative_mse','teacher_selected_relative_mse','different_position_fraction')}
        for age in (5,6,7)}
    result = dict(verdict='CAPACITY_DIAGNOSTIC_COMPLETE', identity=c['identity'],
        warm_reader_sha256=c['warm_reader_sha256'], windows=30, records=records, by_age=summary,
        metric='Mean of key and value per-token relative MSE, before position embeddings; same as training objective.',
        interpretation='Teacher-assisted token selection shows a feasible representational improvement. Teacher inputs are unavailable online; this is neither predictor accuracy nor task success. Reference age4; current ages5..7; camera-local selection allows repeated source tokens.')
    write_json(study.OUTPUT / 'analysis' / 'token_correspondence_capacity.json', result)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
