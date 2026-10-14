import copy

import pytest

from tools.simvla import visual_readout_remote as remote


def contracts():
    source = {k: k for k in remote.FIELDS}
    source.update(identity='sd1_identity', model=dict(sha256='parent', config={
        'checkpoint_revision': 'base', 'norm_stats': '/sd1/norm', 'condition_checkpoint': '/sd1/model'}))
    destination = copy.deepcopy(source)
    destination['identity'] = 'rb2_identity'
    destination['model']['config'].update(norm_stats='/rb2/norm', condition_checkpoint='/rb2/model')
    marker = dict(training_identity='sd1_identity', artifact_sha256=dict(norm_stats='norm', condition_checkpoint='model'))
    return source, destination, marker


def test_host_paths_and_identity_can_differ_when_artifacts_match(monkeypatch):
    monkeypatch.setattr(remote, 'sha', lambda path: path.rsplit('/', 1)[-1])
    source, destination, marker = contracts()
    assert remote.validate_origin(source, destination, marker) is source


@pytest.mark.parametrize('field', remote.FIELDS)
def test_scientific_change_is_rejected(monkeypatch, field):
    monkeypatch.setattr(remote, 'sha', lambda path: path.rsplit('/', 1)[-1])
    source, destination, marker = contracts()
    destination[field] = 'different'
    with pytest.raises(RuntimeError, match='scientific contract'):
        remote.validate_origin(source, destination, marker)


@pytest.mark.parametrize('field', ('norm_stats', 'condition_checkpoint'))
def test_artifact_hash_mismatch_is_rejected(monkeypatch, field):
    monkeypatch.setattr(remote, 'sha', lambda path: path.rsplit('/', 1)[-1])
    source, destination, marker = contracts()
    marker['artifact_sha256'][field] = 'wrong'
    with pytest.raises(RuntimeError, match='Base artifact differs'):
        remote.validate_origin(source, destination, marker)
