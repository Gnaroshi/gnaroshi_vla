"""CPU-only new norm/angle decomposition on the prior 200-window comparison."""
import argparse
from pathlib import Path
import time

import torch
from tqdm import tqdm

from tools.simvla.condition_output_split_pipeline import configuration
from tools.simvla.compile_benchmark import write_json,sha
from methods.latentloop.modules.condition_output_split import geometry
from methods.latentloop.modules.native_simvla_v0 import NativeV0ObservationPair
from architectures.simvla.adapters.latentloop.native_v0_checkpoint import load_native_v0_checkpoint
from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import make_datasets,load_sequence,_balanced_indices


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    torch.set_num_threads(2)
    c=configuration(); c['training_k_c']=4
    if sha(c['condition_checkpoint'])!='d19057768a99f8130bbce279d694a1a9aa9896a7953ad613aadc88d1e8b194db':
        raise RuntimeError('Prior checkpoint differs')
    model,payload=load_native_v0_checkpoint(c['condition_checkpoint'],device='cpu',require_final_150k=True)
    model.eval().requires_grad_(False)
    _,heldout=make_datasets(c,payload)
    indices=_balanced_indices(heldout.identities,limit=200,seed=314159)
    records=[]; began=time.monotonic()
    for index in tqdm(indices,desc='CPU norm/angle decomposition',mininterval=2):
        s=load_sequence(heldout,index,torch.device('cpu'))
        previous=s['anchor_condition']; target=s['teacher_conditions'][:,0]
        valid=s['valid_mask'].bool()
        pair=NativeV0ObservationPair(s['image_sequence'][:,0],s['image_sequence'][:,1],
            s['proprio_sequence'][:,0],s['proprio_sequence'][:,1])
        predicted=model.update_once(previous,pair,valid_mask=valid,group_ids=s['group_ids'],age=1).condition
        records.append(dict(index=index,source=heldout.identities[index],
            hold=geometry(previous,target,valid),ours=geometry(predicted,target,valid)))
    means={arm:{key:sum(r[arm][key] for r in records)/len(records) for key in records[0][arm]}
        for arm in ('hold','ours')}
    for arm,expected in [('hold',.241036),('ours',.317784)]:
        if abs(means[arm]['raw_mse']-expected)>5e-5:
            raise RuntimeError(f'CPU reconstruction differs from reported rows: {arm} {means[arm]}')
    write_json(a.output,dict(verdict='PRIOR_COMPARISON_GEOMETRY_RECONSTRUCTED',windows=200,device='cpu',
        seconds=time.monotonic()-began,means=means,records=records,
        checkpoint_sha256=sha(c['condition_checkpoint']),cache_manifest_sha256=sha(Path(c['cache'])/'manifest.json'),
        note='New norm/angle measurements on the same old windows; no action head, environment evaluation or training. CPU raw MSE checked against original GPU aggregate.'))
    print(means,flush=True)


if __name__=='__main__': main()
