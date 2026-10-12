"""Exact frozen vision/connector features, cached only for updater training."""
import hashlib
from pathlib import Path
import time

import torch

from tools.simvla.compile_benchmark import read_json, write_json, sha, stats


class VisionOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision = model.vlm.model.vision_model
        self.connector = model.vlm.model.connector
        self.requires_grad_(False).eval()

    def forward(self, image_input):
        # The registered LIBERO contract has two valid camera views.
        batch, views = image_input.shape[:2]
        features = self.vision(pixel_values=image_input.flatten(0, 1),
            output_hidden_states=False, return_dict=True).last_hidden_state
        return self.connector(features).reshape(batch, -1, 960)


def feature_path(root, query_id):
    key = hashlib.sha256(query_id.encode()).hexdigest()
    return Path(root)/key[:2]/(key+'.pt')


def datasets(cfg):
    from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import ExactTeacherSequenceDataset
    payload=torch.load(cfg['condition_checkpoint'],map_location='cpu',weights_only=False)
    training=payload['training_config']
    return [ExactTeacherSequenceDataset(cfg['cache'], split=split, window_queries=8,
                heldout_fraction=training['heldout_fraction'],split_seed=training['split_seed'])
            for split in ('train', 'heldout')]


def required_queries(data):
    return sorted({q for dataset in data for window in dataset.windows for q in window[4:]})


def extract(study, shard):
    from tools.simvla.error_compensation_common import configure, snapshots
    from architectures.simvla.adapters.latentloop.native_v0_runtime import load_frozen_simvla, configure_strict_torch_determinism
    from architectures.simvla.adapters.latentloop.native_v0_prepare import _official_training_image_inputs
    from architectures.simvla.adapters.latentloop.native_v0_checkpoint import atomic_torch_save
    from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import _load_rgb_ref
    c=study.checked();cfg=c['model']['config'];configure(cfg)
    torch.set_num_threads(1);configure_strict_torch_determinism(7)
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((total-2*1024**3)/total)
    provenance=snapshots(cfg)
    model,processor,_=load_frozen_simvla(checkpoint=provenance['checkpoint_snapshot'],
        norm_stats=cfg['norm_stats'],smolvlm_model=provenance['backbone_snapshot'],device='cuda')
    vision=VisionOnly(model);data=datasets(cfg);ids=required_queries(data)[shard::4]
    out=study.OUTPUT/'features';records=[];began=time.monotonic()
    with torch.inference_mode():
        for n,query in enumerate(ids):
            path=feature_path(out,query)
            if path.exists():
                saved=torch.load(path,map_location='cpu',weights_only=False)
                if saved['identity']!=c['identity'] or saved['query_id']!=query:
                    raise RuntimeError('Feature cache identity mismatch')
                records.append(dict(query=query,path=str(path),sha256=sha(path)));continue
            record=data[0].store.query(query)
            raw=_load_rgb_ref(record['metadata']['raw_rgb_ref'])
            batch=_official_training_image_inputs(raw,image_size=processor.image_size,num_views=processor.num_views)
            if batch['image_mask'].tolist()!=[[True,True,False]]:
                raise RuntimeError('Expected two valid cameras and one padded camera')
            inputs=batch['image_input'][:,:2].cuda()
            features=vision(inputs).cpu()
            if n==0:
                captured=[]
                handle=model.vlm.model.connector.register_forward_hook(lambda m,a,o:captured.append(o.detach()))
                try:
                    text=processor.encode_language([record['metadata']['language_instruction']])
                    model.forward_vlm_efficient(batch['image_input'].cuda(),batch['image_mask'].cuda(),text['input_ids'].cuda())
                    reference=captured[0].reshape(1,-1,960).cpu()
                    torch.testing.assert_close(features,reference,atol=0,rtol=0)
                finally:handle.remove()
            if features.shape!=(1,72,960) or not torch.isfinite(features).all():
                raise RuntimeError(f'Unexpected vision features {features.shape}')
            path.parent.mkdir(parents=True,exist_ok=True)
            atomic_torch_save(dict(identity=c['identity'],query_id=query,features=features,
                input_sha256=hashlib.sha256(inputs.cpu().numpy().tobytes()).hexdigest()),path)
            records.append(dict(query=query,path=str(path),sha256=sha(path)))
            if n%100==0:
                write_json(out/f'progress_{shard}.json',dict(completed=n+1,total=len(ids),seconds=time.monotonic()-began))
                print(f'VISION_CACHE shard={shard} {n+1}/{len(ids)}',flush=True)
        write_json(out/f'shard_{shard}.json',dict(verdict='FEATURES_COMPLETE',identity=c['identity'],
            shard=shard,queries=len(ids),seconds=time.monotonic()-began,records=records,
            dtype='float32',shape=[1,72,960],source='Original frozen vision encoder and connector; no text transformer'))


class CachedFeatures:
    def __init__(self, root, identity):
        self.root,self.identity=Path(root),identity
        self.sequence=None
    def bind(self, sequence):
        self.sequence=sequence
    def __call__(self, age, images):
        if self.sequence is None:raise RuntimeError('Sequence feature binding missing')
        values=[]
        for ids in self.sequence['query_ids']:
            query=ids[age]
            saved=torch.load(feature_path(self.root,query),map_location='cpu',weights_only=False)
            if saved['identity']!=self.identity or saved['query_id']!=query:
                raise RuntimeError('Feature/query mismatch')
            values.append(saved['features'])
        return torch.cat(values).to(images.device)


def profile(study):
    from tools.simvla.refresh_allocation import setup
    from tools.simvla.compiled_campaign import move
    from tools.simvla.compile_runtime import Compiler
    from architectures.simvla.adapters.latentloop.native_v0_runtime import load_frozen_simvla,configure_strict_torch_determinism
    c=study.checked();cfg=c['model']['config'];setup(cfg)
    torch.set_num_threads(1);configure_strict_torch_determinism(20260815)
    sample=move(torch.load(Path(cfg['recorded_input'])/'recorded_inputs.pt',map_location='cpu',weights_only=False))[0]['queries'][0]
    checkpoint=str(Path(cfg['hf_home'])/'hub/models--YuankaiLuo--SimVLA-LIBERO/snapshots'/cfg['checkpoint_revision'])
    model,_,action=load_frozen_simvla(checkpoint=checkpoint,norm_stats=cfg['norm_stats'],
        smolvlm_model='HuggingFaceTB/SmolVLM-500M-Instruct',device='cuda')
    module=VisionOnly(model);compiler=Compiler(True)
    report={}
    with torch.inference_mode():
        expected=module(sample['image_input'][:,:2])
        module.forward=compiler.wrap('vision_only',module.forward)
        original=model.forward_vlm_efficient
        model.forward_vlm_efficient=compiler.wrap('full_backbone',original)
        for name,fn in [('vision_only',lambda:module(sample['image_input'][:,:2])),
            ('full_backbone',lambda:model.forward_vlm_efficient(sample['image_input'],sample['image_mask'],sample['input_ids'])['vlm_features'])]:
            for _ in range(20):value=fn()
            torch.cuda.synchronize();graphs=compiler.graph_count();times=[]
            for _ in range(200):
                torch.cuda.synchronize();start=time.perf_counter();value=fn();torch.cuda.synchronize()
                times.append(1000*(time.perf_counter()-start))
            if compiler.graph_count()!=graphs:raise RuntimeError('Compilation entered measured iterations')
            report[name]=stats(times)
            if name=='vision_only':
                difference=float((value-expected).abs().max());report['compiled_max_feature_diff']=difference
                torch.testing.assert_close(value,expected,atol=1e-4,rtol=1e-4)
        v=report['vision_only']['mean'];f=report['full_backbone']['mean']
        write_json(study.OUTPUT/'profile/summary.json',dict(verdict='PROFILE_COMPLETE',identity=c['identity'],
            measurements=report,compiler=compiler.records,hardware=torch.cuda.get_device_name(0),
            midpoint_extra_vision_ms_per_action=v/40,late_every_extra_vision_ms_per_action=v/10,
            removing_one_full_call_per_eight_queries_saves_ms_per_action=f/40,
            scope='Warm device-synchronized operators only. Full rollout measures preprocessing, updater, action decoder and queue separately.'))
