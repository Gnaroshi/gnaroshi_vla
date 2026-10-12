"""Resume pending evaluations and test the causal role of current vision."""
import argparse
from types import MethodType, SimpleNamespace
import sys

from tools.simvla import visual_midpoint as study
from tools.simvla import failure_interventions as prior
from tools.simvla.compile_benchmark import ROOT, write_json, sha
from tools.simvla.gpu_followup_queue import run_queue

OUTPUT=study.OUTPUT/'followup'


def contract():
    c=study.checked()
    c=dict(c,parent_identity=c['identity'],
        intervention='Use refresh image_input for scheduled vision measurements; keep current raw_rgb, proprio, CNN deltas and all trained weights unchanged.',
        followup_source_sha256=sha(__file__))
    c['identity']=prior.digest(c)
    return c


def attach_stale(attach,policy,model,interval,compiler):
    policy=attach(policy,model,interval,compiler)
    full,update,reset=policy._full_refresh,policy._v0_update,policy.reset
    def full_reference(self,batch,*,policy_query_index):
        result=full(batch,policy_query_index=policy_query_index)
        self._refresh_image_input=batch['image_input']
        return result
    def update_reference(self,batch,*,age,policy_query_index):
        if self._refresh_image_input is None:raise RuntimeError('Refresh image is missing')
        modified=dict(batch,image_input=self._refresh_image_input)
        return update(modified,age=age,policy_query_index=policy_query_index)
    def reset_reference(self):
        reset();self._refresh_image_input=None
    policy._full_refresh=MethodType(full_reference,policy)
    policy._v0_update=MethodType(update_reference,policy)
    policy.reset=MethodType(reset_reference,policy)
    policy.reset()
    return policy


def evaluate_stale():
    from tools.simvla import visual_midpoint_eval as evaluation
    c=contract();write_json(OUTPUT/'stale_contract.json',c)
    original=evaluation.attach_visual
    evaluation.attach_visual=lambda *a,**kw:attach_stale(original,*a,**kw)
    evaluation.study=SimpleNamespace(OUTPUT=OUTPUT,checked=lambda:c,
        model_payload=study.model_payload,initial_model=study.initial_model,previous=study.previous)
    evaluation.evaluate('midpoint')


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('all','stale'),default='all',nargs='?')
    a=p.parse_args()
    if a.command=='stale':evaluate_stale();return 0
    c=study.checked();plan=[]
    if prior.IS_RB2:
        # The first launch observed a terminal status from a superseded setup
        # attempt. Preserve its active bridge row and resume only its successors.
        for job in study.jobs(c):
            if job['id'].startswith('eval'):
                job=dict(job,deps=[d for d in job.get('deps',[]) if d!='bridge_f4'])
                plan.append(job)
        deps=['eval_midpoint','eval_late_every']
    else:deps=[]
    plan.append(dict(id='stale_midpoint',cmd=[sys.executable,'-u','-m',
        'tools.simvla.visual_midpoint_followup','stale'],deps=deps,
        ready_file=str(study.OUTPUT/'rows/midpoint/summary.json'),
        ready_fields=dict(verdict='EVAL_COMPLETE',identity=c['identity'],episodes=500),
        summary=str(OUTPUT/'rows/midpoint/summary.json'),
        completion=dict(verdict='EVAL_COMPLETE',identity=contract()['identity'],episodes=500)))
    write_json(OUTPUT/'plan.json',dict(jobs=plan,source_sha256=sha(__file__),
        purpose='Matched trained model, same vision call schedule, replace only current vision images with refresh images.'))
    return run_queue(OUTPUT/'queue',plan,gpus=(0,) if prior.IS_RB2 else (4,5,6,7),
        predecessor=[dict(path=str(study.OUTPUT),lock='queue.lock',allow_when_all_assigned=not prior.IS_RB2)],
        environment=prior.environment,cwd=ROOT,timeout=16*3600)


if __name__=='__main__':raise SystemExit(main())
