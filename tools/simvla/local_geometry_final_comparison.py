"""Compare completed variants to the equal-budget mean-pooling control."""
import argparse
import time

from tools.simvla import local_geometry_campaign as campaign
from tools.simvla.attention_block_campaign import paired_changes


def summarize():
    campaign.configure()
    base, training = campaign.base, campaign.training
    c = base.checked()
    rows, outcomes = {}, {}
    for arm in training.ARMS:
        path = training.OUTPUT/'rows'/arm/'summary.json'
        if not path.exists():
            continue
        summary, records = base.merged_result(c,arm)
        if base.read_json(path) != summary:
            raise RuntimeError('Merged summary differs from full episode evidence')
        rows[arm], outcomes[arm] = summary, records
    control = training.ARMS[0]
    matched = {}
    if control in rows:
        for arm in training.ARMS[1:]:
            if arm in rows:
                old_ms, new_ms = rows[control]['policy_ms_per_action'], rows[arm]['policy_ms_per_action']
                matched[arm] = dict(
                    success_rate_difference_pp=100*(rows[arm]['success_rate']-rows[control]['success_rate']),
                    policy_ms_per_action_difference=new_ms-old_ms if old_ms is not None and new_ms is not None else None,
                    paired_outcomes=paired_changes(outcomes[control],outcomes[arm]))
    result = dict(identity=c['identity'],complete=len(rows)==3,
        equal_budget_control=control,rows=rows,matched_comparisons=matched,
        initialization_reference=base.read_json(training.parent.OUTPUT/'rows/visual_predictor_only/summary.json'),
        initial_checkpoint_sha256=c['matched_parent_sha256'],
        interpretation='Each70K variant is compared to the independent70K mean-pooling control. The60K initialization is reported separately.',
        updated_unix=time.time())
    base.write_json(training.OUTPUT/'matched_comparison_summary.json',result)
    print(result,flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait',action='store_true')
    args = parser.parse_args()
    if args.wait:
        began = time.monotonic()
        path = campaign.training.OUTPUT/'queue_status.json'
        while True:
            state = campaign.base.read_json(path) if path.exists() else {}
            if state.get('phase') in ('complete','finished_with_failures'):
                break
            if time.monotonic()-began > 24*3600:
                raise TimeoutError('Main campaign still incomplete after24h')
            print('WAIT full three-arm campaign; no GPU allocation',flush=True)
            time.sleep(60)
    summarize()


if __name__ == '__main__':
    main()
