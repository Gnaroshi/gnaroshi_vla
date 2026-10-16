"""Summarize all completed goal traces and extract explicitly sampled videos."""
import argparse
from pathlib import Path

from tools.simvla import attention_block_train as block
from tools.simvla import attention_query_train as query
from tools.simvla.attention_query_evidence import video_sheet
from tools.simvla.compile_benchmark import read_json, write_json, sha


def classify(record):
    trace = record['query_goal_progress'] + [record['terminal_goal_progress']]
    states = [row['satisfied'] for row in trace]
    ever = [any(state[i] for state in states) for i in range(len(states[0]))]
    final = states[-1]
    gained = [a and not b for a,b in zip(ever,states[0])]
    return dict(never_any=not any(ever), some_at_end=0 < sum(final) < len(final),
        lost_predicate=any(a and not b for a,b in zip(ever,final)),
        all_ever_not_simultaneous=all(ever) and not any(all(s) for s in states),
        newly_achieved_after_start=any(gained),
        lost_newly_achieved=any(a and not b for a,b in zip(gained,final)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    rows, sheets = {}, []
    for root in (query.OUTPUT,block.OUTPUT):
        for path in sorted(root.glob('rows/*/summary.json')):
            arm = path.parent.name
            records = read_json(path.parent/'episode_outcomes.json')
            if len(records) != 500 or len({(r['task_id'],r['trial_id']) for r in records}) != 500:
                raise RuntimeError('Expected500 unique episodes')
            failures = [r for r in records if not r['success']]
            labels = [classify(r) for r in failures]
            counts = {k:sum(label[k] for label in labels) for k in labels[0]}
            per_task = {}
            for task in range(10):
                local = [classify(r) for r in failures if r['task_id']==task]
                per_task[str(task)] = dict(failures=len(local),**{
                    k:sum(label[k] for label in local) for k in counts})
            rows[arm] = dict(summary=read_json(path),summary_path=str(path),summary_sha256=sha(path),
                train=read_json(root/'train'/arm/'summary.json'),
                outcome_path=str(path.parent/'episode_outcomes.json'),
                outcome_sha256=sha(path.parent/'episode_outcomes.json'),
                failures=len(failures),goal_categories=counts,per_task=per_task)
            if arm in ('visual_predictor_only','condition_updater_only'):
                for task in (4,6,8,9):
                    candidates = [r for r in failures if r['task_id']==task]
                    # Sample one failure per task/arm; do not claim exhaustive visual review.
                    record = min(candidates,key=lambda r:r['trial_id'])
                    sheet = args.output/'video_sheets'/arm/f"task{task}_trial{record['trial_id']}.jpg"
                    sheets.append(video_sheet(Path(record['video']),sheet,record))
    write_json(args.output/'evidence.json',dict(rows=rows,videos=sheets,
        goal_trace_episodes=sum(r['summary']['episodes'] for r in rows.values()),
        category_note='Categories overlap. Predicates checked at each query and terminal state; between-query transients can be missed.',
        video_note='Eight selected failed episode contact sheets, twelve frames each; extraction alone is not visual review.'))
    print({arm:dict(sr=row['summary']['success_rate'],ms=row['summary']['policy_ms_per_action'],
        failures=row['failures'],goals=row['goal_categories']) for arm,row in rows.items()})


if __name__ == '__main__':
    main()
