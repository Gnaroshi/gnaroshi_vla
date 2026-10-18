"""Finish the two missing short-interval Bridge action-noise seeds."""
from tools.simvla import matched_reference_seeds as base


def configure():
    base.OUTPUT = base.RESULTS / 'matched_references/short_intervals_second_noise_v1'
    base.MODULE = 'tools.simvla.bridge_remaining_seeds'
    base.ROWS = ('bridge_f2', 'bridge_f3')
    base.ORIGINS = {row:'visual_memory_followup/reference_age_and_controls_v1' for row in base.ROWS}
    base.FILES = base.FILES + ('tools/simvla/bridge_remaining_seeds.py',
        'architectures/simvla/wrappers/run_bridge_remaining_seeds.sh',
        'tests/test_condition_routing.py')


if __name__=='__main__':
    configure()
    raise SystemExit(base.main())
