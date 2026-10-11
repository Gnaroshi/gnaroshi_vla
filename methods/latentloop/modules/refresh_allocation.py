"""Causal placement of two backbone calls in each eight-query block."""
from collections import deque
import math
import random

STRATEGIES = ('even', 'burst', 'random', 'overlap')


class RefreshAllocation:
    def __init__(self, strategy, seed=7):
        if strategy not in STRATEGIES:
            raise ValueError(strategy)
        self.strategy = strategy
        self.rng = random.Random(seed)
        self.history = deque(maxlen=64)
        self.used_second = False
        self.random_slot = None
        self.last_query = -1
        self.last_full = None
        self.records = []

    def known_decision(self, t):
        """None requests a cheap action candidate; bool needs no score."""
        if t != self.last_query + 1:
            raise ValueError('Queries must be consecutive')
        phase = t % 8
        if phase == 0:
            self.used_second = False
            self.random_slot = self.rng.randrange(1, 8)
            return True
        if self.used_second:
            return False
        if self.strategy == 'even':
            return phase == 4
        if self.strategy == 'burst':
            return phase == 1
        if self.strategy == 'random':
            return phase == self.random_slot
        return True if phase == 7 else None

    def decide(self, t, known, score=None):
        if t != self.last_query + 1:
            raise ValueError('Queries must be consecutive')
        if score is not None and (not math.isfinite(score) or score < 0):
            raise ValueError('Score must be finite and nonnegative')
        phase = t % 8
        rank = None
        if known is None:
            if score is None:
                raise ValueError('Missing current-overlap score')
            if len(self.history) < 7:
                full = phase == 4
            else:
                # Rank only against past observations; a higher current score
                # spends the remaining call earlier. The last slot is reserved.
                rank = sum(x < score for x in self.history) / len(self.history)
                full = rank >= 1 - 1/(8-phase)
        else:
            full = known
        if score is not None:
            self.history.append(score)
        if full:
            if phase:
                if self.used_second:
                    raise RuntimeError('Block budget exceeded')
                self.used_second = True
            self.last_full = t
        elif self.last_full is None or t-self.last_full > 7:
            raise RuntimeError('Unsupported Condition prediction age')
        self.last_query = t
        record = dict(query=t, phase=phase, full=full, score=score,
                      past_rank=rank, age=t-self.last_full)
        self.records.append(record)
        return full

    def audit(self):
        for start in range(0, len(self.records), 8):
            block = self.records[start:start+8]
            count = sum(r['full'] for r in block)
            if count not in (1, 2) or (len(block) == 8 and count != 2):
                raise RuntimeError('Refresh budget mismatch')
            if not block[0]['full'] or any(r['age'] > 7 for r in block):
                raise RuntimeError('Invalid anchor or update age')
        return dict(queries=len(self.records), backbone_calls=sum(r['full'] for r in self.records),
                    complete_blocks=len(self.records)//8,
                    unfinished_block_queries=len(self.records)%8)
