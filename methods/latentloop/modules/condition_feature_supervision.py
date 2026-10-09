"""Training-only readers of the existing observation code; no deployment changes."""
import torch
from torch import nn
from torch.nn import functional as F

MODES = ('none', 'absolute', 'delta', 'both')


class ConditionFeatureSupervision(nn.Module):
    def __init__(self, code_dim=128, condition_dim=960, max_tokens=256, seed=7):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.code = nn.Sequential(nn.LayerNorm(code_dim), nn.Linear(code_dim, 64))
            self.position = nn.Embedding(max_tokens, 64)
            self.absolute = nn.Linear(64, condition_dim)
            self.delta = nn.Linear(64, condition_dim)
        self.condition_dim = condition_dim

    def forward(self, code, token_count):
        if not 0 < token_count <= self.position.num_embeddings:
            raise ValueError('Unsupported teacher token count')
        positions = torch.arange(token_count, device=code.device)
        feature = F.gelu(self.code(code).unsqueeze(1) + self.position(positions).unsqueeze(0))
        return {'absolute': self.absolute(feature), 'delta': self.delta(feature)}

    def objective(self, codes, sequence, mode):
        if mode not in MODES or not codes:
            raise ValueError('Invalid feature supervision')
        mask = sequence['valid_mask'].bool() & (sequence['group_ids'] >= 1) & (sequence['group_ids'] <= 3)
        if not bool(mask.any()):
            raise ValueError('No source-identified image tokens')
        previous = F.layer_norm(sequence['anchor_condition'].detach().float(), (self.condition_dim,))
        values = {'absolute': [], 'delta': []}
        for index, code in enumerate(codes):
            current = F.layer_norm(sequence['teacher_conditions'][:, index].detach().float(), (self.condition_dim,))
            predicted = self(code, current.shape[1])
            values['absolute'].append((predicted['absolute'].float()-current).square()[mask].mean())
            values['delta'].append((predicted['delta'].float()-(current-previous)).square()[mask].mean())
            previous = current
        means = {k: torch.stack(v).mean() for k, v in values.items()}
        loss = ((means['absolute'] + means['delta']) / 2 if mode == 'both'
            else codes[0].new_zeros(()) if mode == 'none' else means[mode])
        return loss, means
