import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MaskedLinear"]


class MaskedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.mask = torch.ones_like(self.weight, dtype=torch.bool)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight * self.mask, self.bias)

    def prune_by_threshold(self, thr):
        self.mask *= (self.weight.abs() >= thr)

    def prune_by_rank(self, rank):
        if rank == 0:
            return
        weight_val = self.weight[self.mask == 1.]
        sorted_abs_weight = weight_val.abs().sort()[0]
        thr = sorted_abs_weight[rank]
        self.prune_by_threshold(thr)

    def prune_by_pct(self, pct):
        prune_idx = int(self.num_weight * pct)
        self.prune_by_rank(prune_idx)

    def random_prune_by_pct(self, pct):
        prune_idx = int(self.num_weight * pct)
        rand = torch.rand(size=self.mask.size(), device=self.mask.device, dtype=torch.float)
        rand_val = rand[self.mask == 1]
        sorted_abs_rand = rand_val.sort()[0]
        thr = sorted_abs_rand[prune_idx]
        self.mask *= (rand >= thr)

    def move_data(self, device: torch.device):
        self.mask = self.mask.to(device)

    def to(self, *args, **kwargs):
        device = torch._C._nn._parse_to(*args, **kwargs)[0]

        if device is not None:
            self.move_data(device)

        return super(MaskedLinear, self).to(*args, **kwargs)

    @property
    def num_weight(self) -> int:
        return self.mask.sum().item()
