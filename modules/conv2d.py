import torch
from torch.nn.common_types import _size_2_t

__all__ = ["MaskedConv2d"]


class MaskedConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1,
                 padding: _size_2_t = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True,
                 padding_mode: str = 'zeros'):
        super(MaskedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                           dilation, groups, bias, padding_mode)
        self.mask = torch.ones_like(self.weight, dtype=torch.bool)

    def forward(self, inp):
        return self._conv_forward(inp, self.weight * self.mask)

    def prune_by_threshold(self, thr):
        self.mask *= (torch.abs(self.weight) >= thr)

    def prune_by_rank(self, rank):
        if rank == 0:
            return
        weights_val = self.weight[self.mask == 1]
        sorted_abs_weights = torch.sort(torch.abs(weights_val))[0]
        thr = sorted_abs_weights[rank]
        self.prune_by_threshold(thr)

    def prune_by_pct(self, pct):
        if pct == 0:
            return
        prune_idx = int(self.num_weight * pct)
        self.prune_by_rank(prune_idx)

    def random_prune_by_pct(self, pct):
        prune_idx = int(self.num_weight * pct)
        rand = torch.rand_like(self.mask, device=self.mask.device)
        rand_val = rand[self.mask == 1]
        sorted_abs_rand = torch.sort(rand_val)[0]
        thr = sorted_abs_rand[prune_idx]
        self.mask *= (rand >= thr)

    def move_data(self, device: torch.device):
        self.mask = self.mask.to(device)

    def to(self, *args, **kwargs):
        device = torch._C._nn._parse_to(*args, **kwargs)[0]

        if device is not None:
            self.move_data(device)

        return super(MaskedConv2d, self).to(*args, **kwargs)

    def cuda(self, device=None):
        self.to(device if device is not None else "cuda:0")

    @property
    def num_weight(self):
        return torch.sum(self.mask).int().item()
