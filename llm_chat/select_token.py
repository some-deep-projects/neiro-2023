# select_token.py
import typing as tp
from abc import ABC, abstractmethod

import torch
from torch import Tensor


class TokenSelection(ABC):
    @abstractmethod
    def __call__(self, logits: Tensor):
        pass


class NucleusSelection(TokenSelection):
    def __init__(self, temperature: float = 1., top_k: tp.Optional[int] = None, top_p: tp.Optional[float] = None):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def __call__(self, logits: Tensor):
        top_k, top_p = self.top_k, self.top_p
        inds = torch.arange(logits.size(0))
        logits = logits / self.temperature

        if (top_k is not None) or (top_p is not None):
            inds = logits.argsort(descending=True)
            logits = logits[inds]

        if top_k is not None:
            min_topk_val = logits[min(top_k, logits.size(0) - 1)]
            topk_mask = (logits >= min_topk_val)
            logits = logits[topk_mask]
            inds = inds[topk_mask]

        if top_p is not None:
            sum_probs = torch.softmax(logits.float(), dim=0).cumsum(dim=0)
            probs_mask = (sum_probs <= top_p)
            probs_mask = probs_mask.roll(1)
            probs_mask[0] = True
            logits = logits[probs_mask]
            inds = inds[probs_mask]

        probs = torch.softmax(logits.float(), dim=0)
        return inds[torch.multinomial(probs, num_samples=1)]
