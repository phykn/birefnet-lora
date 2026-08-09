from dataclasses import dataclass

import torch


@dataclass
class Output:
    logits: list[torch.Tensor]
    gdt: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None
