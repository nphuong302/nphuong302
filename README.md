<h1>Hi, I'm Phương Nguyễn 🌿</h1>

<p>
Machine Learning Developer with a background in software engineering and control & automation.  

Interested in AI as both a technical craft and a human one.

</p>

## 🪄 My Work

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, dim, activation):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.norm(self.linear(x)))

class SoulAlignedNet(nn.Module):
    """
    Project: SoulInTheLoop
    Blending Technology, Art, and Ethics with human-centered intent.
    """
    def __init__(self, dim=256):
        super().__init__()
        self.ai = Block(dim, F.relu)                               # make it work
        self.art = Block(dim, torch.tanh)                          # make it felt
        self.ethics = Block(dim, lambda x: torch.clamp(x, -1, 1))  # make it right

    def forward(self, x):
        return self.ethics(self.art(self.ai(x)))

model = SoulAlignedNet()
model.train()  # training continues in life
```
