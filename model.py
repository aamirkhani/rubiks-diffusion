"""Cost-to-go value network (DeepCubeA-style residual MLP, LayerNorm variant)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.l1 = nn.Linear(h, h)
        self.n1 = nn.LayerNorm(h)
        self.l2 = nn.Linear(h, h)
        self.n2 = nn.LayerNorm(h)

    def forward(self, x):
        y = F.relu(self.n1(self.l1(x)))
        y = self.n2(self.l2(y))
        return F.relu(x + y)


class PolicyNet(nn.Module):
    """Diffusion-style denoiser: predicts which move undoes the last scramble
    step. Input: [B, S] int8 states. Output: [B, M] move logits."""

    def __init__(self, S, M, h1=5000, h2=1000, blocks=4, t_dim=0, vocab=6):
        super().__init__()
        self.S, self.M, self.t_dim, self.vocab = S, M, t_dim, vocab
        self.fc1 = nn.Linear(S * vocab + t_dim, h1)
        self.n1 = nn.LayerNorm(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.n2 = nn.LayerNorm(h2)
        self.blocks = nn.ModuleList([ResBlock(h2) for _ in range(blocks)])
        self.out = nn.Linear(h2, M)

    def forward(self, states, t=None):
        x = F.one_hot(states.long(), self.vocab).to(torch.float32).flatten(1)
        if self.t_dim > 0:
            if t is None:  # unknown noise level at solve time -> assume max
                t = torch.full((states.shape[0],), self.t_dim, device=states.device)
            th = F.one_hot(t.clamp(1, self.t_dim).long() - 1, self.t_dim).to(torch.float32)
            x = torch.cat([x, th], dim=1)
        x = F.relu(self.n1(self.fc1(x)))
        x = F.relu(self.n2(self.fc2(x)))
        for b in self.blocks:
            x = b(x)
        return self.out(x)


class ValueNet(nn.Module):
    """Input: [B, S] int8 sticker states. Output: [B] cost-to-go estimate."""

    def __init__(self, S, h1=5000, h2=1000, blocks=4, vocab=6):
        super().__init__()
        self.S, self.vocab = S, vocab
        self.fc1 = nn.Linear(S * vocab, h1)
        self.n1 = nn.LayerNorm(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.n2 = nn.LayerNorm(h2)
        self.blocks = nn.ModuleList([ResBlock(h2) for _ in range(blocks)])
        self.out = nn.Linear(h2, 1)

    def forward(self, states):
        x = F.one_hot(states.long(), self.vocab).to(torch.float32).flatten(1)
        x = F.relu(self.n1(self.fc1(x)))
        x = F.relu(self.n2(self.fc2(x)))
        for b in self.blocks:
            x = b(x)
        return self.out(x).squeeze(1)
