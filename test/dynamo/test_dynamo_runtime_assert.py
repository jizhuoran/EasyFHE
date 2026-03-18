# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._dynamo.utils


class DenseBlock(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)
        self.norm = torch.nn.LayerNorm(dim)
        self.gate = torch.nn.Linear(dim, dim)

    def forward(self, x):
        return self.norm(self.linear(x)) * torch.sigmoid(self.gate(x))


class DenseArch(torch.nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.blocks = torch.nn.ModuleList([DenseBlock(dim) for _ in range(num_layers)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class RecModel(torch.nn.Module):
    """Simplified recommendation model with many nested submodules."""

    def __init__(self, num_events=10, num_layers=6, num_embeddings=12, dim=128):
        super().__init__()
        self.shared_arch = DenseArch(dim, num_layers)
        self.event_submodels = torch.nn.ModuleDict(
            {
                f"event_{i}": torch.nn.Sequential(
                    DenseArch(dim, num_layers),
                    torch.nn.Linear(dim, 1),
                )
                for i in range(num_events)
            }
        )
        self.embeddings = torch.nn.ModuleList(
            [torch.nn.Embedding(1000, dim) for _ in range(num_embeddings)]
        )

    def forward(self, x):
        x = self.shared_arch(x)
        outputs = []
        for submodel in self.event_submodels.values():
            outputs.append(submodel(x))
        return torch.cat(outputs, dim=-1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
