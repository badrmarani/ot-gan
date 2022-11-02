import torch
from torch import nn


class SinkhornLoss(nn.Module):
    def __init__(self) -> None:
        super(SinkhornLoss, self).__init__()

    def _cost(self, x: torch.tensor, y: torch.tensor) -> torch.tensor:
        x_norm = torch.linalg.norm(x, dim=1, keepdim=True)
        y_norm = torch.linalg.norm(y, dim=1, keepdim=True)
        return 1 - torch.matmul(x, y.t()).div(
            torch.max(x_norm * y_norm.t(), torch.tensor([1e-8]))
        )

    def forward(
        self, x: torch.tensor, y: torch.tensor, niter: int = 500, eps: float = 500
    ) -> torch.tensor:
        n = x.size(0)

        C = self._cost(x, y)
        K = torch.exp(-C / eps)
        a, b = torch.empty((n, 1), dtype=torch.float).fill_(1 / n), torch.empty(
            (n, 1), dtype=torch.float
        ).fill_(1 / n)

        for i in range(niter):
            a = torch.empty((n, 1), dtype=torch.float).fill_(1 / n).div(K.mm(b))
            b = torch.empty((n, 1), dtype=torch.float).fill_(1 / n).div(K.t().mm(a))

        loss = torch.matmul(torch.mm(K * C, b).t(), a)

        # a, b = a.squeeze(1), b.squeeze(1)
        # P = torch.diag(a).mm(K).mm(torch.diag(b))

        return loss


class SinkhornDiv(nn.Module):
    def __init__(self) -> None:
        super(SinkhornDiv, self).__init__()

    def forward(
        self,
        x: torch.tensor,
        xp: torch.tensor,
        y: torch.tensor,
        yp: torch.tensor,
        niter: int,
        eps: float,
    ) -> torch.tensor:
        W = SinkhornLoss()
        return (
            W(x, y, niter, eps)
            + W(x, yp, niter, eps)
            + W(xp, y, niter, eps)
            + W(xp, yp, niter, eps)
            - 2 * W(x, xp, niter, eps)
            - 2 * W(y, yp, niter, eps)
        )
