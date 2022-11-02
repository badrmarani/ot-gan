import torch
from torch import nn
import random
import numpy as np
import torchvision


import matplotlib.pyplot as plt

from loss import SinkhornDiv
from model import Critic, Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


lr = 3e-4
betas = (.5, .999)
ngen = 3

batch_size = 32
epochs = 5
niter = 10
eps = 1

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(
    root="datasets/", train=True, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=2*batch_size, shuffle=True)

D = Critic().to(device)
G = Generator().to(device)

SD = SinkhornDiv()

optim_g = torch.optim.Adam(list(G.parameters()), lr=lr, betas=betas)
optim_d = torch.optim.Adam(list(D.parameters()), lr=lr, betas=betas)

train_loss_history = []
eval_loss_history = []
for ep in range(epochs):
    train_loss_G, train_loss_D = 0, 0
    G.train()
    D.train()
    for t, (x, _) in enumerate(train_loader):
        z = 2*torch.rand(size=(batch_size, 100))-1
        zp = 2*torch.rand(size=(batch_size, 100))-1

        x, xp = torch.split(x, batch_size)
        x, xp = x.to(device), xp.to(device)

        y, yp = D(G(z)), D(G(zp))
        x, xp = D(x), D(xp)

        if not (t+1) % (ngen+1):
            optim_d.zero_grad()
            loss_d = - SD(x, xp, y, yp, niter, eps)

            loss_d.backward()
            optim_d.step()
            train_loss_D += loss_d.item()
            # print(train_loss_D, loss_d.item())

        else:
            optim_g.zero_grad()
            loss_g = SD(x, xp, y, yp, niter, eps)

            loss_g.backward()
            optim_g.step()
            train_loss_G += loss_g.item()

            # print(train_loss_G, loss_g.item())

        with torch.no_grad():
            if not (t+1) % 10:
                    print("""
                    Epoch: {:^5}/{:^5}| Batch: {:^5}/{:^5} || SD Generator = {:^9.4f} SD Critic = {:^9.4f}
                    """.format(ep+1, epochs, t+1, len(train_loader), train_loss_G, train_loss_D))


    with torch.no_grad():
        n_samples = 5

        samples = [random.choice(2*torch.rand(size=(200,10,10))-1) for _ in range(n_samples)]
        out = np.concatenate([
            # *|CURSOR_MARCADOR|*
            G(x.view(1,-1).float()).cpu().numpy().transpose((1,2,0)) for x in samples 
        ], axis=1)

        plt.figure(figsize=(50,50))
        plt.imshow((out*255).astype(np.uint8), interpolation="nearest")
        plt.savefig(f"output-ot-gan-{ep+1}.jpg", dpi="figure")
    



# samples = [random.choice(train_dataset.data) for _ in range(n_samples)]
# out = np.concatenate([
#     G(x.view(1, -1).float()).view(3,32,32).detach().cpu().numpy().transpose((1,2,0)) for x in samples
# ], axis=1)


# plt.figure(figsize=(50,50))
# plt.plot(train_loss_history, label="train loss (eps=500, niter=500)")
# plt.plot(eval_loss_history, label="eval loss (eps=500, niter=500)")
# plt.legend()
# plt.grid()
# plt.savefig("sd-loss-ot-gan.jpg", dpi="figure")
