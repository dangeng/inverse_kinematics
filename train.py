import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import InvKin

B = 512         # Batch size

# Make model, loss, and optimizer
model = InvKin()
optimizer = Adam(model.parameters(), lr=1e-1)

losses = []
for iter in tqdm(range(100)):
    # Sample random point in "roughly" reachable space: [-.25, .25]**3
    xs = torch.rand((B, 3)) * .25 - .125

    # Forward through model
    thetas, pred_x = model(xs)

    # Penalize L2 dist btwn target and actual position
    dist = torch.linalg.norm(xs - pred_x, dim=1)
    loss = dist.mean()

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Track losses
    losses.append(loss.item())

# Save model
torch.save(model.state_dict(), 'chkpt.pth')

plt.plot(losses)
plt.show()