import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import InvKin

B = 128         # Batch size

# Make model, loss, and optimizer
model = InvKin()
optimizer = Adam(model.parameters(), lr=1e-4)

losses = []
for iter in tqdm(range(1000)):
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



# Plot arms
'''
xs = torch.rand((16, 3)) * .2 - .1
thetas, pred_x = model(xs)
joint_locs = model.get_joints(thetas)
joint_locs = torch.stack(joint_locs, dim=2)
joint_locs = joint_locs.detach().numpy()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for jls in joint_locs:
    ax.plot(jls[0], jls[1], jls[2])

plt.show()
'''