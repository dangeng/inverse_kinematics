import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class InvKin(nn.Module):
    def __init__(self, n_theta=3, hidden_dim=64):
        super().__init__()

        self.n_theta = n_theta

        self.net = nn.Sequential(
                        nn.Linear(3, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, n_theta)
                   )

    def get_dh_array(self, d, theta, r, alpha):
        row_0 = torch.stack([torch.cos(theta), -torch.sin(theta) * torch.cos(alpha), torch.sin(theta) * torch.sin(alpha), r * torch.cos(theta)])
        row_1 = torch.stack([torch.sin(theta), torch.cos(theta) * torch.cos(alpha), -torch.cos(theta) * torch.sin(alpha), r * torch.sin(theta)])
        row_2 = torch.stack([torch.tensor(0.0), torch.sin(alpha), torch.cos(alpha), d])
        row_3 = torch.tensor([0., 0., 0., 1.])
        dh = torch.stack([row_0, row_1, row_2, row_3])

        return dh

    def get_dh_array_vec(self, ds, thetas, rs, alphas):
        dhs = []
        for d, theta, r, alpha in zip(ds, thetas, rs, alphas):
            dh = self.get_dh_array(d, theta, r, alpha)
            dhs.append(dh)

        dhs = torch.stack(dhs)
        return dhs

    def forward_model(self, thetas):
        # Keep track of all joint locs
        joint_locs = [torch.zeros(thetas.shape[0], 3)]

        zeros = torch.zeros(thetas.shape[0])
        ones = torch.ones(thetas.shape[0])
        rs = [.12, .115]

        dhs = self.get_dh_array_vec(zeros, thetas.T[0], zeros, ones * np.pi / 2)
        for theta, r in zip(thetas.T[1:], rs):
            new_dhs = self.get_dh_array_vec(zeros, theta, ones * r, zeros)
            dhs = torch.bmm(dhs, new_dhs)

            # Append joint locs to list
            joint_locs.append(dhs[:, 0:3, 3])

        # joint_locs reshape: (b, 3) -> (b, 3 (joint idx), 3 (x,y,z))
        joint_locs = torch.stack(joint_locs, dim=1)

        # Return end effector position + all joint location
        x = dhs[:, 0:3, 3]
        return x, joint_locs

    def forward(self, x):
        thetas = self.net(x)
        pred_x, _ = self.forward_model(thetas)
        return thetas, pred_x
