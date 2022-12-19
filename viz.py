import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import InvKin

def plot_joint_locs(joint_locs, target_locs=None, save_path=None):
    '''
    joint_locs (np array): 
        joint locations of shape (batch, joint_idx, xyz) = (B, 3, 3)
    target_locs (np array): 
        optional target locations to plot, of shape (batch, xyz)
    save_path: 
        optional save path
    '''
    # Make 3d plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Plot each arm
    for jls in joint_locs:
        ax.plot(jls[:, 0], jls[:, 1], jls[:, 2])

    # Plot each target
    if target_locs is not None:
        ax.scatter(target_locs[:, 0], target_locs[:, 1], target_locs[:, 2])

    # These ranges are hardcoded...
    ax.set_xlim(-.25, .25)
    ax.set_ylim(-.25, .25)
    ax.set_zlim(-.25, .25)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def sanity_check(model, N=16, save_path=None):
    '''
    Spins all joints at constant rate, half revolution
    '''

    # Make thetas
    thetas = torch.zeros(N, 3)

    # Spin all joints by half revolution
    thetas[:, 0] = torch.linspace(0, 3.141592, N)
    thetas[:, 1] = torch.linspace(0, 3.141592, N)
    thetas[:, 2] = torch.linspace(0, 3.141592, N)

    # Get joint locs for each
    _, joint_locs = model.forward_model(thetas)
    joint_locs = joint_locs.numpy()

    plot_joint_locs(joint_locs, save_path=save_path)

def random_test(model, N=16, save_path=None):
    '''
    Sample random locations
    '''
    with torch.no_grad():
        # Sample random point in "roughly" reachable space: [-.25, .25]**3
        xs = torch.rand((N, 3)) * .25 - .125

        # Forward through model
        thetas, pred_x = model(xs)

        # Get joint locs for each prediction
        _, joint_locs = model.forward_model(thetas)

    plot_joint_locs(joint_locs, target_locs=xs, save_path=save_path)

model = InvKin()
chkpt = torch.load('chkpt_10k_lr1e-4.pth')
model.load_state_dict(chkpt)
model = model.eval()
#sanity_check(model)
random_test(model, N=3)

