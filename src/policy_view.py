# Credit
# https://hackernoon.com/how-to-plot-a-decision-boundary-for-machine-learning-algorithms-in-python-3o1n3w07 (13.08.2024)
# https://stackoverflow.com/questions/45075638/graph-k-nn-decision-boundaries-in-matplotlib (13.08.2024)
# Color from https://scikit-learn.org/stable/auto_examples/neighbors/plot_nca_classification.html#sphx-glr-auto-examples-neighbors-plot-nca-classification-py (14.08.2024)

from load_pretrained import get_pretrained_args, load_pretrained
from huggingface_sb3 import EnvironmentName
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def load_model_env(algo: str, env_id: str):
    args = get_pretrained_args()
    args.algo = algo
    args.env = EnvironmentName(env_id)
    args.no_render = True
    args.env_kwargs = {"render_mode": "rgb_array"}
    return load_pretrained(args)


if __name__ == "__main__":
    env_id = "MountainCar-v0"
    info = load_model_env("dqn", env_id)

    env = info["env"]
    model = info["model"]

    fig, ax = plt.subplots(layout="constrained")

    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
    cmap_bold = ["#FF0000", "#00FF00", "#0000FF"]


    x1grid = np.linspace(-1.2, 0.6, 1_000)
    x2grid = np.linspace(-0.07, 0.07, 1_000)

    xx, yy = np.meshgrid(x1grid, x2grid)

    grid = np.c_[xx.ravel(), yy.ravel()]

    obs = grid
    obs_tensor, _ = model.q_net.obs_to_tensor(obs)
    q_values = model.q_net(obs_tensor).numpy(force=True)

    yhat = np.argmax(q_values, 1)
    zz = yhat.reshape(xx.shape)

    ax.pcolormesh(xx, yy, zz, cmap=cmap_light,
                  linewidth=0, rasterized=True)

    # Create Trajectory
    obs_traj = []
    action_traj = []
    obs = env.reset()
    done = False

    while not done:
        with torch.no_grad():
            obs_tensor, _ = model.q_net.obs_to_tensor(obs)
            q_values = model.q_net(obs_tensor).numpy(force=True)

            action = np.argmax(q_values, 1)

        obs, _, done, _ = env.step(action)
        if not done:
            obs_traj.append(obs)
            action_traj.append(action)

    obs_traj = np.concatenate(obs_traj, 0)
    action_traj = np.concatenate(action_traj)

    # ax.plot(obs_traj[:, 0], obs_traj[:, 1],
    #         c=sns.color_palette("colorblind", as_cmap=True)[6],
    #         linewidth=3)

    # Query State
    qx, qy = obs_traj[35, 0], obs_traj[35, 1]
    qcolor = cmap_bold[1]
    ax.scatter(qx, qy, c=qcolor, s=40)
    ax.text(qx - 0.01, qy - 0.007, "Query State", color=qcolor, fontsize=15)

    # Semi-factual State
    sfx, sfy = obs_traj[46, 0], obs_traj[46, 1]
    sfcolor = cmap_bold[2]
    ax.scatter(sfx, sfy, c=sfcolor, s=40)
    ax.text(sfx - 0.05, sfy - 0.007, "Semifactual State",
            color=sfcolor, fontsize=15)

    # Counterfactual State
    cx, cy = obs_traj[51, 0], obs_traj[51, 1]
    ccolor = cmap_bold[0]
    ax.scatter(cx, cy, c=ccolor, s=40)
    ax.text(cx - 0.1, cy + 0.004, "Counterfactual State",
            color=ccolor, fontsize=15)

    ax.set_ylabel("Velocity", fontsize=15)
    ax.set_xlabel("Position", fontsize=15)
    ax.axvspan(0.5, 0.6, color='yellow', alpha=0.5)

    fig.savefig("runs/cf_sf_example.pdf")
