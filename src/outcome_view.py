# Credit
# https://hackernoon.com/how-to-plot-a-decision-boundary-for-machine-learning-algorithms-in-python-3o1n3w07 (accessed 13.08.2024)
# https://stackoverflow.com/questions/45075638/graph-k-nn-decision-boundaries-in-matplotlib (accessed 13.08.2024)
# Color from https://scikit-learn.org/stable/auto_examples/neighbors/plot_nca_classification.html#sphx-glr-auto-examples-neighbors-plot-nca-classification-py (accessed 14.08.2024)

from load_pretrained import get_pretrained_args, load_pretrained
from huggingface_sb3 import EnvironmentName
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import defaultdict
from utils import seed_everything
plt.rcParams.update({'font.size': 11})


def load_model_env(algo: str, env_id: str):
    args = get_pretrained_args()
    args.algo = algo
    args.env = EnvironmentName(env_id)
    args.no_render = True
    args.env_kwargs = {"render_mode": "rgb_array"}
    return load_pretrained(args)


def sim_trajector(env_id: str, seed: int):
    seed_everything(seed, True)
    info = load_model_env("dqn", env_id)
    env = info["env"]
    model = info["model"]

    containers = defaultdict(list)
    obs = env.reset()
    done = False

    while not done:
        with torch.no_grad():
            obs_tensor, _ = model.q_net.obs_to_tensor(obs)
            q_values = model.q_net(obs_tensor).numpy(force=True)

            action = np.argmax(q_values, 1)

        obs, _, done, _ = env.step(action)
        containers["obs"].append(obs)
        containers["img"].append(env.render())
        containers["action"].append(action)
        containers["value"].append(q_values[0, np.squeeze(action)])

    return (
        np.concatenate(containers["obs"], 0)[:-1],
        np.stack(containers["img"], 0)[:-1],
        np.concatenate(containers["action"])[:-1],
        np.array(containers["value"])[:-1]
    )


def sim_osf_trajector(env_id: str, seed: int, timestep: int, delta: float,
                      org_value):
    seed_everything(seed, True)
    info = load_model_env("dqn", env_id)
    env = info["env"]
    model = info["model"]

    containers = defaultdict(list)
    obs = env.reset()
    done = False
    i = 0
    osf_timestep = -1

    while not done:
        with torch.no_grad():
            obs_tensor, _ = model.q_net.obs_to_tensor(obs)
            q_values = model.q_net(obs_tensor).numpy(force=True)

        if i <= timestep or osf_timestep != -1:
            action = np.argmax(q_values, 1)
        elif org_value[i] - q_values[0, np.squeeze(action)] > delta:
            if osf_timestep == -1:
                osf_timestep = i - 1
            action = np.argmax(q_values, 1)

        i = i + 1

        obs, _, done, _ = env.step(action)
        containers["obs"].append(obs)
        containers["img"].append(env.render())
        containers["action"].append(action)
        containers["value"].append(q_values[0, np.squeeze(action)])

    return (
        np.concatenate(containers["obs"], 0)[:-1],
        np.stack(containers["img"], 0)[:-1],
        np.concatenate(containers["action"])[:-1],
        np.array(containers["value"])[:-1],
        osf_timestep
    )


if __name__ == "__main__":
    env_id = "MountainCar-v0"
    seed = 93
    oobs, oimg, oaction, ovalue = sim_trajector(env_id, seed)

    query_timestep = 35
    delta = 5

    osf_obs, osf_img, osf_action, osf_value, osf_timestep = sim_osf_trajector(
        env_id, seed, query_timestep, delta, ovalue)

    # Plot decision boundaries
    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
    cmap_bold = ["#FF0000", "#00FF00", "#0000FF"]

    fig, ax = plt.subplots(layout="constrained")


    x1grid = np.linspace(-1.2, 0.6, 1_000)
    x2grid = np.linspace(-0.07, 0.07, 1_000)

    xx, yy = np.meshgrid(x1grid, x2grid)

    grid = np.c_[xx.ravel(), yy.ravel()]

    obs = grid
    info = load_model_env("dqn", env_id)
    env = info["env"]
    model = info["model"]
    obs_tensor, _ = model.q_net.obs_to_tensor(obs)
    q_values = model.q_net(obs_tensor).numpy(force=True)

    yhat = np.argmax(q_values, 1)
    zz = yhat.reshape(xx.shape)

    ax.pcolormesh(xx, yy, zz, cmap=cmap_light,
                  linewidth=0, rasterized=True)

    # Query State
    qx, qy = oobs[query_timestep, 0], oobs[query_timestep, 1]
    qcolor = cmap_bold[1]
    ax.plot(oobs[query_timestep:, 0],
            oobs[query_timestep:, 1], color=qcolor, alpha=0.5)
    ax.scatter(qx, qy, c=qcolor, s=40)
    ax.text(qx - 0.01, qy - 0.007, "Query State", color=qcolor,
            fontsize=15)

    # Outcome-based Semi-factual State State
    qx, qy = osf_obs[osf_timestep, 0], osf_obs[osf_timestep, 1]
    osfcolor = cmap_bold[0]
    ax.plot(osf_obs[osf_timestep:, 0],
            osf_obs[osf_timestep:, 1], color=osfcolor, alpha=0.5)
    ax.scatter(qx, qy, c=osfcolor, s=40)
    ax.text(qx + 0.03, qy - 0.008,
            "Outcome-Based\nSemifactual State", color=osfcolor,
            fontsize=15)

    ax.set_ylabel("Velocity", fontsize=15)
    ax.set_xlabel("Position", fontsize=15)
    ax.axvspan(0.5, 0.6, color='yellow', alpha=0.5)

    fig.savefig("runs/osf_example.pdf")

    plt.close(fig)
