# Credit
# https://hackernoon.com/how-to-plot-a-decision-boundary-for-machine-learning-algorithms-in-python-3o1n3w07 (13.08.2024)
# https://stackoverflow.com/questions/45075638/graph-k-nn-decision-boundaries-in-matplotlib (13.08.2024)
# Color from https://scikit-learn.org/stable/auto_examples/neighbors/plot_nca_classification.html#sphx-glr-auto-examples-neighbors-plot-nca-classification-py (14.08.2024)
from src.load_pretrained import get_pretrained_args, load_pretrained
from huggingface_sb3 import EnvironmentName
from src.utils import seed_everything
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Param:
    delta: float
    start_features: np.ndarray


def load_model_env(algo: str, env_id: str):
    args = get_pretrained_args()
    args.algo = algo
    args.env = EnvironmentName(env_id)
    args.no_render = True
    args.env_kwargs = {"render_mode": "rgb_array"}
    return load_pretrained(args)


def simulateMountainCar(start_features=None, seed: int = 0):
    seed_everything(seed, True)
    info = load_model_env("dqn", "MountainCar-v0")
    env = info["env"]
    model = info["model"]

    containers = defaultdict(list)

    done = False
    obs = env.reset()
    if start_features is not None:
        env.envs[0].unwrapped.state = start_features
        obs = env.envs[0].unwrapped.state[np.newaxis, ...]

    while not done:
        with torch.no_grad():
            obs_tensor, _ = model.q_net.obs_to_tensor(obs)
            q_values = model.q_net(obs_tensor).numpy(force=True)

            action = np.argmax(q_values, 1)

        containers["obs"].append(obs)
        containers["img"].append(env.render())
        containers["action"].append(action)
        containers["value"].append(q_values[0, np.squeeze(action)])

        obs, _, done, _ = env.step(action)

    return (
        model,
        np.concatenate(containers["obs"], 0),
        np.stack(containers["img"], 0),
        np.concatenate(containers["action"]),
        np.array(containers["value"])
    )


def simulateMountainCarOSF(start_features, original_value, delta: float, seed: int = 0):
    seed_everything(seed, True)
    info = load_model_env("dqn", "MountainCar-v0")
    env = info["env"]
    model = info["model"]

    containers = defaultdict(list)
    obs = env.reset()
    env.envs[0].unwrapped.state = start_features
    obs = env.envs[0].unwrapped.state[np.newaxis, ...]

    done = False
    i = 0
    osf_timestep = -1

    while not done:
        with torch.no_grad():
            obs_tensor, _ = model.q_net.obs_to_tensor(obs)
            q_values = model.q_net(obs_tensor).numpy(force=True)

        if i == 0 or osf_timestep != -1:
            action = np.argmax(q_values, 1)
        elif original_value[i] - q_values[0, np.squeeze(action)] > delta and np.argmax(q_values, 1) != action:
            if osf_timestep == -1:
                osf_timestep = i - 1
            action = np.argmax(q_values, 1)

        i = i + 1

        containers["obs"].append(obs)
        containers["img"].append(env.render())
        containers["action"].append(action)
        containers["value"].append(q_values[0, np.squeeze(action)])

        obs, _, done, _ = env.step(action)

    return (
        np.concatenate(containers["obs"], 0),
        np.stack(containers["img"], 0),
        np.concatenate(containers["action"]),
        np.array(containers["value"]),
        osf_timestep
    )


def plot_decision_boundary(model, ax):
    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])

    x1grid = np.linspace(-1.2, 0.6, 2_500)
    x2grid = np.linspace(-0.07, 0.07, 2_500)

    xx, yy = np.meshgrid(x1grid, x2grid)

    grid = np.c_[xx.ravel(), yy.ravel()]

    q_values = model.q_net(model.q_net.obs_to_tensor(grid)[
                           0]).numpy(force=True)

    yhat = np.argmax(q_values, 1)
    zz = yhat.reshape(xx.shape)

    ax.pcolormesh(xx, yy, zz, linewidth=0, rasterized=True, cmap=cmap_light,
                  alpha=0.8)


if __name__ == "__main__":
    params = Param(0.5, np.array([-0.52, 0.013]))
    model, obs, img, action, value = simulateMountainCar(
        params.start_features)

    osf_obs, osf_img, osf_action, osf_value, osf_timestep = simulateMountainCarOSF(
        params.start_features, value, params.delta)

    fig, ax = plt.subplots(layout="constrained")
    ax.set_ylabel("Velocity", fontsize=20)
    ax.set_xlabel("Position", fontsize=20)

    plot_decision_boundary(model, ax)

    qs = obs[:, 0], obs[:, 1]
    ax.plot(qs[0], qs[1], linewidth=3, color="C0", alpha=0.8)
    # ax.text(qs[0][0] - 0.1,
    #         qs[1][0] + 0.004, "Query State", color="C0")

    osf = osf_obs[:, 0], osf_obs[:, 1]
    ax.plot(osf[0], osf[1], linewidth=3, color="C1", alpha=0.8)

    ax.scatter(qs[0][0], qs[1][0], color="C0",
               linewidths=5, zorder=10)
    ax.scatter(osf[0][osf_timestep], osf[1][osf_timestep],
               color="C1", linewidths=5, zorder=10)
    # ax.text(osf[0][osf_timestep], osf[1]
    #         [osf_timestep] - 0.005, "OSF State", color="C1")

    plt.tick_params(axis='both', which='major', labelsize=15)
    ax.axvspan(0.5, 0.6, color='yellow', alpha=0.5)

    fig.savefig(
        Path().cwd() / "runs/mc" /
        f"MC__delta={params.delta}__sf={params.start_features.tolist()}.pdf",
        bbox_inches='tight', pad_inches=0.0)

    # Make video
    # fps = 25
    # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(
    #     *'mp4v'), fps, (img.shape[2], img.shape[1]))

    # for i in range(len(osf_img)):
    #     out.write(osf_img[i])

    # out.release()
