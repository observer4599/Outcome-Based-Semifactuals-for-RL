from src.utils import seed_everything
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from functools import partial
from tqdm import trange
from src.play_with_osf import MODEL_PATH
import gymnasium as gym
from src.dqn import make_env, QNetwork
import tyro

ACTION_MAPPER = {"PongNoFrameskip-v4": {1: 0, 4: 2, 5: 3},
                 "BreakoutNoFrameskip-v4": {1: 0}}
ACTION_NAME = {
    "PongNoFrameskip-v4": {0: "Do nothing", 1: "Do nothing", 2: "Go up",
                           3: "Go down", 4: "Go up", 5: "Go down"},
    "BreakoutNoFrameskip-v4": {0: "Do nothing", 1: "Do nothing", 2: "Go right", 3: "Go left"},
}
BLACK = np.array([0, 0, 0])
WHITE_GRAY = np.array([230, 230, 230])
NUM_FRAMES = 10


def change_black_to_white_gray(env_id, img):
    # assuming dim=3 and color is the last channel
    if env_id != "BreakoutNoFrameskip-v4":
        return img
    img = np.copy(img)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            if (img[r, c] == BLACK).all():
                img[r, c] = WHITE_GRAY
    return img


def load_model_env(env_id: str, device: torch.device, seed: int):
    # 0 only envs, 1 only model, 2 both
    seed_everything(seed, True)

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed, 0, False, None, True)])

    model = QNetwork(envs).to(device)
    model.load_state_dict(torch.load(
        Path.cwd() / f"runs/{env_id}__dqn__{MODEL_PATH[env_id]}.cleanrl_model",
        map_location=device)
    )
    model.eval()

    return model, envs


def simulate(env_id: str, device: torch.device, seed: int = 0):
    seed_everything(seed, True)
    model, envs = load_model_env(env_id, device, seed)

    containers = defaultdict(list)

    done = False
    obs = envs.reset(seed=seed)[0]

    while not done:
        with torch.no_grad():
            q_values = model(torch.tensor(obs).to(device)).numpy(force=True)
            actions = np.argmax(q_values, 1)

        containers["obs"].append(obs)
        containers["img"].append(envs.unwrapped.envs[0].render())
        containers["action"].append(actions)
        containers["value"].append(q_values[0, np.squeeze(actions)])

        obs, _, terminations, truncations, _ = envs.step(actions)
        done = terminations or truncations

    return (
        model,
        np.concatenate(containers["obs"], 0),
        np.stack(containers["img"], 0),
        np.concatenate(containers["action"]),
        np.array(containers["value"])
    )


def simulateOSF(env_id: str, query_ts: int, original_value: np.ndarray,
                delta: float, device: torch.device, seed: int = 0):
    seed_everything(seed, True)
    model, envs = load_model_env(env_id, device, seed)

    containers = defaultdict(list)
    obs = envs.reset(seed=seed)[0]

    done = False
    i = 0
    osf_timestep = -1
    while not done:
        with torch.no_grad():
            q_values = model(torch.tensor(obs).to(device)).numpy(force=True)

        if i <= query_ts or osf_timestep != -1:
            actions = np.argmax(q_values, 1)
        elif original_value[i] - q_values[0, np.squeeze(actions)] > delta:
            if osf_timestep == -1:
                osf_timestep = i - 1
            actions = np.argmax(q_values, 1)

        i = i + 1

        containers["obs"].append(obs)
        containers["img"].append(envs.unwrapped.envs[0].render())
        containers["action"].append(actions)
        containers["value"].append(q_values[0, np.squeeze(action)])

        obs, _, terminations, truncations, _ = envs.step(actions)
        done = terminations or truncations

    return (
        np.concatenate(containers["obs"], 0),
        np.stack(containers["img"], 0),
        np.concatenate(containers["action"]),
        np.array(containers["value"]),
        osf_timestep
    )


def register_activation_hook(model, buffer):
    def save_activation_(self, input, output, buffer):
        buffer.append(output.numpy(force=True))
    save_activation = partial(save_activation_, buffer=buffer)

    hook = model.network[8].register_forward_hook(
        save_activation)

    return hook


def simulateCF(env_id: str, n_steps: int, device: torch.device, seed: int = 0):
    seed_everything(seed, True)
    model, envs = load_model_env(env_id, device, seed)

    containers = defaultdict(list)
    hook = register_activation_hook(model, containers["activation"])

    obs = envs.reset(seed=seed)[0]

    for _ in trange(n_steps):
        with torch.no_grad():
            q_values = model(torch.tensor(obs).to(device)).numpy(force=True)
            actions = np.argmax(q_values, 1)

        containers["obs"].append(obs)
        containers["img"].append(envs.unwrapped.envs[0].render())
        containers["action"].append(actions)
        containers["value"].append(q_values[0, np.squeeze(actions)])

        obs = envs.step(actions)[0]

    hook.remove()

    return (
        model,
        np.concatenate(containers["obs"], 0),
        np.stack(containers["img"], 0),
        np.concatenate(containers["action"]),
        np.array(containers["value"]),
        np.concatenate(containers["activation"], 0)
    )


def make_cf(args, folder, device, sample_closest: int = 100):

    _, obs, img, action, value, act = simulateCF(
        args.env_id, args.n_steps, device)
    idx = np.arange(len(obs))

    for key, value in ACTION_MAPPER[args.env_id].items():
        action[action == key] = value
    act_new = act[action != action[args.query_ts]]
    idx = idx[action != action[args.query_ts]]

    dist = np.sum(
        np.sqrt((act[args.query_ts:args.query_ts + 1] - act_new) ** 2), 1)

    random_idx = np.arange(sample_closest)
    np.random.shuffle(random_idx)

    closest = np.argpartition(dist, sample_closest)[
        random_idx[:args.n_img - 1]]

    ncols = args.n_img
    fig, ax = plt.subplots(
        figsize=(6 * ncols, 9), ncols=ncols, layout="constrained")
    for j in range(ncols):
        if j == 0:
            sim_idx = args.query_ts
        else:
            sim_idx = idx[closest[j - 1]]
        for i in range(sim_idx-NUM_FRAMES, sim_idx + 1):
            ax[j].imshow(change_black_to_white_gray(
                args.env_id, img[i]), alpha=args.alpha)
            ax[j].axis("off")

        ax[j].set_title(
            f"{sim_idx} - {ACTION_NAME[args.env_id][action[sim_idx]]}",
            fontsize=40
        )
    fig.savefig(folder / f"CF__env_id={args.env_id}__query={args.query_ts}__bs={args.n_steps}.pdf",
                bbox_inches='tight', pad_inches=0.0)


def ceildiv(a, b):
    # Credit
    # https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    return -(a // -b)


if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")

    @dataclass
    class Args:
        task: str  # CF or OSF
        env_id: str
        delta: float
        query_ts: int
        n_img: int = 5
        alpha: float = 0.6
        seed: int = 0

        # If CF
        n_steps: int = 100_000

    args = tyro.cli(Args)
    assert args.task in (
        "CF", "OSF", "MIX"), f"{args.task} is not a valid task."
    seed_everything(args.seed, True)

    folder = Path.cwd() / \
        f"runs/{args.task}"
    if not folder.exists():
        folder.mkdir(parents=True)

    if args.task == "MIX":

        # COUNTERFACTUAL
        _, obs, img, action, value, act = simulateCF(
            args.env_id, args.n_steps, DEVICE)
        idx = np.arange(len(obs))

        for key, value in ACTION_MAPPER[args.env_id].items():
            action[action == key] = value
        act_new = act[action != action[args.query_ts]]
        idx = idx[action != action[args.query_ts]]

        dist = np.sum(
            np.sqrt((act[args.query_ts:args.query_ts + 1] - act_new) ** 2), 1)
        sim_idx = idx[np.argmin(dist)]
        ncols = 1 + args.n_img

        fig, ax = plt.subplots(
            figsize=(6 * ncols, 9), ncols=ncols, layout="constrained")

        for i in range(9, 0, -1):
            ax[0].imshow(img[args.query_ts - i], alpha=args.alpha)
            ax[1].imshow(img[sim_idx - i], alpha=args.alpha)

        for i in range(ncols):
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['bottom'].set_visible(False)
            ax[i].spines['left'].set_visible(False)

        ax[0].set_title(
            f"{args.query_ts} - {ACTION_NAME[args.env_id][action[args.query_ts]]}",
            fontsize=40
        )
        ax[1].set_title(
            f"{sim_idx} - {ACTION_NAME[args.env_id][action[sim_idx]]}",
            fontsize=40
        )

        # OSF
        model, obs, img, action, value = simulate(
            args.env_id, DEVICE, args.seed)

        osf_obs, osf_img, osf_action, osf_value, osf_timestep = simulateOSF(
            args.env_id, args.query_ts, value, args.delta, DEVICE, args.seed)
        timesplit = ceildiv(osf_timestep - args.query_ts, 3)
        show_steps = [args.query_ts + timesplit,
                      args.query_ts + 2 * timesplit,
                      osf_timestep,
                      args.query_ts + 4 * timesplit]

        for i, timestep in enumerate(show_steps, start=2):
            # OSF
            for j in range(timestep - NUM_FRAMES, timestep + 1):
                ax[i].imshow(change_black_to_white_gray(
                    args.env_id, osf_img[j]), alpha=args.alpha)

            ax[i].set_title(
                f"{timestep} - {ACTION_NAME[args.env_id][osf_action[timestep]]}", fontsize=40)

        fig.savefig("motivation_example.svg",
                    bbox_inches='tight',
                    pad_inches=0.0
                    )

    elif args.task == "CF":
        make_cf(args, folder, DEVICE)

    elif args.task == "OSF":
        model, obs, img, action, value = simulate(
            args.env_id, DEVICE, args.seed)

        osf_obs, osf_img, osf_action, osf_value, osf_timestep = simulateOSF(
            args.env_id, args.query_ts, value, args.delta, DEVICE, args.seed)
        timesplit = ceildiv(osf_timestep - args.query_ts, 3)
        show_steps = [args.query_ts,
                      args.query_ts + timesplit,
                      args.query_ts + 2 * timesplit,
                      osf_timestep,
                      args.query_ts + 4 * timesplit]

        fig, ax = plt.subplots(figsize=(
            6.5 * args.n_img, 9 * 2), ncols=args.n_img, nrows=2, layout="constrained")
        for i, timestep in enumerate(show_steps):
            # Original
            for j in range(timestep - NUM_FRAMES, timestep + 1):
                ax[0, i].imshow(change_black_to_white_gray(
                    args.env_id, img[j]), alpha=args.alpha)

            ax[0, i].set_xticks([])
            ax[0, i].set_yticks([])
            ax[0, i].spines['top'].set_visible(False)
            ax[0, i].spines['right'].set_visible(False)
            ax[0, i].spines['bottom'].set_visible(False)
            ax[0, i].spines['left'].set_visible(False)

            if i == 0:
                ax[0, i].set_title(
                    f"{timestep} - {ACTION_NAME[args.env_id][action[timestep]]}",
                    fontsize=50)
                ax[0, i].set_ylabel("Original", rotation=90, fontsize=50)
            else:
                ax[0, i].set_title(timestep, fontsize=50)

            # OSF
            for j in range(timestep - NUM_FRAMES, timestep + 1):
                ax[1, i].imshow(change_black_to_white_gray(
                    args.env_id, osf_img[j]), alpha=args.alpha)

            ax[1, i].set_xticks([])
            ax[1, i].set_yticks([])
            ax[1, i].spines['top'].set_visible(False)
            ax[1, i].spines['right'].set_visible(False)
            ax[1, i].spines['bottom'].set_visible(False)
            ax[1, i].spines['left'].set_visible(False)

            if i == 0:
                ax[1, i].set_title(
                    f"{timestep} - {ACTION_NAME[args.env_id][action[timestep]]}",
                    fontsize=50)
                ax[1, i].set_ylabel("OSF", rotation=90, fontsize=50)
            else:
                ax[1, i].set_title(timestep, fontsize=50)

            fig.savefig(
                folder /
                f"OSF__env_id={args.env_id}__query={args.query_ts}__delta={args.delta}.pdf",
                bbox_inches='tight',
                pad_inches=0.0
            )
