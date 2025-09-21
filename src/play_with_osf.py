from src.utils import seed_everything
import torch
import numpy as np
from collections import defaultdict
import cv2
from pathlib import Path
from src.dqn import QNetwork, make_env
import gymnasium as gym
from tqdm import trange
import statistics
import tyro
from dataclasses import dataclass
import random
from joblib import Parallel, delayed


MODEL_PATH = {
    "BreakoutNoFrameskip-v4": "0__1725144952/dqn__9999999",
    "PongNoFrameskip-v4": "0__1725144941/dqn__9999999",
    "MsPacmanNoFrameskip-v4": "0__1725182818/dqn__9999999",
    "SpaceInvadersNoFrameskip-v4": "0__1725182859/dqn__9999999",
    "AssaultNoFrameskip-v4": "0__1725220155/dqn__9999999",
    "SeaquestNoFrameskip-v4": "0__1725220186/dqn__9999999"
}


def compute_results(result):
    result_dict = {}

    ts_a = {"ts/a": (np.array(
        result["episodic_length"]) / np.array(result["action_switch"])).tolist()}

    for key, value in {**result, **ts_a}.items():
        try:
            result_dict[f"{key}_mean"] = statistics.mean(value)
            result_dict[f"{key}_stdev"] = statistics.stdev(value)
        except:
            if f"{key}_mean" in result_dict.keys():
                del result_dict[f"{key}_mean"]
            if f"{key}_stdev" in result_dict.keys():
                del result_dict[f"{key}_stdev"]

    return result_dict


def load_model_envs(env_id: str, device, seed: int = 0,
                    mode: int = 2, render: bool = False):
    # 0 only envs, 1 only model, 2 both
    assert mode in (0, 1, 2), f"{mode} is illegal."
    seed_everything(seed, True)

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed, 0, False, None, render)])

    if mode == 0:
        return envs

    model = QNetwork(envs).to(device)
    model.load_state_dict(torch.load(
        Path.cwd() / f"runs/{env_id}__dqn__{MODEL_PATH[env_id]}.cleanrl_model", map_location=device))
    model.eval()

    if mode == 1:
        return model
    return model, envs


def simulate_action_seq(env_id: str, action_seq: list[list[np.ndarray]],
                        device, seed: int = 0):
    seed_everything(seed, True)
    envs = load_model_envs(env_id, device, seed, 0)

    obs = envs.reset(seed=seed)[0]
    for action in action_seq:
        obs = envs.step(action)[0]
    return obs, envs


def count_action_switch(action_seq):
    action_switch_count = 0
    for i in range(1, len(action_seq)):
        if action_seq[i - 1].item() != action_seq[i].item():
            action_switch_count += 1

    return action_switch_count


def simulate_osf_play(env_id: str, device: torch.device,
                      delta: float, make_video: bool, seed: int = 0):
    seed_everything(seed, True)
    model, envs = load_model_envs(env_id, device, seed, 2)
    containers = defaultdict(list)
    containers["seed"].append(seed)
    done = False

    osf_envs = load_model_envs(env_id, device, seed, 0, make_video)
    osf_obs = osf_envs.reset(seed=seed)[0]
    osf_q_values = model(torch.tensor(
        osf_obs).to(device)).numpy(force=True)
    osf_actions = np.argmax(osf_q_values, 1)

    obs = envs.reset(seed=seed)[0]
    # Sanity check
    assert (obs - osf_obs).sum() == 0, "Wrong state"

    while not done:
        with torch.no_grad():
            q_values = model(torch.tensor(
                obs).to(device)).numpy(force=True)
            osf_q_values = model(torch.tensor(
                osf_obs).to(device)).numpy(force=True)

        actions = np.argmax(q_values, 1)

        # Switch OSF action when importance gap larger than delta
        importance_gap = (np.max(q_values, 1)[0] -
                          osf_q_values[0, osf_actions.item()]).item()
        if importance_gap > delta:
            osf_actions = np.argmax(osf_q_values, 1)
            obs, envs = simulate_action_seq(
                env_id, containers["action_seq"], device, seed
            )
            assert (obs - osf_obs).sum() == 0, "Wrong state"
            actions = osf_actions

        # Add information
        containers["action_seq"].append(osf_actions)

        if make_video:
            containers["img"].append(
                osf_envs.unwrapped.envs[0].render())

        # Step in the environment
        obs = envs.step(actions)[0]
        osf_obs, _, _, _, infos = osf_envs.step(osf_actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    containers["episodic_return"].append(
                        info["episode"]["r"].item())
                    containers["episodic_length"].append(
                        info["episode"]["l"].item())

                    # Only one env
                    done = True

    containers["action_switch"].append(count_action_switch(
        containers["action_seq"]))
    del containers["action_seq"]
    return containers


def simulate_play(env_id: str, epsilon,
                  device: torch.device, make_video: bool, seed: int = 0):
    seed_everything(seed, True)
    containers = defaultdict(list)
    containers["seed"].append(seed)
    model, envs = load_model_envs(env_id, device, seed, 2, make_video)

    obs = envs.reset(seed=seed)[0]
    done = False

    while not done:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample()
                                for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                q_values = model(torch.tensor(obs).to(device))
                actions = torch.argmax(q_values, dim=1).numpy(force=True)

        # Add information
        containers["action_seq"].append(actions)
        if make_video:
            containers["img"].append(envs.unwrapped.envs[0].render())

        obs, _, _, _, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    containers["episodic_return"].append(
                        info["episode"]["r"].item())
                    containers["episodic_length"].append(
                        info["episode"]["l"].item())
                    done = True

    containers["action_switch"].append(count_action_switch(
        containers["action_seq"]))
    del containers["action_seq"]
    return containers


if __name__ == "__main__":
    @dataclass
    class Args:
        env_id: str
        delta: float
        epsilon: float
        # Generated using random.org, 643, Min: 1, Max: 1000, 2024-09-10 05:12:07 UTC
        seed: int = 643
        num_ep: int = 30
        n_jobs: int = -1
        make_video: bool = False

    args = tyro.cli(Args)

    assert args.delta == 0 or args.epsilon == 0, "Either delta or epsilon has to be 0."
    assert args.delta >= 0 and args.epsilon >= 0, "Both have to be positive"

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")

    result_play = defaultdict(list)
    if args.delta != 0:
        output_generator = Parallel(n_jobs=-1)(delayed(simulate_osf_play)(
            args.env_id, DEVICE, args.delta, args.make_video,
            seed
        ) for seed in trange(args.seed, args.seed + args.num_ep))

    else:
        output_generator = Parallel(n_jobs=-1)(delayed(simulate_play)(
            args.env_id, args.epsilon, DEVICE, args.make_video,
            seed
        ) for seed in trange(args.seed, args.seed + args.num_ep))

    for result in output_generator:
        for key, value in result.items():
            result_play[key] += value

    if not args.make_video:
        save_folder = Path().cwd() / "runs/osf_play"
        if not save_folder.exists():
            save_folder.mkdir(parents=True)

        save_file = save_folder / f"device={DEVICE}__env_id={args.env_id}__seed={args.seed}"\
            f"__num_ep={args.num_ep}__delta={args.delta:.2f}__epsilon={args.epsilon:.2f}.txt"

        with save_file.open("w") as f:
            f.write(
                f"Device={DEVICE}, Env={args.env_id}, Num episodes: {args.num_ep}, "
                f"epsilon={args.epsilon:.2f}, delta={args.delta:.2f}\n")
            f.write("\n")

            for key, val in compute_results(result_play).items():
                if "seed" in key:
                    continue
                f.write(f"{key}: {val}\n")
            f.write("\n")

            keys = [key for key in result_play.keys()]
            f.write(",".join(keys) + "\n")
            for i in range(len(result_play["seed"])):
                f.write(",".join([str(result_play[key][i])
                        for key in keys]) + "\n")

    else:
        save_folder = Path().cwd() / "runs/osf_play_video"
        if not save_folder.exists():
            save_folder.mkdir(parents=True)

        imgs = np.stack(result_play["img"][0], 0)

        # Make video
        fps = 25
        out = cv2.VideoWriter(
            str(save_folder / f"env_id={args.env_id}__num_ep={args.num_ep}"
                f"__delta={args.delta}__epsilon={args.epsilon}__seed={args.seed}.mp4"),
            cv2.VideoWriter_fourcc(
                *'mp4v'), fps, (imgs.shape[2], imgs.shape[1]))

        for i in range(len(imgs)):
            out.write(cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR))

        out.release()
