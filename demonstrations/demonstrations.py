import gymnasium as gym
import torch
import numpy as np
import argparse
from tianshou.data import Batch
import os
from gymnasium.wrappers.record_video import RecordVideo
from ddpg_policy import get_ddpg_policy
from td3_policy import get_td3_policy
from sac_policy import get_sac_policy
from redq_policy import get_redq_policy

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="FetchReach-v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--ensemble-size", type=int, default=10)
    parser.add_argument("--subset-size", type=int, default=2)
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=3e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=False, action="store_true")
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--update-actor-freq", type=int, default=2)
    parser.add_argument("--start-timesteps", type=int, default=25000)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--step-per-epoch", type=int, default=5000)
    parser.add_argument("--step-per-collect", type=int, default=1)
    parser.add_argument("--update-per-step", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--target-mode", type=str, choices=("min", "mean"), default="min"
    )
    parser.add_argument(
        "--replay-buffer", type=str, default="her", choices=["normal", "her"]
    )
    parser.add_argument("--her-horizon", type=int, default=50)
    parser.add_argument("--her-future-k", type=int, default=8)
    parser.add_argument("--training-num", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="HER-benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    parser.add_argument("--algo-label", type=str, default="")
    return parser.parse_args()

def load_policy(policy_type, env, args):
    print(args.task, args.hidden_sizes)

    # model
    if policy_type == "ddpg":
        policy = get_ddpg_policy(env, args)
    elif policy_type == "td3":
        policy = get_td3_policy(env, args)
    elif policy_type == "sac":
        policy = get_sac_policy(env, args)
    elif policy_type == "redq":
        policy = get_redq_policy(env, args)
    else:
        raise Exception("Unknown policy.")

    return policy

def simulate(task, policy_type, policy_path, hidden_sizes, args=get_args()):
    args.task = task
    args.hidden_sizes = hidden_sizes

    video_name_prefix = policy_type.upper() + "_" + task
    video_folder = os.path.join("", task, policy_type)

    env = RecordVideo(
        env=gym.make(task, render_mode="rgb_array"),
        video_folder=video_folder,
        name_prefix=video_name_prefix,
        video_length=40000
    )
    observation, info = env.reset()

    policy = load_policy(policy_type=policy_type, env=env, args=args)
    policy.load_state_dict(torch.load(policy_path, map_location=args.device))
    print("Loaded agent from: ", policy_path)

    reward = -1     # initialize

    for step_index in range(1000):

        batch = Batch(obs=[observation], info=info)  # the first dimension is batch-size
        action = policy.forward(batch=batch, state=observation).act[0].detach().numpy()  # policy.forward return a batch, use ".act" to extract the action

        if reward == -1:
            observation, reward, terminated, truncated, info = env.step(action)
        else:
            observation, reward, terminated, truncated, info = env.step([0,0,0,-1])    # stay in place

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

if __name__ == '__main__':

    hidden_sizes = {
        "FetchReach-v3" : [256, 256],
        "FetchPush-v2" : [256, 256, 256],
        "FetchPickAndPlace-v2" : [256, 256, 256]
        }

    tasks = [
        "FetchReach-v3",
        "FetchPush-v2",
        "FetchPickAndPlace-v2"
        ]

    tasks_policies = {
        "FetchReach-v3" : {"ddpg", "td3", "sac", "redq"},
        "FetchPush-v2" : {"td3", "redq"},
        "FetchPickAndPlace-v2" : {"redq"}
        }

    # save all simulations
    for task in tasks_policies:
        for policy_type in tasks_policies[task]:

            log_dir = os.path.join("../examples", "benchmark", "log")
            seed = "0"
            partial_path = os.path.join(log_dir, task, policy_type, seed)
            dir = os.listdir(partial_path)[0]   # there is only one directory
            full_path = os.path.join(partial_path, dir, "policy.pth")

            simulate(
                task=task,
                policy_type=policy_type,
                policy_path=full_path,
                hidden_sizes=hidden_sizes[task]
            )
