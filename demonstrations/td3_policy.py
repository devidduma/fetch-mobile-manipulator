from tianshou.exploration import GaussianNoise
from tianshou.policy import TD3Policy
from tianshou.utils.net.common import Net, get_dict_state_decorator
from tianshou.utils.net.continuous import Actor, Critic
import torch
import numpy as np

def get_td3_policy(env, args):

    args.state_shape = {
        'observation': env.observation_space['observation'].shape,
        'achieved_goal': env.observation_space['achieved_goal'].shape,
        'desired_goal': env.observation_space['desired_goal'].shape,
    }
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    args.exploration_noise = args.exploration_noise * args.max_action
    args.policy_noise = args.policy_noise * args.max_action
    args.noise_clip = args.noise_clip * args.max_action
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

    # model
    dict_state_dec, flat_state_shape = get_dict_state_decorator(
        state_shape=args.state_shape,
        keys=['observation', 'achieved_goal', 'desired_goal']
    )
    net_a = dict_state_dec(Net)(
        flat_state_shape, hidden_sizes=args.hidden_sizes, device=args.device
    )
    actor = dict_state_dec(Actor)(
        net_a, args.action_shape, max_action=args.max_action, device=args.device
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = dict_state_dec(Net)(
        flat_state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    net_c2 = dict_state_dec(Net)(
        flat_state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic1 = dict_state_dec(Critic)(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = dict_state_dec(Critic)(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    policy = TD3Policy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        policy_noise=args.policy_noise,
        update_actor_freq=args.update_actor_freq,
        noise_clip=args.noise_clip,
        estimation_step=args.n_step,
        action_space=env.action_space,
    )

    return policy
