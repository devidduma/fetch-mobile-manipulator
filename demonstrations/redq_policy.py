from tianshou.policy import REDQPolicy
from tianshou.utils.net.common import EnsembleLinear, Net, get_dict_state_decorator
from tianshou.utils.net.continuous import ActorProb, Critic
import torch
import numpy as np

def get_redq_policy(env, args):
    args.state_shape = {
        'observation': env.observation_space['observation'].shape,
        'achieved_goal': env.observation_space['achieved_goal'].shape,
        'desired_goal': env.observation_space['desired_goal'].shape,
    }
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    args.exploration_noise = args.exploration_noise * args.max_action
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

    def linear(x, y):
        return EnsembleLinear(args.ensemble_size, x, y)

    # model
    dict_state_dec, flat_state_shape = get_dict_state_decorator(
        state_shape=args.state_shape,
        keys=['observation', 'achieved_goal', 'desired_goal']
    )
    net_a = dict_state_dec(Net)(
        flat_state_shape, hidden_sizes=args.hidden_sizes, device=args.device
    )
    actor = dict_state_dec(ActorProb)(
        net_a, args.action_shape, max_action=args.max_action, device=args.device,
        unbounded=True,
        conditioned_sigma=True,
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    net_c = dict_state_dec(Net)(
        flat_state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
        linear_layer=linear,
    )
    critics = dict_state_dec(Critic)(
        net_c,
        device=args.device,
        linear_layer=linear,
        flatten_input=False,
    ).to(args.device)
    critics_optim = torch.optim.Adam(critics.parameters(), lr=args.critic_lr)

    policy = REDQPolicy(
        actor,
        actor_optim,
        critics,
        critics_optim,
        args.ensemble_size,
        args.subset_size,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        estimation_step=args.n_step,
        actor_delay=args.update_per_step,
        target_mode=args.target_mode,
        action_space=env.action_space,
    )

    return policy