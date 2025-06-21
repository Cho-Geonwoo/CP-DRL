import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from causal_world.task_generators import generate_task
from causal_world.envs.causalworld import CausalWorld
import random
import argparse


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# === Models ===
class TransitionPrediction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, state_dim)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class RewardPrediction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class BetaVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16, beta=4.0):
        super().__init__()
        self.fc_mu_logvar = nn.Linear(input_dim, latent_dim * 2)
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.beta = beta

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu_logvar = self.fc_mu_logvar(x)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


# === Task Variants ===
TASK_VARIANTS = {
    "total": {
        "A": {
            "tool_block_size": 0.05,
            "action_scale": 1.0,
            "fractional_reward_weight": 1.0,
        },
        "B": {
            "tool_block_size": 0.065,
            "action_scale": 2.0,
            "fractional_reward_weight": 2.0,
        },
        "C": {
            "tool_block_size": 0.08,
            "action_scale": 3.0,
            "fractional_reward_weight": 3.0,
        },
        "D": {
            "tool_block_size": 0.1,
            "action_scale": 4.0,
            "fractional_reward_weight": 4.0,
        },
    },
    "only_reward": {
        "A": {
            "tool_block_size": 0.065,
            "action_scale": 1.0,
            "fractional_reward_weight": 1.0,
        },
        "B": {
            "tool_block_size": 0.065,
            "action_scale": 1.0,
            "fractional_reward_weight": 2.0,
        },
        "C": {
            "tool_block_size": 0.065,
            "action_scale": 1.0,
            "fractional_reward_weight": 3.0,
        },
        "D": {
            "tool_block_size": 0.065,
            "action_scale": 1.0,
            "fractional_reward_weight": 4.0,
        },
    },
    "only_action": {
        "A": {
            "tool_block_size": 0.065,
            "action_scale": 1.0,
            "fractional_reward_weight": 1.0,
        },
        "B": {
            "tool_block_size": 0.065,
            "action_scale": 2.0,
            "fractional_reward_weight": 1.0,
        },
        "C": {
            "tool_block_size": 0.065,
            "action_scale": 3.0,
            "fractional_reward_weight": 1.0,
        },
        "D": {
            "tool_block_size": 0.065,
            "action_scale": 4.0,
            "fractional_reward_weight": 1.0,
        },
    },
    "only_block_size": {
        "A": {
            "tool_block_size": 0.05,
            "action_scale": 1.0,
            "fractional_reward_weight": 1.0,
        },
        "B": {
            "tool_block_size": 0.065,
            "action_scale": 1.0,
            "fractional_reward_weight": 1.0,
        },
        "C": {
            "tool_block_size": 0.08,
            "action_scale": 1.0,
            "fractional_reward_weight": 1.0,
        },
        "D": {
            "tool_block_size": 0.1,
            "action_scale": 1.0,
            "fractional_reward_weight": 1.0,
        },
    },
}


def make_env(config):
    task = generate_task(
        task_generator_id="general",
        variables_space="space_a_b",
        tool_block_size=config["tool_block_size"],
        fractional_reward_weight=config.get("fractional_reward_weight", 1.0),
        nums_objects=3,
    )
    return CausalWorld(
        task=task,
        skip_frame=3,
    )


# === Training Utilities ===
def train_models(env, episodes=10):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    hidden_dim = 64

    transition_models = [
        TransitionPrediction(obs_dim, act_dim, hidden_dim).cuda() for _ in range(5)
    ]
    reward_models = [
        RewardPrediction(obs_dim, act_dim, hidden_dim).cuda() for _ in range(5)
    ]
    state_models = [BetaVAE(obs_dim).cuda() for _ in range(5)]
    action_models = [BetaVAE(act_dim).cuda() for _ in range(5)]

    transition_opts = [optim.Adam(m.parameters(), lr=1e-3) for m in transition_models]
    reward_opts = [optim.Adam(m.parameters(), lr=1e-3) for m in reward_models]
    state_opts = [optim.Adam(m.parameters(), lr=1e-3) for m in state_models]
    action_opts = [optim.Adam(m.parameters(), lr=1e-3) for m in action_models]

    data = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            act = env.action_space.sample()
            next_obs, rew, done, _ = env.step(act)
            data.append((obs, act, next_obs, rew))
            obs = next_obs

    states = torch.tensor([d[0] for d in data], dtype=torch.float32).cuda()
    actions = torch.tensor([d[1] for d in data], dtype=torch.float32).cuda()
    next_states = torch.tensor([d[2] for d in data], dtype=torch.float32).cuda()
    rewards = (
        torch.tensor([d[3] for d in data], dtype=torch.float32).cuda().unsqueeze(-1)
    )

    for model, opt in zip(transition_models, transition_opts):
        for _ in range(5):
            pred = model(states, actions)
            loss = nn.MSELoss()(pred, next_states)
            opt.zero_grad()
            loss.backward()
            opt.step()

    for model, opt in zip(reward_models, reward_opts):
        for _ in range(5):
            pred = model(states, actions)
            loss = nn.MSELoss()(pred, rewards)
            opt.zero_grad()
            loss.backward()
            opt.step()

    for model, opt in zip(state_models, state_opts):
        for _ in range(5):
            recon, mu, logvar = model(states)
            recon_loss = nn.MSELoss()(recon, states)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + model.beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()

    for model, opt in zip(action_models, action_opts):
        for _ in range(5):
            recon, mu, logvar = model(actions)
            recon_loss = nn.MSELoss()(recon, actions)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + model.beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()

    return transition_models, reward_models, state_models, action_models


def evaluate_cm_score(
    env,
    transition_models,
    reward_models,
    state_models,
    action_models,
    episodes=5,
    action_scale=1.0,
):
    data = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            act = env.action_space.sample()
            act = act * action_scale  # action_Scale actions if needed
            next_obs, rew, done, _ = env.step(act)
            data.append((obs, act, next_obs, rew))
            obs = next_obs

    states = torch.tensor([d[0] for d in data], dtype=torch.float32).cuda()
    actions = torch.tensor([d[1] for d in data], dtype=torch.float32).cuda()
    next_states = torch.tensor([d[2] for d in data], dtype=torch.float32).cuda()
    rewards = (
        torch.tensor([d[3] for d in data], dtype=torch.float32).cuda().unsqueeze(-1)
    )

    transition_std = (
        torch.stack([m(states, actions) for m in transition_models])
        .std(dim=0)
        .mean()
        .item()
    )
    reward_std = (
        torch.stack([m(states, actions) for m in reward_models])
        .std(dim=0)
        .mean()
        .item()
    )
    state_std = (
        torch.stack([m(states)[0] for m in state_models]).std(dim=0).mean().item()
    )
    action_std = (
        torch.stack([m(actions)[0] for m in action_models]).std(dim=0).mean().item()
    )

    # return transition_std + reward_std + state_std + action_std
    return transition_std, reward_std, state_std, action_std


# === Experiment Runner ===
def run_cm_experiment(task_variant):
    base_env = make_env(TASK_VARIANTS[task_variant]["A"])
    t_models, r_models, s_models, a_models = train_models(base_env, episodes=10)

    scores = {}
    for task_id, cfg in TASK_VARIANTS[task_variant].items():
        # if task_id == "A":
        #     continue
        test_env = make_env(cfg)
        t_score, r_score, s_score, a_score = evaluate_cm_score(
            test_env,
            t_models,
            r_models,
            s_models,
            a_models,
            episodes=5,
            action_scale=cfg.get("action_scale", 1.0),
        )
        scores[task_id] = {
            "transition": t_score,
            "reward": r_score,
            "state": s_score,
            "action": a_score,
            "total": t_score + r_score + s_score + a_score,
        }
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Causal Misalignment Experiment")
    parser.add_argument(
        "--task_variant",
        type=str,
        choices=list(TASK_VARIANTS.keys()),
        default="total",
        help="Task variant to use for the experiment",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    set_seed(args.seed)
    print(f"Using task variant: {args.task_variant}")
    cm_scores = run_cm_experiment(args.task_variant)
    print("\nCausal Misalignment Scores:")
    for task, score in sorted(cm_scores.items()):
        print(
            f"Task {task}: Transition={score['transition']:.4f}, Reward={score['reward']:.4f}, "
            f"State={score['state']:.4f}, Action={score['action']:.4f}, Total={score['total']:.4f}"
        )
