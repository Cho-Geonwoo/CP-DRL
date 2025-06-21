import torch
import torch.optim as optim
import torch.nn as nn
import copy
import numpy as np

from TeachMyAgent.teachers.algos.AbstractTeacher import AbstractTeacher
from currot.deep_sprl.teachers.spl.currot import CurrOT
from TeachMyAgent.models import (
    TransitionPrediction,
    BetaVAE,
    RewardPrediction,
)
from TeachMyAgent import config


class StepBuffer:
    def __init__(self):
        self.states = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        self.actions = []

    def update_buffer(self, state, reward, done, next_state, action):
        self.states.append(state)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)
        self.actions.append(action)

    def reset(self):
        self.states = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        self.actions = []

    def read_buffer(self):
        return self.states, self.rewards, self.dones, self.next_states, self.actions


class CPDRLTeacher(AbstractTeacher):
    def __init__(
        self,
        context_lb,
        context_ub,
        seed,
        env_reward_lb,
        env_reward_ub,
        state_dim,
        action_dim,
        perf_lb=180,
        n_samples=500,
        episodes_per_update=50,
        epsilon=None,
        callback=None,
    ):

        super().__init__(
            context_lb, context_ub, env_reward_lb, env_reward_ub, seed=seed
        )

        if epsilon is None:
            epsilon = 0.05 * np.linalg.norm(np.array(context_ub) - np.array(context_lb))

        if perf_lb is None:
            perf_lb = 0.5 * (env_reward_ub - env_reward_lb) + env_reward_lb

        if episodes_per_update is None:
            episodes_per_update = 0.25 * n_samples
        self.episodes_per_update = episodes_per_update

        # Create an array if we use the same number of bins per dimension
        target_sampler = lambda n: np.random.uniform(
            context_lb, context_ub, size=(n, len(context_lb))
        )
        init_samples = np.random.uniform(
            context_lb, context_ub, size=(n_samples, len(context_lb))
        )

        self.curriculum = CurrOT(
            (np.array(context_lb), np.array(context_ub)),
            init_samples,
            target_sampler,
            perf_lb,
            epsilon,
            wait_until_threshold=False,
        )

        self.context_buffer = []
        self.return_buffer = []
        self.bk = {"teacher_snapshots": []}

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.step_buffer = StepBuffer()
        self.transition_prediction_models = nn.ModuleList(
            [
                TransitionPrediction(
                    state_dim, action_dim, config.CPDRL.TRANSITION_HIDDEN_DIM
                ).to(config.DEVICE)
                for _ in range(config.CPDRL.ENSEMBLE_SIZE)
            ]
        )
        self.state_reconstruction_models = nn.ModuleList(
            [
                BetaVAE(state_dim, config.CPDRL.STATE_VAE_LATENT_DIM, beta=4.0).to(
                    config.DEVICE
                )
                for _ in range(config.CPDRL.ENSEMBLE_SIZE)
            ]
        )
        self.action_reconstruction_models = nn.ModuleList(
            [
                BetaVAE(action_dim, config.CPDRL.ACTION_VAE_LATENT_DIM, beta=4.0).to(
                    config.DEVICE
                )
                for _ in range(config.CPDRL.ENSEMBLE_SIZE)
            ]
        )
        self.reward_prediction_models = nn.ModuleList(
            [
                RewardPrediction(
                    state_dim, action_dim, config.CPDRL.REWARD_HIDDEN_DIM
                ).to(config.DEVICE)
                for _ in range(config.CPDRL.ENSEMBLE_SIZE)
            ]
        )

        self.transition_prediction_optimizers = [
            optim.Adam(model.parameters(), lr=config.CPDRL.LEARNING_RATE)
            for model in self.transition_prediction_models
        ]
        self.reward_prediction_optimizers = [
            optim.Adam(model.parameters(), lr=config.CPDRL.LEARNING_RATE)
            for model in self.reward_prediction_models
        ]
        self.action_reconstruction_optimizers = [
            optim.Adam(model.parameters(), lr=config.CPDRL.LEARNING_RATE)
            for model in self.action_reconstruction_models
        ]
        self.state_reconstruction_optimizers = [
            optim.Adam(model.parameters(), lr=config.CPDRL.LEARNING_RATE)
            for model in self.state_reconstruction_models
        ]

        self.state_reconstruction_error = 0.0
        self.transition_reconstruction_error = 0.0
        self.action_reconstruction_error = 0.0
        self.reward_reconstruction_error = 0.0

    def calculate_errors(self):
        states, rewards, dones, next_states, actions = self.step_buffer.read_buffer()
        if len(states) == 0:
            return
        self.step_buffer.reset()

        states = torch.tensor(states, dtype=torch.float32, device=config.DEVICE)
        next_states = torch.tensor(
            next_states, dtype=torch.float32, device=config.DEVICE
        )
        actions = torch.tensor(actions, dtype=torch.float32, device=config.DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=config.DEVICE)

        state_reconstruction_error = self._calculate_state_reconstruction_error(states)
        transition_reconstruction_error = (
            self._calculate_transition_reconstruction_error(states, actions)
        )
        action_reconstruction_error = self._calculate_action_reconstruction_error(
            actions
        )
        reward_reconstruction_error = self._calculate_reward_reconstruction_error(
            states, actions, rewards
        )

        return (
            state_reconstruction_error,
            transition_reconstruction_error,
            action_reconstruction_error,
            reward_reconstruction_error,
        )

    def episodic_update(self, task, reward, is_success):
        (
            state_reconstruction_error,
            transition_reconstruction_error,
            action_reconstruction_error,
            reward_reconstruction_error,
        ) = self.calculate_errors()

        reward = (
            reward
            + state_reconstruction_error
            + transition_reconstruction_error
            + action_reconstruction_error
            + reward_reconstruction_error
        )

        # self.sampler.update(self.sample_idx, reward)
        self.curriculum.on_rollout_end(task, reward)
        self.context_buffer.append(task)
        self.return_buffer.append(reward)
        # print("Updated task %d" % self.sample_idx)

        if len(self.context_buffer) >= self.episodes_per_update:
            new_snapshot = {
                "context_buffer": copy.deepcopy(self.context_buffer),
                "return_buffer": copy.deepcopy(self.return_buffer),
            }
            contexts = np.array(self.context_buffer)
            returns = np.array(self.return_buffer)
            self.context_buffer.clear()
            self.return_buffer.clear()
            self.curriculum.update_distribution(contexts, returns)
            new_snapshot["current_samples"] = copy.deepcopy(
                self.curriculum.teacher.current_samples
            )
            new_snapshot["success_buffer"] = copy.deepcopy(
                self.curriculum.success_buffer.contexts
            )
            self.bk["teacher_snapshots"].append(new_snapshot)

    def _calculate_state_reconstruction_error(self, states):
        ensemble_predictions = []

        for model, optimizer in zip(
            self.state_reconstruction_models, self.state_reconstruction_optimizers
        ):
            pred, mu, logvar = model(states)
            ensemble_predictions.append(pred)
            loss = torch.nn.functional.mse_loss(pred, states)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss += config.CPDRL.ALPHA * kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        disagreement = torch.stack(ensemble_predictions).std(dim=0).mean()

        if config.CPDRL.ALIGNED:
            self.state_reconstruction_error = (
                config.CPDRL.STATE_DISAGREEMENT_SCALE / disagreement.item()
            )
        else:
            self.state_reconstruction_error = (
                config.CPDRL.STATE_DISAGREEMENT_SCALE * disagreement.item()
            )

    def _calculate_transition_reconstruction_error(self, states, actions):
        current_states = states[:-1]
        next_states = states[1:]
        actions = actions[:-1]

        ensemble_predictions = []

        for model, optimizer in zip(
            self.transition_prediction_models, self.transition_prediction_optimizers
        ):
            pred = model(current_states, actions)
            ensemble_predictions.append(pred)
            loss = torch.nn.functional.mse_loss(pred, next_states)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        disagreement = torch.stack(ensemble_predictions).std(dim=0).mean()

        if config.CPDRL.ALIGNED:
            self.transition_reconstruction_error = (
                config.CPDRL.TRANSITION_DISAGREEMENT_SCALE / disagreement.item()
            )
        else:
            self.transition_reconstruction_error = (
                config.CPDRL.TRANSITION_DISAGREEMENT_SCALE * disagreement.item()
            )

    def _calculate_action_reconstruction_error(self, actions):
        ensemble_predictions = []

        for model, optimizer in zip(
            self.action_reconstruction_models, self.action_reconstruction_optimizers
        ):
            pred, mu, logvar = model(actions)
            ensemble_predictions.append(pred)
            loss = torch.nn.functional.mse_loss(pred, actions)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss += config.CPDRL.ALPHA * kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        disagreement = torch.stack(ensemble_predictions).std(dim=0).mean()

        if config.CPDRL.ALIGNED:
            self.action_reconstruction_error = (
                config.CPDRL.ACTION_DISAGREEMENT_SCALE / disagreement.item()
            )
        else:
            self.action_reconstruction_error = (
                config.CPDRL.ACTION_DISAGREEMENT_SCALE * disagreement.item()
            )

    def _calculate_reward_reconstruction_error(self, states, actions, rewards):
        states = states[:-1]
        actions = actions[:-1]
        rewards = rewards[1:]

        ensemble_predictions = []

        for model, optimizer in zip(
            self.reward_prediction_models, self.reward_prediction_optimizers
        ):
            pred = model(states, actions)
            ensemble_predictions.append(pred)
            loss = torch.nn.functional.mse_loss(pred.squeeze(-1), rewards)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        disagreement = torch.stack(ensemble_predictions).std(dim=0).mean()

        if config.CPDRL.ALIGNED:
            self.reward_reconstruction_error = (
                config.CPDRL.REWARD_DISAGREEMENT_SCALE / disagreement.item()
            )
        else:
            self.reward_reconstruction_error = (
                config.CPDRL.REWARD_DISAGREEMENT_SCALE * disagreement.item()
            )

    def step_update(self, state, action, reward, next_state, done):
        self.step_buffer.update_buffer(state, reward, done, next_state, action)

    def sample_task(self):
        return self.curriculum.sample().astype(np.float32)

    def is_non_exploratory_task_sampling_available(self):
        return True

    def non_exploratory_task_sampling(self):
        task_idx = np.random.randint(
            0, self.curriculum.teacher.current_samples.shape[0]
        )
        task = np.clip(
            self.curriculum.teacher.current_samples[task_idx, :],
            self.curriculum.context_bounds[0],
            self.curriculum.context_bounds[1],
        ).astype(np.float32)
        return {"task": task, "infos": None}

    def save(self, path):
        self.curriculum.save(path)

    def load(self, path):
        self.curriculum.load(path)
