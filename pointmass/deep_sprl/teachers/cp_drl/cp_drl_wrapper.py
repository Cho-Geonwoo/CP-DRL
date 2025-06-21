import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import config

from deep_sprl.teachers.util import Buffer
from deep_sprl.teachers.abstract_teacher import BaseWrapper
from .models import TransitionPrediction, BetaVAE, RewardPrediction


class StepBuffer:
    def __init__(self):
        self.states = []
        self.rewards = []
        self.dones = []
        self.infos = []
        self.actions = []

    def update_buffer(self, state, reward, done, info, action):
        self.states.append(state)
        self.rewards.append(reward)
        self.dones.append(done)
        self.infos.append(info)
        self.actions.append(action)

    def reset(self):
        self.states = []
        self.rewards = []
        self.dones = []
        self.infos = []
        self.actions = []

    def read_buffer(self):
        return self.states, self.rewards, self.dones, self.infos, self.actions


class CPDRLWrapper(BaseWrapper):
    def __init__(
        self,
        env,
        sp_teacher,
        discount_factor,
        context_visible,
        reward_from_info=False,
        use_undiscounted_reward=False,
        episodes_per_update=50,
    ):
        self.use_undiscounted_reward = use_undiscounted_reward
        BaseWrapper.__init__(
            self,
            env,
            sp_teacher,
            discount_factor,
            context_visible,
            reward_from_info=reward_from_info,
        )
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.context_buffer = Buffer(3, episodes_per_update + 1, True)
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
        self.episodes_per_update = episodes_per_update
        self.state_reconstruction_error = 0.0
        self.transition_reconstruction_error = 0.0
        self.action_reconstruction_error = 0.0
        self.reward_reconstruction_error = 0.0
        self.step_count = 0

    def step(self, action):
        step = self.env.step(action)
        if self.context_visible:
            modified_step = (
                np.concatenate((step[0], self.processed_context)),
                step[1],
                step[2],
                step[3],
            )  # (state, reward, done, info, action)
        self.update(step, action)
        self.step_count += 1
        return modified_step

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

    def update(self, step, action):
        reward = step[3]["reward"] if self.reward_from_info else step[1]
        self.undiscounted_reward += reward
        self.discounted_reward += self.cur_disc * reward
        self.cur_disc *= self.discount_factor
        self.step_length += 1.0
        self.step_buffer.update_buffer(step[0], step[1], step[2], reward, action)

        if step[2]:
            states, rewards, _, _, actions = self.step_buffer.read_buffer()
            states = torch.tensor(np.array(states), dtype=torch.float32).to(
                config.DEVICE
            )
            actions = torch.tensor(np.array(actions), dtype=torch.float32).to(
                config.DEVICE
            )
            rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(
                config.DEVICE
            )

            if len(states) > 1:
                self._calculate_state_reconstruction_error(states)
                self._calculate_transition_reconstruction_error(states, actions)
                self._calculate_action_reconstruction_error(actions)
                self._calculate_reward_reconstruction_error(states, actions, rewards)

                self.done_callback(
                    self.cur_initial_state.copy(),
                    self.cur_context,
                    self.state_reconstruction_error,
                    self.transition_reconstruction_error,
                    self.action_reconstruction_error,
                    self.reward_reconstruction_error,
                    self.discounted_reward,
                    self.undiscounted_reward,
                )

            self.stats_buffer.update_buffer(
                (self.undiscounted_reward, self.discounted_reward, self.step_length)
            )
            self.context_trace_buffer.update_buffer(
                (
                    self.undiscounted_reward,
                    self.discounted_reward,
                    self.processed_context.copy(),
                )
            )
            self.undiscounted_reward = 0.0
            self.discounted_reward = 0.0
            self.state_reconstruction_error = 0.0
            self.transition_reconstruction_error = 0.0
            self.action_reconstruction_error = 0.0
            self.reward_reconstruction_error = 0.0
            self.cur_disc = 1.0
            self.step_length = 0.0
            self.step_buffer.reset()

            self.cur_context = None
            self.processed_context = None
            self.cur_initial_state = None

    def done_callback(
        self,
        cur_initial_state,
        cur_context,
        state_reconstruction_error,
        transition_reconstruction_error,
        action_reconstruction_error,
        reward_reconstruction_error,
        discounted_reward,
        undiscounted_reward,
    ):
        reward = (
            undiscounted_reward if self.use_undiscounted_reward else discounted_reward
        )
        ret = (
            reward
            + state_reconstruction_error
            + action_reconstruction_error
            + reward_reconstruction_error
            + transition_reconstruction_error
        )
        self.context_buffer.update_buffer((cur_initial_state, cur_context, ret))

        if hasattr(self.teacher, "on_rollout_end"):
            self.teacher.on_rollout_end(cur_context, ret)

        if len(self.context_buffer) >= self.episodes_per_update:
            __, contexts, returns = self.context_buffer.read_buffer()
            self.teacher.update_distribution(np.array(contexts), np.array(returns))
