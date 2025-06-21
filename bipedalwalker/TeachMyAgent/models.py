import torch
import torch.nn as nn


class TransitionPrediction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(TransitionPrediction, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, state_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        next_state_pred = self.fc2(x)
        return next_state_pred


class RewardPrediction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(RewardPrediction, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        reward_pred = self.fc2(x)
        return reward_pred


class BetaVAE(nn.Module):
    def __init__(self, state_dim, latent_dim, beta=4.0):
        super(BetaVAE, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.state_dim = state_dim

        self.encoder = nn.Linear(state_dim, latent_dim * 2)
        self.decoder = nn.Linear(latent_dim, state_dim)

    def forward(self, x):
        mu, logvar = torch.chunk(self.encoder(x), 2, dim=-1)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
