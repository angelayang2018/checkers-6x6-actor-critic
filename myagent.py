import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as D

# =========================
# Action Encoding
# =========================

BOARD_SIZE = 6
ACTION_SIZE = BOARD_SIZE ** 4  # 6*6*6*6 = 1296

def encode_action(action):
    fr, fc, tr, tc = action
    return fr * 216 + fc * 36 + tr * 6 + tc

def decode_action(idx):
    fr = idx // 216
    fc = (idx % 216) // 36
    tr = (idx % 36) // 6
    tc = idx % 6
    return (fr, fc, tr, tc)


# =========================
# Actor-Critic Network
# =========================

class ActorCritic(nn.Module):
    def __init__(self, action_size=ACTION_SIZE):
        super().__init__()

        self.fc1 = nn.Linear(36, 128)
        self.fc2 = nn.Linear(128, 128)

        # Actor
        self.policy_head = nn.Linear(128, action_size)

        # Critic
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(-1, 36)  # flatten board

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value


# =========================
# Agent Wrapper
# =========================

class ACAgent:
    def __init__(self, lr=1e-3, gamma=0.99):
        self.gamma = gamma

        self.model = ActorCritic(ACTION_SIZE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    # ---------------------
    # Action Selection
    # ---------------------
    def act(self, state, legal_moves):
        state = torch.FloatTensor(state).unsqueeze(0)

        logits, value = self.model(state)
        probs = torch.softmax(logits, dim=-1).squeeze()

        # Mask illegal actions
        mask = torch.zeros_like(probs)

        for move in legal_moves:
            idx = encode_action(move)
            mask[idx] = 1

        # Apply mask
        probs = probs * mask

        # Avoid division by zero
        if probs.sum() == 0:
            probs = mask / mask.sum()
        else:
            probs = probs / probs.sum()

        dist = D.Categorical(probs)
        action_idx = dist.sample()

        action = decode_action(action_idx.item())
        log_prob = dist.log_prob(action_idx)

        return action, log_prob, value

    # ---------------------
    # Learning Update
    # ---------------------
    def update(self, log_probs, values, rewards):
        returns = []
        G = 0

        # Compute discounted returns
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)
        values = torch.cat(values).squeeze()

        # Advantage
        advantage = returns - values.detach()

        # Losses
        policy_loss = -(torch.stack(log_probs) * advantage).mean()
        value_loss = F.mse_loss(values, returns)

        loss = policy_loss + value_loss

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()