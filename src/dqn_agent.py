# src/dqn_agent.py
import random
from collections import deque, namedtuple
import numpy as np

# Try import torch and provide a clear error if missing
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:
    raise ImportError("PyTorch is required to run the DQN agent. "
                      "Install with: pip install torch torchvision") from e

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    def __init__(self, capacity:int = 100000): # <-- FIXED (was _init_)
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size:int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self): # <-- FIXED (was _len_)
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, input_dim:int, hidden_dims=(256, 256), output_dim:int=26): # <-- FIXED (was _init_)
        super().__init__() # <-- FIXED (was _init_)
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__( # <-- FIXED (was _init_)
        self,
        input_dim:int,
        action_dim:int=26,
        lr:float=1e-3,
        gamma:float=0.99,
        epsilon_start:float=1.0,
        epsilon_final:float=0.05,
        epsilon_decay_steps:int=20000,
        buffer_capacity:int=200000,
        batch_size:int=64,
        target_update_freq:int=1000,
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.model = DQN(input_dim, output_dim=action_dim).to(self.device)
        self.target = DQN(input_dim, output_dim=action_dim).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma

        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay_steps = epsilon_decay_steps
        self.frame_idx = 0

        self.replay = ReplayBuffer(capacity=buffer_capacity)
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.loss_fn = nn.MSELoss()

    def act(self, state_np: np.ndarray, legal_mask: np.ndarray=None):
        """
        state_np: numpy array (input vector)
        legal_mask: optional boolean mask of size action_dim (True where allowed)
        """
        self.frame_idx += 1
        eps = self._epsilon()
        if random.random() < eps:
            # random legal action
            if legal_mask is None:
                return random.randrange(self.action_dim)
            else:
                legal_idxs = np.flatnonzero(legal_mask)
                if len(legal_idxs) == 0:
                    return random.randrange(self.action_dim) # Fallback
                return int(np.random.choice(legal_idxs))
        else:
            # greedy
            self.model.eval()
            with torch.no_grad():
                x = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                q = self.model(x).cpu().numpy().squeeze(0)  # (action_dim,)
                if legal_mask is not None:
                    # set illegal actions to -inf so argmax ignores them
                    masked_q = np.where(legal_mask, q, -1e9)
                else:
                    masked_q = q
                return int(np.argmax(masked_q))

    def _epsilon(self):
        # linear decay
        frac = min(1.0, float(self.frame_idx) / float(self.epsilon_decay_steps))
        return self.epsilon_start + frac * (self.epsilon_final - self.epsilon_start)

    def store(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay) < self.batch_size:
            return None

        batch = self.replay.sample(self.batch_size)
        state = torch.tensor(np.stack(batch.state), dtype=torch.float32, device=self.device)
        action = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state = torch.tensor(np.stack(batch.next_state), dtype=torch.float32, device=self.device)
        done = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a)
        q_values = self.model(state).gather(1, action)

        # target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q = self.target(next_state)
            max_next_q = next_q.max(1)[0].unsqueeze(1)
            target_q = reward + (1.0 - done) * self.gamma * max_next_q

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

        # update target network
        if self.frame_idx % self.target_update_freq == 0:
            self.target.load_state_dict(self.model.state_dict())

        return loss.item()

    def save(self, path):
        torch.save({
            "model_state": self.model.state_dict(),
            "target_state": self.target.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.model.load_state_dict(data["model_state"])
        self.target.load_state_dict(data["target_state"])
        if "optimizer_state" in data:
            self.optimizer.load_state_dict(data["optimizer_state"])