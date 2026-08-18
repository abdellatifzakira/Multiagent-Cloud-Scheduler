# RL/Buffer.py

import numpy as np
import torch


class Buffer:
    def __init__(self, capacity, obs_dim, act_dim, device="cpu"):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.device = torch.device(device)

        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.action = np.full(
            (self.capacity, self.act_dim),
            -1,
            dtype=np.int64,
        )
        self.reward = np.zeros(
            (self.capacity, self.act_dim),
            dtype=np.float32,
        )
        self.next_obs = np.zeros(
            (self.capacity, self.obs_dim),
            dtype=np.float32,
        )
        self.next_active = np.zeros(
            self.capacity,
            dtype=np.int64,
        )
        self.done = np.zeros(
            self.capacity,
            dtype=np.bool_,
        )

        self._index = 0
        self._size = 0

    def add(
        self,
        obs,
        action,
        reward,
        next_obs,
        done,
        next_active=None,
    ):
        self.obs[self._index] = np.asarray(
            obs,
            dtype=np.float32,
        )

        self.action[self._index] = np.asarray(
            action,
            dtype=np.int64,
        )

        self.reward[self._index] = np.asarray(
            reward,
            dtype=np.float32,
        )

        self.next_obs[self._index] = np.asarray(
            next_obs,
            dtype=np.float32,
        )

        if next_active is None:
            next_active = self.act_dim

        self.next_active[self._index] = int(
            np.clip(next_active, 0, self.act_dim)
        )

        self.done[self._index] = bool(done)

        self._index = (
            self._index + 1
        ) % self.capacity

        self._size = min(
            self._size + 1,
            self.capacity,
        )

    def sample(self, indices):
        indices = np.asarray(
            indices,
            dtype=np.int64,
        )

        obs = torch.as_tensor(
            self.obs[indices],
            dtype=torch.float32,
            device=self.device,
        )

        action = torch.as_tensor(
            self.action[indices],
            dtype=torch.long,
            device=self.device,
        )

        reward = torch.as_tensor(
            self.reward[indices],
            dtype=torch.float32,
            device=self.device,
        )

        next_obs = torch.as_tensor(
            self.next_obs[indices],
            dtype=torch.float32,
            device=self.device,
        )

        next_active = torch.as_tensor(
            self.next_active[indices],
            dtype=torch.long,
            device=self.device,
        )

        done = torch.as_tensor(
            self.done[indices],
            dtype=torch.float32,
            device=self.device,
        )

        return (
            obs,
            action,
            reward,
            next_obs,
            done,
            next_active,
        )

    def __len__(self):
        return self._size