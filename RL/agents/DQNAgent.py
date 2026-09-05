# RL/agents/DQNAgent.py

import copy
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from Agent import Agent
except ModuleNotFoundError:
    from RL.agents.Agent import Agent

try:
    from Buffer import Buffer
except ModuleNotFoundError:
    from RL.Buffer import Buffer


class StandardQNetwork(nn.Module):
    """
    Feedforward multi-task Q-network that directly consumes the complete
    processed scheduling state and outputs (num_tasks x num_servers) Q-values.
    """

    def __init__(
        self,
        state_size,
        num_servers,
        num_tasks,
        hidden_dim=128,
        layer_config=None,
    ):
        super().__init__()

        self.num_servers = num_servers
        self.num_tasks = num_tasks

        activation_map = {
            "relu": nn.ReLU,
            "leakyrelu": nn.LeakyReLU,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
        }

        if layer_config is None:
            layer_config = [(128, "relu"), (256, "relu"), (128, "relu")]

        modules = []
        input_dim = state_size

        for layer_size, activation_name in layer_config:
            act_cls = activation_map.get(activation_name.lower())
            if act_cls is None:
                raise ValueError(f"Unsupported activation: {activation_name}")
            modules.append(nn.Linear(input_dim, layer_size))
            modules.append(act_cls())
            input_dim = layer_size

        self.feature_extractor = nn.Sequential(*modules)

        # Single-pass projection head outputting Q-values for all tasks.
        self.q_head = nn.Linear(input_dim, num_tasks * num_servers)

    def forward(self, state, actions=None):
        batch_size = state.size(0)

        # Directly consume the complete processed state.
        features = self.feature_extractor(state)

        # Output shape: [batch_size, num_tasks, num_servers]
        q_values = self.q_head(features).view(
            batch_size,
            self.num_tasks,
            self.num_servers,
        )

        if actions is not None:
            chosen_actions = actions.clamp(
                min=0,
                max=self.num_servers - 1,
            )
        else:
            chosen_actions = torch.argmax(q_values, dim=-1)

        return q_values, chosen_actions


class DQNNetwork(nn.Module):
    """
    Consolidated single-step Deep Q-Network.

    The complete processed scheduling state is passed directly to the
    decision network. No server encoder or task encoder is used.
    """

    def __init__(self, decision_network):
        super().__init__()
        self.decision_network = decision_network

    def forward(self, state, actions=None):
        return self.decision_network(
            state,
            actions=actions,
        )


class DQNAgent(Agent):
    def __init__(
        self,
        epsilon=1.0,
        gamma=0.99,
        min_epsilon=0.05,
        epsilon_decay=0.995,
        optimizer="adam",
        layers_config=None,
        decoder_hidden_dim=128,
        learning_rate=1e-3,
        replay_batch_size=64,
        replay_start_size=256,
        train_every=10,
        gradient_steps=4,
        target_update_every=250,
        buffer_capacity=1000,
        model_path = None,
        resume_training = False,
        seed=123,
    ):
        super().__init__(
            "DQN",
            "Single-step Standard Deep Q-Network Agent",
        )

        if layers_config is None:
            layers_config = [
                (128, "relu"),
                (256, "relu"),
                (128, "relu"),
            ]

        self.state = None
        self.active_tasks = 0
        self.resume_training = resume_training

        self.decision_network = None
        self.network = None
        self.target_network = None

        self.servers_state_size = None
        self.tasks_state_size = None
        self.state_size = None

        self.task_token_dim = 3
        self.model_path = model_path

        self.optimizer_name = optimizer
        self.buffer_capacity = buffer_capacity

        self.epsilon = float(epsilon)
        self.min_epsilon = float(min_epsilon)
        self.epsilon_decay = float(epsilon_decay)

        self.gamma = float(gamma)

        self.layers_config = layers_config
        self.decoder_hidden_dim = decoder_hidden_dim

        self.batch_size = None
        self.num_servers = None

        self.rng = random.Random(seed)
        self.seed = seed

        self.buffer = None

        self.replay_batch_size = replay_batch_size
        self.replay_start_size = replay_start_size
        self.train_every = train_every
        self.gradient_steps = gradient_steps
        self.target_update_every = target_update_every

        self.learning_rate = learning_rate
        self.weight_decay = 0.0

        self.step_count = 0
        self.gradient_step_count = 0

        self.last_loss = None
        self.last_q_mean = None
        self.last_target_mean = None

        self.action_counts = None

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._warned_overflow = False
        self.built = False
        self.loaded = False

    def build(self):
        if self.action_space is None:
            raise RuntimeError("Action space not provided by environment")

        if self.batch_size is None:
            raise RuntimeError("Batch size not provided by environment")

        if self.num_servers is None:
            raise RuntimeError("Number of servers not provided by environment")

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.action_space.seed(self.seed)

        self.servers_state_size = 6 * self.num_servers
        self.tasks_state_size = self.task_token_dim * self.batch_size
        self.state_size = self.servers_state_size + self.tasks_state_size

        self.decision_network = StandardQNetwork(
            state_size=self.state_size,
            num_servers=self.num_servers,
            num_tasks=self.batch_size,
            hidden_dim=self.decoder_hidden_dim,
            layer_config=self.layers_config,
        )

        self.network = DQNNetwork(
            self.decision_network,
        ).to(self.device)

        self.target_network = copy.deepcopy(self.network).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

        for parameter in self.target_network.parameters():
            parameter.requires_grad_(False)

        self.buffer = Buffer(
            capacity=self.buffer_capacity,
            obs_dim=self.state_size,
            act_dim=self.batch_size,
            device=self.device,
        )

        optimizer_map = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
        }

        optimizer_class = optimizer_map.get(self.optimizer_name.lower())
        if optimizer_class is None:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        self.optimizer = optimizer_class(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.action_counts = np.zeros(self.num_servers, dtype=np.int64)
        self.network.train()
        self.built = True

    def process(self, state):
        """Preserves identical state transformation, extraction, and z-score scaling."""
        raw_server_features = []
        server_ids = list(state["C_CPU"].keys())

        for server_id in server_ids:
            cpu_capacity = max(float(state["C_CPU"][server_id]), 1e-6)
            cpu_history = np.asarray(state["CPU"][server_id], dtype=np.float32)

            if cpu_history.size == 0:
                current_cpu = 0.0
                max_cpu = 0.0
                mean_cpu = 0.0
            else:
                current_cpu = float(cpu_history[-1])
                max_cpu = float(np.max(cpu_history))
                mean_cpu = float(np.mean(cpu_history))

            available_cpu = (
                float(state["MAX_AVAILABLE_CPU"][server_id]) / cpu_capacity
            )
            queue_pressure = (
                float(state["QUEUE"][server_id]) / cpu_capacity
            )
            effective_compute = float(
                state["EFFECTIVE_COMPUTE"][server_id]
            )

            raw_server_features.append(
                [
                    current_cpu,
                    available_cpu,
                    queue_pressure,
                    max_cpu,
                    mean_cpu,
                    effective_compute,
                ]
            )

        raw_server_features = np.asarray(raw_server_features, dtype=np.float32)

        if len(raw_server_features) > 1:
            mean = raw_server_features.mean(axis=0, keepdims=True)
            std = raw_server_features.std(axis=0, keepdims=True)
            normalized = (raw_server_features - mean) / (std + 1e-6)
        else:
            normalized = raw_server_features

        compressed_state = []
        for row in normalized:
            compressed_state.extend(row.tolist())

        task_keys = list(state["TASKS"].keys())
        num_real_tasks = len(task_keys)

        if num_real_tasks > self.batch_size:
            if not self._warned_overflow:
                warnings.warn(
                    f"{num_real_tasks} ready tasks exceed batch_size={self.batch_size}; truncating."
                )
                self._warned_overflow = True
            task_keys = task_keys[: self.batch_size]
            num_real_tasks = self.batch_size

        if num_real_tasks > 0:
            task_values = []
            for task_id in task_keys:
                task = state["TASKS"][task_id]
                task_values.append(
                    [
                        float(task["R_CPU"]),
                        float(task["INSTRUCTIONS"]),
                        1.0,
                    ]
                )

            raw_tasks = np.asarray(task_values, dtype=np.float32)

            if num_real_tasks > 1:
                mean = raw_tasks[:, :2].mean(axis=0, keepdims=True)
                std = raw_tasks[:, :2].std(axis=0, keepdims=True)
                raw_tasks[:, :2] = (raw_tasks[:, :2] - mean) / (std + 1e-6)

            for row in raw_tasks:
                compressed_state.extend(row.tolist())

        for _ in range(self.batch_size - num_real_tasks):
            compressed_state.extend([0.0, 0.0, 0.0])

        return (
            np.asarray(compressed_state, dtype=np.float32),
            num_real_tasks,
        )

    def observe(self, state):
        self.state, self.active_tasks = self.process(state)

    def take_action(self):
        if self.active_tasks <= 0:
            return np.array([], dtype=np.int64)

        n = min(self.active_tasks, self.batch_size)

        if self.rng.random() < self.epsilon:
            sampled = np.asarray(
                self.action_space.sample(),
                dtype=np.int64,
            )
            actions = sampled[:n]
        else:
            state_tensor = torch.as_tensor(
                self.state,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            was_training = self.network.training
            self.network.eval()

            with torch.no_grad():
                _, action_tensor = self.network(state_tensor)

            if was_training:
                self.network.train()

            actions = (
                action_tensor[0, :n].cpu().numpy().astype(np.int64)
            )

        actions = np.clip(actions, 0, self.num_servers - 1)

        if self.action_counts is not None:
            for action in actions:
                self.action_counts[int(action)] += 1

        return actions

    def get_action_distribution(self, reset=False):
        if self.action_counts is None or self.action_counts.sum() == 0:
            return None

        distribution = self.action_counts / self.action_counts.sum()

        if reset:
            self.action_counts = np.zeros_like(self.action_counts)

        return distribution

    def update(self, action, reward, next_state, done):
        """Preserves reward alignment, padding, and environment transition recording."""
        next_state_arr, next_active = self.process(next_state)

        action = np.asarray(action, dtype=np.int64)
        reward = np.asarray(reward, dtype=np.float32)

        n = min(len(action), len(reward), self.batch_size)

        padded_action = np.full(self.batch_size, -1, dtype=np.int64)
        padded_reward = np.zeros(self.batch_size, dtype=np.float32)

        if n > 0:
            padded_action[:n] = action[:n]
            padded_reward[:n] = reward[:n]

        self.buffer.add(
            self.state,
            padded_action,
            padded_reward,
            next_state_arr,
            done,
            next_active=next_active,
        )

        self.step_count += 1

        if (
            self.step_count % self.train_every == 0
            and len(self.buffer) >= self.replay_start_size
        ):
            available = min(self.replay_batch_size, len(self.buffer))
            for _ in range(self.gradient_steps):
                self.replay(available)

        self.state = next_state_arr
        self.active_tasks = next_active

    def replay(self, batch_size):
        if len(self.buffer) < batch_size:
            return

        indices = self.rng.sample(range(len(self.buffer)), batch_size)

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            next_active,
        ) = self.buffer.sample(indices)

        actions = actions.long()
        mask = (actions >= 0).float()
        safe_actions = actions.clamp(min=0, max=self.num_servers - 1)

        # Forward pass: current step Q-values.
        current_q_values, _ = self.network(
            states,
            actions=safe_actions,
        )

        current_q = current_q_values.gather(
            2, safe_actions.unsqueeze(-1)
        ).squeeze(-1)

        # Standard non-sequential target calculation.
        with torch.no_grad():
            _, next_actions = self.network(
                next_states,
            )

            next_target_q_values, _ = self.target_network(
                next_states,
                actions=next_actions,
            )

            next_q = next_target_q_values.gather(
                2, next_actions.unsqueeze(-1)
            ).squeeze(-1)

            next_mask = (
                torch.arange(self.batch_size, device=self.device).unsqueeze(0)
                < next_active.unsqueeze(1)
            ).float()

            bootstrap_mask = (1.0 - dones.unsqueeze(1)) * next_mask

            target_q = (
                rewards + self.gamma * next_q * bootstrap_mask
            )

        valid_count = mask.sum().clamp(min=1.0)

        self.last_q_mean = ((current_q * mask).sum() / valid_count).item()
        self.last_target_mean = ((target_q * mask).sum() / valid_count).item()

        loss_per_element = F.smooth_l1_loss(
            current_q, target_q, reduction="none"
        )
        loss = (loss_per_element * mask).sum() / valid_count

        if not torch.isfinite(loss):
            return

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.network.parameters(), max_norm=10.0
        )

        self.optimizer.step()

        self.gradient_step_count += 1
        self.last_loss = float(loss.detach().cpu().item())

        if self.gradient_step_count % self.target_update_every == 0:
            self.target_network.load_state_dict(self.network.state_dict())
            self.target_network.eval()

    def decay_epsilon(self):
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay,
        )
        
        
    
    def save_model(self, path):

        torch.save(
            self.network.state_dict(),
            path,
        )

        print(f"[DQN] Model saved to: {path}")


    def load_model(self, path=None):
        if not path :
            self.loaded = False
            return False

        self.network.load_state_dict(
            torch.load(
                path,
                map_location=self.device,
                weights_only=True,
            )
        )

        # Keep target network consistent too.
        self.target_network.load_state_dict(
            self.network.state_dict()
        )

        self.network.eval()
        self.target_network.eval()

        print(f"[DQN] Model loaded from: {path}")
        self.loaded = True
        return True