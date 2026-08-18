# RL/agents/DQNAgent.py

import copy
import warnings
import random

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


class QNetwork(nn.Module):
    def __init__(
        self,
        state_size,
        num_actions,
        layer_config,
        num_tasks=None,
    ):
        super().__init__()

        activation_map = {
            "relu": nn.ReLU,
            "leakyrelu": nn.LeakyReLU,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
        }

        modules = []
        input_dim = state_size

        for layer_size, activation_name in layer_config:
            if activation_name.lower() not in activation_map:
                raise ValueError(
                    f"Unsupported activation: {activation_name}"
                )

            modules.append(
                nn.Linear(
                    input_dim,
                    layer_size,
                )
            )
            modules.append(
                activation_map[
                    activation_name.lower()
                ]()
            )
            input_dim = layer_size

        final_output_size = (
            num_actions * num_tasks
            if num_tasks is not None
            else num_actions
        )

        modules.append(
            nn.Linear(
                input_dim,
                final_output_size,
            )
        )

        self.model = nn.Sequential(*modules)
        self.num_actions = num_actions
        self.num_tasks = num_tasks

    def forward(self, x):
        out = self.model(x)

        if self.num_tasks is not None:
            out = out.view(
                -1,
                self.num_tasks,
                self.num_actions,
            )

        return out


class SequentialDecisionNetwork(nn.Module):
    def __init__(
        self,
        servers_encoding_dim,
        tasks_context_dim,
        num_servers,
        num_tasks,
        task_token_dim=3,
        hidden_dim=128,
    ):
        super().__init__()

        self.num_servers = num_servers
        self.num_tasks = num_tasks
        self.task_token_dim = task_token_dim

        self.hidden_init = nn.Linear(
            servers_encoding_dim + tasks_context_dim,
            hidden_dim,
        )

        self.gru = nn.GRUCell(
            task_token_dim + num_servers,
            hidden_dim,
        )

        self.q_head = nn.Linear(
            hidden_dim + servers_encoding_dim,
            num_servers,
        )

        self.capacity_update = nn.Sequential(
            nn.Linear(
                servers_encoding_dim
                + num_servers
                + task_token_dim,
                servers_encoding_dim,
            ),
            nn.Tanh(),
        )

        self.capacity_norm = nn.LayerNorm(
            servers_encoding_dim
        )

    def forward(
        self,
        servers_encoding,
        tasks_context,
        tasks_raw,
        actions=None,
    ):
        batch = servers_encoding.size(0)
        device = servers_encoding.device

        h = torch.tanh(
            self.hidden_init(
                torch.cat(
                    [
                        servers_encoding,
                        tasks_context,
                    ],
                    dim=-1,
                )
            )
        )

        capacity_state = servers_encoding

        prev_onehot = torch.zeros(
            batch,
            self.num_servers,
            device=device,
        )

        all_q_values = []
        chosen_actions = []

        for t in range(self.num_tasks):
            start = t * self.task_token_dim
            end = start + self.task_token_dim

            task_token = tasks_raw[:, start:end]

            gru_input = torch.cat(
                [
                    task_token,
                    prev_onehot,
                ],
                dim=-1,
            )

            h = self.gru(
                gru_input,
                h,
            )

            q_values_t = self.q_head(
                torch.cat(
                    [
                        h,
                        capacity_state,
                    ],
                    dim=-1,
                )
            )

            all_q_values.append(q_values_t)

            if actions is not None:
                action_t = actions[:, t]
                action_t = action_t.clamp(
                    min=0,
                    max=self.num_servers - 1,
                )
            else:
                action_t = torch.argmax(
                    q_values_t,
                    dim=-1,
                )

            chosen_actions.append(action_t)

            onehot = F.one_hot(
                action_t,
                num_classes=self.num_servers,
            ).float()

            capacity_delta = self.capacity_update(
                torch.cat(
                    [
                        capacity_state,
                        onehot,
                        task_token,
                    ],
                    dim=-1,
                )
            )

            capacity_state = self.capacity_norm(
                capacity_state + capacity_delta
            )

            prev_onehot = onehot

        q_values = torch.stack(
            all_q_values,
            dim=1,
        )

        actions_used = torch.stack(
            chosen_actions,
            dim=1,
        )

        return q_values, actions_used


class DQNNetwork(nn.Module):
    def __init__(
        self,
        servers_network,
        tasks_network,
        decision_network,
    ):
        super().__init__()

        self.servers_network = servers_network
        self.tasks_network = tasks_network
        self.decision_network = decision_network

    def forward(
        self,
        servers_state,
        tasks_state,
        actions=None,
    ):
        servers_encoding = self.servers_network(
            servers_state
        )

        tasks_encoding = self.tasks_network(
            tasks_state
        )

        return self.decision_network(
            servers_encoding,
            tasks_encoding,
            tasks_state,
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
        seed=123,
    ):
        super().__init__(
            "DQN",
            "Deep Q-Network Agent "
            "(sequential decision decoder)",
        )

        if layers_config is None:
            layers_config = [
                (128, "relu"),
                (256, "relu"),
                (128, "relu"),
            ]

        self.state = None
        self.active_tasks = 0

        self.servers_network = None
        self.tasks_network = None
        self.decision_network = None
        self.network = None
        self.target_network = None

        self.servers_state_size = None
        self.tasks_state_size = None

        self.task_token_dim = 3

        self.optimizer_name = optimizer

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

        self.replay_batch_size = 64
        self.replay_start_size = 256
        self.train_every = 10
        self.gradient_steps = 4
        self.target_update_every = 250

        self.learning_rate = 1e-3
        self.weight_decay = 0.0

        self.step_count = 0
        self.gradient_step_count = 0

        self.last_loss = None
        self.last_q_mean = None
        self.last_target_mean = None

        self.action_counts = None

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self._warned_overflow = False
        self.built = False

    def build(self):
        if self.action_space is None:
            raise RuntimeError(
                "Action space not provided by environment"
            )

        if self.batch_size is None:
            raise RuntimeError(
                "Batch size not provided by environment"
            )

        if self.num_servers is None:
            raise RuntimeError(
                "Number of servers not provided by environment"
            )

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.action_space.seed(self.seed)

        self.servers_state_size = (
            6 * self.num_servers
        )

        self.tasks_state_size = (
            self.task_token_dim * self.batch_size
        )

        self.servers_network = QNetwork(
            state_size=self.servers_state_size,
            num_actions=self.servers_state_size,
            layer_config=self.layers_config,
        )

        self.tasks_network = QNetwork(
            state_size=self.tasks_state_size,
            num_actions=self.tasks_state_size,
            layer_config=self.layers_config,
        )

        self.decision_network = (
            SequentialDecisionNetwork(
                servers_encoding_dim=self.servers_state_size,
                tasks_context_dim=self.tasks_state_size,
                num_servers=self.num_servers,
                num_tasks=self.batch_size,
                task_token_dim=self.task_token_dim,
                hidden_dim=self.decoder_hidden_dim,
            )
        )

        self.network = DQNNetwork(
            self.servers_network,
            self.tasks_network,
            self.decision_network,
        ).to(self.device)

        self.target_network = copy.deepcopy(
            self.network
        ).to(self.device)

        self.target_network.load_state_dict(
            self.network.state_dict()
        )

        self.target_network.eval()

        for parameter in self.target_network.parameters():
            parameter.requires_grad_(False)

        self.buffer = Buffer(
            capacity=10_000,
            obs_dim=(
                self.servers_state_size
                + self.tasks_state_size
            ),
            act_dim=self.batch_size,
            device=self.device,
        )

        optimizer_map = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
        }

        optimizer_class = optimizer_map.get(
            self.optimizer_name.lower()
        )

        if optimizer_class is None:
            raise ValueError(
                f"Unsupported optimizer: "
                f"{self.optimizer_name}"
            )

        self.optimizer = optimizer_class(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.action_counts = np.zeros(
            self.num_servers,
            dtype=np.int64,
        )

        self.network.train()
        self.built = True

    def process(self, state):
        raw_server_features = []

        server_ids = list(
            state["C_CPU"].keys()
        )

        for server_id in server_ids:
            cpu_capacity = max(
                float(state["C_CPU"][server_id]),
                1e-6,
            )

            cpu_history = np.asarray(
                state["CPU"][server_id],
                dtype=np.float32,
            )

            if cpu_history.size == 0:
                current_cpu = 0.0
                max_cpu = 0.0
                mean_cpu = 0.0
            else:
                current_cpu = float(
                    cpu_history[-1]
                )
                max_cpu = float(
                    np.max(cpu_history)
                )
                mean_cpu = float(
                    np.mean(cpu_history)
                )

            available_cpu = (
                float(
                    state["MAX_AVAILABLE_CPU"][
                        server_id
                    ]
                )
                / cpu_capacity
            )

            queue_pressure = (
                float(
                    state["QUEUE"][server_id]
                )
                / cpu_capacity
            )

            effective_compute = float(
                state["EFFECTIVE_COMPUTE"][
                    server_id
                ]
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

        raw_server_features = np.asarray(
            raw_server_features,
            dtype=np.float32,
        )

        if len(raw_server_features) > 1:
            mean = raw_server_features.mean(
                axis=0,
                keepdims=True,
            )
            std = raw_server_features.std(
                axis=0,
                keepdims=True,
            )

            normalized = (
                raw_server_features - mean
            ) / (std + 1e-6)
        else:
            normalized = raw_server_features

        compressed_state = []

        for row in normalized:
            compressed_state.extend(
                row.tolist()
            )

        task_keys = list(
            state["TASKS"].keys()
        )

        num_real_tasks = len(task_keys)

        if num_real_tasks > self.batch_size:
            if not self._warned_overflow:
                warnings.warn(
                    f"{num_real_tasks} ready tasks exceed "
                    f"batch_size={self.batch_size}; "
                    f"truncating the task batch."
                )
                self._warned_overflow = True

            task_keys = task_keys[
                : self.batch_size
            ]

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

            raw_tasks = np.asarray(
                task_values,
                dtype=np.float32,
            )

            if num_real_tasks > 1:
                mean = raw_tasks[:, :2].mean(
                    axis=0,
                    keepdims=True,
                )

                std = raw_tasks[:, :2].std(
                    axis=0,
                    keepdims=True,
                )

                raw_tasks[:, :2] = (
                    raw_tasks[:, :2] - mean
                ) / (std + 1e-6)

            for row in raw_tasks:
                compressed_state.extend(
                    row.tolist()
                )

        for _ in range(
            self.batch_size - num_real_tasks
        ):
            compressed_state.extend(
                [0.0, 0.0, 0.0]
            )

        return (
            np.asarray(
                compressed_state,
                dtype=np.float32,
            ),
            num_real_tasks,
        )

    def observe(self, state):
        self.state, self.active_tasks = (
            self.process(state)
        )

    def take_action(self):
        if self.active_tasks <= 0:
            return np.array(
                [],
                dtype=np.int64,
            )

        n = min(
            self.active_tasks,
            self.batch_size,
        )

        if self.rng.random() < self.epsilon:
            sampled = np.asarray(
                self.action_space.sample(),
                dtype=np.int64,
            )

            actions = sampled[:n]
        else:
            servers_state = self.state[
                : self.servers_state_size
            ]

            tasks_state = self.state[
                self.servers_state_size :
            ]

            server_tensor = torch.as_tensor(
                servers_state,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            task_tensor = torch.as_tensor(
                tasks_state,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            was_training = self.network.training
            self.network.eval()

            with torch.no_grad():
                _, action_tensor = self.network(
                    server_tensor,
                    task_tensor,
                )

            if was_training:
                self.network.train()

            actions = (
                action_tensor[
                    0,
                    :n,
                ]
                .cpu()
                .numpy()
                .astype(np.int64)
            )

        actions = np.clip(
            actions,
            0,
            self.num_servers - 1,
        )

        if self.action_counts is not None:
            for action in actions:
                self.action_counts[
                    int(action)
                ] += 1

        return actions

    def get_action_distribution(self, reset=False):
        if (
            self.action_counts is None
            or self.action_counts.sum() == 0
        ):
            return None

        distribution = (
            self.action_counts
            / self.action_counts.sum()
        )

        if reset:
            self.action_counts = np.zeros_like(
                self.action_counts
            )

        return distribution

    def update(
        self,
        action,
        reward,
        next_state,
        done,
    ):
        next_state_arr, next_active = (
            self.process(next_state)
        )

        action = np.asarray(
            action,
            dtype=np.int64,
        )

        reward = np.asarray(
            reward,
            dtype=np.float32,
        )

        n = min(
            len(action),
            len(reward),
            self.batch_size,
        )

        padded_action = np.full(
            self.batch_size,
            -1,
            dtype=np.int64,
        )

        padded_reward = np.zeros(
            self.batch_size,
            dtype=np.float32,
        )

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
            self.step_count
            % self.train_every
            == 0
            and len(self.buffer)
            >= self.replay_start_size
        ):
            available = min(
                self.replay_batch_size,
                len(self.buffer),
            )

            for _ in range(
                self.gradient_steps
            ):
                self.replay(available)

        self.state = next_state_arr
        self.active_tasks = next_active

    def replay(self, batch_size):
        if len(self.buffer) < batch_size:
            return

        indices = self.rng.sample(
            range(len(self.buffer)),
            batch_size,
        )

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            next_active,
        ) = self.buffer.sample(indices)

        actions = actions.long()

        mask = (
            actions >= 0
        ).float()

        safe_actions = actions.clamp(
            min=0,
            max=self.num_servers - 1,
        )

        servers_states = states[
            :, : self.servers_state_size
        ]

        tasks_states = states[
            :, self.servers_state_size :
        ]

        next_servers_states = next_states[
            :, : self.servers_state_size
        ]

        next_tasks_states = next_states[
            :, self.servers_state_size :
        ]

        current_q_values, _ = self.network(
            servers_states,
            tasks_states,
            actions=safe_actions,
        )

        current_q = current_q_values.gather(
            2,
            safe_actions.unsqueeze(-1),
        ).squeeze(-1)

        with torch.no_grad():
            _, next_actions = self.network(
                next_servers_states,
                next_tasks_states,
            )

            next_target_q_values, _ = (
                self.target_network(
                    next_servers_states,
                    next_tasks_states,
                    actions=next_actions,
                )
            )

            next_q = next_target_q_values.gather(
                2,
                next_actions.unsqueeze(-1),
            ).squeeze(-1)

            next_mask = (
                torch.arange(
                    self.batch_size,
                    device=self.device,
                )
                .unsqueeze(0)
                < next_active.unsqueeze(1)
            ).float()

            bootstrap_mask = (
                1.0 - dones.unsqueeze(1)
            ) * next_mask

            target_q = (
                rewards
                + self.gamma
                * next_q
                * bootstrap_mask
            )

        valid_count = mask.sum().clamp(
            min=1.0
        )

        self.last_q_mean = (
            (current_q * mask).sum()
            / valid_count
        ).item()

        self.last_target_mean = (
            (target_q * mask).sum()
            / valid_count
        ).item()

        loss_per_element = (
            F.smooth_l1_loss(
                current_q,
                target_q,
                reduction="none",
            )
        )

        loss = (
            loss_per_element * mask
        ).sum() / valid_count

        if not torch.isfinite(loss):
            return

        self.optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.network.parameters(),
            max_norm=10.0,
        )

        self.optimizer.step()

        self.gradient_step_count += 1
        self.last_loss = float(
            loss.detach().cpu().item()
        )

        if (
            self.gradient_step_count
            % self.target_update_every
            == 0
        ):
            self.target_network.load_state_dict(
                self.network.state_dict()
            )

            self.target_network.eval()

    def decay_epsilon(self):
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay,
        )