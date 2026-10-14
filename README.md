# Multiagent Cloud Scheduler

## Overview

This project explores cloud workload scheduling through simulation. The goal is to model a cloud environment, represent task dependencies, and investigate strategies for assigning workloads to available resources.

## What Has Been Done

- Developed a simulation framework for cloud workload scheduling.
- Modeled servers, virtual machines, and tasks with dependencies represented using Directed Acyclic Graphs (DAGs).
- Incorporated scheduling considerations such as resource constraints, task queues, energy consumption, and network transfers.
- Implemented baseline scheduling heuristics to explore and compare scheduling strategies.
- Explored the use of reinforcement learning for scheduling decisions.

## Current Limitations

- The reinforcement learning approach is still experimental and has not reached a reliable, fully working implementation.
- The integration and evaluation of the RL-based scheduler remain incomplete.
- Further testing and validation are needed to establish meaningful comparisons between scheduling strategies.

## Current Bottlenecks

- **Reward instability:** The reinforcement learning reward curve exhibits significant fluctuations during training.
- **Policy instability:** The learned policy remains unstable and may perform inconsistently during evaluation.
- **Reward and interaction design:** Further refinement of the reward function and state-action-reward interactions is needed to improve learning stability and scheduling performance.

## Project Status

This repository represents an ongoing exploration of cloud scheduling. The simulation framework and baseline heuristics form the foundation of the work, while the reinforcement learning component **remains unfinished**.
