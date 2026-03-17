"""
q_learning.py

Tabular Q-learning implementation for discrete-state, discrete-action
Gymnasium environments such as FrozenLake-v1 and Taxi-v3.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym


@dataclass
class QLearningConfig:
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    n_episodes: int = 1000
    max_steps_per_episode: int = 200
    seed: Optional[int] = None


class QLearningAgent:
    """
    Tabular Q-learning agent for discrete environments.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        config: QLearningConfig,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.config = config

        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)
        self.rng = np.random.default_rng(config.seed)

    def select_action(self, state: int, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy during training,
        and greedy policy during evaluation.
        """
        if training and self.rng.random() < self.config.epsilon:
            return int(self.rng.integers(self.n_actions))

        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """
        Standard Q-learning update rule:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        """
        current_q = self.q_table[state, action]

        if done:
            td_target = reward
        else:
            td_target = reward + self.config.gamma * np.max(self.q_table[next_state])

        td_error = td_target - current_q
        self.q_table[state, action] += self.config.alpha * td_error

    def decay_epsilon(self) -> None:
        """
        Decay epsilon after each episode, while respecting epsilon_min.
        """
        self.config.epsilon = max(
            self.config.epsilon_min,
            self.config.epsilon * self.config.epsilon_decay,
        )

    def train_episode(self, env: gym.Env, episode_seed: Optional[int] = None) -> Dict[str, float]:
        """
        Run one training episode.
        """
        state, _ = env.reset(seed=episode_seed)
        total_reward = 0.0
        steps = 0

        for _ in range(self.config.max_steps_per_episode):
            action = self.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            self.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        self.decay_epsilon()

        return {
            "episode_reward": total_reward,
            "episode_length": steps,
        }

    def evaluate_episode(self, env: gym.Env, episode_seed: Optional[int] = None) -> Dict[str, float]:
        """
        Run one greedy evaluation episode without exploration.
        """
        state, _ = env.reset(seed=episode_seed)
        total_reward = 0.0
        steps = 0

        for _ in range(self.config.max_steps_per_episode):
            action = self.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)

            state = next_state
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        return {
            "episode_reward": total_reward,
            "episode_length": steps,
        }

    def train(
        self,
        env: gym.Env,
    ) -> Dict[str, List[float]]:
        """
        Train for multiple episodes and return logged results.
        """
        rewards: List[float] = []
        lengths: List[float] = []
        epsilons: List[float] = []

        base_seed = self.config.seed

        for episode_idx in range(self.config.n_episodes):
            episode_seed = None if base_seed is None else base_seed + episode_idx

            result = self.train_episode(env, episode_seed=episode_seed)

            rewards.append(result["episode_reward"])
            lengths.append(result["episode_length"])
            epsilons.append(self.config.epsilon)

        return {
            "episode_rewards": rewards,
            "episode_lengths": lengths,
            "epsilons": epsilons,
        }

    def evaluate(
        self,
        env: gym.Env,
        n_eval_episodes: int = 20,
    ) -> Dict[str, float]:
        """
        Evaluate the trained policy over multiple episodes.
        """
        rewards = []
        lengths = []

        base_seed = self.config.seed

        for i in range(n_eval_episodes):
            episode_seed = None if base_seed is None else 10_000 + base_seed + i
            result = self.evaluate_episode(env, episode_seed=episode_seed)
            rewards.append(result["episode_reward"])
            lengths.append(result["episode_length"])

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_length": float(np.mean(lengths)),
            "std_length": float(np.std(lengths)),
        }


def run_q_learning(
    env: gym.Env,
    config: QLearningConfig,
) -> Tuple[QLearningAgent, Dict[str, List[float]], Dict[str, float]]:
    """
    Convenience wrapper to:
    1. create agent
    2. train agent
    3. evaluate learned policy
    """
    if not hasattr(env.observation_space, "n"):
        raise TypeError("Environment must have discrete observation space.")

    if not hasattr(env.action_space, "n"):
        raise TypeError("Environment must have discrete action space.")

    n_states = int(env.observation_space.n)
    n_actions = int(env.action_space.n)

    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        config=config,
    )

    training_logs = agent.train(env)
    eval_logs = agent.evaluate(env)

    return agent, training_logs, eval_logs