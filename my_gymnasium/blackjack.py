from collections import defaultdict
from dataclasses import dataclass, asdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm

class BlackjackAgent:
    def __init__(
            self,
            env:gym.Env,
            learning_rate:float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float =0.95,
            debug:bool=False
    ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay=epsilon_decay
        self.final_epsilon = final_epsilon
        self.debug=debug
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool])-> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample() #explore
        else:
            return int(np.argmax(self.q_values[obs]))
    
    def update(
            self,
            obs: tuple[int,int, bool],
            action: int,
            reward:float,
            terminated:bool,
            next_obs: tuple[int, int, bool]
    ):
        if self.debug:
            print(self._summarize_q_values())
        future_q_value=(not terminated)* np.max(self.q_values[next_obs])
        target = reward + self.discount_factor*future_q_value
        temporal_difference = target - self.q_values[obs][action]
        self.q_values[obs][action]=(
            self.q_values[obs][action]+self.lr*temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
    
    def _summarize_q_values(self):
        num_states = len(self.q_values)
        if num_states == 0:
            return {"states": 0, "action_dim": None, "dtype": None}

        # Take one example value (all should have same shape)
        sample = next(iter(self.q_values.values()))
        action_dim = sample.shape
        dtype = sample.dtype

        return {
            "states": num_states,
            "action_dim": action_dim,
            "dtype": dtype,
        }

@dataclass
class Config:
    learning_rate = 0.01
    n_episodes = 100_000
    start_epsilon = 1.0
    final_epsilon = 0.1
    debug=False
    @property
    def epsilon_decay(self)->float:
        return self.start_epsilon/(self.n_episodes/2)
cfg = Config()
env = gym.make("Blackjack-v1", sab=False)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=cfg.n_episodes)
agent = BlackjackAgent(env=env,
                       learning_rate=cfg.learning_rate,
                       initial_epsilon=cfg.start_epsilon,
                       final_epsilon=cfg.final_epsilon,
                       epsilon_decay=cfg.epsilon_decay
                       )

for episode in tqdm(range(cfg.n_episodes)):
    obs, info = env.reset()
    done=False   
    if cfg.debug:     
        print(f"============Episode {episode}====================")


    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        if cfg.debug:
            print(f"Action: {action}")
            print(f"Next observation: {next_obs}")
            print(f"Reward: {reward}")
            print(f"Terminated: {terminated}")
            print(f"Info: {info}")
            print(f"Truncated: {truncated}")
        agent.update(obs, action, reward, terminated, next_obs)
        done = terminated or truncated
        obs = next_obs
    agent.decay_epsilon