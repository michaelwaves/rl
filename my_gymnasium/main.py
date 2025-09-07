import gymnasium as gym


def main():
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    observation, info = env.reset()
    print(f"Starting observation: {observation}")
    episode_over = False
    total_reward = 0

    while not episode_over:
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_over = terminated or truncated
    print(f"Episode finished. Total reward: {total_reward}")
    env.close()
    print(env)


if __name__ == "__main__":
    main()
