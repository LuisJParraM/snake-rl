from snake_env import SnakeEnv
from dqn_agent import DQNAgent
import numpy as np

def play_one_episode():
    # env with render enabled
    env = SnakeEnv(render=True)
    state = env.reset()

    state_size = state.shape[0]
    action_size = env.action_space

    # create agent and load trained model
    agent = DQNAgent(state_size, action_size)
    agent.load("dqn_snake.h5")
    agent.epsilon = 0.0  # no random actions

    done = False
    total_reward = 0.0

    while not done:
        # choose best action from model
        action = agent.act(state)

        # step env
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        # render frame
        env.render()

        state = next_state

    print("Episode finished, total reward:", total_reward)
    env.close()

if __name__ == "__main__":
    play_one_episode()
