from pathlib import Path
from src.snake_env import SnakeEnv
from src.dqn_agent import DQNAgent
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "dqn_snake.h5"

NUM_EPISODES = 500       # puedes empezar con 200 para probar
TARGET_UPDATE_FREQ = 10  # episodios

def train():
    env = SnakeEnv(render=False)   # sin render para entrenar rápido
    state_size = env.reset().shape[0]
    action_size = env.action_space

    agent = DQNAgent(state_size, action_size)

    for e in range(1, NUM_EPISODES + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        step_count = 0

        while not done:
            # choose action
            action = agent.act(state)

            # env step
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # store
            agent.remember(state, action, reward, next_state, done)

            # train
            agent.replay()

            state = next_state
            step_count += 1

            # optional: limit steps per episode
            if step_count > 500:
                done = True

        # update target network
        if e % TARGET_UPDATE_FREQ == 0:
            agent._update_target_model()

        print(f"Episode {e}/{NUM_EPISODES} - reward: {total_reward:.2f}, "
              f"epsilon: {agent.epsilon:.3f}")

    # save model
    agent.save(str(MODEL_PATH))
    env.close()

if __name__ == "__main__":
    train()
