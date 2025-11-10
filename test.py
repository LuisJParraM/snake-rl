# test_env.py
from snake_env import SnakeEnv

env = SnakeEnv(render=True)
state = env.reset()

done = False
while not done:
    action = 0  # always left, solo para probar
    state, reward, done, info = env.step(action)
    env.render()

env.close()
