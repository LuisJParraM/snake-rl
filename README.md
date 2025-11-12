# 🐍 Snake AI - Deep Q-Learning (DQN)

This project implements the classic **Snake** game (Nokia-style) and trains an AI agent to play it using **Deep Reinforcement Learning (DQN)** built with **TensorFlow**.

---

## 🎯 Objective

The goal is to develop a custom **Reinforcement Learning (RL)** environment where an agent learns to play Snake by:
- Observing the game state.
- Taking discrete actions (left, right, up, down).
- Receiving rewards based on its behavior.

---

## 🧠 Project Structure
snake-rl/
│
├── snake_env.py # Custom game environment (similar to OpenAI Gym)
├── dqn_agent.py # Deep Q-Network agent implementation
├── train_dqn.py # Training script for the agent
├── play_trained.py # Visual test of the trained model
└── README.md # Project documentation

---

## 🕹️ Environment Overview (`snake_env.py`)

- **Grid size:** 84×48 logical pixels (Nokia screen style).  
- **Action space:**  
  `0 = LEFT`, `1 = RIGHT`, `2 = UP`, `3 = DOWN`.
- **Render mode:** Pygame window (optional).

### 🧩 State representation (11 features)

| # | Feature | Description |
|---|----------|-------------|
| 1 | `danger_straight` | 1 if there is an obstacle straight ahead |
| 2 | `danger_right` | 1 if there is an obstacle on the right |
| 3 | `danger_left` | 1 if there is an obstacle on the left |
| 4 | `dir_left` | Current direction: moving left |
| 5 | `dir_right` | Current direction: moving right |
| 6 | `dir_up` | Current direction: moving up |
| 7 | `dir_down` | Current direction: moving down |
| 8 | `food_left` | Food is located to the left |
| 9 | `food_right` | Food is located to the right |
| 10 | `food_up` | Food is above |
| 11 | `food_down` | Food is below |

---

## 🧮 Reward Function

| Situation | Reward |
|------------|--------|
| Snake dies (collision) | **-10.0** |
| Moves closer to food | **+1.0** |
| Moves away from food | **-0.5** |
| Eats the food | **+10.0** |

This dense reward encourages exploration while guiding the agent toward the food.

---

## 🧩 Deep Q-Network Agent (`dqn_agent.py`)

- Fully connected neural network (Dense layers with ReLU activation).  
- Uses **experience replay** to stabilize learning.  
- Updates a **target network** every few episodes.  
- Implements **ε-greedy exploration** strategy.

---

## ⚙️ Training Script (`train_dqn.py`)

Run the following command to train your model:

```bash
python train_dqn.py

```
You will see console output similar to:
```bash
Episode 1/500 - reward: -2.6, epsilon: 0.59
Episode 2/500 - reward: -1.4, epsilon: 0.42
Episode 3/500 - reward:  0.3, epsilon: 0.30
...
```
After training, the model is saved automatically as :

```bash
dqn_snake.h5
```
---

## 🧪 Watching the Trained Agent (play_trained.py):
To visualize the agent playing the game:
```bash
python play_trained.py
```
A Pygame window will open, showing the Snake controlled by the trained neural network policy.

---

## 📈 Possible Improvements

* Plot training curves (average reward, loss, epsilon decay).

* Implement Prioritized Experience Replay.

* Use a CNN-based model for visual state input.

* Save the model using the modern .keras format.

* Add sound or UI effects for visual polish.

---

# 👨‍💻 Author
Luis J. Parra
Engineering Student – ENIM / ENSAM
Master in Industrial Performance and Innovation (IPI 4.0)
Project developed for the Artificial Intelligence and Reinforcement Learning course.

---

## 📄 License
You are free to use, modify, and distribute this code, as long as credit is given to the author.

---



