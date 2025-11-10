import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # replay buffer
        self.memory = deque(maxlen=50000)

        # hyperparams
        self.gamma = 0.99       # discount
        self.epsilon = 1.0      # exploration start
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.learning_rate = 1e-3

        # main & target networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self._update_target_model()

    def _build_model(self):
        """Simple MLP for Q-values."""
        model = models.Sequential()
        model.add(layers.Input(shape=(self.state_size,)))
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dense(self.action_size, activation="linear"))
        model.compile(
            loss="mse",
            optimizer=optimizers.Adam(learning_rate=self.learning_rate)
        )
        return model

    def _update_target_model(self):
        """Copy weights from main to target."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store experience."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Epsilon-greedy action."""
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        q_vals = self.model.predict(state[np.newaxis, :], verbose=0)
        return np.argmax(q_vals[0])

    def replay(self):
        """Train on random minibatch."""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([m[0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])

        # Q(s,a) update target
        target_q = self.model.predict(states, verbose=0)
        next_q = self.target_model.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                target_q[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])

        self.model.fit(states, target_q, epochs=1, verbose=0)

        # epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        # load model architecture + weights, ignore old loss/metrics
        self.model = tf.keras.models.load_model(path, compile=False)

        # (optional) recompile if you also want to keep training later
        self.model.compile(
            loss="mse",
            optimizer=optimizers.Adam(learning_rate=self.learning_rate)
        )

        # update target network
        self._update_target_model()
