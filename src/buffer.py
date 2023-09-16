import tensorflow as tf
import numpy as np

class Buffer:
    def __init__(self, max_size=20000, batch_size=100):
        self.max_size = max_size
        self.batch_size = batch_size
        self.state_memory = np.zeros((self.max_size, 2))
        self.state_next_memory = np.zeros((self.max_size, 2))
        self.action_memory = np.zeros((self.max_size, 1))
        self.reward_memory = np.zeros((self.max_size, 1))
        self.counter = 0

    def store(self, observation):
        index = self.counter % self.max_size

        self.state_memory[index] = observation[0]
        self.state_next_memory[index] = observation[1]
        self.action_memory[index] = observation[2]
        self.reward_memory[index] = observation[3]
        self.counter += 1

    def sample(self):
        sample_range = min(self.counter, self.max_size)
        sample = np.random.choice(sample_range, self.batch_size)

        states  = tf.convert_to_tensor(self.state_memory[sample], dtype=tf.float32)
        states_ = tf.convert_to_tensor(self.state_next_memory[sample], dtype=tf.float32)
        actions = tf.convert_to_tensor(self.action_memory[sample], dtype=tf.float32)
        rewards = tf.convert_to_tensor(self.reward_memory[sample], dtype=tf.float32)

        return states, states_, actions, rewards

