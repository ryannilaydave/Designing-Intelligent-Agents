import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from buffer import Buffer
from noise import Noise

# Class influenced by paper https://arxiv.org/pdf/2105.07998.pdf (not direct copy, rather as guidance, namely policy method), credit given. 
class Agent:
    def __init__(self, env, load_path):
        self.env = env

        self.tau = 0.01 
        self.discount = 0.99
        self.batch_size = 100
        self.exp_required = 100

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

        self.actor = self.init_actor()
        self.critic = self.init_critic()
        self.target_actor = self.init_actor()
        self.target_critic = self.init_critic()

        self.buffer = Buffer(batch_size = self.batch_size)
        self.noise = Noise(sigma=0.5, theta=0.5)

        if load_path != "":
            self.load_models(load_path)
        else :
            self.target_actor.set_weights(self.actor.get_weights())
            self.target_critic.set_weights(self.critic.get_weights())

    def init_actor(self):
        kernel_init = tf.random_uniform_initializer(minval=-0.05, maxval=0.05)
        inputs = layers.Input(shape=(2))
        out = layers.Dense(64, activation="relu")(inputs)
        out = layers.Dense(64, activation="relu")(out)
        outputs = (layers.Dense(1, activation="tanh", kernel_initializer=kernel_init)(out)) * self.env.action_space.high[0]
        return tf.keras.Model(inputs, outputs)

    def init_critic(self):
        state_in = layers.Input(shape=(2))
        state_out = layers.Dense(32, activation="relu")(state_in)
        action_in = layers.Input(shape=(1))
        action_out = layers.Dense(32, activation="relu")(action_in)
        concat = layers.Concatenate()([state_out, action_out])
        out = layers.Dense(64, activation="relu")(concat)
        out = layers.Dense(64, activation="relu")(out)
        out = layers.Dense(1)(out)
        return tf.keras.Model([state_in, action_in], out)

    def update_actor(self, states, states_, actions, rewards):
        with tf.GradientTape() as tape:
            actions = self.actor(states, training=True)
            q = self.critic([states, actions], training=True)
            actor_loss = -tf.math.reduce_mean(q)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

    def update_critic(self, states, states_, actions, rewards):
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_, training=True)
            q_next_target = self.target_critic([states_, target_actions], training=True)
            q_target = rewards + self.discount * q_next_target
            q = self.critic([states, actions], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(q_target - q))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))
     
    def soft_update(self, target_model, model, tau):
        for (target_model_variables, model_variables) in zip(target_model.variables, model.variables):
            target_model_variables.assign(model_variables * tau + model_variables * (1 - tau))

    def policy(self, state, noise): 
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        sampled_actions = tf.squeeze(self.actor(state)).numpy() + self.noise.get_noise() 
        restrict_bounds = np.clip(sampled_actions, self.env.action_space.low[0], self.env.action_space.high[0])
        return [np.squeeze(restrict_bounds)] 

    def load_models(self, path):
        self.actor.load_weights(path + "/actor.h5")
        self.target_actor.load_weights = (path + "/target_actor.h5")
        self.critic.load_weights(path + "/critic.h5")
        self.target_critic.load_weights(path + "/target_critic.h5")
        print("Model weights loaded from " + path)

    def save_models(self, path):
        self.actor.save_weights(path + "/actor.h5")
        self.target_actor.save_weights(path + "/target_actor.h5")
        self.critic.save_weights(path + "/critic.h5")
        self.target_critic.save_weights(path + "/target_critic.h5")
        print("Model weights saved to " + path)

    def run_episodes(self, train):
        episode_rewards = []
        average_rewards = []

        for episode in range(50):
            episode_reward = 0
            state = self.env.reset()

            while True:
                self.env.render()

                action = self.policy(state, self.noise)
                state_, reward, done, info = self.env.step(action)
                self.buffer.store((state, state_, action, reward))
                episode_reward += reward 

                if train and self.buffer.counter > self.exp_required:
                    states, states_, actions, rewards = self.buffer.sample()
                    
                    self.update_actor(states, states_, actions, rewards)
                    self.update_critic(states, states_, actions, rewards)
                    self.soft_update(self.target_actor, self.actor, self.tau)
                    self.soft_update(self.target_critic, self.critic, self.tau)

                state = state_

                if done: break
            
            episode_rewards.append(episode_reward)
            average_rewards.append(np.average(episode_rewards))
            print(str(episode) + "  " + str(episode_reward))
            
        self.env.close()
         
        return episode_rewards, average_rewards 