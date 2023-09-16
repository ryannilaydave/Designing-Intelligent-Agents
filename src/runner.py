import gym
import matplotlib.pyplot as plt
import numpy as np
from agent import Agent
import os

class Runner: 
    def __init__(self, environments=[1,2,3], num_trials=5): 
        self.environments = environments
        self.num_trials = num_trials

    def get_abs(self, path):
        script_dir = os.path.dirname(__file__)
        return os.path.join(script_dir, path) 

    def get_load_path(self, problem, trial):
        rel_path = "models/" + problem + "/training/experiment" + str(trial)
        return self.get_abs(rel_path)

    def reset_plot(self):
        plt.clf()
        
    def save_plot(self, path, rewards, label):
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.xlim(0,50)
        plt.ylabel(label)
        plt.ylim(-100,100)
        plt.savefig(path + ".png") 
        print("Plot saved to " + path)  

    def save_reward_list(self, path, list):
        with open(path + ".txt", "w") as f:
            for num in list:
                f.write("%s\n" % num)
        print("Episode rewards saved to " + path) 

    def save_metrics(self, env_version, exp_type, trial, train, rewards, average_rewards):
        path = "MountainCarContinuous-v" + str(env_version) + "/" + exp_type + "/experiment" 
        self.save_reward_list(self.get_abs("reward_lists/" + path + str(trial)), rewards)
        if train: 
            agent.save_models(self.get_abs("models/" + path + str(trial)))
            self.save_plot(self.get_abs("plots/" + path), average_rewards, "Average Reward")
        else:
            self.save_plot(self.get_abs("plots/" + path), rewards, "Reward")

    def run_experiment(self, env_version, exp_type, trial, train, load):
        problem = "MountainCarContinuous-v" + str(env_version) 
        env = gym.make(problem)
        agent = Agent(env, load) 
        rewards, average_rewards = agent.run_episodes(train)
        self.save_metrics(env_version, exp_type, trial, train, rewards, average_rewards)

    def run_control_experiments(self, version):
        self.reset_plot() 
        for i in range(1,self.num_trials+1):
            self.run_experiment(env_version = version, exp_type = "control", trial = i, train = False, load = "")
        
    def run_training_experiments(self, version):
        self.reset_plot()
        for i in range(1,self.num_trials+1):
            self.run_experiment(env_version = version, exp_type = "training", trial = i, train = True, load = "")

    def run_test_experiments(self, version): 
        for i in self.environments:
            self.reset_plot()
            for j in range(1,self.num_trials+1):
                load_path = self.get_load_path("MountainCarContinuous-v" + str(i), j)
                self.run_experiment(env_version = version, exp_type = "test-v" + str(i), trial = j, train = False, load = load_path)

    def run_all_control_experiments(self): 
        for i in self.environments:
            self.run_control_experiments(i)

    def run_all_training_experiments(self):
        for i in self.environments:
            self.run_training_experiments(i)

    def run_all_test_experiments(self):
        for i in self.environments:
            self.run_test_experiments(i)

    def run_all_experiments(self): 
        self.run_all_control_experiments()
        self.run_all_training_experiments()
        self.run_all_test_experiments()