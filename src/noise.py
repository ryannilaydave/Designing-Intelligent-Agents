import numpy as np

class Noise:
    def __init__(self, sigma=0.25, theta=0.2): 
        self.sigma = sigma
        self.theta = theta
        self.mu = np.zeros(1)
        self.prev = np.zeros_like(self.mu)
        self.dt = 1e-2

    def decay(self):
        self.sigma = max(0.25, self.sigma*0.99)
        self.theta = max(0.2, self.theta*0.99)
    
    def get_noise(self): # Converted from matlab code in https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

        x = self.prev + self.theta * (self.mu - self.prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.prev = x
        
        return x
