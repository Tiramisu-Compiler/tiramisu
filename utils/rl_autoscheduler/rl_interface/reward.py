from builtins import property
import math

class Reward:
    def __init__(self,reward):
        self._reward=reward
    
    @property
    def reward(self):
        return self.log_reward()
    
    @reward.setter
    def reward(self,value):
        self._reward = value
    
    def log_reward(self,base=4):
        return math.log(abs(self._reward),base)