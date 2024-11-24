import numpy as np
import time
# import pygame

class Mole:
    def __init__(self, size, colour, required_strength):
        self.size = None
        self.colour = 'black'
        self.duration = duration    #depends on how long we want the mole to stay o the screen
        self.required_strength = 0

    def get_hit(self, agent_strength):
        return(agent_strength >= self.strength)

class SmallMole(Mole):
    def __init__(self, position):
        super().__init__(position)
        self.size = 'small'
        self.required_strength = 0.2
        self.color = 'yellow'
        self.score = 10


class LargeMole(Mole):
    def __init__(self, position):
        super().__init__(position)
        self.size = 'large'
        self.required_strength = 0.7
        self.color = 'brown'
        self.score = 20



class Agent:
    def __init__(self, avg_reaction_time):
        self.reaction_time = np.random.normal(avg_reaction_time, 0.05)  #not sure if we need this or not,imitating the situation people see the mole but still missed it        
        self.hit_strength = None  
        self.accuracy = 0.95  
        self.score = 0
        #self.learning_rate = 0.1  

    def hit_mole(self,mole):
        if mole.required_strength <= self.hit_strength:
            if np.random.random() < self.accuracy:  # Simulating hit accuracy not sure if we need this or not,imitating the situation people see the mole but still missed it 
                self.add_score(mole)  # Add points when it is a successful hit
                return True
        return False

    def get_score(self):
        return self.score
    
    def add_score(self, Mole):
        self.score += Mole.score

    def change_strength(self, new_strength):
        self.hit_strength= new_strength

