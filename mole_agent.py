import numpy as np
import random

# import pygame


class Mole:
    def __init__(self, size, colour, required_strength):
        self.size = None
        self.colour = 'black'
        self.duration = duration  # depends on how long we want the mole to stay o the screen
        self.required_strength = 0


class Agent:
    def __init__(self, avg_reaction_time):
        # not sure if we need this or not,imitating the situation people see the mole but still missed it
        self.reaction_time = np.random.normal(avg_reaction_time, 0.05)
        self.hit_strength = None
        self.accuracy = 0.95
        self.score = 0
        # self.learning_rate = 0.1

    def hit_mole(self, mole):
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
        self.hit_strength = new_strength
