import utils, sys
from panda3d.core import Vec3
import utils
import torch

sys.path.append('/home/roman/Desktop/ML/pipeline')

from custom_models.Gym import GymNN

__MOBILITY__ = {
        'SOLID': {
            'linear': {'min': [0, 0, 0], 'max': [0, 0, 0]},
            'angular': {'min': [0, 0, 0], 'max': [0, 0, 0]}
            },

        'LOOSE': {
            'linear': {'min': [0, 0, -1], 'max': [0, 0, 1]},
            'angular': {'min': [-0.2, -0.2, 0], 'max': [0.2, 0.2, 0]}
            }
}

class Character:
    def __init__(self, render, loader, world, input_dim, hidden_dim) -> None:
        self.render = render
        self.world = world
        self.loader = loader
        self.brain = GymNN(input_dim=input_dim, hidden_dim=hidden_dim)
        self.objects = []

    def create_new(self, position, rotation=[0, 0, 0], scale=[1, 1, 1], color=[.78, .78, .78], static=False, mass=1):
        char = utils.get_box('character', position, rotation, scale, color, static, mass)
        self.world.attachRigidBody(char.node())
        self.objects.append(char)
        # pos = 3 * [0]
        legs = []
        for i in [-2.2, 2.2]:
            for j in [-2.2, 2.2]:
                v = [i, j, -scale[0] - 2]
                p2 = self.add_new_weight(char, position=v, scale=[scale[0]/2.5, scale[0]/2.5, scale[0]/7], mass=.5, mobility=__MOBILITY__['LOOSE'])
                legs.append(p2)
        
        self.state = {
            'character': char,
            'legs': {'objects': legs, 'collisions': [None for n in legs]},
            'affect': {
                'movement': {'active': False, 'force': None, 'value': [0, 0, 0]},
                'rotation': {'active': False, 'force': None, 'value': [0, 0, 0]}
            }
        }

    def add_new_weight(self, p1, position, scale=[1, 1, 1], mass=1, mobility=None):
        position = Vec3(*position) + p1.getPos()
        p2 = utils.get_box('leg', position=position, scale=scale, mass=mass)
        self.world.attachRigidBody(p2.node())
        constraint = utils.join(p1, p2, mobility)   
        self.world.attachConstraint(constraint)
        self.objects.append(p2)
        return p2
    
    
    def assign_target(self, target):
        self.target = target

    def interact(self):
        pass

    def evaluate(self):
        pass
    