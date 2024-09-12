import utils, random, sys
import numpy as np
from panda3d.core import Vec3, BitMask32
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
    def __init__(self, render, loader, world) -> None:
        self.render = render
        self.world = world
        self.loader = loader
        self.brain = GymNN(input_dim=7, hidden_dim=100)
        self.objects = []

    def create_char_spikes(self, position, rotation=[0, 0, 0], scale=[1, 1, 1], color=[.78, .78, .78], static=False, mass=1):
        player = self.get_box(position, rotation, scale, color, static, mass)
        self.objects.append(player)
        pos = 3 * [0]
        for i in range(15):
            v = np.random.uniform(1, 3, size=3)
            if random.random() < .5: v[0] *= -1
            if random.random() < .5: v[1] *= -1
            if random.random() < .5: v[2] *= -1
            v = v.tolist()
            self.add_new_weight(player, position=v, scale=[.5, .5, .5], mass=.5, mobility=__MOBILITY__['SOLID'])
            utils.get_line(v, pos).reparentTo(player)
        return player
    
    def push_leg(self, char, leg_num, active):
        char['affect']['movement']['active'] = active
        if active and char['legs']['collisions'][leg_num] is not None:
            leg = char['legs']['objects'][leg_num]
            dist = char['character'].getPos(leg)
            force = dist.normalized() * 50
            char['affect']['movement']['force'] = force
        else:
            char['affect']['movement']['force'] = Vec3(0, 0, 0)

    def turn_leg(self, char, leg_num, active):
        char['affect']['rotation']['active'] = active
        if active:
            force = [0, 0, 0]
            if leg_num == 0:
                force[0] = 100
            elif leg_num == 1:
                force[0] = -100
            elif leg_num == 2:
                force[2] = 100
            elif leg_num == 3:
                force[2] = -100

            char['affect']['rotation']['force'] = force

    def create_new(self, position, rotation=[0, 0, 0], scale=[1, 1, 1], color=[.78, .78, .78], static=False, mass=1):
        player = self.get_box('Player', position, rotation, scale, color, static, mass)
        self.objects.append(player)
        pos = 3 * [0]
        legs = []
        for i in [-2, 2]:
            for j in [-2, 2]:
                v = [i, j, -scale[0] - 2]
                p2 = self.add_new_weight(player, position=v, scale=[scale[0]/2, scale[0]/2, scale[0]/7], mass=.5, mobility=__MOBILITY__['LOOSE'])
                legs.append(p2)
        
        char = {
            'character': player,
            'legs': {'objects': legs, 'collisions': [None for n in legs]},
            'affect': {
                'movement': {'active': False, 'force': None},
                'rotation': {'active': False, 'force': None}
            }
        }
        
        return char

    def get_box(self, name, position, rotation=[0, 0, 0], scale=[1, 1, 1], color=[.78, .78, .78], static=False, mass=1):
        node = utils.new_box_node(scale, static, mass)
        np = self.render.attachNewNode(node)
        np.setName(name)
        np.setHpr(*rotation)  
        np.setPos(*position)
        np.setCollideMask(BitMask32.allOn())

        half_scale = [scale[0]/2, scale[1]/2, scale[2]/2]
        model = self.loader.loadModel('models/box.egg')
        model.setTextureOff(1)
        model.setScale(*scale)
        model.setColor(*color)
        model.setPos(*[-h for h in half_scale])

        model.reparentTo(np)
        self.world.attachRigidBody(node)
        return np

    def add_new_weight(self, p1, position, scale=[1, 1, 1], mass=1, mobility=None):
        position = Vec3(*position) + p1.getPos()
        p2 = self.get_box('leg', position=position, scale=scale, mass=mass)
        constraint = utils.join(p1, p2, mobility)   
        self.world.attachConstraint(constraint)
        self.objects.append(p2)
        return p2
    
    def interact(self, collisions):
        pos = self.objects[0].getPos()
        rot = self.objects[0].getHpr()
        
        pos = torch.tensor([pos.x, pos.y, pos.z])
        rot = torch.tensor([rot.x, rot.y, rot.z])

        # add loose param
        input_t = torch.cat((torch.cat((pos, rot)), torch.tensor([1])))
        return self.brain(input_t.unsqueeze(0))