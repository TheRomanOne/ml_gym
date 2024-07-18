import utils, random
import numpy as np
from panda3d.core import LineSegs, TransformState, Vec4, Vec3, Point3, Geom, GeomNode, GeomVertexFormat, GeomVertexData, GeomTriangles, GeomVertexWriter, NodePath, Material, DirectionalLight, AmbientLight

__MOBILITY__ = {
        'SOLID': {
            'linear': {'min': [0, 0, 0], 'max': [0, 0, 0]},
            'angular': {'min': [0, 0, 0], 'max': [0, 0, 0]}
            },

        'LOOSE': {
            'linear': {'min': [0, 0, 0], 'max': [00, 0, 0]},
            'angular': {'min': [-20, 0, 0], 'max': [20, 0, 0]}
            }
}

class Character:
    def __init__(self, render, loader, world) -> None:
        self.render = render
        self.world = world
        self.loader = loader

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
    
    def create_new(self, num_weights, position, rotation=[0, 0, 0], scale=[1, 1, 1], color=[.78, .78, .78], static=False, mass=1):
        player = self.get_box(position, rotation, scale, color, static, mass)
        self.objects.append(player)
        pos = 3 * [0]
        chars = [player]
        for i in range(15):
            p1 = chars[i]
            v = np.random.uniform(1, 3, size=3)
            if random.random() < .5: v[0] *= -1
            if random.random() < .5: v[1] *= -1
            if random.random() < .5: v[2] *= -1
            v = v.tolist()
            p2 = self.add_new_weight(p1, position=v, scale=[.5, .5, .5], mass=.5, mobility=__MOBILITY__['SOLID'])
            chars.append(p2)
            utils.get_line(v, pos).reparentTo(p1)
        return player
    
    def get_box(self, position, rotation=[0, 0, 0], scale=[1, 1, 1], color=[.78, .78, .78], static=False, mass=1):
        node = utils.new_box_node(scale, static, mass)
        np = self.render.attachNewNode(node)
        np.setHpr(*rotation)  
        np.setPos(*position)

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
        p2 = self.get_box(position=position, scale=scale, mass=mass)
        constraint = utils.join(p1, p2, mobility)   
        self.world.attachConstraint(constraint)
        self.objects.append(p2)
        return p2
    
    