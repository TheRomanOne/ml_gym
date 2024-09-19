import utils
from Characters.walker import Walker
import torch
from panda3d.bullet import BulletDebugNode, BulletWorld
from panda3d.core import Vec3


class Scene:
    def __init__(self, x, y, render, loader) -> None:
        self.render = render
        self.loader = loader
        self.x_coord = x
        self.y_coord = y
        self.model_scale = 3
        self.plane_scale = 90
        self.field_models = []
        self.field_rigids = []
        self.actor = None
        self.world = None
        self.reset_world()
        self.init_model()
        self.build_new_training_field()
        self.add_target()
    
    def reset_world(self):
        if self.world is not None:
            self.actor.state['character'].removeNode()
            for l in self.actor.state['legs']['objects']:
                l.removeNode()
            self.actor.target.removeNode()
            del self.actor.target
            del self.actor
            
            for body in list(self.world.getRigidBodies()):
                self.world.remove_rigid_body(body)

            # Remove all constraints from the world
            for constraint in list(self.world.getConstraints()):
                self.world.remove_constraint(constraint)

            # Remove any ghost objects if you have them
            for ghost in list(self.world.getGhosts()):
                self.world.remove_ghost(ghost)
            del self.world

        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -2*9.81))

    def init_model(self):
        char = Walker(self.render, self.loader, self.world, input_dim=7, hidden_dim=70, num_categories=4)
        char.create_new(
            [self.x_coord, self.y_coord - .7 * self.plane_scale * .5, 1], 
            [0, 0, 0],
            [self.model_scale] * 3,
            [.2, .3, .8],
            False
        )
        self.actor = char
    
    def add_target(self):
        # place target
        t_x = self.x_coord + 1.5 * (torch.rand(1) - .5) * self.plane_scale/2.
        t_y = self.y_coord + 1.5 * torch.rand(1) * self.plane_scale/4.
        # t_y = y + .9*plane_scale/2.
        t_position = [t_x, t_y, -3]
        box, target = utils.get_box(  
            name="Target",
            position=t_position,
            scale=[3, 3, 3],
            color=[.8, .2, .3],
            static=True
        )
         
        self.world.attachRigidBody(box.node())
        self.actor.assign_target(target)
    
    def build_new_training_field(self):
        self.field_models = []
        self._built_training_block(name="Plane", position=[self.x_coord, self.y_coord, -5], scale=[self.plane_scale, self.plane_scale, .1], color=[.2, .8, .3], static=True)
        self._built_training_block(name="Border", position=[self.x_coord + self.plane_scale/2, self.y_coord, -3.5], scale=[1, self.plane_scale, 3], color=[.2, .8, .3], static=True)
        self._built_training_block(name="Border", position=[self.x_coord, self.y_coord + self.plane_scale/2, -3.5], scale=[self.plane_scale, 1, 3], color=[.2, .8, .3], static=True)
        self._built_training_block(name="Border", position=[self.x_coord - self.plane_scale/2, self.y_coord, -3.5], scale=[1, self.plane_scale, 3], color=[.2, .8, .3], static=True)
        self._built_training_block(name="Border", position=[self.x_coord, self.y_coord - self.plane_scale/2, -3.5], scale=[self.plane_scale, 1, 3], color=[.2, .8, .3], static=True)

    def _built_training_block(self, name, position, rotation=[0, 0, 0], scale=[1, 1, 1], color=[.78, .78, .78], static=False, mass=1):
        box, model = utils.get_box(name, position, rotation, scale, color, static, mass)
        self.field_models.append(model)
        self.field_rigids.append(box)
        self.world.attachRigidBody(box.node())
    
    def advance(self, dt):
        self.actor.interact()
        self.world.doPhysics(dt)

    def evaluate(self):
        self.actor.evaluate()

    def reset_brain(self, brain):
        self.reset_world()
        for r in self.field_rigids:
            self.world.attachRigidBody(r.node())
        self.init_model()
        self.add_target()

        model = self.actor.brain
        utils.unflatten_model(model, brain)
        
    def score(self):
        return self.actor.score