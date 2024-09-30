import utils
import torch
from Characters.character import Character, __MOBILITY__
from NeuralNetworks.thrower import ThrowerNN

class Thrower(Character):
    def __init__(self, render, loader, world) -> None:
        super().__init__(render, loader, world)
        self.brain = ThrowerNN(input_dim=2, hidden_dim=30, output_dim=3)
        self.score = 9999
        self.terminated = False
        self.proj_shot = False

    def remove(self):
        super().remove()
        self.state['projectile'].removeNode()

    @torch.no_grad()
    def interact(self):
        if self.proj_shot:
            return

        proj = self.state['projectile']
        
        # feed forward brain
        pos = self.state['projectile'].getPos()
        t_pos = self.target.getPos()
        direction = (t_pos - pos)
        input_t = torch.tensor([direction.x, direction.y]).unsqueeze(0)
        force_v = self.brain(input_t).detach().squeeze()
        
        utils.affect('movement', proj, force_v * 150)
        self.proj_shot = True

    
    @torch.no_grad()
    def evaluate(self):
        if self.terminated:
            return
        
        proj = self.state['projectile']
        proj_collision = utils.get_collisions(self.world, proj.node())
        if proj_collision is not None: # hit something

            pos = self.state['projectile'].getPos()
            t_pos = self.target.getPos()
            self.score = (t_pos - pos).length()
            self.terminated = True
        
    def create_new(self, position, rotation=[0, 0, 0], scale=[1, 1, 1], color=[.78, .78, .78], static=False, mass=1):
        char, model = utils.get_box('character', position, rotation, scale, color, static, mass)
        self.world.attachRigidBody(char.node())
        self.objects.append(char)
        self.models.append(model)
        
        p = position
        p[2] += .5+(scale[2] + 1.5) / 2
        projectile, p_model = utils.get_box('projectile', p, rotation, [1.5, 1.5, 1.5], [.5, .67, .33], static, mass)
        self.world.attachRigidBody(projectile.node())
        self.models.append(p_model)

        self.state = {
            'character': char,
            'projectile': projectile,
            'affect': {
                'movement': {'active': False, 'force': None, 'value': [0, 0, 0]},
                'rotation': {'active': False, 'force': None, 'value': [0, 0, 0]}
            }
        }