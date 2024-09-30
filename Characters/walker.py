import utils
from panda3d.core import Vec3
import utils
import torch
from Characters.character import Character, __MOBILITY__
from NeuralNetworks.walker import WalkerNN

class Walker(Character):
    def __init__(self, render, loader, world, input_dim, hidden_dim, num_categories) -> None:
        super().__init__(render, loader, world)
        self.brain = WalkerNN(input_dim=input_dim, hidden_dim=hidden_dim, num_categories=num_categories)
        self.prev_distance_to_target = -1
        self.score = 0

        self.terminated = False

    def push_leg(self, leg_num, active, force):
        self.state['affect']['movement']['active'] = active
        if active and self.state['legs']['collisions'][leg_num] is not None:
            leg = self.state['legs']['objects'][leg_num]
            dist = self.state['character'].getPos(leg)
            force = dist.normalized() * force * 10
            self.state['affect']['movement']['force'] = force
        else:
            self.state['affect']['movement']['force'] = Vec3(0, 0, 0)

    def turn_leg(self, leg_num, active):
        self.state['affect']['rotation']['active'] = active
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

            self.state['affect']['rotation']['force'] = force

    def remove(self):
        super().remove()
        for l in self.state['legs']['objects']:
            l.removeNode()

    @torch.no_grad()
    def interact(self):
        if self.terminated:
            return

        legs = self.state['legs']

        body_collision = utils.get_collisions(self.world, self.state['character'].node())
        if body_collision is not None: # hit something
            self.terminated = True
            self.world.remove_rigid_body(self.state['character'].node())
            for l in legs['objects']:
                self.world.remove_rigid_body(l.node())
            self.state['character'].getChildren()[0].setColor(.0, 0, 0, 1)
            return

        leg_binary = [0] * len(legs['objects'])
        for i, leg_obj in enumerate(legs['objects']):
            c = utils.get_collisions(self.world, leg_obj.node())
            if c is not None and c['obj'].name == 'Plane':
                legs['collisions'][i] = c
                leg_obj.getChildren()[0].setColor(1, 1, 1, 1)
                leg_binary[i] = 1
            else:
                legs['collisions'][i] = None
                leg_obj.getChildren()[0].setColor(.5, .5, .5, 1)
                leg_binary[i] = 0

        pos = self.objects[0].getPos()
        pos_t = torch.tensor([pos.x, pos.y, pos.z])


        object_forward = self.objects[0].getQuat().getForward()# normalized

        t_pos = self.target.getPos()
        distance_v = t_pos - pos
        distance_n = distance_v.normalized()

        cos_theta = object_forward.dot(distance_n)
        distance = distance_v.length()
        velocity = self.get_velocity().norm()
        # rot = self.objects[0].getHpr()
        # rot_t = torch.tensor([rot.x, rot.y, rot.z])

        # create input tensor
        # [position.xyz, velocity, angle, distance, 1]
        input_t = torch.tensor([velocity, cos_theta, distance, 1])
        input_t = torch.cat((pos_t, input_t))
        # input_t = torch.cat((velocity_t, input_t))
        
        # feed forward brain
        leg_num, is_active, force = self.brain(input_t.unsqueeze(0), torch.tensor(leg_binary).unsqueeze(0))

        # react
        # leg_num = torch.argmax(leg_num.detach().squeeze())
        # is_active = is_active[0].item() > .5
        is_active = True
        for l in torch.where(leg_num > .8)[1]:
            self.push_leg(l, True, force)

        for key, a in self.state['affect'].items():
            if a['active']:
                force = a['force']
                utils.affect(key, self.state['character'], force)

    
    def get_velocity(self):
        v = self.objects[0].node().get_linear_velocity()
        return torch.tensor([v.x, v.y, v.z])

    @torch.no_grad()
    def evaluate(self):
        if self.terminated:
            self.score -= 1
            return
        
        pos = self.objects[0].getPos()
        t_pos = self.target.getPos()
        distance = (t_pos - pos).length()

        if self.prev_distance_to_target > 0:
            ds = self.prev_distance_to_target - distance
            self.score += ds - .01
            
        self.prev_distance_to_target = distance
        
    def create_new(self, position, rotation=[0, 0, 0], scale=[1, 1, 1], color=[.78, .78, .78], static=False, mass=1):
        char, model = utils.get_box('character', position, rotation, scale, color, static, mass)
        self.world.attachRigidBody(char.node())
        self.objects.append(char)
        self.models.append(model)
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
        p2, model = utils.get_box('leg', position=position, scale=scale, mass=mass)
        self.world.attachRigidBody(p2.node())
        constraint = utils.join(p1, p2, mobility)   
        self.world.attachConstraint(constraint)
        self.objects.append(p2)
        self.models.append(model)
        return p2