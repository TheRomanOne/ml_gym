import utils, sys
from panda3d.core import Vec3
import utils
import torch
from Characters.character import Character


class Walker(Character):
    def __init__(self, render, loader, world, input_dim, hidden_dim, num_categories) -> None:
        super().__init__(render, loader, world, input_dim, hidden_dim, num_categories)
        self.prev_distance_to_target = -1
        self.score = 0

        # self.terminated = False

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

    @torch.no_grad()
    def interact(self):
        legs = self.state['legs']
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
        leg_num = torch.argmax(leg_num.detach().squeeze())
        is_active = is_active[0].item() > .5
        self.push_leg(leg_num, is_active, force)

        for key, a in self.state['affect'].items():
            if a['active']:
                force = a['force']
                utils.affect(key, self.state['character'], force)

    def get_velocity(self):
        v = self.objects[0].node().get_linear_velocity()
        return torch.tensor([v.x, v.y, v.z])

    @torch.no_grad()
    def evaluate(self):
        pos = self.objects[0].getPos()
        t_pos = self.target.getPos()
        distance = (t_pos - pos).length()

        if self.prev_distance_to_target > 0:
            ds = self.prev_distance_to_target - distance
            self.score += ds - .01
            
        self.prev_distance_to_target = distance
        