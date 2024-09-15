import utils, sys
from panda3d.core import Vec3
import utils
import torch
from Characters.character import Character

sys.path.append('/home/roman/Desktop/ML/pipeline')

class Walker(Character):
    def __init__(self, render, loader, world, input_dim=9, hidden_dim=100) -> None:
        super().__init__(render, loader, world, input_dim, hidden_dim)
        self.prev_distance_to_target = 9999

    def push_leg(self, leg_num, active):
        self.state['affect']['movement']['active'] = active
        if active and self.state['legs']['collisions'][leg_num] is not None:
            leg = self.state['legs']['objects'][leg_num]
            dist = self.state['character'].getPos(leg)
            force = dist.normalized() * 70
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

    
    def interact(self):
        legs = self.state['legs']
        for i, leg_obj in enumerate(legs['objects']):
            c = utils.get_collisions(self.world, leg_obj.node())
            if c is not None and c['obj'].name == 'Plane':
                legs['collisions'][i] = c
                leg_obj.getChildren()[0].setColor(1, 1, 1, 1)
            else:
                legs['collisions'][i] = None
                leg_obj.getChildren()[0].setColor(.5, .5, .5, 1)

        pos = self.objects[0].getPos()
        pos_t = torch.tensor([pos.x, pos.y, pos.z])


        object_forward = self.objects[0].getQuat().getForward()# normalized

        t_pos = self.target.getPos()
        distance_v = t_pos - pos
        distance_n = distance_v.normalized()

        cos_theta = object_forward.dot(distance_n)
        distance = distance_v.length()
        velocity_t = self.get_velocity()
        # rot = self.objects[0].getHpr()
        # rot_t = torch.tensor([rot.x, rot.y, rot.z])

        # create input tensor
        # [position.xyz, velocity.xyz, angle, distance, 1]
        input_t = torch.tensor([cos_theta, distance, 1])
        input_t = torch.cat((velocity_t, input_t))
        input_t = torch.cat((pos_t, input_t))
        # input_t = torch.cat((t_pos_t, input_t))
        # input_t = torch.cat((rot_t, input_t))

        # feed forward brain
        leg_num, is_active = self.brain(input_t.unsqueeze(0))

        # react
        leg_num = torch.argmax(leg_num.detach().squeeze())
        is_active = is_active[0].item() > .5
        self.push_leg(leg_num, is_active)

        for key, a in self.state['affect'].items():
            if a['active']:
                force = a['force']
                utils.affect(key, self.state['character'], force)

    def get_velocity(self):
        v = self.objects[0].node().get_linear_velocity()
        return torch.tensor([v.x, v.y, v.z])

    def evaluate(self):
        pass