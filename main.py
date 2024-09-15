from panda3d.core import Vec4, Vec3, WindowProperties, DirectionalLight, AmbientLight
from panda3d.bullet import BulletDebugNode, BulletWorld
from direct.showbase.ShowBase import ShowBase
import random, utils
import numpy as np
from Character import Character
import sys
import torch

sys.path.append('/home/roman/Desktop/ML/pipeline')

from custom_models.Gym import GymNN

def apply_drag(body):
    fluid_density = .1225  # Density of air at sea level in kg/m^3

    velocity = body.getLinearVelocity()
    speed = velocity.length()
    drag_force_magnitude = -fluid_density * speed**2
    drag_force = velocity.normalized() * drag_force_magnitude
    body.applyCentralForce(drag_force)


# def rotate_body(body, angles):
#     current_hpr = body.getHpr()
#     new_hpr = Point3(current_hpr)
#     new_hpr.setX(new_hpr.getX() + angles[0])
#     new_hpr.setY(new_hpr.getY() + angles[1])
#     new_hpr.setZ(new_hpr.getZ() + angles[2])
#     body.setHpr(new_hpr)

class World(ShowBase):
    def __init__(self) -> None:
        ShowBase.__init__(self)
        self.setBackgroundColor(0.4, .7, .8, 1)
        self.camera = base.cam
        self.camera.setPos(-150, -350, 120)
        self.camera.lookAt(0, 0, -10)
        self.setFrameRateMeter(True)

        props = WindowProperties()
        # props.setFullscreen(True)
        props.setSize(1920, 1080)  # Set the resolution you want

        # Apply the window properties
        self.win.requestProperties(props)

        # World
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.81))

        taskMgr.add(self.update, 'update')


        self.actors = []
        self.init_models()
        self.add_lighting()


        # self.setup_keys()
        # self.setup_debugger()
    
    # def setup_keys(self):
    #     f = 100
        
        # self.accept('q', self.Character.push_leg, [self.player, 0, True])
        # self.accept('q-up', self.Character.push_leg, [self.player, 0, False])

        # self.accept('w', self.Character.push_leg, [self.player, 1, True])
        # self.accept('w-up', self.Character.push_leg, [self.player, 1, False])

        # self.accept('a', self.Character.push_leg, [self.player, 2, True])
        # self.accept('a-up', self.Character.push_leg, [self.player, 2, False])

        # self.accept('s', self.Character.push_leg, [self.player, 3, True])
        # self.accept('s-up', self.Character.push_leg, [self.player, 3, False])

        # self.accept('e', self.Character.turn_leg, [self.player, 0, True])
        # self.accept('e-up', self.Character.turn_leg, [self.player, 0, False])

        # self.accept('r', self.Character.turn_leg, [self.player, 1, True])
        # self.accept('r-up', self.Character.turn_leg, [self.player, 1, False])

        # self.accept('d', self.Character.turn_leg, [self.player, 2, True])
        # self.accept('d-up', self.Character.turn_leg, [self.player, 2, False])

        # self.accept('f', self.Character.turn_leg, [self.player, 3, True])
        # self.accept('f-up', self.Character.turn_leg, [self.player, 3, False])

    


    def init_models(self):
        n_instances = 6
        multiplier = 70
        r = np.linspace(-n_instances / 2, n_instances / 2 + 1, n_instances)
        model_scale = 3
        plane_scale = 70
        for i in r:
            for j in r:
                char = Character(render, loader, self.world)
                x = i * multiplier
                y = j * multiplier
                char.create_new(
                    [x, y - .7 * plane_scale * .5, 5], 
                    [0, 0, 0],
                    [model_scale] * 3,
                    [.2, .3, .8],
                    False
                )
                self.actors.append(char)
                self.add_box(
                    name="Plane",
                    position=[x, y, -5],
                    scale=[plane_scale, plane_scale, .1],
                    color=[.2, .8, .3],
                    static=True
                )

                self.add_box(name="Border", position=[x + plane_scale/2, y, -3.5], scale=[1, plane_scale, 3], color=[.2, .8, .3], static=True)
                self.add_box(name="Border", position=[x, y + plane_scale/2, -3.5], scale=[plane_scale, 1, 3], color=[.2, .8, .3], static=True)
                self.add_box(name="Border", position=[x - plane_scale/2, y, -3.5], scale=[1, plane_scale, 3], color=[.2, .8, .3], static=True)
                self.add_box(name="Border", position=[x, y - plane_scale/2, -3.5], scale=[plane_scale, 1, 3], color=[.2, .8, .3], static=True)

                # place target
                t_x = x + 1.5 * (torch.rand(1) - .5) * plane_scale/2.
                t_y = y + .9*plane_scale/2.
                t_position = [t_x, t_y, -3]
                self.add_box(  
                    name="Target",
                    position=t_position,
                    scale=[3, 3, 3],
                    color=[.8, .2, .3],
                    static=True
                )

                char.assign_target(t_position)


            
    def add_random_box(self, static=False):
        p = [(random.random() - .5) * 5 for _ in range(3)]
        r = [(random.random() - .5) * 180 for _ in range(3)]
        s = [(.2 + .8 * random.random()) for _ in range(3)]
        # s = 3 * [1 + .5 * random.random()]
        # p[2] += 5
        return self.add_box(p, r, s, [.8, .3, .2], False)

    def add_box(self, name, position, rotation=[0, 0, 0], scale=[1, 1, 1], color=[.78, .78, .78], static=False, mass=1):
        box = utils.get_box(name, position, rotation, scale, color, static, mass)
        self.world.attachRigidBody(box.node())
        return box

    # def add_plane(self):
    #     self.add_box(
    #         name="Plane",
    #         position=[0, 0, -5],
    #         scale=[100, 100, .1],
    #         color=[.2, .8, .3],
    #         static=True
    #     )

    def add_lighting(self):
            # Create a directional light
        directional_light = DirectionalLight("directional_light")
        directional_light.setColor(Vec4(1, 1, 1, 1))
        directional_light.setShadowCaster(True, 512, 512)
        directional_light.getLens().setNearFar(1, 5500)
        directional_light.getLens().setFilmSize(50, 50)

        # Attach the light to the render node
        directional_light_np = self.render.attachNewNode(directional_light)
        directional_light_np.setHpr(-70, -60, 0)
        self.render.setLight(directional_light_np)

        # Create an ambient light
        ambient_light = AmbientLight("ambient_light")
        ambient_light.setColor(Vec4(0.8, 0.8, 0.8, 1))

        # Attach the ambient light to the render node
        ambient_light_np = self.render.attachNewNode(ambient_light)
        self.render.setLight(ambient_light_np)

        self.render.setShaderAuto()

    # def create_material(self, color):
    #     # Create a new material
    #     mat = Material()
        
    #     # Set the diffuse color to red
    #     mat.setDiffuse((*color, 1))
        
    #     # Optionally, set other material properties
    #     # mat.setAmbient((0.2, 0, 0, 1))
    #     mat.setSpecular((1, 1, 1, 1))
    #     mat.setShininess(50.0)
        
    #     return mat
    
    def setup_debugger(self):
        debug_node = BulletDebugNode('Debug')
        debug_node.showWireframe(True)
        debug_node.showConstraints(True)
        debug_node.showBoundingBoxes(True)
        debug_node.showNormals(True)
        debug_np = self.render.attachNewNode(debug_node)
        debug_np.show()
        self.world.setDebugNode(debug_np.node())

    def update(self, task):
        dt = globalClock.getDt()

        for player in self.actors:

            # for o in self.Character.objects:
            #     apply_drag(o.node())

            legs = player.state['legs']
            for i, leg_obj in enumerate(legs['objects']):
                c = utils.get_collisions(self.world, leg_obj.node())
                if c is not None and c['obj'].name == 'Plane':
                    legs['collisions'][i] = c
                    leg_obj.getChildren()[0].setColor(1, 1, 1, 1)
                else:
                    legs['collisions'][i] = None
                    leg_obj.getChildren()[0].setColor(.5, .5, .5, 1)
            

            leg_num, is_active = player.interact(legs['collisions'])
            leg_num = torch.argmax(leg_num.detach().squeeze())
            is_active = is_active[0].item() > .5
            player.push_leg(leg_num, is_active)

            for key, a in player.state['affect'].items():
                if a['active']:
                    force = a['force']
                    utils.affect(key, player.state['character'], force)

            # player.state['character'].setCollideMask(utils.NO_COLLISION_MASK)

        self.world.doPhysics(dt)

        return task.cont

world = World()
world.run()
