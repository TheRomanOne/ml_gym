from panda3d.core import Vec4, Vec3, WindowProperties, DirectionalLight, AmbientLight
from panda3d.bullet import BulletDebugNode, BulletWorld
from direct.showbase.ShowBase import ShowBase
import sys, torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/roman/Desktop/ML/pipeline')

import utils
from Characters.walker import Walker
from train.gen_algo import GeneticAlgorithm

def apply_drag(body):
    fluid_density = .1225  # Density of air at sea level in kg/m^3

    velocity = body.getLinearVelocity()
    speed = velocity.length()
    drag_force_magnitude = -fluid_density * speed**2
    drag_force = velocity.normalized() * drag_force_magnitude
    body.applyCentralForce(drag_force)



class World(ShowBase):
    def __init__(self) -> None:
        ShowBase.__init__(self)
        self.setBackgroundColor(0.4, .7, .8, 1)
        self.camera = base.cam
        self.camera.setPos(-140, -340, 70)
        self.camera.lookAt(0, 0, -70)
        self.setFrameRateMeter(True)

        props = WindowProperties()
        # props.setFullscreen(True)
        props.setSize(1920, 1080)  # Set the resolution you want

        # Apply the window properties
        self.win.requestProperties(props)
        
        # World

        taskMgr.add(self.update, 'update')


        self.actors = []
        self.models = []
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -2*9.81))
        self.reset_world()
        self.add_lighting()

        self.epochs = 0
        self.max_epochs = 250

        self.mean_scores = []
        self.ga = GeneticAlgorithm(
                fitness_function=lambda a: a.score,
                random_genome_value=lambda n: (torch.rand(n) - 0.5) * 2,
                maximize=True,
                elitism=.3,
                maxPopulationSize=len(self.actors)
            )
    def init_population(self, grid_size):
        multiplier = 70
        r = np.linspace(-grid_size / 2, grid_size / 2 + 1, grid_size)
        model_scale = 3
        plane_scale = 90
        for i in r:
            for j in r:
                char = Walker(render, loader, self.world, input_dim=7, hidden_dim=100, num_categories=4)
                x = i * multiplier
                y = j * multiplier
                char.create_new(
                    [x, y - .7 * plane_scale * .5, 1], 
                    [0, 0, 0],
                    [model_scale] * 3,
                    [.2, .3, .8],
                    False
                )
                self.actors.append(char)
                self.add_training_env(x, y, plane_scale)

                # place target
                t_x = x + 1.5 * (torch.rand(1) - .5) * plane_scale/2.
                t_y = y + 1.5 * torch.rand(1) * plane_scale/4.
                # t_y = y + .9*plane_scale/2.
                t_position = [t_x, t_y, -3]
                target = self.add_box(  
                    name="Target",
                    position=t_position,
                    scale=[3, 3, 3],
                    color=[.8, .2, .3],
                    static=True
                )
                char.assign_target(target)


    def add_training_env(self, x, y, plane_scale):
        env = []
        env.append(self.add_box(name="Plane", position=[x, y, -5], scale=[plane_scale, plane_scale, .1], color=[.2, .8, .3], static=True))
        env.append(self.add_box(name="Border", position=[x + plane_scale/2, y, -3.5], scale=[1, plane_scale, 3], color=[.2, .8, .3], static=True))
        env.append(self.add_box(name="Border", position=[x, y + plane_scale/2, -3.5], scale=[plane_scale, 1, 3], color=[.2, .8, .3], static=True))
        env.append(self.add_box(name="Border", position=[x - plane_scale/2, y, -3.5], scale=[1, plane_scale, 3], color=[.2, .8, .3], static=True))
        env.append(self.add_box(name="Border", position=[x, y - plane_scale/2, -3.5], scale=[plane_scale, 1, 3], color=[.2, .8, .3], static=True))
        return env

    def add_box(self, name, position, rotation=[0, 0, 0], scale=[1, 1, 1], color=[.78, .78, .78], static=False, mass=1):
        box, model = utils.get_box(name, position, rotation, scale, color, static, mass)
        self.models.append(model)
        self.world.attachRigidBody(box.node())
        return box

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

    def setup_debugger(self):
        debug_node = BulletDebugNode('Debug')
        debug_node.showWireframe(True)
        debug_node.showConstraints(True)
        debug_node.showBoundingBoxes(True)
        debug_node.showNormals(True)
        debug_np = self.render.attachNewNode(debug_node)
        debug_np.show()
        self.world.setDebugNode(debug_np.node())

    def reset_world(self, new_population=None):
        for m in self.models:
            m.removeNode()
            m.detachNode()
        
        for p in self.actors:
            for m in p.models:
                m.removeNode()
                m.detachNode()

        self.models.clear()
        self.actors.clear()

        for body in list(self.world.getRigidBodies()):
            self.world.remove_rigid_body(body)

        num = int(np.sqrt(len(new_population))) if new_population is not None else 5

        self.init_population(num)

        if new_population is not None:
            for i, new_brain in enumerate(new_population):
                model = self.actors[i].brain
                utils.unflatten_model(model, new_brain)


        self.epochs = 0

    def update(self, task):
        dt = globalClock.getDt()

        if self.epochs < self.max_epochs:
            for player in self.actors:
                player.interact()

            self.world.doPhysics(dt)

            for player in self.actors:
                player.evaluate()

            self.epochs += 1

        else:
            self.actors.sort(key=lambda a: a.score, reverse=True)
            brains = [utils.flatten_model(a.brain) for a in self.actors[:5]]
            self.ga.init_population(brains)
            mean_score = np.array([a.score for a in self.actors]).mean() / self.max_epochs
            print(f'Generation {len(self.mean_scores)}| reward {mean_score}')
            self.mean_scores.append(mean_score)
            # Reset the world with the new population
            self.reset_world(self.ga.population)

        return task.cont
    
world = World()
world.run()
