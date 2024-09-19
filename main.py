from panda3d.core import Vec4, Vec3, WindowProperties, DirectionalLight, AmbientLight, TransformState
from panda3d.bullet import BulletDebugNode, BulletWorld
from direct.showbase.ShowBase import ShowBase
import sys, torch
import numpy as np
import matplotlib.pyplot as plt
import gc

sys.path.append('/home/roman/Desktop/ML/pipeline')

from Scene.scene import Scene
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


        # self.actors = []
        # self.models = []
        # self.world = BulletWorld()
        # self.world.setGravity(Vec3(0, 0, -2*9.81))
        self.scenes = []
        self.grid_size = 5
        self.restart_simulation()
        self.add_lighting()

        self.epochs = 0
        self.max_epochs = 250
        self.mean_scores = []
        self.ga = GeneticAlgorithm(
                fitness_function=lambda a: a.score,
                random_genome_value=lambda n: (torch.rand(n) - 0.5) * 2,
                maximize=True,
                elitism=.2,
                mutation=.3,
                maxPopulationSize=len(self.scenes)
            )

    def add_lighting(self):
            # Create a directional light
        directional_light = DirectionalLight("directional_light")
        directional_light.setColor(Vec4(1, 1, 1, 1))
        directional_light.setShadowCaster(True, 1024, 1024)
        directional_light.getLens().setNearFar(1, 5500)
        directional_light.getLens().setFilmSize(50, 50)

        # Attach the light to the render node
        directional_light_np = self.render.attachNewNode(directional_light)
        directional_light_np.setHpr(-70, -60, 0)
        self.render.setLight(directional_light_np)

        # Create an ambient light
        ambient_light = AmbientLight("ambient_light")
        ambient_light.setColor(Vec4(0.8, 0.8, 0.8, 1))

        # # Attach the ambient light to the render node
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

    def restart_simulation(self, new_population=None):
        
        multiplier = 70
        h_grid = self.grid_size / 2
        r = np.linspace(-h_grid, h_grid + 1, self.grid_size)

        if new_population is None:
            for i in r:
                for j in r:
                    self.scenes.append(Scene(i * multiplier, j * multiplier, render, loader))
        else:
            for i, new_brain in enumerate(new_population):
                self.scenes[i].reset_brain(new_brain)


        self.epochs = 0
        gc.collect()

    def update(self, task):
        dt = globalClock.getDt()

        if self.epochs < self.max_epochs:
            for scene in self.scenes:
                scene.advance(dt)
                scene.evaluate()

            self.epochs += 1

        else:
            n_parents = max(5, int(len(self.scenes) * .2))
            sorted_scenes = sorted(self.scenes, key=lambda a: a.score(), reverse=True)[:n_parents]
            mean_score = np.array([a.score() for a in sorted_scenes]).mean() / self.max_epochs

            brains = [utils.flatten_model(s.actor.brain) for s in sorted_scenes]
            self.ga.init_population(brains)
            print(f'Generation {len(self.mean_scores)} | reward {mean_score}')
            self.mean_scores.append(mean_score)

            self.restart_simulation(self.ga.population)

        return task.cont
    
world = World()
world.run()
