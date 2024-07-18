from panda3d.core import LineSegs, TransformState, Vec4, Vec3, Point3, Geom, GeomNode, GeomVertexFormat, GeomVertexData, GeomTriangles, GeomVertexWriter, NodePath, Material, DirectionalLight, AmbientLight
from panda3d.bullet import BulletGenericConstraint, BulletWorld
from direct.showbase.ShowBase import ShowBase
import random, utils
import numpy as np
from Character import Character



class World(ShowBase):
    def __init__(self) -> None:
        ShowBase.__init__(self)
        self.setBackgroundColor(0.4, .7, .8, 1)
        self.camera = base.cam
        self.camera.setPos(0, -70, 10)
        self.camera.lookAt(0, 0, 0)

        # World
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -19.81))

        taskMgr.add(self.update, 'update')

        self.status = {
            'rotation': [0, 0, 0],
            'movement': [0, 0, 0],
        }

        self.Character = Character(render, loader, self.world)
        self.player = None
        self._leg_num = -1
        self.add_plane()
        self.init_models()
        self.setup_keys()
        self.add_lighting()
    
    def setup_keys(self):
        f = 100
        
        self.accept('q', self.move_leg, [0, True])
        self.accept('q-up', self.move_leg, [0, False])

        self.accept('w', self.move_leg, [1, True])
        self.accept('w-up', self.move_leg, [1, False])

        self.accept('a', self.move_leg, [2, True])
        self.accept('a-up', self.move_leg, [2, False])

        self.accept('s', self.move_leg, [3, True])
        self.accept('s-up', self.move_leg, [3, False])

    def move_leg(self, leg_num, active):
        self._leg_num = leg_num if active else -1

    def set_body_rotation(self, axis, value):
        self.status['rotation'][axis] = value

    def set_body_movement(self, axis, value):
        self.status['movement'][axis] = value

    def affect(self, a_type, np, value):
        if a_type == 'rotation':
            np.node().applyTorque(Vec3(*value))
        elif a_type == 'movement':
            np.node().applyCentralForce(Vec3(*value))


    def apply_drag(self, body):
        fluid_density = .1225  # Density of air at sea level in kg/m^3

        velocity = body.getLinearVelocity()
        speed = velocity.length()
        drag_force_magnitude = -fluid_density * speed**2
        drag_force = velocity.normalized() * drag_force_magnitude
        body.applyCentralForce(drag_force)


    def rotate_body(self, body, angles):
        current_hpr = body.getHpr()  # Get current HPR (heading, pitch, roll)
        new_hpr = Point3(current_hpr)  # Create a new Point3 to modify
        new_hpr.setX(new_hpr.getX() + angles[0])  # Increment X (heading) by 10 degrees
        new_hpr.setY(new_hpr.getY() + angles[1])  # Increment X (heading) by 10 degrees
        new_hpr.setZ(new_hpr.getZ() + angles[2])  # Increment X (heading) by 10 degrees
        body.setHpr(new_hpr)  # Apply the new heading, pitch, and roll to p1


    def init_models(self):
        n = 1
        m = 10
        r = np.linspace(-n / 2, n / 2 + 1, n)
        s = 3
        for i in r:
            for j in r:
                c = self.Character.create_new(
                    [i * m, j * m, 5], 
                    [0, 0, 0],
                    [s, s, s],
                    [.2, .3, .8],
                    False
                )
                if self.player is None:
                    self.player = c

            
    def add_random_box(self, static=False):
        p = [(random.random() - .5) * 5 for _ in range(3)]
        r = [(random.random() - .5) * 180 for _ in range(3)]
        s = [(.2 + .8 * random.random()) for _ in range(3)]
        # s = 3 * [1 + .5 * random.random()]
        # p[2] += 5
        return self.add_box(p, r, s, [.8, .3, .2], False)

    def add_box(self, position, rotation=[0, 0, 0], scale=[1, 1, 1], color=[.78, .78, .78], static=False, mass=1):
        return self.Character.get_box(position, rotation, scale, color, static, mass)

    def add_plane(self):
        self.add_box(
            position=[0, 0, -5],
            scale=[100, 100, .1],
            color=[.2, .8, .3],
            static=True
        )

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

    def create_material(self, color):
        # Create a new material
        mat = Material()
        
        # Set the diffuse color to red
        mat.setDiffuse((*color, 1))
        
        # Optionally, set other material properties
        # mat.setAmbient((0.2, 0, 0, 1))
        mat.setSpecular((1, 1, 1, 1))
        mat.setShininess(50.0)
        
        return mat

        model = loader.loadModel('models/box.egg')
        model.setTextureOff(1)
        model.setScale(*scale)
        model.setColor(*color)
        model.setPos(*[-h for h in half_scale])

        model.reparentTo(np)

        return np

    def get_cube_model(self):
        format = GeomVertexFormat.getV3()
        data = GeomVertexData("Data", format, Geom.UHStatic)
        vertices = GeomVertexWriter(data, "vertex")

        size = .5
        vertices.addData3f(-size, -size, -size)
        vertices.addData3f(+size, -size, -size)
        vertices.addData3f(-size, +size, -size)
        vertices.addData3f(+size, +size, -size)
        vertices.addData3f(-size, -size, +size)
        vertices.addData3f(+size, -size, +size)
        vertices.addData3f(-size, +size, +size)
        vertices.addData3f(+size, +size, +size)

        triangles = GeomTriangles(Geom.UHStatic)

        def addQuad(v0, v1, v2, v3):
            triangles.addVertices(v0, v1, v2)
            triangles.addVertices(v0, v2, v3)
            triangles.closePrimitive()

        addQuad(4, 5, 7, 6) # Z+
        addQuad(0, 2, 3, 1) # Z-
        addQuad(3, 7, 5, 1) # X+
        addQuad(4, 6, 2, 0) # X-
        addQuad(2, 6, 7, 3) # Y+
        addQuad(0, 1, 5, 4) # Y+

        geom = Geom(data)
        geom.addPrimitive(triangles)

        node = GeomNode("CubeMaker")
        node.addGeom(geom)

        return NodePath(node)


    def update(self, task):
        dt = globalClock.getDt()
        player = self.player['character']
        # for key, value in self.status.items():
            # if sum([abs(x) for x in value]) > 0:
        if self._leg_num > -1:
            leg = self.player['legs'][self._leg_num]
            dist = player.getPos(leg)
            force = dist.normalized() * 20
            self.affect('movement', player, force)
              
        # for o in self.Character.objects:
        #     self.apply_drag(o.node())

        self.world.doPhysics(dt)
        return task.cont

world = World()
world.run()
