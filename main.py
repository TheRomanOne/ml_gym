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
        self.camera.setPos(0, -50, 10)
        self.camera.lookAt(0, 0, 0)

        # World
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.81))

        taskMgr.add(self.update, 'update')

        self.status = {
            'rotation': [0, 0, 0],
            'movement': [0, 0, 0],
        }

        self.Character = Character(render, loader, self.world)
        self.add_lighting()
        self.add_plane()
        self.init_models()

        f = 100
        buttons = {'x': ['r', 'f'], 'y': ['e', 'd'], 'z': ['t', 'g'] }

        for i, key in enumerate(buttons.keys()):
            for j, k in enumerate(buttons[key]):
                self.accept(k, self.set_body_rotation, [i, f * (1 if j % 2 == 0 else -1)])
                self.accept(f'{k}-up', self.set_body_rotation, [i, 0])


    def set_body_rotation(self, axis, value):
        self.status['rotation'][axis] = value

    def apply_force(self, body, force):
        body.applyCentralForce(force)

    def apply_drag(self, body):
        fluid_density = .1225  # Density of air at sea level in kg/m^3

        velocity = body.getLinearVelocity()
        speed = velocity.length()
        drag_force_magnitude = -fluid_density * speed**2
        drag_force = velocity.normalized() * drag_force_magnitude
        body.applyCentralForce(drag_force)

    def apply_torque(self, np, torque):
        np.node().applyTorque(Vec3(*torque))

    def rotate_body(self, body, angles):
        current_hpr = body.getHpr()  # Get current HPR (heading, pitch, roll)
        new_hpr = Point3(current_hpr)  # Create a new Point3 to modify
        new_hpr.setX(new_hpr.getX() + angles[0])  # Increment X (heading) by 10 degrees
        new_hpr.setY(new_hpr.getY() + angles[1])  # Increment X (heading) by 10 degrees
        new_hpr.setZ(new_hpr.getZ() + angles[2])  # Increment X (heading) by 10 degrees
        body.setHpr(new_hpr)  # Apply the new heading, pitch, and roll to p1

    def get_line(self, point_a, point_b, thickness=10):
        self.line_segs = LineSegs()
        
        # Set line color (optional)
        self.line_segs.setColor(1, 0, 0, 1)  # Red color
                
        # Add the line segment
        self.line_segs.moveTo(*point_a)
        self.line_segs.drawTo(*point_b)
        
        # Create a NodePath from the LineSegs
        self.line_node = self.line_segs.create()
        
        # Attach the line NodePath to the render tree
        self.line_np = NodePath(self.line_node)
        
        # Optionally, set the thickness of the line
        self.line_np.setRenderModeThickness(thickness)
        return self.line_np

    def init_models(self):
        n = 8
        m = 6
        r = np.linspace(-n / 2, n / 2, n)
        for i in r:
            for j in r:
                self.Character.create_new(
                    3,
                    [i * m, j * m, 5], 
                    [0, 0, 0],
                    [1, 1, 1],
                    [.2, .3, .8],
                    False
                )

            
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
            scale=[40, 40, .1],
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
        if sum([abs(x) for x in self.status['rotation']]) > 0:
            self.apply_torque(self.player, self.status['rotation'])
            
        for o in self.Character.objects:
            self.apply_drag(o.node())

        self.world.doPhysics(dt)
        return task.cont

world = World()
world.run()
