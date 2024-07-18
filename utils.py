from panda3d.bullet import BulletGenericConstraint, BulletWorld, BulletPlaneShape, BulletRigidBodyNode, BulletBoxShape
from panda3d.core import LineSegs, TransformState, Vec4, Vec3, Point3, Geom, GeomNode, GeomVertexFormat, GeomVertexData, GeomTriangles, GeomVertexWriter, NodePath, Material, DirectionalLight, AmbientLight

def new_box_node(scale=[1, 1, 1], static=False, mass=1):
    # Box
    half_scale = [scale[0]/2, scale[1]/2, scale[2]/2]
    shape = BulletBoxShape(Vec3(*half_scale))
    node = BulletRigidBodyNode('Box')
    if not static:
        node.setMass(mass)
    node.addShape(shape)

    return node
    
def affect(self, a_type, np, value):
    if a_type == 'rotation':
        np.node().applyTorque(Vec3(*value))
    elif a_type == 'movement':
        if not np.node().isActive():
            np.node().setActive(True)
            print('activated')
        np.node().applyCentralForce(Vec3(*value))
        
def join(p1, p2, mobility):
    # t1 = TransformState.makePos(Point3(0, 0, 0))
    t2 = TransformState.makePos(p2.getPos() - p1.getPos())

    t1 = TransformState.makePosHpr(Vec3(0, 0, 0), Vec3(0, 0, 0))
    # t2 = TransformState.makePosHpr(Vec3(0, 0, 0), Vec3(0, 0, 5))

    con = BulletGenericConstraint(p2.node(), p1.node(), t1, t2, True)
    
    for i in range(3):
        linear = mobility['linear']
        angular = mobility['angular']
        con.setLinearLimit(i, linear['min'][i], linear['max'][i])
        con.setAngularLimit(i, angular['min'][i], angular['max'][i])
    
    return con

def get_line(point_a, point_b, thickness=10):
    line_segs = LineSegs()
    
    # Set line color (optional)
    line_segs.setColor(1, 0, 0, 1)  # Red color
            
    # Add the line segment
    line_segs.moveTo(*point_a)
    line_segs.drawTo(*point_b)
    
    # Create a NodePath from the LineSegs
    line_node = line_segs.create()
    
    # Attach the line NodePath to the render tree
    line_np = NodePath(line_node)
    
    # Optionally, set the thickness of the line
    line_np.setRenderModeThickness(thickness)
    return line_np