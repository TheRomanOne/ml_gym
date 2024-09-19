from panda3d.bullet import BulletContactResult, BulletGenericConstraint, BulletRigidBodyNode, BulletBoxShape
from panda3d.core import LineSegs, TransformState, BitMask32, Vec3, NodePath
import torch


NO_COLLISION_MASK = BitMask32.bit(0)
# DEFAULT_MASK = BitMask32.allOn()
DEFAULT_MASK = BitMask32.bit(1)

def new_box_node(scale=[1, 1, 1], static=False, mass=1):
    # Box
    half_scale = [scale[0]/2, scale[1]/2, scale[2]/2]
    shape = BulletBoxShape(Vec3(*half_scale))
    node = BulletRigidBodyNode('Box')
    if not static:
        node.setMass(mass)
    node.addShape(shape)

    return node
    
def affect(a_type, np, value):
    node = np.node()
    if not node.isActive():
        node.setActive(True)
        # print('activated')
    # if a_type == 'rotation':
    #     node.applyTorque(Vec3(*value))
    #     print(value)
    elif a_type == 'movement':
        node.applyCentralForce(Vec3(*value))
    else:
        raise("not supported")

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

def get_collisions(world, obj):
    contact_result = world.contactTest(obj)
    # contacts = []
    if contact_result.getNumContacts() > 0:
        for contact in contact_result.getContacts():
            point = contact.getManifoldPoint()
            
            c = {
                'obj': contact.getNode1(),
                'pos': point.getPositionWorldOnB(),
                'normal': point.getNormalWorldOnB()
            }
            # contacts.append(c)
            return c
    return None


def get_box(name, position, rotation=[0, 0, 0], scale=[1, 1, 1], color=[.78, .78, .78], static=False, mass=1):
    node = new_box_node(scale, static, mass)
    np = render.attachNewNode(node)
    np.setName(name)
    np.setHpr(*rotation)  
    np.setPos(*position)
    np.setCollideMask(DEFAULT_MASK)

    half_scale = [scale[0]/2, scale[1]/2, scale[2]/2]
    model = loader.loadModel('models/box.egg')
    model.setName(name)
    model.setTextureOff(1)
    model.setScale(*scale)
    model.setColor(*color)
    model.setPos(*[-h for h in half_scale])

    model.reparentTo(np)
    
    return np, model

def flatten_model(model):
    """Flatten the model's parameters into a 1D tensor."""
    flat_params = []
    for param in model.parameters():
        flat_params.append(param.view(-1))
    return torch.cat(flat_params)

def unflatten_model(model, flat_tensor):
    """Restore the model's parameters from a 1D tensor without affecting gradients."""
    offset = 0
    with torch.no_grad():  # Disable gradient tracking
        for param in model.parameters():
            param_size = param.numel()
            param_data = flat_tensor[offset:offset + param_size]
            param.copy_(param_data.view_as(param))  # Copy the data into the parameter
            offset += param_size


# def rotate_body(body, angles):
#     current_hpr = body.getHpr()
#     new_hpr = Point3(current_hpr)
#     new_hpr.setX(new_hpr.getX() + angles[0])
#     new_hpr.setY(new_hpr.getY() + angles[1])
#     new_hpr.setZ(new_hpr.getZ() + angles[2])
#     body.setHpr(new_hpr)

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