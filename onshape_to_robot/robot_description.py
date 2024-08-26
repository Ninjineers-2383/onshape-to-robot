import math
from pathlib import Path
from xml.etree.ElementTree import fromstring, tostring, Element, register_namespace
import numpy as np

base_robot = """<?xml version="1.0"?>
<robot>
</robot>
"""

def isclose(x, y, rtol=1.e-5, atol=1.e-8):
    return abs(x-y) <= atol + rtol * abs(y)

def euler_angles_from_rotation_matrix(R):
    '''
    From a paper by Gregory G. Slabaugh (undated),
    "Computing Euler angles from a rotation matrix
    '''
    phi = 0.0
    if isclose(R[2,0],-1.0):
        theta = math.pi/2.0
        psi = math.atan2(R[0,1],R[0,2])
    elif isclose(R[2,0],1.0):
        theta = -math.pi/2.0
        psi = math.atan2(-R[0,1],-R[0,2])
    else:
        theta = -math.asin(R[2,0])
        cos_theta = math.cos(theta)
        psi = math.atan2(R[2,1]/cos_theta, R[2,2]/cos_theta)
        phi = math.atan2(R[1,0]/cos_theta, R[0,0]/cos_theta)
    return psi, theta, phi

class Link:
    def __init__(self, name: str):
        self.xml = Element('link', {'name': name})
        self.xml.append(Element(f'xacro:{name}'))

    def add_mesh(self, type: str, transform: np.matrix, mesh_name: str):
        top = Element(type)
        geom = Element("geometry")
        path = mesh_name.split('/')[-1]
        mesh = Element("mesh", {'filename': 'file:///home/henrylec/frc_ws/robot_cad/' + path})
        material = Element('material', {"name": "grey"})
        geom.append(mesh)
        top.append(material)
        top.append(geom)
        rpy = euler_angles_from_rotation_matrix(transform)
        origin = Element('origin', {
            'xyz': f"{transform.item((0, 3))} {transform.item((1, 3))} {transform.item((2, 3))}",
            'rpy': f"{rpy[0]} {rpy[1]} {rpy[2]}"
            })
        top.append(origin)
        self.xml.append(top)

class Joint:
    def __init__(self, name: str, type: str):
        self.xml = Element('joint', {'name': name, 'type': type})
        self.xml.append(Element('axis', {'xyz': '0 0 1'}))

    def set_origin(self, transform: np.matrix):
        rpy = euler_angles_from_rotation_matrix(transform[:3, :3])
        top = Element('origin', {
            'xyz': f"{transform.item((0, 3))} {transform.item((1, 3))} {transform.item((2, 3))}",
            'rpy': f"{rpy[0]} {rpy[1]} {rpy[2]}"})
        self.xml.append(top)

    def set_parent(self, parent: str):
        parent = Element('parent', {'link': parent})
        self.xml.append(parent)

    def set_child(self, child: str):
        child = Element('child', {'link': child})
        self.xml.append(child)

    def set_limits(self, upper: float, lower: float, effort: float, velocity: float):
        limit = Element('limit', {
            'lower': str(lower),
            'upper': str(upper),
            'effort': str(effort),
            'velocity': str(velocity)
        })
        self.xml.append(limit)

class RobotDescription:
    def __init__(self, name: str):
        self.xml = fromstring(base_robot.format(name=name))
        self.xml.attrib['xmlns:xacro'] = 'http://www.ros.org/wiki/xacro'
        color = Element('material', {'name': 'grey'})
        color.append(Element('color', {'rgba': '0.5 0.5 0.5 1'}))
        self.xml.append(color)

    def add_link(self, name: str) -> Link:
        link = Link(name)
        self.xml.append(link.xml)
        return link
    
    def add_joint(self, name: str, type: str) -> Joint:
        joint = Joint(name, type)
        self.xml.append(joint.xml)
        return joint

    def dump(self) -> str:
        return tostring(self.xml)
    
