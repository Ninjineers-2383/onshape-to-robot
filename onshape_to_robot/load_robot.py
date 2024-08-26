from collections import defaultdict
from dataclasses import dataclass
import hashlib
import math
import os
from sys import exit
import sys
from typing import Dict, List, Set, Tuple
import graphviz
import numpy as np
import uuid
from .onshape_api.client import Client
from .config import config, configFile
from colorama import Fore, Back, Style
from .tree import OnshapePart
from .robot_description import Link, RobotDescription
import json, pprint
from anytree import RenderTree
from stl.mesh import Mesh
from .stl_combine import load_mesh, apply_matrix, combine_meshes, save_mesh
from pathlib import Path
from anytree import AnyNode
from anytree.exporter import UniqueDotExporter

# OnShape API client
workspaceId = None
client = Client(logging=False, creds=configFile)
client.useCollisionsConfigurations = config["useCollisionsConfigurations"]

def load_assemby_json() -> dict:
    with open('assembly_cache.json', 'r') as f:
        return json.load(f)
    # If a versionId is provided, it will be used, else the main workspace is retrieved
    if config["versionId"] != "":
        print(
            "\n"
            + Style.BRIGHT
            + "* Using configuration version ID "
            + config["versionId"]
            + " ..."
            + Style.RESET_ALL
        )
    elif config["workspaceId"] != "":
        print(
            "\n"
            + Style.BRIGHT
            + "* Using configuration workspace ID "
            + config["workspaceId"]
            + " ..."
            + Style.RESET_ALL
        )
        workspaceId = config["workspaceId"]
    else:
        print("\n" + Style.BRIGHT + "* Retrieving workspace ID ..." + Style.RESET_ALL)
        document = client.get_document(config["documentId"]).json()
        workspaceId = document["defaultWorkspace"]["id"]
        print(Fore.GREEN + "+ Using workspace id: " + workspaceId + Style.RESET_ALL)

    # Now, finding the assembly, according to given name in configuration, or else the first possible one
    print(
        "\n"
        + Style.BRIGHT
        + "* Retrieving elements in the document, searching for the assembly..."
        + Style.RESET_ALL
    )
    if config["versionId"] != "":
        elements = client.list_elements(
            config["documentId"], config["versionId"], "v"
        ).json()
    else:
        elements = client.list_elements(config["documentId"], workspaceId).json()
    assemblyId = None
    assemblyName = ""
    for element in elements:
        if element["type"] == "Assembly" and (
            config["assemblyName"] is False or element["name"] == config["assemblyName"]
        ):
            print(
                Fore.GREEN
                + "+ Found assembly, id: "
                + element["id"]
                + ', name: "'
                + element["name"]
                + '"'
                + Style.RESET_ALL
            )
            assemblyName = element["name"]
            assemblyId = element["id"]

    if assemblyId == None:
        print(
            Fore.RED + "ERROR: Unable to find assembly in this document" + Style.RESET_ALL
        )
        exit(1)

    # Retrieving the assembly
    print(
        "\n"
        + Style.BRIGHT
        + '* Retrieving assembly "'
        + assemblyName
        + '" with id '
        + assemblyId
        + Style.RESET_ALL
    )
    if config["versionId"] != "":
        assembly = client.get_assembly(
            config["documentId"],
            config["versionId"],
            assemblyId,
            "v",
            configuration=config["configuration"],
        )
    else:
        assembly = client.get_assembly(
            config["documentId"],
            workspaceId,
            assemblyId,
            configuration=config["configuration"],
        )
    with open('assembly_cache.json', 'w') as f:
        json.dump(assembly, f)
    return assembly

def get_part_filename(did, mid, eid, partId, config) -> Path:
    m = hashlib.sha1()
    m.update(partId.encode('utf-8'))
    pid = m.hexdigest()

    key = (did, mid, eid, pid, config)

    if type(key) == tuple:
        key = '_'.join(list(key))
    fileName = 'part_stl' + '__' + key

    m = hashlib.sha1()
    m.update(fileName.encode('utf-8'))
    fileName = m.hexdigest()

    file = Path(sys.argv[1]) / "parts" / (fileName + '.stl')
    return file

def get_part_stl(part: OnshapePart) -> Mesh:
    file = get_part_filename(part.documentId, part.documentMicroversion, part.elementId, part.partId, part.config)

    try:
        return Mesh.from_file(file)
    except Exception as e:
        print(f"ERROR: could not load mesh for {part}")

def download_all_parts(assembly: dict):
    for part in assembly["parts"]:
        if part["partId"] == "":
            continue
        file = get_part_filename(part["documentId"], part["documentMicroversion"], part["elementId"], part["partId"], part["configuration"])

        if not os.path.exists(file):
            with open(file, "wb") as f:
                stl = client.part_studio_stl_no_cache(part["documentId"], part["documentMicroversion"], part["elementId"], part["partId"], part["configuration"]).content
                f.write(stl)
                print(f"Downloaded: {part["documentId"], part["documentMicroversion"], part["elementId"], part["partId"], part["configuration"]}")
                

def load_robot_assemblies(assembly: dict) -> Dict[Tuple[str], List[OnshapePart]]:
    root = assembly["rootAssembly"]

    # Create massive list of all instances in the whole doc
    instances = []
    instances.extend(root["instances"])
    for assembly in assembly["subAssemblies"]:
        instances.extend(assembly["instances"])

    instances_by_id = {}
    for instance in instances:
        instances_by_id[instance['id']] = instance
    
    assmeblies: Dict[Tuple[str], List[OnshapePart]] = {}

    hidden: List[Tuple[str]] = []
    
    for ocurrence in root["occurrences"]:
        path = ocurrence["path"]
        instanceId = ocurrence["path"][-1]
        instance = instances_by_id[instanceId]

        if assmeblies.get(tuple(path[:-1]), None) is None:
            assmeblies[tuple(path[:-1])] = []
        if instance["suppressed"] or ocurrence["hidden"]:
            hidden.append(tuple(ocurrence["path"]))
            continue

        part_path = ()
        hid = False
        for part in ocurrence["path"]:
            part_path += (part,)
            if part_path in hidden:
                hid = True
                break
        if hid:
            continue

        assmeblies[tuple(path[:-1])].append(OnshapePart(
            instance["name"],
            instance["type"],
            instance["id"],
            instance["documentId"],
            instance["elementId"],
            instance["documentMicroversion"],
            (instance["partId"] if instance["type"] == "Part" else ""),
            ocurrence["path"],
            instance["configuration"],
            np.matrix(np.reshape(ocurrence["transform"], (4, 4))),
        ))

    

    return assmeblies

def split_by_group(assemblies: Dict[Tuple[str], List[OnshapePart]], doc: Dict) -> Dict[Tuple[str], List[List[OnshapePart]]]:
    groups_by_ass: Dict[Tuple[str], List[List[OnshapePart]]] = {}
    fasteneds: List[Tuple[Tuple[str], Tuple[str]]] = []
    origin_fastends: List[Tuple[Tuple[str], Tuple[str]]] = []
    grouped_items: Dict[Tuple[str], Tuple[Tuple[str], int]] = {}
    for key, assembly in assemblies.items():
        i = 1
        groups_by_ass[key] = [[]]
        groups = groups_by_ass[key]
        if key == ():
            subassembly = doc["rootAssembly"]
        else:
            parent_key = key[:-1]
            parent = next((x for x in assemblies[parent_key] if x.instanceId == key[-1]), None)
            if parent is None:
                continue
                raise Exception()

            subassembly = next((x for x in doc["subAssemblies"] if x["elementId"] == parent.elementId), None)
            if subassembly is None:
                continue
                raise Exception()
        
        group_feats = []
        
        for feature in subassembly["features"]:
            if feature["featureType"] == "mateGroup":
                group_feats.append(feature)

        for pattern in subassembly["patterns"]:
            ids = list(pattern["seedToPatternInstances"].keys())
            sub_ids = []
            for id in ids:
                sub_ids.extend(pattern["seedToPatternInstances"][id])
            ids.extend(sub_ids)
            ocurrences = []
            for id in ids:
                ocurrences.append({'occurrence': [id]})
            group_feats.append({'featureData': {'occurrences': ocurrences}})

        for group in group_feats:
            if len(groups) == i:
                groups.append([])
            items: List[OnshapePart] = []
            for ocurrence in group["featureData"]["occurrences"]:
                item = next(x for x in assembly if x.instanceId == ocurrence["occurrence"][-1])
                items.append(item)
                # TODO!: Need to make sure items arent duplicated when moving into new groups
                #! Currently items cannot be in two groups
            # Item_path group_path group_index
            groups_to_combine: List[Tuple[Tuple[str], Tuple[str], int]] = []
            for item in items:
                gp = grouped_items.get(tuple(item.path), None)
                if gp is not None:
                    groups_to_combine.append((tuple(item.path), gp[0], gp[1]))
            if len(groups_to_combine) == 0:
                for item in items:
                    grouped_items[tuple(item.path)] = (key, i)
                groups[i].extend(items)
                i += 1
            else:
                root = groups_to_combine[0]
                root_group = groups_by_ass[root[1]][root[2]]
                for other in groups_to_combine[1:]:
                    grouped_items[other[0]] = (root[1], root[2])
                    other_group = groups_by_ass[other[1]][other[2]]
                    root_group.extend(other_group)
                    other_group.clear()
            

        fastened_feats = []
        for feature in subassembly["features"]:
            if feature["featureType"] == "mate" and feature["featureData"]["mateType"] == "FASTENED":
                fastened_feats.append(feature)

        for fastened in fastened_feats:
            if len(fastened["featureData"]["matedEntities"]) != 2 or len(fastened["featureData"]["matedEntities"][0]["matedOccurrence"]) == 0:
                continue
            child_key = key + tuple(fastened["featureData"]["matedEntities"][0]["matedOccurrence"])

            if len(fastened["featureData"]["matedEntities"][1]["matedOccurrence"]) == 0:
                parent_key = key
                print(f"Origin mate with parent {key}")
                origin_fastends.append((parent_key, child_key))
            else:
                parent_key = key + tuple(fastened["featureData"]["matedEntities"][1]["matedOccurrence"])
                fasteneds.append((parent_key, child_key))
            


    for key, assembly in assemblies.items():
        for part in assembly:
            found = False
            for group in groups_by_ass[key]:
                for g_part in group:
                    if tuple(part.path) == tuple(g_part.path):
                        found = True
                        continue
            if not found:
                groups_by_ass[key].append([part])

    root_group = groups_by_ass[()][0]

    for fasten in origin_fastends + fasteneds:
        parent_group = None
        parent_group_idx = -1
        parent_group_key = None
        child_group = None
        child_group_idx = -1
        child_group_key = None

        if fasten[0] == ():
            parent_group = root_group
            parent_group_key = ()
            parent_group_idx = 0

        for key, groups in groups_by_ass.items():
            for idx, group in enumerate(groups):
                for onshape_part in group:
                    if tuple(onshape_part.path) == fasten[0]:
                        if parent_group != None:
                            print(f"Found {onshape_part.path} in multiple places!!!")
                        parent_group = group
                        parent_group_idx = idx
                        parent_group_key = key
                    if tuple(onshape_part.path) == fasten[1]:
                        if child_group != None:
                            print(f"Found {onshape_part.path} in multiple places!!!")
                        child_group = group
                        child_group_idx = idx
                        child_group_key = key

        if child_group is None:
            print(f"Error: Could not find child group {fasten[1]}")
            continue

        if parent_group is None:
            print(f"Error: Could not find parent group {fasten[0]}")
            continue

        if child_group == root_group:
            parent_group, child_group = child_group, parent_group
            parent_group_idx, child_group_idx = child_group_idx, parent_group_idx
            parent_group_key, child_group_key = child_group_key, parent_group_key
            print("switched parent and child to preserve root")

        if parent_group == child_group:
            print("Fasten between items in the same group")
            continue

        parent_group.extend(child_group)
        child_group.clear()
        del groups_by_ass[child_group_key][child_group_idx]

    return groups_by_ass



def group_to_stl_name(key: Tuple[str], index: int) -> str:
    return os.path.join(sys.argv[1], "_".join(key).replace('/', '') + "_" + str(index) + "_combined.stl")

def group_to_stl(key: Tuple[str], index: int, group: List[OnshapePart]):
    root = None
    for node in group:
        if node.type == "Assembly":
            continue
        
        print(node.name)

        mesh = get_part_stl(node)

        if mesh is None:
            continue

        apply_matrix(mesh, node.transform)
        root = combine_meshes(root, mesh)
    
    if root is None:
        return
    path = group_to_stl_name(key, index)
    save_mesh(root, path)
    print(f"Writing {path}")

@dataclass
class JointCoordinateSystem:
    xAxis: Tuple[float, float, float]
    yAxis: Tuple[float, float, float]
    zAxis: Tuple[float, float, float]
    origin: Tuple[float, float, float]

@dataclass
class JointInfo:
    parentCs: JointCoordinateSystem
    childCs: JointCoordinateSystem


@dataclass
class JointConfig:
    parent_key: Tuple[str]
    parent_group_idx: int
    parent_transform: np.matrix
    child_key: Tuple[str]
    child_group_idx: int
    child_transform: np.matrix
    info: JointInfo
    type: str

def generate_joints_config(groups_by_sub: Dict[Tuple[str], List[List[OnshapePart]]], doc: dict) -> List[JointConfig]:
    joint_configs: List[JointConfig] = []
    for key, _ in groups_by_sub.items():
        if key == ():
            subassembly = doc["rootAssembly"]
        else:
            parent: OnshapePart = None
            for new_key, groups in groups_by_sub.items():
                if parent is not None:
                    break
                for group in groups:
                    if parent is not None:
                        break
                    for item in group:
                        if parent is not None:
                            break
                        if item.instanceId == key[-1]:
                            parent = item

            if parent is None:
                print(f"Error: Could not find assembly instance {key}")
                continue
                raise Exception()

            subassembly = next((x for x in doc["subAssemblies"] if x["elementId"] == parent.elementId), None)
            if subassembly is None:
                raise Exception()
        
        joints = [feat for feat in subassembly["features"] if feat["featureType"] == "mate" and feat["featureData"]["mateType"] in ["REVOLUTE", "SLIDER"]]
        for joint in joints:
            parent = key + tuple(joint["featureData"]["matedEntities"][1]["matedOccurrence"])
            child = key + tuple(joint["featureData"]["matedEntities"][0]["matedOccurrence"])

            parent_key: Tuple[str] = None
            parent_group_idx: int = -1
            parent_part: OnshapePart = None
            child_key: Tuple[str] = None
            child_group_idx: int = -1
            child_part: OnshapePart = None


            for new_key, groups in groups_by_sub.items():
                for idx, group in enumerate(groups):
                    for onshape_part in group:
                        if tuple(onshape_part.path) == parent:
                            if parent_key != None:
                                print(f"Found {onshape_part.path} in multiple places!!!")
                            parent_key = new_key
                            parent_group_idx = idx
                            parent_part = onshape_part
                        if tuple(onshape_part.path) == child:
                            if child_key != None:
                                print(f"Found {onshape_part.path} in multiple places!!!")
                            child_key = new_key
                            child_group_idx = idx
                            child_part = onshape_part

            if parent_key is None:
                raise Exception()
            if child_key is None:
                raise Exception()
            
            joint_configs.append(JointConfig(
                parent_key,
                parent_group_idx,
                parent_part.transform,
                child_key,
                child_group_idx,
                child_part.transform,
                JointInfo(
                    JointCoordinateSystem(
                        tuple(joint["featureData"]["matedEntities"][1]["matedCS"]["xAxis"]),
                        tuple(joint["featureData"]["matedEntities"][1]["matedCS"]["yAxis"]),
                        tuple(joint["featureData"]["matedEntities"][1]["matedCS"]["zAxis"]),
                        tuple(joint["featureData"]["matedEntities"][1]["matedCS"]["origin"]),
                    ),
                    JointCoordinateSystem(
                        tuple(joint["featureData"]["matedEntities"][0]["matedCS"]["xAxis"]),
                        tuple(joint["featureData"]["matedEntities"][0]["matedCS"]["yAxis"]),
                        tuple(joint["featureData"]["matedEntities"][0]["matedCS"]["zAxis"]),
                        tuple(joint["featureData"]["matedEntities"][0]["matedCS"]["origin"]),
                    )
                ),
                joint["featureData"]["mateType"]))
    return joint_configs

def get_T_part_mate(jointCoords: JointCoordinateSystem):
    T_part_mate = np.eye(4)
    T_part_mate[:3, :3] = np.stack(
        (
            np.array(jointCoords.xAxis),
            np.array(jointCoords.yAxis),
            np.array(jointCoords.zAxis),
        )
    ).T
    T_part_mate[:3, 3] = jointCoords.origin

    return T_part_mate

@dataclass(unsafe_hash=True)
class RenderLink:
    key: Tuple[str]
    idx: int

@dataclass(unsafe_hash=True)
class RenderJoint:
    parent: Link
    child: Link
    idx: int


def render_tree(joints: List[JointConfig]):
    dot = graphviz.Digraph('links')

    links: Set[RenderLink] = set()
    joints_s: Set[RenderJoint] = set()

    for idx, joint in enumerate(joints):
        parent = RenderLink(joint.parent_key, joint.parent_group_idx)
        child = RenderLink(joint.child_key, joint.child_group_idx)
        links.add(parent)
        links.add(child)
        joints_s.add(RenderJoint(parent, child, idx))

    for link in links:
        name = "_".join(link.key) + "_" + str(link.idx)
        print(f"adding node {name}")
        dot.node(name)

    for joint in joints_s:
        dot.edge("_".join(joint.parent.key) + "_" + str(joint.parent.idx),
                 "_".join(joint.child.key) + "_" + str(joint.child.idx),
                 str(joint.idx))
        
    dot.save()
    print(dot.render())

ONSHAPE_TO_URDF_JOINTS = {"REVOLUTE": "continuous", "SLIDER": "prismatic"}

def generate_xml(groups_by_sub: Dict[Tuple[str], List[List[OnshapePart]]], joint_configs: List[JointConfig], robot: RobotDescription):
    # convert joint_configs into tree order
    root_node = AnyNode(name="_0")
    nodes: Dict[str, AnyNode] = {"_0": root_node}

    joints_to_add: List[JointConfig] = []
    joints_to_add.extend(joint_configs)
    i = 0
    while i < len(joints_to_add):
        joint = joints_to_add[i]
        parent_name = "_".join(joint.parent_key) + "_" + str(joint.parent_group_idx)
        child_name = "_".join(joint.child_key) + "_" + str(joint.child_group_idx)

        if parent_name not in nodes:
            i += 1
            joints_to_add.append(joint)
            continue
        
        if child_name in nodes:
            raise Exception("Joints do not represent a tree structure")
        
        nodes[child_name] = AnyNode(name=child_name, joint=joint, parent=nodes[parent_name])

        i += 1

    names = config['linkRemaps']

    root_link = robot.add_link(names['_0'])
    root_link.add_mesh('visual', np.identity(4), group_to_stl_name((), 0))

    joints_in_tree: List[JointConfig] = [joint.joint for joint in root_node.descendants]

    render_tree(joints_in_tree)

    for idx, node in enumerate(root_node.descendants):
        joint: JointConfig = node.joint
        parent_name = "_".join(joint.parent_key) + "_" + str(joint.parent_group_idx)
        child_name = "_".join(joint.child_key) + "_" + str(joint.child_group_idx)

        link = robot.add_link(names[child_name])
        joint_e = robot.add_joint(names[parent_name] + "_to_" + names[child_name], ONSHAPE_TO_URDF_JOINTS[joint.type])
        
        if joint.type == "SLIDER":
            joint_e.set_limits(0, 0, 0, 0)

        origin_to_joint = joint.parent_transform @ get_T_part_mate(joint.info.parentCs)
        child_transform = np.linalg.inv(get_T_part_mate(joint.info.parentCs)) @ np.linalg.inv(joint.parent_transform)

        grand_parent = node.parent
        if grand_parent is root_node:
            origin_to_prev = np.eye(4)
        else:
            origin_to_prev = grand_parent.joint.parent_transform @ get_T_part_mate(grand_parent.joint.info.parentCs)

        parent_transform = np.linalg.inv(origin_to_prev) @ origin_to_joint

        joint_e.set_origin(parent_transform)
        joint_e.set_parent(names[parent_name])
        joint_e.set_child(names[child_name])
        
        link.add_mesh('visual', child_transform, group_to_stl_name(joint.child_key, joint.child_group_idx))

        

    with open('robot.urdf', 'wb') as f:
        f.write(robot.dump())

# # Finds a (leaf) instance given the full path, typically A B C where A and B would be subassemblies and C
# # the final part


# def findInstance(path, instances=None):
#     global assembly

#     if instances is None:
#         instances = assembly["rootAssembly"]["instances"]

#     for instance in instances:
#         if instance["id"] == path[0]:
#             if len(path) == 1:
#                 # If the length of remaining path is 1, the part is in the current assembly/subassembly
#                 return instance
#             else:
#                 # Else, we need to find the matching sub assembly to find the proper part (recursively)
#                 d = instance["documentId"]
#                 m = instance["documentMicroversion"]
#                 e = instance["elementId"]
#                 for asm in assembly["subAssemblies"]:
#                     if (
#                         asm["documentId"] == d
#                         and asm["documentMicroversion"] == m
#                         and asm["elementId"] == e
#                     ):
#                         return findInstance(path[1:], asm["instances"])

#     print(Fore.RED + "Could not find instance for " + str(path) + Style.RESET_ALL)


# # Collecting occurrences, the path is the assembly / sub assembly chain
# occurrences = {}
# for occurrence in root["occurrences"]:
#     occurrence["assignation"] = None
#     occurrence["instance"] = findInstance(occurrence["path"])
#     occurrence["transform"] = np.matrix(np.reshape(occurrence["transform"], (4, 4)))
#     occurrence["linkName"] = None
#     occurrences[tuple(occurrence["path"])] = occurrence

# # Gets an occurrence given its full path


# def getOccurrence(path):
#     return occurrences[tuple(path)]


# # Assignations are pieces that will be in the same link. Note that this is only for top-level
# # item of the path (all sub assemblies and parts in assemblies are naturally in the same link as
# # the parent), but other parts that can be connected with mates in top assemblies are then assigned to
# # the link
# assignations = {}

# # Frames (mated with frame_ name) will be special links in the output file allowing to track some specific
# # manually identified frames
# frames = defaultdict(list)


# def assignParts(root, parent):
#     assignations[root] = parent
#     for occurrence in occurrences.values():
#         if occurrence["path"][0] == root:
#             occurrence["assignation"] = parent


# from .features import init as features_init, getLimits

# features_init(client, config, root, workspaceId, assemblyId)


# def get_T_part_mate(matedEntity: dict):
#     T_part_mate = np.eye(4)
#     T_part_mate[:3, :3] = np.stack(
#         (
#             np.array(matedEntity["matedCS"]["xAxis"]),
#             np.array(matedEntity["matedCS"]["yAxis"]),
#             np.array(matedEntity["matedCS"]["zAxis"]),
#         )
#     ).T
#     T_part_mate[:3, 3] = matedEntity["matedCS"]["origin"]

#     return T_part_mate


# # First, features are scanned to find the DOFs. Links that they connects are then tagged
# print(
#     "\n"
#     + Style.BRIGHT
#     + "* Getting assembly features, scanning for DOFs..."
#     + Style.RESET_ALL
# )
# trunk = None
# relations = {}
# features = []
# for sub in assembly["subAssemblies"]
# features = root["features"]
# for feature in features:
#     if feature["featureType"] == "mateConnector":
#         name = feature["name"]
#         if name[0:5] == "link_":
#             name = name[5:]
#             occurrences[(feature["featureData"]["occurrence"][0],)]["linkName"] = name
#     else:
#         if feature["suppressed"]:
#             continue

#         data = feature["featureData"]

#         if (
#             "matedEntities" not in data
#             or len(data["matedEntities"]) != 2
#             or len(data["matedEntities"][0]["matedOccurrence"]) == 0
#             or len(data["matedEntities"][1]["matedOccurrence"]) == 0
#         ):
#             continue

#         child = data["matedEntities"][0]["matedOccurrence"][0]
#         parent = data["matedEntities"][1]["matedOccurrence"][0]

#         if data["name"].startswith("closing_"):
#             for k in 0, 1:
#                 matedEntity = data["matedEntities"][k]
#                 occurrence = matedEntity["matedOccurrence"][0]

#                 T_world_part = getOccurrence(matedEntity["matedOccurrence"])[
#                     "transform"
#                 ]
#                 T_part_mate = get_T_part_mate(matedEntity)
#                 T_world_mate = T_world_part * T_part_mate

#                 frames[occurrence].append([f"{data['name']}_{k+1}", T_world_mate])
#         elif data["name"].startswith("dof_"):
#             parts = data["name"].split("_")
#             del parts[0]
#             data["inverted"] = False
#             if parts[-1] == "inv" or parts[-1] == "inverted":
#                 data["inverted"] = True
#                 del parts[-1]
#             name = "_".join(parts)
#             if name == "":
#                 print(
#                     Fore.RED
#                     + "ERROR: a DOF dones't have any name (\""
#                     + data["name"]
#                     + '" should be "dof_...")'
#                     + Style.RESET_ALL
#                 )
#                 exit()

#             limits = None
#             if data["mateType"] == "REVOLUTE" or data["mateType"] == "CYLINDRICAL":
#                 if "wheel" in parts or "continuous" in parts:
#                     jointType = "continuous"
#                 else:
#                     jointType = "revolute"

#                 if not config["ignoreLimits"]:
#                     limits = getLimits(jointType, data["name"])
#             elif data["mateType"] == "SLIDER":
#                 jointType = "prismatic"
#                 if not config["ignoreLimits"]:
#                     limits = getLimits(jointType, data["name"])
#             elif data["mateType"] == "FASTENED":
#                 jointType = "fixed"
#             else:
#                 print(
#                     Fore.RED
#                     + 'ERROR: "'
#                     + name
#                     + '" is declared as a DOF but the mate type is '
#                     + data["mateType"]
#                     + ""
#                 )
#                 print(
#                     "       Only REVOLUTE, CYLINDRICAL, SLIDER and FASTENED are supported"
#                     + Style.RESET_ALL
#                 )
#                 exit(1)

#             # We compute the axis in the world frame
#             matedEntity = data["matedEntities"][0]
#             T_world_part = getOccurrence(matedEntity["matedOccurrence"])["transform"]

#             # jointToPart is the (rotation only) matrix from joint to the part
#             # it is attached to
#             T_part_mate = get_T_part_mate(matedEntity)

#             if data["inverted"]:
#                 if limits is not None:
#                     limits = (-limits[1], -limits[0])

#                 # Flipping the joint around X axis
#                 flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
#                 T_part_mate[:3, :3] = T_part_mate[:3, :3] @ flip

#             T_world_mate = T_world_part * T_part_mate

#             limitsStr = ""
#             if limits is not None:
#                 limitsStr = (
#                     "["
#                     + str(round(limits[0], 3))
#                     + ": "
#                     + str(round(limits[1], 3))
#                     + "]"
#                 )
#             print(
#                 Fore.GREEN
#                 + "+ Found DOF: "
#                 + name
#                 + " "
#                 + Style.DIM
#                 + "("
#                 + jointType
#                 + ")"
#                 + limitsStr
#                 + Style.RESET_ALL
#             )

#             if child in relations:
#                 print(Fore.RED)
#                 print(
#                     "Error, the relation "
#                     + name
#                     + " is connected a child that is already connected"
#                 )
#                 print("Be sure you ordered properly your relations, see:")
#                 print(
#                     "https://onshape-to-robot.readthedocs.io/en/latest/design.html#specifying-degrees-of-freedom"
#                 )
#                 print(Style.RESET_ALL)
#                 exit()

#             relations[child] = {
#                 "parent": parent,
#                 "worldAxisFrame": T_world_mate,
#                 "zAxis": np.array([0, 0, 1]),
#                 "name": name,
#                 "type": jointType,
#                 "limits": limits,
#             }

#             assignParts(child, child)
#             assignParts(parent, parent)

# print(
#     Fore.GREEN
#     + Style.BRIGHT
#     + "* Found total "
#     + str(len(relations))
#     + " DOFs"
#     + Style.RESET_ALL
# )

# # If we have no DOF
# if len(relations) == 0:
#     trunk = root["instances"][0]["id"]
#     assignParts(trunk, trunk)


# def connectParts(child, parent):
#     assignParts(child, parent)


# # Spreading parts assignations, this parts mainly does two things:
# # 1. Finds the parts of the top level assembly that are not directly in a sub assembly and try to assign them
# #    to an existing link that was identified before
# # 2. Among those parts, finds the ones that are frames (connected with a frame_* connector)
# changed = True
# while changed:
#     changed = False
#     for feature in features:
#         if feature["featureType"] != "mate" or feature["suppressed"]:
#             continue

#         data = feature["featureData"]

#         if (
#             len(data["matedEntities"]) != 2
#             or len(data["matedEntities"][0]["matedOccurrence"]) == 0
#             or len(data["matedEntities"][1]["matedOccurrence"]) == 0
#         ):
#             continue

#         occurrenceA = data["matedEntities"][0]["matedOccurrence"][0]
#         occurrenceB = data["matedEntities"][1]["matedOccurrence"][0]

#         if (occurrenceA not in assignations) != (occurrenceB not in assignations):
#             if data["name"].startswith("frame_"):
#                 # In case of a constraint naemd "frame_", we add it as a frame, we draw it if drawFrames is True
#                 name = "_".join(data["name"].split("_")[1:])
#                 if occurrenceA in assignations:
#                     frames[occurrenceA].append(
#                         [name, data["matedEntities"][1]["matedOccurrence"]]
#                     )
#                     assignParts(
#                         occurrenceB,
#                         {True: assignations[occurrenceA], False: "frame"}[
#                             config["drawFrames"]
#                         ],
#                     )
#                 else:
#                     frames[occurrenceB].append(
#                         [name, data["matedEntities"][0]["matedOccurrence"]]
#                     )
#                     assignParts(
#                         occurrenceA,
#                         {True: assignations[occurrenceB], False: "frame"}[
#                             config["drawFrames"]
#                         ],
#                     )
#                 changed = True
#             else:
#                 if occurrenceA in assignations:
#                     connectParts(occurrenceB, assignations[occurrenceA])
#                     changed = True
#                 else:
#                     connectParts(occurrenceA, assignations[occurrenceB])
#                     changed = True

# # Building and checking robot tree, here we:
# # 1. Search for robot trunk (which will be the top-level link)
# # 2. Scan for orphaned parts (if you add something floating with no mate to anything)
# #    that are then assigned to trunk by default
# # 3. Collect all the pieces of the robot tree
# print("\n" + Style.BRIGHT + "* Building robot tree" + Style.RESET_ALL)

# for childId in relations:
#     entry = relations[childId]
#     if entry["parent"] not in relations:
#         trunk = entry["parent"]
#         break
# trunkOccurrence = getOccurrence([trunk])
# print(
#     Style.BRIGHT + "* Trunk is " + trunkOccurrence["instance"]["name"] + Style.RESET_ALL
# )

# for occurrence in occurrences.values():
#     if occurrence["assignation"] is None:
#         print(
#             Fore.YELLOW
#             + "WARNING: part ("
#             + occurrence["instance"]["name"]
#             + ") has no assignation, connecting it with trunk"
#             + Style.RESET_ALL
#         )
#         child = occurrence["path"][0]
#         connectParts(child, trunk)

# # If a sub-assembly is suppressed, we also mark as suppressed the parts in this sub-assembly
# for occurrence in occurrences.values():
#     if not occurrence["instance"]["suppressed"]:
#         for k in range(len(occurrence["path"]) - 1):
#             upper_path = tuple(occurrence["path"][0 : k + 1])
#             if (
#                 upper_path in occurrences
#                 and occurrences[upper_path]["instance"]["suppressed"]
#             ):
#                 occurrence["instance"]["suppressed"] = True


# def collect(id):
#     part = {}
#     part["id"] = id
#     part["children"] = []
#     for childId in relations:
#         entry = relations[childId]
#         if entry["parent"] == id:
#             child = collect(childId)
#             child["axis_frame"] = entry["worldAxisFrame"]
#             child["z_axis"] = entry["zAxis"]
#             child["dof_name"] = entry["name"]
#             child["jointType"] = entry["type"]
#             child["jointLimits"] = entry["limits"]
#             part["children"].append(child)
#     return part


# tree = collect(trunk)
