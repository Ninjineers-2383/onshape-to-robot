import numpy as np
from apikey.client import Client

# You should fil creds.json with API key for your acount obtained from
# https://dev-portal.onshape.com/keys

# The first part in assembly should be the trunk
# The sub-parts should revolute around Z-axis

client = Client(logging=False)

# Document that should be parsed
# XXX: Argv-ize
documentId = '483c803918afc4d52e2647f0'

print('* Retrieving workspace ID ...')
document = client.get_document(documentId).json()
workspaceId = document['defaultWorkspace']['id']
print('- Workspace id: '+workspaceId)

print('* Retrieving elements in the document, searching for the assembly...')
elements = client.list_elements(documentId).json()
assemblyId = None
for element in elements:
    if element['type'] == 'Assembly':
        print("- Found assembly, id: "+element['id']+', name: "'+element['name']+'"')
        assemblyId = element['id']

if assemblyId == None:
    print("! Unable to find assembly in this document")
    exit(1)

print('* Retrieving assembly')
assembly = client.get_assembly(documentId, workspaceId, assemblyId).json()

# Collecting transform matrix for all the parts via occurences
root = assembly['rootAssembly']
occurrences = {}
for occurrence in root['occurrences']:
    path = occurrence['path']
    target = path[-1]
    matrix = np.reshape(occurrence['transform'], (4, 4))
    occurrences[target] = {
        'parent': path[0],
        'transform': matrix,
    }

# Collecting parts instance from assembly and subassemblies
parts = {}
def collectParts(instances):
    for instance in instances:
        if instance['type'] == 'Part':
            id = instance['id']
            occurrence = occurrences[id]
            instance['parent'] = occurrence['parent']
            instance['transform'] = occurrence['transform']
            parts[id] = instance

collectParts(root['instances'])
for asm in assembly['subAssemblies']:
    collectParts(asm['instances'])

print('* Getting assembly features, scanning for DOFs...')
relations = []
features = client.get_assembly_features(documentId, workspaceId, assemblyId).json()
for feature in features['features']:
    if feature['message']['name'][0:3] == 'dof':
        connectors = feature['message']['mateConnectors']
        a = connectors[0]['message']['parameters'][1]['message']['queries'][0]['message']['path'][0]
        b = connectors[1]['message']['parameters'][1]['message']['queries'][0]['message']['path'][0]
        relations.append([a,b])
print('- Found '+str(len(relations))+' DOFs')
    
print('* Building robot tree')
def collect(id, parent = None):
    part = {}
    part['id'] = id
    part['children'] = []
    for entry in relations:
        if entry[0] == id and entry[1] != parent:
            part['children'].append(collect(entry[1], id))
        if entry[1] == id and entry[0] != parent:
            part['children'].append(collect(entry[0], id))
    return part

trunk = root['instances'][0]['id']
robot = collect(trunk)

