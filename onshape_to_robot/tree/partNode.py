from typing import List
import numpy as np

class OnshapePart:
    def __init__(self, name: str, type: str, instanceId: str, documentId: str, elementId: str, documentMicroversion: str, partId: str, path: list[str], config: str, transform: np.matrix):
        self.type = type
        self.instanceId = instanceId
        self.documentId = documentId
        self.documentMicroversion = documentMicroversion
        self.partId = partId
        self.elementId = elementId
        self.path = path
        self.config = config
        self.transform = transform
        self.name = name
