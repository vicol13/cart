import numpy as np
from typing import Protocol
from collections import defaultdict
from typing import DefaultDict
class Node(Protocol):
    """
    Class which represent base class for nodes
    """
    pass


class DecisionNode(Node):
    def __init__(self, feature_index:int=None, threshold:np.float32=None, left:"Node"=None, right:"Node"=None, info_gain:np.float32=None, type=None)->None:
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.type = type
    
    def feature_importance(self):
        kk = {}
        def compute(dd:dict,node:DecisionNode):
            if type(node.left) == DecisionNode:
                compute(dd,node.left)
            if type(node.right) == DecisionNode:
                compute(dd,node.right)
            if node.feature_index in dd:
                dd[node.feature_index]+=1
            else:
                dd[node.feature_index]=1
        compute(kk,self)
        return kk

        


class LeafNode(Node):
    def __init__(self,value)->None:
        self.value = value