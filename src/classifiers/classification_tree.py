import numpy as np
from utils.tree_utils import TreeUtils
from domain.node import Node,LeafNode
from typing import List

class ClassificationTree:
    """
    Represent Single tree clasifier 
    """
    def __init__(self,min_samples_split:int=2, max_depth:int=5)->None:
        # initialize the root of the tree 
        self.__root = None
        
        # stopping conditions
        self.__min_samples_split = min_samples_split
        self.__max_depth = max_depth
        
    def fit(self, x:np.ndarray, y:np.ndarray):
        """
        Function which will build the tree via tuils
        """
        dataset = np.concatenate((x, y), axis=1)
        self.__root = TreeUtils.build_tree(dataset, self.__min_samples_split,self.__max_depth)
    
    def predict(self, x:np.ndarray)->np.ndarray:
        """
        Function which iterates over the input and predict each entry
        """ 
        return np.array([self.__predict(item, self.__root) for item in x])
    
    def __predict(self, x, tree:Node):
        """
        Recursively will parse the tree and return corresponding value for input 
        """ 
        if type(tree)==LeafNode: 
            return tree.value
        
        feature_val = x[tree.feature_index]
        if type(feature_val) == str:
            if feature_val in tree.threshold:
                return self.__predict(x, tree.left)
            else:
                return self.__predict(x, tree.right)
        else:
            if feature_val<=tree.threshold:
                return self.__predict(x, tree.left)
            else:
                return self.__predict(x, tree.right)
    

    def feature_importance(self):
        return self.__root.feature_importance()