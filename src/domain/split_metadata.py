import numpy as np
from typing import Tuple,List
from numbers import Number


class SplitMetadata:
    """
    """
    def __init__(self)->None:
        self.information_gain =  -float("inf")
    
    def with_feature_index(self,feature_index)->"SplitMetadata":
        self.feature_index = feature_index
        return self
    
    def split(dataset:np.ndarray)->Tuple[np.ndarray,np.ndarray]:
        raise NotImplementedError("Call on base class use child implementation")
    
    def with_information_gain(self,information_gain:float)->"SplitMetadata":
        self.information_gain = information_gain
        return self



class CategoricalSplitMetadata(SplitMetadata):
    """
    """
    def __init__(self)->None:
        super().__init__()
        self.threshold = None

    def with_threshold(self,threshold:List[str])->"CategoricalSplitMetadata":
        self.threshold = threshold
        return self

    def split(self,data):
        if self.feature_index is None and self.threshold is None : 
            raise ValueError("Before usege of split define [threshold] and [feature index]")
        
        left = lambda dataset: np.array([row for row in dataset if row[self.feature_index] in self.threshold])
        right = lambda dataset: np.array([row for row in dataset if row[self.feature_index] not in self.threshold])
       
        return left(data),right(data) 



class NumericalSplitMetadata(SplitMetadata):
    """
    
    """
    def __init__(self)->None:
        super().__init__()
        self.threshold = None

    def split(self,data:np.ndarray)->Tuple[np.ndarray,np.ndarray]:
        """
        Function which will split data into 2 sides based thresold and features_index define previously
        """
        
        return self.split_with_threshold(data,self.threshold)

    def split_with_threshold(self,data:np.ndarray,threshold:float)->Tuple[np.ndarray,np.ndarray]:
        """
        Function which will split data into 2 sides based input thresold and features_index define previously
        """
        if self.feature_index is None  and threshold is None: 
            raise ValueError("Before usege of split define [threshold] and [feature index]")
        
        left = lambda dataset: np.array([row for row in dataset if row[self.feature_index]<=threshold])
        right = lambda dataset: np.array([row for row in dataset if row[self.feature_index]>threshold])
        
        return left(data),right(data) 

    def with_threshold(self,threshold:Number)->"SplitMetadata":
        self.threshold = threshold
        return self
