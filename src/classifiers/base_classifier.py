from typing import Protocol
import numpy as np 
from typing import Iterable
import pandas as pd
class BaseClassifier(Protocol):
    """
    Class which defines the contract for any classifier
    """

    def fit(self,df:pd.DataFrame)->None:
        """
        Should build the tree based on input data
        """
    
    def predict(self, x: Iterable)->np.ndarray:
        """
        Should return the predicted values for input
        """