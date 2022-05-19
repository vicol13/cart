from .base_classifier import BaseClassifier
from .classification_tree import ClassificationTree
from typing import List,Tuple,Iterable
from collections import defaultdict
import pandas as pd
import numpy as np 
from utils.voting_utils import modified_plurality

class RandomForest(BaseClassifier):
    """
    This class represent the implementation of RandomForest
    """
    def __init__(self,features: int, n_trees: int):
        self.__features = features
        self.__n_trees = n_trees
        self.__trees: List[Tuple[ClassificationTree,Tuple[int]]] = []
        self.__feature_importance = defaultdict(lambda: defaultdict(lambda: 0))

    
    def fit(self, df: pd.DataFrame) -> None:
        """
        Method which will generate random datasets, and train on them
        classification trees
        """
        x = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # columns =list(x.columns)
        name_to_index = {name:i for i,name in enumerate(x.columns)}
        self.__index_to_name = {i:name for i,name in enumerate(x.columns)}
        # column_indexes = [[ columns.index(column) for column in df.columns]]
        
        for i in range(self.__n_trees):
            sampled_data = self.__sample_dataset(x)
            
            for index,key in enumerate(sampled_data.columns):
                self.__feature_importance[name_to_index[key]][index]+=1
            indexes = tuple([name_to_index[column] for column in sampled_data.columns])
            tree = self.__build_tree(sampled_data.join(y))
            self.__trees.append((tree,indexes))
            #represnt small chunk of data
    
    def predict(self, x: Iterable) -> List[object]:
        """
        Method used for prediction collection of inputs
        """
        return [self.__predict(item) for item in x]


    def __predict(self, x: np.ndarray) -> object:
        """
        Method for prediction singel instance 
        :rtype: returns an object as we can have as labels string and integers
        """
        votes = []
        for tree,features in self.__trees: 
            vote = tree.predict([x[tuple([features])]])
            votes.append(vote[0])
        return modified_plurality(votes)


    def __sample_dataset(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Function which will extract from x, self.features amount of 
        random class and will try to build tree based on them
        """
        new_columns = np.random.permutation(x.columns)[:self.__features]
        # temp = x.sample(self.__features, axis='columns')
        temp = x[new_columns]
        return temp.sample(len(temp), axis='rows')
    
    def __build_tree(self,df:pd.DataFrame)->ClassificationTree:
        """
        Function which will build a tree 
        """
        sx = df.iloc[:, :-1]
        sy = df.iloc[:, - 1].values.reshape(-1, 1)
        sx = sx.values
        tree = ClassificationTree()
        tree.fit(sx, sy)

        return tree
    
    def features_importance(self):
        dd = defaultdict(lambda: 0)
        total = 0
        for tree,_ in self.__trees: 
            for feature_index,counter in tree.feature_importance().items():
                dd[feature_index] += counter
                total +=counter
        rr={}
        for key,value in dd.items():
            # dd[key] = dd[key]/total
            rr[self.__index_to_name[key]]= value/total
        return rr
