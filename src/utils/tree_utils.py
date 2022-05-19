from typing import Dict
from domain.node import DecisionNode,LeafNode,Node
from utils.metric_utils import MetricsUtils
import numpy as np
from itertools import combinations
from domain.split_metadata import NumericalSplitMetadata,SplitMetadata,CategoricalSplitMetadata


class TreeUtils:
    """
    Utils class which is responsible for building a Tree for Decision Tree
    """
    @staticmethod
    def build_tree(dataset:np.ndarray,min_samples_split:int,max_depth:int, curr_depth:int=0)->Node:
        """
        Recursive function which builds the tree based in input dataset
        :rparam dataset: represents the input dataset
        :rparam min_sample: represent minimal amount of instance per node
        :rparam max_depth: represent the maximum amount of depth allowed for tree to grow
        :rparam curr_depth: represent tracker of depth while building the tree 

        :return builded node
        """
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        
        # split until stopping conditions are met
        if num_samples>= min_samples_split and curr_depth<= max_depth:
            # find the best split
            best_split:SplitMetadata = TreeUtils.split(dataset, num_features)
            # check if information gain is positive
            if best_split.information_gain > 0:
                left_subset,right_subset = best_split.split(dataset)
                # build the left tree
                left = TreeUtils.build_tree(left_subset,min_samples_split, max_depth, curr_depth+1)
                
                # build the right tree
                right = TreeUtils.build_tree(right_subset,min_samples_split, max_depth, curr_depth+1)
                
                # return decision node
                return DecisionNode(best_split.feature_index, best_split.threshold, left, right, best_split.information_gain)
        
        # compute leaf node
        leaf_value = MetricsUtils.calculate_leaf_value(Y)
        
        # return leaf node
        return LeafNode(value=leaf_value)
    
    @staticmethod
    def split(dataset:np.ndarray, num_features:int)->Dict[str,str]:
        """
        Function which will find the best split of dataset, based on features
        
        """
        best_split = SplitMetadata()
     
        # loop over all the features
        for feature_index in range(num_features):
            # get all possible values of a feature
            feature_values = dataset[:, feature_index]
            # based on the type of the features get the split
            current_split  = TreeUtils.split_categorical(dataset,feature_index)  if type(feature_values[0]) is str \
                else TreeUtils.split_numerical(dataset,feature_index)

            if current_split.information_gain > best_split.information_gain:
                best_split = current_split
        
        return best_split

    
    @staticmethod
    def split_numerical(dataset:np.ndarray,feature_index:int=None)->SplitMetadata:
        """
        Function which will best split dataset on numerical attributes
        """
        numerical_split = NumericalSplitMetadata()\
            .with_feature_index(feature_index)
        
        thresholds = np.unique(dataset[:, feature_index])
        for threshold in thresholds:

            left,right = numerical_split.split_with_threshold(dataset,threshold)
            # check if childs are not null
            if len(left)>0 and len(right)>0:
                y, left_y, right_y = dataset[:, -1], left[:, -1], right[:, -1]
                
                curr_info_gain = MetricsUtils.gini_gain(y, left_y, right_y)
          
                if curr_info_gain > numerical_split.information_gain:
                        numerical_split \
                            .with_threshold(threshold) \
                            .with_information_gain(curr_info_gain)

        return numerical_split


    def split_categorical(dataset,feature_index):
        """
        Function which will split  categorical attributes
        """
        categorical_split = CategoricalSplitMetadata()\
            .with_feature_index(feature_index)

        features = dataset[:, feature_index]
        unique_values = set(np.unique(features))
        y = dataset[:, -1]
        
        for combination in combinations(unique_values,len(unique_values)-1):
                base = list(combination)
                diff = list(unique_values.difference(combination))
                if len(base)>0 and len(diff)>0:
                        base_x = y[np.in1d(features,base)]
                        dif_x =  y[np.in1d(features,diff)] 
                        
                        current_info_gain= MetricsUtils.gini_gain(y,base_x,dif_x)

                        if current_info_gain > categorical_split.information_gain:
                                categorical_split\
                                    .with_information_gain(current_info_gain)\
                                    .with_threshold(diff) 
        return categorical_split

