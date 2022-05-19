import numpy as np 

class MetricsUtils:

    @staticmethod
    def gini_index(y):
        ''' function to compute gini index '''
        
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
    

    @staticmethod
    def calculate_leaf_value(Y):
        ''' function to compute leaf node '''
        
        Y = list(Y)
        return max(Y, key=Y.count)
    
    @staticmethod
    def entropy(y):
        ''' function to compute entropy '''
        
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    @staticmethod
    def gini_gain( parent, l_child, r_child):
        ''' function to compute information gain '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
      
        return  MetricsUtils.gini_index(parent) - \
                (weight_l*MetricsUtils.gini_index(l_child) +\
                     weight_r*MetricsUtils.gini_index(r_child))
         