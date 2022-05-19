
from classifiers.decision_forest import DecisionForest
from classifiers.random_forest import RandomForest
from classifiers.base_classifier import BaseClassifier
from typing import Iterable
from sklearn.metrics import classification_report,balanced_accuracy_score,f1_score,precision_score,recall_score,accuracy_score
from tabulate import tabulate
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from math import log2,sqrt
import sys

def print_metrics(y_true,y_predicted,title=""):
    """
    Util function which print the metrics of predictions
    """
    print("\n\n\n================================================================")
    print(f"\t{title}")
    balanced_acc = round(balanced_accuracy_score(y_true,y_predicted)*100,2)
    accuracy  = round(accuracy_score(y_true,y_predicted)*100,2)
    f1 = round(f1_score(y_true,y_predicted,average='weighted')*100,2)
    precision = round(precision_score(y_true,y_predicted, average='weighted')*100,2)
    recall =  round(recall_score(y_true,y_predicted,average='weighted')*100,2)
    tabulatet = tabulate([['balanced accuracy', balanced_acc], ['accuracy', accuracy],['f1', f1],['precision', precision], ['recall',recall]], headers=['Metric', 'Score'], tablefmt='orgtbl')
    print(tabulatet)
    # print(f"\n\n\t\tClassification report [{title}] ::" )
    # print(classification_report(y_true,y_predicted))
    print("================================================================")
    return round(accuracy_score(y_true,y_predicted)*100,2)

def train_and_predict(classifier_class,data_frame:pd.DataFrame,F:int,NT:int,x_test:Iterable):
    classifier = classifier_class(F,NT)
    classifier.fit(data_frame)
    return classifier.predict(x_test),classifier.features_importance()


if __name__ == "__main__":
    dataset = sys.argv[1]
    # init data
    df = pd.read_csv(f'./data/{dataset}.csv')
    x_train, x_test =  train_test_split(df, test_size=0.33, random_state=42)
    x_tr = x_train.iloc[:, :-1].values
    y_tr = x_train.iloc[:, -1].values.reshape(-1,1)
    x_te = x_test.iloc[:, :-1].values
    y_te = x_test.iloc[:, -1].values.reshape(-1,1)
    size = len(df.columns) -1

    classifiers = [DecisionForest,RandomForest]
    NTs = [1,10,25,50]
    

    for classifier in classifiers:
        Fs = None #keep Fs as seet in order to avoid duplicates
        model_name = None
        if classifier is DecisionForest:
            Fs = [int(size/4),int(size/3),int((size*3)/4),random.randint(1, size)]
            model_name = 'DecisionForest'
        if classifier is RandomForest:
            Fs = [1,3,int(log2(size)+1),int(sqrt(size))]
            model_name = 'RandomForest'
        for nt in NTs:
            for f in Fs:
                    tt = f'{model_name} with  NT[{nt}] F[{f}]'
                    y_hat,features_importance = train_and_predict(classifier,x_train,f,nt,x_te)
                    accuracy = print_metrics(y_te,y_hat,title=tt)
                    print(features_importance)


            