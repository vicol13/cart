# Practical work 2 
## Supervised and experiential learning


This project represents the implementation of CART, Decision Forest and Random Forest which represent the practical assignment 2
of SEL MAI


```
📦src
 ┣ 📂classifiers                # folder with classifiers
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜base_classifier.py
 ┃ ┣ 📜classification_tree.py
 ┃ ┣ 📜decision_forest.py
 ┃ ┗ 📜random_forest.py
 ┣ 📂data                       # folder datasets
 ┃ ┣ 📜breast-cancer.csv
 ┃ ┣ 📜contact-lenses.csv
 ┃ ┣ 📜obesity.csv
 ┃ ┗ 📜wine.csv
 ┣ 📂domain                     # core clases used for building/splitting the tree
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜node.py
 ┃ ┗ 📜split_metadata.py
 ┣ 📂utils                      # util methods for building the tree
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜metric_utils.py
 ┃ ┣ 📜tree_utils.py
 ┃ ┗ 📜voting_utils.py
 ┣ 📜__init__.py
 ┣ 📜interpreter.py            # entry point for for project
 ┗ 📜main.ipynb

```


## Set up the project
1. create virtual environment
```shell
python3 -m venv venv/
```
2. enter virtual environment
```shell
source venv/bin/activate
```
3. install dependencies 
```shell
pip3 install -r requirements.txt
```
## Running the algorithm

1. enter the virtual environment
```shell
source venv/bin/activate
```

2. run the algorithm
```
python3 src/interpreter.py <data_set_name>
```
make sure <data_set_name> is in folder src/data