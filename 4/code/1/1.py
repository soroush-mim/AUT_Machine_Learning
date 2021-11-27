import numpy as np
import pandas as pd
from sklearn.svm import *
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import itertools

#reading data
DATASET = 'parkinsons.data' 
INDEX = 'name' 
TARGET = 'status'
df = pd.read_csv(DATASET)
df = df.set_index(INDEX)



def model_generator(model, params):
    """a functions that generates all possible models from given parameters

    Args:
        model ([sk_learn obj]): [model from sklearn]
        params ([dict]): [parameters of model]
    """    
    
    def model_name_generator(model, param):
        #generate name for a given model and params
        name = model.__name__ + ''.join(map(str, param.items()))
        name = name.replace(', ', '=').replace("'", "").replace(")(", ", ")
        return name
    
    def dict_product(d):
        #generate params from given ranges
        lists = list(itertools.product(*d.values()))
        dicts = [dict(zip(d.keys(), l)) for l in lists]
        return dicts

    params = dict_product(params)
    models = {model_name_generator(model, param):model(**param) for param in params}

    return models

#models is a dict that for different kernels and params has names and models
models = {
    **model_generator(SVC,
                      {'kernel': ['linear']}),#linear kernel
    
    **model_generator(SVC,
                      {'kernel': ['poly'],
                       'degree': [2,3,5,12],
                       'coef0': [1,2,5,10]}), #polynomial kernels 
          
    **model_generator(SVC,
                      {'kernel': ['rbf'],
                       'gamma': ['auto', 'scale',.001,.01 ,5 ,10]}), #RBF kernels
          
    **model_generator(SVC,
                      {'kernel': ['sigmoid'],
                       'coef0': [.001,4,10,15]}) #sigmoid kernels  
}

#splitting data to train and test (20% test - 80% train)
test_size = 0.25
train_x, test_x, train_y, test_y = train_test_split(df.drop(columns=[TARGET]), df[TARGET], test_size=test_size)
#difining metrics
metrics = [accuracy_score, f1_score]

#create a data frame which each row has model parameters and accuracy and f1 score dataframe is sorted based on accuracy
leaderboard = pd.DataFrame({
    metric.__name__:[metric(test_y, model.fit(train_x, train_y).predict(test_x)) \
        for model in models.values()] \
            for metric in metrics},\
                index=models.keys()).sort_values(metrics[0].__name__, ascending=False)

print(leaderboard.head(65))