import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import *
from sklearn.metrics import *
import itertools
from sklearn.svm import *
#reading data
DATASET = 'pima_indians_diabetes.csv'
TARGET = 'class'
df = pd.read_csv(DATASET)

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
    **model_generator(BaggingClassifier,
                      {'base_estimator':[SVC()],
                       'n_estimators':range(8,11)}),
          
    **model_generator(BaggingClassifier,
                      {'n_estimators':range(8,11)}),
      
    **model_generator(AdaBoostClassifier,
                      {}),
    
    **model_generator(GradientBoostingClassifier,
                      {'n_estimators': range(1, 100, 20)}),
}

#splitting data to train and test (20% test - 80% train)
test_size = 0.25
train_x, test_x, train_y, test_y = train_test_split(df.drop(columns=[TARGET]), df[TARGET], test_size=test_size)
#difining metrics
metrics = [accuracy_score]
#create a data frame which each row has model parameters and accuracy. dataframe is sorted based on accuracy
leaderboard_test = pd.DataFrame({
    metric.__name__:[metric(test_y, model.fit(train_x, train_y).predict(test_x)) \
        for model in models.values()] \
            for metric in metrics},\
                index=models.keys()).sort_values(metrics[0].__name__, ascending=False)



print('test accuracy: ')
print(leaderboard_test.to_string())
