import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn import linear_model

def select_features(X_train , columns , trashhold):
    """select features from train set based on correlation

    Args:
        X_train ([dataframe]): [features for train]
        columns ([list]): [names of features sorted based on corr with target]
        trashhold ([float]): [trashhold for selecting features]

    Returns:
        [list]: [names of selected features]
    """    
    corr = X_train.corr()
    corr = corr.abs()
    featuer_names = []
    featuer_names.append(columns[0])
    i = 2
    for column in columns[1:]:
        flag = True
        for j in featuer_names:
            if corr[column].loc[j] > trashhold:
                flag = False
                break
        if flag:
            featuer_names.append(column)

    return featuer_names

#reading dataset
df = pd.read_csv('SeoulBikeData.csv')

target_col = 'Rented Bike Count'
categorical_cols = ['Seasons' , 'Holiday' , 'Functioning Day']
#changing cetegorical columns to numerical
df[categorical_cols] = df[categorical_cols].astype('category')
df[categorical_cols] = df[categorical_cols].apply(lambda x: x.cat.codes)
#dropping date column
df = df.drop(columns=['Date'])
#slecting target and features
target = df[target_col].to_numpy()
features = df.drop(columns=[target_col])
#normalizing features
for column in features.columns:
    features[column]=(features[column]-features[column].min())/(features[column].max()-features[column].min())

#initilizing kfold validation
kf = KFold(n_splits = 5, shuffle = True, random_state = 2)
#this dict is for saving score for each trashhold
score = {}
#this dict is for saving selected features based on different trashholds
selected = {}
#trashholds for selecting features
trashhold = [.35 , .4 , .5 ,  .75 ,.9 , 1]
for i in trashhold:
    score[i] = 0
#performing kfold for different trashholds
for train_index, test_index in kf.split(features):
    #splitting test and train
    X_train, X_test = features.iloc[train_index], features.iloc[test_index]
    y_train, y_test = target[train_index], target[test_index]
    new_df = df.iloc[train_index]
    #sorting feature names based on corr with target
    corr_target = new_df.corr()[target_col]
    corr_target = corr_target.abs()
    columns = corr_target.sort_values(ascending = False)[1:].index
    
    for i in trashhold:
        #selecting features
        selected_features = select_features(X_train , columns , i)
        selected[i] = selected_features
        new_X_train = X_train[selected_features]
        new_X_test = X_test[selected_features]
        #performing regression
        regr = linear_model.LinearRegression()
        regr.fit(new_X_train, y_train)
        y_pred = regr.predict(new_X_test)
        score[i]+= regr.score(new_X_test,y_test)
#calculating maen of scores
for i in score.keys():
    score[i]/=5
    
for i in score.keys():
    print('trashhold: ' , i , ' score: ' , score[i])
    print('selected features: ' , selected[i])
