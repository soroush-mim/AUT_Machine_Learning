import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn import linear_model
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


#dict for saving scores for different alphas
score={}
alphas = [.9,.98  ,1 , 1.02 , 1.5]
#initilizing kfold
kf = KFold(n_splits = 5, shuffle = True, random_state = 2)
for i in alphas:
    score[i] = 0
for train_index, test_index in kf.split(features):
    
    X_train, X_test = features.iloc[train_index], features.iloc[test_index]
    y_train, y_test = target[train_index], target[test_index]
    
    for alpha in alphas:
        clf = linear_model.Lasso(alpha=alpha)
        clf.fit(X_train , y_train)
        score[alpha]+= clf.score(X_test , y_test)
        

for i in score.keys():
    score[i]/=5
    
print('scores: ')
for i in score.keys():
    print('alpha: ' , i , 'score: ' , score[i])

#choosing best alpha
t=0
for i in score.keys():
    if score[i]>t:
        t = score[i]
        best_alpha = i
        
clf = linear_model.Lasso(alpha=best_alpha)
clf.fit(features , target)

print('dropped featuers with lasso: ',list(features.columns[clf.coef_==0]))
