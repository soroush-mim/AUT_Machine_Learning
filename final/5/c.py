import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
#reading dataset
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
#replacing ? marks with nan values
df.replace({'?': np.nan},inplace =True)
#selecting target column
target_name = 'DEATH_EVENT'
target = df[target_name]
#selecting features
features = df.drop(columns = [target_name])
features = features.astype('float64')
#dropping rows with 4 or more null values
new_featurs = features.drop([213,238,273,299])
target = target.drop([213,238,273,299])
#filling null values in categorical columns with mode
for column in ['sex' ,  'diabetes' ]:
    new_featurs[column].fillna(int(new_featurs[column].mode()), inplace=True)
#filling null values in categorical columns with mean
for column in ['age' ,  'creatinine_phosphokinase' ,'creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine' ,'serum_sodium','time']:
    new_featurs[column].fillna(int(new_featurs[column].mean()), inplace=True)
#reset indices
new_featurs = new_featurs.reset_index(drop = True)
target = target.reset_index(drop=True)
#normalizing
for column in new_featurs.columns:
    new_featurs[column] = (new_featurs[column] - new_featurs[column].min())/(new_featurs[column].max() - new_featurs[column].min())

#initilizing base classifiers
clf1 = LogisticRegression()
clf2 = RandomForestClassifier(n_estimators=50)
clf3 = GaussianNB()
clf4 = DecisionTreeClassifier(max_depth=4)
clf5 = KNeighborsClassifier(n_neighbors=7)
clf6 = SVC(gamma=.1, kernel='rbf' , probability=True)

#initializing voting classifiers
eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2),
                                    ('gnb', clf3)],
                        voting='soft')
eclf2 = VotingClassifier(estimators=[('lr', clf1), ('dt', clf4),
                                    ('knn', clf5)],
                        voting='soft')
eclf3 = VotingClassifier(estimators=[('svm', clf6), ('lr', clf1),
                                    ('dt', clf4)],
                        voting='soft')

for i , eclf in enumerate([eclf1 , eclf2 , eclf3]):
    scores = cross_val_score(eclf, new_featurs, target, cv=5)
    print('score for ' , i+1, 'st', 'voting classifier: ' , scores.mean())