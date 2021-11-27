import numpy as np
import pandas as pd

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
#selecting rows with null values
null_rows = features[features.isnull().sum(axis=1)!=0]

print('number of null values in rows with them:')
print(null_rows.isnull().sum(axis=1))

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