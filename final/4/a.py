import pandas as pd
import numpy as np

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
