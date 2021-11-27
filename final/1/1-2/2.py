import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from sklearn.cluster import KMeans

def normaliz(col):
    #normalize a feature range
    return (col - col.min())/(col.max() - col.min())

#preproccesing
#reading data
df = pd.read_csv('Shill Bidding Dataset.csv')
#feature selection
#dropping id and target columns
train_data = df.drop(columns = ['Record_ID' , 'Class' ,'Auction_ID' ,'Bidder_ID' ])
#normalizing Auction_Duration feature 
train_data['Auction_Duration'] = normaliz(train_data['Auction_Duration'])
train_data = train_data.to_numpy()

inertia = []
#performiing kmeans with different Ks
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(train_data)
    inertia.append(kmeans.inertia_)
    
#plotting error vs K
plt.plot(range(1,11) , inertia)
plt.grid()
plt.xticks(range(12))
plt.xlabel('K')
plt.ylabel('inertia')
plt.title('elbow method')
plt.savefig('1-2-2')