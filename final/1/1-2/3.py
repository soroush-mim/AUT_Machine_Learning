import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from sklearn.cluster import KMeans

def purity(g_truth , labels):
    #a function for calculating purity
    #does not works good for imbalanced data
    class_ = []
    #finding must frequent class in each cluster
    for i in range(2):
        zero_num = np.count_nonzero(g_truth[labels==i]==0)
        one_num = np.count_nonzero(g_truth[labels==i]==1)
        if zero_num > one_num:
            class_.append(0)
        else:
            class_.append(1)
    
    return sum([np.count_nonzero(g_truth[labels==i]==class_[i]) for i in range(2)])/len(g_truth)
    

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

#performing kmeans
kmeans = KMeans(n_clusters=2 , random_state=0).fit(train_data)
#kmeans labels
labels = kmeans.labels_
#actual labels
g_truth = df['Class'].to_numpy()

print('purity: ',purity(g_truth , labels))