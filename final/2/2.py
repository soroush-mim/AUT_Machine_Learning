import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from sklearn.cluster import DBSCAN

def purity(g_truth , labels):
    """a function for calculate purity

    Args:
        g_truth ([numpy array]): [real class labels]
        labels ([numpy array]): [cluster labels]

    Returns:
        [int]: [purity]
    """    
    t=0
    for i in np.unique(labels):
        if i !=-1:
            counts = np.unique(g_truth[labels==i],return_counts=True)[1]
            t+=np.max(counts)
    
    return t/len(g_truth)

def dbscan_plot(df_name , eps , min_samples):
    """a function for performing dbscan on datasets
        and plotting them it also prints purity for dataset

    Args:
        df_name ([str]): [dataset name]
        eps ([float]): [eps for dbscan]
        min_samples ([int]): [min_samples for dbscan]
    """    
    #fixing column names based on dataset
    if df_name!='rings':
        columns = ['x' , 'y' , 'class']
    else:
        columns = ['class','x' , 'y','z']
    #reading dataset and converting to numpy
    df = pd.read_csv(df_name + '.txt' , sep = '\t' , header=None , names = columns)
    if df_name!='rings':
        train = df[columns[:-1]].to_numpy()
    else:
        train = df[columns[1:]].to_numpy()
    #performing dbscan
    clustering = DBSCAN(eps = eps , min_samples=min_samples).fit(train)
    
    #plotting
    if df_name == 'D31':
        labels = clustering.labels_
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
            # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = train[class_member_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

        plt.title('Estimated number of clusters: '+ str(len(np.unique(clustering.labels_))) + ', purity: ' + str(purity(df['class'].to_numpy() , clustering.labels_)))
        plt.savefig(df_name)
        plt.clf()
        
    elif df_name == 'rings':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(train[: , 0], train[: , 1], train[: , 2], marker='o' , c=clustering.labels_, cmap=plt.cm.Set1,edgecolor='k')
        handles = scatter.legend_elements()[0]
        labels = list(np.unique(clustering.labels_))
        labels = [x if x != -1 else 'noise' for x in labels]
        legend1 = ax.legend(handles, labels, title="clusters")
        plt.title('number of clusetrs: ' + str(len(np.unique(clustering.labels_)))+', purity: ' + str(purity(df['class'].to_numpy() , clustering.labels_)))
        plt.savefig(df_name)
        plt.clf()
    else:
        fig, ax = plt.subplots()
        scatter = ax.scatter(train[:,0], train[:, 1] ,c=clustering.labels_, cmap=plt.cm.Set1,edgecolor='k')
        handles = scatter.legend_elements()[0]
        labels = list(np.unique(clustering.labels_))
        labels = [x if x !=  -1 else 'noise' for x in labels]
        legend1 = ax.legend(handles, labels,loc="upper right", title="clusters",bbox_to_anchor=(1.1, 1))

        ax.add_artist(legend1)
        plt.title('number of clusetrs: ' + str(len(np.unique(clustering.labels_)))+', purity: ' + str(purity(df['class'].to_numpy() , clustering.labels_)))
        plt.savefig(df_name)
        plt.clf()


#for 2D datasets
datasets_conf = [('Compound' ,1.45 ,3) , ('pathbased',1.98,8) ,  ('spiral',2.5,5) , ('D31',.78,23) , ('rings' , 10 , 20)]

for df , eps , min_s in datasets_conf:
    dbscan_plot(df , eps , min_s)




