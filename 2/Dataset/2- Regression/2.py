import pandas as pd
import numpy as np

#pip install xlrd==1.2.0
#reading data
data = pd.read_excel('regression.xlsx' )
data.columns = ['x1' , 'x2' ,'x3'  , 'x4' ,'x5','y' ]

#preprocessing
Y = data['y']
X = data.drop('y' , axis = 1)
#normalize Attributes to 0-10 range
X = (X - X.min())/(X.max() - X.min())*10
Y = Y.to_numpy().reshape(414,1)
X = X.to_numpy()

#splittin data to train and test sets
x_train = X[:int(len(X)*0.7)]
y_train = Y[:int(len(X)*0.7)]
x_test = X[int(len(X)*0.7):]
y_test = Y[int(len(X)*0.7):]

def MSE(y_true , y_pred):
    """calculating MSE 

    Args:
        y_true ([numpy array m*1]): [true values]
        y_pred ([numpy array m*1]): [predicted values]

    Returns:
        [float]: [MSE value]
    """    
    return np.sum(np.power((y_true - y_pred) , 2))/(2*y_true.shape[0])

def KNN( X_train , Y_train , X_test , K = 1 , dist_type = 'Euc' ):
    """this functions execute KNN algorithm for regression
    
    Args:
        X_train ([numpy array m*n]): [features for training]
        Y_train ([numpy array m*1]): [targets for training]
        X_test ([numpy array z*n]): [features for predicting]
        K (int, optional): [num of neighbours]. Defaults to 1.
        dist_type (str, optional): [type of distance, it should be one of these: 'Euc','Mnhtn','Cosin']. Defaults to 'Euc'.

    Returns:
        [numpy array z*1]: [predicted values for X_test]
    """
    
    #for all types of distances the dists is a z*m matrix which z is num of test points and m is num of train points
    #and in the ith row of dists we have distances for ith test point from all of train points
    if dist_type == 'Euc':
        #calculating L2 norm for each test point from each train point , L2 norm is the Euclidean distance
        dists = np.linalg.norm(X_test[:,np.newaxis]-X_train ,axis = 2)
    
    if dist_type == 'Mnhtn':
        #calculating manhatan distance for each test point from each train point
        dists = np.sum(np.absolute(X_test[:,np.newaxis]-X_train) , axis = 2)
        
    if dist_type == 'Cosin':
        #calculating cosin distance for each test point from each train point
        norm_xtrain = np.linalg.norm(X_train,axis = 1 ).reshape(X_train.shape[0],1)
        norm_xtest = np.linalg.norm(X_test,axis = 1 ).reshape(X_test.shape[0],1)
        norms = norm_xtest @ norm_xtrain.T
        dists = 1 - ((X_test @ X_train.T) / norms)
    
    #choosing min distances
    min_dists_indices = np.argpartition(dists,K,axis = 1)[:,:K]
    #calculating predicted value
    Y_sum_min_distances = np.sum(Y_train[min_dists_indices] , axis = 1)
    y_pred = Y_sum_min_distances / K
    return y_pred

k = 9 
pred = KNN(x_train , y_train , x_test , k)
train_pred= KNN(x_train , y_train , x_train , k)
print('k: ',k)
print('test error: ', MSE(y_test , pred))
print('train error: ', MSE(y_train , train_pred))