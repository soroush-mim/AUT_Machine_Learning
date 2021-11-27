import pandas as pd
import numpy as np


def accuracy(y_true , y_predicted):
    """caculate num of correctly predicted samples / num of all samples
    
    class of samples should be 0 or 1
    Args:
        y_true ([numpy array n*1]): [true value of labels]
        y_predicted ([numpy array n*1]): [predicted value of labels]

    Returns:
        [float]: [accuracy of prediction]
    """
    
    difference = y_true - y_predicted
    return np.count_nonzero(difference == 0) / difference.shape[0]

def confusion_matrix1(y_true , y_predicted):
    """calculate confusion matrix and print it
        class of samples should be 0 or 1

    Args:
        y_true ([numpy array n*1]): [true value of labels]
        y_predicted ([numpy array n*1]): [predicted value of labels]
    """    
    # class 1 is posetive
    TP , TN , FP , FN = 0,0,0,0
    difference = y_true - y_predicted
    
    #true ->0 , predicted->1
    FP = np.count_nonzero(difference == -1)
    #true ->1 , predicted->0
    FN = np.count_nonzero(difference == 1)
    
    #smaples that has predicted correctly
    T = difference == 0
    #smaples that has predicted posetive
    P = y_predicted == 1
    #smaples that has predicted negative
    N = y_predicted == 0
    
    #smaples that has predicted correctly and posetive
    tp = T*P
    #smaples that has predicted correctly and negative
    tn = T*N
    
    TP = np.count_nonzero(tp)
    TN = np.count_nonzero(tn)
    
    print('class 1 is posetive')
    print('----------------------------')
    print('actual\predict|__0__1_')
    print('            0 |',TN,FP)
    print('            1 |',FN,TP)
    print('----------------------------')
    print()
    
def KNN( X_train , Y_train , X_test , K = 1 , dist_type = 'Euc' ):
    """this functions execute KNN algorithm

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
    #calculating prediction labels
    Y_sum_min_distances = np.sum(Y_train[min_dists_indices] , axis = 1)
    y_pred = Y_sum_min_distances > (K-1)/2
    
    return y_pred.astype('int64')

def K_fold(X , Y, K = 1 , dist_type = 'Euc' , folds_num = 10 ):
    """this function uses k-fold cross validation and KNN for predicting labes

    Args:
        X ([numpy array m*n]): [features]
        Y ([numpy array m*1]): [targets]
        K (int, optional): [num of neighbours for KNN]. Defaults to 1.
        dist_type (str, optional): [type of distance for KNN, it should be one of these: 'Euc','Mnhtn','Cosin']. Defaults to 'Euc'.
        folds_num (int, optional): [number of folds]. Defaults to 10.

    Returns:
        [numpy array m*1]: [predicted labels for X with k-fold cross validation and KNN]
    """    
    #initializing predictions vector
    Y_pred = np.zeros((Y.shape[0] , 1))
    
    #picking folds ranges
    folds = np.linspace(0.0, 1.0, num=folds_num+1)
    
    #number of samples
    m = X.shape[0]
    
    #execuiting KNN for diffrent folds
    for i in range(folds_num):
        #calculating start and end of folds
        start = int(m * folds[i])
        end = int(m * folds[i+1])
        
        #calculating fold set and training set
        x_train = np.delete(X , np.arange(start,end) , axis = 0)
        x_test = X[start:end , :]
        y_train = np.delete(Y , np.arange(start,end))
        
        Y_pred[start:end] = KNN(x_train , y_train , x_test , K , dist_type).reshape(end - start,1)
 
    return Y_pred


#reading data
data = pd.read_csv('mammographic_masses.data', sep=",", header=None)
data.columns = ['BI-RADS' , 'Age' ,'Shape'  , 'Margin' ,'Density','Severity' ]

#preprocessing

#fill ? entries with NaN
data = data.replace(['?'] , np.nan)
#changeing type of data to numeric
data = data.apply(pd.to_numeric)
#drop raws with 2 or more null values
data = data.dropna(thresh=5)
#filling remaining null values with the mean of its columns
data = data.fillna((data.mean()+0.5).astype('int64'))

data = data.astype('float64')
#normalize Age Attribute to 0-5 range
data['Age'] = ((data['Age'] - data['Age'].min())/(data['Age'].max()-data['Age'].min()))*5


#splitting targets and features
X = data.drop('Severity' , axis = 1) 
Y = data['Severity']
X = X.to_numpy()
Y = Y.to_numpy().reshape(931,1)


#calculating 10-fold cross validation with KNN for k = 1,3,5,7,15,30 and distance type = Euc
max_acc = 0
for k in [1,3,5,7,15,30]:

    y_pred = K_fold(X , Y , k)
    print('K = ' , k)
    acc = accuracy(Y , y_pred)
    if acc > max_acc:
        max_acc = acc
        best_k = k
    print('accuracy: ' ,acc)
    confusion_matrix1(Y , y_pred)

print('best k is: ', best_k , 'with ',max_acc ,' accuracy')