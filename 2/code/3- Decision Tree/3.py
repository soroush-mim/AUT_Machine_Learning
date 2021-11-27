import numpy as np
import pandas as pd
from sklearn import tree


def accuracy(y_true , y_predicted):
    """caculate num of correctly predicted samples / num of all samples
    
    class of samples should be 0 or 1
    Args:
        y_true ([numpy array n*1]): [true value of labels]
        y_predicted ([numpy array n*1]): [predicted value of labels]

    Returns:
        [float]: [accuracy of prediction]
    """
    y_predicted = y_predicted.reshape(y_true.shape[0],1)
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
    y_predicted = y_predicted.reshape(y_true.shape[0],1)
    y_predicted = (y_predicted / 2)-1
    y_true = (y_true / 2)-1
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
    
    print('class 4 is posetive')
    print('----------------------------')
    print('actual\predict|__2__4_')
    print('            2 |',TN,FP)
    print('            4 |',FN,TP)
    print()
    

#reading data
train_data = pd.read_csv('breast-cancer-wisconsin-train.data', sep=",", header=None)
test_data = pd.read_csv('breast-cancer-wisconsin-test.data', sep=",", header=None)

#preprocessing
#add column names
train_data.columns = ['id' , 'x1','x2','x3','x4','x5','x6','x7','x8','x9','y']
test_data.columns = ['id' , 'x1','x2','x3','x4','x5','x6','x7','x8','x9','y']

#droping id column
train_data = train_data.drop('id' ,axis = 1)
test_data = test_data.drop('id' ,axis = 1)
#fill ? entries with NaN
train_data = train_data.replace(['?'] , np.nan)
#changeing type of data to numeric
train_data = train_data.apply(pd.to_numeric)
#fill null values with a new value
train_data = train_data.fillna(11)

#splitting targets and featurs
x_train = train_data.drop('y' ,axis = 1).to_numpy()
y_train = train_data['y'].to_numpy().reshape(499)
x_test = test_data.drop('y' , axis = 1).to_numpy()
y_test = test_data['y'].to_numpy().reshape(200,1)

#building model
clf = tree.DecisionTreeClassifier()
#fitting model on train set
clf = clf.fit(x_train, y_train)

y_predict = clf.predict(x_test)

confusion_matrix1(y_test , y_predict)

print('accuracy: ',accuracy(y_test , y_predict))