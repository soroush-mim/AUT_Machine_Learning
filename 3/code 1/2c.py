import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def NBC(train , smoothing = 0):
    
    """calculate priors and likelihoods of training setfor naive bayes

    Returns:
        [dict]: [contains probabilities for priors and likelihoods]
    """    
    probs = {}
    tables = {}
    #calculating priors
    for class_ in train['class'].unique():
        probs[class_] = train['class'].value_counts()[class_]/train.shape[0]
        #we use this tables for calculating likelihoods
        tables[class_] = train[train['class'] == class_]
    
    #calculating likelihoods for each feature and its value and target class
    for feature in train.columns[:-1]:
        probs[feature]={}
        k = len(train[feature].unique())
        for value in train[feature].unique():
            probs[feature][value]={}
            for class_ in train['class'].unique():
                
                probs[feature][value][class_] = (tables[class_][feature].value_counts()[value] + smoothing)/(tables[class_].shape[0] + smoothing*k )
    
    return probs

def predict1(row , probs , class_ , t ):
    

    p = probs[class_]
    for feature in ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']:
        p *= probs[feature][row[feature]][class_]
        
    return int(p >= t)
    #acc = 0  , unacc = 1  , vgood = 2   , good = 3 
    
def true(row):
    """turn categorical target classes to numeric values

    """    
    if row['class'] == 'acc':
        return 0 
    if row['class'] == 'unacc':
        return 1
    if row['class'] == 'vgood':
        return 2
    
    return 3

def TPR_FPR(y_true , y_predicted):
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
    
    return [FP / (TN + FP),TP / (TP + FN)]


#loading data
data = pd.read_csv('car.data', sep=",", header=None)
data.columns = ['buying' , 'maint' ,'doors'  , 'persons' ,'lug_boot','safety','class']
#shuffling
data = data.sample(frac = 1).reset_index(drop=True)
data = data.astype('category')
#splitting data to train and test
train = data[:int(len(data)*0.7)]
test = data[int(len(data)*0.7):]

#calculating prior and likelihoods of train data
probs = NBC(train)

#adding a column for changing categorical target calss to numerical values

test =test.assign(true =  test.apply(lambda row:true(row) , axis = 1).values)

#changing predict column and true column type to numpy
y_test_true = test['true'].to_numpy()

#difining trashholds for NBC
trashhold = np.linspace(0,0.0005,300)
classes  =['acc' , 'unacc', 'vgood' , 'good']
rates = [[],[],[],[]]

#for each class we have to calculate diffrent TPRs and FPRs for diffrent trashholds
for i in range(4):
    y_true = (y_test_true == i).astype('int32')
    for t in trashhold:
        y_pred= test.apply(lambda row:predict1(row , probs , classes[i] , t),axis = 1).to_numpy()
        rates[i].append(TPR_FPR(y_true , y_pred))

for i in range(4):
    plt.plot(*zip(*rates[i]) , label = classes[i])
    
plt.legend()
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.savefig('2c.png')