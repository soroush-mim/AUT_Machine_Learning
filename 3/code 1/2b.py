import pandas as pd
import numpy as np


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

def predict(row , probs):
    """predict target class for row

    Args:
        row (pandas series): a sample in dataset
        probs (dict): [probabilities for NBC]

    Returns:
        [int]: [predicted class (0 to 3 which 0 is acc , 1 is unacc , 2 is vgood , 3 is good)]
    """    
    posteriors = []
    #calculating each postrior
    for i in['acc' , 'unacc' , 'vgood' , 'good']:
        p = probs[i]
        for feature in ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']:
            p *= probs[feature][row[feature]][i]
        posteriors.append(p)
        
    return posteriors.index(max(posteriors))
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

def confusion(y_true , y_pred):
    """a function for calculating confusion matrix for multiclass problems

    
    """    
    conf = np.zeros((4,4))
    for tr_la in range(4):
        for pr_la in range(4):
            true = y_true == tr_la
            pre = y_pred[true]
            conf[tr_la,pr_la]=np.count_nonzero(pre==pr_la)
        
    return conf.astype('int64')


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
probs = NBC(train , 1)

#adding predict column and a column for changing categorical target calss to numerical values
test = test.assign (predict  = test.apply(lambda row:predict(row , probs),axis = 1).values)
test =test.assign(true =  test.apply(lambda row:true(row) , axis = 1).values)

train = train.assign (predict  = train.apply(lambda row:predict(row , probs),axis = 1).values)
train =train.assign(true =  train.apply(lambda row:true(row) , axis = 1).values)

#changing predict column and true column type to numpt
y_test_true = test['true'].to_numpy()
y_test_pred = test['predict'].to_numpy()

y_train_true = train['true'].to_numpy()
y_train_pred = train['predict'].to_numpy()

#calculating confusion matrix
conf_test = confusion(y_test_true , y_test_pred)
conf_train = confusion(y_train_true , y_train_pred)

#calculating FN , FP , ‫‪sensitivity‬‬ , ‫‪specificity‬‬
#column sums  - diagonal
fp_test = conf_test.sum(axis = 0) - np.diag(conf_test)
fp_train = conf_train.sum(axis = 0) - np.diag(conf_train)
#row sums - diagonal
fn_test = conf_test.sum(axis = 1) - np.diag(conf_test)
fn_train = conf_train.sum(axis = 1) - np.diag(conf_train)
sens_test = np.diag(conf_test) / (np.diag(conf_test) + fn_test)
sens_train = np.diag(conf_train) / (np.diag(conf_train) + fn_train)
tn_test = -(fp_test + fn_test + np.diag(conf_test)) + conf_test.sum()
tn_train = - (fp_train + fn_train + np.diag(conf_train)) + conf_train.sum()
spec_test = (tn_test / (tn_test + fp_test)).reshape(4,)
spec_train = (tn_train / (tn_train + fp_train)).reshape(4,)

print('test data: ')
print()
print('confusion matrix:')
print(conf_test)
classes = ['acc' , 'unacc' , 'vgood' , 'good']
for i in range(4):
    print('positive class: ' , i , ' (' + classes[i] + ' )')
    print('FN : ' , fn_test[i])
    print('FP : ' , fp_test[i])
    print('‫‪sensitivity‬‬: ' , sens_test[i])
    print('‫‪specificity‬‬: ' , spec_test[i])
    print()
print()
print('train data: ')
print()
print('confusion matrix:')
print(conf_train)

for i in range(4):
    print('positive class: ' , i,' (' + classes[i] + ' )')
    print('FN : ' , fn_train[i])
    print('FP : ' , fp_train[i])
    print('‫‪sensitivity‬‬: ' , sens_train[i])
    print('‫‪specificity‬‬: ' , spec_train[i])
    print()
print()