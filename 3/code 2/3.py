from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def predict(models , x):
    """this function calculates which class has the biggest probability

    Args:
        models ([type]): [description]
        x ([type]): [description]

    Returns:
        [type]: [description]
    """    
    x = x.reshape(1 , 784)
    max_prob = 0
    max_index = 0
    
    for i in range(10):
        if models[i].predict_proba(x)[0,1]>max_prob:
            max_prob = models[i].predict_proba(x)[0,1]
            max_index = i
            
    
    return max_index


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


def one_vs_all(x_train , y_train):
    models = []
    for positive_class in range(10):
    
        #seperaing positive labels
        y_train_positive = (y_train == positive_class).astype('uint8')
        #initializing model
        model = LogisticRegression( multi_class = 'ovr' , solver = 'lbfgs' , max_iter =500)
        print('starting training for positive class = ' , positive_class)
        #fitting model
        model.fit(x_train, y_train_positive)
        models.append(model)
        print('training for positive class = ' , positive_class , ' has ended')
    return models

    
def confusion(y_true , y_pred):
    conf = np.zeros((10,10))
    for tr_la in range(10):
        for pr_la in range(10):
            true = y_true == tr_la
            pre = y_pred[true]
            conf[tr_la,pr_la]=np.count_nonzero(pre==pr_la)
        
    return conf.astype('int64')
    
    
#fetching mnist dataset
mnist = fetch_openml('mnist_784')
x = mnist.data
y = mnist.target

#scaling
x = x/255

y = y.astype('uint8')
x = x.astype('float32')


#splitting dataset into test and train
x_train = x[:60000]
y_train = y[:60000]
x_test = x[60000:]
y_test = y[60000:]

#training models with one vs all
models = one_vs_all(x_train,y_train) 

#predicting lables for each sample
y_pred_test = np.array([predict(models , x) for x in x_test])
y_pred_train = np.array([predict(models , x) for x in x_train])

print('test accuracy: ',accuracy(y_test , y_pred_test))
print('train accuracy: ',accuracy(y_train , y_pred_train))



        
confusion_test = confusion(y_test,y_pred_test)
confusion_train = confusion(y_train,y_pred_train)
print('confusion matrix for train data: ')
print(confusion_train)
print()
print('confusion matrix for test data: ')
print(confusion_test)


random_ind = np.random.randint(0,10000,25)

plt.figure(figsize=(20,20))
for index, (image, label , pre) in enumerate(zip(x_test[random_ind], y_test[random_ind] , y_pred_test[random_ind])):
 
    plt.subplot(5, 5, index + 1)
    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
    plt.title('y_org: ' + str(label) + ' y_pre: ' +str(pre) , fontsize = 10)
    
plt.savefig('3b.png')