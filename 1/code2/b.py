import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# a function for scaling features between -0.5 and 0.5
def scaling(x,d):
    """[x(i) := (x(i) - mean) / range]

    Args:
        x ([numpy array (m * d+1)]): [input matrix for scaling]
        d ([integer]): [degree of the polynomial model]

    Returns:
        [numpy array]: [scaled martix except the first colmun that is 1]
    """    
   
    # slicing the first column that is 1 from matrix
    temp = x[: , 1:]
    # range should not be zero (max - min != 0)
    # scaling
    temp = [(temp[:,i] - temp[:,i].mean())/(temp[:,i].max() - temp[:,i].min()) for i in range(d)]

    temp = np.array(temp)
    temp = temp.T

    return np.append(np.ones(( x.shape[0], 1)) , temp , axis = 1) #adding [1] column

# a function for adding polynomial features
def make_featurs(x , d = 1):
    """[adding polynomial features to the feature matrix]

    Args:
        x ([numpy array (m * 1)]): [raw features matrix]
        d (int, optional): [degree of the polynomial model]. Defaults to 1.

    Returns:
        [numpy array (m * d + 1)]: [features matrix with polynomial features added]
    """    
    
    #adding [1] column
    x = np.append(np.ones(x.shape) , x , axis = 1)

    # adding polynomial features
    for i in range(d-1):
        x = np.append(x , np.power(x[: , 1] ,i+2 ).reshape(x.shape[0] , 1), axis = 1)
    return x

# a function for calculating MSE error
def MSE(x , y , theta , m):
    """[calculating MSE error]

    Args:
        x ([numpy array (m * d+1)]): [features matrix]
        y ([numpy array (m * 1)]): [target vector]
        theta ([numpy array (d + 1 * 1)]): [theta vector]
        m ([int]): [num of samples]

    Returns:
        [float]: [MSE error]
    """    
    return np.sum(np.power((np.dot(x , theta) - y) , 2))/(2*m)

# gradiant decent
def GD(target , features, reg , d=1 , iter_num=1000 , alpha =0.0001  ):
    """[gradiant decent]

    Args:
        target ([numpy array (m * 1)]): [target vector]
        features ([numpy array (m * 1)]): [raw features vector]
        reg ([float]): [regularization factor]
        d (int, optional): [degree of model]. Defaults to 1.
        iter_num (int, optional): [number of iterations]. Defaults to 1000.
        alpha (float, optional): [learning rate]. Defaults to 0.0001.

    Returns:
        [numpy array (d+1 * 1)]: [theta vector]
    """    
    
    theta = np.zeros((d+1,1)) # initializing theta to zero vector
    theta_temp = theta

    m = features.shape[0] # m = num of samples
    y = target

    x = make_featurs(features , d) # adding polynomial features to features matrix
    x = scaling(x , d) #scaling features
        
    for i in range(iter_num):

        theta_temp = theta*(1 - (alpha *reg) / m) - alpha * (1 / m) * np.dot(x.T , np.dot(x , theta) - y) #theta = theta - alpha * d (J) + lambda * theta power 2
        theta_temp[0] = theta_temp[0] + (alpha * reg)/m * theta[0] #this line is becuase of regularization , we should not regularize theta0

        theta = theta_temp
        #print('error ', i , ' :    ' , MSE(x , y , theta ,m))

    return theta
        
def PolyCoefficients(x, coeffs):
    """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

    The coefficients must be in ascending order (``x**0`` to ``x**o``).
    """
    o = len(coeffs)
    #print(f'# This is a polynomial of order {ord}.')
    y = 0
    for i in range(o):
        y += coeffs[i]*x**i
    return y

def plot_curve(x , y , theta , fig_name , d , alpha ,test_error ,  train_error  , iter_num):

    """
    a function for generating plots and saving them
    """
    xx = np.linspace(-0.4, 1,100000)

    coeffs = list(map(float , theta))

    plt.plot(x[:,1] , y , '.' , label = 'data') #plotting data
    plt.plot(xx, PolyCoefficients(xx, coeffs) , label = 'fitted curve') #plotting the curve
    
    # adding lables and titles to the plot
    plt.title('d = ' + str(d)+' alpha= ' + str(alpha) + ' train error= ' + str(train_error) + ' \ntest error = ' + str(test_error) + ' num of iterations = ' + str(iter_num))
    plt.xlabel('x')
    plt.ylabel('y')

    plt.legend()
    
    #saving plot
    plt.savefig(fig_name + '.png')



# reading data from csv file
firstData = pd.read_csv('Dataset2.csv')

#shuffling data
firstData = firstData.sample(frac = 1).reset_index(drop=True)

x = firstData['x'].to_numpy().reshape(len(firstData),1)
y = firstData['y'].to_numpy().reshape(len(firstData),1)

# splitting data to train and test sets
train = firstData[:int(len(firstData)*0.7)]
test = firstData[int(len(firstData)*0.7):]

x_train = train['x'].to_numpy().reshape(int(len(firstData)*0.7),1)
y_train = train['y'].to_numpy().reshape(int(len(firstData)*0.7),1)

x_test = test['x'].to_numpy().reshape(len(firstData) - int(len(firstData)*0.7),1)
y_test = test['y'].to_numpy().reshape(len(firstData) - int(len(firstData)*0.7),1)





d = 1
alpha = 1

theta = GD(y_train , x_train , reg = 0 , d = d , iter_num=2000 , alpha = alpha)

test_error = MSE(scaling(make_featurs(x_test,d),d) ,y_test, theta , y_train.shape[0])

train_error = MSE(scaling(make_featurs(x_train,d),d) ,y_train , theta , y_train.shape[0])

plot_curve(scaling(make_featurs(x,d),d) , y , theta , 'second linear', d , alpha ,test_error, train_error  , 2000)

plt.clf()


