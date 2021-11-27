import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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

def plot_curve(x , y , theta , fig_name , d  , train_error , test_error ):

    """
    a function for generating plots and saving them
    """
    xx = np.linspace(-230, 1000,1000000)

    coeffs = list(map(float , theta))

     #plotting data
    plt.plot(xx, PolyCoefficients(xx, coeffs) , label =   ' d = ' + str(d)+' train error= ' + str(train_error) + ' \ntest error = ' + str(test_error)) #plotting the curve
    
    # adding lables and titles to the plot
    plt.title( 'normal equation')


    plt.legend()
    
    
    
    
def normal_eq(x , y , d):
    """[summary]

    Args:
        x ([numpy array (m * 1)]): [raw feature vectores]
        y ([numpy array (m * 1)]): [targets vector]
        d ([int]): [degree of model]

    Returns:
        [numpy array (d+1 * 1)]: [theta vector]
    """    
    
    x = make_featurs(x , d) # adding polynomial features to features matrix
    
    temp = np.linalg.inv(np.dot(x.T , x)) #(X^T * X) ^ -1
    theta = np.dot(np.dot(temp , x.T) , y) # temp * X^T * y
    
    return theta



# reading data from csv file
firstData = pd.read_csv('Dataset1.csv')

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

#plotting data
plt.plot(x, y , '.' , label = 'data')
plt.xlabel('x')
plt.ylabel('y')

d = [1 , 2, 4]

for j in d:

    theta = normal_eq(x_train , y_train , j)

    test_error = MSE(make_featurs(x_test,j) ,y_test , theta , y_test.shape[0])

    train_error = MSE(make_featurs(x_train,j) ,y_train , theta , y_train.shape[0])
    
    
    plot_curve(make_featurs(x,j), y , theta , str(j) ,j, train_error , test_error)

    
    print(theta)
    
#saving plot


plt.savefig('5e.png')


