import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



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


regr =LinearRegression() 

#fitting the model  
regr.fit(x_train, y_train) 

y_pred = regr.predict(x_train) 

#calculating mse
test_error = mean_squared_error(y_test, regr.predict(x_test))
train_error =  mean_squared_error(y_train,y_pred )

#plotting 
plt.scatter(x, y, color ='b') 
plt.plot(x_train, y_pred, color ='k') 
plt.title('test error= ' + str(test_error) + '  train error= ' + str(train_error))
plt.xlabel('x')
plt.ylabel('y')

plt.legend()
    
#saving plot
plt.savefig('sklearn.png')