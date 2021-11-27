import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# reading data from csv file
firstData = pd.read_csv('Dataset1.csv')

#plotting the data and saving it
firstData.plot(kind = 'scatter' , x = 'x' , y = 'y' , color = 'red')
plt.savefig('output1.png')




