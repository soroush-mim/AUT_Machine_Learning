import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# reading data from csv file
firstData = pd.read_csv('Dataset1.csv')

#shuffling data
firstData = firstData.sample(frac = 1).reset_index(drop=True)




