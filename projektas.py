import numpy as np
import pandas as pd
from scipy import stats

dataframe = pd.read_csv('input/weatherAUS.csv')
print("size is: ", dataframe.shape)
dataframe = dataframe.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am','Location','RISK_MM','Date'],axis=1)
dataframe = dataframe.dropna(how='any')
z=np.abs(stats.zscore(dataframe._get_numeric_data()))
print(z)
dataframe=dataframe[(z<3).all(axis=1)]
print(dataframe.shape)
dataframe['RainToday'].replace({'No':0,'Yes':1},inplace = True)
dataframe['RainTomorrow'].replace({'No':0,'Yes':1},inplace=True)
print(dataframe[0:5])
#print(dataframe.shape)
#print(dataframe[0:5])
#not_empty = dataframe.count().sort_values()
#good_tags = list()
#for i in range(len(not_empty)):
#    print(not_empty[i])
#    if not_empty[i] > (dataframe.shape[0] * 0.66) :
#        good_tags.append(i)
#        print(not_empty[i], " lel ", (dataframe.shape[0] * 0.6) )
