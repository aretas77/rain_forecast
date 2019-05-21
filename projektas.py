import numpy as np
import pandas as pd
from scipy import stats

def readAndFormat(path):
	df = pd.read_csv(path)
	df = df.drop(columns=['Sunshine', 'Evaporation','Cloud3pm','Cloud9am','Location','RISK_MM','Date'],axis=1)
	df = df.dropna(how='any')
	z  = np.abs(stats.zscore(df._get_numeric_data()))
	df = df[(z < 3).all(axis=1)]
	df['RainToday'].replace({'No': 0,'Yes': 1}, inplace = True)
	df['RainTomorrow'].replace({'No': 0,'Yes': 1}, inplace = True)
	categorical_columns = ['WindGustDir','WindDir3pm','WindDir9am']
	df = pd.get_dummies(df, columns=categorical_columns)
	return df


dataframe = readAndFormat('input/weatherAUS.csv')
print(dataframe[0:5])