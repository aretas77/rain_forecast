import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import preprocessing

def readAndFormat(path):
	#nuskaitom faila
	df = pd.read_csv(path)
	#ismetame nereikalingus duomenis
	df = df.drop(columns=['Sunshine', 'Evaporation',
		'Cloud3pm','Cloud9am','Location','RISK_MM','Date'],axis=1)
	#ismetas N/A values
	df = df.dropna(how='any')
	#apskaiciuojam normalizacija
	z  = np.abs(stats.zscore(df._get_numeric_data()))
	#ismetam duomenis kurie neatitinka normu
	df = df[(z < 3).all(axis=1)]
	#pakeiciam Yes ir No i 1 ir 0 del paprastumo
	df['RainToday'].replace({'No': 0,'Yes': 1}, inplace = True)
	df['RainTomorrow'].replace({'No': 0,'Yes': 1}, inplace = True)
	#kadangi vejo gusiai klasifikuojami raidemis, sukuriam papildomus columns
	#kad galima butu suzymeti 1 ir 0 kekvienai direkcijai
	categorical_columns = ['WindGustDir','WindDir3pm','WindDir9am']
	df = pd.get_dummies(df, columns=categorical_columns)
	return df

def Preprocess(dataframe, count):
	#standartizuojam  visa data su MinMaxScaler'iu
	scaler = preprocessing.MinMaxScaler()
	scaler.fit(dataframe)
	dataframe = pd.DataFrame(scaler.transform(dataframe),
		index=dataframe.index, columns = dataframe.columns)
	#naudojam SelectKBest metoda isrenkant labiausiai susijusias values su lietum rytoj
	x = dataframe.loc[:,dataframe.columns!='RainTomorrow']
	y = dataframe[['RainTomorrow']]
	selector = SelectKBest(chi2,k=count)
	selector.fit(x,y)
	x_new = selector.transform(x)
	return (x.columns[selector.get_support(indices=True)])


dataframe = readAndFormat('input/weatherAUS.csv')
print(dataframe[0:5])
#selectinam top n values pagal kurias apmokysim 
elements = Preprocess(dataframe, 4)
#selectinam tik tuos values is dataframe
dataframe = dataframe[elements]