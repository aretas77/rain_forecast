import pandas as pd


def readAndFormat(path):
    #nuskaitom faila
    df = pd.read_csv(path)
    return df

def getMinMax(dataframe):
    print("Date: ", dataframe['Date'].min(), " ", dataframe['Date'].max()) 
    print("Locations: " , dataframe['Location'].unique())
    print("MinTemp: ", dataframe['MinTemp'].min(), " ", dataframe['MinTemp'].max()) 
    print("MaxTemp: ", dataframe['MaxTemp'].min(), " ", dataframe['MaxTemp'].max()) 
    print("Rainfall: ", dataframe['Rainfall'].min(), " ", dataframe['Rainfall'].max()) 
    print("Evaporation: ", dataframe['Evaporation'].min(), " ", dataframe['Evaporation'].max()) 
    print("Sunshine: ", dataframe['Sunshine'].min(), " ", dataframe['Sunshine'].max()) 
    print("WindGustDir: " , dataframe['WindGustDir'].unique())
    print("WindGustSpeed: ", dataframe['WindGustSpeed'].min(), " ", dataframe['WindGustSpeed'].max()) 
    print("WindDir9am: " , dataframe['WindDir9am'].unique())
    print("WindDir3pm: " , dataframe['WindDir3pm'].unique())
    FormatMinMax('Humidity9am',dataframe)
    FormatMinMax('Humidity3pm',dataframe)
    FormatMinMax('Pressure9am',dataframe)
    FormatMinMax('Pressure3pm',dataframe)
    FormatMinMax('Cloud9am',dataframe)
    FormatMinMax('Cloud3pm',dataframe)
    FormatMinMax('Temp9am'  ,dataframe)
    FormatMinMax('Temp3pm',dataframe)
    print("RainToday: " , dataframe['RainToday'].unique())
    FormatMinMax('RISK_MM',dataframe)
    print("RainTomorrow:",dataframe['RainTomorrow'].unique())

def FormatMinMax(field, df):
    print(field, ": [", df[field].min(), " ", df[field].max(), "]")

dataframe = readAndFormat('/home/stud/Documents/PythonDev/rain_forecast/input/weatherAUS.csv')
getMinMax(dataframe)

