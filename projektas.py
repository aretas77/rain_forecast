import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
import time

from scipy import stats
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_samples, silhouette_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
from sklearn import svm #Support vector machine
from sklearn.tree import DecisionTreeClassifier 
from plotly import tools
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D


def readAndFormat(path):
    #nuskaitom faila
    df = pd.read_csv(path)

    print('Size of weather data frame is:', df.shape)
    print('Data before cleanup.')
    print(df[0:5])

    # Ismetame nereikalingus duomenis.
    #
    # Sunshine - number of hours of brightness in the day.
    #	* Too little data.
    # Cloud3pm - fraction of sky obscured by clouds at 3pm.
    #	* Too little data.
    # Cloud9am - fraction of sky obscured by clouds at 9am.
    #	* Too little data.
    # Location - the name of the city located in Australia.
    #	* We are determining Will it rain in Australia? So we don't need locations.
    # RISK_MM - amount of next day rain in mm.
    #	* This could leak prediction data to our model, so drop it.
    df = df.drop(columns=['Sunshine', 'Evaporation', 'Cloud3pm','Cloud9am','Location','RISK_MM','Date'],axis=1)
    print('Data after cleanup.')
    print(df[0:5])

    # Drop any null values
    df = df.dropna(how='any')

    # Show Pearson correlation coefficients
    fig, ax = plt.subplots(figsize=[12,12])
    sns.heatmap(df.corr(), ax=ax, cmap='Blues', annot=True)
    ax.set_title("Pearson correlation coefficients", size=20)
    plt.show()

    #params=['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
    #        'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']
    #pd.plotting.scatter_matrix(df[params], alpha=0.2, figsize=(20, 20))
    #plt.show()


    # Apskaiciuojam normalizacija.
    # We will remove outliers in our data. We are using Z-score to detect and remove
    # the outliers. In other words, this removes values that are 'not normal' and are
    # deviated too much.
    z  = np.abs(stats.zscore(df._get_numeric_data()))
    # Select all data which is less than z value of 3.
    df = df[(z < 3).all(axis=1)]

    # pakeiciam Yes ir No i 1 ir 0 del paprastumo
    df['RainToday'].replace({'No': 0,'Yes': 1}, inplace = True)
    df['RainTomorrow'].replace({'No': 0,'Yes': 1}, inplace = True)

    # kadangi vejo gusiai klasifikuojami raidemis, sukuriam papildomus columns
    # kad galima butu suzymeti 1 ir 0 kekvienai direkcijai
    categorical_columns = ['WindGustDir','WindDir3pm','WindDir9am']
    df = pd.get_dummies(df, columns=categorical_columns)
    print(df[0:5])

    return df

def Preprocess(dataframe, count):
    # standartizuojam  visa data su MinMaxScaler'iu
    # Now data is between 0 and 1.
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(dataframe)
    dataframe = pd.DataFrame(scaler.transform(dataframe),
            index=dataframe.index, columns = dataframe.columns)
    print('\nDone MinMaxScaler().')
    print(dataframe[0:5])

    # naudojam SelectKBest metoda isrenkant labiausiai susijusias values su
    # lietum rytoj

    # Take all columns but skip RainTommorrow and put it in a different set.
    x = dataframe.loc[:, dataframe.columns != 'RainTomorrow']
    y = dataframe[['RainTomorrow']]

    # We use chi2 method to determine features that are dependent on the
    # target. If, for example, a feature is not dependent on the target, then
    # the feature is uninformative for classifying observations.
    selector = SelectKBest(chi2,k=count)
    selector.fit(x,y)
    x_new = selector.transform(x)

    # Return top count columns.
    return (x.columns[selector.get_support(indices=True)])

def LogReg(x,y):
    t0 = time.time()
    # x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
    x_train = x[['Humidity3pm', 'Rainfall', 'RainToday']]
    x_test = y[['Humidity3pm', 'Rainfall', 'RainToday']]
    y_train = x[['RainTomorrow']]
    y_test = y[['RainTomorrow']]

    # Call LogisticRegression method
    clf_logreg = LogisticRegression(random_state=0, solver='liblinear')
    clf_logreg.fit(x_train,y_train.values.ravel())

    y_pred = clf_logreg.predict(x_test)
    score = accuracy_score(y_test,y_pred)
    time_taken = time.time()-t0
    return score, time_taken

def SVM(x,y):
    X_train = x[['Humidity3pm', 'Rainfall', 'RainToday']]
    X_test = y[['Humidity3pm', 'Rainfall', 'RainToday']]
    y_train = x[['RainTomorrow']]
    y_test = y[['RainTomorrow']]


    t0=time.time()
  #  X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
    clf_svc = svm.SVC(kernel='linear')
    clf_svc.fit(X_train,y_train)
    y_pred = clf_svc.predict(X_test)
    score = accuracy_score(y_test,y_pred)
    time_taken = time.time()-t0
    return score, time_taken


def DecisionTree(x,y): 
    X_train = x[['Humidity3pm', 'Rainfall', 'RainToday']]
    X_test = y[['Humidity3pm', 'Rainfall', 'RainToday']]
    y_train = x[['RainTomorrow']]
    y_test = y[['RainTomorrow']]


    t0=time.time()
  #  X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

    clf_dt = DecisionTreeClassifier(random_state=0)
    clf_dt.fit(X_train,y_train)
    y_pred = clf_dt.predict(X_test)
    score = accuracy_score(y_test,y_pred)
    time_taken = time.time()-t0
    return score, time_taken

def KMeansMethod(data):
    t0 = time.time()

    train = data[['Humidity3pm', 'Rainfall', 'RainToday']]
    test = data[['RainTomorrow']]

    # show the relation of tomorrow rain with humidity3pm and rainfall
    sns.set_style('whitegrid')
    sns.lmplot('Humidity3pm', 'Rainfall', data=data, col='RainTomorrow', hue='RainTomorrow',
            palette='coolwarm', height=6, aspect=1, fit_reg=False)
    
    # show histogram for humidity3pm
    sns.set_style('darkgrid')
    sns.FacetGrid(data, hue='RainTomorrow', palette='coolwarm', height=6, aspect=2,
            col='RainTomorrow').map(plt.hist, 'Humidity3pm', bins=20, alpha=0.7)

    # show histogram for rainfall
    sns.FacetGrid(data, hue='RainTomorrow', palette='coolwarm', height=6, aspect=2,
            col='RainTomorrow').map(plt.hist, 'Rainfall', bins=20, alpha=0.7)

    # Calculate the number of clusters (just for diagrams)
    # GetClusterSizeElbow(data)

    # Apply kmeans to the dataset / Creat kmeans classifier
    kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10,
            random_state=0)
    y_kmeans = kmeans.fit_predict(train)

    print(confusion_matrix(data['RainTomorrow'], kmeans.labels_))
    print(classification_report(data['RainTomorrow'], kmeans.labels_))

    plt.show()
    data['RainTomorrow'] = y_kmeans

    sns.set_style('whitegrid')
    sns.lmplot('Humidity3pm', 'Rainfall', data=data, hue='RainTomorrow',
               palette='coolwarm', height=6, aspect=1, fit_reg=False, scatter=True)
    plt.scatter(kmeans.cluster_centers_[0, 0], kmeans.cluster_centers_[0, 1], color='r')
    plt.scatter(kmeans.cluster_centers_[1, 0], kmeans.cluster_centers_[1, 1], color='r')

    plt.legend()
    plt.show()

def GetTrainingData(data, to, fro):
    frames = list()
    frames.append(data[:to])
    frames.append(data[fro:])
    return pd.concat(frames)

def CrossTestHarness(dataframe, method):
    segment_size = int(dataframe.shape[0]/10)
    test_datas = list()
    train_datas = list()
    scores_times = list()
    average_score, average_time = 0.0, 0.0

    for i in range(10):
        train_datas.append(GetTrainingData(dataframe,segment_size*i,segment_size*(i+1)))
        for j in range(10):
            if j==i:
                test_datas.append(dataframe[segment_size*i:segment_size*(i+1)])

    
    for i in range(10):
        if method == 'LogisticRegression':
            scores_times.append(LogReg(train_datas[i],test_datas[i]))
        if method == "SVM":
            scores_times.append(SVM(train_datas[i],test_datas[i]))
        if method == 'DecisionTreeClassifier':
            scores_times.append(DecisionTree(train_datas[i],test_datas[i]))
    #add your own methods here with if statements
    for i in range(len(scores_times)):
        average_score += scores_times[i][0]
        average_time += scores_times[i][1]
    average_score = average_score/len(scores_times)
    average_time = average_time/len(scores_times)
    print(method, "Average score =", average_score * 100, "% ", "average time =", average_time, "s")

def GetClusterSizeElbow(data):
    # WCSS - within cluster sum of squares
    wcss = []

    for i in range(1, 7):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 7), wcss)
    plt.title('The elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

# Method to calculate how many clusters are recommended for KMeans
def GetClusterSizeSilhouette(data):
    n_clusters = [ 2, 3, 4, 5 ]
    for n in n_clusters:
        t = time.time()

        clusterer = KMeans(n_clusters=n, random_state=0)
        cluster_labels = clusterer.fit_predict(data)

        sillhouette_avg = silhouette_score(data, cluster_labels)
        time_taken = time.time() - t
        print("For n_clusters =", n, "The average silhouette_score is :", sillhouette_avg,
                "Taken time is :", time_taken)

dataframe = readAndFormat('/home/stud/Documents/PythonDev/rain_forecast/input/weatherAUS.csv')
#selectinam top n values pagal kurias apmokysim 
elements = Preprocess(dataframe, 4)

modified_dataFrame = dataframe[['Humidity3pm', 'Rainfall', 'RainToday', 'RainTomorrow']]
#CrossTestHarness(modified_dataFrame, 'LogisticRegression')
CrossTestHarness(modified_dataFrame, 'DecisionTreeClassifier')
# Unsupervised methods with preprocessed data
#KMeansMethod(modified_dataFrame)
