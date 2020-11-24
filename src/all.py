import numpy as np 
import pandas as pd
import math

import warnings
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks, peak_widths
from itertools import chain

import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


#filepath = '/teams/DSC180A_FA20_A00/b05vpnxray/data/unzipped'
filepath = 'test'
data = os.listdir(filepath)

    
def find_intervalPeaks(Data, height_min):
    """
    Finds the peak heights in 2->1 Bytes, and then evaluates the seconds in each interval 
    between peaks
    
    data: takes in a dataframe of network stats data 
    height_min: takes in minimum peak height to create peaks for 
    
    returns array of interval lengths
    """
    x = Data['Time']
    y = Data['2->1Bytes']
    peaks, _ = find_peaks(y, height=height_min)
    s = x[peaks].diff().apply(lambda x: x.seconds)
    #returns array of interval lengths
    s = s[1:].to_numpy()
    s = s[s>1]
    return s

def packetSizes(data):
    """
    Finds number of packets being downloaded that are larger than 1200 bytes
    
    Data: takes in dataframe of network stats file dataframe 
    """
    packets = data[['packet_sizes','packet_dirs']]
    #filter out upload rows
    packets = packets.loc[packets['packet_dirs'] != '1;']
    #next, we can break up each packet size and corresponding direction into its own row 
    packets = packets.apply(lambda x: x.str.split(';').explode())
    #drop empty rows, and packet sizes above 1500 (this should not be happening)
    packets['packet_sizes'].replace('', np.nan, inplace=True)
    packets = packets.dropna(subset=['packet_sizes'])
    packets = packets.loc[packets['packet_sizes'] != '1500']
    #drop direction 1 
    packets = packets.loc[packets['packet_dirs'] != '1'].reset_index() 
    #Find all packets larger than 1200 bytes 
    sizes = packets['packet_sizes']
    count = len(sizes.astype(int).loc[sizes.astype(int) > 1200])
    #returns series of packet sizes being downloaded
    return count


def find_threshold(Data):
    """
    Finds the threshold we want to use in finding interval peaks
    
    Data: takes in dataframe of network stats data 
    """
    x = Data['Time']
    y = Data['2->1Bytes']
    peaks, _ = find_peaks(y, height=0)
    mean = y[peaks].mean() - 10000
    return mean


def label(data):
    """
    creates labels for data from file name
    """
    labels =[]
    for file in data:
        if 'novideo' in file:
            labels.append(0)
        else:
            labels.append(1)
    return labels

#function to create features

warnings.filterwarnings("ignore")
def findFeatures(filepath):
    """
    takes in list of data files
    outputs clean features and labels
    """
    data = os.listdir(filepath)
    bags = []
    # files that start with '._' have no data, taking these out
    bad_data = '._'
    data = [x for x in data if not x.startswith(bad_data)]
    # we dont want to look at novpn data right now, take these out as well
    data  = [x for x in data if not ('novpn' in x)]
    data = [x for x in data if x.endswith('.csv')]
    for file in data: 
        file = pd.read_csv(filepath + '/' + file)
        file['Time'] = pd.to_datetime(file['Time'], unit='s')
        mean = find_threshold(file)
        x = np.array([find_intervalPeaks(file, mean)]).std()
        #if bags has nan value that means there is only one peak or no peaks, meaning probably non streaming?
        #replace that value with 99 for now 
        if math.isnan(x):
            x = 100
        bags.append([x, packetSizes(file)]) 
    #bags = reshape(bags)
    #print(bags)
    labels = label(data)
    print('data labels: ' + str(labels) + ' (check to see if test data is working!)')
    return bags, labels

def create_data(data):
    """
    creates training and testing data from raw data list
    """
    X, y = findFeatures(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test

def mlModel(data):
    """
    trains model on data
    returns trained model and test accuracy 
    """
    X_train, X_test, y_train, y_test = create_data(data)
    model = KNeighborsClassifier(n_neighbors=3) #TO CHANGE
    model.fit(X_train, y_train)
    #print(X_test)
    y_pred = model.predict(X_test)
    y_true = y_test
    print ('Test Accuracy is: ' + str(accuracy_score(y_true, y_pred)))
    return model


#warnings.filterwarnings("ignore")
#model = mlModel(data)

def inputFile():
    filepath = 'inputFile'
    data = os.listdir(filepath)
    
    bags = []
    if (data == ['.ipynb_checkpoints'] or data == [] or data == ['.DS_Store']):
        return ('No Input File Found!')
    for file in data:
        #finds first csv file 
        if file.endswith(".csv"):
            file = pd.read_csv(filepath + '/' + file)
            file['Time'] = pd.to_datetime(file['Time'], unit='s')
            #find feature peaks
            mean = find_threshold(file)
            x = np.array([find_intervalPeaks(file, mean)]).std()
            if math.isnan(x):
                x = 100
            bags.append([x, packetSizes(file)]) 
            return np.array(bags)


#if type(inputFile()) == str:
    #print('No Input File Found!')
#elif model.predict(inputFile())[0] == 1:
   # print ("This is predicted to be video streaming!")
#else:
     #print ("This is predicted to be NOT video streaming!")
