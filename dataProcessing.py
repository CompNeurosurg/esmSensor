from mne.filter import filter_data
import pandas as pd
import pyedflib
import numpy as np
from datetime import timedelta
from scipy.signal import find_peaks_cwt

def extractAllSensors(leftWristFile,rightWristFile,chestFile,featureWindowLength=60):
    # Read in the three files
    #Process leftWristData
    labels, timeStamps, data, sr = readData(leftWristFile)
    #Could be left wrist specific or general
    alignedTimes, leftWristFeatures, labels = extractFeatures(data, timeStamps, sr, featureWindowLength)
    labels=[l + 'L' for l in labels]
    leftWristDF=pd.DataFrame(leftWristFeatures.T,columns=labels,index=alignedTimes)
    # Same for right wrist
    labels, timeStamps, data, sr = readData(rightWristFile)
    #Could be right wrist specific or general
    alignedTimes, rightWristFeatures, labels = extractFeatures(data, timeStamps, sr, featureWindowLength)
    labels=[l + 'R' for l in labels]
    rightWristDF=pd.DataFrame(rightWristFeatures.T,columns=labels,index=alignedTimes)
    # Same for chest
    labels, timeStamps, data, sr = readData(chestFile)
    #Could be chest specific or general
    alignedTimes, chestFeatures, labels = extractFeatures(data, timeStamps, sr, featureWindowLength)
    labels=[l + 'C' for l in labels]
    chestDF=pd.DataFrame(chestFeatures.T,columns=labels,index=alignedTimes)
    return leftWristDF,rightWristDF,chestDF    

def readData(filename):
    #Extract data
    f = pyedflib.EdfReader(filename)
    sr = f.getSampleFrequencies()[0]
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)
    #Get starting time
    startingTime=filename[-19:-4]
    startingTime=pd.to_datetime(startingTime, format='%Y%m%d_%H%M%S', errors='ignore')
    timeStamps=[]
    last=startingTime
    for time in range(f.getNSamples()[0]):
        timeStamps.append(last)
        last=last+pd.Timedelta('%d ms' % (1000/sr))
    f._close()
    return signal_labels, timeStamps, sigbufs, sr

def extractFeatures(data, timeStamps, sr, windowLength):
    #Filter data between 4 and 8 Hz
    #filtData = filter_data(data, sr, 4,8)

    #Extract some sort of feature for all windows and corresponding time stamps
    numSamples=data.shape[1]
    # Getting number and names of features
    tremorNames, _ = tremorFeatures(data[:,:windowLength*sr], sr)
    bradyNames, _ = bradykinesiaFeatures(data[:,:windowLength*sr], sr)
    
    
    features=np.zeros((len(tremorNames) + len(bradyNames),int(numSamples/(windowLength*sr))))
    alignedTimes=[]
    for i,win in enumerate(range(0,numSamples,windowLength*sr)):
        if i<features.shape[1]:
            #Average power per channel
            #features[:,i]=np.mean(filtData[:,win:win+windowLength*sr]**2,axis=1)
            _, features[:len(tremorNames),i] = tremorFeatures(data[:,win:win+windowLength*sr],sr)
            _, features[len(tremorNames):,i] = bradykinesiaFeatures(data[:,win:win+windowLength*sr],sr)
            # Add the power between 3.6 and 9.4
            #Timestamp at beginning of each window
            alignedTimes.append(timeStamps[win])
    return alignedTimes, features, tremorNames + bradyNames

def tremorFeatures(windowData,sr):
    gyroChannel={'X':6,'Y':7,'Z':8}
    freq = np.fft.rfftfreq(windowData[0,:].shape[0], d=1./sr)
    selected=np.logical_and(freq>3.5,freq<7.5)
    features=[]
    featureNames=[]
    for ch in gyroChannel:
        spec = np.log(np.abs(np.fft.rfft(windowData[gyroChannel[ch],:])))
        features.append(np.sum(spec[selected]))
        featureNames.append('BandPower' + ch)
    return featureNames, features

def bradykinesiaFeatures(windowData,sr):
    features=[]
    featureNames=[]
    accelerometerChannel={'X':3,'Y':4,'Z':5}
    for ch in accelerometerChannel.keys():
        peaks=find_peaks_cwt(windowData[accelerometerChannel[ch],:],np.arange(40,50))
        features.append(len(peaks))
        featureNames.append('#Movements' + ch)
        features.append(np.mean(np.diff(peaks))/sr)
        featureNames.append('MovementDuration' + ch)
    features.append(np.max(windowData[3:6,:]))
    featureNames.append('MaxMovement')
    return featureNames, features

def alignFeaturesESM(listOfDF,esmFrame,esmColumns,esmWindowLength=15):
    
    combinedColumns=esmColumns
    for featureFrame in listOfDF:
        combinedColumns= combinedColumns + featureFrame.keys().tolist()
    esmFeatures=pd.DataFrame(columns=combinedColumns) # Create new empty dataframe with feature and esm columns

    hop=np.mean(np.diff(listOfDF[0].index))
    for beep in range(esmFrame.shape[0]): #Loop through all the ESM Beeps

        beepTime=esmFrame.index[beep] # Get the corresponding time
        
        esmData=np.matlib.repmat(esmFrame.iloc[beep][esmColumns],esmWindowLength,1)
        combined=esmData
        
        subIndex=[beepTime-hop*t for t in range(esmWindowLength)][::-1]
        for featureFrame in listOfDF:
        
        
            timediff = np.min(np.abs(featureFrame.index-esmFrame.index[beep])) 
            # Find corresponding moment for beep time in the sensor data
            if timediff>timedelta(minutes=esmWindowLength):
                # If corresponding time is too far off, remove beep
                #print("Couldn't find corresponding sensor data")
                continue
            pos=np.argmin(np.abs(featureFrame.index-esmFrame.index[beep]))
            # For the smallest time difference find the position in the sensor data
            if pos>esmWindowLength:
                featColumns=featureFrame.keys().tolist() #The names of the features                
                featData=featureFrame.iloc[pos-esmWindowLength:pos][featColumns].values
                # Get corresponding timestamps
                
                # Repeat ESM data for each data point in the window
                combined=np.concatenate((combined,featData),axis=1)
                #Combine ESM & feature data
        if combined.shape[1]==len(combinedColumns):
            esmFeatures=esmFeatures.append(pd.DataFrame(combined,columns=combinedColumns,index=subIndex))
                #Append combined data to the dataframe
    return esmFeatures

if __name__ == "__main__":
    featureWindowLength=60
    esmWindowLength=15
    leftWristFile="13337_20180203_094429.bdf"
    rightWristFile="13337_20180203_094429.bdf"
    chestFile="13337_20180203_094429.bdf"
    #Extract features for chest and both wrists
    l,r,c = extractAllSensors(leftWristFile,rightWristFile,chestFile,60)
