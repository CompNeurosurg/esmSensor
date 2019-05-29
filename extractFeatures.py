import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from mne.filter import filter_data
from scipy.signal import find_peaks_cwt, welch
from entropy import shannon_entropy, sample_entropy
import time
from os.path import join

def extractFeatures(data, esm,sr, windowLength=60):

    numSamples=data.shape[0]
    # Getting number and names of features
    tremorNames, _ = tremorFeatures(data[0,:windowLength*sr,:], sr,windowLength=windowLength)
    bradyNames, _ = bradykinesiaFeatures(data[0,:windowLength*sr,:], sr,windowLength=windowLength)
    cols=[]
    for s in ['L','R','C']:
        cols.extend([c + s for c in tremorNames])
        cols.extend([c + s for c in bradyNames])
    cols.extend(esm.keys())
    aligned = pd.DataFrame(columns=cols)
    accelerometerChannel = [a + b for a in  [0,6,12] for b in range(3)]
    for beep in range(data.shape[0]):
        t=time.time()
        allFeat = []
        numWindows = int(data.shape[1]/sr/windowLength)
        buff = data[beep,:,:]
        buff[:,accelerometerChannel] = (buff[:,accelerometerChannel].T - np.mean(buff[:,accelerometerChannel].T,axis=0)).T
        #buff[:,accelerometerChannel] = filter_data(buff[:,accelerometerChannel].T,sr,0,3,method='iir',verbose='WARNING').T
        for s,sID in enumerate([range(6),range(6,12),range(12,18)]):
            
            features=np.zeros((numWindows,len(tremorNames)+len(bradyNames)))
            for i in range(0,numWindows):
                win = i * windowLength * sr
                _, features[i,:len(tremorNames)] = tremorFeatures(buff[win:win+windowLength*sr,sID],sr,windowLength=windowLength)
                _, features[i,len(tremorNames):] = bradykinesiaFeatures(buff[win:win+windowLength*sr,sID],sr,windowLength=windowLength)
            allFeat.append(features)
        allFeat = np.concatenate(allFeat,axis=1)
        allFeat = np.concatenate((allFeat, np.matlib.repmat(esm.iloc[beep,:],numWindows,1)),axis=1)
        aligned = aligned.append(pd.DataFrame(allFeat,columns = cols),ignore_index=True)
        #print(time.time()-t)
        # Add the power between 3.6 and 9.4
        #Timestamp at beginning of each window
        #alignedTimes.append(startTime + pd.Timedelta('%d s ' % (i * windowLength)))
    return aligned



def tremorFeatures(windowData,sr,windowLength=60):
    tremorChannel={'AccX':1,'AccY':2,'AccZ':3,'X':3,'Y':4,'Z':5} # gyro is xyz 3-4-5
    if windowData.shape[0]!=sr*windowLength:
        print(windowData.shape,sr*windowLength)
    #freq = np.fft.rfftfreq(windowLength*sr, d=1./sr)
    #selected=np.logical_and(freq>3.5,freq<7.5)
    features=[]
    featureNames=[]
    for ch in tremorChannel.keys():
        #spec = np.mean(np.log(np.abs(np.fft.rfft(windowData[:,tremorChannel[ch]]))[selected]))
        f, spec = welch(windowData[:,tremorChannel[ch]], fs=sr, nperseg=sr )
        selected = np.logical_and(f>3.5,f<7.5)
        spec = np.mean(np.log(spec[selected])) # np.log
        features.append(spec)
        featureNames.append('BandPower' + ch)
        
        
    return featureNames, features


# In[145]:


def bradykinesiaFeatures(windowData,sr,windowLength=60):
    features=[]
    featureNames=[]
    accelerometerChannel={'X':0,'Y':1,'Z':2} # assuming acc xyz 0-1-2 is
    windowData = filter_data(windowData[:,list(accelerometerChannel.values())].T,sr,0,3,method='iir',verbose='WARNING').T
    

    # lowpass filter signal <3 Hz
    
    # Features: (Patel et al IEEE 2009)
    # rms
    # range
    # entropy
    # normalized cross-correlation value and time point
    # dominat frequency and ratio between dominant and rest
    freq = np.fft.rfftfreq(windowLength*sr, d=1./sr)
    
    for ch in accelerometerChannel.keys():
        #ent = shannon_entropy(windowData[:,accelerometerChannel[ch]])
        #features.append(ent)
        #featureNames.append('Entropy' + ch)
        
        spec = np.abs(np.fft.rfft(windowData[:,accelerometerChannel[ch]]))
        domFreq = freq[np.argmax(spec)]
        features.append(domFreq)
        featureNames.append('DomFreq' + ch)
        
        domEnergyRatio = np.max(spec) / np.sum(spec)
        features.append(domEnergyRatio)
        featureNames.append('DomEnergyRatio' + ch)
        
        rms = np.sqrt(np.mean(windowData[:,accelerometerChannel[ch]]**2))
        features.append(rms)
        featureNames.append('RMS' + ch)
        
        ampRange = np.max(windowData[:,accelerometerChannel[ch]]) - np.min(windowData[:,accelerometerChannel[ch]])
        features.append(ampRange)
        featureNames.append('AmpRange' + ch)
    
    cCMax=[]
    cCLocs=[]
    for i, ch1 in enumerate(accelerometerChannel.keys()):
        for j,ch2 in enumerate(list(accelerometerChannel.keys())[i+1:]):
            crossCorr = np.correlate(windowData[:,accelerometerChannel[ch1]],windowData[:,accelerometerChannel[ch2]],'same')
            crossCorr = crossCorr/(np.std(windowData[:,accelerometerChannel[ch1]]) * np.std(windowData[:,accelerometerChannel[ch2]]))
            
            cCMax.append(np.max(crossCorr))
            cCLocs.append(np.argmax(crossCorr))
    features.append(np.max(cCMax))
    featureNames.append('MaxCC')
    features.append(cCLocs[np.argmax(cCMax)])
    featureNames.append('MaxCCLoc')
        
        #peaks=find_peaks_cwt(windowData[accelerometerChannel[ch],:],np.arange(1,10))
        #peaks=[1,2,3]
        #features.append(len(peaks))
        #featureNames.append('#Movements' + ch)
        
    #features.append(np.max(windowData[0:3,:]))
    #featureNames.append('MaxMovement')
    return featureNames, features




if __name__ == '__main__':
    allSubs = ['110001','110002','110003','110004','110005','110006','110007','110008','110009','110010','110011','110013','110014','110016','110017','110018',]
    allSubs = ['110018']
    outPath = 'C:/data/processed/ESM_pilot/'
    sr=100
    winL = 30 # seconds
    for subject in allSubs:
        print(subject)
        trialData = np.load(join(outPath, subject + '_trials.npy')).astype(np.float64)
        esm = pd.read_csv(join(outPath,  subject + '_esm.csv'))
        alignedFeatures = extractFeatures(trialData,esm,sr,windowLength=winL)
        alignedFeatures.to_csv(join(outPath,  subject + '_features' +  str(winL)  + '.csv'),index=False)


