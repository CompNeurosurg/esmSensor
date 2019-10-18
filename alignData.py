from os import listdir
from os.path import isfile, join
from datetime import timedelta
import time

from scipy.signal import decimate
import pandas as pd
import matplotlib.pyplot as plt
import pyedflib
import numpy as np


def loadESM(path):
    esm = pd.read_stata('C:/data/raw/MOX/SANPAR_BE.dta', convert_categoricals=False)
    esm = esm[['subjno', 'mood_well', 'mood_down', 'mood_fright', 'mood_tense', 'phy_sleepy', 'phy_tired',
               'mood_cheerf', 'mood_relax', 'thou_concent', 'pat_hallu', 'loc_where',
               'soc_who', 'soc_who02', 'soc_who03', 'act_what', 'act_what02',
               'act_what03', 'act_norpob', 'sanpar_been', 'sanpar_stil',
               'sanpar_spreken', 'sanpar_lopen', 'sanpar_tremor', 'sanpar_traag',
               'sanpar_stijf', 'sanpar_spann', 'sanpar_beweeg', 'sanpar_onoff',
               'sanpar_medic', 'beep_disturb', '_datetime', '_datetime_e', 'dayno_n', 'beepno_n']]
    esm['duration'] = esm['_datetime_e'] - esm['_datetime']

    # rename to english
    esm = esm.rename(index=str,
                     columns={'sanpar_been': 'prob_mobility',
                              'sanpar_stil': 'prob_stillness',
                              'sanpar_spreken': 'prob_speech',
                              'sanpar_lopen': 'prob_walking',
                              'sanpar_tremor': 'tremor',
                              'sanpar_traag': 'slowness',
                              'sanpar_stijf': 'stiffness',
                              'sanpar_spann': 'tension',
                              'sanpar_beweeg': 'dyskinesia',
                              'sanpar_onoff': 'onoff',
                              'sanpar_medic': 'medic'})

    mapNames = {}
    for i in range(25):
        mapNames[9009989 + i] = 110001 + i

    esm['castorID'] = [mapNames[e] for e in esm['subjno']]
    return esm


def getFileLists(localPath, subject):
    # create list of files per L/R/chest from directory (mypath)
    localPath = join(localPath, subject)
    leftSensors = ['13797', '13799', '13794', '13806']
    rightSensors = ['13805', '13801', '13793', '13795']
    chestSensors = ['13804', '13792', '13803', '13796']

    bdffiles = [f for f in listdir(localPath) if isfile(join(localPath, f)) and f[0] != '_' and f[-3:] == 'edf']
    # bdffiles are the files in mypath, not directories

    leftFiles = []
    rightFiles = []
    chestFiles = []

    for f in bdffiles:
        if f[0:5] in leftSensors:
            leftFiles.append(join(localPath, f))
        elif f[0:5] in rightSensors:
            rightFiles.append(join(localPath, f))
        elif f[0:5] in chestSensors:
            chestFiles.append(join(localPath, f))

    leftFiles = sorted(leftFiles)
    rightFiles = sorted(rightFiles)
    chestFiles = sorted(chestFiles)
    return leftFiles, rightFiles, chestFiles


# In[34]:


def extractRawTrials(leftFiles,rightFiles,chestFiles,esmFrame,esmWindowLength=15,featureWindowLength=60):
    # Read in the three list of files
    #Process leftWristData
    leftWristDF=[]
    rightWristDF=[]
    chestDF=[]
    
    files = [leftFiles, rightFiles, chestFiles]
    trials = [[[] for _ in range(esmFrame.shape[0])],[[] for _ in range(esmFrame.shape[0])], [[] for _ in range(esmFrame.shape[0])]]
    identifiers = ['l', 'r', 'c']
    foundTrials = np.zeros((esmFrame.shape[0],3))
    for i, f in enumerate(files):
        for file in f:
            print(file)
            try:
                labels, timeStamps, data, sr = readData(file) ## as input instead: leftFiles
                if data.shape[1]<sr * featureWindowLength:
                    raise ValueError('File too short to proceed.')
            except:
                print('%s is broken' % file)
                continue
            data = pd.DataFrame(data.T,index=timeStamps)
            for beep in range(esmFrame.shape[0]):
                if foundTrials[beep,i]==1:
                    continue
                beepTime=esmFrame['_datetime'].iloc[beep] # Get the corresponding time
                timediff = np.min(np.abs(data.index-beepTime)) 
                # Find corresponding moment for beep time in the sensor data
                #print(timediff)
                if timediff>timedelta(minutes=esmWindowLength):
                # If corresponding time is too far off, remove beep
                #print("Couldn't find corresponding sensor data")
                    continue
                pos=np.argmin(np.abs(data.index-beepTime))
                # For the smallest time difference find the position in the sensor data
                if pos>esmWindowLength*windowLength*sr:
                    trials[i][beep] = data.iloc[pos-(int(esmWindowLength*windowLength*sr)):pos]
                    foundTrials[beep,i]=1

    keep = np.sum(foundTrials,axis=1)==3
    trialData = np.zeros((np.sum(keep),int(esmWindowLength*windowLength*sr), 3 * 6))
    counter =0
    for beep in range(esmFrame.shape[0]):
        if keep[beep]:
            temp = np.concatenate((trials[0][beep],trials[1][beep],trials[2][beep]),axis=1)
            trialData[counter,:,:]=temp
            counter+=1
    foundESM = esmFrame.iloc[keep,:]   
    return trialData,  foundESM


# In[31]:


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
    #startingTime=f.getStartdatetime() #needs to be tested
    startingTime=filename[-19:-4]
    startingTime=pd.to_datetime(startingTime, format='%Y%m%d_%H%M%S', errors='ignore')
    #print(startingTime)
    sigbufs = decimate(sigbufs,downsampling,axis=1)
    sr=sr/downsampling
    timeStamps=pd.date_range(start=startingTime,periods=sigbufs.shape[1],freq='%d ms' % (1000/sr))
    return signal_labels, timeStamps, sigbufs, sr


# In[32]:


def alignFeaturesESM(listOfDF,esmFrame,esmColumns,esmWindowLength=15):
    
    combinedColumns=esmColumns
    for featureFrame in listOfDF:
        combinedColumns= combinedColumns + featureFrame.keys().tolist()  
    esmFeatures=pd.DataFrame(columns=combinedColumns) # Create new empty dataframe with feature and esm columns

    hop=np.mean(np.diff(listOfDF[0].index))
    for beep in range(esmFrame.shape[0]): #Loop through all the ESM Beeps
        beepTime=esmFrame['_datetime'].iloc[beep] # Get the corresponding time
        
        esmData=np.matlib.repmat(esmFrame.iloc[beep][esmColumns],esmWindowLength,1)
        combined=esmData
        
        subIndex=[beepTime-hop*t for t in range(esmWindowLength)][::-1]
        for featureFrame in listOfDF:
        
        
            timediff = np.min(np.abs(featureFrame.index-beepTime)) 
            # Find corresponding moment for beep time in the sensor data
            #print(timediff)
            if timediff>timedelta(minutes=esmWindowLength):
                # If corresponding time is too far off, remove beep
                #print("Couldn't find corresponding sensor data")
                continue
            pos=np.argmin(np.abs(featureFrame.index-beepTime))
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


if __name__ == '__main__':

    path = "Y:/ADBS"
    downsampling=2
    localPath = 'C:/data/raw/MOX/'
    outPath = 'C:/data/processed/ESM_pilot/'
    featureWindowLength=60
    windowLength=60
    esmWindowLength=15

    esm = loadESM(path)
    #
    allSubs = ['110001','110002','110003','110004','110005','110006','110007','110008','110009','110010','110011','110013','110014','110015','110016','110017','110018','110019','110020','110021']
    #allSubs = ['110015','110017','110018','110019','110020','110021']
    #allSubs = ['110020'] #'110020',
    for subject in allSubs:
        leftFiles, rightFiles, chestFiles = getFileLists(localPath, subject)
        t=time.time()
        trialData,selectedESM = extractRawTrials(leftFiles,rightFiles,chestFiles,esm[esm['castorID']==int(subject)])
        print(time.time()-t)
        print(trialData.shape)
        print(selectedESM.shape)
        np.save(join(outPath,subject + '_trials.npy'),trialData.astype(np.float32))
        #np.save(join(outPath,subject + '_trials64.npy'),trialData)
        selectedESM.to_csv(join(outPath, subject + '_esm.csv'),index=False)



'''
### Transform Gyro data into orientation estimation
from madgwickahrs import MadgwickAHRS
mw = MadgwickAHRS(sampleperiod=1/sr)
euler = np.zeros((3,sigbufs.shape[1]))
for sample in range(sigbufs.shape[1]):
    mw.update_imu(sigbufs[6:,sample],sigbufs[3:6,sample])
    euler[:,sample] = mw.quaternion.to_euler123()


# In[ ]:


plt.matshow(euler,aspect='auto')
plt.yticks([0,1,2],['Roll', 'Pitch', 'Yaw'])
plt.xlabel('Time in samples')
plt.show()

'''