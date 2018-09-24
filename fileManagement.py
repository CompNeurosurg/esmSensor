import pyedflib

def readData(filename): 
    #Extract data
    t = time.time()
    f = pyedflib.EdfReader(mypath+filename)
    sr = f.getSampleFrequencies()[0]
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)
    print(time.time()-t)
    #Get starting time
    t=time.time()
    startingTime=filename[-19:-4]
    startingTime=pd.to_datetime(startingTime, format='%Y%m%d_%H%M%S', errors='ignore')
    timeStamps=[]
    last=startingTime
    
    for sample in range(f.getNSamples()[0]):
        timeStamps.append(last)
        last=last+pd.Timedelta('%d ms' % (1000/sr))
    f._close()
    print(time.time()-t)
    return signal_labels, timeStamps, sigbufs, sr