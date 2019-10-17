from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_predict, KFold
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#import seaborn as sns
#sns.set_context('paper')

allSubs = ['110001','110002','110003','110004','110005','110006','110007','110008','110009','110010','110011','110013','110014','110016','110017','110018'] 
#'110015'
allSubs = ['110018']
phenotype = np.array([2,1,2,2,2,2,1,2,2,1,1,1,2,2,2,1]) # removed 15


outPath = 'C:/data/processed/ESM_pilot/'

esmColumns = ['subjno', 'mood_well', 'mood_down', 'mood_fright', 'mood_tense', 'phy_sleepy', 'phy_tired',
       'mood_cheerf', 'mood_relax', 'thou_concent', 'pat_hallu', 'loc_where',
       'soc_who', 'soc_who02', 'soc_who03', 'act_what', 'act_what02',
       'act_what03', 'act_norpob', 'prob_mobility', 'prob_stillness',
       'prob_speech', 'prob_walking', 'tremor', 'slowness',
       'stiffness', 'tension', 'dyskinesia', 'onoff',
       'medic', 'beep_disturb', '_datetime', '_datetime_e', 'dayno_n', 'beepno_n','duration']
esmInterest = ['mood_well', 'mood_down', 'mood_fright', 'mood_tense', 'phy_sleepy', 'phy_tired',
       'mood_cheerf', 'mood_relax', 'thou_concent', 'pat_hallu', 'act_norpob', 'prob_mobility', 'prob_stillness',
       'prob_speech', 'prob_walking', 'tremor', 'slowness',
       'stiffness', 'tension', 'dyskinesia', 'onoff',
       'medic', 'beep_disturb']

drop=[ 'subjno','soc_who', 'soc_who02', 'soc_who03', 'act_what', 'act_what02', 'loc_where',
       'soc_who', 'soc_who02', 'soc_who03', 'act_what', 'act_what02',
       'act_what03','_datetime', '_datetime_e', 'dayno_n', 'beepno_n','duration','castorID']
esmInterest = ['act_norpob', 'prob_mobility', 'prob_stillness',
       'prob_speech', 'prob_walking','tremor', 'slowness', 'stiffness', 'tension', 'dyskinesia', 'onoff']
esmInterest = ['tremor','onoff']

winLength = [30, 60, 120, 180, 300, 450, 900]
winLength = [900]
folds=10
rs=np.zeros((len(winLength),len(allSubs),len(esmInterest)))
rsFolds = np.zeros((len(winLength),len(allSubs),len(esmInterest),10))
ps=np.zeros((len(winLength),len(allSubs),len(esmInterest)))
rmses=np.zeros((len(winLength),len(allSubs),len(esmInterest)))
rands=np.zeros((len(winLength),len(allSubs),len(esmInterest),2))

for w, wl in enumerate(winLength):
    for s, sub in enumerate(allSubs):
       
            esmFeatures = pd.read_csv('C:/data/processed/ESM_pilot/' + sub + '_features' + str(wl) + '.csv',index_col=False)
            esmFeatures = esmFeatures.drop(drop, axis=1, errors='ignore')
            #esmFeatures = esmFeatures.dropna(axis=1)
            dat = esmFeatures.drop(esmColumns,axis=1,errors='ignore')
            #dat=esmFeatures.filter(like='BandPower')
            feats = [c for c in dat.keys() if c[-1]!='C' ]
            dat = dat[feats].values
            #feats = [c for c in dat.keys() if c[:9]=='BandPower' ]
            #dat = dat[feats]
            #est=RandomForestRegressor(n_estimators=100)
            est=LinearRegression()
            #est=SVR()
            #est=Lasso(alpha=1.0)
            for eN, e in enumerate(esmInterest):
                x=dat[~np.isnan(esmFeatures[e]),:]
                y=esmFeatures[e][~np.isnan(esmFeatures[e])].values
                #est=LinearRegression()
                #est=RandomForestRegressor(n_estimators=10)
                prediction = np.zeros(y.shape)
                kf = KFold(n_splits=folds)
                for f, (train, test) in enumerate(kf.split(x)):
                        est.fit(x[train,:],y[train])
                        pred = est.predict(x[test,:])
                        prediction[test]=pred
                        rsFolds[w,s,eN, f],_ = pearsonr(pred,y[test])
                rs[w,s,eN], ps[w,s,eN] = pearsonr(prediction,y)
                rmses[w,s,eN] = np.sqrt(np.mean((y-prediction)**2))
                if wl==900:
                    fig, ax = plt.subplots()
                    
                    line1, = plt.plot(np.arange(len(y))[25:], y[25:])
                    line2, = plt.plot(np.arange(len(prediction))[25:],prediction[25:],'--')
                    plt.legend([line1,line2],['Actual','Prediction'])
                    plt.xlabel('Beep')
                    plt.ylabel('ESM Answer')

                    plt.savefig(e + '_prediction.png', dpi=150)
                    plt.show()
                # Establish baseline
                rounds = 1000
                rRands=[]
                rmseRands=[]
                for r in range(rounds):
                    perm = np.random.permutation(y)
                    r, p = pearsonr(perm,y)
                    rRands.append(r)
                    rmse = np.sqrt(np.mean((y-perm)**2))
                    rmseRands.append(rmse)
                alpha=0.05
                rands[w,s,eN,0] = np.sort(rRands)[int(- alpha * rounds)]
                rands[w,s,eN,1] = np.sort(rmseRands)[int(alpha*rounds)]
                

for eN, e in enumerate(esmInterest):
    fig, ax = plt.subplots()
    line1, = ax.plot(winLength, rs[:,0,eN], 'g-o')
    line2, = ax.plot(winLength, rands[:,0,eN,0], 'r-x')
    line2 = ax.fill_between(winLength, np.zeros((len(winLength))),rands[:,0,eN,0],color='r',alpha=0.5)
    plt.legend([line1, line2],['Prediction', 'Random'])
    plt.xlabel('Window Length (s)')
    plt.ylabel('Pearson r')
    plt.ylim(0,0.5)
    plt.xticks(winLength,[str(w) for w in winLength])
    plt.grid()
    plt.savefig(e + '_corrs.png',dpi=150)
    plt.show()

    fig, ax = plt.subplots()
    line1 = ax.errorbar(winLength, np.mean(rsFolds[:,0,eN,:],axis=1),np.std(rsFolds[:,0,eN,:],axis=1)/np.sqrt(10),color='g', linestyle='-')
    line2, = ax.plot(winLength, rands[:,0,eN,0], 'r-x')
    line2 = ax.fill_between(winLength, np.zeros((len(winLength))),rands[:,0,eN,0],color='r',alpha=0.5)
    plt.legend([line1, line2],['Prediction', 'Random'])
    plt.xlabel('Window Length (s)')
    plt.ylabel('Pearson r')
    plt.ylim(0,0.5)
    plt.xticks(winLength,[str(w) for w in winLength])
    plt.grid()
    #plt.savefig(e + '_corrs.png',dpi=150)
    plt.show()


    fig, ax = plt.subplots()
    line1, = ax.plot(winLength, rmses[:,0,eN],'-o')
    line2, = ax.plot(winLength, rands[:,0,eN,1], '-x')
    plt.legend([line1, line2],['Prediction', 'Random'])
    plt.xlabel('Window Length (s)')
    plt.ylabel('RMSE')
    plt.savefig(e + '_rmse.png',dpi=150)
    plt.show()