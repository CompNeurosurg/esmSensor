from scipy.stats import spearmanr, pearsonr
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_context('paper')

#allSubs = ['110001','110002','110003','110004','110005','110006','110007','110008','110009','110010','110011','110013','110014','110016','110017','110018','110019','110020','110021',]
allSubs = ['110018']
#'110015'
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
esmInterest = ['act_norpob', 'prob_mobility', 'prob_stillness',
       'prob_speech', 'prob_walking','tremor', 'slowness', 'stiffness', 'tension', 'dyskinesia', 'onoff']
esmInterest = ['onoff']
#esmInterest = ['tremor']


drop=[ 'subjno','soc_who', 'soc_who02', 'soc_who03', 'act_what', 'act_what02', 'loc_where',
       'soc_who', 'soc_who02', 'soc_who03', 'act_what', 'act_what02',
       'act_what03','_datetime', '_datetime_e', 'dayno_n', 'beepno_n','duration','castorID']

rs=np.zeros((len(allSubs),len(esmInterest)))
ps=np.zeros((len(allSubs),len(esmInterest)))

## Sub Individual

folds = 5
for i,sub in enumerate(allSubs):

    esmFeatures = pd.read_csv('C:/data/processed/ESM_pilot/' + sub + '_features900.csv',index_col=False)
    esmFeatures = esmFeatures.drop(drop, axis=1, errors='ignore')
    #esmFeatures = esmFeatures.dropna(axis=1)
    dat = esmFeatures.drop(esmColumns,axis=1,errors='ignore')
    #dat=esmFeatures.filter(like='BandPower')
    feats = [c for c in dat.keys() if c[-1]!='C'] # if c[-1]!='C'
    dat = dat[feats]
    #feats = [c for c in dat.keys() if c[:9]=='BandPower' ]
    #dat = dat[feats]

    est=LinearDiscriminantAnalysis()
    est=LogisticRegression(solver='lbfgs')


    e =esmInterest[0]
    x=dat.loc[~np.isnan(esmFeatures[e]),:].values
    y=esmFeatures[e][~np.isnan(esmFeatures[e])].values
    y=y==3
    if np.sum(y)/len(y)<0.2 or np.sum(y)/len(y)>0.8:
        continue
    print(np.sum(y)/len(y))
    kf = KFold(n_splits=folds)
    weights=np.zeros((folds,x.shape[1]))
    scores = np.zeros((y.shape[0],2))
    aucs=[]
    for fold, (train,test) in enumerate(kf.split(x)): 
        est.fit(x[train,:],y[train])
        #scores[test,:] = est.predict_proba(x[test,:])
        if hasattr(est, "predict_proba"):
            prob_pos = est.predict_proba(x[test,:])#[:, 0]
        else:  # use decision function
            prob_pos = est.decision_function(x[test,:])
            #prob_pos = \
            #    (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        scores[test,:]=prob_pos
        #importances[fold,:] = est.feature_importances_
        weights[fold,:] = est.coef_
        aucs.append(roc_auc_score(y[test]==1, scores[test,0]))
    #scores = cross_val_score(est,x,y,cv=KFold(10))
    auc = roc_auc_score(y==1, scores[:,0])
    (fpr, tpr, treshs) = roc_curve(y==1, scores[:,0])
    print('Participant %s has auc %f' %(sub,auc))
    print(np.mean(aucs), np.std(aucs))


    # Figure
    sns.set_context('paper')
    fig, ax= plt.subplots(figsize=(4,4))

    ax.plot(fpr,tpr,'o-')
    ax.plot(np.arange(0,1.1,0.1),np.arange(0,1.1,0.1))
    ax.set_xlabel('False Positive Rate', fontsize=16)
    ax.set_ylabel('True Positive Rate',fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    sns.despine()
    plt.grid()
    plt.tight_layout()
    plt.savefig('roc.png',dpi=300)
    plt.show()

    prediction = cross_val_predict(est, x,y)

    fig, ax = plt.subplots(figsize=(8,4))

    line1, = plt.plot(np.arange(len(y)), y,color='k')
    line2, = plt.plot(np.arange(len(y)),prediction,'--', color='grey')
    plt.legend([line1,line2],['Actual','Prediction'])
    plt.xlabel('Beep')
    plt.ylabel('ESM Answer')
    sns.despine()
    plt.grid()
    plt.tight_layout()
    plt.savefig('prediction.png',dpi=300)
    plt.show()