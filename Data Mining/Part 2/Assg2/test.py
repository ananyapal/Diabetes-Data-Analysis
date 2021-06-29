import pickle
import pandas as pd
import numpy as np


def normalize(df):
   df=(df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))
   #df=df/df.max(axis=0)
   return df

testfile = pd.read_csv('test.csv',usecols=np.arange(25))

features = pd.DataFrame()
fmean=testfile.mean(axis=1)
fmax=testfile.max(axis=1)
fmin=testfile.min(axis=1)
fstd=testfile.std(axis=1)
fvar=testfile.var(axis=1)
fmedian=testfile.median(axis=1)
fskew=testfile.skew(axis=1)
fkurt=testfile.kurtosis(axis=1)

features=pd.concat([fmean, fmax,fmin,fstd,fvar,fmedian,fskew,fkurt], axis = 1)

f11=[]

#RMS
for index,row in testfile.iterrows():
    rms = 0
    rms = rms + np.square(index)
    res= np.sqrt(rms / row.shape)
    f11.append(res)
    
f1 = pd.DataFrame(f11)
features=pd.concat([features,f1],axis=1)
features.columns=['mean', 'max','min','std','var','median','skewness','kurtosis','rms']

testfeatures=normalize(features)
print(testfeatures)


svclassifier = pickle.load(open('model.pkl', 'rb'))
y_pred = svclassifier.predict(testfeatures)
print('Generated result of SVM CLassifier')
svdf = pd.DataFrame(y_pred, columns=['Class(Meal=1,NoMeal=0)'])
svdf.to_csv("Result.csv")
