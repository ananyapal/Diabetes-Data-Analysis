import pickle 
import numpy as np
import pandas as pd
#import pywt
from scipy.stats import skew
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict


#-----------------EXTRACTED MEAL AND NO MEAL FILES---------------------
meal = pd.read_csv('MealSet.csv',usecols=np.arange(25))
nomeal = pd.read_csv('NoMealSet.csv',usecols=np.arange(25))
#----------------------------------------------------------------------

def normalize(df):
   df=(df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))
   #df=df/df.max(axis=0)
   return df


#--------------------MEAL-----------------------------
   
features = pd.DataFrame()
fmean=meal.mean(axis=1)
fmax=meal.max(axis=1)
fmin=meal.min(axis=1)
fstd=meal.std(axis=1)
fvar=meal.var(axis=1)
fmedian=meal.median(axis=1)
fskew=meal.skew(axis=1)
fkurt=meal.kurtosis(axis=1)

features=pd.concat([fmean, fmax,fmin,fstd,fvar,fmedian,fskew,fkurt], axis = 1)

f11=[]

#RMS
for index,row in meal.iterrows():
    rms = 0
    rms = rms + np.square(index)
    res= np.sqrt(rms / row.shape)
    f11.append(res)
    
f1 = pd.DataFrame(f11)
features=pd.concat([features,f1],axis=1)
features.columns=['mean', 'max','min','std','var','median','skewness','kurtosis','rms']

mealfeatures=normalize(features)
mealfeatures['Class']=1     #Meal = 1
#print(mealfeatures)

#--------------------NOMEAL-----------------------------

features2 = pd.DataFrame()
fmean2=nomeal.mean(axis=1)
fmax2=nomeal.max(axis=1)
fmin2=nomeal.min(axis=1)
fstd2=nomeal.std(axis=1)
fvar2=nomeal.var(axis=1)
fmedian2=nomeal.median(axis=1)
fskew2=nomeal.skew(axis=1)
fkurt2=nomeal.kurtosis(axis=1)

features2=pd.concat([fmean2, fmax2,fmin2,fstd2,fvar2,fmedian2,fskew2,fkurt2], axis = 1)

f12=[]

#RMS
for index,row in nomeal.iterrows():
    rms = 0
    rms = rms + np.square(index)
    res= np.sqrt(rms / row.shape)
    f12.append(res)
    
f2 = pd.DataFrame(f12)
features2=pd.concat([features2,f2],axis=1)
features2.columns=['mean', 'max','min','std','var','median','skewness','kurtosis','rms']

nomealfeatures=normalize(features2)
nomealfeatures['Class']=0     #NoMeal = 0

#--------------------ALLDATA-----------------------------

alldata=pd.concat([mealfeatures,nomealfeatures],axis=0)
#print(alldata)

X = alldata.drop('Class', axis=1)
y = alldata['Class']

#--------------------SVM CLASSIFIER MODEL-----------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


print("ACCURACY:")
print(accuracy_score(y_test,y_pred))
print("F1 SCORE:")
print(f1_score(y_test, y_pred))
print("RECALL:")
print(recall_score(y_test, y_pred))
print("PRECISION:")
print(precision_score(y_test, y_pred))


#--------------MODEL PICKLE FILE GENERATED----------------------
with open('model.pkl', 'wb') as (file):
    pickle.dump(svclassifier, file)
    
    
#--------------VALIDATION----------------------

print("\nVALIDATION")



cv_r2_scores_rf = cross_val_score(svclassifier, X, y, cv=5,scoring='r2')
print(cv_r2_scores_rf)
print("Mean 5-Fold R Squared: {}".format(np.mean(cv_r2_scores_rf)))

kf = KFold(n_splits = 5, shuffle = True)

scores = []
for i in range(5):
    result = next(kf.split(alldata), None)
    x_train = alldata.iloc[result[0]]
    x_test = alldata.iloc[result[1]]
    y_train = y.iloc[result[0]]
    y_test = y.iloc[result[1]]
    model = svclassifier.fit(x_train,y_train)
    predictions = svclassifier.predict(x_test)
    scores.append(model.score(x_test,y_test))
print('Scores from each Iteration: ', scores)
print('Average K-Fold Score :' , np.mean(scores))

